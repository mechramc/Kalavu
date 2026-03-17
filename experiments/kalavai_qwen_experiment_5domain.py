#!/usr/bin/env python3
"""
KALAVAI: Qwen2.5-1.5B 5-Domain Specialist Fusion Experiment
===========================================================
Fine-tunes Qwen2.5-1.5B on five maximally divergent domains, then fuses
via learned MoE routing. Tests whether fusion improvement compounds with
more specialists (scaling law for cooperative training).

Domains (chosen for maximal stylistic divergence):
  - math:     GSM8K (arithmetic word problems + step-by-step solutions)
  - science:  SciQ (long scientific passages + Q&A — MUST include `support`)
  - code:     code_search_net python (function source code)
  - legal:    pile-of-law/pile-of-law (court opinions, contracts)
  - creative: emozilla/pg19 (Project Gutenberg full books)

Key design decisions (same as 2-domain experiment):
- Packed tokenization (concatenate → 512-token chunks, NO padding)
- Full fine-tuning of unfrozen layers (NOT LoRA)
- Freeze first 2 transformer blocks + embeddings
- AdamW with betas=(0.9, 0.95), linear warmup + cosine decay

Verified results (in-distribution):
    5-domain MoE fused mixed loss: 0.8165
    Best individual mixed loss:    1.3221
    Improvement: +38.24%

Scaling pattern: 2 overlapping domains (+0.9%) → 2 divergent (+17.15%) → 5 domains (+38.24%)
"""

import copy
import json
import math
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B"
DOMAINS = ["math", "science", "code", "legal", "creative"]
FREEZE_LAYERS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_STEPS = 200
BATCH_SIZE = 2
GRAD_ACCUM = 2
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
N_TRAIN_SAMPLES = 2000
ROUTER_STEPS = 300
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
SEED = 42
CHECKPOINT_DIR = Path("kalavai_checkpoints_5domain")
RESULTS_PATH = Path("kalavai_checkpoints_5domain/qwen_5domain_results.json")


# ============================================================================
# Packed tokenization
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_len: int = SEQ_LEN,
                 max_chars: int = 1500):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# Data loading — one function per domain
# ============================================================================

def load_math_texts(split: str, n_samples: int) -> list[str]:
    from datasets import load_dataset
    hf_split = "test" if split == "eval" else "train"
    ds = load_dataset("openai/gsm8k", "main", split=hf_split, streaming=True)
    texts = []
    for item in ds:
        content = item.get("question", "") + "\n" + item.get("answer", "")
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n_samples:
            break
    return texts


def load_science_texts(split: str, n_samples: int) -> list[str]:
    """CRITICAL: include `support` field — long scientific passage."""
    from datasets import load_dataset
    hf_split = "validation" if split == "eval" else "train"
    ds = load_dataset("allenai/sciq", split=hf_split, streaming=True)
    texts = []
    for item in ds:
        content = (
            item.get("support", "") + "\n"
            + item.get("question", "") + "\n"
            + item.get("correct_answer", "")
        )
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n_samples:
            break
    return texts


def load_code_texts(split: str, n_samples: int) -> list[str]:
    """code_search_net python: whole_func_string includes docstring."""
    from datasets import load_dataset
    hf_split = "test" if split == "eval" else "train"
    ds = load_dataset("code_search_net", "python", split=hf_split, streaming=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n_samples:
            break
    return texts


def load_legal_texts(split: str, n_samples: int) -> list[str]:
    """pile-of-law: court opinions and legal documents."""
    from datasets import load_dataset
    # pile-of-law has many subsets; stream from the main dataset
    # using atticus_contracts which are well-structured legal documents
    try:
        ds = load_dataset(
            "pile-of-law/pile-of-law", "atticus_contracts",
            split="train", streaming=True,
        )
    except Exception:
        # Fallback to us_court_opinions if atticus_contracts unavailable
        ds = load_dataset(
            "pile-of-law/pile-of-law", "us_court_opinions",
            split="train", streaming=True,
        )
    skip = n_samples if split == "eval" else 0
    texts = []
    skipped = 0
    for item in ds:
        content = item.get("text", "")
        if len(content) < 200:
            continue
        if skipped < skip:
            skipped += 1
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    return texts


def load_creative_texts(split: str, n_samples: int) -> list[str]:
    """pg19: Project Gutenberg books. Test split for eval."""
    from datasets import load_dataset
    hf_split = "test" if split == "eval" else "train"
    ds = load_dataset("emozilla/pg19", split=hf_split, streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:3000]
        if len(content) < 500:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    return texts


LOADERS = {
    "math":     load_math_texts,
    "science":  load_science_texts,
    "code":     load_code_texts,
    "legal":    load_legal_texts,
    "creative": load_creative_texts,
}


# ============================================================================
# Freezing
# ============================================================================

def freeze_first_n_layers(model, n: int):
    for p in model.model.embed_tokens.parameters():
        p.requires_grad = False
    for i in range(n):
        for p in model.model.layers[i].parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# 5-Expert MoE
# ============================================================================

class FiveExpertMoE(nn.Module):
    """
    Sequence-level MoE over five specialist models.
    All specialists run fully; router combines their logits via softmax weights.
    Only router weights are trained.
    """
    def __init__(self, specialists: list, hidden_size: int):
        super().__init__()
        assert len(specialists) == 5
        self.specialists = nn.ModuleList(specialists)
        for spec in self.specialists:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 5, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach()
        last_h = out.hidden_states[-1].detach()
        h_pooled = last_h.mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        all_logits = []
        all_h = []
        for spec in self.specialists:
            logits, h = self._run_specialist(spec, input_ids)
            all_logits.append(logits)
            all_h.append(h)

        h_avg = torch.stack(all_h, dim=0).mean(dim=0)  # (B, H)
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, 5)

        fused = sum(gates[:, i:i+1, None] * all_logits[i] for i in range(5))

        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss, fused, gates


# ============================================================================
# Training
# ============================================================================

def train_specialist(model, dataset: PackedChunkDataset, domain: str, device: str):
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(MAX_STEPS * WARMUP_FRACTION))
    step = 0
    accum = 0
    total_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum += 1
        total_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

            if step < warmup_steps:
                lr = LR * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, MAX_STEPS - warmup_steps)
                lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()
            accum = 0
            step += 1

            if step % 50 == 0 or step == MAX_STEPS:
                avg = total_loss / step
                elapsed = time.time() - t0
                print(f"  [{domain}] step {step}/{MAX_STEPS} | loss {avg:.4f} | {elapsed:.0f}s")

    print(f"  {domain} training done in {time.time()-t0:.0f}s")
    return model


# ============================================================================
# Eval
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset: PackedChunkDataset, device: str,
              batch_size: int = 4, is_fused: bool = False) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(input_ids, labels=labels)
        else:
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


def print_table(results: dict, domains: list[str]):
    col_w = 10
    header = f"{'Model':<28}" + "".join(f"{d[:7]:>{col_w}}" for d in domains) + f"{'Average':>{col_w}}"
    print(header)
    print("-" * len(header))
    for name, losses in results.items():
        avg = sum(losses[d] for d in domains if d in losses) / len(domains)
        row = f"{name:<28}" + "".join(f"{losses.get(d, float('nan')):>{col_w}.4f}" for d in domains) + f"{avg:>{col_w}.4f}"
        print(row)


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(SEED)

    print("=" * 70)
    print("KALAVAI: Qwen2.5-1.5B 5-Domain Fusion Experiment")
    print("=" * 70)
    print(f"Model:   {MODEL_ID}")
    print(f"Domains: {', '.join(DOMAINS)}")
    print(f"Steps:   {MAX_STEPS} per specialist, freeze_layers={FREEZE_LAYERS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load all domain data
    print("\nLoading data for all domains...")
    train_texts = {}
    eval_texts = {}
    for domain in DOMAINS:
        print(f"  Loading {domain}...")
        train_texts[domain] = LOADERS[domain]("train", N_TRAIN_SAMPLES)
        eval_texts[domain]  = LOADERS[domain]("eval",  N_TRAIN_SAMPLES // 5)
        print(f"    {domain}: {len(train_texts[domain])} train, {len(eval_texts[domain])} eval")

    # Build packed datasets
    print("\nBuilding packed datasets...")
    train_sets = {d: PackedChunkDataset(train_texts[d], tokenizer) for d in DOMAINS}
    eval_sets  = {d: PackedChunkDataset(eval_texts[d],  tokenizer) for d in DOMAINS}

    # Mixed eval and train
    eval_mixed = PackedChunkDataset.__new__(PackedChunkDataset)
    eval_mixed.chunks = [c for d in DOMAINS for c in eval_sets[d].chunks]
    train_mixed = PackedChunkDataset.__new__(PackedChunkDataset)
    train_mixed.chunks = [c for d in DOMAINS for c in train_sets[d].chunks]
    eval_sets["mixed"] = eval_mixed

    for domain in DOMAINS:
        print(f"  {domain}_train_chunks={len(train_sets[domain])}")

    # Load base model for eval
    print(f"\nLoading base model for eval...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()
    hidden_size = base_model.config.hidden_size

    # Train all specialists
    specialists = {}
    for domain in DOMAINS:
        print(f"\n{'='*60}\nTraining {domain} specialist\n{'='*60}")
        spec = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        freeze_first_n_layers(spec, FREEZE_LAYERS)
        train_specialist(spec, train_sets[domain], domain, device)
        spec.eval()
        spec.save_pretrained(CHECKPOINT_DIR / f"{domain}_specialist")
        tokenizer.save_pretrained(CHECKPOINT_DIR / f"{domain}_specialist")
        specialists[domain] = spec

    # 5-expert MoE
    print(f"\nBuilding 5-expert MoE and training router ({ROUTER_STEPS} steps)...")
    spec_list = [specialists[d] for d in DOMAINS]
    moe = FiveExpertMoE(spec_list, hidden_size).to(device)

    router_optimizer = torch.optim.AdamW(moe.router.parameters(), lr=ROUTER_LR)
    router_loader = DataLoader(train_mixed, batch_size=ROUTER_BATCH, shuffle=True,
                               drop_last=True, collate_fn=_collate)
    router_iter = cycle(router_loader)
    moe.train()
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(router_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        router_optimizer.zero_grad()
        loss.backward()
        router_optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"  Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()
    torch.save(moe.router.state_dict(), CHECKPOINT_DIR / "router_5domain.pt")

    # Evaluate all variants
    print("\nEvaluating all variants...")
    all_domains_and_mixed = DOMAINS + ["mixed"]

    results = {}

    # Base model
    results["Base Qwen2.5-1.5B"] = {}
    for domain in all_domains_and_mixed:
        loss = eval_loss(base_model, eval_sets[domain], device)
        results["Base Qwen2.5-1.5B"][domain] = round(loss, 6)

    # Each specialist on all eval sets
    for domain in DOMAINS:
        name = f"Specialist ({domain})"
        results[name] = {}
        for eval_domain in all_domains_and_mixed:
            loss = eval_loss(specialists[domain], eval_sets[eval_domain], device)
            results[name][eval_domain] = round(loss, 6)

    # MoE fused
    results["MoE fused (5 experts)"] = {}
    for domain in all_domains_and_mixed:
        loss = eval_loss(moe, eval_sets[domain], device, batch_size=2, is_fused=True)
        results["MoE fused (5 experts)"][domain] = round(loss, 6)

    print(f"\nRESULTS — 5-Domain Fusion (in-distribution eval):")
    print_table(results, all_domains_and_mixed)

    # Improvement
    best_individual_mixed = min(
        results[f"Specialist ({d})"]["mixed"] for d in DOMAINS
    )
    fused_mixed = results["MoE fused (5 experts)"]["mixed"]
    improvement = (best_individual_mixed - fused_mixed) / best_individual_mixed * 100
    print(f"\nImprovement over best individual (mixed): {improvement:+.2f}%")

    # Save
    output = {
        "experiment": "qwen_5domain",
        "model_id": MODEL_ID,
        "domains": DOMAINS,
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "seq_len": SEQ_LEN,
            "n_experts": 5,
        },
        "eval_loss": results,
        "best_individual_mixed": round(best_individual_mixed, 6),
        "improvement_pct": round(improvement, 4),
        "thesis_holds": improvement > 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
