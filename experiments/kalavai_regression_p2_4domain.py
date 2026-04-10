#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Regression Extension P2 — 4-Specialist (Code + Science + Fiction + Legal)
====================================================================================
Adds a new (divergence, gain) data point to the divergence-gain regression by adding
a 4th specialist (EU legal text from lex_glue/eurlex) to the existing 3-domain setup.

Expected outcome:
  - Code div: ~10%,  Science div: ~12%,  Fiction div: ~25%,  Legal div: ~28-35%
  - Mean divergence: ~18-22%  (fills gap between Exp2=18.52% and Exp1=25.65%)
  - Expected gain: ~12-15% vs best specialist (from regression: 0.82*20 - 2.72 ≈ 13.7%)

Regression value: this is an UPPER-MID-RANGE data point (n=8 total after both P1+P2).

Config: identical to kalavai_pythia_experiment.py except N_EXPERTS=4 and DOMAINS includes
legal. Router hyperparameters unchanged (500 steps, same LR).

Usage:
    python experiments/kalavai_regression_p2_4domain.py
"""

import copy
import json
import os
import statistics
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
FREEZE_LAYERS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
SEEDS = [42, 137, 2026]

# P2: 4 specialists — add legal to the existing 3-domain set
DOMAINS = ["code", "science", "fiction", "legal"]
N_EXPERTS = len(DOMAINS)

RESULTS_DIR = Path("results/regression_extension")
CHECKPOINT_DIR = Path("checkpoints/regression_extension/p2_4domain")

HIDDEN_SIZE = 1024
NUM_LAYERS = 24


# ============================================================================
# PackedChunkDataset
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_len: int = SEQ_LEN,
                 max_chars: int = 5000):
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


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n_samples: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading code (n={n_samples}) from code_search_net python...")
    ds = load_dataset("code_search_net", "python", split="train", streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) <= 200:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} code samples")
    return texts


def load_science_texts(n_samples: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading science (n={n_samples}) from allenai/sciq...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
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
    print(f"    Loaded {len(texts)} science samples")
    return texts


def load_fiction_texts(n_samples: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading fiction (n={n_samples}) from emozilla/pg19...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        raw = item.get("text", "")
        content = raw[:5000]
        if len(content) < 500:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} fiction samples")
    return texts


def load_legal_texts(n_samples: int) -> list[str]:
    """EU legislation from lex_glue/eurlex — same dataset as Phase 2 Exp 2 (legal)."""
    from datasets import load_dataset
    print(f"  Loading legal (n={n_samples}) from lex_glue/eurlex...")
    ds = load_dataset("lex_glue", "eurlex", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} legal samples")
    return texts


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    n = len(chunks)
    train_end = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]


# ============================================================================
# Freezing
# ============================================================================

def freeze_first_n_layers(model, n: int):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# N-Expert MoE (generalised)
# ============================================================================

class NExpertMoE(nn.Module):
    """
    Sequence-level MoE over N specialist models.
    Linear router: mean-pooled hidden state -> N gates.
    Specialists frozen; router trained on mixed data.
    Logit-space combination (equivalent to log-linear mixture).
    """
    def __init__(self, specialists: list, hidden_size: int):
        super().__init__()
        self.specialists = nn.ModuleList(specialists)
        for spec in self.specialists:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.n = len(specialists)
        self.router = nn.Linear(hidden_size, self.n, bias=False)

    @torch.no_grad()
    def _run_specialist(self, model, input_ids):
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

        h_avg = torch.stack(all_h, dim=0).mean(dim=0)
        gates = torch.softmax(self.router(h_avg), dim=-1)

        fused = sum(
            gates[:, i:i+1, None] * all_logits[i]
            for i in range(self.n)
        )

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

def train_specialist(model, domain: str, train_chunks: list, seed: int,
                     device: str, log_every: int = 50) -> list:
    set_seed(seed)
    freeze_first_n_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    print(f"  {domain} train_chunks={len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    step = 0
    accum = 0
    running_loss = 0.0
    loss_history = []
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch_to_device(batch, device))
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1

            if step % log_every == 0 or step == MAX_STEPS:
                avg = running_loss / step
                elapsed = time.time() - t0
                print(f"  [{domain}] step {step}/{MAX_STEPS} | loss {avg:.4f} | {elapsed:.0f}s")
                loss_history.append((step, round(avg, 6)))

    print(f"  {domain} training done in {time.time()-t0:.0f}s")
    return loss_history


# ============================================================================
# Evaluation — per-domain equal-weight protocol
# ============================================================================

@torch.no_grad()
def eval_loss_domain(model, dataset, device: str,
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


def eval_all_domains(model, held_out_by_domain: dict, device: str,
                     is_fused: bool = False) -> dict:
    losses = {}
    for domain, ds in held_out_by_domain.items():
        loss = eval_loss_domain(model, ds, device, is_fused=is_fused)
        losses[domain] = round(loss, 6)
        print(f"    [{domain:10s}]: {loss:.4f}")
    ew_avg = statistics.mean(losses.values())
    losses["equal_weight_avg"] = round(ew_avg, 6)
    print(f"    [equal_weight_avg]: {ew_avg:.4f}")
    return losses


# ============================================================================
# Router training
# ============================================================================

def train_router(moe: NExpertMoE, train_chunks_by_domain: dict, device: str):
    all_chunks = []
    for chunks in train_chunks_by_domain.values():
        all_chunks.extend(chunks)
    combined = make_dataset_from_chunks(all_chunks)

    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()

    print(f"\n  Training router ({ROUTER_STEPS} steps, mixed={len(combined)} chunks)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")


# ============================================================================
# Router distribution eval
# ============================================================================

@torch.no_grad()
def eval_router_distribution(moe: NExpertMoE, held_out_by_domain: dict,
                              device: str, n_batches: int = 20) -> dict:
    moe.eval()
    results = {}
    for domain, ds in held_out_by_domain.items():
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                            drop_last=True, collate_fn=_collate)
        gate_sums = [0.0] * moe.n
        count = 0
        for batch in loader:
            if count >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(moe.n):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        if count > 0:
            results[domain] = [round(g / count, 4) for g in gate_sums]
        else:
            results[domain] = [1.0 / moe.n] * moe.n
    return results


# ============================================================================
# Main experiment loop
# ============================================================================

def run_seed(seed: int, all_domain_chunks: dict, device: str) -> dict:
    print(f"\n{'=' * 70}")
    print(f"SEED {seed}")
    print(f"{'=' * 70}")
    set_seed(seed)

    held_out_by_domain = {
        d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
        for d in DOMAINS
    }

    # Base model eval
    print(f"\n[Seed {seed}] Loading base model for eval...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()
    print(f"[Seed {seed}] Evaluating base model...")
    base_losses = eval_all_domains(base_model, held_out_by_domain, device)
    base_ew = base_losses["equal_weight_avg"]
    del base_model
    torch.cuda.empty_cache()

    # Train specialists
    specialists = {}
    spec_losses_by_domain = {}

    for domain in DOMAINS:
        print(f"\n[Seed {seed}] Training {domain} specialist...")
        spec = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        train_specialist(spec, domain, all_domain_chunks[domain]["train"], seed, device)
        spec.eval()

        print(f"[Seed {seed}] Evaluating {domain} specialist on all domains...")
        spec_losses = eval_all_domains(spec, held_out_by_domain, device)
        spec_losses_by_domain[domain] = spec_losses

        specialists[domain] = spec.cpu()
        torch.cuda.empty_cache()

    # Per-specialist divergence
    per_specialist_divergence = {}
    for domain in DOMAINS:
        base_d = base_losses[domain]
        spec_d = spec_losses_by_domain[domain][domain]
        div = (base_d - spec_d) / base_d * 100.0
        per_specialist_divergence[domain] = round(div, 4)
    mean_divergence = round(statistics.mean(per_specialist_divergence.values()), 4)

    print(f"\n[Seed {seed}] Divergence: {per_specialist_divergence}")
    print(f"[Seed {seed}] Mean divergence: {mean_divergence:.2f}%")

    # Best specialist
    best_spec_domain = min(DOMAINS, key=lambda d: spec_losses_by_domain[d]["equal_weight_avg"])
    best_spec_ew = spec_losses_by_domain[best_spec_domain]["equal_weight_avg"]
    print(f"\n[Seed {seed}] Best specialist: {best_spec_domain} (EW loss={best_spec_ew:.4f})")

    # Build and train MoE
    print(f"\n[Seed {seed}] Building 4-expert MoE...")
    spec_list = [specialists[d].to(device) for d in DOMAINS]
    moe = NExpertMoE(spec_list, HIDDEN_SIZE).to(device)

    train_chunks_by_domain = {d: all_domain_chunks[d]["train"] for d in DOMAINS}
    train_router(moe, train_chunks_by_domain, device)

    # MoE eval
    print(f"\n[Seed {seed}] Evaluating MoE...")
    moe.eval()
    moe_losses = eval_all_domains(moe, held_out_by_domain, device, is_fused=True)
    moe_ew = moe_losses["equal_weight_avg"]

    # Router distribution
    router_dist = eval_router_distribution(moe, held_out_by_domain, device)
    domain_labels = DOMAINS
    print(f"\n[Seed {seed}] Router distribution:")
    for domain, gates in router_dist.items():
        gate_str = "  ".join(f"{domain_labels[i]}:{gates[i]:.3f}" for i in range(N_EXPERTS))
        print(f"  {domain:10s}: {gate_str}")

    # Metrics
    improvement_vs_spec = (best_spec_ew - moe_ew) / best_spec_ew * 100.0
    improvement_vs_base = (base_ew - moe_ew) / base_ew * 100.0

    print(f"\n[Seed {seed}] RESULTS:")
    print(f"  Base EW loss:       {base_ew:.4f}")
    print(f"  Best spec EW loss:  {best_spec_ew:.4f} ({best_spec_domain})")
    print(f"  MoE EW loss:        {moe_ew:.4f}")
    print(f"  vs best specialist: {improvement_vs_spec:+.2f}%")
    print(f"  vs base:            {improvement_vs_base:+.2f}%")
    print(f"  Mean divergence:    {mean_divergence:.2f}%")

    result = {
        "seed": seed,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "n_experts": N_EXPERTS,
        "domains": DOMAINS,
        "experiment": "regression_p2_4domain",
        "eval_batch_size": 4,
        "eval_batches": EVAL_BATCHES,
        "eval_method": "per-domain-separate-then-equal-weight-avg",
        "base_losses": base_losses,
        "specialist_losses": spec_losses_by_domain,
        "moe_losses": moe_losses,
        "per_specialist_divergence": per_specialist_divergence,
        "router_distribution": {d: {"labels": domain_labels, "gates": g}
                                 for d, g in router_dist.items()},
        "metrics": {
            "base_equal_weight": round(base_ew, 6),
            "best_spec_equal_weight": round(best_spec_ew, 6),
            "best_spec_domain": best_spec_domain,
            "moe_equal_weight": round(moe_ew, 6),
            "improvement_vs_spec": round(improvement_vs_spec, 4),
            "improvement_vs_base": round(improvement_vs_base, 4),
            "mean_divergence": mean_divergence,
        },
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "router_steps": ROUTER_STEPS,
            "router_lr": ROUTER_LR,
            "router_batch": ROUTER_BATCH,
            "n_samples_per_domain": N_SAMPLES_PER_DOMAIN,
        },
    }

    del moe
    for spec in spec_list:
        del spec
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 70)
    print("KALAVAI Regression Extension P2: 4-Domain (Code+Science+Fiction+Legal)")
    print("=" * 70)
    print(f"Model:    {MODEL_ID} @ {REVISION}")
    print(f"Domains:  {DOMAINS}")
    print(f"Seeds:    {SEEDS}")
    print(f"Goal:     Fill regression gap ~19-22% mean divergence")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: CPU mode will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data for all 4 domains...")
    loaders = {
        "code":    lambda: load_code_texts(N_SAMPLES_PER_DOMAIN),
        "science": lambda: load_science_texts(N_SAMPLES_PER_DOMAIN),
        "fiction": lambda: load_fiction_texts(N_SAMPLES_PER_DOMAIN),
        "legal":   lambda: load_legal_texts(N_SAMPLES_PER_DOMAIN),
    }
    all_domain_chunks = {}
    for domain, loader_fn in loaders.items():
        texts = loader_fn()
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, indist_c, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {
            "train": train_c,
            "indist": indist_c,
            "held_out": held_c,
        }
        print(f"  {domain}: total={len(ds_full)}, "
              f"train={len(train_c)}, held_out={len(held_c)}")
        if len(train_c) < 1500:
            print(f"  WARNING: {domain} has <1500 train chunks")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in SEEDS:
        result = run_seed(seed, all_domain_chunks, device)
        all_results.append(result)

        out_path = RESULTS_DIR / f"p2_4domain_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {out_path}")

    # Summary
    gains = [r["metrics"]["improvement_vs_spec"] for r in all_results]
    divs = [r["metrics"]["mean_divergence"] for r in all_results]
    mean_gain = statistics.mean(gains)
    std_gain = statistics.stdev(gains) if len(gains) > 1 else 0.0
    mean_div = statistics.mean(divs)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — 3-Seed Results")
    print("=" * 70)
    print(f"  Seeds: {SEEDS}")
    print(f"  Per-seed gain vs spec: {[f'{g:.2f}%' for g in gains]}")
    print(f"  Mean gain:  {mean_gain:+.2f}% ± {std_gain:.2f}pp")
    print(f"  Mean div:   {mean_div:.2f}%")
    print(f"\n  REGRESSION POINT: (div={mean_div:.2f}%, gain={mean_gain:.2f}%)")
    print(f"  Predicted by regression: {0.82 * mean_div - 2.72:.2f}%")

    summary = {
        "experiment": "regression_p2_4domain",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "domains": DOMAINS,
        "seeds": SEEDS,
        "per_seed_gain_vs_spec": gains,
        "per_seed_mean_div": divs,
        "mean_gain_vs_spec": round(mean_gain, 4),
        "std_gain_vs_spec": round(std_gain, 4),
        "mean_divergence": round(mean_div, 4),
        "regression_prediction": round(0.82 * mean_div - 2.72, 4),
    }
    summary_path = RESULTS_DIR / "p2_4domain_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
