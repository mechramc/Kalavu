#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Real-Model Specialist Divergence Experiment
=====================================================
Trains code vs literary fiction specialists on Qwen2.5-1.5B for 1000 steps,
checks they've actually diverged, then fuses with MoE and evaluates on held-out data.

Domains chosen for maximum stylistic divergence:
  - Code:    bigcode/starcoderdata (Python) — brackets, indentation, symbols
  - Fiction: emozilla/pg19 (Project Gutenberg) — flowing prose, dialogue, narrative

This is the real-model validation for the KALAVU paper.
"""

import copy
import json
import statistics
import sys
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

MODEL_ID = "Qwen/Qwen2.5-1.5B"
FREEZE_LAYERS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_STEPS = 1000
BATCH_SIZE = 2
GRAD_ACCUM = 2
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
N_TRAIN_SAMPLES = 5000
N_EVAL_SAMPLES = 500
ROUTER_STEPS = 300
ROUTER_LR = 1e-3
SEEDS = [42, 137, 2026]
RESULTS_PATH = Path("results/real/qwen_divergent_domains.json")
EVAL_BATCHES = 50
DIVERGENCE_GAP_THRESHOLD = 0.1


# ============================================================================
# PackedChunkDataset — reused from kalavu/evaluate.py pattern
# ============================================================================

class PackedChunkDataset(Dataset):
    """
    Concatenates all texts into one stream, splits into fixed SEQ_LEN chunks.
    No padding. Every token is real content.
    Labels = input_ids (causal LM; HF models handle the shift internally).
    """
    def __init__(self, texts: list[str], tokenizer, seq_len: int = SEQ_LEN,
                 max_chars: int = 3000):
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


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(split: str, n_samples: int) -> list[str]:
    """
    Load Python functions from code_search_net (ungated, parquet-based).
    train split → training data; test split → held-out eval.
    """
    from datasets import load_dataset
    hf_split = "test" if split == "held_out" else "train"
    print(f"  Loading code ({hf_split}, n={n_samples}) from code_search_net python...")
    ds = load_dataset("code_search_net", "python", split=hf_split, streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("func_code_string", "")
        if len(content) <= 200:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} code samples")
    return texts


def load_fiction_texts(split: str, n_samples: int) -> list[str]:
    """Load Project Gutenberg books from emozilla/pg19."""
    from datasets import load_dataset
    # pg19 has train/validation/test splits
    hf_split = "test" if split == "held_out" else "train"
    print(f"  Loading fiction ({hf_split}, n={n_samples}) from emozilla/pg19...")
    ds = load_dataset("emozilla/pg19", split=hf_split, streaming=True)
    texts = []
    for item in ds:
        raw = item.get("text", "")
        # Truncate each book to first 3000 chars
        content = raw[:3000]
        if len(content) < 500:
            continue
        texts.append(content)
        if len(texts) >= n_samples:
            break
    print(f"    Loaded {len(texts)} fiction samples")
    return texts


# ============================================================================
# freeze_first_n_layers — reused from kalavu/train_hf.py
# ============================================================================

def freeze_first_n_layers(model, n: int):
    """Freeze embedding + first n transformer blocks."""
    for p in model.model.embed_tokens.parameters():
        p.requires_grad = False
    for i in range(n):
        for p in model.model.layers[i].parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# TwoExpertMoE — copied from fuse_and_eval_ablation.py
# ============================================================================

class TwoExpertMoE(nn.Module):
    """
    Sequence-level MoE over two specialist models.
    Router: mean of last hidden states from both experts -> small MLP -> 2 gates (softmax).
    Specialists are frozen; only router is trained.
    """
    def __init__(self, spec_a, spec_b, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        for p in self.spec_a.parameters():
            p.requires_grad_(False)
        for p in self.spec_b.parameters():
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 2, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach()             # (B, T, V)
        last_h = out.hidden_states[-1].detach()  # (B, T, H)
        h_pooled = last_h.mean(dim=1).float()    # (B, H)
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)

        h_avg = (h_a + h_b) / 2.0
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, 2)

        fused = gates[:, 0:1, None] * logits_a + gates[:, 1:2, None] * logits_b  # (B, T, V)

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

def train_specialist(model, tokenizer, domain: str, train_texts: list[str],
                     seed: int, device: str):
    set_seed(seed)
    freeze_first_n_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = PackedChunkDataset(train_texts, tokenizer, seq_len=SEQ_LEN)
    print(f"  {domain}_train_chunks={len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

    step = 0
    accum = 0
    total_loss = 0.0
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
        total_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum = 0
            step += 1

            if step % 100 == 0 or step == MAX_STEPS:
                avg = total_loss / step
                elapsed = time.time() - t0
                print(f"  [{domain}] step {step}/{MAX_STEPS} | loss {avg:.4f} | {elapsed:.0f}s")

    print(f"  {domain} training done in {time.time()-t0:.0f}s")
    return model


# ============================================================================
# Eval helpers
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


# ============================================================================
# Divergence check
# ============================================================================

def check_divergence(spec_code, spec_fiction, base_model,
                     eval_code: PackedChunkDataset,
                     eval_fiction: PackedChunkDataset,
                     device: str, seed: int):
    """
    Verify specialists have actually diverged from base before proceeding to fusion.
    Returns (passed, checks_dict, losses_dict).
    """
    print(f"\nDIVERGENCE CHECK (seed={seed}):")

    losses = {}
    for name, model in [("base", base_model), ("code", spec_code), ("fiction", spec_fiction)]:
        losses[name] = {
            "code":    eval_loss(model, eval_code, device),
            "fiction": eval_loss(model, eval_fiction, device),
        }

    base_code    = losses["base"]["code"]
    base_fiction = losses["base"]["fiction"]
    spec_c_code  = losses["code"]["code"]
    spec_c_fict  = losses["code"]["fiction"]
    spec_f_code  = losses["fiction"]["code"]
    spec_f_fict  = losses["fiction"]["fiction"]

    checks = {
        "code_beats_base_on_code":        spec_c_code  < base_code,
        "fiction_beats_base_on_fiction":  spec_f_fict  < base_fiction,
        "code_worse_than_base_on_fiction": spec_c_fict > base_fiction,
        "divergence_gap":                 (spec_c_fict - spec_c_code) > DIVERGENCE_GAP_THRESHOLD,
    }

    def delta_pct(new, base):
        return (new - base) / base * 100

    def check_sym(v):
        return "\u2713" if v else "\u2717"

    print(f"  Code specialist on code:         {spec_c_code:.4f} "
          f"(base: {base_code:.4f}, d = {delta_pct(spec_c_code, base_code):+.2f}%)  "
          f"{check_sym(checks['code_beats_base_on_code'])}")
    print(f"  Code specialist on fiction:      {spec_c_fict:.4f} "
          f"(base: {base_fiction:.4f}, d = {delta_pct(spec_c_fict, base_fiction):+.2f}%)  "
          f"{check_sym(checks['code_worse_than_base_on_fiction'])}")
    print(f"  Fiction specialist on fiction:   {spec_f_fict:.4f} "
          f"(base: {base_fiction:.4f}, d = {delta_pct(spec_f_fict, base_fiction):+.2f}%)  "
          f"{check_sym(checks['fiction_beats_base_on_fiction'])}")
    print(f"  Divergence gap (code cross-domain): "
          f"{spec_c_fict - spec_c_code:.4f} > {DIVERGENCE_GAP_THRESHOLD}  "
          f"{check_sym(checks['divergence_gap'])}")

    passed = all(checks.values())
    print(f"  All checks passed: {'YES' if passed else 'NO'}")

    return passed, checks, losses


# ============================================================================
# Weight averaging
# ============================================================================

def weight_average(spec_a, spec_b):
    avg = copy.deepcopy(spec_a)
    state_a = spec_a.state_dict()
    state_b = spec_b.state_dict()
    avg_state = {k: (state_a[k].float() + state_b[k].float()) / 2.0 for k in state_a}
    avg_state = {k: v.to(torch.bfloat16) for k, v in avg_state.items()}
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Router training
# ============================================================================

def train_router(moe: TwoExpertMoE, train_code: PackedChunkDataset,
                 train_fiction: PackedChunkDataset, device: str):
    # Mixed dataset: interleave chunks from both domains
    combined = PackedChunkDataset.__new__(PackedChunkDataset)
    combined.chunks = train_code.chunks + train_fiction.chunks

    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=BATCH_SIZE * GRAD_ACCUM, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()

    print(f"\n  Training router ({ROUTER_STEPS} steps)...")
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
# Print results table
# ============================================================================

def print_results_table(results: dict, seed: int):
    domains = ["code", "fiction", "mixed"]
    col_w = 10
    print(f"\nFUSION RESULTS \u2014 Held-Out Eval (seed={seed}):")
    header = f"{'Model':<25}" + "".join(f"{d:>{col_w}}" for d in ["Code", "Fiction", "Mixed", "Average"])
    print(header)
    print("\u2500" * len(header))

    for name, losses in results.items():
        avg = sum(losses[d] for d in domains) / len(domains)
        row = f"{name:<25}" + "".join(f"{losses[d]:>{col_w}.4f}" for d in domains) + f"{avg:>{col_w}.4f}"
        print(row)


# ============================================================================
# Single seed run
# ============================================================================

def run_seed(seed: int, tokenizer, device: str,
             train_code: list[str], train_fiction: list[str],
             eval_code_texts: list[str], eval_fiction_texts: list[str]) -> dict | None:
    print(f"\n{'#'*70}")
    print(f"# SEED {seed}")
    print(f"{'#'*70}")

    # Build packed datasets
    print("\nBuilding packed datasets...")
    train_code_ds = PackedChunkDataset(train_code, tokenizer, seq_len=SEQ_LEN)
    train_fict_ds = PackedChunkDataset(train_fiction, tokenizer, seq_len=SEQ_LEN)
    eval_code_ds  = PackedChunkDataset(eval_code_texts, tokenizer, seq_len=SEQ_LEN)
    eval_fict_ds  = PackedChunkDataset(eval_fiction_texts, tokenizer, seq_len=SEQ_LEN)

    # Build mixed eval set
    eval_mixed_ds = PackedChunkDataset.__new__(PackedChunkDataset)
    eval_mixed_ds.chunks = eval_code_ds.chunks + eval_fict_ds.chunks

    print(f"  code_train_chunks={len(train_code_ds)}, fiction_train_chunks={len(train_fict_ds)}")
    print(f"  code_eval_chunks={len(eval_code_ds)}, fiction_eval_chunks={len(eval_fict_ds)}")

    if len(train_code_ds) < 1000 or len(train_fict_ds) < 1000:
        print("WARNING: fewer than 1000 training chunks — results may be unreliable")

    # Load base model fresh for each seed
    print(f"\nLoading base model: {MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()

    hidden_size = base_model.config.hidden_size

    # Train code specialist
    print(f"\nTraining code specialist (seed={seed})...")
    spec_code = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    train_specialist(spec_code, tokenizer, "code", train_code, seed, device)
    spec_code.eval()

    # Train fiction specialist
    print(f"\nTraining fiction specialist (seed={seed})...")
    spec_fiction = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    train_specialist(spec_fiction, tokenizer, "fiction", train_fiction, seed + 1000, device)
    spec_fiction.eval()

    # Divergence check — must pass before fusion
    passed, checks, div_losses = check_divergence(
        spec_code, spec_fiction, base_model,
        eval_code_ds, eval_fict_ds, device, seed,
    )

    if not passed:
        print(f"\nDIVERGENCE CHECK FAILED for seed={seed}. Skipping fusion.")
        print("Diagnostics:")
        for check_name, result in checks.items():
            print(f"  {check_name}: {'PASS' if result else 'FAIL'}")
        # Clean up
        del spec_code, spec_fiction, base_model
        torch.cuda.empty_cache()
        return {
            "seed": seed,
            "divergence_passed": False,
            "divergence_checks": checks,
            "divergence_losses": {
                name: {d: round(v, 6) for d, v in ls.items()}
                for name, ls in div_losses.items()
            },
        }

    # Weight average baseline
    print("\nComputing weight average...")
    weight_avg_model = weight_average(spec_code, spec_fiction)
    weight_avg_model = weight_avg_model.to(device)

    # Build MoE and train router
    print("\nBuilding MoE...")
    moe = TwoExpertMoE(spec_code, spec_fiction, hidden_size).to(device)
    train_router(moe, train_code_ds, train_fict_ds, device)
    moe.eval()

    # Eval all 5 variants on held-out data
    print("\nEvaluating all variants on held-out data...")
    eval_sets = {"code": eval_code_ds, "fiction": eval_fict_ds, "mixed": eval_mixed_ds}

    results = {}
    for name, model, is_fused in [
        ("Base model",           base_model,       False),
        ("Specialist (code)",    spec_code,         False),
        ("Specialist (fiction)", spec_fiction,      False),
        ("Weight averaged",      weight_avg_model,  False),
        ("MoE fused",            moe,               True),
    ]:
        bs = 2 if is_fused else 4
        results[name] = {}
        for domain, ds in eval_sets.items():
            loss = eval_loss(model, ds, device, batch_size=bs, is_fused=is_fused)
            results[name][domain] = round(loss, 6)

    print_results_table(results, seed)

    # Compute improvement over best individual on mixed held-out
    best_individual = min(
        results["Specialist (code)"]["mixed"],
        results["Specialist (fiction)"]["mixed"],
    )
    fused_mixed = results["MoE fused"]["mixed"]
    improvement_pct = (best_individual - fused_mixed) / best_individual * 100

    print(f"\nImprovement over best individual (mixed): {improvement_pct:+.1f}%")

    # Clean up VRAM
    del moe, weight_avg_model, base_model, spec_code, spec_fiction
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "divergence_passed": True,
        "divergence_checks": checks,
        "divergence_losses": {
            name: {d: round(v, 6) for d, v in ls.items()}
            for name, ls in div_losses.items()
        },
        "eval_heldout": {k: {d: round(v, 6) for d, v in vs.items()} for k, vs in results.items()},
        "improvement_pct": round(improvement_pct, 4),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU: Real-Model Specialist Divergence Experiment")
    print("=" * 70)
    print(f"Model:   {MODEL_ID}")
    print(f"Domains: code (code_search_net python) vs fiction (emozilla/pg19)")
    print(f"Steps:   {MAX_STEPS} per specialist")
    print(f"Seeds:   {SEEDS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be extremely slow.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data once — same data across all seeds, only model init varies
    print("\nLoading training data...")
    train_code    = load_code_texts("train",    N_TRAIN_SAMPLES)
    train_fiction = load_fiction_texts("train", N_TRAIN_SAMPLES)

    print("\nLoading held-out evaluation data...")
    eval_code    = load_code_texts("held_out",    N_EVAL_SAMPLES)
    eval_fiction = load_fiction_texts("held_out", N_EVAL_SAMPLES)

    all_results = []
    for seed in SEEDS:
        result = run_seed(
            seed, tokenizer, device,
            train_code, train_fiction,
            eval_code, eval_fiction,
        )
        if result is not None:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY ACROSS ALL SEEDS:")
    print(f"{'Seed':<8} {'Divergence':<14} {'Improvement':>12}")
    print("-" * 36)

    passed_results = [r for r in all_results if r.get("divergence_passed")]
    for r in all_results:
        div = "YES" if r.get("divergence_passed") else "NO"
        imp = f"{r['improvement_pct']:+.1f}%" if r.get("divergence_passed") else "N/A"
        print(f"{r['seed']:<8} {div:<14} {imp:>12}")

    if passed_results:
        improvements = [r["improvement_pct"] for r in passed_results]
        mean_imp = statistics.mean(improvements)
        std_imp = statistics.stdev(improvements) if len(improvements) > 1 else 0.0
        print("-" * 36)
        print(f"{'Mean':<8} {'':14} {mean_imp:>+11.1f}%")
        if len(improvements) > 1:
            print(f"{'Std':<8} {'':14} {std_imp:>11.1f}%")

        print(f"\nINTERPRETATION:")
        if mean_imp > 20:
            print(f"  A: Strong result ({mean_imp:.1f}%) — fusion works on divergent real-model domains.")
        elif mean_imp > 5:
            print(f"  B: Modest result ({mean_imp:.1f}%) — fusion works but benefit is incremental.")
        else:
            print(f"  C: Weak result ({mean_imp:.1f}%) — fusion not working on real models yet.")
    else:
        print("\nNo seeds passed divergence check — increase MAX_STEPS or check data loading.")

    # Save
    import time as time_mod
    output = {
        "experiment": "qwen_divergent_domains",
        "model_id": MODEL_ID,
        "domains": ["code", "fiction"],
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "seq_len": SEQ_LEN,
            "n_train_samples": N_TRAIN_SAMPLES,
            "n_eval_samples": N_EVAL_SAMPLES,
            "router_steps": ROUTER_STEPS,
        },
        "seeds": SEEDS,
        "per_seed": all_results,
        "summary": {
            "seeds_passed_divergence": len(passed_results),
            "improvement_mean": round(statistics.mean([r["improvement_pct"] for r in passed_results]), 4) if passed_results else None,
            "improvement_std": round(statistics.stdev([r["improvement_pct"] for r in passed_results]), 4) if len(passed_results) > 1 else None,
        },
        "timestamp": time_mod.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
