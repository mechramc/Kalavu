#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-1B Maturity Sweep
=================================
Trains specialists at 4 checkpoints (step5000, step20000, step50000, step143000)
and combines with step10000 results from kalavai_pythia_1b_experiment.py to
produce the maturity curve for Pythia-1B.

Also produces the combined 410M vs 1B maturity figure — the paper hero figure.

Checkpoints tested:
  step5000   (~3.5% of full training)
  step10000  (~7.0%) <- loaded from main_result_summary.json, not retrained
  step20000  (~14.0%)
  step50000  (~35.0%)
  step143000 (~100.0%)

Paper narrative:
  Early training (step10000): highest fusion improvement
  As base knowledge increases: improvement diminishes
  This mirrors the synthetic -> real -> Qwen arc
"""

import copy
import json
import math
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

MODEL_BASE = "EleutherAI/pythia-1b"
REVISIONS = [
    ("step5000",   5000,   3.5),
    # step10000 comes from kalavai_pythia_1b_experiment.py — reuse those checkpoints
    ("step20000",  20000,  14.0),
    ("step50000",  50000,  35.0),
    ("step143000", 143000, 100.0),
]
FREEZE_LAYERS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE = 2048
NUM_LAYERS = 16
DOMAINS = ["code", "science", "fiction"]
SEEDS_PHASE_A = [42]
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50

RESULTS_DIR = Path("results/pythia/pythia_1b/maturity_sweep")
CHECKPOINT_DIR = Path("checkpoints/pythia/pythia_1b/maturity_sweep")
FIGURES_DIR = Path("figures/pythia")

# Step10000 results from the main 1B experiment — loaded from file, not recomputed
MAIN_1B_RESULTS_PATH = Path("results/pythia/pythia_1b/main_result_summary.json")

# 410M maturity sweep results — for combined figure
MATURITY_SWEEP_410M_PATH = Path("results/pythia/maturity_sweep_410m/summary.json")

# Qwen result for combined figure annotation
QWEN_IMPROVEMENT_PCT = -1.0


# ============================================================================
# PackedChunkDataset — copied from kalavai_qwen_divergent_domains.py
# ============================================================================

class PackedChunkDataset(Dataset):
    """
    Concatenates all texts into one stream, splits into fixed SEQ_LEN chunks.
    No padding. Every token is real content.
    Labels = input_ids (causal LM; HF models handle the shift internally).
    """
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
    """Create a PackedChunkDataset-like object from pre-split chunks."""
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n_samples: int) -> list[str]:
    """Load Python functions from code_search_net."""
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
    """
    SciQ: support + question + correct_answer.
    CRITICAL: the `support` field is a long scientific passage — do NOT strip it.
    """
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
    """Load Project Gutenberg books from emozilla/pg19, first 5000 chars each."""
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


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    """Split packed chunks 80/10/10 into train, indist, held_out."""
    n = len(chunks)
    train_end = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]


# ============================================================================
# Pythia-specific freezing
# ============================================================================

def freeze_first_n_layers(model, n: int):
    """Freeze embedding + first n transformer blocks (GPT-NeoX architecture)."""
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# ThreeExpertMoE
# ============================================================================

class ThreeExpertMoE(nn.Module):
    """
    Sequence-level MoE over three specialist models.
    Router: single linear layer (HIDDEN_SIZE -> 3) — matches paper spec.
    Specialists are frozen; only router is trained.
    """
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in self.spec_a.parameters():
            p.requires_grad_(False)
        for p in self.spec_b.parameters():
            p.requires_grad_(False)
        for p in self.spec_c.parameters():
            p.requires_grad_(False)
        # Single linear layer router, matches 1B spec
        self.router = nn.Linear(hidden_size, 3, bias=False)

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
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)

        h_avg = (h_a + h_b + h_c) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, 3)

        fused = (
            gates[:, 0:1, None] * logits_a
            + gates[:, 1:2, None] * logits_b
            + gates[:, 2:3, None] * logits_c
        )  # (B, T, V)

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

def train_specialist(model, domain: str, train_chunks: list, tokenizer,
                     seed: int, device: str, log_every: int = 50) -> list[float]:
    """Train a specialist. Returns loss history for plotting."""
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
    loss_history = []   # (step, avg_loss)
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

            # Warmup: linearly scale LR
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
# Eval helpers
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset, device: str,
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
# Weight averaging (3-way)
# ============================================================================

def weight_average_three(spec_a, spec_b, spec_c):
    """Three-way weight average of unfrozen-layer weights."""
    avg = copy.deepcopy(spec_a)
    sa = spec_a.state_dict()
    sb = spec_b.state_dict()
    sc = spec_c.state_dict()
    avg_state = {
        k: ((sa[k].float() + sb[k].float() + sc[k].float()) / 3.0).to(torch.bfloat16)
        for k in sa
    }
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Router training
# ============================================================================

def train_router(moe: ThreeExpertMoE, train_datasets: dict, device: str):
    """Train the router on mixed data from all three domains."""
    all_chunks = []
    for ds in train_datasets.values():
        all_chunks.extend(ds.chunks)
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
# One checkpoint: load data, train specialists, fuse, eval
# ============================================================================

def run_one_checkpoint(revision: str, training_pct: float, tokenizer,
                       all_domain_chunks: dict, device: str) -> dict:
    """
    Run full specialist-train + fusion pipeline for one revision.
    Returns result dict with improvement_pct and eval_matrix.
    """
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {revision} ({training_pct:.1f}% of full training)")
    print(f"{'='*70}")

    ckpt_base = CHECKPOINT_DIR / revision
    ckpt_base.mkdir(parents=True, exist_ok=True)

    seed = SEEDS_PHASE_A[0]  # seed=42 only

    # Build eval datasets
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)
    train_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                  for d in DOMAINS}

    # Eval base at this revision
    print(f"\nLoading base model {MODEL_BASE} @ {revision}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, revision=revision,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()

    base_losses = {}
    for domain, ds in held_out_sets.items():
        l = eval_loss(base_model, ds, device)
        base_losses[domain] = round(l, 6)
        print(f"  Base [{domain:8s}] @ {revision}: {l:.4f}")

    del base_model
    torch.cuda.empty_cache()

    # Train specialists
    specialists = {}
    for domain in DOMAINS:
        ckpt_path = ckpt_base / f"{domain}_specialist_seed{seed}.pt"

        if ckpt_path.exists():
            print(f"\nLoading cached {domain} specialist from {ckpt_path}...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE, revision=revision,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            spec.load_state_dict(torch.load(ckpt_path, map_location=device))
            spec.eval()
        else:
            print(f"\nTraining {domain} specialist @ {revision} (seed={seed})...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE, revision=revision,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            train_specialist(spec, domain, all_domain_chunks[domain]["train"],
                             tokenizer, seed, device)
            spec.eval()
            torch.save(spec.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        specialists[domain] = spec

    # 3-way weight average
    print(f"\nComputing 3-way weight average @ {revision}...")
    weight_avg_model = weight_average_three(
        specialists["code"], specialists["science"], specialists["fiction"]
    ).to(device)

    # Build MoE and train router
    print(f"\nBuilding ThreeExpertMoE @ {revision}...")
    moe = ThreeExpertMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    train_router(moe, train_sets, device)
    moe.eval()

    # Eval all 6 variants
    print(f"\nEvaluating all variants @ {revision}...")
    eval_matrix = {}
    for model_key, model, is_fused in [
        ("base",          None,                       False),   # base reloaded below
        ("code_spec",     specialists["code"],        False),
        ("science_spec",  specialists["science"],     False),
        ("fiction_spec",  specialists["fiction"],     False),
        ("weight_avg",    weight_avg_model,           False),
        ("moe",           moe,                        True),
    ]:
        if model_key == "base":
            # Reload base for eval
            bm = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE, revision=revision,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            bm.eval()
            eval_matrix[model_key] = {}
            for domain, ds in held_out_sets.items():
                l = eval_loss(bm, ds, device, batch_size=4, is_fused=False)
                eval_matrix[model_key][domain] = round(l, 6)
            del bm
            torch.cuda.empty_cache()
        else:
            bs = 2 if is_fused else 4
            eval_matrix[model_key] = {}
            for domain, ds in held_out_sets.items():
                l = eval_loss(model, ds, device, batch_size=bs, is_fused=is_fused)
                eval_matrix[model_key][domain] = round(l, 6)

    # Print results table
    col_w = 11
    domains_display = ["code", "science", "fiction", "mixed"]
    print(f"\n{'Model':<20}" + "".join(f"{d.capitalize():>{col_w}}" for d in domains_display)
          + f"{'Average':>{col_w}}")
    print("-" * (20 + col_w * (len(domains_display) + 1)))
    for mk, losses in eval_matrix.items():
        avg = sum(losses[d] for d in domains_display) / len(domains_display)
        row = f"{mk:<20}" + "".join(f"{losses[d]:>{col_w}.4f}" for d in domains_display)
        row += f"{avg:>{col_w}.4f}"
        print(row)

    # Compute improvement
    best_individual_mixed = min(
        eval_matrix["code_spec"]["mixed"],
        eval_matrix["science_spec"]["mixed"],
        eval_matrix["fiction_spec"]["mixed"],
    )
    moe_mixed = eval_matrix["moe"]["mixed"]
    improvement_pct = (best_individual_mixed - moe_mixed) / best_individual_mixed * 100
    print(f"\nImprovement over best individual (mixed): {improvement_pct:+.1f}%")
    print(f"[kalavai] 1b maturity sweep: step={revision} improvement={improvement_pct:.1f}%")

    result = {
        "revision": revision,
        "training_pct": training_pct,
        "seed": seed,
        "base_losses": base_losses,
        "eval_heldout": eval_matrix,
        "improvement_pct": round(improvement_pct, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / f"result_{revision}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    del weight_avg_model, moe
    for m in specialists.values():
        del m
    torch.cuda.empty_cache()

    return result


# ============================================================================
# Load step10000 result from main 1B experiment
# ============================================================================

def load_step10000_result() -> dict:
    """Load step10000 improvement from main 1B experiment summary."""
    if not MAIN_1B_RESULTS_PATH.exists():
        print(f"WARNING: {MAIN_1B_RESULTS_PATH} not found.")
        print("  Run kalavai_pythia_1b_experiment.py first, or set improvement_pct manually.")
        # Return a placeholder so sweep can still run
        return {
            "revision": "step10000",
            "training_pct": 7.0,
            "seed": 42,
            "improvement_pct": None,
            "source": "MISSING — run kalavai_pythia_1b_experiment.py first",
        }

    with open(MAIN_1B_RESULTS_PATH) as f:
        summary = json.load(f)

    # Extract seed=42 improvement
    per_seed = summary.get("per_seed_fusion", {})
    seed42_result = per_seed.get(42, per_seed.get("42", {}))
    improvement_pct = seed42_result.get("improvement_pct", None)

    if improvement_pct is None:
        # Try top-level summary
        improvement_pct = summary.get("summary", {}).get("improvement_mean_pct", None)

    print(f"  Loaded step10000 result from {MAIN_1B_RESULTS_PATH}")
    print(f"  improvement_pct (seed=42) = {improvement_pct}")

    return {
        "revision": "step10000",
        "training_pct": 7.0,
        "seed": 42,
        "improvement_pct": improvement_pct,
        "source": str(MAIN_1B_RESULTS_PATH),
    }


# ============================================================================
# Load 410M maturity sweep results
# ============================================================================

def load_410m_sweep() -> list[dict]:
    """Load 410M maturity sweep results for combined figure."""
    if not MATURITY_SWEEP_410M_PATH.exists():
        print(f"  WARNING: {MATURITY_SWEEP_410M_PATH} not found.")
        print("  410M curve will be omitted from combined figure.")
        return []

    with open(MATURITY_SWEEP_410M_PATH) as f:
        data = json.load(f)

    # Expected format: list of {revision, training_pct, improvement_pct}
    # or dict with "checkpoints" key
    if isinstance(data, list):
        return data
    elif "checkpoints" in data:
        return data["checkpoints"]
    elif "results" in data:
        return data["results"]
    else:
        # Try to extract from top-level keys
        print(f"  WARNING: Unexpected format in {MATURITY_SWEEP_410M_PATH}, trying to parse...")
        results = []
        for k, v in data.items():
            if isinstance(v, dict) and "improvement_pct" in v:
                results.append(v)
        return results


# ============================================================================
# Figures
# ============================================================================

def save_maturity_curve_1b(sweep_results: list[dict]):
    """
    fig_maturity_curve_1b.png — same format as fig_maturity_curve_410m.png but for 1B.
    X-axis: training_pct, Y-axis: improvement_pct.
    Mark step10000 with star annotation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Sort by training_pct
        sorted_results = sorted(sweep_results, key=lambda r: r["training_pct"])
        xs = [r["training_pct"] for r in sorted_results]
        ys = [r["improvement_pct"] if r["improvement_pct"] is not None else 0.0
              for r in sorted_results]
        revisions = [r["revision"] for r in sorted_results]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(xs, ys, "r-o", linewidth=2, markersize=8, label="Pythia-1B")
        ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5, linewidth=1)

        # Annotate each point
        for x, y, rev in zip(xs, ys, revisions):
            ax.annotate(rev, (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)

        # Star annotation for step10000
        step10000_results = [r for r in sorted_results if r["revision"] == "step10000"]
        if step10000_results:
            sx = step10000_results[0]["training_pct"]
            sy = step10000_results[0]["improvement_pct"] or 0.0
            ax.plot(sx, sy, "r*", markersize=18, zorder=5, label="step10000 (from main exp)")

        ax.set_xlabel("% of Full Training")
        ax.set_ylabel("Improvement over Best Individual (%)")
        ax.set_title("Pythia-1B: Fusion Improvement vs. Base Model Maturity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_maturity_curve_1b.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save maturity curve 1B figure: {e}")


def save_maturity_curve_combined(sweep_1b: list[dict], sweep_410m: list[dict]):
    """
    fig_maturity_curve_combined.png — paper hero figure.
    Two curves: 410M (blue circles) vs 1B (red squares).
    Shaded KALAVAI target regime (0-20%).
    Qwen as gray diamond at x=100.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(11, 6))

        # Sort both curves
        sorted_1b = sorted(sweep_1b, key=lambda r: r["training_pct"])
        xs_1b = [r["training_pct"] for r in sorted_1b]
        ys_1b = [r["improvement_pct"] if r["improvement_pct"] is not None else 0.0
                 for r in sorted_1b]

        sorted_410m = sorted(sweep_410m, key=lambda r: r.get("training_pct", 0))
        xs_410m = [r.get("training_pct", 0) for r in sorted_410m]
        ys_410m = [r.get("improvement_pct", 0) if r.get("improvement_pct") is not None else 0.0
                   for r in sorted_410m]

        # 1B curve (red squares)
        if xs_1b:
            ax.plot(xs_1b, ys_1b, "rs-", linewidth=2, markersize=9,
                    label="Pythia-1B", color="#c0392b", zorder=3)
            # Star for step10000 on 1B curve
            step10k_1b = [r for r in sorted_1b if r["revision"] == "step10000"]
            if step10k_1b:
                sx = step10k_1b[0]["training_pct"]
                sy = step10k_1b[0]["improvement_pct"] or 0.0
                ax.plot(sx, sy, "r*", markersize=20, zorder=5, color="#c0392b",
                        label="step10000 (1B, main exp.)")

        # 410M curve (blue circles)
        if xs_410m:
            ax.plot(xs_410m, ys_410m, "bo-", linewidth=2, markersize=9,
                    label="Pythia-410M", color="#2980b9", zorder=3)

        # Qwen marker (gray diamond, separate model family)
        ax.plot(100, QWEN_IMPROVEMENT_PCT, "gD", markersize=12,
                color="#7f8c8d", zorder=4,
                label=f"Qwen-1.5B (different family, {QWEN_IMPROVEMENT_PCT:+.1f}%)")
        ax.annotate("Qwen-1.5B\n(different family)",
                    (100, QWEN_IMPROVEMENT_PCT),
                    textcoords="offset points", xytext=(-60, -30),
                    ha="center", fontsize=8, color="#7f8c8d",
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))

        # Horizontal zero line
        ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5, linewidth=1.2)

        # Shaded KALAVAI target regime (early training zone 0-20%)
        ax.axvspan(0, 20, alpha=0.08, color="gold", zorder=1)
        ax.text(10, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 5,
                "KALAVAI\ntarget regime",
                ha="center", fontsize=9, color="#b7950b", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="#b7950b", alpha=0.7))

        ax.set_xlabel("% of Full Training", fontsize=12)
        ax.set_ylabel("Fusion Improvement over Best Individual (%)", fontsize=12)
        ax.set_title("Fusion Improvement vs. Base Model Maturity", fontsize=14)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 105)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_maturity_curve_combined.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save combined maturity curve figure: {e}")


def save_maturity_comparison_bar(sweep_1b: list[dict], sweep_410m: list[dict]):
    """
    fig_maturity_curve_comparison_bar.png — grouped bar chart comparing 410M vs 1B.
    At matching checkpoints. Blue=410M, red=1B.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Build lookup by revision
        lookup_1b = {r["revision"]: r.get("improvement_pct", 0.0) or 0.0
                     for r in sweep_1b}
        lookup_410m = {r.get("revision", r.get("step", "")): r.get("improvement_pct", 0.0) or 0.0
                       for r in sweep_410m}

        # All revisions in order
        all_revisions = ["step5000", "step10000", "step20000", "step50000", "step143000"]
        labels = ["step5000\n(3.5%)", "step10000\n(7%)", "step20000\n(14%)",
                  "step50000\n(35%)", "step143000\n(100%)"]

        vals_1b   = [lookup_1b.get(rev, 0.0) for rev in all_revisions]
        vals_410m = [lookup_410m.get(rev, 0.0) for rev in all_revisions]

        x = np.arange(len(all_revisions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 5))
        bars_410m = ax.bar(x - width/2, vals_410m, width, label="Pythia-410M",
                           color="#2980b9", alpha=0.85)
        bars_1b   = ax.bar(x + width/2, vals_1b,   width, label="Pythia-1B",
                           color="#c0392b", alpha=0.85)

        # Value labels on bars
        for bar in list(bars_410m) + list(bars_1b):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                    f"{h:+.1f}%", ha="center", va="bottom", fontsize=8)

        ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fusion Improvement over Best Individual (%)")
        ax.set_title("Scaling Behavior: 410M vs 1B Fusion Improvement at Matching Checkpoints")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_maturity_curve_comparison_bar.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save comparison bar figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-1B Maturity Sweep")
    print("=" * 70)
    print(f"Model:    {MODEL_BASE}")
    print(f"Revisions: {[r[0] for r in REVISIONS]} + step10000 (from main experiment)")
    print(f"Domains:  {DOMAINS}")
    print(f"Steps:    {MAX_STEPS} per specialist (freeze_layers={FREEZE_LAYERS})")
    print(f"Seeds:    {SEEDS_PHASE_A} (phase A — single seed)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be extremely slow.")

    # Load tokenizer once (same NeoX tokenizer across all 1B checkpoints)
    print(f"\nLoading tokenizer from {MODEL_BASE} (step143000)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, revision="step143000")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and pack data once — same across all checkpoints
    print("\nLoading and packing data (shared across all checkpoints)...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, indist_c, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {
            "train": train_c,
            "indist": indist_c,
            "held_out": held_c,
        }
        print(f"  {domain}: total={len(ds_full)}, "
              f"train={len(train_c)}, indist={len(indist_c)}, held_out={len(held_c)}")
        if len(train_c) < 2000:
            print(f"  WARNING: {domain} has <2000 train chunks — results may be unreliable!")

    # Step10000 — load from main experiment, do NOT retrain
    print("\n" + "=" * 70)
    print("STEP10000: Loading from main 1B experiment (not retraining)")
    print("=" * 70)
    step10000_result = load_step10000_result()

    # Run all other checkpoints
    sweep_results = [step10000_result]  # include step10000 in sweep
    revision_results = {}

    for revision, step_num, training_pct in REVISIONS:
        result = run_one_checkpoint(
            revision=revision,
            training_pct=training_pct,
            tokenizer=tokenizer,
            all_domain_chunks=all_domain_chunks,
            device=device,
        )
        sweep_results.append(result)
        revision_results[revision] = result

    # Load 410M sweep for combined figure
    print("\nLoading 410M maturity sweep results for combined figure...")
    sweep_410m = load_410m_sweep()
    if sweep_410m:
        print(f"  Loaded {len(sweep_410m)} 410M checkpoints")
    else:
        print("  No 410M results found — combined figure will show 1B only")

    # Save summary
    summary = {
        "experiment": "pythia_1b_maturity_sweep",
        "model_base": MODEL_BASE,
        "domains": DOMAINS,
        "seeds": SEEDS_PHASE_A,
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "seq_len": SEQ_LEN,
            "warmup_fraction": WARMUP_FRACTION,
            "n_samples_per_domain": N_SAMPLES_PER_DOMAIN,
            "router_steps": ROUTER_STEPS,
            "eval_batches": EVAL_BATCHES,
            "hidden_size": HIDDEN_SIZE,
        },
        "checkpoints": sorted(sweep_results, key=lambda r: r["training_pct"]),
        "step10000_source": step10000_result.get("source", str(MAIN_1B_RESULTS_PATH)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Print sweep table
    print(f"\n{'Revision':<15} {'Training%':>12} {'Improvement':>14}")
    print("-" * 45)
    for r in sorted(sweep_results, key=lambda r: r["training_pct"]):
        imp = r["improvement_pct"]
        imp_str = f"{imp:>+13.1f}%" if imp is not None else "      N/A (missing)"
        print(f"{r['revision']:<15} {r['training_pct']:>11.1f}% {imp_str}")

    # Generate figures
    print("\nGenerating figures...")

    print("\nFigure 1: fig_maturity_curve_1b.png")
    save_maturity_curve_1b(sweep_results)

    print("\nFigure 2: fig_maturity_curve_combined.png (paper hero figure)")
    save_maturity_curve_combined(sweep_results, sweep_410m)

    print("\nFigure 3: fig_maturity_curve_comparison_bar.png")
    save_maturity_comparison_bar(sweep_results, sweep_410m)

    print("\n[kalavai] 1b maturity sweep complete -- combined figure generated")

    print("\n" + "=" * 70)
    print("Maturity sweep complete!")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
