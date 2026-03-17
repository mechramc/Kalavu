#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-410M Freeze Depth Ablation
==========================================
Sweeps freeze_layers in [0, 2, 4, 6, 8, 12] with seed=42.
Uses the SIMPLE Linear(1024, 3) router to isolate freeze depth as the variable.
freeze=0 is the BTX baseline (no shared backbone).
freeze=4 reuses the main experiment checkpoints (skips retraining).

Phase 1: seed=42 sweep over all 6 freeze depths
Phase 2: 3 seeds on the best 2 depths (besides freeze=4) for error bars
"""

import json
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
# Config — specialist training matches main experiment exactly
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
LR = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE = 1024
DOMAINS = ["code", "science", "fiction"]

ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
N_SAMPLES_PER_DOMAIN = 3000

FREEZE_SWEEP = [0, 2, 4, 6, 8, 12]
SEEDS_SWEEP = [42]           # Phase 1: seed=42 only
SEEDS_MULTISEED = [42, 137, 2026]  # Phase 2: top-2 depths get all 3 seeds

CHECKPOINT_DIR = Path("checkpoints/pythia")
RESULTS_DIR = Path("results/pythia")
FIGURES_DIR = Path("figures/pythia")

# freeze=4/seed=42 already trained in main experiment — reuse those
MAIN_EXP_FREEZE = 4
MAIN_EXP_SEED = 42


# ============================================================================
# Dataset
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_dataset_from_chunks(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks, train_frac=0.8, indist_frac=0.1):
    n = len(chunks)
    return (chunks[:int(n * train_frac)],
            chunks[int(n * train_frac):int(n * (train_frac + indist_frac))],
            chunks[int(n * (train_frac + indist_frac)):])


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train", streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) > 200:
            texts.append(content)
        if len(texts) >= n: break
    return texts


def load_science_texts(n):
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("support", "") + "\n" + item.get("question", "") + "\n" + item.get("correct_answer", "")
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n: break
    return texts


def load_fiction_texts(n):
    from datasets import load_dataset
    print(f"  Loading fiction (n={n})...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n: break
    return texts


# ============================================================================
# Freeze (GPT-NeoX architecture)
# ============================================================================

def freeze_first_n_layers(model, n: int):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"  freeze={n}: trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M ({pct:.1f}%)")
    return pct


# ============================================================================
# Simple linear router MoE
# ============================================================================

class SimpleLinearMoE(nn.Module):
    """MoE with single-layer linear router (what the original spec called for)."""
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run(self.spec_a, input_ids)
        lb, hb = self._run(self.spec_b, input_ids)
        lc, hc = self._run(self.spec_c, input_ids)
        h_avg = (ha + hb + hc) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = (gates[:, 0:1, None] * la
                 + gates[:, 1:2, None] * lb
                 + gates[:, 2:3, None] * lc)
        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, gates


# ============================================================================
# Training
# ============================================================================

def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_specialist(model, domain, train_chunks, seed, device):
    set_seed(seed)
    model.train()
    dataset = make_dataset_from_chunks(train_chunks)
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
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS: break
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
            if step % 500 == 0 or step == MAX_STEPS:
                print(f"    [{domain}] step {step}/{MAX_STEPS} | loss {running_loss/step:.4f} | {time.time()-t0:.0f}s")

    print(f"    {domain} done in {time.time()-t0:.0f}s, final loss={running_loss/MAX_STEPS:.4f}")
    return model


def train_router(moe, combined_train, device):
    combined = make_dataset_from_chunks(combined_train)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({ROUTER_STEPS} steps)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")


@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=4, is_fused=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(ids, labels=lbl)
        else:
            loss = model(input_ids=ids, labels=lbl).loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


def compute_divergence_gap(specialists, held_out_sets, device):
    """
    Mean pairwise cross-domain loss gap: how much worse is each specialist
    on other domains vs its own domain. Higher = more divergence.
    """
    gaps = []
    for spec_domain, spec in specialists.items():
        own_loss = eval_loss(spec, held_out_sets[spec_domain], device)
        other_losses = [
            eval_loss(spec, held_out_sets[d], device)
            for d in DOMAINS if d != spec_domain
        ]
        gaps.append(sum(other_losses) / len(other_losses) - own_loss)
    return sum(gaps) / len(gaps)


# ============================================================================
# Figure
# ============================================================================

def save_freeze_ablation_figure(sweep_results, multiseed_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        freeze_vals = [r["freeze_layers"] for r in sweep_results]
        improvements = [r["improvement_pct"] for r in sweep_results]
        div_gaps = [r["divergence_gap"] for r in sweep_results]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Primary: improvement %
        line1, = ax1.plot(freeze_vals, improvements, "o-", color="#3498db",
                          linewidth=2.5, markersize=8, label="Improvement % (left)")

        # Add error bars for multi-seed points
        for freeze_n, ms in multiseed_results.items():
            fn = int(freeze_n)
            if fn in freeze_vals:
                idx = freeze_vals.index(fn)
                ax1.errorbar(fn, improvements[idx], yerr=ms["std"],
                             fmt="none", color="#2980b9", capsize=6, linewidth=2)

        # Secondary: divergence gap
        line2, = ax2.plot(freeze_vals, div_gaps, "s--", color="#e74c3c",
                          linewidth=2, markersize=7, alpha=0.8, label="Divergence gap (right)")

        # Annotations
        ax1.axvline(x=MAIN_EXP_FREEZE, color="#2ecc71", linestyle=":", linewidth=2, alpha=0.8)
        ax1.text(MAIN_EXP_FREEZE + 0.2, min(improvements) * 0.99,
                 "main exp\n(freeze=4)", fontsize=8, color="#27ae60")

        ax1.axvline(x=0, color="#95a5a6", linestyle=":", linewidth=2, alpha=0.8)
        ax1.text(0.2, min(improvements) * 0.99,
                 "BTX\nbaseline", fontsize=8, color="#7f8c8d")

        for i, (fv, imp) in enumerate(zip(freeze_vals, improvements)):
            ax1.annotate(f"{imp:+.1f}%", (fv, imp),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8)

        ax1.set_xlabel("Frozen Layers (out of 24)")
        ax1.set_ylabel("MoE Improvement over Best Individual (%)", color="#3498db")
        ax2.set_ylabel("Mean Divergence Gap (cross-domain loss - own-domain loss)", color="#e74c3c")
        ax1.set_title("Fusion Improvement vs Freeze Depth (Pythia-410M, simple linear router)")
        ax1.set_xticks(freeze_vals)
        ax1.grid(True, alpha=0.3)

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_ablation_freeze.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save figure: {e}")


# ============================================================================
# Single freeze depth run (one seed)
# ============================================================================

def run_freeze_depth(freeze_n, seed, all_domain_chunks, held_out_sets, device,
                     reuse_checkpoints=False):
    """Train 3 specialists at freeze_n, fuse with simple linear router, return results."""
    print(f"\n{'='*55}")
    print(f"freeze={freeze_n}, seed={seed}")
    print(f"{'='*55}")

    specialists = {}
    combined_train = []

    for domain in DOMAINS:
        ckpt_path = CHECKPOINT_DIR / f"freeze{freeze_n}_{domain}_seed{seed}.pt"
        combined_train.extend(all_domain_chunks[domain]["train"])

        if reuse_checkpoints and ckpt_path.exists():
            print(f"  Reusing checkpoint: {ckpt_path}")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
            ).to(device)
            spec.load_state_dict(torch.load(ckpt_path, map_location=device))
            spec.eval()
            specialists[domain] = spec
        else:
            print(f"  Training {domain} specialist (freeze={freeze_n}, seed={seed})...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
            ).to(device)
            freeze_first_n_layers(spec, freeze_n)
            train_specialist(spec, domain, all_domain_chunks[domain]["train"], seed, device)
            spec.eval()
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(spec.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")
            specialists[domain] = spec

    # Divergence check
    print(f"  Computing divergence gap...")
    div_gap = compute_divergence_gap(specialists, held_out_sets, device)
    print(f"  Divergence gap: {div_gap:.4f}")

    # Base model eval for improvement baseline
    print(f"  Loading base model for comparison...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base.eval()
    base_mixed = eval_loss(base, held_out_sets["mixed"], device)
    del base
    torch.cuda.empty_cache()

    best_individual = min(
        eval_loss(specialists[d], held_out_sets["mixed"], device) for d in DOMAINS
    )

    # Build MoE and train router
    print(f"  Building SimpleLinearMoE...")
    moe = SimpleLinearMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    train_router(moe, combined_train, device)
    moe.eval()

    moe_mixed = eval_loss(moe, held_out_sets["mixed"], device,
                          batch_size=2, is_fused=True)
    improvement = (best_individual - moe_mixed) / best_individual * 100

    print(f"\n  Results (freeze={freeze_n}, seed={seed}):")
    print(f"    Base mixed:       {base_mixed:.4f}")
    print(f"    Best individual:  {best_individual:.4f}")
    print(f"    MoE mixed:        {moe_mixed:.4f}")
    print(f"    Improvement:      {improvement:+.1f}%")
    print(f"    Divergence gap:   {div_gap:.4f}")

    del moe
    for s in specialists.values():
        del s
    torch.cuda.empty_cache()

    return {
        "freeze_layers": freeze_n,
        "seed": seed,
        "base_mixed_loss": round(base_mixed, 6),
        "best_individual_mixed": round(best_individual, 6),
        "moe_mixed_loss": round(moe_mixed, 6),
        "improvement_pct": round(improvement, 4),
        "divergence_gap": round(div_gap, 4),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-410M Freeze Depth Ablation")
    print("=" * 70)
    print(f"Sweep: freeze_layers={FREEZE_SWEEP}")
    print(f"Router: Simple Linear(1024, 3)")
    print(f"Phase 1: seed=42 across all depths")
    print(f"Phase 2: 3 seeds on best 2 depths")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer + data once
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting chunks (80/10/10)...")
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    # =========================================================================
    # PHASE 1: seed=42 sweep across all freeze depths
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: seed=42 sweep")
    print("=" * 70)

    sweep_results = []
    partial_path = RESULTS_DIR / "ablation_freeze_partial.json"

    for freeze_n in FREEZE_SWEEP:
        # freeze=4/seed=42: reuse main experiment checkpoints
        if freeze_n == MAIN_EXP_FREEZE:
            print(f"\nfreeze={freeze_n}: reusing main experiment checkpoints (seed=42)...")
            # Copy main exp checkpoints to freeze-labeled names if needed
            for domain in DOMAINS:
                src = CHECKPOINT_DIR / f"{domain}_specialist_seed42.pt"
                dst = CHECKPOINT_DIR / f"freeze{freeze_n}_{domain}_seed42.pt"
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy(src, dst)
                    print(f"  Copied {src.name} -> {dst.name}")

        result = run_freeze_depth(
            freeze_n, seed=42, all_domain_chunks=all_domain_chunks,
            held_out_sets=held_out_sets, device=device,
            reuse_checkpoints=True,
        )
        sweep_results.append(result)

        # Save partial results after each depth
        with open(partial_path, "w") as f:
            json.dump({"phase1_partial": sweep_results}, f, indent=2)
        print(f"  Partial results saved: {partial_path}")

    # Print phase 1 summary table
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS (seed=42, simple linear router)")
    print("=" * 70)
    print(f"{'Freeze':>8} {'Shared%':>9} {'Mixed Loss':>12} {'Improvement':>13} {'Div Gap':>10}")
    print("-" * 58)
    num_layers = 24
    for r in sweep_results:
        fn = r["freeze_layers"]
        shared_pct = fn / num_layers * 100
        main_marker = " *" if fn == MAIN_EXP_FREEZE else ""
        print(f"{fn:>8} {shared_pct:>8.0f}% {r['moe_mixed_loss']:>12.4f} "
              f"{r['improvement_pct']:>+12.1f}%{main_marker} {r['divergence_gap']:>9.4f}")
    print("  * = main experiment value")

    # Determine best 2 non-4 depths for multi-seed
    non4 = [r for r in sweep_results if r["freeze_layers"] != MAIN_EXP_FREEZE]
    top2 = sorted(non4, key=lambda r: r["improvement_pct"], reverse=True)[:2]
    top2_freeze = sorted([r["freeze_layers"] for r in top2])
    print(f"\nTop 2 depths for multi-seed run: {top2_freeze}")

    # =========================================================================
    # PHASE 2: multi-seed on top 2 depths
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: 3 seeds on freeze depths {top2_freeze}")
    print("=" * 70)

    multiseed_results = {}

    for freeze_n in top2_freeze:
        multiseed_results[str(freeze_n)] = {"seeds": [], "per_seed": []}

        for seed in SEEDS_MULTISEED:
            if seed == 42:
                # Already have from phase 1 — find the result
                existing = next((r for r in sweep_results
                                 if r["freeze_layers"] == freeze_n), None)
                if existing:
                    multiseed_results[str(freeze_n)]["per_seed"].append(existing)
                    multiseed_results[str(freeze_n)]["seeds"].append(42)
                    print(f"  freeze={freeze_n}, seed=42: reusing phase 1 result")
                    continue

            result = run_freeze_depth(
                freeze_n, seed=seed, all_domain_chunks=all_domain_chunks,
                held_out_sets=held_out_sets, device=device,
                reuse_checkpoints=False,
            )
            multiseed_results[str(freeze_n)]["per_seed"].append(result)
            multiseed_results[str(freeze_n)]["seeds"].append(seed)

            # Save partial
            with open(partial_path, "w") as f:
                json.dump({"phase1": sweep_results, "phase2_partial": multiseed_results}, f, indent=2)

        # Aggregate
        imps = [r["improvement_pct"]
                for r in multiseed_results[str(freeze_n)]["per_seed"]]
        mean_imp = statistics.mean(imps)
        std_imp = statistics.stdev(imps) if len(imps) > 1 else 0.0
        multiseed_results[str(freeze_n)]["mean"] = round(mean_imp, 4)
        multiseed_results[str(freeze_n)]["std"] = round(std_imp, 4)
        print(f"\nfreeze={freeze_n}: mean={mean_imp:+.1f}% ± {std_imp:.1f}%")

    # Also add freeze=4 multi-seed from main experiment results
    main_exp_path = RESULTS_DIR / "step5_final_summary.json"
    if main_exp_path.exists():
        with open(main_exp_path) as f:
            main_summary = json.load(f)
        main_imp = main_summary.get("summary", {}).get("improvement_mean_pct", 14.2)
        main_std = main_summary.get("summary", {}).get("improvement_std_pct", 0.0)
        multiseed_results["4"] = {
            "seeds": [42, 137, 2026],
            "mean": main_imp,
            "std": main_std,
            "note": "from main experiment (2-layer router, reported for reference)",
        }

    # Final results table
    print("\n" + "=" * 70)
    print("FREEZE DEPTH ABLATION — FINAL RESULTS")
    print("=" * 70)
    print(f"{'Freeze':>8} {'Shared%':>9} {'Improvement (seed=42)':>22} {'Multi-seed':>22}")
    print("-" * 70)
    for r in sweep_results:
        fn = r["freeze_layers"]
        shared_pct = fn / 24 * 100
        ms = multiseed_results.get(str(fn))
        if ms:
            ms_str = f"{ms['mean']:+.1f}% +/- {ms['std']:.1f}%"
        else:
            ms_str = "seed=42 only"
        main_marker = " *" if fn == MAIN_EXP_FREEZE else ""
        print(f"{fn:>8} {shared_pct:>8.0f}% {r['improvement_pct']:>+21.1f}%{main_marker} {ms_str:>22}")

    # Determine optimal freeze
    best_r = max(sweep_results, key=lambda r: r["improvement_pct"])
    print(f"\nOptimal freeze depth: {best_r['freeze_layers']} layers ({best_r['improvement_pct']:+.1f}%)")

    btx_result = next((r for r in sweep_results if r["freeze_layers"] == 0), None)
    if btx_result:
        btx_imp = btx_result["improvement_pct"]
        main_imp = next(r["improvement_pct"] for r in sweep_results if r["freeze_layers"] == MAIN_EXP_FREEZE)
        if main_imp > btx_imp:
            print(f"BTX baseline (freeze=0): {btx_imp:+.1f}% vs freeze=4: {main_imp:+.1f}%")
            print("Frozen layers HELP fusion — central thesis SUPPORTED.")
        else:
            print(f"BTX baseline (freeze=0): {btx_imp:+.1f}% vs freeze=4: {main_imp:+.1f}%")
            print("WARNING: freeze=0 >= freeze=4. Thesis needs qualification.")

    # Save figures
    print("\nSaving figure...")
    save_freeze_ablation_figure(sweep_results, multiseed_results)

    # Save JSON
    output = {
        "experiment": "freeze_depth_ablation",
        "base_model": f"pythia-410m-{REVISION}",
        "router_type": "simple_linear",
        "freeze_sweep": FREEZE_SWEEP,
        "phase1_seed42_results": sweep_results,
        "top2_freeze_depths": top2_freeze,
        "multi_seed_results": multiseed_results,
        "optimal_freeze_layers": best_r["freeze_layers"],
        "btx_baseline_improvement_pct": btx_result["improvement_pct"] if btx_result else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "ablation_freeze_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
