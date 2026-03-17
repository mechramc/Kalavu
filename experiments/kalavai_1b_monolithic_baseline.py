#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-1B Monolithic Baseline — Equal-Compute Comparison
=================================================================
Trains a single Pythia-1B model on mixed-domain data for 6000 steps
(= 3 specialists × 2000 steps = same total compute as specialist-then-fuse).

Mirrors kalavai_pythia_monolithic_baseline.py exactly, scaled to 1B.
The ONLY differences from the 410M version:
  - MODEL_ID = pythia-1b
  - RESULTS_DIR = results/pythia/pythia_1b
  - Figure filenames carry _1b_ suffix
  - MoE reference loaded from main_result_summary.json (1B format)

Compare against:
  - 1B base model:        mixed=2.160 (from main_result_summary.json)
  - 1B best specialist:   mixed≈1.992
  - 1B KALAVAI MoE:       mixed≈1.696
"""

import json
import statistics
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ============================================================================
# Config — matches 1B experiment exactly except max_steps=6000 and mixed data
# ============================================================================

MODEL_ID      = "EleutherAI/pythia-1b"
REVISION      = "step10000"
FREEZE_LAYERS = 4           # 4/16 = 25%, same as 1B specialist training
LR            = 2e-5
WEIGHT_DECAY  = 0.1
MAX_STEPS     = 6000        # 3 specialists × 2000 steps = equal compute
BATCH_SIZE    = 2
GRAD_ACCUM    = 4           # effective batch = 8
GRADIENT_CLIP = 1.0
SEQ_LEN       = 512
WARMUP_FRACTION = 0.1
DOMAINS       = ["code", "science", "fiction"]
SEEDS         = [42, 137, 2026]
N_SAMPLES_PER_DOMAIN = 3000
EVAL_INTERVAL = 500
EVAL_BATCHES  = 50

RESULTS_DIR   = Path("results/pythia/pythia_1b")
FIGURES_DIR   = Path("figures/pythia")
CHECKPOINT_DIR = Path("checkpoints/pythia/pythia_1b")

# MoE results from the 1B main experiment
MOE_RESULTS_PATH = RESULTS_DIR / "main_result_summary.json"


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
        "labels":    torch.stack([b["labels"]    for b in batch]),
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
# Data loading — identical to 1B main experiment
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
        content = (item.get("support", "") + "\n"
                   + item.get("question", "") + "\n"
                   + item.get("correct_answer", ""))
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
# Freeze (GPT-NeoX — same architecture at 1B)
# ============================================================================

def freeze_first_n_layers(model, n):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# Eval
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        loss = model(input_ids=ids, labels=lbl).loss
        if loss is not None:
            total += loss.item()
            count += 1
    return round(total / count, 6) if count > 0 else float("inf")


def eval_all(model, held_out_sets, device):
    return {d: eval_loss(model, ds, device) for d, ds in held_out_sets.items()}


# ============================================================================
# Training
# ============================================================================

def train_monolithic(model, mixed_train_chunks, held_out_sets, seed, device):
    """Train on mixed data for MAX_STEPS, eval every EVAL_INTERVAL steps."""
    set_seed(seed)
    freeze_first_n_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(mixed_train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer    = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    checkpoints = []

    # Step 0 eval
    print(f"  Step 0 eval (base)...")
    model.eval()
    step0 = eval_all(model, held_out_sets, device)
    checkpoints.append({"step": 0, "held_out": step0, "train_loss": None})
    print(f"    mixed={step0['mixed']:.4f}  code={step0['code']:.4f}  "
          f"science={step0['science']:.4f}  fiction={step0['fiction']:.4f}")
    model.train()

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS: break

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(**{k: v.to(device) for k, v in batch.items()})
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum        += 1
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

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                avg_train = running_loss / step
                print(f"  step {step}/{MAX_STEPS} | train={avg_train:.4f} | {time.time()-t0:.0f}s | eval...")
                model.eval()
                ckpt_eval = eval_all(model, held_out_sets, device)
                model.train()
                print(f"    mixed={ckpt_eval['mixed']:.4f}  code={ckpt_eval['code']:.4f}  "
                      f"science={ckpt_eval['science']:.4f}  fiction={ckpt_eval['fiction']:.4f}")
                checkpoints.append({
                    "step":       step,
                    "held_out":   ckpt_eval,
                    "train_loss": round(avg_train, 6),
                })

    print(f"  Training done in {time.time()-t0:.0f}s")
    model.eval()
    return checkpoints


# ============================================================================
# Figures
# ============================================================================

def save_comparison_figure(results_by_seed, base_losses, moe_info):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def mean_std(key_fn):
            vals = [key_fn(r) for r in results_by_seed]
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

        mono_mean, mono_std = mean_std(lambda r: r["final_held_out"]["mixed"])
        moe_m    = moe_info.get("moe_mixed_mean")
        moe_s    = moe_info.get("moe_mixed_std", 0.0)
        best_ind = moe_info.get("best_spec_mean")
        wavg_m   = moe_info.get("weight_avg_mean")
        base_mixed = base_losses["mixed"]

        labels = ["Base\nmodel", "Monolithic\n(6000 steps)", "Best\nindividual",
                  "Weight\navg", "MoE\nfused"]
        means  = [base_mixed, mono_mean, best_ind, wavg_m, moe_m]
        errs   = [0.0,        mono_std,  0.0,      0.0,   moe_s]
        colors = ["#95a5a6",  "#e67e22", "#3498db", "#f39c12", "#9b59b6"]

        valid = [(l, m, e, c) for l, m, e, c in zip(labels, means, errs, colors)
                 if m is not None]
        labels, means, errs, colors = zip(*valid)

        y_min = min(means) * 0.995
        y_max = max(means) * 1.005

        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5,
                      yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Held-Out Mixed Loss (lower is better)")
        ax.set_title("Equal-Compute Comparison: Monolithic vs Specialist Fusion\n"
                     "(Pythia-1B, 3 seeds, freeze=4)")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, loss in zip(bars, means):
            imp   = (base_mixed - loss) / base_mixed * 100
            label = f"{loss:.4f}"
            if abs(imp) > 0.01:
                label += f"\n({imp:+.1f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (y_max - y_min) * 0.003,
                    label, ha="center", va="bottom", fontsize=8, fontweight="bold")

        if moe_m and mono_mean:
            gap_pct = (mono_mean - moe_m) / mono_mean * 100
            ax.annotate(
                f"MoE vs Monolithic:\n{gap_pct:+.1f}%",
                xy=(len(means) - 1, moe_m),
                xytext=(-120, 30), textcoords="offset points",
                fontsize=9, color="#8e44ad",
                arrowprops=dict(arrowstyle="->", color="#8e44ad"),
            )

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_1b_monolithic_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING comparison figure: {e}")


def save_trajectory_figure(all_seed_checkpoints, moe_mixed_mean, base_mixed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_set = sorted(set(c["step"] for ckpts in all_seed_checkpoints for c in ckpts))
        avg_by_step = {}
        for step in steps_set:
            vals = []
            for ckpts in all_seed_checkpoints:
                match = next((c for c in ckpts if c["step"] == step), None)
                if match:
                    vals.append(match["held_out"]["mixed"])
            if vals:
                avg_by_step[step] = statistics.mean(vals)

        steps  = sorted(avg_by_step.keys())
        losses = [avg_by_step[s] for s in steps]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axhline(y=base_mixed,     color="#95a5a6", linestyle="--", linewidth=2,
                   label=f"Base model ({base_mixed:.4f})", zorder=2)
        ax.axhline(y=moe_mixed_mean, color="#9b59b6", linestyle="--", linewidth=2.5,
                   label=f"MoE fused ({moe_mixed_mean:.4f})", zorder=2)
        ax.plot(steps, losses, "o-", color="#e67e22", linewidth=2.5, markersize=6,
                label="Monolithic (mixed, held-out)", zorder=3)
        ax.axvline(x=2000, color="#3498db", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(2000 + 50, max(losses) * 0.999, "Specialists\nfinish here",
                fontsize=8, color="#2980b9")

        mono_at_6000 = avg_by_step.get(6000, losses[-1])
        if moe_mixed_mean < mono_at_6000:
            ax.fill_between([0, 6000],
                            [moe_mixed_mean, moe_mixed_mean],
                            [mono_at_6000, mono_at_6000],
                            alpha=0.08, color="#9b59b6", label="MoE advantage region")

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Held-Out Mixed Loss (lower is better)")
        ax.set_title("Monolithic Training Trajectory vs Fusion Result\n"
                     "(Pythia-1B, avg across 3 seeds)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_1b_monolithic_trajectory.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING trajectory figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: 1B Monolithic Baseline — Equal-Compute Comparison")
    print("=" * 70)
    print(f"Model: {MODEL_ID} @ {REVISION}")
    print(f"Monolithic steps: {MAX_STEPS} (= 3 specialists × 2000)")
    print(f"Seeds: {SEEDS}")
    print(f"Eval every {EVAL_INTERVAL} steps")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data once
    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting (80/10/10)...")
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    mixed_train = []
    for d in DOMAINS:
        mixed_train.extend(all_domain_chunks[d]["train"])
    print(f"\n  Mixed training chunks: {len(mixed_train)}")

    # Base model losses
    print("\nEvaluating base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()
    base_losses = eval_all(base_model, held_out_sets, device)
    print(f"  Base: " + "  ".join(f"{d}={base_losses[d]:.4f}"
                                   for d in DOMAINS + ["mixed"]))
    del base_model
    torch.cuda.empty_cache()

    # Load 1B MoE reference results
    moe_info = {}
    if MOE_RESULTS_PATH.exists():
        with open(MOE_RESULTS_PATH) as f:
            s = json.load(f)
        moe_info["improvement_mean_pct"] = s["summary"]["improvement_mean_pct"]
        moe_info["improvement_std_pct"]  = s["summary"]["improvement_std_pct"]

        moe_mixed_vals  = []
        best_spec_vals  = []
        weight_avg_vals = []
        for _, sd in s.get("per_seed_fusion", {}).items():
            eh = sd.get("eval_heldout", {})
            v = eh.get("moe", {}).get("mixed")
            if v: moe_mixed_vals.append(v)
            best_ind = min(
                eh.get("code_spec",    {}).get("mixed", float("inf")),
                eh.get("science_spec", {}).get("mixed", float("inf")),
                eh.get("fiction_spec", {}).get("mixed", float("inf")),
            )
            if best_ind < float("inf"): best_spec_vals.append(best_ind)
            wavg = eh.get("weight_avg", {}).get("mixed")
            if wavg: weight_avg_vals.append(wavg)

        if moe_mixed_vals:
            moe_info["moe_mixed_mean"] = statistics.mean(moe_mixed_vals)
            moe_info["moe_mixed_std"]  = (statistics.stdev(moe_mixed_vals)
                                          if len(moe_mixed_vals) > 1 else 0.0)
        if best_spec_vals:
            moe_info["best_spec_mean"] = statistics.mean(best_spec_vals)
        if weight_avg_vals:
            moe_info["weight_avg_mean"] = statistics.mean(weight_avg_vals)
    else:
        print(f"  WARNING: MoE results not found at {MOE_RESULTS_PATH}")

    # =========================================================================
    # Run 3 seeds
    # =========================================================================
    all_seed_results     = []
    all_seed_checkpoints = []

    for seed in SEEDS:
        print(f"\n{'='*55}")
        print(f"SEED {seed}")
        print(f"{'='*55}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        ).to(device)

        checkpoints = train_monolithic(model, mixed_train, held_out_sets, seed, device)
        all_seed_checkpoints.append(checkpoints)

        final = checkpoints[-1]["held_out"]
        print(f"\n  Final (step {checkpoints[-1]['step']}):")
        print(f"    mixed={final['mixed']:.4f}  code={final['code']:.4f}  "
              f"science={final['science']:.4f}  fiction={final['fiction']:.4f}")

        seed_result = {
            "seed":           seed,
            "final_held_out": final,
            "checkpoints":    checkpoints,
        }
        all_seed_results.append(seed_result)

        # Per-seed JSON
        out_path = RESULTS_DIR / f"monolithic_baseline_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump({
                "seed":          seed,
                "model_id":      MODEL_ID,
                "revision":      REVISION,
                "max_steps":     MAX_STEPS,
                "freeze_layers": FREEZE_LAYERS,
                "base_losses":   base_losses,
                "checkpoints":   checkpoints,
            }, f, indent=2)
        print(f"  Saved: {out_path}")

        # Save checkpoint for benchmarks (seed=42 only)
        if seed == 42:
            ckpt_path = CHECKPOINT_DIR / "monolithic_seed42.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        del model
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    mono_mixed_vals = [r["final_held_out"]["mixed"] for r in all_seed_results]
    mono_mean = statistics.mean(mono_mixed_vals)
    mono_std  = statistics.stdev(mono_mixed_vals) if len(mono_mixed_vals) > 1 else 0.0

    moe_mixed_mean = moe_info.get("moe_mixed_mean")
    fused_vs_mono  = None
    if moe_mixed_mean:
        fused_vs_mono = round((mono_mean - moe_mixed_mean) / mono_mean * 100, 4)

    mono_vs_base = round((base_losses["mixed"] - mono_mean) / base_losses["mixed"] * 100, 4)
    moe_vs_base  = moe_info.get("improvement_mean_pct")

    print("\n" + "=" * 70)
    print("1B MONOLITHIC BASELINE — FINAL RESULTS")
    print("=" * 70)
    col    = 10
    header = f"{'Model':<28}" + "".join(f"{d.capitalize():>{col}}" for d in DOMAINS + ["Mixed"])
    print(header)
    print("-" * len(header))

    def row(name, losses_dict):
        return (f"{name:<28}"
                + "".join(f"{losses_dict[d]:>{col}.4f}" for d in DOMAINS)
                + f"{losses_dict['mixed']:>{col}.4f}")

    print(row("Base model", base_losses))
    mono_mean_losses = {d: statistics.mean(r["final_held_out"][d] for r in all_seed_results)
                        for d in DOMAINS + ["mixed"]}
    print(row(f"Monolithic ({MAX_STEPS} steps)", mono_mean_losses))

    if moe_mixed_mean:
        moe_losses = {d: moe_info.get(f"moe_{d}_mean", moe_mixed_mean) for d in DOMAINS}
        moe_losses["mixed"] = moe_mixed_mean

    print(f"\nMonolithic vs Base:      {mono_vs_base:+.1f}%")
    if moe_vs_base:
        print(f"MoE fused vs Base:       {moe_vs_base:+.1f}%")
    if fused_vs_mono is not None:
        print(f"MoE fused vs Monolithic: {fused_vs_mono:+.1f}%  <-- KEY NUMBER")
        if fused_vs_mono > 0:
            print(f"  Specialist-then-fuse BEATS monolithic by {fused_vs_mono:.1f}%")
        else:
            print(f"  Monolithic BEATS specialist-then-fuse by {-fused_vs_mono:.1f}%")

    # Figures
    print("\nSaving figures...")
    save_comparison_figure(all_seed_results, base_losses, moe_info)
    if moe_mixed_mean:
        save_trajectory_figure(all_seed_checkpoints, moe_mixed_mean, base_losses["mixed"])

    # Summary JSON
    summary = {
        "experiment":          "1b_monolithic_baseline",
        "model_id":            MODEL_ID,
        "revision":            REVISION,
        "total_steps":         MAX_STEPS,
        "equivalent_compute":  "3 specialists × 2000 steps",
        "freeze_layers":       FREEZE_LAYERS,
        "seeds":               SEEDS,
        "base_losses":         base_losses,
        "results": {
            "mean": {
                "monolithic_mixed":        round(mono_mean, 6),
                "moe_fused_mixed":         round(moe_mixed_mean, 6) if moe_mixed_mean else None,
                "best_specialist_mixed":   round(moe_info["best_spec_mean"], 6) if "best_spec_mean" in moe_info else None,
                "weight_avg_mixed":        round(moe_info["weight_avg_mean"], 6) if "weight_avg_mean" in moe_info else None,
                "fused_vs_monolithic_pct": fused_vs_mono,
                "monolithic_vs_base_pct":  mono_vs_base,
                "moe_vs_base_pct":         moe_vs_base,
            },
            "std": {
                "monolithic_mixed": round(mono_std, 6),
            },
        },
        "per_seed": [
            {
                "seed":        r["seed"],
                "final_mixed": r["final_held_out"]["mixed"],
                "final_code":  r["final_held_out"]["code"],
                "final_science": r["final_held_out"]["science"],
                "final_fiction": r["final_held_out"]["fiction"],
            }
            for r in all_seed_results
        ],
        "interpretation": (
            f"KALAVAI MoE beats 1B monolithic by {fused_vs_mono:.1f}%"
            if fused_vs_mono and fused_vs_mono > 0
            else "Monolithic matches or exceeds MoE — check scaling story"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "monolithic_baseline_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
