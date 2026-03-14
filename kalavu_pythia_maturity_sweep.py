#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Pythia-410M Base Model Maturity Sweep
==============================================
Sweeps 6 Pythia-410M training checkpoints measuring fusion improvement
vs base model maturity (how far through training the base was when frozen).

Hypothesis: Fusion improvement should be highest at intermediate maturity —
too early (random weights) → specialists can't diverge meaningfully;
too late (fully trained) → model already knows everything, no room to improve.

Data split: ALL domains use a single 80/10/10 split on packed chunks.
ALL reported numbers use held_out_chunks only.
Data is loaded ONCE and reused across all checkpoint revisions.
"""

import copy
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
# Config
# ============================================================================

MODEL_BASE = "EleutherAI/pythia-410m"
REVISIONS = [
    ("step5000",   5000,   3.5),
    ("step10000",  10000,  7.0),   # main experiment — load existing checkpoints if present
    ("step20000",  20000,  14.0),
    ("step50000",  50000,  35.0),
    ("step100000", 100000, 70.0),
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
HIDDEN_SIZE = 1024
DOMAINS = ["code", "science", "fiction"]
SEEDS_PHASE_A = [42]
SEEDS_PHASE_B = [137, 2026]   # run on 2 most interesting checkpoints after phase A
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50

RESULTS_DIR = Path("results/pythia/maturity_sweep_410m")
CHECKPOINT_DIR = Path("checkpoints/pythia/maturity_sweep_410m")
FIGURES_DIR = Path("figures/pythia")

# Existing main-experiment checkpoints live here (step10000, seed=42)
MAIN_EXP_CHECKPOINT_DIR = Path("checkpoints/pythia")


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
# Simple linear router MoE (3-expert)
# ============================================================================

class SimpleLinearMoE(nn.Module):
    """MoE with single-layer linear router — simple Linear(1024, 3)."""
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
# Training helpers
# ============================================================================

def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_specialist(model, domain, train_chunks, seed, device):
    """Train one specialist. Assumes freeze already applied."""
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


def train_router(moe, combined_train_chunks, device):
    """Train the router on mixed chunks."""
    combined = make_dataset_from_chunks(combined_train_chunks)
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


def weight_average_three(spec_a, spec_b, spec_c):
    """Three-way weight average."""
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
# Single checkpoint run (one revision × one seed)
# ============================================================================

def run_checkpoint_seed(revision, step_n, training_pct, seed,
                        all_domain_chunks, held_out_sets, device):
    """
    Train 3 specialists at the given revision + seed, fuse, return results dict.
    For step10000 + seed=42: reuses main experiment checkpoints if present.
    """
    print(f"\n{'='*60}")
    print(f"revision={revision} (step={step_n}, {training_pct:.1f}% trained) seed={seed}")
    print(f"{'='*60}")

    specialists = {}
    combined_train = []
    specialist_beats_base = {}

    # Determine checkpoint paths for this run
    ckpt_subdir = CHECKPOINT_DIR / f"step{step_n}"
    ckpt_subdir.mkdir(parents=True, exist_ok=True)

    # Eval base model first
    print(f"  Loading base model at {revision}...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE, revision=revision, torch_dtype=torch.bfloat16,
    ).to(device)
    base.eval()

    base_losses = {}
    for domain in DOMAINS:
        l = eval_loss(base, held_out_sets[domain], device)
        base_losses[domain] = round(l, 6)
    base_mixed = eval_loss(base, held_out_sets["mixed"], device)
    base_losses["mixed"] = round(base_mixed, 6)
    print(f"  Base losses: " + ", ".join(f"{d}={v:.4f}" for d, v in base_losses.items()))
    del base
    torch.cuda.empty_cache()

    # Train or load specialists
    for domain in DOMAINS:
        combined_train.extend(all_domain_chunks[domain]["train"])

        # Check for existing checkpoint
        ckpt_path = ckpt_subdir / f"{domain}_seed{seed}.pt"

        # Special case: step10000 / seed=42 → try main experiment checkpoints
        main_exp_ckpt = MAIN_EXP_CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
        if step_n == 10000 and seed == 42 and main_exp_ckpt.exists() and not ckpt_path.exists():
            import shutil
            shutil.copy(main_exp_ckpt, ckpt_path)
            print(f"  Copied main-exp checkpoint: {main_exp_ckpt.name} -> {ckpt_path}")

        if ckpt_path.exists():
            print(f"  Loading existing checkpoint: {ckpt_path}")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE, revision=revision, torch_dtype=torch.bfloat16,
            ).to(device)
            spec.load_state_dict(torch.load(ckpt_path, map_location=device))
            spec.eval()
        else:
            print(f"  Training {domain} specialist (revision={revision}, seed={seed})...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE, revision=revision, torch_dtype=torch.bfloat16,
            ).to(device)
            freeze_first_n_layers(spec, FREEZE_LAYERS)
            train_specialist(spec, domain, all_domain_chunks[domain]["train"], seed, device)
            spec.eval()
            torch.save(spec.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

        specialists[domain] = spec

    # Divergence check: does each specialist beat base on its own domain?
    print("  Divergence check...")
    for domain in DOMAINS:
        spec_loss = eval_loss(specialists[domain], held_out_sets[domain], device)
        base_loss = base_losses[domain]
        beats = spec_loss < base_loss
        specialist_beats_base[domain] = {
            "beats_base": beats,
            "spec_loss": round(spec_loss, 6),
            "base_loss": base_loss,
            "delta_pct": round((spec_loss - base_loss) / base_loss * 100, 2),
        }
        sym = "+" if beats else "-"
        print(f"    [{sym}] {domain}: spec={spec_loss:.4f} vs base={base_loss:.4f} "
              f"({specialist_beats_base[domain]['delta_pct']:+.1f}%)")

    # Best individual on mixed
    best_individual = min(
        eval_loss(specialists[d], held_out_sets["mixed"], device) for d in DOMAINS
    )

    # 3-way weight average
    print("  Computing 3-way weight average...")
    wa = weight_average_three(specialists["code"], specialists["science"], specialists["fiction"]).to(device)
    wa_mixed = eval_loss(wa, held_out_sets["mixed"], device)
    del wa
    torch.cuda.empty_cache()

    # Build simple linear MoE and train router
    print("  Building SimpleLinearMoE and training router...")
    moe = SimpleLinearMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    train_router(moe, combined_train, device)
    moe.eval()

    moe_mixed = eval_loss(moe, held_out_sets["mixed"], device, batch_size=2, is_fused=True)
    improvement = (best_individual - moe_mixed) / best_individual * 100

    print(f"\n  Results (revision={revision}, seed={seed}):")
    print(f"    Base mixed:       {base_mixed:.4f}")
    print(f"    Best individual:  {best_individual:.4f}")
    print(f"    Weight avg mixed: {wa_mixed:.4f}")
    print(f"    MoE mixed:        {moe_mixed:.4f}")
    print(f"    Improvement:      {improvement:+.1f}%")

    print(f"[kalavu] maturity sweep 410m: step={step_n} improvement={improvement:.1f}%")

    del moe
    for s in specialists.values():
        del s
    torch.cuda.empty_cache()

    return {
        "revision": revision,
        "step_n": step_n,
        "training_pct": training_pct,
        "seed": seed,
        "base_losses": base_losses,
        "best_individual_mixed": round(best_individual, 6),
        "weight_avg_mixed": round(wa_mixed, 6),
        "moe_mixed_loss": round(moe_mixed, 6),
        "improvement_pct": round(improvement, 4),
        "specialist_beats_base": specialist_beats_base,
    }


# ============================================================================
# Figure
# ============================================================================

def save_maturity_curve(phase_a_results, multiseed_results, interesting_steps):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        training_pcts = [r["training_pct"] for r in phase_a_results]
        improvements   = [r["improvement_pct"] for r in phase_a_results]
        step_ns        = [r["step_n"] for r in phase_a_results]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="y=0 (no improvement)")

        ax.plot(training_pcts, improvements, "o-", color="#3498db",
                linewidth=2.5, markersize=8, label="Improvement % (seed=42)")

        # Error bars for multi-seed points
        for step_n, ms in multiseed_results.items():
            sn = int(step_n)
            match = next((r for r in phase_a_results if r["step_n"] == sn), None)
            if match:
                ax.errorbar(match["training_pct"], match["improvement_pct"],
                            yerr=ms["std"], fmt="none", color="#2980b9", capsize=6, linewidth=2)

        # Annotate step10000 (main experiment)
        main = next((r for r in phase_a_results if r["step_n"] == 10000), None)
        if main:
            ax.annotate("main exp\n(step10000)",
                        xy=(main["training_pct"], main["improvement_pct"]),
                        xytext=(main["training_pct"] + 3, main["improvement_pct"] + 1.5),
                        fontsize=8, color="#27ae60",
                        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
            ax.plot(main["training_pct"], main["improvement_pct"],
                    "*", color="#f1c40f", markersize=16, zorder=5)

        # Annotate interesting steps
        for sn in interesting_steps:
            match = next((r for r in phase_a_results if r["step_n"] == sn), None)
            if match and sn != 10000:
                ax.annotate(f"step{sn}",
                            xy=(match["training_pct"], match["improvement_pct"]),
                            xytext=(match["training_pct"] + 2, match["improvement_pct"] - 1.5),
                            fontsize=7, color="#8e44ad",
                            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.0))

        # Value labels
        for pct, imp in zip(training_pcts, improvements):
            ax.annotate(f"{imp:+.1f}%", (pct, imp),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8)

        ax.set_xlabel("Base Model Training Progress (%)", fontsize=12)
        ax.set_ylabel("MoE Improvement over Best Individual (%)", fontsize=12)
        ax.set_title("Fusion improvement vs base model maturity (Pythia-410M)", fontsize=13)
        ax.set_xticks(training_pcts)
        ax.set_xticklabels([f"{p:.1f}%" for p in training_pcts])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_maturity_curve_410m.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save maturity curve figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU: Pythia-410M Base Model Maturity Sweep")
    print("=" * 70)
    print(f"Model:     {MODEL_BASE}")
    print(f"Revisions: {[r[0] for r in REVISIONS]}")
    print(f"Domains:   {DOMAINS}")
    print(f"Phase A:   seeds={SEEDS_PHASE_A} (all 6 checkpoints)")
    print(f"Phase B:   seeds={SEEDS_PHASE_B} (2 most interesting checkpoints)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load tokenizer + data ONCE — same chunks reused across all revisions
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer (step10000 for tokenization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, revision="step10000")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data (loaded ONCE, shared across all revisions)...")
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

    # =====================================================================
    # PHASE A: seed=42 across all 6 checkpoints
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE A: seed=42 sweep across all 6 checkpoints")
    print("=" * 70)

    phase_a_results = []
    partial_path = RESULTS_DIR / "maturity_sweep_partial.json"

    for revision, step_n, training_pct in REVISIONS:
        result = run_checkpoint_seed(
            revision, step_n, training_pct,
            seed=42,
            all_domain_chunks=all_domain_chunks,
            held_out_sets=held_out_sets,
            device=device,
        )
        phase_a_results.append(result)

        # Save result
        out_path = RESULTS_DIR / f"checkpoint_step{step_n}_seed42.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

        # Save partial after each checkpoint
        with open(partial_path, "w") as f:
            json.dump({"phase_a_partial": phase_a_results}, f, indent=2)

    # Print Phase A table
    print("\n" + "=" * 70)
    print("PHASE A RESULTS (seed=42)")
    print("=" * 70)
    print(f"{'Step':>10} {'Train%':>8} {'Base Loss':>11} {'Best Indiv':>11} {'MoE Loss':>10} {'Improvement':>13}")
    print("-" * 68)
    for r in phase_a_results:
        main_marker = " *" if r["step_n"] == 10000 else ""
        print(f"{r['step_n']:>10} {r['training_pct']:>7.1f}% "
              f"{r['base_losses']['mixed']:>11.4f} {r['best_individual_mixed']:>11.4f} "
              f"{r['moe_mixed_loss']:>10.4f} {r['improvement_pct']:>+12.1f}%{main_marker}")
    print("  * = main experiment checkpoint")

    # Identify 2 most interesting checkpoints for Phase B:
    # 1) highest improvement; 2) improvement drops below 5% (or the lowest)
    sorted_by_imp = sorted(phase_a_results, key=lambda r: r["improvement_pct"], reverse=True)
    best_step = sorted_by_imp[0]["step_n"]

    # Find the checkpoint where improvement drops below 5%, or the one with lowest improvement
    below_threshold = [r for r in phase_a_results if r["improvement_pct"] < 5.0]
    if below_threshold:
        interesting_step2 = min(below_threshold, key=lambda r: r["improvement_pct"])["step_n"]
    else:
        interesting_step2 = sorted_by_imp[-1]["step_n"]

    interesting_steps = sorted(set([best_step, interesting_step2]))
    print(f"\nPhase B targets (2 most interesting): {interesting_steps}")
    print(f"  - step{best_step}: highest improvement = {sorted_by_imp[0]['improvement_pct']:+.1f}%")
    match2 = next(r for r in phase_a_results if r["step_n"] == interesting_step2)
    print(f"  - step{interesting_step2}: improvement = {match2['improvement_pct']:+.1f}%")

    # =====================================================================
    # PHASE B: seeds 137, 2026 on the 2 interesting checkpoints
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"PHASE B: seeds={SEEDS_PHASE_B} on steps={interesting_steps}")
    print("=" * 70)

    multiseed_results = {}

    for step_n in interesting_steps:
        revision = next(r[0] for r in REVISIONS if r[1] == step_n)
        training_pct = next(r[2] for r in REVISIONS if r[1] == step_n)
        multiseed_results[str(step_n)] = {"per_seed": [], "seeds": [42]}

        # Seed=42 already done in Phase A
        phase_a_result = next(r for r in phase_a_results if r["step_n"] == step_n)
        multiseed_results[str(step_n)]["per_seed"].append(phase_a_result)

        for seed in SEEDS_PHASE_B:
            result = run_checkpoint_seed(
                revision, step_n, training_pct,
                seed=seed,
                all_domain_chunks=all_domain_chunks,
                held_out_sets=held_out_sets,
                device=device,
            )
            multiseed_results[str(step_n)]["per_seed"].append(result)
            multiseed_results[str(step_n)]["seeds"].append(seed)

            out_path = RESULTS_DIR / f"checkpoint_step{step_n}_seed{seed}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_path}")

            with open(partial_path, "w") as f:
                json.dump({
                    "phase_a": phase_a_results,
                    "phase_b_partial": {k: v["per_seed"] for k, v in multiseed_results.items()},
                }, f, indent=2)

        # Aggregate multi-seed stats
        imps = [r["improvement_pct"] for r in multiseed_results[str(step_n)]["per_seed"]]
        mean_imp = statistics.mean(imps)
        std_imp = statistics.stdev(imps) if len(imps) > 1 else 0.0
        multiseed_results[str(step_n)]["mean"] = round(mean_imp, 4)
        multiseed_results[str(step_n)]["std"] = round(std_imp, 4)
        print(f"\nstep{step_n}: mean={mean_imp:+.1f}% +/- {std_imp:.1f}% (n={len(imps)} seeds)")

    print("[kalavu] maturity sweep 410m: multi-seed complete")

    # =====================================================================
    # Final summary table
    # =====================================================================
    print("\n" + "=" * 70)
    print("MATURITY SWEEP — FINAL RESULTS")
    print("=" * 70)
    print(f"{'Step':>10} {'Train%':>8} {'Improvement (s=42)':>20} {'Multi-seed':>24}")
    print("-" * 70)
    for r in phase_a_results:
        sn = str(r["step_n"])
        ms = multiseed_results.get(sn)
        if ms:
            ms_str = f"{ms['mean']:+.1f}% +/- {ms['std']:.1f}%"
        else:
            ms_str = "seed=42 only"
        main_marker = " *" if r["step_n"] == 10000 else ""
        print(f"{r['step_n']:>10} {r['training_pct']:>7.1f}% "
              f"{r['improvement_pct']:>+19.1f}%{main_marker} {ms_str:>24}")
    print("  * = main experiment checkpoint")

    # =====================================================================
    # Figure
    # =====================================================================
    print("\nSaving maturity curve figure...")
    save_maturity_curve(phase_a_results, multiseed_results, interesting_steps)

    # =====================================================================
    # Save summary JSON
    # =====================================================================
    summary = {
        "experiment": "maturity_sweep_410m",
        "base_model": MODEL_BASE,
        "domains": DOMAINS,
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
        },
        "curve": [
            {
                "revision": r["revision"],
                "step_n": r["step_n"],
                "training_pct": r["training_pct"],
                "improvement_pct_seed42": r["improvement_pct"],
                "moe_mixed_loss": r["moe_mixed_loss"],
                "base_mixed_loss": r["base_losses"]["mixed"],
                "multiseed": multiseed_results.get(str(r["step_n"]), None),
            }
            for r in phase_a_results
        ],
        "phase_b_interesting_steps": interesting_steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")

    if partial_path.exists():
        partial_path.unlink()

    # Final commit-style print
    improvements_str = ", ".join(
        f"step{r['step_n']}={r['improvement_pct']:+.1f}%" for r in phase_a_results
    )
    print(f"\n[kalavu] maturity sweep 410m: curve = [{improvements_str}]")


if __name__ == "__main__":
    main()
