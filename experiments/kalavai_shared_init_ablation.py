#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Shared Initialization Necessity Ablation
=================================================
Tests whether specialists MUST start from the same checkpoint.

The paper claims shared initialization is the core structural requirement —
but this is currently argued by narrative, not demonstrated experimentally.
This ablation provides direct empirical evidence.

Three conditions (all Pythia-410M, 3-domain code/science/fiction, 2000 steps):

  Condition 1 (control):    All 3 specialists from step10000  [3 seeds]
    → Expected: +14.2% improvement (re-confirms paper result)

  Condition 2 (large gap):  Code from step5000, Science from step10000,
                             Fiction from step20000             [3 seeds]
    → Expected: significant degradation (init divergence breaks fusibility)

  Condition 3 (small gap):  Code from step8000, Science from step10000,
                             Fiction from step12000             [1 seed = 42]
    → Expected: partial degradation vs control

Output: Table with condition | init checkpoints | MoE loss | improvement vs
        best specialist | improvement vs base

Decision criterion: Clear degradation in Condition 2 confirms the
shared-initialization claim. Gradient Condition 3 < Condition 1 and
Condition 2 < Condition 3 gives a "fusibility vs. init distance" curve.
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
# Config — matches kalavai_pythia_experiment.py exactly
# ============================================================================

MODEL_ID      = "EleutherAI/pythia-410m"
FREEZE_LAYERS = 4
LR            = 2e-5
WEIGHT_DECAY  = 0.1
MAX_STEPS     = 2000
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
GRADIENT_CLIP = 1.0
SEQ_LEN       = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE   = 1024      # Pythia-410M hidden size
DOMAINS       = ["code", "science", "fiction"]
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS  = 500
ROUTER_LR     = 1e-3
ROUTER_BATCH  = 4
EVAL_BATCHES  = 50

# Condition definitions — (domain -> revision to load from)
CONDITIONS = [
    {
        "name":  "Condition 1 (control): shared step10000",
        "short": "control",
        "inits": {
            "code":    "step10000",
            "science": "step10000",
            "fiction": "step10000",
        },
        "seeds": [42, 137, 2026],
    },
    {
        "name":  "Condition 2 (large gap): step5000/step10000/step20000",
        "short": "large_gap",
        "inits": {
            "code":    "step5000",
            "science": "step10000",
            "fiction": "step20000",
        },
        "seeds": [42, 137, 2026],
    },
    {
        "name":  "Condition 3 (small gap): step8000/step10000/step12000",
        "short": "small_gap",
        "inits": {
            "code":    "step8000",
            "science": "step10000",
            "fiction": "step12000",
        },
        "seeds": [42],
    },
]

RESULTS_DIR    = Path("results/pythia/shared_init_ablation")
FIGURES_DIR    = Path("figures/pythia")
CHECKPOINT_DIR = Path("checkpoints/pythia/shared_init_ablation")


# ============================================================================
# Dataset (identical to all other experiments)
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
# Data loading
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
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
# Freeze
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
def eval_loss(model, dataset, device, batch_size=4, is_fused=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES: break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(input_ids, labels=labels)
        else:
            out  = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        if loss is not None:
            total += loss.item()
            count += 1
    return round(total / count, 6) if count > 0 else float("inf")


def eval_all(model, held_out_sets, device, is_fused=False):
    return {d: eval_loss(model, ds, device, is_fused=is_fused)
            for d, ds in held_out_sets.items()}


# ============================================================================
# Specialist training
# ============================================================================

def train_specialist(model, domain, train_chunks, seed, device):
    set_seed(seed)
    freeze_first_n_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer    = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

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

            if step % 200 == 0 or step == MAX_STEPS:
                avg = running_loss / step
                print(f"  [{domain}] step {step}/{MAX_STEPS} | loss={avg:.4f} | {time.time()-t0:.0f}s")

    model.eval()
    return model


# ============================================================================
# ThreeExpertMoE (same as main experiment)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for sp in [self.spec_a, self.spec_b, self.spec_c]:
            for p in sp.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits   = out.logits.detach()
        h_pooled = out.hidden_states[-1].detach().mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)
        h_avg  = (h_a + h_b + h_c) / 3.0
        gates  = torch.softmax(self.router(h_avg), dim=-1)
        fused  = (gates[:, 0:1, None] * logits_a
                + gates[:, 1:2, None] * logits_b
                + gates[:, 2:3, None] * logits_c)
        loss = None
        if labels is not None:
            sl = fused[:, :-1, :].contiguous()
            ll = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1))
        return loss, fused, gates


# ============================================================================
# Weight averaging
# ============================================================================

def weight_average_three(spec_a, spec_b, spec_c):
    avg = copy.deepcopy(spec_a)
    sa, sb, sc = spec_a.state_dict(), spec_b.state_dict(), spec_c.state_dict()
    avg.load_state_dict({
        k: ((sa[k].float() + sb[k].float() + sc[k].float()) / 3.0).to(torch.bfloat16)
        for k in sa
    })
    avg.eval()
    return avg


# ============================================================================
# Router training
# ============================================================================

def train_router(moe, train_datasets, device):
    all_chunks = []
    for ds in train_datasets.values():
        all_chunks.extend(ds.chunks)
    combined = make_dataset_from_chunks(all_chunks)

    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader    = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                           drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()

    print(f"\n  Training router ({ROUTER_STEPS} steps)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch     = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")


@torch.no_grad()
def eval_router_distribution(moe, eval_datasets, device, n_batches=20):
    moe.eval()
    results = {}
    for domain, ds in eval_datasets.items():
        if domain == "mixed": continue
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                            drop_last=True, collate_fn=_collate)
        gate_sums = [0.0, 0.0, 0.0]
        count = 0
        for batch in loader:
            if count >= n_batches: break
            ids = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            _, _, gates = moe(ids, labels=lbl)
            for i in range(3):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        if count > 0:
            results[domain] = [round(g / count, 4) for g in gate_sums]
    return results


# ============================================================================
# Run one condition for one seed
# ============================================================================

def run_condition_seed(condition, seed, all_domain_chunks, base_losses,
                        tokenizer, device):
    """Train 3 specialists from their assigned init checkpoints, fuse, eval."""
    cond_name = condition["short"]
    inits     = condition["inits"]

    print(f"\n  --- Seed {seed} ---")

    # Load each specialist from its own init checkpoint
    specialists = {}
    for domain in DOMAINS:
        revision = inits[domain]
        print(f"  Loading {domain} specialist from {revision}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=revision, torch_dtype=torch.bfloat16,
        ).to(device)
        print(f"  Training {domain} specialist (seed={seed}, init={revision})...")
        model = train_specialist(
            model, domain,
            all_domain_chunks[domain]["train"],
            seed, device,
        )
        specialists[domain] = model

    # Eval each specialist on held-out data
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    print(f"\n  Evaluating specialists...")
    spec_losses = {}
    for domain, model in specialists.items():
        spec_losses[domain] = eval_all(model, held_out_sets, device)
        print(f"    {domain}: " + "  ".join(
            f"{d}={spec_losses[domain][d]:.4f}" for d in DOMAINS + ["mixed"]))

    best_spec_mixed = min(spec_losses[d]["mixed"] for d in DOMAINS)

    # Weight average
    print(f"  Computing weight average...")
    weight_avg = weight_average_three(
        specialists["code"], specialists["science"], specialists["fiction"]
    ).to(device)
    wavg_losses = eval_all(weight_avg, held_out_sets, device)
    print(f"    weight_avg: " + "  ".join(
        f"{d}={wavg_losses[d]:.4f}" for d in DOMAINS + ["mixed"]))
    del weight_avg
    torch.cuda.empty_cache()

    # MoE fusion
    print(f"  Building MoE...")
    train_datasets = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                      for d in DOMAINS}
    moe = ThreeExpertMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    train_router(moe, train_datasets, device)

    print(f"  Evaluating MoE...")
    moe_losses = eval_all(moe, held_out_sets, device, is_fused=True)
    print(f"    MoE: " + "  ".join(
        f"{d}={moe_losses[d]:.4f}" for d in DOMAINS + ["mixed"]))

    router_dist = eval_router_distribution(moe, held_out_sets, device)

    # Improvement metrics
    base_mixed  = base_losses["mixed"]
    moe_mixed   = moe_losses["mixed"]
    imp_vs_base = round((base_mixed - moe_mixed) / base_mixed * 100, 4)
    imp_vs_spec = round((best_spec_mixed - moe_mixed) / best_spec_mixed * 100, 4)

    print(f"\n  Results:")
    print(f"    Base mixed:          {base_mixed:.4f}")
    print(f"    Best specialist:     {best_spec_mixed:.4f}")
    print(f"    Weight avg:          {wavg_losses['mixed']:.4f}")
    print(f"    MoE fused:           {moe_mixed:.4f}")
    print(f"    Improvement vs base: {imp_vs_base:+.2f}%")
    print(f"    Improvement vs spec: {imp_vs_spec:+.2f}%")
    print(f"    Router distribution: {router_dist}")

    # Clean up
    del moe, specialists["code"], specialists["science"], specialists["fiction"]
    torch.cuda.empty_cache()

    return {
        "seed":             seed,
        "condition":        cond_name,
        "init_revisions":   inits,
        "base_losses":      base_losses,
        "spec_losses":      spec_losses,
        "weight_avg_losses": wavg_losses,
        "moe_losses":       moe_losses,
        "best_spec_mixed":  best_spec_mixed,
        "improvement_vs_base_pct":  imp_vs_base,
        "improvement_vs_spec_pct":  imp_vs_spec,
        "router_distribution":      router_dist,
    }


# ============================================================================
# Summary table
# ============================================================================

def print_summary(all_results):
    print("\n" + "=" * 80)
    print("SHARED INITIALIZATION ABLATION — SUMMARY")
    print("=" * 80)
    hdr = (f"{'Condition':<42} {'Seeds':>5} {'MoE Mixed':>10} "
           f"{'vs Base':>9} {'vs Best Spec':>13}")
    print(hdr)
    print("-" * 80)

    for cond in CONDITIONS:
        cshort = cond["short"]
        seed_results = [r for r in all_results if r["condition"] == cshort]
        if not seed_results: continue

        imps_base = [r["improvement_vs_base_pct"] for r in seed_results]
        imps_spec = [r["improvement_vs_spec_pct"] for r in seed_results]
        moe_vals  = [r["moe_losses"]["mixed"] for r in seed_results]

        m_base = statistics.mean(imps_base)
        m_spec = statistics.mean(imps_spec)
        m_moe  = statistics.mean(moe_vals)
        s_base = statistics.stdev(imps_base) if len(imps_base) > 1 else 0.0

        print(f"{cond['name']:<42} {len(seed_results):>5} "
              f"{m_moe:>10.4f} {m_base:>+8.2f}% {m_spec:>+12.2f}%"
              + (f" (±{s_base:.2f})" if s_base > 0 else ""))

    print("\nInterpretation:")
    cond_results = {}
    for cond in CONDITIONS:
        cshort = cond["short"]
        seed_results = [r for r in all_results if r["condition"] == cshort]
        if seed_results:
            cond_results[cshort] = statistics.mean(
                r["improvement_vs_base_pct"] for r in seed_results)

    ctrl  = cond_results.get("control")
    large = cond_results.get("large_gap")
    small = cond_results.get("small_gap")

    if ctrl is not None and large is not None:
        drop = ctrl - large
        print(f"  Control → Large gap: {ctrl:+.2f}% → {large:+.2f}% (drop={drop:.2f}pp)")
        if drop > 5:
            print("  STRONG EVIDENCE: shared initialization is critical for fusibility")
        elif drop > 2:
            print("  MODERATE EVIDENCE: shared initialization helps fusibility")
        else:
            print("  WEAK EVIDENCE: degradation is small — may not be critical")

    if ctrl is not None and small is not None:
        drop = ctrl - small
        print(f"  Control → Small gap: {ctrl:+.2f}% → {small:+.2f}% (drop={drop:.2f}pp)")


# ============================================================================
# Figure
# ============================================================================

def save_figure(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: improvement vs base by condition
        ax = axes[0]
        cond_means, cond_stds, cond_labels = [], [], []
        for cond in CONDITIONS:
            cshort = cond["short"]
            seed_results = [r for r in all_results if r["condition"] == cshort]
            if not seed_results: continue
            vals = [r["improvement_vs_base_pct"] for r in seed_results]
            cond_means.append(statistics.mean(vals))
            cond_stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            cond_labels.append(cond["short"].replace("_", "\n"))

        colors = ["#2ecc71", "#e74c3c", "#f39c12"][:len(cond_means)]
        x = np.arange(len(cond_means))
        ax.bar(x, cond_means, yerr=cond_stds, capsize=6,
               color=colors, width=0.55, zorder=3,
               error_kw={"elinewidth": 1.5, "ecolor": "#374151"})
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=9)
        ax.set_ylabel("MoE Improvement vs Base (%)")
        ax.set_title("Shared Initialization Ablation\n(MoE improvement by condition)")
        ax.axhline(0, color="#374151", lw=1.0)
        ax.grid(True, axis="y", alpha=0.3, zorder=0)
        for i, (m, s) in enumerate(zip(cond_means, cond_stds)):
            ax.text(i, m + s + 0.3, f"{m:+.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Right: per-seed scatter
        ax2 = axes[1]
        for i, cond in enumerate(CONDITIONS):
            cshort = cond["short"]
            seed_results = [r for r in all_results if r["condition"] == cshort]
            if not seed_results: continue
            seeds = [r["seed"] for r in seed_results]
            imps  = [r["improvement_vs_base_pct"] for r in seed_results]
            ax2.scatter(
                [i] * len(seeds), imps,
                color=colors[i], s=80, zorder=3,
                label=cond["short"].replace("_", " "),
            )
            mean_imp = statistics.mean(imps)
            ax2.hlines(mean_imp, i - 0.2, i + 0.2,
                       colors=colors[i], linewidths=2.5, zorder=4)

        ax2.set_xticks(range(len(CONDITIONS)))
        ax2.set_xticklabels([c["short"].replace("_", "\n") for c in CONDITIONS], fontsize=9)
        ax2.set_ylabel("MoE Improvement vs Base (%)")
        ax2.set_title("Per-Seed Results by Condition")
        ax2.axhline(0, color="#374151", lw=1.0)
        ax2.grid(True, axis="y", alpha=0.3, zorder=0)
        ax2.legend(fontsize=8)

        fig.suptitle("Does Shared Initialization Matter?\n"
                     "(Pythia-410M, 3-domain, freeze=4, 2000 steps)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_shared_init_ablation.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: figure failed: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Shared Initialization Necessity Ablation")
    print("=" * 70)
    for cond in CONDITIONS:
        print(f"  {cond['name']}")
        for d, r in cond["inits"].items():
            print(f"    {d}: {r}")
        print(f"  Seeds: {cond['seeds']}")
        print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer (revision doesn't matter for tokenizer on same model family)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision="step10000")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data once — same split used for all conditions
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

    # Base losses — use step10000 as canonical base (same as control condition)
    print("\nEvaluating base model (step10000)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision="step10000", torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    base_losses = {d: eval_loss(base_model, ds, device)
                   for d, ds in held_out_sets.items()}
    print(f"  Base (step10000): " + "  ".join(
        f"{d}={base_losses[d]:.4f}" for d in DOMAINS + ["mixed"]))
    del base_model
    torch.cuda.empty_cache()

    # =========================================================================
    # Run all conditions
    # =========================================================================
    all_results = []

    for cond in CONDITIONS:
        print(f"\n{'#'*65}")
        print(f"# {cond['name']}")
        print(f"{'#'*65}")

        for seed in cond["seeds"]:
            result = run_condition_seed(
                condition=cond,
                seed=seed,
                all_domain_chunks=all_domain_chunks,
                base_losses=base_losses,
                tokenizer=tokenizer,
                device=device,
            )
            all_results.append(result)

            # Save per-condition per-seed JSON
            out_path = RESULTS_DIR / f"result_{cond['short']}_seed{seed}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_summary(all_results)

    # Summary JSON
    summary_by_condition = {}
    for cond in CONDITIONS:
        cshort = cond["short"]
        seed_results = [r for r in all_results if r["condition"] == cshort]
        if not seed_results: continue
        imps  = [r["improvement_vs_base_pct"] for r in seed_results]
        specs = [r["improvement_vs_spec_pct"]  for r in seed_results]
        moes  = [r["moe_losses"]["mixed"]       for r in seed_results]
        summary_by_condition[cshort] = {
            "condition_name":      cond["name"],
            "init_revisions":      cond["inits"],
            "seeds_run":           [r["seed"] for r in seed_results],
            "improvement_vs_base": {
                "mean": round(statistics.mean(imps), 4),
                "std":  round(statistics.stdev(imps) if len(imps) > 1 else 0.0, 4),
                "values": imps,
            },
            "improvement_vs_spec": {
                "mean": round(statistics.mean(specs), 4),
                "std":  round(statistics.stdev(specs) if len(specs) > 1 else 0.0, 4),
                "values": specs,
            },
            "moe_mixed": {
                "mean": round(statistics.mean(moes), 6),
                "std":  round(statistics.stdev(moes) if len(moes) > 1 else 0.0, 6),
                "values": moes,
            },
        }

    ctrl_imp  = summary_by_condition.get("control",   {}).get("improvement_vs_base", {}).get("mean")
    large_imp = summary_by_condition.get("large_gap", {}).get("improvement_vs_base", {}).get("mean")
    small_imp = summary_by_condition.get("small_gap", {}).get("improvement_vs_base", {}).get("mean")

    conclusion = "undetermined"
    if ctrl_imp is not None and large_imp is not None:
        drop = ctrl_imp - large_imp
        if drop > 5:
            conclusion = "STRONG: shared init is critical"
        elif drop > 2:
            conclusion = "MODERATE: shared init helps"
        else:
            conclusion = "WEAK: shared init may not be critical"

    summary = {
        "experiment":   "shared_init_ablation",
        "model_id":     MODEL_ID,
        "max_steps":    MAX_STEPS,
        "freeze_layers": FREEZE_LAYERS,
        "base_losses":  base_losses,
        "by_condition": summary_by_condition,
        "conclusion":   conclusion,
        "control_improvement_pct":   ctrl_imp,
        "large_gap_improvement_pct": large_imp,
        "small_gap_improvement_pct": small_imp,
        "degradation_control_to_large": (
            round(ctrl_imp - large_imp, 4) if ctrl_imp and large_imp else None
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_path}")

    # Figure
    print("\nSaving figure...")
    save_figure(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
