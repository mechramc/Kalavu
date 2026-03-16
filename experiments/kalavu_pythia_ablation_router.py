#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Pythia-410M Router Architecture Ablation
=================================================
Reuses seed=42 specialist checkpoints from the main experiment.
Tests 3 router variants on the same frozen specialists:
  A) Uniform (no router — 1/3 each)
  B) Simple Linear(1024, 3)  [what the spec originally called for]
  C) 2-Layer Linear(1024->256)->ReLU->Linear(256,3)  [what main exp used]

Purpose: confirm +14.2% result is about the fusion mechanism, not router complexity.
"""

import json
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Config — must match main experiment exactly
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
FREEZE_LAYERS = 4
SEQ_LEN = 512
HIDDEN_SIZE = 1024
DOMAINS = ["code", "science", "fiction"]
SEED = 42

ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50

N_SAMPLES_PER_DOMAIN = 3000

CHECKPOINT_DIR = Path("checkpoints/pythia")
RESULTS_DIR = Path("results/pythia")
FIGURES_DIR = Path("figures/pythia")


# ============================================================================
# Dataset (copied from main experiment)
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
# Data loading (same as main experiment)
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
# MoE variants
# ============================================================================

class UniformMoE(nn.Module):
    """No router — equal 1/3 weight to each specialist."""
    def __init__(self, spec_a, spec_b, spec_c):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            return model(input_ids=input_ids).logits.detach()

    def forward(self, input_ids, labels=None):
        la = self._run(self.spec_a, input_ids)
        lb = self._run(self.spec_b, input_ids)
        lc = self._run(self.spec_c, input_ids)
        fused = (la + lb + lc) / 3.0
        gates = torch.full((input_ids.shape[0], 3), 1/3, device=input_ids.device)
        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, gates


class LearnedMoE(nn.Module):
    """MoE with a pluggable router module."""
    def __init__(self, spec_a, spec_b, spec_c, router: nn.Module):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        self.router = router
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)

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
# Training / Eval helpers
# ============================================================================

def train_router(moe, train_chunks_combined, device):
    combined = make_dataset_from_chunks(train_chunks_combined)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({ROUTER_STEPS} steps, {len(combined)} chunks)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    step {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")


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


@torch.no_grad()
def eval_router_dist(moe, held_out_sets, device, n_batches=20):
    moe.eval()
    results = {}
    for domain, ds in held_out_sets.items():
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
        results[domain] = [round(g / max(count, 1), 4) for g in gate_sums]
    return results


# ============================================================================
# Figure
# ============================================================================

def save_router_ablation_figure(variant_results, best_individual):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Best\nIndividual", "Uniform\n(no router)", "Simple\nLinear", "2-Layer\n(main exp)"]
        keys = ["best_individual", "uniform", "simple_linear", "two_layer"]
        losses = [best_individual] + [variant_results[k]["mixed_loss"] for k in ["uniform", "simple_linear", "two_layer"]]
        imps = [0.0] + [variant_results[k]["improvement_pct"] for k in ["uniform", "simple_linear", "two_layer"]]
        colors = ["#95a5a6", "#f39c12", "#3498db", "#9b59b6"]

        # Y-axis range: zoom in to show differences clearly
        y_min = min(losses) * 0.995
        y_max = max(losses) * 1.005

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, losses, color=colors, alpha=0.85, width=0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Held-Out Mixed Loss (lower is better)")
        ax.set_title("Router Architecture Ablation (Pythia-410M, seed=42, freeze=4)")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, loss, imp in zip(bars, losses, imps):
            label = f"{loss:.4f}"
            if imp != 0.0:
                label += f"\n({imp:+.1f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (y_max - y_min) * 0.005,
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_ablation_router.png"
        fig.savefig(path, dpi=150)
        import matplotlib.pyplot as plt2; plt2.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU: Pythia-410M Router Architecture Ablation")
    print("=" * 70)
    print(f"Reusing seed=42 specialist checkpoints (freeze={FREEZE_LAYERS})")
    print(f"Testing: uniform, simple_linear, two_layer routers")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data and build splits (same as main experiment)
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

    combined_train = []
    for d in DOMAINS:
        combined_train.extend(all_domain_chunks[d]["train"])

    # Load specialists (seed=42)
    print("\nLoading seed=42 specialists...")
    specialists = {}
    for domain in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed42.pt"
        spec = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        ).to(device)
        spec.load_state_dict(torch.load(ckpt, map_location=device))
        spec.eval()
        specialists[domain] = spec
        print(f"  Loaded {domain} from {ckpt}")

    # Eval best individual (baseline)
    print("\nEvaluating best individual baseline...")
    best_individual_loss = min(
        eval_loss(specialists[d], held_out_sets["mixed"], device)
        for d in DOMAINS
    )
    print(f"  Best individual mixed loss: {best_individual_loss:.4f}")

    variant_results = {}

    # -------------------------------------------------------------------------
    # Variant A: Uniform (no router)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Variant A: Uniform (no learned router, 1/3 each)")
    print("-" * 50)
    moe_uniform = UniformMoE(specialists["code"], specialists["science"],
                             specialists["fiction"]).to(device)
    uniform_loss = eval_loss(moe_uniform, held_out_sets["mixed"], device,
                             batch_size=2, is_fused=True)
    uniform_imp = (best_individual_loss - uniform_loss) / best_individual_loss * 100
    gate_dist = {d: [0.333, 0.333, 0.333] for d in DOMAINS}
    print(f"  Mixed loss: {uniform_loss:.4f} ({uniform_imp:+.1f}%)")
    variant_results["uniform"] = {
        "mixed_loss": round(uniform_loss, 6),
        "improvement_pct": round(uniform_imp, 4),
        "gate_distribution": gate_dist,
    }
    del moe_uniform
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Variant B: Simple Linear(1024, 3)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Variant B: Simple Linear(1024, 3)")
    print("-" * 50)
    router_simple = nn.Linear(HIDDEN_SIZE, 3, bias=False).to(device)
    moe_simple = LearnedMoE(specialists["code"], specialists["science"],
                            specialists["fiction"], router_simple).to(device)
    train_router(moe_simple, combined_train, device)
    moe_simple.eval()
    simple_loss = eval_loss(moe_simple, held_out_sets["mixed"], device,
                            batch_size=2, is_fused=True)
    simple_imp = (best_individual_loss - simple_loss) / best_individual_loss * 100
    simple_gate_dist = eval_router_dist(moe_simple, held_out_sets, device)
    print(f"  Mixed loss: {simple_loss:.4f} ({simple_imp:+.1f}%)")
    print(f"  Gate dist: {simple_gate_dist}")
    variant_results["simple_linear"] = {
        "mixed_loss": round(simple_loss, 6),
        "improvement_pct": round(simple_imp, 4),
        "gate_distribution": simple_gate_dist,
    }
    del moe_simple
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Variant C: 2-Layer (main experiment router)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Variant C: 2-Layer Linear(1024->256)->ReLU->Linear(256,3)")
    print("-" * 50)
    router_2layer = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 3, bias=False),
    ).to(device)
    moe_2layer = LearnedMoE(specialists["code"], specialists["science"],
                            specialists["fiction"], router_2layer).to(device)
    train_router(moe_2layer, combined_train, device)
    moe_2layer.eval()
    twolayer_loss = eval_loss(moe_2layer, held_out_sets["mixed"], device,
                              batch_size=2, is_fused=True)
    twolayer_imp = (best_individual_loss - twolayer_loss) / best_individual_loss * 100
    twolayer_gate_dist = eval_router_dist(moe_2layer, held_out_sets, device)
    print(f"  Mixed loss: {twolayer_loss:.4f} ({twolayer_imp:+.1f}%)")
    print(f"  Gate dist: {twolayer_gate_dist}")
    variant_results["two_layer"] = {
        "mixed_loss": round(twolayer_loss, 6),
        "improvement_pct": round(twolayer_imp, 4),
        "gate_distribution": twolayer_gate_dist,
    }
    del moe_2layer
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Results table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROUTER ABLATION RESULTS (Pythia-410M, seed=42, freeze=4)")
    print("=" * 70)
    print(f"{'Router Type':<35} {'Mixed Loss':>12} {'Improvement':>13}")
    print("-" * 62)
    print(f"{'Best individual (baseline)':<35} {best_individual_loss:>12.4f} {'(baseline)':>13}")
    for name, label in [("uniform", "Uniform (no router)"),
                         ("simple_linear", "Simple Linear(1024,3)"),
                         ("two_layer", "2-Layer (main exp)")]:
        r = variant_results[name]
        print(f"{label:<35} {r['mixed_loss']:>12.4f} {r['improvement_pct']:>+12.1f}%")

    # Figure
    print("\nSaving figure...")
    save_router_ablation_figure(variant_results, best_individual_loss)

    # Save JSON
    output = {
        "experiment": "router_ablation",
        "base_model": f"pythia-410m-{REVISION}",
        "seed": SEED,
        "freeze_layers": FREEZE_LAYERS,
        "best_individual_mixed": round(best_individual_loss, 6),
        "variants": variant_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "ablation_router_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    print("\nConclusion:")
    sv = variant_results["simple_linear"]["improvement_pct"]
    tv = variant_results["two_layer"]["improvement_pct"]
    diff = abs(sv - tv)
    if diff < 1.0:
        print(f"  Router depth DOES NOT matter ({diff:.1f}% difference).")
        print(f"  Result is robust to router architecture.")
        print(f"  Paper can use simple linear router for all comparisons.")
    else:
        print(f"  Router depth matters ({diff:.1f}% difference).")
        print(f"  Use consistent router architecture across all experiments.")


if __name__ == "__main__":
    main()
