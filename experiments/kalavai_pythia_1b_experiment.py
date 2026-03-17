#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-1B Three-Domain Specialist Fusion Experiment
===========================================================
Loads Pythia-1B at step10000 (~7% through training), trains three domain
specialists (code, science, fiction), verifies divergence, then fuses with
weight averaging and three-expert MoE routing.

Paper narrative:
  - Synthetic (zero prior):       +60.7%  <- mechanism works
  - Pythia-410M (early training): +14.2%  <- transfers to real models
  - Pythia-1B (early training):   +X.X%   <- scales to 1B
  - Qwen (fully trained):         -1.0%   <- diminishes with base knowledge

Hypothesis: Pythia-1B at step10000 knows basic English but has shallow domain
knowledge — 2000 fine-tuning steps can produce genuine specialist divergence.

Data split: All domains use a single 80/10/10 split on packed chunks.
ALL reported numbers use held_out_chunks only.
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

MODEL_ID = "EleutherAI/pythia-1b"
REVISION = "step10000"
FREEZE_LAYERS = 4        # 4/16 = 25%
LR = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4           # effective batch = 8
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE = 2048       # Pythia-1B hidden size
NUM_LAYERS = 16          # Pythia-1B layers
DOMAINS = ["code", "science", "fiction"]
SEEDS = [42, 137, 2026]
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50

RESULTS_DIR = Path("results/pythia/pythia_1b")
CHECKPOINT_DIR = Path("checkpoints/pythia/pythia_1b")
FIGURES_DIR = Path("figures/pythia")


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
        # Single linear layer router (no hidden layer), matches 1B spec
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
# Router distribution eval
# ============================================================================

@torch.no_grad()
def eval_router_distribution(moe: ThreeExpertMoE, eval_datasets: dict,
                              device: str, n_batches: int = 20) -> dict:
    """
    For each domain, run n_batches through the MoE and average gate weights.
    Returns {domain: [gate_0_mean, gate_1_mean, gate_2_mean]}.
    """
    moe.eval()
    results = {}
    for domain, ds in eval_datasets.items():
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                            drop_last=True, collate_fn=_collate)
        gate_sums = [0.0, 0.0, 0.0]
        count = 0
        for batch in loader:
            if count >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(3):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        if count > 0:
            results[domain] = [round(g / count, 4) for g in gate_sums]
        else:
            results[domain] = [0.333, 0.333, 0.333]
    return results


# ============================================================================
# Figures
# ============================================================================

def save_training_curves(loss_histories: dict, seed: int):
    """fig_1b_training_curves_seed42.png: 3-line loss plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"code": "#e74c3c", "science": "#2ecc71", "fiction": "#3498db"}
        for domain, history in loss_histories.items():
            if not history:
                continue
            steps = [h[0] for h in history]
            losses = [h[1] for h in history]
            ax.plot(steps, losses, label=f"{domain.capitalize()} specialist",
                    color=colors.get(domain, "gray"), linewidth=2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Average Loss")
        ax.set_title(f"Pythia-1B Specialist Training Curves (seed={seed})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"fig_1b_training_curves_seed{seed}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save training curves figure: {e}")


def save_divergence_heatmap(loss_matrix: dict, seed: int):
    """fig_1b_divergence_heatmap.png: 4x3 heatmap (base + 3 specialists x 3 domains)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        row_names = ["Base", "Code spec.", "Science spec.", "Fiction spec."]
        col_names = ["Code", "Science", "Fiction"]
        model_keys = ["base", "code", "science", "fiction"]

        data = np.zeros((4, 3))
        for i, mk in enumerate(model_keys):
            if mk in loss_matrix:
                for j, dk in enumerate(["code", "science", "fiction"]):
                    data[i, j] = loss_matrix[mk].get(dk, 0.0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(3))
        ax.set_xticklabels(col_names)
        ax.set_yticks(range(4))
        ax.set_yticklabels(row_names)
        ax.set_title(f"Divergence Heatmap — Held-Out Losses (Pythia-1B, seed={seed})")

        for i in range(4):
            for j in range(3):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="black")

        fig.colorbar(im, ax=ax, label="Cross-Entropy Loss")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_1b_divergence_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save divergence heatmap: {e}")


def save_fusion_comparison(fusion_results: dict, base_losses: dict, seed: int):
    """fig_1b_fusion_comparison.png: grouped bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        domains = ["code", "science", "fiction", "mixed"]
        model_order = [
            "base", "code_spec", "science_spec", "fiction_spec",
            "weight_avg", "moe"
        ]
        display_names = [
            "Base", "Code\nspec.", "Science\nspec.", "Fiction\nspec.",
            "Weight\navg.", "MoE"
        ]
        colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

        n_groups = len(domains)
        n_bars = len(model_order)
        x = np.arange(n_groups)
        width = 0.12

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (mk, name, color) in enumerate(zip(model_order, display_names, colors)):
            vals = [fusion_results.get(mk, {}).get(d, 0.0) for d in domains]
            offset = (i - n_bars / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains])
        ax.set_ylabel("Cross-Entropy Loss (lower is better)")
        ax.set_title(f"Fusion Comparison — Held-Out Eval (Pythia-1B, seed={seed})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_1b_fusion_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save fusion comparison figure: {e}")


def save_router_distribution(router_dist: dict, seed: int):
    """fig_1b_router_distribution.png: 3x3 bar chart of router weights per domain."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        domains = ["code", "science", "fiction"]
        expert_names = ["Code exp.", "Science exp.", "Fiction exp."]
        colors = ["#e74c3c", "#2ecc71", "#3498db"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, domain in zip(axes, domains):
            gates = router_dist.get(domain, [0.333, 0.333, 0.333])
            bars = ax.bar(expert_names, gates, color=colors, alpha=0.85)
            ax.set_title(f"Input: {domain.capitalize()}")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Mean gate weight" if domain == "code" else "")
            ax.set_xticklabels(expert_names, rotation=30, ha="right", fontsize=8)
            ax.axhline(y=1/3, linestyle="--", color="gray", alpha=0.5, label="Uniform")
            for bar, val in zip(bars, gates):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(f"Router Gate Distribution by Input Domain (Pythia-1B, seed={seed})")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_1b_router_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save router distribution figure: {e}")


# ============================================================================
# Step 0: Environment check
# ============================================================================

def step0_env_check():
    print("\n" + "=" * 70)
    print("STEP 0: Environment Check")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be extremely slow.")

    print(f"\nLoading {MODEL_ID} at revision={REVISION}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    n_layers = len(model.gpt_neox.layers)
    hidden_size = model.config.hidden_size

    print(f"\nModel parameters: {total_params/1e6:.1f}M")
    print(f"Layers: {n_layers}")
    print(f"Hidden size: {hidden_size}")

    assert 900e6 <= total_params <= 1100e6, f"Unexpected param count: {total_params/1e6:.1f}M"
    assert n_layers == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {n_layers}"
    assert hidden_size == HIDDEN_SIZE, f"Expected hidden_size={HIDDEN_SIZE}, got {hidden_size}"
    print("\nArchitecture verified (params ~1B, 16 layers, hidden=2048) OK")

    # Quick loss sanity check
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_text = "The quick brown fox jumps over the lazy dog."
    ids = tokenizer(test_text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    quick_loss = out.loss.item()
    print(f"Quick eval loss on test sentence: {quick_loss:.4f}")

    if quick_loss > 8.0:
        print("WARNING: loss > 8 — consider using step20000 instead of step10000.")
    else:
        print("Loss in acceptable range OK")

    del model
    torch.cuda.empty_cache()
    return device


# ============================================================================
# Step 1: Base model eval
# ============================================================================

def step1_base_eval(tokenizer, device: str) -> dict:
    print("\n" + "=" * 70)
    print("STEP 1: Base Model Eval + Data Stats")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 3 domains
    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    # Pack all into chunks first, then split 80/10/10
    print("\nPacking and splitting chunks (80/10/10)...")
    print(f"Note: Pythia-1B uses same NeoX tokenizer as 410M — chunk counts should match.")
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

    # Load base model
    print(f"\nLoading base model: {MODEL_ID} (revision={REVISION})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()

    # Eval on held-out chunks per domain + mixed
    print("\nEvaluating base model on held-out data...")
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}

    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    base_losses = {}
    for domain, ds in held_out_sets.items():
        loss = eval_loss(base_model, ds, device)
        base_losses[domain] = round(loss, 6)
        print(f"  Base [{domain:8s}]: {loss:.4f}")

    print(f"\nSanity check — losses should be in 3-6 range:")
    for d, l in base_losses.items():
        ok = "OK" if 2.0 <= l <= 8.0 else "UNEXPECTED"
        print(f"  {d}: {l:.4f} {ok}")

    del base_model
    torch.cuda.empty_cache()

    result = {
        "step": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "data_stats": {
            d: {
                "train_chunks": len(all_domain_chunks[d]["train"]),
                "indist_chunks": len(all_domain_chunks[d]["indist"]),
                "held_out_chunks": len(all_domain_chunks[d]["held_out"]),
            }
            for d in DOMAINS
        },
        "base_held_out_losses": base_losses,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "step1_base_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    return result, all_domain_chunks


# ============================================================================
# Step 2: Train specialists
# ============================================================================

def step2_train_specialists(tokenizer, all_domain_chunks: dict, device: str) -> dict:
    print("\n" + "=" * 70)
    print("STEP 2: Train Specialists (3 seeds x 3 domains = 9 runs)")
    print("=" * 70)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_loss_histories = {}   # seed -> domain -> history

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")

        all_loss_histories[seed] = {}

        for domain in DOMAINS:
            print(f"\nTraining {domain} specialist (seed={seed})...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)

            train_chunks = all_domain_chunks[domain]["train"]
            history = train_specialist(
                model, domain, train_chunks, tokenizer, seed, device
            )
            all_loss_histories[seed][domain] = history
            model.eval()

            # Save checkpoint
            ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            del model
            torch.cuda.empty_cache()

        # Save training curves for seed=42
        if seed == 42:
            print("\nSaving training curves figure...")
            save_training_curves(all_loss_histories[42], seed=42)

    print("\n[kalavai] pythia-1b step 2: specialists trained")
    return all_loss_histories


# ============================================================================
# Step 3: Divergence check
# ============================================================================

def step3_divergence_check(tokenizer, all_domain_chunks: dict, device: str) -> dict:
    print("\n" + "=" * 70)
    print("STEP 3: Divergence Check (3x3 matrix + base = 4x3)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build held-out eval datasets
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}

    seed42_loss_matrix = None
    all_seed_results = {}
    seeds_failed_all = []

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"DIVERGENCE CHECK — Seed {seed}")
        print(f"{'='*50}")

        # Load base model for comparison
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        base_model.eval()

        # Load all 3 specialists
        specialists = {}
        for domain in DOMAINS:
            ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
            print(f"  Loading {domain} specialist from {ckpt_path}...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            spec.load_state_dict(torch.load(ckpt_path, map_location=device))
            spec.eval()
            specialists[domain] = spec

        # Eval all 4 models (base + 3 specialists) on all 3 held-out domain sets
        loss_matrix = {}  # model_key -> {domain -> loss}

        for model_key, model in [("base", base_model)] + list(specialists.items()):
            loss_matrix[model_key] = {}
            for domain, ds in held_out_sets.items():
                l = eval_loss(model, ds, device)
                loss_matrix[model_key][domain] = round(l, 6)

        # Print 4x3 matrix
        print(f"\n{'Model':<20}" + "".join(f"{d:>12}" for d in DOMAINS))
        print("-" * (20 + 12 * 3))
        for mk in ["base"] + DOMAINS:
            row = f"{mk:<20}" + "".join(
                f"{loss_matrix[mk][d]:>12.4f}" for d in DOMAINS
            )
            print(row)

        # Checks: each specialist must beat base on its own domain
        checks = {}
        for domain in DOMAINS:
            base_loss = loss_matrix["base"][domain]
            spec_loss = loss_matrix[domain][domain]
            passed = spec_loss < base_loss
            checks[f"{domain}_beats_base"] = {
                "passed": passed,
                "base_loss": loss_matrix["base"][domain],
                "spec_loss": loss_matrix[domain][domain],
                "delta_pct": round((spec_loss - base_loss) / base_loss * 100, 2),
            }
            sym = "OK" if passed else "FAIL"
            print(f"  {sym} {domain} specialist on {domain}: "
                  f"{spec_loss:.4f} vs base {base_loss:.4f} "
                  f"({checks[f'{domain}_beats_base']['delta_pct']:+.1f}%)")

        seed_passed = all(c["passed"] for c in checks.values())
        print(f"\n  All divergence checks passed: {'YES' if seed_passed else 'NO'}")

        if not seed_passed:
            print(f"  DIVERGENCE CHECK FAILED for seed={seed}!")
            # Check if ALL 3 domain checks failed for this seed
            all_failed_for_seed = not any(c["passed"] for c in checks.values())
            if all_failed_for_seed:
                seeds_failed_all.append(seed)
                print(f"  CRITICAL: seed={seed} failed ALL 3 domain checks!")

        seed_result = {
            "seed": seed,
            "passed": seed_passed,
            "loss_matrix": loss_matrix,
            "checks": checks,
        }
        all_seed_results[seed] = seed_result

        out_path = RESULTS_DIR / f"step3_divergence_check_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(seed_result, f, indent=2)
        print(f"  Saved: {out_path}")

        if seed == 42:
            seed42_loss_matrix = loss_matrix

        del base_model
        for m in specialists.values():
            del m
        torch.cuda.empty_cache()

    # Save heatmap for seed=42
    if seed42_loss_matrix is not None:
        print("\nSaving divergence heatmap figure...")
        save_divergence_heatmap(seed42_loss_matrix, seed=42)

    # STOP if any seed failed ALL 3 domain checks
    if seeds_failed_all:
        print(f"\nFATAL: Seeds {seeds_failed_all} failed ALL 3 domain divergence checks.")
        print("This indicates the model is not specializing at all — stopping experiment.")
        print("Consider: more training steps, higher LR, fewer frozen layers, or different data.")
        sys.exit(1)

    return all_seed_results


# ============================================================================
# Step 4: Fusion
# ============================================================================

def step4_fusion(tokenizer, all_domain_chunks: dict, device: str,
                 divergence_results: dict) -> dict:
    print("\n" + "=" * 70)
    print("STEP 4: Fusion (Weight Avg + MoE)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build held-out eval datasets
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    # Build train datasets for router
    train_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                  for d in DOMAINS}

    seed42_fusion = None
    all_fusion_results = {}

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"FUSION — Seed {seed}")
        print(f"{'='*50}")

        div_check = divergence_results.get(seed, {})
        if not div_check.get("passed", True):
            print(f"  WARNING: seed={seed} failed divergence check, proceeding anyway.")

        # Load base model
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        base_model.eval()

        # Load 3 specialists
        specialists = {}
        for domain in DOMAINS:
            ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION,
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            spec.load_state_dict(torch.load(ckpt_path, map_location=device))
            spec.eval()
            specialists[domain] = spec
            print(f"  Loaded {domain} specialist")

        # 3-way weight average
        print("\nComputing 3-way weight average...")
        weight_avg_model = weight_average_three(
            specialists["code"], specialists["science"], specialists["fiction"]
        ).to(device)

        # Build MoE and train router
        print("\nBuilding ThreeExpertMoE (router: Linear(2048, 3))...")
        moe = ThreeExpertMoE(
            specialists["code"], specialists["science"], specialists["fiction"],
            hidden_size=HIDDEN_SIZE,
        ).to(device)
        train_router(moe, train_sets, device)
        moe.eval()

        # Router distribution eval (before full eval)
        print("\nEvaluating router distribution...")
        router_dist = eval_router_distribution(moe, held_out_sets, device)
        print("  Router gate weights per domain:")
        for domain, gates in router_dist.items():
            if domain == "mixed":
                continue
            print(f"    {domain:10s}: code={gates[0]:.3f}, science={gates[1]:.3f}, fiction={gates[2]:.3f}")

        # Eval all 6 variants on held-out data
        print("\nEvaluating all variants on held-out data...")

        eval_matrix = {}
        for model_key, model, is_fused in [
            ("base",          base_model,                False),
            ("code_spec",     specialists["code"],       False),
            ("science_spec",  specialists["science"],    False),
            ("fiction_spec",  specialists["fiction"],    False),
            ("weight_avg",    weight_avg_model,          False),
            ("moe",           moe,                       True),
        ]:
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

        # Compute improvement: MoE vs best individual on mixed held-out
        best_individual_mixed = min(
            eval_matrix["code_spec"]["mixed"],
            eval_matrix["science_spec"]["mixed"],
            eval_matrix["fiction_spec"]["mixed"],
        )
        moe_mixed = eval_matrix["moe"]["mixed"]
        improvement_pct = (best_individual_mixed - moe_mixed) / best_individual_mixed * 100
        print(f"\nImprovement over best individual (mixed): {improvement_pct:+.1f}%")
        print(f"[kalavai] pythia-1b seed={seed}: improvement={improvement_pct:.1f}%")

        seed_result = {
            "seed": seed,
            "eval_heldout": eval_matrix,
            "improvement_pct": round(improvement_pct, 4),
            "router_distribution": router_dist,
        }
        all_fusion_results[seed] = seed_result

        out_path = RESULTS_DIR / f"main_result_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(seed_result, f, indent=2)
        print(f"\nSaved: {out_path}")

        # Figures for seed=42 only
        if seed == 42:
            seed42_fusion = (eval_matrix, router_dist)
            print("\nSaving fusion figures...")
            save_fusion_comparison(eval_matrix, eval_matrix.get("base", {}), seed=42)
            save_router_distribution(router_dist, seed=42)

        del base_model, weight_avg_model, moe
        for m in specialists.values():
            del m
        torch.cuda.empty_cache()

    return all_fusion_results


# ============================================================================
# Step 5: Final summary
# ============================================================================

def step5_final_summary(step1_result: dict, step3_results: dict,
                         step4_results: dict) -> dict:
    print("\n" + "=" * 70)
    print("STEP 5: Final Summary")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    improvements = [r["improvement_pct"] for r in step4_results.values()]
    mean_imp = statistics.mean(improvements)
    std_imp = statistics.stdev(improvements) if len(improvements) > 1 else 0.0

    seeds_passed_divergence = sum(
        1 for r in step3_results.values() if r.get("passed", False)
    )

    print(f"\n{'Seed':<8} {'Div. Check':<14} {'Improvement':>14}")
    print("-" * 40)
    for seed in SEEDS:
        div_ok = step3_results.get(seed, {}).get("passed", False)
        imp = step4_results.get(seed, {}).get("improvement_pct", float("nan"))
        div_str = "PASS" if div_ok else "FAIL"
        print(f"{seed:<8} {div_str:<14} {imp:>+13.1f}%")

    print("-" * 40)
    print(f"{'Mean':<8} {'':14} {mean_imp:>+13.1f}%")
    if len(improvements) > 1:
        print(f"{'Std':<8} {'':14} {std_imp:>13.1f}%")

    print(f"\nINTERPRETATION:")
    if mean_imp > 20:
        print(f"  A: Strong result ({mean_imp:.1f}%) — fusion generalizes well to 1B scale.")
    elif mean_imp > 5:
        print(f"  B: Modest result ({mean_imp:.1f}%) — fusion works incrementally at 1B scale.")
    elif mean_imp > 0:
        print(f"  C: Weak positive ({mean_imp:.1f}%) — marginal improvement, confirms pattern direction.")
    else:
        print(f"  D: Neutral/negative ({mean_imp:.1f}%) — fusion not helping at this training stage.")

    # Cross-scale comparison
    SCALE_410M_MEAN = 14.2
    SCALE_410M_STD = 0.0
    print(f"\nSCALE COMPARISON (step10000, 3 domains, freeze=4)")
    print(f"{'Model Size':<14} {'Improvement':<14} {'Std'}")
    print("-" * 42)
    print(f"{'410M':<14} {SCALE_410M_MEAN:>+.1f}%        +/-{SCALE_410M_STD:.1f}%")
    print(f"{'1B':<14} {mean_imp:>+.1f}%        +/-{std_imp:.1f}%")

    print(f"\nPaper narrative context:")
    print(f"  Synthetic (zero prior):    +60.7%")
    print(f"  Pythia-410M (early):       +14.2% +/-0.0%")
    print(f"  Pythia-1B (early):         {mean_imp:+.1f}% +/-{std_imp:.1f}%")
    print(f"  Qwen (fully trained):      -1.0%")

    summary = {
        "step": 5,
        "experiment": "pythia_1b_three_domain",
        "model_id": MODEL_ID,
        "revision": REVISION,
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
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
        },
        "seeds": SEEDS,
        "data_stats": step1_result.get("data_stats", {}),
        "base_held_out_losses": step1_result.get("base_held_out_losses", {}),
        "per_seed_divergence": {
            seed: {
                "passed": r.get("passed", False),
                "checks": r.get("checks", {}),
            }
            for seed, r in step3_results.items()
        },
        "per_seed_fusion": {
            seed: {
                "eval_heldout": r.get("eval_heldout", {}),
                "improvement_pct": r.get("improvement_pct", None),
                "router_distribution": r.get("router_distribution", {}),
            }
            for seed, r in step4_results.items()
        },
        "summary": {
            "seeds_passed_divergence": seeds_passed_divergence,
            "improvement_mean_pct": round(mean_imp, 4),
            "improvement_std_pct": round(std_imp, 4),
        },
        "scale_comparison": {
            "410m_mean_pct": SCALE_410M_MEAN,
            "410m_std_pct": SCALE_410M_STD,
            "1b_mean_pct": round(mean_imp, 4),
            "1b_std_pct": round(std_imp, 4),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "main_result_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n[kalavai] pythia-1b complete: mean={mean_imp:.1f}% std={std_imp:.1f}%")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-1B Three-Domain Specialist Fusion Experiment")
    print("=" * 70)
    print(f"Model:    {MODEL_ID} @ revision={REVISION}")
    print(f"Domains:  {DOMAINS}")
    print(f"Steps:    {MAX_STEPS} per specialist (freeze_layers={FREEZE_LAYERS})")
    print(f"Seeds:    {SEEDS}")
    print(f"Config:   LR={LR}, WD={WEIGHT_DECAY}, BATCH={BATCH_SIZE}x{GRAD_ACCUM}")
    print(f"Hidden:   {HIDDEN_SIZE}, Layers: {NUM_LAYERS}")

    # Step 0
    device = step0_env_check()

    # Load tokenizer once
    print(f"\nLoading tokenizer from {MODEL_ID} (revision={REVISION})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1
    step1_result, all_domain_chunks = step1_base_eval(tokenizer, device)
    print("\n[Step 1 complete]")

    # Step 2
    step2_train_specialists(tokenizer, all_domain_chunks, device)
    print("\n[Step 2 complete]")

    # Step 3
    step3_results = step3_divergence_check(tokenizer, all_domain_chunks, device)
    print("\n[Step 3 complete]")

    # Step 4
    step4_results = step4_fusion(tokenizer, all_domain_chunks, device, step3_results)
    print("\n[Step 4 complete]")

    # Step 5
    step5_final_summary(step1_result, step3_results, step4_results)
    print("\n[Step 5 complete]")

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
