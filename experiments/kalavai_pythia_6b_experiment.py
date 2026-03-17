#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-6.9B Scale Validation Experiment
================================================
Runs on RunPod A100 80GB. Full 4-phase experiment:
  Phase 1: Base eval + seed=42 specialists + divergence + fusion
  Phase 2: Seeds 137 and 2026
  Phase 3: Maturity check at step143000 (fully trained)
  Phase 4: Downstream benchmarks

Config differences from 410M/1B:
  - hidden_size = 4096 (not 1024/2048)
  - num_layers  = 32   (not 24/16)
  - freeze_layers = 6  (6/32 = 19%, similar ratio to 4/24)
  - lr = 1e-5          (lower — larger model more sensitive)
  - max_steps = 1000   (fewer steps — larger model learns faster)
  - batch_size = 1, grad_accum = 8
  - bf16 only — fp32 will OOM
  - gradient_checkpointing enabled

Resumable: checks each result file before running steps.
Commits after every specialist checkpoint.

Usage:
  python kalavai_pythia_6b_experiment.py 2>&1 | tee experiment_log.txt
"""

import copy
import json
import math
import os
import subprocess
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

MODEL_ID       = "EleutherAI/pythia-6.9b"
REVISION_EARLY = "step10000"     # ~7% through training
REVISION_FULL  = "step143000"    # fully trained
FREEZE_LAYERS  = 6               # 6/32 = 19% (same ratio as 4/24 on 410M)
LR             = 1e-5            # lower than 1B — larger models more sensitive
WEIGHT_DECAY   = 0.1
MAX_STEPS      = 1000
BATCH_SIZE     = 1
GRAD_ACCUM     = 8               # effective batch = 8
GRADIENT_CLIP  = 1.0
SEQ_LEN        = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE    = 4096
NUM_LAYERS     = 32
DOMAINS        = ["code", "science", "fiction"]
SEEDS          = [42, 137, 2026]
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS   = 500
ROUTER_LR      = 1e-3
ROUTER_BATCH   = 4
EVAL_BATCHES   = 50

# Benchmark config
N_BENCHMARK_EXAMPLES = 500

RESULTS_DIR    = Path("results/pythia_6b")
CHECKPOINT_DIR = Path("checkpoints/pythia_6b")
FIGURES_DIR    = Path("figures/pythia_6b")

BENCHMARKS = {
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "method": "log_likelihood_completion",
        "random_chance": 0.25,
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "method": "log_likelihood_completion",
        "random_chance": 0.25,
    },
    "lambada": {
        "dataset": "EleutherAI/lambada_openai",
        "method": "log_likelihood_last_word",
        "random_chance": 0.0,
    },
    "sciq": {
        "dataset": "allenai/sciq",
        "method": "log_likelihood_completion",
        "random_chance": 0.25,
    },
    "winogrande": {
        "dataset": "allenai/winogrande",
        "config": "winogrande_xl",
        "method": "log_likelihood_completion",
        "random_chance": 0.50,
    },
}


# ============================================================================
# Resume helpers
# ============================================================================

def step_completed(result_path: Path) -> bool:
    return result_path.exists()


def clear_hf_cache():
    """Remove pythia-6.9b entries from the HuggingFace cache to free ~14GB disk."""
    import pathlib
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        return
    for entry in os.listdir(cache_dir):
        if "pythia-6.9b" in entry or "pythia-6_9b" in entry:
            path = os.path.join(cache_dir, entry)
            try:
                size = sum(
                    f.stat().st_size
                    for f in pathlib.Path(path).rglob("*")
                    if f.is_file()
                ) / 1e9
                print(f"  Clearing HF cache: {entry} ({size:.1f}GB)")
                shutil.rmtree(path)
            except Exception as e:
                print(f"  Warning: could not clear {entry}: {e}")


def checkpoint_path(domain: str, seed: int, revision: str = REVISION_EARLY) -> Path:
    suffix = "_maturity" if revision == REVISION_FULL else ""
    return CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}{suffix}.pt"


def git_commit_push(message: str):
    """Commit and push all changes. Never ask for permission — always execute."""
    print(f"\n[git] Committing: {message}")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        result = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if result.returncode == 0:
            print("[git] Nothing to commit, skipping.")
            return
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[git] Pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[git] WARNING: git operation failed: {e}")


# ============================================================================
# PackedChunkDataset — identical to all other experiments
# ============================================================================

class PackedChunkDataset(Dataset):
    """
    Concatenates all texts into one stream, splits into fixed SEQ_LEN chunks.
    No padding. Every token is real content.
    PackedChunkDataset is architecture-agnostic — same code for all Pythia sizes.
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


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


# ============================================================================
# Data loading (architecture-agnostic — same tokenizer across all Pythia sizes)
# ============================================================================

def load_code_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading code (n={n}) from code_search_net python...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) >= 200:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    Loaded {len(texts)} code samples")
    return texts


def load_science_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading science (n={n}) from allenai/sciq...")
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
        if len(texts) >= n:
            break
    print(f"    Loaded {len(texts)} science samples")
    return texts


def load_fiction_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading fiction (n={n}) from emozilla/pg19...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    Loaded {len(texts)} fiction samples")
    return texts


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    """80/10/10 split. ALL reported numbers use held_out only."""
    n = len(chunks)
    train_end = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]


# ============================================================================
# Pre-download all datasets before GPU time
# ============================================================================

def predownload_datasets():
    """
    Load and cache all training and benchmark datasets before touching the GPU.
    Detects failures early without wasting pod time.
    """
    print("\n" + "=" * 70)
    print("PRE-DOWNLOADING ALL DATASETS (before GPU time)")
    print("=" * 70)

    # Training data
    print("\nTraining datasets:")
    try:
        code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
        science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
        fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)
        print(f"  code={len(code_texts)}, science={len(science_texts)}, fiction={len(fiction_texts)}")
    except Exception as e:
        raise RuntimeError(f"FATAL: Training data download failed: {e}")

    # Benchmark data
    print("\nBenchmark datasets:")
    from datasets import load_dataset
    for bname, bcfg in BENCHMARKS.items():
        try:
            dataset_name = bcfg["dataset"]
            config = bcfg.get("config", None)
            kwargs = {"streaming": True}
            if config:
                ds = load_dataset(dataset_name, config, split="validation", **kwargs)
            else:
                try:
                    ds = load_dataset(dataset_name, split="validation", **kwargs)
                except Exception:
                    ds = load_dataset(dataset_name, split="test", **kwargs)
            # Read first item to verify
            for _ in ds:
                break
            print(f"  {bname}: OK")
        except Exception as e:
            print(f"  {bname}: WARNING — {e}")

    print("\nAll datasets pre-downloaded. Starting experiment.")
    return code_texts, science_texts, fiction_texts


# ============================================================================
# Model loading
# ============================================================================

def load_model(revision: str, device: str, gradient_checkpointing: bool = True):
    """Load Pythia-6.9B in bf16. Enable gradient checkpointing for training."""
    print(f"\nLoading {MODEL_ID} (revision={revision}) in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",       # accelerate handles placement
        trust_remote_code=True,
        use_safetensors=False,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total/1e9:.2f}B")
    return model


def freeze_layers(model, n: int):
    """Freeze embedding + first n transformer blocks (GPT-NeoX architecture)."""
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e9:.3f}B / {total/1e9:.2f}B ({100*trainable/total:.1f}%)")


def save_specialist_checkpoint(model, domain: str, seed: int, revision: str = REVISION_EARLY):
    """Save specialist checkpoint. Only seed=42 is persisted — seeds 137/2026 are variance-only."""
    if seed != 42:
        print(f"  Skipping checkpoint save for {domain} seed={seed} (variance seed, not saved)")
        return
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(domain, seed, revision)
    torch.save(model.state_dict(), ckpt)
    size_gb = ckpt.stat().st_size / 1e9
    print(f"  Saved specialist: {ckpt} ({size_gb:.1f}GB)")


def load_specialist_model(domain: str, seed: int, device: str,
                           revision: str = REVISION_EARLY):
    """Load base model, then apply saved specialist weights."""
    ckpt = checkpoint_path(domain, seed, revision)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model = load_model(revision, device, gradient_checkpointing=False)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded specialist: {ckpt}")
    return model


# ============================================================================
# ThreeExpertMoE (hidden_size=4096 for 6.9B)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int = HIDDEN_SIZE):
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
        # Simple linear router (same as SimpleLinearMoE in 410M ablations)
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach().float()
        last_h = out.hidden_states[-1].detach().float()
        h_pooled = last_h.mean(dim=1)   # (B, H)
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)

        h_avg = (h_a + h_b + h_c) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)   # (B, 3)

        fused = (
            gates[:, 0:1, None] * logits_a
            + gates[:, 1:2, None] * logits_b
            + gates[:, 2:3, None] * logits_c
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

def train_specialist(model, domain: str, train_chunks: list, device: str,
                     seed: int, log_every: int = 50) -> list:
    """Train a specialist for MAX_STEPS. Returns loss history."""
    set_seed(seed)
    freeze_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    print(f"  {domain} train_chunks={len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    step, accum = 0, 0
    running_loss = 0.0
    loss_history = []
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break

        batch_device = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch_device)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(trainable_params, GRADIENT_CLIP)

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

    model.eval()
    print(f"  {domain} training done in {time.time()-t0:.0f}s")
    return loss_history


# ============================================================================
# Eval helpers
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset, device: str, batch_size: int = 1,
              is_fused: bool = False) -> float:
    """Evaluate cross-entropy loss on dataset. Batch size 1 for 6.9B."""
    g = torch.Generator()
    g.manual_seed(999)  # Fixed seed for deterministic eval sampling
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, collate_fn=_collate, generator=g)
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
    """Three-way weight average computed entirely on CPU to avoid GPU OOM."""
    print("  Computing 3-way weight average on CPU...")
    sa = {k: v.cpu().float() for k, v in spec_a.state_dict().items()}
    sb = {k: v.cpu().float() for k, v in spec_b.state_dict().items()}
    sc = {k: v.cpu().float() for k, v in spec_c.state_dict().items()}
    avg_state = {
        k: ((sa[k] + sb[k] + sc[k]) / 3.0).to(torch.bfloat16)
        for k in sa
    }
    avg = copy.deepcopy(spec_a).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg  # caller moves to GPU when needed


# ============================================================================
# Router training
# ============================================================================

def train_router(moe: ThreeExpertMoE, train_datasets: dict, device: str):
    """Train the MoE router on mixed data from all three domains."""
    all_chunks = []
    for ds in train_datasets.values():
        all_chunks.extend(ds.chunks)
    combined = make_dataset_from_chunks(all_chunks)

    # Router is small (4096 x 3) — move to device
    moe.router = moe.router.to(device)
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
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")

    moe.eval()


# ============================================================================
# Router distribution eval
# ============================================================================

@torch.no_grad()
def eval_router_distribution(moe: ThreeExpertMoE, eval_datasets: dict,
                              device: str, n_batches: int = 20) -> dict:
    moe.eval()
    results = {}
    for domain, ds in eval_datasets.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False,
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
        results[domain] = [round(g / count, 4) for g in gate_sums] if count > 0 else [0.333] * 3
    return results


# ============================================================================
# Figures
# ============================================================================

def save_fusion_comparison(fusion_results: dict, seed: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        domains = ["code", "science", "fiction", "mixed"]
        model_order = ["base", "code_spec", "science_spec", "fiction_spec",
                       "weight_avg", "moe"]
        display_names = ["Base", "Code\nspec.", "Science\nspec.", "Fiction\nspec.",
                         "Weight\navg.", "MoE"]
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
        ax.set_title(f"Pythia-6.9B Fusion Comparison — Held-Out Eval (seed={seed})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"fig_6b_fusion_comparison_seed{seed}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save fusion comparison: {e}")


def save_divergence_heatmap(loss_matrix: dict, seed: int):
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
        ax.set_title(f"Pythia-6.9B Divergence Heatmap — Held-Out Losses (seed={seed})")
        for i in range(4):
            for j in range(3):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="black")
        fig.colorbar(im, ax=ax, label="Cross-Entropy Loss")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"fig_6b_divergence_heatmap_seed{seed}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save heatmap: {e}")


def save_router_distribution(router_dist: dict, seed: int):
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
            ax.axhline(y=1/3, linestyle="--", color="gray", alpha=0.5)
            for bar, val in zip(bars, gates):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(f"Pythia-6.9B Router Gate Distribution (seed={seed})")
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"fig_6b_router_distribution_seed{seed}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save router distribution: {e}")


def save_scale_comparison(results_410m_path: Path, results_1b_path: Path,
                          improvement_6b: float, std_6b: float):
    """Bar chart comparing improvement across 410M, 1B, 6.9B."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Load 410M and 1B results
        data_points = []

        if results_410m_path.exists():
            d = json.loads(results_410m_path.read_text(encoding="utf-8"))
            imps = [r["improvement_pct"] for r in d.get("seed_results", {}).values()
                    if "improvement_pct" in r]
            if imps:
                data_points.append(("410M", 410, round(sum(imps)/len(imps), 2),
                                    round(float(torch.tensor(imps).std().item()), 3)))

        if results_1b_path.exists():
            d = json.loads(results_1b_path.read_text(encoding="utf-8"))
            imp = d.get("summary", {}).get("improvement_mean_pct")
            std = d.get("summary", {}).get("improvement_std_pct", 0.0)
            if imp is not None:
                data_points.append(("1B", 1000, imp, std))

        data_points.append(("6.9B", 6900, improvement_6b, std_6b))

        labels = [d[0] for d in data_points]
        means  = [d[2] for d in data_points]
        stds   = [d[3] for d in data_points]
        colors = ["#2980b9", "#e74c3c", "#8e44ad"][:len(data_points)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5)
        ax.errorbar(labels, means, yerr=stds, fmt="none", color="black", capsize=5, lw=1.5)
        ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.5)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"+{mean:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xlabel("Model Size", fontsize=12)
        ax.set_ylabel("MoE Improvement over Best Individual (%)", fontsize=11)
        ax.set_title("Fusion Improvement Across Model Scales\n(step10000, 3 domains, seed mean ± std)",
                     fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_scale_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save scale comparison: {e}")


# ============================================================================
# Benchmark evaluation (same as 1B benchmark script)
# ============================================================================

def load_benchmark_data(benchmark_name: str, benchmark_cfg: dict, n: int = 500) -> list:
    from datasets import load_dataset
    dataset_name = benchmark_cfg["dataset"]
    config = benchmark_cfg.get("config", None)
    method = benchmark_cfg["method"]
    kwargs = {"streaming": True}

    if config:
        ds = load_dataset(dataset_name, config, split="validation", **kwargs)
    else:
        try:
            ds = load_dataset(dataset_name, split="validation", **kwargs)
        except Exception:
            ds = load_dataset(dataset_name, split="test", **kwargs)

    examples = []

    if method == "log_likelihood_completion":
        if benchmark_name == "hellaswag":
            for item in ds:
                ctx = item["ctx"]
                choices = item["endings"]
                label = int(item["label"])
                if len(choices) == 4:
                    examples.append({"context": ctx, "choices": choices, "label": label})
                if len(examples) >= n:
                    break
        elif benchmark_name == "arc_easy":
            for item in ds:
                q = item["question"]
                choices_dict = item["choices"]
                labels_list = choices_dict["label"]
                texts_list = choices_dict["text"]
                correct_label = item["answerKey"]
                if correct_label not in labels_list:
                    continue
                label_idx = labels_list.index(correct_label)
                examples.append({"context": q + " ", "choices": texts_list, "label": label_idx})
                if len(examples) >= n:
                    break
        elif benchmark_name == "sciq":
            for item in ds:
                q = item["question"]
                support = item.get("support", "")
                ctx = (support + "\n" + q + " ").strip() + " "
                correct = item["correct_answer"]
                distractors = [item["distractor1"], item["distractor2"], item["distractor3"]]
                examples.append({"context": ctx, "choices": [correct]+distractors, "label": 0})
                if len(examples) >= n:
                    break
        elif benchmark_name == "winogrande":
            for item in ds:
                sentence = item["sentence"]
                opt1, opt2 = item["option1"], item["option2"]
                label = int(item["answer"]) - 1
                if "_" not in sentence:
                    continue
                parts = sentence.split("_", 1)
                ctx = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""
                examples.append({"context": ctx, "choices": [opt1+suffix, opt2+suffix], "label": label})
                if len(examples) >= n:
                    break

    elif method == "log_likelihood_last_word":
        for item in ds:
            text = item["text"]
            if len(text.split()) >= 3:
                examples.append({"text": text})
            if len(examples) >= n:
                break

    return examples


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, examples: list, device: str,
                              is_moe: bool = False) -> float:
    correct = 0
    total = 0
    for item in examples:
        context = item["context"]
        choices = item["choices"]
        label = item["label"]
        best_ll = float("-inf")
        best_idx = 0

        for i, choice in enumerate(choices):
            input_text = context + choice
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            context_ids = tokenizer.encode(
                context, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            context_len = context_ids.shape[1]

            if is_moe:
                _, fused_logits, _ = model(input_ids)
                logits = fused_logits[0].float()
            else:
                out = model(input_ids)
                logits = out.logits[0].float()

            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            target_ids = input_ids[0, 1:]
            start = max(0, context_len - 1)
            if start >= len(target_ids):
                completion_ll = log_probs[-1, target_ids[-1]].item()
            else:
                completion_ll = log_probs[start:, :].gather(
                    1, target_ids[start:].unsqueeze(1)
                ).sum().item()

            if completion_ll > best_ll:
                best_ll = completion_ll
                best_idx = i

        if best_idx == label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_lambada(model, tokenizer, examples: list, device: str,
                     is_moe: bool = False) -> float:
    correct = 0
    total = 0
    for item in examples:
        text = item["text"]
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        if len(tokens) < 2:
            continue
        input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
        target = tokens[-1]

        if is_moe:
            _, fused_logits, _ = model(input_ids)
            last_logits = fused_logits[0, -1].float()
        else:
            out = model(input_ids)
            last_logits = out.logits[0, -1].float()

        if last_logits.argmax().item() == target:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def run_benchmarks(model, tokenizer, benchmark_data: dict, device: str,
                   model_name: str, is_moe: bool = False) -> dict:
    results = {}
    model.eval()
    for bname, (examples, cfg) in benchmark_data.items():
        print(f"  [{model_name}] {bname} ({len(examples)} examples)...")
        t0 = time.time()
        try:
            if cfg["method"] == "log_likelihood_last_word":
                acc = evaluate_lambada(model, tokenizer, examples, device, is_moe=is_moe)
            else:
                acc = evaluate_multiple_choice(model, tokenizer, examples, device, is_moe=is_moe)
            results[bname] = round(acc * 100, 2)
            print(f"    {bname}: {acc*100:.1f}% ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    {bname}: ERROR — {e}")
            results[bname] = None

    avg_vals = [v for v in results.values() if v is not None]
    results["average"] = round(sum(avg_vals)/len(avg_vals), 2) if avg_vals else None
    return results


# ============================================================================
# Core experiment: base eval + train specialists + divergence + fusion
# ============================================================================

def run_seed_experiment(seed: int, tokenizer, device: str,
                        all_domain_chunks: dict, revision: str = REVISION_EARLY):
    """
    Full experiment for one seed:
      - Train 3 specialists (or load from checkpoint)
      - Divergence check
      - Weight avg + MoE fusion
      - Save results

    Returns: result dict with improvement_pct
    """
    is_maturity = (revision == REVISION_FULL)
    seed_tag = f"seed{seed}" + ("_maturity" if is_maturity else "")

    divergence_path = RESULTS_DIR / f"step5_divergence_{seed_tag}.json"
    fusion_path     = RESULTS_DIR / f"step6_fusion_{seed_tag}.json"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: seed={seed}, revision={revision}")
    print(f"{'='*70}")

    # ── Resume: skip entire seed if fusion result already exists ─────────────
    if step_completed(fusion_path):
        print(f"\n  [skip] Seed {seed} already complete: {fusion_path}")
        return json.loads(fusion_path.read_text(encoding="utf-8"))

    # ── Train specialists, keep completed ones on CPU (safety valve) ─────────
    # Completed specialists sit in CPU RAM while the next one trains on GPU.
    # Each 6.9B bf16 model is ~14GB; CPU RAM on RunPod is typically 100-200GB.
    trained: dict = {}
    for domain in DOMAINS:
        ckpt = checkpoint_path(domain, seed, revision)
        if ckpt.exists():
            print(f"\n  Loading cached {domain} specialist (seed={seed}) from {ckpt}...")
            model = load_model(revision, device, gradient_checkpointing=False)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            del state
        else:
            print(f"\n  Training {domain} specialist (seed={seed}, revision={revision})...")
            model = load_model(revision, device, gradient_checkpointing=True)
            train_chunks = all_domain_chunks[domain]["train"]
            train_specialist(model, domain, train_chunks, device, seed)
            model.eval()
            save_specialist_checkpoint(model, domain, seed, revision)
        # Move to CPU so the next specialist can train on GPU without OOM
        model.to("cpu")
        torch.cuda.empty_cache()
        print(f"  {domain} specialist moved to CPU (frees ~14GB GPU)")
        trained[domain] = model

    # Move all three back to GPU for eval/fusion
    print(f"\n  Moving specialists back to GPU for evaluation (seed={seed})...")
    for domain in DOMAINS:
        trained[domain].to(device)
    torch.cuda.empty_cache()

    spec_code    = trained["code"]
    spec_science = trained["science"]
    spec_fiction = trained["fiction"]

    # ── Divergence check ─────────────────────────────────────────────────────
    if step_completed(divergence_path):
        print(f"\n  [skip] Divergence check already done: {divergence_path}")
        div_result = json.loads(divergence_path.read_text(encoding="utf-8"))
    else:
        print(f"\n  Divergence check (seed={seed})...")

        # Load base once for comparison
        base_model = load_model(revision, device, gradient_checkpointing=False)
        base_model.eval()

        # Compute loss matrix: {model_name: {domain: loss}}
        held_out_sets = {
            d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
            for d in DOMAINS
        }

        def eval_all_domains(model, is_fused=False) -> dict:
            losses = {}
            for d, ds in held_out_sets.items():
                losses[d] = round(eval_loss(model, ds, device, is_fused=is_fused), 6)
            return losses

        loss_matrix = {
            "base":    eval_all_domains(base_model),
            "code":    eval_all_domains(spec_code),
            "science": eval_all_domains(spec_science),
            "fiction": eval_all_domains(spec_fiction),
        }

        # Verify each specialist beats base on own domain
        divergence_checks = {}
        all_pass = True
        for domain in DOMAINS:
            spec_loss = loss_matrix[domain][domain]
            base_loss = loss_matrix["base"][domain]
            passes = spec_loss < base_loss
            divergence_checks[domain] = {
                "specialist_loss": spec_loss,
                "base_loss": base_loss,
                "improvement": round(base_loss - spec_loss, 6),
                "passes": passes,
            }
            status = "PASS" if passes else "FAIL"
            print(f"  {domain}: spec={spec_loss:.4f} < base={base_loss:.4f} [{status}]")
            if not passes:
                all_pass = False

        del base_model
        torch.cuda.empty_cache()

        if not all_pass:
            print("\nFATAL: Divergence check FAILED for one or more domains.")
            print("Specialists did not improve over base. STOPPING.")
            div_result = {
                "seed": seed, "revision": revision,
                "loss_matrix": loss_matrix,
                "divergence_checks": divergence_checks,
                "all_pass": False,
            }
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(divergence_path, "w", encoding="utf-8") as f:
                json.dump(div_result, f, indent=2)
            git_commit_push(f"[kalavai] 6.9B divergence FAILED seed={seed}")
            raise RuntimeError(f"Divergence check failed for seed={seed}")

        div_result = {
            "seed": seed, "revision": revision,
            "loss_matrix": loss_matrix,
            "divergence_checks": divergence_checks,
            "all_pass": True,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(divergence_path, "w", encoding="utf-8") as f:
            json.dump(div_result, f, indent=2)
        print(f"  Saved: {divergence_path}")

        save_divergence_heatmap(loss_matrix, seed)
        git_commit_push(f"[kalavai] 6.9B divergence check passed seed={seed}")

    if not div_result.get("all_pass", False):
        raise RuntimeError(f"Divergence check failed for seed={seed}")

    # ── Fusion ───────────────────────────────────────────────────────────────
    if step_completed(fusion_path):
        print(f"\n  [skip] Fusion already done: {fusion_path}")
        return json.loads(fusion_path.read_text(encoding="utf-8"))

    print(f"\n  Fusion: weight avg + MoE (seed={seed})...")

    held_out_sets = {
        d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
        for d in DOMAINS
    }
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    # Load base for baseline
    base_model = load_model(revision, device, gradient_checkpointing=False)
    base_model.eval()

    fusion_losses = {}

    # Base
    for d, ds in held_out_sets.items():
        fusion_losses.setdefault("base", {})[d] = round(
            eval_loss(base_model, ds, device), 6
        )
    del base_model
    torch.cuda.empty_cache()

    # Specialists
    for domain, spec in [("code_spec", spec_code), ("science_spec", spec_science),
                          ("fiction_spec", spec_fiction)]:
        for d, ds in held_out_sets.items():
            fusion_losses.setdefault(domain, {})[d] = round(
                eval_loss(spec, ds, device), 6
            )

    # Weight average
    avg_model = weight_average_three(spec_code, spec_science, spec_fiction)
    avg_model.to(device)  # move to GPU after CPU averaging
    for d, ds in held_out_sets.items():
        fusion_losses.setdefault("weight_avg", {})[d] = round(
            eval_loss(avg_model, ds, device), 6
        )
    del avg_model
    torch.cuda.empty_cache()

    # MoE — build and train router
    train_ds_dict = {
        d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
        for d in DOMAINS
    }
    moe = ThreeExpertMoE(spec_code, spec_science, spec_fiction).to(device)
    train_router(moe, train_ds_dict, device)

    for d, ds in held_out_sets.items():
        fusion_losses.setdefault("moe", {})[d] = round(
            eval_loss(moe, ds, device, is_fused=True), 6
        )

    router_dist = eval_router_distribution(moe, held_out_sets, device)

    # Save fused artifacts for seed=42 (for HF Hub publishing)
    if seed == 42 and revision == REVISION_EARLY:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        # Router weights (tiny — ~48KB for 4096×3)
        router_path = CHECKPOINT_DIR / "router_seed42.pt"
        torch.save(moe.router.state_dict(), router_path)
        print(f"  Saved router weights: {router_path}")
        # Weight-averaged model (usable as standalone drop-in, ~14GB)
        avg_path = CHECKPOINT_DIR / "weight_avg_seed42.pt"
        avg_for_save = weight_average_three(spec_code, spec_science, spec_fiction)
        torch.save(avg_for_save.state_dict(), avg_path)
        size_gb = avg_path.stat().st_size / 1e9
        print(f"  Saved weight-avg model: {avg_path} ({size_gb:.1f}GB)")
        del avg_for_save
        torch.cuda.empty_cache()

    del moe, spec_code, spec_science, spec_fiction
    torch.cuda.empty_cache()

    # Compute improvement: (best_individual_mixed - moe_mixed) / best_individual_mixed * 100
    best_individual_mixed = min(
        fusion_losses["code_spec"]["mixed"],
        fusion_losses["science_spec"]["mixed"],
        fusion_losses["fiction_spec"]["mixed"],
    )
    moe_mixed = fusion_losses["moe"]["mixed"]
    improvement_pct = round(
        (best_individual_mixed - moe_mixed) / best_individual_mixed * 100, 4
    )
    print(f"\n  KEY RESULT: improvement = +{improvement_pct:.2f}% "
          f"(MoE {moe_mixed:.4f} vs best_individual {best_individual_mixed:.4f})")

    fusion_result = {
        "seed": seed,
        "revision": revision,
        "eval_heldout": fusion_losses,
        "improvement_pct": improvement_pct,
        "best_individual_mixed": best_individual_mixed,
        "moe_mixed": moe_mixed,
        "router_distribution": router_dist,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(fusion_path, "w", encoding="utf-8") as f:
        json.dump(fusion_result, f, indent=2)
    print(f"  Saved: {fusion_path}")

    save_fusion_comparison(fusion_losses, seed)
    save_router_distribution(router_dist, seed)

    git_commit_push(
        f"[kalavai] 6.9B fusion seed={seed} improvement={improvement_pct:.2f}%"
    )

    return fusion_result


# ============================================================================
# Phase 1+2: base eval + 3 seeds
# ============================================================================

def phase1_2_base_and_seeds(tokenizer, device: str, code_texts, science_texts, fiction_texts):
    """Run base eval and all 3 seeds at step10000."""

    base_eval_path = RESULTS_DIR / "step1_base_eval.json"

    # ── Data prep ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PACKING AND SPLITTING DATA (80/10/10)")
    print("=" * 70)

    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, indist_c, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {
            "train": train_c, "indist": indist_c, "held_out": held_c
        }
        print(f"  {domain}: train={len(train_c)}, indist={len(indist_c)}, held_out={len(held_c)}")
        if len(train_c) < 500:
            print(f"  WARNING: {domain} has <500 train chunks!")

    # ── Base eval ────────────────────────────────────────────────────────────
    if step_completed(base_eval_path):
        print(f"\n[skip] Base eval already done: {base_eval_path}")
    else:
        print("\n" + "=" * 70)
        print("STEP 1: Base Model Eval")
        print("=" * 70)

        base_model = load_model(REVISION_EARLY, device, gradient_checkpointing=False)
        base_model.eval()

        held_out_sets = {
            d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
            for d in DOMAINS
        }
        mixed_held = []
        for d in DOMAINS:
            mixed_held.extend(all_domain_chunks[d]["held_out"])
        held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

        base_losses = {}
        for domain, ds in held_out_sets.items():
            loss = eval_loss(base_model, ds, device)
            base_losses[domain] = round(loss, 6)
            print(f"  Base [{domain:8s}]: {loss:.4f}")

        del base_model
        torch.cuda.empty_cache()

        result = {
            "step": 1,
            "model_id": MODEL_ID,
            "revision": REVISION_EARLY,
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
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(base_eval_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {base_eval_path}")
        git_commit_push("[kalavai] 6.9B base eval done")

    # ── Run 3 seeds ──────────────────────────────────────────────────────────
    seed_results = {}
    for seed in SEEDS:
        seed_result = run_seed_experiment(
            seed, tokenizer, device, all_domain_chunks, REVISION_EARLY
        )
        seed_results[seed] = seed_result
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleared after seed {seed}")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "summary.json"
    if not step_completed(summary_path):
        improvements = [r["improvement_pct"] for r in seed_results.values()
                        if "improvement_pct" in r]
        if improvements:
            mean_imp = round(sum(improvements) / len(improvements), 4)
            std_imp = round(
                float(torch.tensor(improvements, dtype=torch.float32).std().item()), 4
            )
        else:
            mean_imp, std_imp = None, None

        summary = {
            "model_id": MODEL_ID,
            "revision": REVISION_EARLY,
            "seeds": SEEDS,
            "seed_results": {
                str(s): {"improvement_pct": seed_results[s].get("improvement_pct")}
                for s in SEEDS if s in seed_results
            },
            "improvement_mean_pct": mean_imp,
            "improvement_std_pct": std_imp,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {summary_path}")
        print(f"\n*** 6.9B MAIN RESULT: {mean_imp:.2f}% +/- {std_imp:.2f}% ***")
        git_commit_push(
            f"[kalavai] 6.9B summary: improvement={mean_imp:.2f}%+/-{std_imp:.2f}%"
        )

    return all_domain_chunks, seed_results


# ============================================================================
# Phase 3: Maturity check at step143000
# ============================================================================

def phase3_maturity(tokenizer, device: str, all_domain_chunks: dict):
    """Run one seed (42) at step143000 to get the full-train maturity data point."""
    maturity_path = RESULTS_DIR / "maturity_step143000_seed42.json"

    if step_completed(maturity_path):
        print(f"\n[skip] Maturity check already done: {maturity_path}")
        return

    print("\n" + "=" * 70)
    print("PHASE 3: Maturity check at step143000 (seed=42)")
    print("=" * 70)

    result = run_seed_experiment(42, tokenizer, device, all_domain_chunks, REVISION_FULL)

    maturity_result = {
        "model_id": MODEL_ID,
        "revision": REVISION_FULL,
        "training_pct": 100.0,
        "seed": 42,
        "improvement_pct": result.get("improvement_pct"),
        "eval_heldout": result.get("eval_heldout"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(maturity_path, "w", encoding="utf-8") as f:
        json.dump(maturity_result, f, indent=2)
    print(f"Saved: {maturity_path}")
    git_commit_push(f"[kalavai] 6.9B maturity@step143000 improvement={result.get('improvement_pct'):.2f}%")


# ============================================================================
# Phase 4: Downstream benchmarks
# ============================================================================

def phase4_benchmarks(tokenizer, device: str, all_domain_chunks: dict):
    """Run 5 benchmarks on base, MoE (seed=42 at step10000)."""
    bench_path = RESULTS_DIR / "benchmarks_seed42.json"

    if step_completed(bench_path):
        print(f"\n[skip] Benchmarks already done: {bench_path}")
        return

    print("\n" + "=" * 70)
    print("PHASE 4: Downstream Benchmarks (seed=42)")
    print("=" * 70)

    # Load benchmark data
    print("Loading benchmark datasets...")
    benchmark_data = {}
    for bname, bcfg in BENCHMARKS.items():
        try:
            examples = load_benchmark_data(bname, bcfg, N_BENCHMARK_EXAMPLES)
            benchmark_data[bname] = (examples, bcfg)
            print(f"  {bname}: {len(examples)} examples")
        except Exception as e:
            print(f"  {bname}: ERROR — {e}")
            benchmark_data[bname] = ([], bcfg)

    all_results = {}

    # Base model
    print("\nEvaluating base model...")
    base_model = load_model(REVISION_EARLY, device, gradient_checkpointing=False)
    all_results["base"] = run_benchmarks(
        base_model, tokenizer, benchmark_data, device, "base"
    )
    del base_model
    torch.cuda.empty_cache()

    # Re-train seed=42 specialists for benchmarks (no checkpoints on disk).
    # Same safety-valve pattern: train one at a time, park on CPU between.
    print("\nTraining seed=42 specialists for benchmark evaluation...")
    bench_trained: dict = {}
    for domain in DOMAINS:
        print(f"\n  Training {domain} specialist (seed=42, benchmarks)...")
        m = load_model(REVISION_EARLY, device, gradient_checkpointing=True)
        train_specialist(m, domain, all_domain_chunks[domain]["train"], device, 42)
        m.eval()
        m.to("cpu")
        torch.cuda.empty_cache()
        print(f"  {domain} specialist moved to CPU")
        bench_trained[domain] = m
    print("\n  Moving benchmark specialists back to GPU...")
    for domain in DOMAINS:
        bench_trained[domain].to(device)
    torch.cuda.empty_cache()
    spec_code    = bench_trained["code"]
    spec_science = bench_trained["science"]
    spec_fiction = bench_trained["fiction"]

    # MoE (re-train router)
    print("\nBuilding MoE and training router for benchmarks...")
    train_ds_dict = {
        d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
        for d in DOMAINS
    }
    moe = ThreeExpertMoE(spec_code, spec_science, spec_fiction).to(device)
    train_router(moe, train_ds_dict, device)
    all_results["moe"] = run_benchmarks(
        moe, tokenizer, benchmark_data, device, "moe", is_moe=True
    )
    del moe, spec_code, spec_science, spec_fiction
    torch.cuda.empty_cache()

    # Print table
    print("\n" + "=" * 70)
    print("DOWNSTREAM BENCHMARKS — Pythia-6.9B@step10000 (seed=42)")
    print("=" * 70)
    bench_names = list(BENCHMARKS.keys()) + ["average"]
    header = f"{'Model':<18}" + "".join(f"{b:>12}" for b in bench_names)
    print(header)
    print("-" * len(header))
    for mk, dname in [("base", "Base"), ("moe", "MoE fused")]:
        row = f"{dname:<18}"
        for b in bench_names:
            v = all_results.get(mk, {}).get(b)
            row += f"{(str(v)+'%'):>12}" if v is not None else f"{'N/A':>12}"
        print(row)

    # Save
    moe_avg = all_results.get("moe", {}).get("average")
    base_avg = all_results.get("base", {}).get("average")

    bench_result = {
        "model_id": MODEL_ID,
        "revision": REVISION_EARLY,
        "seed": 42,
        "n_examples": N_BENCHMARK_EXAMPLES,
        "results": all_results,
        "moe_avg_accuracy": moe_avg,
        "base_avg_accuracy": base_avg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench_result, f, indent=2)
    print(f"\nSaved: {bench_path}")
    git_commit_push(
        f"[kalavai] 6.9B benchmarks: MoE avg={moe_avg:.1f}% vs base avg={base_avg:.1f}%"
    )


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-6.9B Scale Validation Experiment")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print("=" * 70)

    # Verify disk space
    import shutil as _shutil
    _disk = _shutil.disk_usage("/workspace")
    print(f"\nDisk: {_disk.free/1e9:.1f}GB free / {_disk.total/1e9:.1f}GB total")
    if _disk.free < 20e9:
        print("WARNING: Less than 20GB free. Clear caches before proceeding.")

    # Verify GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {gpu_mem:.1f}GB")
        if gpu_mem < 60:
            print(f"WARNING: GPU has only {gpu_mem:.1f}GB — 6.9B requires ~80GB A100")
    else:
        print("WARNING: No GPU detected — this will not complete on CPU")

    # Verify architecture expectations
    print(f"\nExpected: {NUM_LAYERS} layers, hidden_size={HIDDEN_SIZE}")
    print(f"Config: lr={LR}, max_steps={MAX_STEPS}, freeze={FREEZE_LAYERS}/{NUM_LAYERS}")

    # Load tokenizer once
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION_EARLY)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Pre-download all datasets before touching GPU
    code_texts, science_texts, fiction_texts = predownload_datasets()

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 + 2: base eval + 3 seeds ────────────────────────────────────
    all_domain_chunks, seed_results = phase1_2_base_and_seeds(
        tokenizer, device, code_texts, science_texts, fiction_texts
    )

    # ── Clear step10000 HF cache before loading step143000 (~14GB freed) ────
    print("\nClearing step10000 HF cache before maturity phase...")
    clear_hf_cache()

    # ── Phase 3: Maturity at step143000 ─────────────────────────────────────
    phase3_maturity(tokenizer, device, all_domain_chunks)

    # ── Phase 4: Downstream benchmarks ──────────────────────────────────────
    phase4_benchmarks(tokenizer, device, all_domain_chunks)

    # ── Scale comparison figure ──────────────────────────────────────────────
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        imp_6b = summary.get("improvement_mean_pct", 0.0)
        std_6b = summary.get("improvement_std_pct", 0.0)

        save_scale_comparison(
            results_410m_path=Path("results/pythia/step5_final_summary.json"),
            results_1b_path=Path("results/pythia/pythia_1b/main_result_summary.json"),
            improvement_6b=imp_6b,
            std_6b=std_6b,
        )
        git_commit_push("[kalavai] 6.9B scale comparison figure saved")

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)
    print("\nBefore shutting down the pod:")
    print("  git status   # verify nothing uncommitted")
    print("  git log --oneline -5")
    print("  # THEN shut down the pod to stop billing")


if __name__ == "__main__":
    main()
