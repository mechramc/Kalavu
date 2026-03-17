#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Specialist Error Mutual Information (Concern 2)
=======================================================
Measures whether specialist errors are complementary (uncorrelated) or merely
diverse (correlated). Loads pythia-410m @ step10000, loads 3 specialists
(seed=42), computes per-token log-likelihoods on mixed held-out data, then
calculates error correlation, complementarity score, and mutual information
between specialist error indicators.

Result: mutual_information.json + fig_mutual_information.png
"""

import json
import subprocess
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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
SEED = 42
DOMAINS = ["code", "science", "fiction"]

# Max chunks to analyze for per-token MI computation
MI_MAX_CHUNKS = 200

RESULTS_DIR = Path("results/pythia")
CHECKPOINT_DIR = Path("checkpoints/pythia")
FIGURES_DIR = Path("figures/pythia")

HIDDEN_SIZE = 1024   # pythia-410m
NUM_LAYERS = 24      # pythia-410m


# ============================================================================
# Utilities
# ============================================================================

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
# PackedChunkDataset — copied verbatim from kalavai_pythia_experiment.py
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
# Data loading — copied verbatim from kalavai_pythia_experiment.py
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
# Pythia-specific freezing — copied verbatim from kalavai_pythia_experiment.py
# ============================================================================

def freeze_bottom_layers(model, n: int):
    """Freeze embedding + first n transformer blocks (GPT-NeoX architecture)."""
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


# ============================================================================
# ThreeExpertMoE — copied verbatim from kalavai_pythia_experiment.py
# ============================================================================

class ThreeExpertMoE(nn.Module):
    """
    Sequence-level MoE over three specialist models.
    Router: mean of last hidden states from all experts -> small MLP -> 3 gates.
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
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 3, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach()
        last_h = out.hidden_states[-1].detach()
        h_pooled = last_h.mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)

        h_avg = (h_a + h_b + h_c) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)

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
# Training helpers — copied verbatim from kalavai_pythia_experiment.py
# ============================================================================

def train_specialist(model, domain: str, train_chunks: list, tokenizer,
                     seed: int, device: str, log_every: int = 50) -> list[float]:
    """Train a specialist. Returns loss history for plotting."""
    set_seed(seed)
    freeze_bottom_layers(model, FREEZE_LAYERS)
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
    from torch.optim.lr_scheduler import CosineAnnealingLR
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
# Per-token loss computation
# ============================================================================

@torch.no_grad()
def compute_per_token_losses(model, chunks: list, device: str,
                              max_chunks: int = MI_MAX_CHUNKS) -> torch.Tensor:
    """
    Compute per-token cross-entropy loss for up to max_chunks chunks.
    Returns a flat 1D tensor of per-token losses, shape (N_tokens,).
    Each token's loss = -log p(token | context).
    """
    model.eval()
    all_losses = []
    n = min(len(chunks), max_chunks)
    print(f"  Computing per-token losses on {n} chunks...")

    for i, chunk in enumerate(chunks[:n]):
        if i % 50 == 0:
            print(f"    chunk {i}/{n}...")
        input_ids = chunk.unsqueeze(0).to(device)   # (1, T)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids)
        logits = out.logits.float()   # (1, T, V)

        # Per-token loss: predict each token from its left context
        # token losses shape: (T-1,)
        token_losses = F.cross_entropy(
            logits[0, :-1],      # (T-1, V)
            input_ids[0, 1:],    # (T-1,)
            reduction="none",
        )
        all_losses.append(token_losses.cpu())

    return torch.cat(all_losses, dim=0)   # (N_tokens,)


# ============================================================================
# Mutual Information computation
# ============================================================================

def compute_mutual_information(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """
    Compute mutual information between two binary error indicator vectors.
    MI(X;Y) = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )
    where x,y in {0,1} (0=correct, 1=error).
    """
    n = len(e1)
    assert len(e2) == n

    e1 = e1.bool()
    e2 = e2.bool()

    # Joint distribution
    p00 = ((~e1) & (~e2)).float().mean().item()
    p01 = ((~e1) & e2).float().mean().item()
    p10 = (e1 & (~e2)).float().mean().item()
    p11 = (e1 & e2).float().mean().item()

    # Marginals
    p1_0 = (~e1).float().mean().item()
    p1_1 = e1.float().mean().item()
    p2_0 = (~e2).float().mean().item()
    p2_1 = e2.float().mean().item()

    eps = 1e-12
    mi = 0.0
    for pxy, px, py in [
        (p00, p1_0, p2_0),
        (p01, p1_0, p2_1),
        (p10, p1_1, p2_0),
        (p11, p1_1, p2_1),
    ]:
        if pxy > eps and px > eps and py > eps:
            mi += pxy * (torch.log(torch.tensor(pxy / (px * py) + eps))).item()

    return float(mi)


def compute_error_correlation(e_i: torch.Tensor, e_j: torch.Tensor) -> float:
    """
    Given two binary error vectors, compute the conditional error overlap:
    P(spec_j wrong | spec_i wrong) = |both wrong| / |spec_i wrong|
    """
    both_wrong = (e_i & e_j).float().sum().item()
    i_wrong = e_i.float().sum().item()
    if i_wrong == 0:
        return 0.0
    return both_wrong / i_wrong


# ============================================================================
# Figure
# ============================================================================

def save_mi_figure(error_corr_matrix: list, complementarity: float,
                   mi: float, frac_complementary: float):
    """fig_mutual_information.png: correlation heatmap + summary bar."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: 3x3 error correlation heatmap
        ax = axes[0]
        data = np.array(error_corr_matrix)
        im = ax.imshow(data, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
        labels = ["Code\nspec.", "Science\nspec.", "Fiction\nspec."]
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Specialist j")
        ax.set_ylabel("Specialist i (wrong)")
        ax.set_title("Error Correlation Matrix\nP(j wrong | i wrong)")
        fig.colorbar(im, ax=ax, label="Conditional error rate")

        for i in range(3):
            for j in range(3):
                color = "white" if data[i, j] > 0.6 else "black"
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

        # Right: summary metrics bar chart
        ax = axes[1]
        metric_names = ["Complementarity\nscore", "Frac. compl.\ntokens"]
        metric_vals = [complementarity, frac_complementary]
        colors = ["#27ae60" if v > 0.5 else "#e74c3c" for v in metric_vals]
        bars = ax.bar(metric_names, metric_vals, color=colors, alpha=0.85, width=0.4)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect complementarity")
        ax.set_ylabel("Score (higher = more complementary)")
        ax.set_title(f"Complementarity Metrics\nMI={mi:.4f}")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, metric_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        fig.suptitle("Specialist Error Mutual Information (Pythia-410M, seed=42)", fontsize=12)
        fig.tight_layout()

        path = FIGURES_DIR / "fig_mutual_information.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Specialist Error Mutual Information (Concern 2)")
    print("=" * 70)
    print(f"Model:  {MODEL_ID} @ revision={REVISION}")
    print(f"Seed:   {SEED}")
    print(f"Domains: {DOMAINS}")
    print(f"MI max chunks: {MI_MAX_CHUNKS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be very slow.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_ID} (revision={REVISION})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading data (3 domains)...")
    code_texts    = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting chunks (80/10/10)...")
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
        print(f"  {domain}: total={len(ds_full)}, train={len(train_c)}, "
              f"indist={len(indist_c)}, held_out={len(held_c)}")

    # Mixed held-out chunks (capped at MI_MAX_CHUNKS for per-token analysis)
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    print(f"  mixed held-out total: {len(mixed_held)} chunks "
          f"(will analyze up to {MI_MAX_CHUNKS})")

    # ---- Load base model ----
    print(f"\nLoading base model: {MODEL_ID} (revision={REVISION})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()

    print("Computing per-token losses for base model...")
    base_token_losses = compute_per_token_losses(base_model, mixed_held, device)
    print(f"  Base model: {len(base_token_losses)} tokens analyzed, "
          f"mean loss={base_token_losses.mean():.4f}")

    del base_model
    torch.cuda.empty_cache()

    # ---- Load or train specialists ----
    print(f"\nLoading specialists (seed={SEED})...")
    specialist_token_losses = {}

    for domain in DOMAINS:
        ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed{SEED}.pt"
        spec = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)

        if ckpt_path.exists():
            print(f"  Loading {domain} specialist from {ckpt_path}...")
            spec.load_state_dict(torch.load(ckpt_path, map_location=device))
            spec.eval()
        else:
            print(f"  Checkpoint not found at {ckpt_path} — training fresh...")
            train_specialist(spec, domain, all_domain_chunks[domain]["train"],
                             tokenizer, SEED, device)
            spec.eval()
            torch.save(spec.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        print(f"  Computing per-token losses for {domain} specialist...")
        tok_losses = compute_per_token_losses(spec, mixed_held, device)
        specialist_token_losses[domain] = tok_losses
        print(f"    {domain}: mean loss={tok_losses.mean():.4f}")

        del spec
        torch.cuda.empty_cache()

    # ---- Compute error indicators ----
    # A token is "wrong" for a specialist if specialist_loss >= base_loss for that token
    # (specialist doesn't improve over base on this token)
    print("\nComputing error indicators...")
    n_tokens = len(base_token_losses)
    assert all(len(v) == n_tokens for v in specialist_token_losses.values()), \
        "Token count mismatch between models — all models must be run on identical chunks."

    error_indicators = {}
    for domain in DOMAINS:
        spec_losses = specialist_token_losses[domain]
        # Error = specialist is NOT better than base (loss >= base_loss)
        errors = (spec_losses >= base_token_losses)
        error_indicators[domain] = errors
        frac_wrong = errors.float().mean().item()
        print(f"  {domain}: {frac_wrong*100:.1f}% of tokens where specialist >= base loss")

    # ---- Error correlation matrix ----
    # C[i,j] = P(spec_j wrong | spec_i wrong)
    print("\nComputing error correlation matrix...")
    domain_list = DOMAINS
    corr_matrix = []
    for i, di in enumerate(domain_list):
        row = []
        for j, dj in enumerate(domain_list):
            corr = compute_error_correlation(error_indicators[di], error_indicators[dj])
            row.append(round(corr, 6))
        corr_matrix.append(row)
        print(f"  {di:10s}: " + "  ".join(
            f"P({dj[:3]} wrong|{di[:3]} wrong)={corr_matrix[i][j]:.3f}"
            for j, dj in enumerate(domain_list)
        ))

    # Off-diagonal mean (average conditional overlap between different specialists)
    off_diagonal = [
        corr_matrix[i][j]
        for i in range(3) for j in range(3)
        if i != j
    ]
    mean_off_diag = sum(off_diagonal) / len(off_diagonal)
    complementarity_score = 1.0 - mean_off_diag

    print(f"\n  Mean off-diagonal correlation: {mean_off_diag:.4f}")
    print(f"  Complementarity score (1 - mean_off_diag): {complementarity_score:.4f}")

    # ---- Mutual information ----
    print("\nComputing mutual information between error indicators...")
    mi_values = []
    for i, di in enumerate(domain_list):
        for j, dj in enumerate(domain_list):
            if i < j:
                mi_val = compute_mutual_information(error_indicators[di], error_indicators[dj])
                mi_values.append(mi_val)
                print(f"  MI({di[:3]}, {dj[:3]}) = {mi_val:.6f}")

    mean_mi = sum(mi_values) / len(mi_values) if mi_values else 0.0
    print(f"  Mean pairwise MI: {mean_mi:.6f}")

    # ---- Fraction of complementary tokens ----
    # A token is "complementary" if NOT all specialists are wrong simultaneously
    all_wrong = error_indicators[DOMAINS[0]]
    for d in DOMAINS[1:]:
        all_wrong = all_wrong & error_indicators[d]

    frac_complementary = 1.0 - all_wrong.float().mean().item()
    print(f"\n  Fraction of complementary tokens (not all wrong): {frac_complementary:.4f}")
    print(f"  Tokens where all 3 specialists fail: {all_wrong.float().mean().item()*100:.1f}%")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Complementarity score:       {complementarity_score:.4f}")
    print(f"  (1.0 = perfectly uncorrelated, 0.0 = perfectly correlated)")
    print(f"  Mean pairwise MI:            {mean_mi:.6f}")
    print(f"  Frac. complementary tokens:  {frac_complementary:.4f}")
    print(f"  Total tokens analyzed:       {n_tokens:,}")

    if complementarity_score > 0.6:
        print("\n  VERDICT: Errors are substantially complementary "
              f"(score={complementarity_score:.3f}) — specialists cover different failure modes.")
    elif complementarity_score > 0.3:
        print(f"\n  VERDICT: Modest complementarity (score={complementarity_score:.3f}) — "
              "some independent coverage, some overlap.")
    else:
        print(f"\n  VERDICT: Errors are largely correlated (score={complementarity_score:.3f}) "
              "— specialists tend to fail on the same tokens.")

    # ---- Save results ----
    result = {
        "error_correlation_matrix": corr_matrix,
        "complementarity_score": round(complementarity_score, 6),
        "mutual_information": round(mean_mi, 6),
        "fraction_complementary_tokens": round(frac_complementary, 6),
        "n_tokens_analyzed": n_tokens,
        "mean_off_diagonal_correlation": round(mean_off_diag, 6),
        "pairwise_mi": {
            f"{di[:3]}_{dj[:3]}": round(mi_values[k], 6)
            for k, (i, j) in enumerate(
                (i, j) for i in range(3) for j in range(3) if i < j
            )
            for di, dj in [(domain_list[i], domain_list[j])]
        },
        "model_id": MODEL_ID,
        "revision": REVISION,
        "seed": SEED,
        "mi_max_chunks": MI_MAX_CHUNKS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "mutual_information.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results: {out_path}")

    # ---- Save figure ----
    save_mi_figure(corr_matrix, complementarity_score, mean_mi, frac_complementary)

    # ---- Git commit + push ----
    commit_msg = f"[kalavai] mutual information: complementarity={complementarity_score:.3f}"
    git_commit_push(commit_msg)

    print("\n" + "=" * 70)
    print("Mutual information analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
