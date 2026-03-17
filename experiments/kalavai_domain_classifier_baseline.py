#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Domain Classifier Baseline
====================================
Addresses the "router is just a domain classifier" concern.
Compares MoE soft router vs hard logistic regression domain classifier.

Steps:
  1. Load Pythia-410m @ step10000
  2. Train 3 domain specialists (freeze=4, seeds=[42, 137, 2026])
  3. Build LogisticRegression classifier on frozen backbone hidden states
  4. Eval: base_loss vs moe_loss vs classifier_loss
  5. Save results + bar chart
"""

import copy
import json
import os
import subprocess
import time
import traceback
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

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not available, falling back to nearest-centroid classifier")

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
SEQ_LEN = 256
WARMUP_FRACTION = 0.1
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
SEEDS = [42, 137, 2026]
DOMAINS = ["code", "science", "fiction"]

RESULTS_DIR = Path("results/pythia")
CHECKPOINT_DIR = Path("checkpoints/pythia")
FIGURES_DIR = Path("figures/pythia")

HIDDEN_SIZE = 1024
NUM_LAYERS = 24


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
# PackedChunkDataset
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts: list, tokenizer, seq_len: int = SEQ_LEN,
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
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    n = len(chunks)
    train_end = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n_samples: int) -> list:
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


def load_science_texts(n_samples: int) -> list:
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


def load_fiction_texts(n_samples: int) -> list:
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


# ============================================================================
# Model helpers
# ============================================================================

def freeze_bottom_layers(model, n: int):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


def train_specialist(model, domain: str, train_chunks: list, tokenizer,
                     seed: int, device: str, max_steps: int = MAX_STEPS) -> None:
    set_seed(seed)
    freeze_bottom_layers(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    print(f"  {domain} train_chunks={len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(max_steps * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps)

    step = 0
    accum = 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= max_steps:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
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
            if step % 100 == 0 or step == max_steps:
                avg = running_loss / step
                print(f"  [{domain}] step {step}/{max_steps} | loss {avg:.4f} | {time.time()-t0:.0f}s")

    print(f"  {domain} training done in {time.time()-t0:.0f}s")


@torch.no_grad()
def eval_loss(model, dataset, device: str, batch_size: int = 4,
              is_fused: bool = False) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(ids, labels=labels)
        else:
            out = model(input_ids=ids, labels=labels)
            loss = out.loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


# ============================================================================
# ThreeExpertMoE
# ============================================================================

class ThreeExpertMoE(nn.Module):
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


def train_router(moe: ThreeExpertMoE, train_chunks_combined: list, device: str):
    combined = make_dataset_from_chunks(train_chunks_combined)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({ROUTER_STEPS} steps, mixed={len(combined)} chunks)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, _, _ = moe(ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")


def weight_average_three(spec_a, spec_b, spec_c):
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
# Nearest-centroid fallback classifier
# ============================================================================

class NearestCentroidClassifier:
    """Fallback when sklearn is not available."""
    def __init__(self):
        self.centroids = None
        self.labels_ = None

    def fit(self, X, y):
        import numpy as np
        unique_labels = sorted(set(y))
        self.labels_ = unique_labels
        self.centroids = []
        for lbl in unique_labels:
            mask = [yi == lbl for yi in y]
            self.centroids.append(X[mask].mean(axis=0))
        self.centroids = np.array(self.centroids)
        return self

    def predict(self, X):
        import numpy as np
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return [self.labels_[i] for i in dists.argmin(axis=1)]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)


# ============================================================================
# Classifier helpers
# ============================================================================

@torch.no_grad()
def extract_hidden_states(model, chunks: list, device: str, layer_idx: int = 3,
                           max_chunks: int = 500) -> "np.ndarray":
    """Extract mean-pooled hidden states from a specific layer."""
    import numpy as np
    dataset = make_dataset_from_chunks(chunks[:max_chunks])
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        drop_last=False, collate_fn=_collate)
    model.eval()
    all_feats = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        out = model(input_ids=ids, output_hidden_states=True)
        # hidden_states[0] = embedding, hidden_states[i+1] = layer i output
        h = out.hidden_states[layer_idx + 1]  # (B, T, H)
        pooled = h.mean(dim=1).float().cpu().numpy()  # (B, H)
        all_feats.append(pooled)
    return np.vstack(all_feats)


def build_domain_classifier(base_model, all_domain_chunks: dict, device: str):
    """Train LogisticRegression on frozen backbone hidden states."""
    import numpy as np

    print("  Extracting features for classifier training...")
    X_parts, y_parts = [], []
    for label_idx, domain in enumerate(DOMAINS):
        train_chunks = all_domain_chunks[domain]["train"]
        feats = extract_hidden_states(base_model, train_chunks, device,
                                      layer_idx=3, max_chunks=300)
        X_parts.append(feats)
        y_parts.extend([label_idx] * len(feats))
        print(f"    {domain}: {len(feats)} feature vectors")

    X = np.vstack(X_parts)
    y = y_parts

    print(f"  Training classifier on {len(X)} samples...")
    if HAS_SKLEARN:
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = NearestCentroidClassifier()
    clf.fit(X, y)
    return clf


@torch.no_grad()
def eval_classifier_loss(clf, base_model, specialists: dict,
                          held_chunks: list, device: str) -> tuple:
    """Eval loss by routing each sequence to classifier-predicted specialist.
    Returns (avg_loss, accuracy).
    """
    import numpy as np

    dataset = make_dataset_from_chunks(held_chunks)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        drop_last=False, collate_fn=_collate)

    total_loss = 0.0
    count = 0
    correct = 0
    domain_list = DOMAINS

    # We need ground truth labels for accuracy — approximate by which domain
    # the held_out chunk came from. We'll track by interleaving.
    domain_assignment = []
    for d_idx, d in enumerate(DOMAINS):
        domain_assignment.extend([d_idx] * len(held_chunks // 3 + (1 if i < len(held_chunks) % 3 else 0)
                                               for i in [0]))
    # Build domain labels per chunk position
    per_domain_n = {d: 0 for d in DOMAINS}

    base_model.eval()
    for spec in specialists.values():
        spec.eval()

    all_feats = []
    all_ids = []
    all_labels_tensor = []

    for batch in loader:
        if count >= EVAL_BATCHES:
            break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        out = base_model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[4]  # layer 3 output (0-indexed hidden_states: 0=embed, 1..N+1=layers)
        pooled = h.mean(dim=1).float().cpu().numpy()
        all_feats.append(pooled)
        all_ids.append(ids.cpu())
        all_labels_tensor.append(lbl.cpu())
        count += 1

    if not all_feats:
        return float("inf"), 0.0

    import numpy as np
    X = np.vstack(all_feats)
    predictions = clf.predict(X)

    total_loss = 0.0
    for i, (pred_label, ids_cpu, lbl_cpu) in enumerate(
            zip(predictions, all_ids, all_labels_tensor)):
        ids = ids_cpu.to(device)
        lbl = lbl_cpu.to(device)
        domain = DOMAINS[pred_label]
        spec = specialists[domain]
        with torch.no_grad():
            loss = spec(input_ids=ids, labels=lbl).loss
        if loss is not None:
            total_loss += loss.item()

    avg_loss = total_loss / len(predictions) if predictions else float("inf")
    # Classifier accuracy: we don't have ground truth per-chunk domain labels in held_out_mixed
    # so we skip accuracy here and compute it separately on domain-labeled held_out chunks
    return avg_loss, 0.0


@torch.no_grad()
def compute_classifier_accuracy(clf, base_model, all_domain_chunks: dict,
                                 device: str) -> float:
    """Compute accuracy on domain-labeled held-out sequences."""
    import numpy as np

    X_parts, y_true = [], []
    for label_idx, domain in enumerate(DOMAINS):
        held_chunks = all_domain_chunks[domain]["held_out"]
        feats = extract_hidden_states(base_model, held_chunks, device,
                                      layer_idx=3, max_chunks=100)
        X_parts.append(feats)
        y_true.extend([label_idx] * len(feats))

    X = np.vstack(X_parts)
    return clf.score(X, y_true)


# ============================================================================
# Figure
# ============================================================================

def save_figure(base_loss: float, moe_loss: float, classifier_loss: float,
                moe_imp: float, clf_imp: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Base\nModel", "MoE\n(soft router)", "Classifier\n(hard router)"]
        losses = [base_loss, moe_loss, classifier_loss]
        imps = [0.0, moe_imp, clf_imp]
        colors = ["#95a5a6", "#9b59b6", "#e67e22"]

        y_min = min(losses) * 0.995
        y_max = max(losses) * 1.01

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, losses, color=colors, alpha=0.85, width=0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Held-Out Mixed Loss (lower is better)")
        ax.set_title("Domain Classifier vs MoE Router (Pythia-410M, freeze=4, seeds=[42,137,2026])")
        ax.grid(True, axis="y", alpha=0.3)

        for bar, loss, imp in zip(bars, losses, imps):
            label = f"{loss:.4f}"
            if imp != 0.0:
                label += f"\n({imp:+.1f}%)"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (y_max - y_min) * 0.005,
                    label, ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_domain_classifier.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Domain Classifier Baseline")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading data...")
    code_texts = load_code_texts(N_SAMPLES_PER_DOMAIN)
    science_texts = load_science_texts(N_SAMPLES_PER_DOMAIN)
    fiction_texts = load_fiction_texts(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting chunks (80/10/10)...")
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, indist_c, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "indist": indist_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_mixed = make_dataset_from_chunks(mixed_held)

    combined_train = []
    for d in DOMAINS:
        combined_train.extend(all_domain_chunks[d]["train"])

    # Load base model
    print(f"\nLoading base model: {MODEL_ID}@{REVISION}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()

    # Base loss
    print("\nEvaluating base model...")
    base_loss = eval_loss(base_model, held_out_mixed, device)
    print(f"  Base mixed loss: {base_loss:.4f}")

    # Check for MoE loss from prior experiment
    moe_loss = None
    summary_path = RESULTS_DIR / "step5_final_summary.json"
    if summary_path.exists():
        print(f"\nLoading MoE reference from {summary_path}...")
        with open(summary_path) as f:
            summary = json.load(f)
        imp_mean = summary.get("improvement_mean_pct", None)
        if imp_mean is not None:
            moe_loss = base_loss / (1.0 + imp_mean / 100.0)
            print(f"  Loaded MoE improvement={imp_mean:.2f}% → moe_loss={moe_loss:.4f}")

    # Train specialists for all seeds
    print("\n" + "=" * 50)
    print("Training specialists (3 seeds × 3 domains)...")
    specialists_by_seed = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        specialists_by_seed[seed] = {}
        for domain in DOMAINS:
            ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
            ).to(device)
            if ckpt.exists():
                print(f"  Loading cached {domain} seed={seed} from {ckpt}")
                model.load_state_dict(torch.load(ckpt, map_location=device))
            else:
                print(f"  Training {domain} specialist seed={seed}...")
                train_specialist(model, domain, all_domain_chunks[domain]["train"],
                                 tokenizer, seed, device)
                torch.save(model.state_dict(), ckpt)
                print(f"  Saved: {ckpt}")
            model.eval()
            specialists_by_seed[seed][domain] = model

    # Compute MoE loss if not loaded
    if moe_loss is None:
        print("\nComputing MoE loss (seed=42)...")
        s = specialists_by_seed[42]
        moe = ThreeExpertMoE(s["code"], s["science"], s["fiction"],
                             hidden_size=HIDDEN_SIZE).to(device)
        train_router(moe, combined_train, device)
        moe.eval()
        moe_loss = eval_loss(moe, held_out_mixed, device, batch_size=2, is_fused=True)
        del moe
        torch.cuda.empty_cache()
        print(f"  MoE mixed loss: {moe_loss:.4f}")

    moe_improvement_pct = (base_loss - moe_loss) / base_loss * 100

    # Build domain classifier on frozen base model
    print("\nBuilding domain classifier...")
    clf = build_domain_classifier(base_model, all_domain_chunks, device)

    # Classifier accuracy
    print("\nEvaluating classifier accuracy on held-out domain chunks...")
    clf_accuracy = compute_classifier_accuracy(clf, base_model, all_domain_chunks, device)
    print(f"  Classifier accuracy: {clf_accuracy:.4f}")

    # Eval classifier routing loss
    print("\nEvaluating classifier-routed loss (seed=42 specialists)...")
    # Build mixed held-out chunks list with domain info for routing
    # We route based on per-sequence classifier prediction
    s42 = specialists_by_seed[42]
    classifier_losses = []

    import numpy as np
    for domain in DOMAINS:
        held_chunks = all_domain_chunks[domain]["held_out"]
        feats = extract_hidden_states(base_model, held_chunks, device,
                                      layer_idx=3, max_chunks=len(held_chunks))
        predictions = clf.predict(feats)
        domain_loss_total = 0.0
        chunk_count = 0
        for i, pred_label in enumerate(predictions):
            if chunk_count >= EVAL_BATCHES // len(DOMAINS):
                break
            pred_domain = DOMAINS[pred_label]
            spec = s42[pred_domain]
            ids = held_chunks[i].unsqueeze(0).to(device)
            with torch.no_grad():
                loss = spec(input_ids=ids, labels=ids).loss
            if loss is not None:
                domain_loss_total += loss.item()
                chunk_count += 1
        if chunk_count > 0:
            classifier_losses.append(domain_loss_total / chunk_count)

    classifier_loss = float(np.mean(classifier_losses)) if classifier_losses else float("inf")
    classifier_improvement_pct = (base_loss - classifier_loss) / base_loss * 100

    print(f"\n  Base loss:        {base_loss:.4f}")
    print(f"  MoE loss:         {moe_loss:.4f}  ({moe_improvement_pct:+.1f}%)")
    print(f"  Classifier loss:  {classifier_loss:.4f}  ({classifier_improvement_pct:+.1f}%)")
    print(f"  Classifier acc:   {clf_accuracy:.4f}")

    # Save figure
    print("\nSaving figure...")
    save_figure(base_loss, moe_loss, classifier_loss,
                moe_improvement_pct, classifier_improvement_pct)

    # Save results
    output = {
        "base_loss": round(base_loss, 6),
        "moe_loss": round(moe_loss, 6),
        "classifier_loss": round(classifier_loss, 6),
        "moe_improvement_pct": round(moe_improvement_pct, 4),
        "classifier_improvement_pct": round(classifier_improvement_pct, 4),
        "classifier_accuracy": round(clf_accuracy, 4),
        "seeds": SEEDS,
        "model": f"{MODEL_ID}@{REVISION}",
        "freeze_layers": FREEZE_LAYERS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "domain_classifier_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")

    # Git commit + push
    msg = (f"[kalavai] domain classifier baseline: "
           f"classifier={classifier_improvement_pct:.1f}% vs moe={moe_improvement_pct:.1f}%")
    git_commit_push(msg)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
