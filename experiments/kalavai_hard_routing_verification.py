#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Hard Routing Verification (Concern 7)
=============================================
Verifies that argmax (hard) routing matches softmax (soft) routing performance.
Loads pythia-410m @ step10000, loads 3 specialists (seed=42), trains a soft
MoE router (500 steps), then evaluates BOTH soft and hard routing on the
same held-out mixed data using the SAME trained router weights.

Result: hard_routing_verification.json + fig_hard_routing_verification.png
"""

import copy
import json
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
# HardRoutingMoE — argmax inference, same router weights as ThreeExpertMoE
# ============================================================================

class HardRoutingMoE(nn.Module):
    """
    Identical to ThreeExpertMoE at training time, but at inference uses argmax
    routing: each batch element is routed 100% to its single highest-gate expert.
    Shares the same trained router weights via parameter reference.
    """
    def __init__(self, spec_a, spec_b, spec_c, router: nn.Sequential):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        # Share the already-trained router — no new parameters
        self.router = router

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
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, 3)

        # Hard routing: pick single best expert per batch element
        best = gates.argmax(dim=-1)  # (B,)
        fused = torch.where(best.unsqueeze(-1).unsqueeze(-1) == 0, logits_a,
                torch.where(best.unsqueeze(-1).unsqueeze(-1) == 1, logits_b, logits_c))

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
# Figure
# ============================================================================

def save_hard_routing_figure(base_loss: float, soft_loss: float, hard_loss: float,
                              soft_improvement: float, hard_improvement: float):
    """fig_hard_routing_verification.png: bar chart comparing soft vs hard routing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Left: absolute losses
        ax = axes[0]
        labels = ["Base", "Soft MoE\n(softmax)", "Hard MoE\n(argmax)"]
        values = [base_loss, soft_loss, hard_loss]
        colors = ["#95a5a6", "#9b59b6", "#e67e22"]
        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
        ax.set_ylabel("Cross-Entropy Loss (lower is better)")
        ax.set_title("Absolute Loss: Soft vs Hard Routing")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        # Right: % improvement over base
        ax = axes[1]
        labels2 = ["Soft MoE\n(softmax)", "Hard MoE\n(argmax)"]
        imps = [soft_improvement, hard_improvement]
        colors2 = ["#9b59b6", "#e67e22"]
        bars2 = ax.bar(labels2, imps, color=colors2, alpha=0.85, width=0.4)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("% Improvement over Base (higher is better)")
        ax.set_title("Routing Mechanism Comparison")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars2, imps):
            ypos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.3
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.suptitle("Hard vs Soft Routing Verification (Pythia-410M, seed=42)", fontsize=12)
        fig.tight_layout()

        path = FIGURES_DIR / "fig_hard_routing_verification.png"
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
    print("KALAVAI: Hard Routing Verification (Concern 7)")
    print("=" * 70)
    print(f"Model:  {MODEL_ID} @ revision={REVISION}")
    print(f"Seed:   {SEED}")
    print(f"Domains: {DOMAINS}")

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

    # Mixed held-out set
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    mixed_ds = make_dataset_from_chunks(mixed_held)
    print(f"  mixed held-out: {len(mixed_held)} chunks")

    # Train datasets for router
    train_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                  for d in DOMAINS}

    # ---- Load base model ----
    print(f"\nLoading base model: {MODEL_ID} (revision={REVISION})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    base_model.eval()

    base_loss = eval_loss(base_model, mixed_ds, device)
    print(f"  Base model mixed held-out loss: {base_loss:.4f}")

    del base_model
    torch.cuda.empty_cache()

    # ---- Load or train specialists ----
    print(f"\nLoading specialists (seed={SEED})...")
    specialists = {}
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

        specialists[domain] = spec

    # ---- Build soft MoE and train router ----
    print("\nBuilding ThreeExpertMoE (soft routing)...")
    soft_moe = ThreeExpertMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    train_router(soft_moe, train_sets, device)
    soft_moe.eval()

    # ---- Eval soft MoE ----
    print("\nEvaluating soft MoE on mixed held-out...")
    soft_loss = eval_loss(soft_moe, mixed_ds, device, batch_size=2, is_fused=True)
    print(f"  Soft MoE mixed held-out loss: {soft_loss:.4f}")

    # ---- Build hard MoE using SAME router weights ----
    print("\nBuilding HardRoutingMoE (argmax, same router weights)...")
    hard_moe = HardRoutingMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        router=soft_moe.router,   # Share the SAME trained router
    ).to(device)
    hard_moe.eval()

    # ---- Eval hard MoE ----
    print("Evaluating hard MoE on mixed held-out...")
    hard_loss = eval_loss(hard_moe, mixed_ds, device, batch_size=2, is_fused=True)
    print(f"  Hard MoE mixed held-out loss: {hard_loss:.4f}")

    # ---- Compute improvements ----
    soft_improvement_pct = (base_loss - soft_loss) / base_loss * 100
    hard_improvement_pct = (base_loss - hard_loss) / base_loss * 100
    difference_pp = hard_improvement_pct - soft_improvement_pct

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Base loss:              {base_loss:.4f}")
    print(f"  Soft MoE loss:          {soft_loss:.4f}  ({soft_improvement_pct:+.2f}% vs base)")
    print(f"  Hard MoE loss:          {hard_loss:.4f}  ({hard_improvement_pct:+.2f}% vs base)")
    print(f"  Difference (hard-soft): {difference_pp:+.2f} pp")

    if abs(difference_pp) < 1.0:
        print("\n  VERDICT: Hard and soft routing perform within 1pp — hard routing confirmed.")
    elif difference_pp > 0:
        print(f"\n  VERDICT: Hard routing outperforms soft by {difference_pp:.2f}pp.")
    else:
        print(f"\n  VERDICT: Soft routing outperforms hard by {abs(difference_pp):.2f}pp.")

    # ---- Save results ----
    result = {
        "base_loss": round(base_loss, 6),
        "soft_moe_loss": round(soft_loss, 6),
        "hard_routing_loss": round(hard_loss, 6),
        "soft_improvement_pct": round(soft_improvement_pct, 4),
        "hard_improvement_pct": round(hard_improvement_pct, 4),
        "difference_pp": round(difference_pp, 4),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "seed": SEED,
        "eval_batches": EVAL_BATCHES,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "hard_routing_verification.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results: {out_path}")

    # ---- Save figure ----
    save_hard_routing_figure(base_loss, soft_loss, hard_loss,
                              soft_improvement_pct, hard_improvement_pct)

    # ---- Git commit + push ----
    commit_msg = (
        f"[kalavai] hard routing: hard={hard_improvement_pct:.2f}% "
        f"vs soft={soft_improvement_pct:.2f}%"
    )
    git_commit_push(commit_msg)

    print("\n" + "=" * 70)
    print("Hard routing verification complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
