#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Training Duration Crossover
======================================
Tests whether freeze=4 advantage over freeze=0 emerges at longer training horizons.

For each combination of (steps, freeze) where:
  - steps in [500, 1000, 2000, 5000, 10000, 20000]
  - freeze in [0, 4]
Trains 3 specialists (code/science/fiction), fuses via MoE, evals held-out loss.

Saves intermediate results after each (steps, freeze) pair to enable resumption.
Identifies crossover_steps: where freeze=0 improvement falls below freeze=4.

Total estimated time: ~12.5 hours (dominated by 20000-step runs).
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

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
LR = 2e-5
WEIGHT_DECAY = 0.1
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
SEED = 42
DOMAINS = ["code", "science", "fiction"]

STEPS_SWEEP = [500, 1000, 2000, 5000, 10000, 20000]
FREEZE_SWEEP = [0, 4]

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
                     seed: int, device: str, max_steps: int) -> None:
    set_seed(seed)
    freeze_bottom_layers(model, 0 if max_steps == 0 else model._freeze_n)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(max_steps * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))

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
            if step % max(max_steps // 10, 50) == 0 or step == max_steps:
                avg = running_loss / step
                print(f"    [{domain}] step {step}/{max_steps} | loss {avg:.4f} | {time.time()-t0:.0f}s")

    print(f"  {domain} done ({max_steps} steps) in {time.time()-t0:.0f}s")


def _prepare_model_with_freeze(freeze_layers: int, device: str):
    """Load a fresh copy of the base model and attach freeze_n attribute."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    # Attach freeze count so train_specialist can use it
    model._freeze_n = freeze_layers
    # Actually freeze
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(freeze_layers):
        model.gpt_neox.layers[i].requires_grad_(False)
    return model


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
    print(f"  Training router ({ROUTER_STEPS} steps)...")
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
# Run one (steps, freeze) combination
# ============================================================================

def run_one_combo(steps: int, freeze: int, all_domain_chunks: dict,
                  tokenizer, held_out_mixed, base_loss: float,
                  device: str) -> dict:
    """Train 3 specialists for `steps` steps with `freeze` frozen layers, fuse, eval."""
    print(f"\n{'='*60}")
    print(f"  steps={steps}, freeze={freeze}")
    print(f"{'='*60}")
    t_start = time.time()

    combined_train = []
    for d in DOMAINS:
        combined_train.extend(all_domain_chunks[d]["train"])

    specialists = {}
    for domain in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"crossover_{domain}_freeze{freeze}_steps{steps}_seed{SEED}.pt"
        model = _prepare_model_with_freeze(freeze, device)
        if ckpt.exists():
            print(f"  Loading cached {domain} freeze={freeze} steps={steps} from {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"  Training {domain} specialist (freeze={freeze}, steps={steps})...")
            # Override train function to use our freeze-aware model
            train_specialist_direct(
                model=model,
                domain=domain,
                train_chunks=all_domain_chunks[domain]["train"],
                tokenizer=tokenizer,
                seed=SEED,
                device=device,
                max_steps=steps,
                freeze_n=freeze,
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved: {ckpt}")
        model.eval()
        specialists[domain] = model

    # Fuse via MoE
    moe = ThreeExpertMoE(specialists["code"], specialists["science"],
                         specialists["fiction"], hidden_size=HIDDEN_SIZE).to(device)
    train_router(moe, combined_train, device)
    moe.eval()

    moe_loss = eval_loss(moe, held_out_mixed, device, batch_size=2, is_fused=True)
    improvement_pct = (base_loss - moe_loss) / base_loss * 100

    print(f"  Result: moe_loss={moe_loss:.4f}, improvement={improvement_pct:+.1f}%")
    print(f"  Combo time: {time.time()-t_start:.0f}s")

    # Cleanup
    del moe
    for spec in specialists.values():
        del spec
    torch.cuda.empty_cache()

    return {"loss": round(moe_loss, 6), "improvement_pct": round(improvement_pct, 4)}


def train_specialist_direct(model, domain: str, train_chunks: list, tokenizer,
                             seed: int, device: str, max_steps: int,
                             freeze_n: int) -> None:
    """Train specialist with explicit freeze_n parameter."""
    set_seed(seed)

    # Ensure freezing is applied
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(freeze_n):
        model.gpt_neox.layers[i].requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    [{domain}] freeze={freeze_n}, trainable={trainable/1e6:.1f}M/{total/1e6:.1f}M")

    model.train()
    dataset = make_dataset_from_chunks(train_chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(max_steps * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(max_steps - warmup_steps, 1))

    step = 0
    accum = 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()
    log_interval = max(max_steps // 10, 50)

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
            if step % log_interval == 0 or step == max_steps:
                avg = running_loss / step
                elapsed = time.time() - t0
                print(f"    [{domain}] step {step}/{max_steps} | loss {avg:.4f} | {elapsed:.0f}s")

    print(f"  {domain} done in {time.time()-t0:.0f}s")


# ============================================================================
# Crossover detection
# ============================================================================

def find_crossover(steps_list: list, freeze0_imps: list, freeze4_imps: list):
    """Find first step count where freeze=4 consistently outperforms freeze=0."""
    crossover_steps = None
    for i, s in enumerate(steps_list):
        if freeze0_imps[i] is not None and freeze4_imps[i] is not None:
            if freeze4_imps[i] > freeze0_imps[i]:
                crossover_steps = s
                break
    return crossover_steps


# ============================================================================
# Figure
# ============================================================================

def save_figure(steps_list: list, freeze0_imps: list, freeze4_imps: list,
                crossover_steps):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter out None values
        valid0 = [(s, v) for s, v in zip(steps_list, freeze0_imps) if v is not None]
        valid4 = [(s, v) for s, v in zip(steps_list, freeze4_imps) if v is not None]

        if valid0:
            s0, v0 = zip(*valid0)
            ax.plot(np.log10(s0), v0, "b--o", label="freeze=0", linewidth=2, markersize=6)

        if valid4:
            s4, v4 = zip(*valid4)
            ax.plot(np.log10(s4), v4, "r-o", label="freeze=4", linewidth=2, markersize=6)

        if crossover_steps is not None:
            ax.axvline(x=np.log10(crossover_steps), color="green", linestyle=":",
                       alpha=0.7, label=f"Crossover at {crossover_steps} steps")

        # X-axis labels
        all_steps = sorted(set(steps_list))
        ax.set_xticks([np.log10(s) for s in all_steps])
        ax.set_xticklabels([str(s) for s in all_steps])
        ax.set_xlabel("Training Steps (log scale)")
        ax.set_ylabel("MoE Improvement over Base (%)")
        ax.set_title("Training Duration Crossover: freeze=0 vs freeze=4\n"
                     "(Pythia-410M, seed=42, 3 specialists)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_training_duration_crossover.png"
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
    print("KALAVU: Training Duration Crossover (freeze=0 vs freeze=4)")
    print("=" * 70)
    print(f"Steps: {STEPS_SWEEP}")
    print(f"Freeze: {FREEZE_SWEEP}")
    print(f"Total combos: {len(STEPS_SWEEP) * len(FREEZE_SWEEP)}")
    print(f"Seed: {SEED}")
    print(f"Estimated runtime: ~12.5 hours (dominated by 20000-step runs)")

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

    # Load data once
    print("\nLoading data...")
    code_texts = load_code_texts(N_SAMPLES_PER_DOMAIN)
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

    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_mixed = make_dataset_from_chunks(mixed_held)

    # Eval base model
    print(f"\nLoading base model for baseline eval...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()
    base_loss = eval_loss(base_model, held_out_mixed, device)
    print(f"  Base mixed loss: {base_loss:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # Load existing intermediate results if available
    inter_path = RESULTS_DIR / "training_duration_crossover.json"
    if inter_path.exists():
        print(f"\nLoading existing results from {inter_path}...")
        with open(inter_path) as f:
            results = json.load(f)
        # Convert existing data to lookup dict
        completed = {}
        for s_idx, s in enumerate(results.get("steps", STEPS_SWEEP)):
            f0_imps = results.get("freeze0_improvement", [])
            f4_imps = results.get("freeze4_improvement", [])
            f0_loss = results.get("freeze0_loss", [])
            f4_loss = results.get("freeze4_loss", [])
            if s_idx < len(f0_imps) and f0_imps[s_idx] is not None:
                completed[(s, 0)] = {
                    "loss": f0_loss[s_idx] if s_idx < len(f0_loss) else None,
                    "improvement_pct": f0_imps[s_idx],
                }
            if s_idx < len(f4_imps) and f4_imps[s_idx] is not None:
                completed[(s, 4)] = {
                    "loss": f4_loss[s_idx] if s_idx < len(f4_loss) else None,
                    "improvement_pct": f4_imps[s_idx],
                }
        print(f"  Found {len(completed)} completed combos")
    else:
        completed = {}

    # Run all (steps, freeze) combinations
    for steps in STEPS_SWEEP:
        for freeze in FREEZE_SWEEP:
            key = (steps, freeze)
            if key in completed:
                print(f"\n  Skipping steps={steps}, freeze={freeze} (already done: "
                      f"improvement={completed[key]['improvement_pct']:.1f}%)")
                continue

            result = run_one_combo(
                steps=steps,
                freeze=freeze,
                all_domain_chunks=all_domain_chunks,
                tokenizer=tokenizer,
                held_out_mixed=held_out_mixed,
                base_loss=base_loss,
                device=device,
            )
            completed[key] = result

            # Save intermediate results after each combo
            freeze0_imps = [completed.get((s, 0), {}).get("improvement_pct", None)
                            for s in STEPS_SWEEP]
            freeze4_imps = [completed.get((s, 4), {}).get("improvement_pct", None)
                            for s in STEPS_SWEEP]
            freeze0_loss = [completed.get((s, 0), {}).get("loss", None)
                            for s in STEPS_SWEEP]
            freeze4_loss = [completed.get((s, 4), {}).get("loss", None)
                            for s in STEPS_SWEEP]

            # Detect crossover from completed data
            valid_steps = [s for s in STEPS_SWEEP
                           if completed.get((s, 0), {}).get("improvement_pct") is not None
                           and completed.get((s, 4), {}).get("improvement_pct") is not None]
            valid_f0 = [completed[(s, 0)]["improvement_pct"] for s in valid_steps]
            valid_f4 = [completed[(s, 4)]["improvement_pct"] for s in valid_steps]
            crossover_steps = find_crossover(valid_steps, valid_f0, valid_f4)

            intermediate_output = {
                "steps": STEPS_SWEEP,
                "freeze0_improvement": freeze0_imps,
                "freeze4_improvement": freeze4_imps,
                "freeze0_loss": freeze0_loss,
                "freeze4_loss": freeze4_loss,
                "crossover_steps": crossover_steps,
                "base_loss": round(base_loss, 6),
                "seed": SEED,
                "model": f"{MODEL_ID}@{REVISION}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            with open(inter_path, "w") as f:
                json.dump(intermediate_output, f, indent=2)
            print(f"\n  Intermediate results saved to {inter_path}")

    # Final results
    freeze0_imps = [completed.get((s, 0), {}).get("improvement_pct", None)
                    for s in STEPS_SWEEP]
    freeze4_imps = [completed.get((s, 4), {}).get("improvement_pct", None)
                    for s in STEPS_SWEEP]
    freeze0_loss = [completed.get((s, 0), {}).get("loss", None) for s in STEPS_SWEEP]
    freeze4_loss = [completed.get((s, 4), {}).get("loss", None) for s in STEPS_SWEEP]

    valid_steps = [s for s in STEPS_SWEEP
                   if completed.get((s, 0), {}).get("improvement_pct") is not None
                   and completed.get((s, 4), {}).get("improvement_pct") is not None]
    valid_f0 = [completed[(s, 0)]["improvement_pct"] for s in valid_steps]
    valid_f4 = [completed[(s, 4)]["improvement_pct"] for s in valid_steps]
    crossover_steps = find_crossover(valid_steps, valid_f0, valid_f4)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Steps':>8} {'freeze=0':>10} {'freeze=4':>10} {'delta':>8}")
    print("-" * 40)
    for i, s in enumerate(STEPS_SWEEP):
        f0 = freeze0_imps[i]
        f4 = freeze4_imps[i]
        f0_str = f"{f0:+.1f}%" if f0 is not None else "N/A"
        f4_str = f"{f4:+.1f}%" if f4 is not None else "N/A"
        delta_str = f"{(f4 - f0):+.1f}pp" if (f0 is not None and f4 is not None) else "N/A"
        print(f"{s:>8} {f0_str:>10} {f4_str:>10} {delta_str:>8}")

    if crossover_steps is not None:
        print(f"\nCrossover detected at steps={crossover_steps}")
    else:
        print(f"\nNo crossover detected in range {STEPS_SWEEP}")

    # Save figure
    print("\nSaving figure...")
    save_figure(STEPS_SWEEP, freeze0_imps, freeze4_imps, crossover_steps)

    # Save final JSON
    final_output = {
        "steps": STEPS_SWEEP,
        "freeze0_improvement": freeze0_imps,
        "freeze4_improvement": freeze4_imps,
        "freeze0_loss": freeze0_loss,
        "freeze4_loss": freeze4_loss,
        "crossover_steps": crossover_steps,
        "base_loss": round(base_loss, 6),
        "seed": SEED,
        "model": f"{MODEL_ID}@{REVISION}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(inter_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved: {inter_path}")

    # Git commit + push
    cs_str = str(crossover_steps) if crossover_steps is not None else "null"
    msg = f"[kalavu] training duration crossover: crossover_steps={cs_str}"
    git_commit_push(msg)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
