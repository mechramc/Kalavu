#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-6.9B Specialist Step-Budget Sweep (Experiment B1)
=================================================================
Tests whether the 6.9B result (+2.4%) is bottlenecked by only 1,000 training
steps. Sweeps step counts and freeze depths to find the optimal config.

Phase 1: Step count sweep at freeze=6 (default)
  - 1,000 steps  (re-confirms existing result)
  - 2,000 steps
  - 4,000 steps

Phase 2: Freeze depth sweep at the best step count from Phase 1
  - K=4  (4/32 = 12.5%)
  - K=6  (6/32 = 18.8%)  ← already done in Phase 1
  - K=8  (8/32 = 25%)

Phase 3: 3 seeds on the best (steps, freeze) combination

Success criterion: if any config reaches +5% vs best specialist, the 6.9B
scaling story strengthens materially. Flat at ~2.4% → honest weak-scaling.

Runs on RunPod A100 80GB.
Resumable: each condition writes a result JSON; script skips it on re-run.

Usage:
  python kalavai_6b_step_sweep.py 2>&1 | tee sweep_log.txt
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

MODEL_ID        = "EleutherAI/pythia-6.9b"
REVISION        = "step10000"
HIDDEN_SIZE     = 4096
NUM_LAYERS      = 32
LR              = 1e-5
WEIGHT_DECAY    = 0.1
BATCH_SIZE      = 1
GRAD_ACCUM      = 8           # effective batch = 8
GRADIENT_CLIP   = 1.0
SEQ_LEN         = 512
WARMUP_FRACTION = 0.1
DOMAINS         = ["code", "science", "fiction"]
N_SAMPLES       = 3000
ROUTER_STEPS    = 500
ROUTER_LR       = 1e-3
ROUTER_BATCH    = 4
EVAL_BATCHES    = 50
EXPLORATION_SEED = 42

# Grid definition
STEP_COUNTS    = [1000, 2000, 4000]
FREEZE_DEPTHS  = [4, 6, 8]          # swept at best step count only
DEFAULT_FREEZE = 6                   # used during step count sweep

RESULTS_DIR    = Path("results/pythia/pythia_6b_step_sweep")
FIGURES_DIR    = Path("figures/pythia")
CHECKPOINT_BASE = Path("checkpoints/pythia_6b_step_sweep")

# ============================================================================
# Resume helpers
# ============================================================================

def condition_key(steps: int, freeze: int, seed: int) -> str:
    return f"steps{steps}_k{freeze}_seed{seed}"

def result_path(steps: int, freeze: int, seed: int) -> Path:
    return RESULTS_DIR / f"result_{condition_key(steps, freeze, seed)}.json"

def checkpoint_dir(steps: int, freeze: int) -> Path:
    return CHECKPOINT_BASE / f"steps{steps}_k{freeze}"

def specialist_ckpt(steps: int, freeze: int, domain: str) -> Path:
    return checkpoint_dir(steps, freeze) / f"{domain}_specialist_seed42.pt"

def git_commit_push(message: str):
    print(f"\n[git] {message}")
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            print("[git] Nothing to commit.")
            return
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[git] Pushed.")
    except subprocess.CalledProcessError as e:
        print(f"[git] WARNING: {e}")

# ============================================================================
# Dataset
# ============================================================================

class PackedChunkDataset(Dataset):
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
    labels    = torch.stack([b["labels"]    for b in batch])
    return {"input_ids": input_ids, "labels": labels}


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks: list, train_frac: float = 0.8, indist_frac: float = 0.1):
    n = len(chunks)
    train_end  = int(n * train_frac)
    indist_end = int(n * (train_frac + indist_frac))
    return chunks[:train_end], chunks[train_end:indist_end], chunks[indist_end:]

# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) >= 200:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    {len(texts)} code samples")
    return texts


def load_science_texts(n: int) -> list[str]:
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
        if len(texts) >= n:
            break
    print(f"    {len(texts)} science samples")
    return texts


def load_fiction_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading fiction (n={n})...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n:
            break
    print(f"    {len(texts)} fiction samples")
    return texts


def predownload_datasets():
    print("\n" + "=" * 70)
    print("PRE-DOWNLOADING ALL DATASETS")
    print("=" * 70)
    code_texts    = load_code_texts(N_SAMPLES)
    science_texts = load_science_texts(N_SAMPLES)
    fiction_texts = load_fiction_texts(N_SAMPLES)
    print(f"\nDownload complete: code={len(code_texts)}, "
          f"science={len(science_texts)}, fiction={len(fiction_texts)}")
    return code_texts, science_texts, fiction_texts

# ============================================================================
# Model loading
# ============================================================================

def load_model(device: str, gradient_checkpointing: bool = True):
    print(f"\nLoading {MODEL_ID} (revision={REVISION}) in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
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


def apply_freeze(model, n: int):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen {n}/{NUM_LAYERS} layers. "
          f"Trainable: {trainable/1e9:.3f}B / {total/1e9:.2f}B "
          f"({100*trainable/total:.1f}%)")

# ============================================================================
# ThreeExpertMoE
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
        self.router = nn.Linear(hidden_size, 3, bias=False)

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach().float()
        last_h = out.hidden_states[-1].detach().float()
        h_pooled = last_h.mean(dim=1)
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
# Training
# ============================================================================

def train_specialist(model, domain: str, train_chunks: list, device: str,
                     seed: int, max_steps: int, freeze: int,
                     log_every: int = 50) -> list:
    set_seed(seed)
    apply_freeze(model, freeze)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)

    warmup_steps    = int(max_steps * WARMUP_FRACTION)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps))

    step, accum = 0, 0
    running_loss = 0.0
    loss_history = []
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= max_steps:
            break
        batch_device = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(**batch_device)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        accum        += 1
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
            accum  = 0
            step  += 1
            if step % log_every == 0 or step == max_steps:
                avg     = running_loss / step
                elapsed = time.time() - t0
                print(f"  [{domain}] step {step}/{max_steps} | loss {avg:.4f} | {elapsed:.0f}s")
                loss_history.append((step, round(avg, 6)))

    model.eval()
    print(f"  {domain} done ({time.time()-t0:.0f}s)")
    return loss_history

# ============================================================================
# Eval
# ============================================================================

@torch.no_grad()
def eval_loss(model, dataset, device: str, batch_size: int = 1,
              is_fused: bool = False) -> float:
    g = torch.Generator()
    g.manual_seed(999)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, collate_fn=_collate, generator=g)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES:
            break
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
    return total / count if count > 0 else float("inf")


@torch.no_grad()
def eval_router_distribution(moe, eval_datasets: dict, device: str,
                              n_batches: int = 20) -> dict:
    moe.eval()
    results = {}
    for domain, ds in eval_datasets.items():
        loader    = DataLoader(ds, batch_size=1, shuffle=False,
                               drop_last=True, collate_fn=_collate)
        gate_sums = [0.0, 0.0, 0.0]
        count     = 0
        for batch in loader:
            if count >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(3):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        results[domain] = [round(g / count, 4) for g in gate_sums] if count > 0 else [0.333] * 3
    return results

# ============================================================================
# Weight averaging
# ============================================================================

def weight_average_three(spec_a, spec_b, spec_c):
    print("  Weight averaging on CPU...")
    sa = {k: v.cpu().float() for k, v in spec_a.state_dict().items()}
    sb = {k: v.cpu().float() for k, v in spec_b.state_dict().items()}
    sc = {k: v.cpu().float() for k, v in spec_c.state_dict().items()}
    avg_state = {k: ((sa[k] + sb[k] + sc[k]) / 3.0).to(torch.bfloat16) for k in sa}
    avg = copy.deepcopy(spec_a).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg

# ============================================================================
# Router training
# ============================================================================

def train_router(moe: ThreeExpertMoE, train_datasets: dict, device: str):
    all_chunks = []
    for ds in train_datasets.values():
        all_chunks.extend(ds.chunks)
    combined  = make_dataset_from_chunks(all_chunks)
    moe.router = moe.router.to(device)
    optimizer  = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader     = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                            drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"\n  Router training ({ROUTER_STEPS} steps, {len(combined)} chunks)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Run one condition: (steps, freeze, seed)
# ============================================================================

def run_condition(steps: int, freeze: int, seed: int, device: str,
                  tokenizer, all_domain_chunks: dict,
                  base_losses: dict) -> dict:
    """
    Train 3 specialists at (steps, freeze), fuse with router, evaluate.
    Returns result dict. Skips if result file already exists.
    """
    rpath = result_path(steps, freeze, seed)
    if rpath.exists():
        print(f"\n[skip] {condition_key(steps, freeze, seed)} already done.")
        return json.loads(rpath.read_text(encoding="utf-8"))

    print(f"\n{'='*70}")
    print(f"CONDITION: steps={steps}, freeze={freeze}, seed={seed}")
    print(f"{'='*70}")

    ckpt_dir = checkpoint_dir(steps, freeze)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Train or load specialists ─────────────────────────────────────────
    trained: dict = {}
    for domain in DOMAINS:
        ckpt = specialist_ckpt(steps, freeze, domain) if seed == EXPLORATION_SEED else None
        if ckpt and ckpt.exists():
            print(f"\n  Loading cached {domain} (steps={steps}, k={freeze})...")
            model = load_model(device, gradient_checkpointing=False)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            del state
        else:
            print(f"\n  Training {domain} specialist (steps={steps}, k={freeze}, seed={seed})...")
            model = load_model(device, gradient_checkpointing=True)
            train_specialist(model, domain, all_domain_chunks[domain]["train"],
                             device, seed, max_steps=steps, freeze=freeze)
            model.eval()
            # Save checkpoint for seed=42 only (variance seeds are ephemeral)
            if seed == EXPLORATION_SEED:
                torch.save(model.state_dict(), ckpt)
                print(f"  Saved: {ckpt} ({ckpt.stat().st_size/1e9:.1f}GB)")

        model.to("cpu")
        torch.cuda.empty_cache()
        print(f"  {domain} moved to CPU")
        trained[domain] = model

    # ── Move all back to GPU ──────────────────────────────────────────────
    print(f"\n  Moving specialists to GPU...")
    for domain in DOMAINS:
        trained[domain].to(device)
    torch.cuda.empty_cache()

    spec_code    = trained["code"]
    spec_science = trained["science"]
    spec_fiction = trained["fiction"]

    # ── Build eval datasets ───────────────────────────────────────────────
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS:
        mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed"] = make_dataset_from_chunks(mixed_held)

    # ── Eval specialists ──────────────────────────────────────────────────
    fusion_losses = {}
    fusion_losses["base"] = base_losses  # from main(), computed once

    for label, spec in [("code_spec", spec_code), ("science_spec", spec_science),
                         ("fiction_spec", spec_fiction)]:
        losses = {}
        for d, ds in held_out_sets.items():
            losses[d] = round(eval_loss(spec, ds, device), 6)
        fusion_losses[label] = losses

    # ── Weight average ────────────────────────────────────────────────────
    avg_model = weight_average_three(spec_code, spec_science, spec_fiction)
    avg_model.to(device)
    wa_losses = {}
    for d, ds in held_out_sets.items():
        wa_losses[d] = round(eval_loss(avg_model, ds, device), 6)
    fusion_losses["weight_avg"] = wa_losses
    del avg_model
    torch.cuda.empty_cache()

    # ── MoE fusion ────────────────────────────────────────────────────────
    train_ds_dict = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"])
                     for d in DOMAINS}
    moe = ThreeExpertMoE(spec_code, spec_science, spec_fiction).to(device)
    train_router(moe, train_ds_dict, device)

    moe_losses = {}
    for d, ds in held_out_sets.items():
        moe_losses[d] = round(eval_loss(moe, ds, device, is_fused=True), 6)
    fusion_losses["moe"] = moe_losses

    router_dist = eval_router_distribution(moe, held_out_sets, device)

    del moe, spec_code, spec_science, spec_fiction
    torch.cuda.empty_cache()

    # ── Compute metrics ───────────────────────────────────────────────────
    best_spec_mixed = min(
        fusion_losses["code_spec"]["mixed"],
        fusion_losses["science_spec"]["mixed"],
        fusion_losses["fiction_spec"]["mixed"],
    )
    moe_mixed   = fusion_losses["moe"]["mixed"]
    base_mixed  = base_losses["mixed"]

    improvement_vs_spec = round((best_spec_mixed - moe_mixed) / best_spec_mixed * 100, 4)
    improvement_vs_base = round((base_mixed - moe_mixed) / base_mixed * 100, 4)

    print(f"\n  KEY RESULT (steps={steps}, k={freeze}, seed={seed}):")
    print(f"    Base mixed:      {base_mixed:.4f}")
    print(f"    Best spec mixed: {best_spec_mixed:.4f}")
    print(f"    MoE mixed:       {moe_mixed:.4f}")
    print(f"    vs spec:         +{improvement_vs_spec:.2f}%")
    print(f"    vs base:         +{improvement_vs_base:.2f}%")

    result = {
        "steps":               steps,
        "freeze":              freeze,
        "seed":                seed,
        "model_id":            MODEL_ID,
        "revision":            REVISION,
        "eval_heldout":        fusion_losses,
        "base_mixed":          base_mixed,
        "best_spec_mixed":     best_spec_mixed,
        "moe_mixed":           moe_mixed,
        "improvement_vs_spec": improvement_vs_spec,
        "improvement_vs_base": improvement_vs_base,
        "router_distribution": router_dist,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Saved: {rpath}")

    git_commit_push(
        f"[kalavai] 6.9B step sweep: steps={steps} k={freeze} seed={seed} "
        f"→ +{improvement_vs_spec:.2f}% vs spec, +{improvement_vs_base:.2f}% vs base"
    )
    return result

# ============================================================================
# Figures
# ============================================================================

def save_step_sweep_figure(results_by_steps: dict):
    """Line plot: improvement vs steps at default freeze depth."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps_sorted = sorted(results_by_steps.keys())
        vs_spec = [results_by_steps[s]["improvement_vs_spec"] for s in steps_sorted]
        vs_base = [results_by_steps[s]["improvement_vs_base"] for s in steps_sorted]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(steps_sorted, vs_spec, "o-", color="#9b59b6", lw=2,
                label="vs Best Specialist")
        ax.plot(steps_sorted, vs_base, "s--", color="#3498db", lw=2,
                label="vs Base Model")
        ax.axhline(5.0, color="green", linestyle=":", lw=1.5,
                   label="NeurIPS gate (5%)")
        for s, v in zip(steps_sorted, vs_spec):
            ax.annotate(f"+{v:.1f}%", (s, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel("Specialist Training Steps", fontsize=12)
        ax.set_ylabel("MoE Improvement (%)", fontsize=12)
        ax.set_title(f"Pythia-6.9B: Improvement vs Step Budget (freeze={DEFAULT_FREEZE}, seed=42)",
                     fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = FIGURES_DIR / "fig_6b_step_sweep.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save step sweep figure: {e}")


def save_freeze_sweep_figure(best_steps: int, results_by_freeze: dict):
    """Bar chart: improvement vs freeze depth at best step count."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        freezes  = sorted(results_by_freeze.keys())
        vs_spec  = [results_by_freeze[k]["improvement_vs_spec"] for k in freezes]
        labels   = [f"K={k}\n({100*k//NUM_LAYERS}%)" for k in freezes]
        colors   = ["#e74c3c", "#2ecc71", "#3498db"][:len(freezes)]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, vs_spec, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, vs_spec):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"+{val:.1f}%", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        ax.axhline(5.0, color="green", linestyle=":", lw=1.5,
                   label="NeurIPS gate (5%)")
        ax.set_xlabel("Freeze Depth", fontsize=12)
        ax.set_ylabel("MoE Improvement vs Best Specialist (%)", fontsize=11)
        ax.set_title(f"Pythia-6.9B: Freeze Depth Sweep ({best_steps} steps, seed=42)",
                     fontsize=11)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        path = FIGURES_DIR / f"fig_6b_freeze_sweep_{best_steps}steps.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save freeze sweep figure: {e}")


def save_summary_figure(all_results: list):
    """Grid heatmap of all (steps, freeze) conditions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        steps_list  = sorted(set(r["steps"]  for r in all_results))
        freeze_list = sorted(set(r["freeze"] for r in all_results))

        data = np.full((len(freeze_list), len(steps_list)), np.nan)
        for r in all_results:
            if r["seed"] != EXPLORATION_SEED:
                continue
            ri = freeze_list.index(r["freeze"])
            ci = steps_list.index(r["steps"])
            data[ri, ci] = r["improvement_vs_spec"]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=max(8, float(np.nanmax(data)) + 1))
        ax.set_xticks(range(len(steps_list)))
        ax.set_xticklabels([f"{s}k" for s in steps_list])
        ax.set_xlabel("Specialist Training Steps")
        ax.set_yticks(range(len(freeze_list)))
        ax.set_yticklabels([f"K={f}" for f in freeze_list])
        ax.set_ylabel("Freeze Depth")
        ax.set_title("Pythia-6.9B Step & Freeze Sweep — MoE Improvement vs Best Spec (%)")
        for i in range(len(freeze_list)):
            for j in range(len(steps_list)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i,j]:.1f}%", ha="center", va="center",
                            fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Improvement (%)")
        fig.tight_layout()

        path = FIGURES_DIR / "fig_6b_step_freeze_grid.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save summary grid: {e}")


# ============================================================================
# Summary JSON
# ============================================================================

def save_summary(all_results: list, best_config: dict):
    summary = {
        "experiment":    "6b_step_sweep",
        "model_id":      MODEL_ID,
        "revision":      REVISION,
        "grid_results":  all_results,
        "best_config":   best_config,
        "conclusion":    (
            f"Best: steps={best_config.get('steps')}, "
            f"freeze={best_config.get('freeze')} → "
            f"+{best_config.get('improvement_vs_spec', 0):.2f}% vs spec"
        ),
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = RESULTS_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved: {path}")
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-6.9B Step-Budget Sweep (B1)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Download data ─────────────────────────────────────────────────────
    code_texts, science_texts, fiction_texts = predownload_datasets()

    # ── Pack and split data (once, shared across all conditions) ──────────
    print("\n" + "=" * 70)
    print("PACKING DATA (shared across all conditions)")
    print("=" * 70)
    set_seed(42)  # deterministic data split
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        train_c, indist_c, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {
            "train": train_c, "indist": indist_c, "held_out": held_c
        }
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # ── Base eval (once) ──────────────────────────────────────────────────
    base_eval_path = RESULTS_DIR / "base_eval.json"
    if base_eval_path.exists():
        print(f"\n[skip] Base eval already done.")
        base_losses = json.loads(base_eval_path.read_text(encoding="utf-8"))
    else:
        print("\n" + "=" * 70)
        print("BASE EVAL")
        print("=" * 70)
        base_model = load_model(device, gradient_checkpointing=False)
        base_model.eval()

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

        del base_model
        torch.cuda.empty_cache()

        base_eval_path.write_text(json.dumps(base_losses, indent=2), encoding="utf-8")
        git_commit_push("[kalavai] 6.9B step sweep: base eval complete")

    print(f"\n  Base mixed loss: {base_losses['mixed']:.4f}")

    # ========================================================================
    # PHASE 1: Step count sweep at default freeze
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1: Step count sweep (freeze={DEFAULT_FREEZE}, seed={EXPLORATION_SEED})")
    print("=" * 70)

    step_results = {}
    for steps in STEP_COUNTS:
        result = run_condition(steps, DEFAULT_FREEZE, EXPLORATION_SEED,
                               device, tokenizer, all_domain_chunks, base_losses)
        step_results[steps] = result

    save_step_sweep_figure(step_results)

    # Determine best step count
    best_steps = max(step_results, key=lambda s: step_results[s]["improvement_vs_spec"])
    best_vs_spec = step_results[best_steps]["improvement_vs_spec"]
    print(f"\n  Best step count: {best_steps} steps → +{best_vs_spec:.2f}% vs spec")

    # ========================================================================
    # PHASE 2: Freeze depth sweep at best step count
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: Freeze depth sweep (steps={best_steps}, seed={EXPLORATION_SEED})")
    print("=" * 70)

    freeze_results = {}
    for freeze in FREEZE_DEPTHS:
        result = run_condition(best_steps, freeze, EXPLORATION_SEED,
                               device, tokenizer, all_domain_chunks, base_losses)
        freeze_results[freeze] = result

    save_freeze_sweep_figure(best_steps, freeze_results)

    # Determine best freeze depth
    best_freeze = max(freeze_results, key=lambda f: freeze_results[f]["improvement_vs_spec"])
    best_overall_vs_spec = freeze_results[best_freeze]["improvement_vs_spec"]
    print(f"\n  Best freeze depth: K={best_freeze} → +{best_overall_vs_spec:.2f}% vs spec")

    # ========================================================================
    # PHASE 3: 3 seeds on best (steps, freeze)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 3: 3 seeds on best config (steps={best_steps}, freeze={best_freeze})")
    print("=" * 70)

    seed_results = []
    for seed in [42, 137, 2026]:
        result = run_condition(best_steps, best_freeze, seed,
                               device, tokenizer, all_domain_chunks, base_losses)
        seed_results.append(result)

    improvements = [r["improvement_vs_spec"] for r in seed_results]
    mean_imp = round(sum(improvements) / len(improvements), 4)
    std_imp  = round(
        (sum((x - mean_imp)**2 for x in improvements) / len(improvements)) ** 0.5, 4
    )

    print(f"\n  3-seed result: +{mean_imp:.2f}% ± {std_imp:.2f}%")
    print(f"  Individual: {[round(v, 2) for v in improvements]}")

    # ========================================================================
    # Final summary
    # ========================================================================
    all_results = (
        list(step_results.values()) +
        [v for k, v in freeze_results.items() if k != DEFAULT_FREEZE] +
        [r for r in seed_results if r["seed"] != EXPLORATION_SEED]
    )

    best_config = {
        "steps":               best_steps,
        "freeze":              best_freeze,
        "improvement_vs_spec": mean_imp,
        "std_vs_spec":         std_imp,
        "seeds":               [r["seed"] for r in seed_results],
        "per_seed_improvements": improvements,
    }

    save_summary_figure(all_results + list(step_results.values()) + list(freeze_results.values()))
    summary = save_summary(all_results, best_config)

    git_commit_push(
        f"[kalavai] 6.9B step sweep COMPLETE: best={best_steps}steps k={best_freeze} "
        f"+{mean_imp:.2f}%±{std_imp:.2f}% (3 seeds)"
    )

    print("\n" + "=" * 70)
    print("B1 COMPLETE")
    print("=" * 70)
    print(f"  Best config:     steps={best_steps}, freeze={best_freeze}")
    print(f"  Improvement:     +{mean_imp:.2f}% ± {std_imp:.2f}% vs best specialist")
    print(f"  NeurIPS gate:    {'PASS ✓' if mean_imp >= 5.0 else 'BELOW 5% — consider TMLR'}")
    print(f"  Results dir:     {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
