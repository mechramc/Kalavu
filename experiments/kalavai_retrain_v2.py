#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Phase 1 Retrain v2
============================
Experiments that require full specialist retraining from scratch.
All specialists are retrained and immediately evaluated with the corrected
per-domain equal-weight protocol. New results saved to results/pythia/v2/.
Old results preserved.

Experiments:
  shared_init   Shared initialization necessity: control vs large_gap vs small_gap
                Tests whether specialists MUST start from the same checkpoint.
  heterogeneous Heterogeneous cooperative: control vs diff_batch / diff_lr / diff_steps
                Tests protocol robustness to realistic contributor variation.
  all           Run both experiments in sequence

Checkpoints saved to:
  checkpoints/pythia/shared_init_v2/   (condition_domain_seed.pt)
  checkpoints/pythia/heterogeneous_v2/ (condition_domain_seed42.pt)

Usage:
  cd /c/Github/Kalavai
  python experiments/kalavai_retrain_v2.py --experiment shared_init
  python experiments/kalavai_retrain_v2.py --experiment heterogeneous
  python experiments/kalavai_retrain_v2.py --experiment all
"""

import argparse
import copy
import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants
# ============================================================================

MODEL_ID    = "EleutherAI/pythia-410m"
FREEZE_LAYERS = 4
LR          = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS   = 2000
BATCH_SIZE  = 2
GRAD_ACCUM  = 4
GRADIENT_CLIP = 1.0
SEQ_LEN     = 512
WARMUP_FRAC = 0.1
DOMAINS     = ["code", "science", "fiction"]
N_SAMPLES   = 3000
ROUTER_STEPS = 500
ROUTER_LR   = 1e-3
ROUTER_BATCH = 4
EVAL_BATCH  = 4
EVAL_BATCHES = 50
HIDDEN_SIZE = 1024

CKPT_DIR     = Path("checkpoints/pythia")
CKPT_SI_V2   = Path("checkpoints/pythia/shared_init_v2")
CKPT_HET_V2  = Path("checkpoints/pythia/heterogeneous_v2")
RESULTS_V2   = Path("results/pythia/v2")

# Shared-init conditions
#   control:    all from step10000 (3 seeds)
#   large_gap:  code=step5000, science=step10000, fiction=step20000 (3 seeds)
#   small_gap:  code=step8000, science=step10000, fiction=step12000 (seed42 only)
SI_CONDITIONS = {
    "control":    {"code": "step10000", "science": "step10000", "fiction": "step10000"},
    "large_gap":  {"code": "step5000",  "science": "step10000", "fiction": "step20000"},
    "small_gap":  {"code": "step8000",  "science": "step10000", "fiction": "step12000"},
}
SI_SEEDS = {"control": [42, 137, 2026], "large_gap": [42, 137, 2026], "small_gap": [42]}

# Heterogeneous conditions
HET_CONDITIONS = {
    "control":    {
        "code":    {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "fiction": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
    },
    "diff_batch": {
        # total tokens held equal: 2*4*512*2000 = 8,192,000 tokens per specialist
        "code":    {"batch": 1, "accum": 4, "steps": 4000, "lr": 2e-5},
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "fiction": {"batch": 4, "accum": 4, "steps": 1000, "lr": 2e-5},
    },
    "diff_lr": {
        "code":    {"batch": 2, "accum": 4, "steps": 2000, "lr": 1e-5},
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "fiction": {"batch": 2, "accum": 4, "steps": 2000, "lr": 5e-5},
    },
    "diff_steps": {
        "code":    {"batch": 2, "accum": 4, "steps": 1000, "lr": 2e-5},
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "fiction": {"batch": 2, "accum": 4, "steps": 3000, "lr": 2e-5},
    },
}
HET_SEED = 42

# ============================================================================
# Dataset
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated), return_tensors="pt", truncation=False,
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


def make_dataset(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks, train_frac=0.8, held_out_frac=0.1):
    n = len(chunks)
    a = int(n * train_frac)
    b = int(n * (train_frac + held_out_frac))
    return chunks[:a], chunks[a:b], chunks[b:]

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
    print(f"    {len(texts)} samples")
    return texts


def load_science_texts(n):
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = (item.get("support", "") + "\n" +
                   item.get("question", "") + "\n" +
                   item.get("correct_answer", ""))
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
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
    print(f"    {len(texts)} samples")
    return texts


def load_data(tokenizer, n=N_SAMPLES):
    print("\nLoading data...")
    raw = {
        "code":    load_code_texts(n),
        "science": load_science_texts(n),
        "fiction": load_fiction_texts(n),
    }
    print("\nPacking and splitting (80/10/10)...")
    train_chunks, held_out_chunks = {}, {}
    for domain, texts in raw.items():
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        tr, _, ho = split_chunks(ds_full.chunks)
        train_chunks[domain]    = tr
        held_out_chunks[domain] = ho
        print(f"  {domain}: train={len(tr)}, held_out={len(ho)}")
    return train_chunks, held_out_chunks

# ============================================================================
# MoE (same as corrected_eval.py — must match exactly)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    """410M 2-layer MLP router. Matches main experiment."""
    def __init__(self, spec_a, spec_b, spec_c):
        super().__init__()
        self.spec_a, self.spec_b, self.spec_c = spec_a, spec_b, spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 3, bias=False),
        )

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run(self.spec_a, input_ids)
        lb, hb = self._run(self.spec_b, input_ids)
        lc, hc = self._run(self.spec_c, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc) / 3.0), dim=-1)
        fused = gates[:, 0:1, None] * la + gates[:, 1:2, None] * lb + gates[:, 2:3, None] * lc
        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, gates

# ============================================================================
# Evaluation — corrected protocol
# ============================================================================

@torch.no_grad()
def eval_loss_domain(model, dataset, device, bs=EVAL_BATCH, n_batches=EVAL_BATCHES, is_fused=False):
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= n_batches: break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(input_ids, labels=labels)
        else:
            loss = model(input_ids=input_ids, labels=labels).loss
        if loss is not None:
            total += loss.item(); count += 1
    return total / count if count > 0 else float("inf")


def eval_all_domains(model, held_out_sets, device, bs=EVAL_BATCH, n_batches=EVAL_BATCHES,
                     is_fused=False, label=""):
    losses = {}
    for domain, ds in held_out_sets.items():
        t0 = time.time()
        loss = eval_loss_domain(model, ds, device, bs, n_batches, is_fused)
        losses[domain] = round(loss, 6)
        print(f"    {label+' ' if label else ''}{domain:8s}: {loss:.4f}  ({time.time()-t0:.1f}s)")
    losses["equal_weight_avg"] = round(
        sum(losses[d] for d in held_out_sets) / len(held_out_sets), 6)
    return losses


def pct_improvement(before, after):
    return round((before - after) / before * 100, 4)

# ============================================================================
# Specialist training
# ============================================================================

def load_base_from_revision(revision: str, device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model


def freeze_layers(model, n):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  freeze={n}: {trainable/1e6:.1f}M / {total/1e6:.1f}M trainable "
          f"({100*trainable/total:.1f}%)")


def train_specialist(model, train_chunks, domain, seed, device, steps, batch_size, accum,
                     lr, warmup_frac=WARMUP_FRAC, grad_clip=GRADIENT_CLIP,
                     weight_decay=WEIGHT_DECAY):
    """Train one specialist. Returns trained model (in-place)."""
    torch.manual_seed(seed)
    dataset = make_dataset(train_chunks)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=weight_decay)
    warmup_steps = int(steps * warmup_frac)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps, eta_min=lr * 0.1)

    print(f"  Training {domain} specialist: {steps} steps, lr={lr}, "
          f"batch={batch_size}×accum{accum} (seed={seed})")
    model.train()
    optimizer.zero_grad()
    for step in range(1, steps + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        loss = model(input_ids=input_ids, labels=labels).loss
        (loss / accum).backward()
        if step % accum == 0:
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if step > warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
        if step % 500 == 0 or step == steps:
            print(f"    step {step}/{steps}: loss={loss.item():.4f}")
    model.eval()
    return model


def train_router(moe, train_chunks_by_domain, device):
    all_chunks = []
    for chunks in train_chunks_by_domain.values():
        all_chunks.extend(chunks)
    combined  = make_dataset(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader    = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
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
    moe.eval()

# ============================================================================
# Experiment: shared_init
# ============================================================================

def run_shared_init(train_chunks, held_out_sets, device):
    """
    Tests whether specialists MUST start from the same checkpoint.

    Condition 1 (control):   all from step10000, 3 seeds
    Condition 2 (large_gap): code=step5000, science=step10000, fiction=step20000, 3 seeds
    Condition 3 (small_gap): code=step8000, science=step10000, fiction=step12000, seed42 only

    Control reuses existing seed42 main-experiment specialists where possible.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: shared_init (initialization gap ablation)")
    print("="*70)

    CKPT_SI_V2.mkdir(parents=True, exist_ok=True)
    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "shared_init_v2.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_keys = {(r["condition"], r["seed"]) for r in existing.get("results", [])}
        results_list = existing.get("results", [])
        print(f"  Resuming: {len(done_keys)} (condition, seed) combos already done")
    else:
        done_keys = set()
        results_list = []

    # Eval base model once
    base_key = ("_base", 0)
    if base_key not in done_keys:
        print("\n[base model]")
        base = load_base_from_revision("step10000", device)
        base_eval = eval_all_domains(base, held_out_sets, device)
        results_list.append({"condition": "_base", "seed": 0,
                              "eval": {k: v for k, v in base_eval.items() if not k.startswith("_")}})
        del base; torch.cuda.empty_cache()
        _save(out_path, "shared_init_v2", results_list)

    for condition_name, inits in SI_CONDITIONS.items():
        for seed in SI_SEEDS[condition_name]:
            key = (condition_name, seed)
            if key in done_keys:
                print(f"\n[{condition_name}, seed={seed}] — already done, skipping")
                continue

            print(f"\n{'─'*60}")
            print(f"[{condition_name}, seed={seed}]  "
                  + "  ".join(f"{d}←{rev}" for d, rev in inits.items()))
            print(f"{'─'*60}")

            # Check if we can reuse existing checkpoints (control/seed42)
            specialists = {}
            for domain in DOMAINS:
                revision = inits[domain]
                ckpt_name = f"{condition_name}_{domain}_seed{seed}.pt"
                ckpt_path = CKPT_SI_V2 / ckpt_name

                # Reuse main experiment checkpoint for control condition
                main_ckpt = CKPT_DIR / f"{domain}_specialist_seed{seed}.pt"
                if condition_name == "control" and main_ckpt.exists():
                    print(f"  [{domain}] loading from main experiment: {main_ckpt.name}")
                    spec = load_base_from_revision(revision, device)
                    spec.load_state_dict(
                        torch.load(main_ckpt, map_location=device, weights_only=True))
                    spec.eval()
                elif ckpt_path.exists():
                    print(f"  [{domain}] loading from v2 checkpoint: {ckpt_path.name}")
                    spec = load_base_from_revision(revision, device)
                    spec.load_state_dict(
                        torch.load(ckpt_path, map_location=device, weights_only=True))
                    spec.eval()
                else:
                    print(f"  [{domain}] training from {revision}...")
                    spec = load_base_from_revision(revision, device)
                    freeze_layers(spec, FREEZE_LAYERS)
                    spec = train_specialist(
                        spec, train_chunks[domain], domain, seed, device,
                        steps=MAX_STEPS, batch_size=BATCH_SIZE, accum=GRAD_ACCUM,
                        lr=LR, warmup_frac=WARMUP_FRAC,
                    )
                    torch.save(spec.state_dict(), ckpt_path)
                    print(f"    Saved: {ckpt_path}")

                specialists[domain] = spec

            # Eval specialists
            spec_evals = {}
            for domain, spec in specialists.items():
                spec_evals[domain] = eval_all_domains(spec, held_out_sets, device)

            # Build + train router
            moe = ThreeExpertMoE(*[specialists[d] for d in DOMAINS]).to(device)
            train_router(moe, train_chunks, device)
            moe.eval()

            moe_eval = eval_all_domains(moe, held_out_sets, device, is_fused=True)
            best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
            moe_eq = moe_eval["equal_weight_avg"]

            entry = {
                "condition":  condition_name,
                "seed":       seed,
                "init_revisions": inits,
                "moe_equal_weight":      moe_eq,
                "best_spec_equal_weight": best_spec_eq,
                "improvement_vs_spec":   pct_improvement(best_spec_eq, moe_eq),
                "per_domain_moe":  {k: v for k, v in moe_eval.items() if not k.startswith("_")},
                "per_domain_specs": {d: {k: v for k, v in e.items() if not k.startswith("_")}
                                     for d, e in spec_evals.items()},
                "eval_method": "per-domain-separate-equal-weight",
                "eval_batch_size": EVAL_BATCH,
            }
            results_list.append(entry)
            print(f"  => improvement_vs_spec: {entry['improvement_vs_spec']:+.2f}%")

            _save(out_path, "shared_init_v2", results_list)

            del moe
            for s in specialists.values(): del s
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SHARED_INIT SUMMARY (v2, corrected eval)")
    print("="*70)
    import statistics as st
    by_cond = {}
    for r in results_list:
        if r["condition"].startswith("_"): continue
        by_cond.setdefault(r["condition"], []).append(r)
    print(f"{'condition':>16}  {'seeds':>12}  {'vs spec (mean)':>16}  {'std':>8}")
    print("-" * 58)
    for cond in ["control", "large_gap", "small_gap"]:
        if cond not in by_cond: continue
        entries = by_cond[cond]
        imps = [e["improvement_vs_spec"] for e in entries]
        seeds_str = ",".join(str(e["seed"]) for e in entries)
        mean_imp = st.mean(imps)
        std_imp  = st.stdev(imps) if len(imps) > 1 else 0.0
        print(f"{cond:>16}  {seeds_str:>12}  {mean_imp:>+15.2f}%  {std_imp:>7.3f}%")

    return results_list

# ============================================================================
# Experiment: heterogeneous
# ============================================================================

def run_heterogeneous(train_chunks, held_out_sets, device):
    """
    Tests protocol robustness to realistic contributor variation.
    4 conditions, seed=42 only. Each condition trains all 3 specialists
    with different training configs, then fuses and evaluates.
    Success: fusion within 2pp of control across all conditions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: heterogeneous (training heterogeneity robustness)")
    print("="*70)

    CKPT_HET_V2.mkdir(parents=True, exist_ok=True)
    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "heterogeneous_v2.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_keys = set(existing.get("done_conditions", []))
        results_list = existing.get("results", [])
        print(f"  Resuming: {len(done_keys)} conditions already done")
    else:
        done_keys = set()
        results_list = []

    for condition_name, per_domain_cfg in HET_CONDITIONS.items():
        if condition_name in done_keys:
            print(f"\n[{condition_name}] — already done, skipping")
            continue

        print(f"\n{'─'*60}")
        print(f"[{condition_name}]")
        for d, cfg in per_domain_cfg.items():
            print(f"  {d}: steps={cfg['steps']}, batch={cfg['batch']}×{cfg['accum']}, "
                  f"lr={cfg['lr']:.0e}")
        print(f"{'─'*60}")

        specialists = {}
        for domain in DOMAINS:
            cfg = per_domain_cfg[domain]
            ckpt_name = f"{condition_name}_{domain}_seed{HET_SEED}.pt"
            ckpt_path = CKPT_HET_V2 / ckpt_name

            # Reuse main-experiment checkpoint for control condition
            main_ckpt = CKPT_DIR / f"{domain}_specialist_seed{HET_SEED}.pt"
            if condition_name == "control" and main_ckpt.exists():
                print(f"  [{domain}] loading from main experiment: {main_ckpt.name}")
                spec = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision="step10000", dtype=torch.bfloat16, trust_remote_code=True
                ).to(device)
                spec.load_state_dict(
                    torch.load(main_ckpt, map_location=device, weights_only=True))
                spec.eval()
            elif ckpt_path.exists():
                print(f"  [{domain}] loading v2 checkpoint: {ckpt_path.name}")
                spec = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision="step10000", dtype=torch.bfloat16, trust_remote_code=True
                ).to(device)
                spec.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=True))
                spec.eval()
            else:
                print(f"  [{domain}] training from step10000 "
                      f"({cfg['steps']} steps, lr={cfg['lr']:.0e})...")
                spec = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision="step10000", dtype=torch.bfloat16, trust_remote_code=True
                ).to(device)
                freeze_layers(spec, FREEZE_LAYERS)
                spec = train_specialist(
                    spec, train_chunks[domain], domain, HET_SEED, device,
                    steps=cfg["steps"], batch_size=cfg["batch"], accum=cfg["accum"],
                    lr=cfg["lr"],
                )
                torch.save(spec.state_dict(), ckpt_path)
                print(f"    Saved: {ckpt_path}")

            specialists[domain] = spec

        # Eval each specialist
        spec_evals = {}
        for domain, spec in specialists.items():
            spec_evals[domain] = eval_all_domains(spec, held_out_sets, device)

        # Build MoE + train router
        moe = ThreeExpertMoE(*[specialists[d] for d in DOMAINS]).to(device)
        train_router(moe, train_chunks, device)
        moe.eval()

        moe_eval     = eval_all_domains(moe, held_out_sets, device, is_fused=True)
        best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
        moe_eq       = moe_eval["equal_weight_avg"]

        entry = {
            "condition":  condition_name,
            "seed":       HET_SEED,
            "per_domain_config": per_domain_cfg,
            "moe_equal_weight":      moe_eq,
            "best_spec_equal_weight": best_spec_eq,
            "improvement_vs_spec":   pct_improvement(best_spec_eq, moe_eq),
            "per_domain_moe":   {k: v for k, v in moe_eval.items() if not k.startswith("_")},
            "per_domain_specs": {d: {k: v for k, v in e.items() if not k.startswith("_")}
                                 for d, e in spec_evals.items()},
            "eval_method": "per-domain-separate-equal-weight",
            "eval_batch_size": EVAL_BATCH,
        }
        results_list.append(entry)
        done_keys.add(condition_name)
        print(f"  => improvement_vs_spec: {entry['improvement_vs_spec']:+.2f}%")

        out = {"experiment": "heterogeneous_v2",
               "done_conditions": list(done_keys),
               "results": results_list}
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Saved: {out_path}")

        del moe
        for s in specialists.values(): del s
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("HETEROGENEOUS SUMMARY (v2, corrected eval)")
    print("="*70)
    control_imp = next(
        (r["improvement_vs_spec"] for r in results_list if r["condition"] == "control"), None)
    for r in sorted(results_list, key=lambda x: x["condition"]):
        delta = ""
        if control_imp is not None and r["condition"] != "control":
            d = r["improvement_vs_spec"] - control_imp
            delta = f"  (Δ vs control: {d:+.2f}pp)"
        print(f"  {r['condition']:>14}: {r['improvement_vs_spec']:+.2f}%{delta}")

    print("\n  Target: all conditions within 2pp of control")
    if control_imp is not None:
        for r in results_list:
            if r["condition"] == "control": continue
            d = abs(r["improvement_vs_spec"] - control_imp)
            status = "✓ PASS" if d <= 2.0 else "✗ FAIL"
            print(f"  {r['condition']:>14}: Δ={d:.2f}pp  {status}")

    return results_list

# ============================================================================
# Helpers
# ============================================================================

def _save(path, experiment_name, results_list):
    out = {"experiment": experiment_name, "results": results_list}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Saved: {path}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KALAVAI Phase 1 Retrain v2")
    parser.add_argument("--experiment", choices=["shared_init", "heterogeneous", "all"],
                        required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nKALAVAI Retrain v2")
    print(f"  experiment:      {args.experiment}")
    print(f"  eval_method:     per-domain equal-weight average (corrected protocol)")
    print(f"  eval_batch_size: {EVAL_BATCH}")
    print(f"  output dir:      {RESULTS_V2}")
    print(f"  device:          {device}")
    if device == "cuda":
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    print(f"\nLoading tokenizer ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision="step10000",
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_chunks, held_out_chunks = load_data(tokenizer)
    held_out_sets = {d: make_dataset(held_out_chunks[d]) for d in DOMAINS}

    experiments = (["shared_init", "heterogeneous"]
                   if args.experiment == "all"
                   else [args.experiment])

    t_start = time.time()
    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# Starting: {exp}")
        print(f"{'#'*70}")
        t0 = time.time()
        if exp == "shared_init":
            run_shared_init(train_chunks, held_out_sets, device)
        elif exp == "heterogeneous":
            run_heterogeneous(train_chunks, held_out_sets, device)
        print(f"\n  {exp} done in {(time.time()-t0)/60:.1f} min")

    print(f"\n{'='*70}")
    print(f"ALL DONE — total {(time.time()-t_start)/60:.1f} min")
    print(f"Results in: {RESULTS_V2.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
