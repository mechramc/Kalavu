#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Phase 1 Ablations v2
==============================
Re-evaluates all Phase 1 ablations using the corrected per-domain equal-weight protocol.
Loads existing checkpoints — no specialist retraining. Only router is trained fresh (500 steps).

Old results in results/pythia/ are PRESERVED. New results go to results/pythia/v2/.

Experiments:
  crossover    Training duration × freeze depth (12 combinations: 6 step counts × 2 freeze depths)
  router       Router architecture comparison (uniform / linear / mlp)
  freeze       Freeze depth sweep (0,2,4,6,8,12) with multi-seed where checkpoints exist
  monolithic   Equal-compute monolithic and Pythia-1.4B wider model baselines
  classifier   Single-expert dispatch vs soft MoE (catastrophic forgetting analysis)
  five_domain  N-specialist scaling (2→5 experts: code/science/fiction/math/multilingual)
  maturity     Base model maturity sweep (410M: 6 checkpoints; 1B: 4 checkpoints)
  hard_routing Soft vs hard (argmax) routing with the same trained router
  benchmarks   Downstream NLP accuracy benchmarks (fresh router, log-likelihood scoring)
  inference    Throughput/VRAM/latency for base, specialist, MoE configurations
  all          Run all experiments in sequence

Usage:
  cd /c/Github/Kalavai
  python experiments/kalavai_ablations_v2.py --experiment five_domain
  python experiments/kalavai_ablations_v2.py --experiment maturity
  python experiments/kalavai_ablations_v2.py --experiment hard_routing
  python experiments/kalavai_ablations_v2.py --experiment benchmarks
  python experiments/kalavai_ablations_v2.py --experiment inference
  python experiments/kalavai_ablations_v2.py --experiment all
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Constants
# ============================================================================

MODEL_ID   = "EleutherAI/pythia-410m"
MODEL_ID_WIDE = "EleutherAI/pythia-1.4b"
REVISION   = "step10000"
HIDDEN_SIZE = 1024
SEQ_LEN    = 512
DOMAINS    = ["code", "science", "fiction"]
N_SAMPLES  = 3000        # per domain, 80/10/10 split
EVAL_BATCH = 4
EVAL_BATCHES = 50

ROUTER_STEPS = 500
ROUTER_LR    = 1e-3
ROUTER_BATCH = 4

CKPT_DIR    = Path("checkpoints/pythia")
RESULTS_V2  = Path("results/pythia/v2")

# Freeze depth sweep config
FREEZE_DEPTHS  = [0, 2, 4, 6, 8, 12]
# Which depths have multi-seed checkpoints (besides seed=42)
FREEZE_MULTISEED = {0: [42, 137, 2026], 2: [42, 137, 2026]}
FREEZE_SEED42_ONLY = [4, 6, 8, 12]

# Crossover step sweep
CROSSOVER_STEPS  = [500, 1000, 2000, 5000, 10000, 20000]
CROSSOVER_FREEZE = [0, 4]

# Five-domain
FIVE_CKPT_DIR = Path("checkpoints/pythia/five_domain")
DOMAINS_5     = ["code", "science", "fiction", "math", "multilingual"]
SUBSETS_5 = {
    "2_specialists": ["code", "fiction"],
    "3_specialists": ["code", "science", "fiction"],
    "4_specialists": ["code", "science", "fiction", "math"],
    "5_specialists": ["code", "science", "fiction", "math", "multilingual"],
}
SEEDS_5 = [42, 137, 2026]

# Maturity sweep
MATURITY_410M_DIR = Path("checkpoints/pythia/maturity_sweep_410m")
MATURITY_1B_DIR   = Path("checkpoints/pythia/pythia_1b/maturity_sweep")
MATURITY_410M_STEPS = ["step5000", "step10000", "step20000", "step50000", "step100000", "step143000"]
MATURITY_1B_STEPS   = ["step5000", "step20000", "step50000", "step143000"]
# Revisions have multi-seed checkpoints
MATURITY_410M_MULTISEED = {"step5000": [42, 137, 2026], "step10000": [42, 137, 2026],
                            "step20000": [42, 137, 2026]}

# Benchmarks
BENCHMARKS    = ["hellaswag", "arc_easy", "piqa", "winogrande", "lambada_openai", "sciq"]
RANDOM_CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "piqa": 0.50,
                 "winogrande": 0.50, "lambada_openai": 0.00, "sciq": 0.25}

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
# Data loading (matches kalavai_corrected_eval.py exactly)
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
    """Load all 3 domains, pack + split 80/10/10. Returns (train_chunks, held_out_chunks)."""
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
# Additional data loaders (math + multilingual for five_domain)
# ============================================================================

def load_math_texts(n):
    from datasets import load_dataset
    print(f"  Loading math/gsm8k (n={n})...")
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item["question"] + "\n" + item["answer"]
        if len(content) > 50:
            texts.append(content)
        if len(texts) >= n: break
    print(f"    {len(texts)} samples")
    return texts


def load_multilingual_texts(n):
    from datasets import load_dataset
    for lang_config in ["20231101.es", "20231101.fr", "20231101.de"]:
        try:
            print(f"  Loading multilingual/wikipedia ({lang_config}, n={n})...")
            ds = load_dataset("wikimedia/wikipedia", lang_config, split="train", streaming=True)
            texts = []
            for item in ds:
                content = item["text"][:3000]
                if len(content) >= 500:
                    texts.append(content)
                if len(texts) >= n: break
            if texts:
                print(f"    {len(texts)} samples ({lang_config})")
                return texts
        except Exception as e:
            print(f"  {lang_config} failed ({e}), trying next...")
    raise RuntimeError("All multilingual dataset options exhausted")


def load_5domain_data(tokenizer, n=N_SAMPLES):
    """Load all 5 domains, pack + split 80/10/10."""
    print("\nLoading 5-domain data...")
    raw = {
        "code":         load_code_texts(n),
        "science":      load_science_texts(n),
        "fiction":      load_fiction_texts(n),
        "math":         load_math_texts(n),
        "multilingual": load_multilingual_texts(n),
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
# MoE architectures (must match originals exactly)
# ============================================================================

class ThreeExpertMoE_MLP(nn.Module):
    """410M MoE with 2-layer MLP router. Matches main experiment."""
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


class ThreeExpertMoE_Linear(nn.Module):
    """MoE with single linear router. hidden_size inferred from spec config if not provided."""
    def __init__(self, spec_a, spec_b, spec_c, hidden_size=None):
        super().__init__()
        self.spec_a, self.spec_b, self.spec_c = spec_a, spec_b, spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)
        if hidden_size is None:
            hidden_size = getattr(spec_a.config, "hidden_size", HIDDEN_SIZE)
        self.router = nn.Linear(hidden_size, 3, bias=False)

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


class ThreeExpertMoE_Uniform(nn.Module):
    """Uniform 1/3 weighting — no router."""
    def __init__(self, spec_a, spec_b, spec_c):
        super().__init__()
        self.spec_a, self.spec_b, self.spec_c = spec_a, spec_b, spec_c
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
        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, None

class FlexibleMoE(nn.Module):
    """N-expert MoE with linear router. Used for five_domain subsets."""
    def __init__(self, specialist_list):
        super().__init__()
        self.specs = nn.ModuleList(specialist_list)
        self.n = len(specialist_list)
        for spec in self.specs:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(HIDDEN_SIZE, self.n, bias=False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        logits_list, hidden_list = [], []
        for spec in self.specs:
            lg, h = self._run(spec, input_ids)
            logits_list.append(lg)
            hidden_list.append(h)
        h_avg = sum(hidden_list) / len(hidden_list)
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(gates[:, i:i+1, None] * logits_list[i] for i in range(self.n))
        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, gates


def train_flexible_router(moe, train_chunks_by_domain, device,
                          steps=ROUTER_STEPS, lr=ROUTER_LR, bs=ROUTER_BATCH):
    """Train router for FlexibleMoE (no .router attribute path differs)."""
    all_chunks = []
    for chunks in train_chunks_by_domain.values():
        all_chunks.extend(chunks)
    combined = make_dataset(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=lr)
    loader = DataLoader(combined, batch_size=bs, shuffle=True, drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({steps} steps, {len(combined)} chunks, {moe.n} experts)...")
    for step in range(1, steps + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            print(f"    step {step}/{steps}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Evaluation — corrected protocol
# ============================================================================

@torch.no_grad()
def eval_loss_domain(model, dataset, device, bs=EVAL_BATCH, n_batches=EVAL_BATCHES, is_fused=False):
    """Single domain at fixed batch size."""
    loader = DataLoader(dataset, batch_size=bs, shuffle=False,
                        drop_last=True, collate_fn=_collate)
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
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


def eval_all_domains(model, held_out_sets, device, bs=EVAL_BATCH, n_batches=EVAL_BATCHES,
                     is_fused=False, label=""):
    """Per-domain separate eval + equal-weight average."""
    losses = {}
    for domain, ds in held_out_sets.items():
        t0 = time.time()
        loss = eval_loss_domain(model, ds, device, bs, n_batches, is_fused)
        losses[domain] = round(loss, 6)
        print(f"    {label + ' ' if label else ''}{domain:8s}: {loss:.4f}  ({time.time()-t0:.1f}s)")
    losses["equal_weight_avg"] = round(sum(losses[d] for d in held_out_sets) / len(held_out_sets), 6)
    return losses

# ============================================================================
# Router training
# ============================================================================

def train_router(moe, train_chunks, device, steps=ROUTER_STEPS, lr=ROUTER_LR, bs=ROUTER_BATCH):
    all_chunks = []
    for chunks in train_chunks.values():
        all_chunks.extend(chunks)
    combined = make_dataset(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=lr)
    loader = DataLoader(combined, batch_size=bs, shuffle=True, drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    print(f"  Training router ({steps} steps, {len(combined)} chunks)...")
    for step in range(1, steps + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps:
            print(f"    step {step}/{steps}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Checkpoint loading helpers
# ============================================================================

def load_base_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model


def load_from_ckpt(ckpt_path: Path, model_id: str, revision: str, device):
    """Load any Pythia checkpoint into the correct architecture."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_specialist(domain: str, seed: int, device):
    """Load main-experiment specialist: {domain}_specialist_seed{seed}.pt"""
    path = CKPT_DIR / f"{domain}_specialist_seed{seed}.pt"
    return load_from_ckpt(path, MODEL_ID, REVISION, device)


def load_freeze_specialist(domain: str, freeze: int, seed: int, device):
    """
    Load freeze-ablation specialist.
    freeze=4/seed=42 → main experiment checkpoints (same as load_specialist).
    Others → freeze{N}_{domain}_seed{seed}.pt
    """
    if freeze == 4 and seed == 42:
        return load_specialist(domain, seed, device)
    path = CKPT_DIR / f"freeze{freeze}_{domain}_seed{seed}.pt"
    return load_from_ckpt(path, MODEL_ID, REVISION, device)


def load_crossover_specialist(domain: str, freeze: int, steps: int, device):
    """Load crossover checkpoint: crossover_{domain}_freeze{freeze}_steps{steps}_seed42.pt"""
    path = CKPT_DIR / f"crossover_{domain}_freeze{freeze}_steps{steps}_seed42.pt"
    return load_from_ckpt(path, MODEL_ID, REVISION, device)


def load_5domain_specialist(domain: str, seed: int, device):
    """Load five_domain specialist: checkpoints/pythia/five_domain/{domain}_seed{N}.pt"""
    path = FIVE_CKPT_DIR / f"{domain}_seed{seed}.pt"
    return load_from_ckpt(path, MODEL_ID, REVISION, device)


def load_maturity_specialist_410m(domain: str, step_rev: str, seed: int, device):
    """Load 410M maturity sweep specialist: maturity_sweep_410m/{step}/{domain}_seed{N}.pt"""
    path = MATURITY_410M_DIR / step_rev / f"{domain}_seed{seed}.pt"
    return load_from_ckpt(path, MODEL_ID, step_rev, device)


def load_maturity_base_410m(step_rev: str, device):
    """Load 410M base model at given revision."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=step_rev, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model


def load_maturity_specialist_1b(domain: str, step_rev: str, device):
    """Load 1B maturity sweep specialist: pythia_1b/maturity_sweep/{step}/{domain}_specialist_seed42.pt"""
    path = MATURITY_1B_DIR / step_rev / f"{domain}_specialist_seed42.pt"
    return load_from_ckpt(path, "EleutherAI/pythia-1b", step_rev, device)


def load_maturity_base_1b(step_rev: str, device):
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b", revision=step_rev, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model

# ============================================================================
# Improvement metric helpers
# ============================================================================

def pct_improvement(before, after):
    """Positive = after is better (lower loss)."""
    return round((before - after) / before * 100, 4)

# ============================================================================
# Experiment: crossover
# ============================================================================

def run_crossover(train_chunks, held_out_sets, device):
    """
    Re-evaluates training duration × freeze depth crossover.
    Loads existing checkpoints, trains a fresh router, evals with corrected protocol.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: crossover (training duration × freeze depth)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "crossover_v2.json"

    # Load incremental results if they exist (for resumption)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_keys = {(r["steps"], r["freeze"]) for r in existing.get("results", [])}
        results_list = existing.get("results", [])
        print(f"  Resuming: {len(done_keys)} combinations already done")
    else:
        done_keys = set()
        results_list = []

    for freeze in CROSSOVER_FREEZE:
        for steps in CROSSOVER_STEPS:
            key = (steps, freeze)
            if key in done_keys:
                print(f"\n[crossover] steps={steps}, freeze={freeze} — already done, skipping")
                continue

            print(f"\n[crossover] steps={steps}, freeze={freeze}")

            # Check all 3 checkpoints exist
            ckpts = [CKPT_DIR / f"crossover_{d}_freeze{freeze}_steps{steps}_seed42.pt" for d in DOMAINS]
            missing = [c for c in ckpts if not c.exists()]
            if missing:
                print(f"  MISSING checkpoints: {[c.name for c in missing]}")
                print(f"  Skipping this combination.")
                continue

            t0 = time.time()
            specs = [load_crossover_specialist(d, freeze, steps, device) for d in DOMAINS]
            print(f"  Loaded specialists ({time.time()-t0:.1f}s)")

            # MoE with MLP router (matches main 410M experiment)
            moe = ThreeExpertMoE_MLP(*specs).to(device)
            train_router(moe, train_chunks, device)
            moe.eval()

            print("  Evaluating MoE...")
            moe_eval = eval_all_domains(moe, held_out_sets, device, is_fused=True)

            # Also eval individual specialists for best-spec baseline
            spec_evals = {}
            for domain, spec in zip(DOMAINS, specs):
                print(f"  Evaluating {domain}_spec...")
                spec_evals[domain] = eval_all_domains(spec, held_out_sets, device)

            best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
            moe_eq = moe_eval["equal_weight_avg"]

            entry = {
                "steps":             steps,
                "freeze":            freeze,
                "moe_equal_weight":  moe_eq,
                "best_spec_equal_weight": best_spec_eq,
                "improvement_vs_spec": pct_improvement(best_spec_eq, moe_eq),
                "per_domain_moe":    {k: v for k, v in moe_eval.items() if not k.startswith("_")},
                "per_domain_specs":  {d: {k: v for k, v in e.items() if not k.startswith("_")}
                                      for d, e in spec_evals.items()},
                "eval_method": "per-domain-separate-equal-weight",
                "eval_batch_size": EVAL_BATCH,
            }
            results_list.append(entry)
            print(f"  => improvement_vs_spec: {entry['improvement_vs_spec']:+.2f}%  "
                  f"(moe={moe_eq:.4f}, best_spec={best_spec_eq:.4f})")

            # Save incrementally
            out = {"experiment": "crossover_v2", "results": results_list}
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"  Saved: {out_path}")

            del moe
            for s in specs: del s
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "="*70)
    print("CROSSOVER SUMMARY (v2, corrected eval)")
    print("="*70)
    print(f"{'steps':>8}  {'freeze':>8}  {'vs spec':>10}  {'moe_eq':>8}")
    print("-" * 42)
    for r in sorted(results_list, key=lambda x: (x["steps"], x["freeze"])):
        print(f"{r['steps']:>8}  {r['freeze']:>8}  "
              f"{r['improvement_vs_spec']:>+9.2f}%  {r['moe_equal_weight']:>8.4f}")

    return results_list

# ============================================================================
# Experiment: router
# ============================================================================

def run_router(train_chunks, held_out_sets, device):
    """
    Router architecture comparison: Uniform / Linear / MLP.
    Loads seed=42 main specialists, tests 3 router variants.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: router (architecture comparison)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "router_ablation_v2.json"

    # Load seed=42 specialists
    print("\nLoading seed=42 specialists...")
    specs = [load_specialist(d, 42, device) for d in DOMAINS]
    spec_evals = {}
    for domain, spec in zip(DOMAINS, specs):
        print(f"\n[{domain}_spec]")
        spec_evals[domain] = eval_all_domains(spec, held_out_sets, device, label=domain)

    best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
    print(f"\n  Best specialist equal-weight avg: {best_spec_eq:.4f}")

    results = {
        "experiment": "router_ablation_v2",
        "eval_method": "per-domain-separate-equal-weight",
        "eval_batch_size": EVAL_BATCH,
        "per_domain_specs": {d: e for d, e in spec_evals.items()},
        "best_spec_equal_weight": best_spec_eq,
        "router_variants": {},
    }

    # Variant A: Uniform (no router)
    print("\n[uniform router]")
    moe_uniform = ThreeExpertMoE_Uniform(*specs).to(device)
    uniform_eval = eval_all_domains(moe_uniform, held_out_sets, device, is_fused=True, label="uniform")
    results["router_variants"]["uniform"] = {
        "eval": {k: v for k, v in uniform_eval.items() if not k.startswith("_")},
        "improvement_vs_spec": pct_improvement(best_spec_eq, uniform_eval["equal_weight_avg"]),
    }
    print(f"  improvement_vs_spec: {results['router_variants']['uniform']['improvement_vs_spec']:+.2f}%")
    del moe_uniform; torch.cuda.empty_cache()

    # Variant B: Linear router
    print("\n[linear router]")
    moe_linear = ThreeExpertMoE_Linear(*specs).to(device)
    train_router(moe_linear, train_chunks, device)
    moe_linear.eval()
    linear_eval = eval_all_domains(moe_linear, held_out_sets, device, is_fused=True, label="linear")
    results["router_variants"]["linear"] = {
        "eval": {k: v for k, v in linear_eval.items() if not k.startswith("_")},
        "improvement_vs_spec": pct_improvement(best_spec_eq, linear_eval["equal_weight_avg"]),
    }
    print(f"  improvement_vs_spec: {results['router_variants']['linear']['improvement_vs_spec']:+.2f}%")
    del moe_linear; torch.cuda.empty_cache()

    # Variant C: MLP router (matches main experiment)
    print("\n[mlp router (main)]")
    moe_mlp = ThreeExpertMoE_MLP(*specs).to(device)
    train_router(moe_mlp, train_chunks, device)
    moe_mlp.eval()
    mlp_eval = eval_all_domains(moe_mlp, held_out_sets, device, is_fused=True, label="mlp")
    results["router_variants"]["mlp"] = {
        "eval": {k: v for k, v in mlp_eval.items() if not k.startswith("_")},
        "improvement_vs_spec": pct_improvement(best_spec_eq, mlp_eval["equal_weight_avg"]),
    }
    print(f"  improvement_vs_spec: {results['router_variants']['mlp']['improvement_vs_spec']:+.2f}%")
    del moe_mlp; torch.cuda.empty_cache()

    for s in specs: del s
    torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n" + "="*70)
    print("ROUTER SUMMARY (v2, corrected eval)")
    print("="*70)
    print(f"{'Variant':<12}  {'eq_avg':>8}  {'vs_spec':>10}")
    print("-" * 34)
    print(f"{'best_spec':<12}  {best_spec_eq:>8.4f}  {'(baseline)':>10}")
    for name, data in results["router_variants"].items():
        eq = data["eval"]["equal_weight_avg"]
        imp = data["improvement_vs_spec"]
        print(f"{name:<12}  {eq:>8.4f}  {imp:>+9.2f}%")

    return results

# ============================================================================
# Experiment: freeze
# ============================================================================

def run_freeze(train_chunks, held_out_sets, device):
    """
    Freeze depth sweep: [0, 2, 4, 6, 8, 12] with seed=42.
    Multi-seed (42, 137, 2026) for freeze=0 and freeze=2.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: freeze (depth sweep)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "freeze_ablation_v2.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_keys = {(r["freeze"], r["seed"]) for r in existing.get("results", [])}
        results_list = existing.get("results", [])
        print(f"  Resuming: {len(done_keys)} (freeze, seed) combos already done")
    else:
        done_keys = set()
        results_list = []

    # Build (freeze, seed) job list
    jobs = []
    for freeze in FREEZE_DEPTHS:
        if freeze in FREEZE_MULTISEED:
            for seed in FREEZE_MULTISEED[freeze]:
                jobs.append((freeze, seed))
        else:
            jobs.append((freeze, 42))

    for freeze, seed in jobs:
        key = (freeze, seed)
        if key in done_keys:
            print(f"\n[freeze={freeze}, seed={seed}] — already done, skipping")
            continue

        print(f"\n[freeze={freeze}, seed={seed}]")

        # Verify checkpoints
        if freeze == 4 and seed == 42:
            ckpt_names = [f"{d}_specialist_seed42.pt" for d in DOMAINS]
        else:
            ckpt_names = [f"freeze{freeze}_{d}_seed{seed}.pt" for d in DOMAINS]
        ckpts = [CKPT_DIR / n for n in ckpt_names]
        missing = [c for c in ckpts if not c.exists()]
        if missing:
            print(f"  MISSING: {[c.name for c in missing]}")
            print(f"  Skipping.")
            continue

        specs = [load_freeze_specialist(d, freeze, seed, device) for d in DOMAINS]

        # MoE with MLP router
        moe = ThreeExpertMoE_MLP(*specs).to(device)
        train_router(moe, train_chunks, device)
        moe.eval()

        print("  Evaluating MoE...")
        moe_eval = eval_all_domains(moe, held_out_sets, device, is_fused=True)

        spec_evals = {}
        for domain, spec in zip(DOMAINS, specs):
            spec_evals[domain] = eval_all_domains(spec, held_out_sets, device)

        best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
        moe_eq = moe_eval["equal_weight_avg"]

        entry = {
            "freeze":    freeze,
            "seed":      seed,
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
        print(f"  => improvement_vs_spec: {entry['improvement_vs_spec']:+.2f}%  "
              f"(moe={moe_eq:.4f}, best_spec={best_spec_eq:.4f})")

        out = {"experiment": "freeze_ablation_v2", "results": results_list}
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Saved: {out_path}")

        del moe
        for s in specs: del s
        torch.cuda.empty_cache()

    # Aggregate multi-seed entries
    print("\n" + "="*70)
    print("FREEZE SUMMARY (v2, corrected eval)")
    print("="*70)
    by_freeze = {}
    for r in results_list:
        by_freeze.setdefault(r["freeze"], []).append(r)

    print(f"{'freeze':>8}  {'seeds':>12}  {'vs spec (mean)':>16}  {'std':>8}")
    print("-" * 52)
    for freeze in sorted(by_freeze.keys()):
        entries = by_freeze[freeze]
        imps = [e["improvement_vs_spec"] for e in entries]
        seeds_str = ",".join(str(e["seed"]) for e in entries)
        import statistics as st
        mean_imp = st.mean(imps)
        std_imp  = st.stdev(imps) if len(imps) > 1 else 0.0
        print(f"{freeze:>8}  {seeds_str:>12}  {mean_imp:>+15.2f}%  {std_imp:>7.3f}%")

    return results_list

# ============================================================================
# Experiment: monolithic
# ============================================================================

def run_monolithic(train_chunks, held_out_sets, device):
    """
    Equal-compute monolithic and wider model baselines.
    Compares: base | monolithic (seed42) | wider (pythia-1.4b) | MoE (seed42)
    """
    print("\n" + "="*70)
    print("EXPERIMENT: monolithic (equal-compute + wider model baselines)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "monolithic_v2.json"

    results = {
        "experiment": "monolithic_v2",
        "eval_method": "per-domain-separate-equal-weight",
        "eval_batch_size": EVAL_BATCH,
        "models": {},
    }

    # Base model
    print("\n[base]")
    base = load_base_model(device)
    results["models"]["base"] = eval_all_domains(base, held_out_sets, device)
    del base; torch.cuda.empty_cache()

    # Monolithic (equal compute, seed=42)
    mono_path = CKPT_DIR / "monolithic_seed42.pt"
    if mono_path.exists():
        print(f"\n[monolithic seed=42]  {mono_path.name}")
        mono = load_from_ckpt(mono_path, MODEL_ID, REVISION, device)
        results["models"]["monolithic_seed42"] = eval_all_domains(mono, held_out_sets, device)
        del mono; torch.cuda.empty_cache()
    else:
        print(f"\n[monolithic seed=42]  NOT FOUND — skipping")

    # Wider model (Pythia-1.4B trained on mixed data)
    wider_path = CKPT_DIR / "wider_1b4_seed42.pt"
    if wider_path.exists():
        print(f"\n[wider model pythia-1.4b]  {wider_path.name}")
        wider = load_from_ckpt(wider_path, MODEL_ID_WIDE, REVISION, device)
        results["models"]["wider_1b4_seed42"] = eval_all_domains(wider, held_out_sets, device)
        del wider; torch.cuda.empty_cache()
    else:
        print(f"\n[wider model]  NOT FOUND — skipping")

    # MoE (seed=42, for direct comparison)
    print("\n[moe seed=42]")
    specs = [load_specialist(d, 42, device) for d in DOMAINS]
    spec_evals = {}
    for domain, spec in zip(DOMAINS, specs):
        spec_evals[domain] = eval_all_domains(spec, held_out_sets, device, label=domain)
    results["models"].update({f"{d}_spec_seed42": e for d, e in spec_evals.items()})

    moe = ThreeExpertMoE_MLP(*specs).to(device)
    train_router(moe, train_chunks, device)
    moe.eval()
    results["models"]["moe_seed42"] = eval_all_domains(moe, held_out_sets, device, is_fused=True)
    del moe
    for s in specs: del s
    torch.cuda.empty_cache()

    # Compute improvement metrics
    best_spec_eq = min(results["models"][f"{d}_spec_seed42"]["equal_weight_avg"] for d in DOMAINS)
    moe_eq       = results["models"]["moe_seed42"]["equal_weight_avg"]
    results["metrics"] = {
        "moe_vs_best_spec":  pct_improvement(best_spec_eq, moe_eq),
    }
    if "monolithic_seed42" in results["models"]:
        mono_eq = results["models"]["monolithic_seed42"]["equal_weight_avg"]
        results["metrics"]["moe_vs_monolithic"] = pct_improvement(mono_eq, moe_eq)
        results["metrics"]["monolithic_vs_best_spec"] = pct_improvement(best_spec_eq, mono_eq)
    if "wider_1b4_seed42" in results["models"]:
        wider_eq = results["models"]["wider_1b4_seed42"]["equal_weight_avg"]
        results["metrics"]["moe_vs_wider"]    = pct_improvement(wider_eq, moe_eq)
        results["metrics"]["wider_vs_best_spec"] = pct_improvement(best_spec_eq, wider_eq)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n" + "="*70)
    print("MONOLITHIC SUMMARY (v2, corrected eval)")
    print("="*70)
    print(f"{'Model':<28}  {'eq_avg':>8}  {'vs_moe':>10}")
    print("-" * 52)
    for name, data in results["models"].items():
        eq = data["equal_weight_avg"]
        vs_moe = pct_improvement(moe_eq, eq)  # positive = model is better than moe (unlikely)
        print(f"{name:<28}  {eq:>8.4f}  {vs_moe:>+9.2f}%")
    print(f"\n  MoE vs best spec: {results['metrics']['moe_vs_best_spec']:+.2f}%")

    return results

# ============================================================================
# Experiment: classifier
# ============================================================================

def run_classifier(train_chunks, held_out_sets, device):
    """
    Single-expert dispatch vs soft MoE.

    Tests: what happens if you route to ONE expert for ALL tokens?
    - Oracle dispatch: code→code_spec, science→science_spec, fiction→fiction_spec (best possible)
    - Fixed dispatch: always use code_spec for everything (worst specialist dispatch)
    - Per-domain specialist matrix: each specialist evaluated on each domain

    The key finding: soft MoE running all specialists concurrently outperforms
    any single-expert dispatch, even oracle dispatch.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: classifier (single-expert dispatch analysis)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "classifier_v2.json"

    # Load seed=42 specialists (and 137, 2026 for context)
    results = {
        "experiment": "classifier_v2",
        "eval_method": "per-domain-separate-equal-weight",
        "eval_batch_size": EVAL_BATCH,
        "cross_domain_matrix": {},  # per specialist per domain
        "dispatch_scenarios": {},
        "moe_results": {},
    }

    print("\nLoading specialists (seeds 42, 137, 2026)...")

    # Full cross-domain matrix for seed=42
    specs_42 = [load_specialist(d, 42, device) for d in DOMAINS]
    for domain, spec in zip(DOMAINS, specs_42):
        print(f"\n[{domain}_spec seed=42] — all domains")
        results["cross_domain_matrix"][f"{domain}_spec_seed42"] = eval_all_domains(
            spec, held_out_sets, device, label=f"{domain}_s42"
        )

    # Oracle dispatch: each domain routed to its own specialist
    # (equal_weight_avg of: code_spec[code], science_spec[science], fiction_spec[fiction])
    oracle_losses = {
        d: results["cross_domain_matrix"][f"{d}_spec_seed42"][d]
        for d in DOMAINS
    }
    oracle_eq = round(sum(oracle_losses.values()) / len(oracle_losses), 6)
    results["dispatch_scenarios"]["oracle_per_domain"] = {
        "description": "Route each domain to its own specialist (oracle domain classifier)",
        "per_domain_losses": oracle_losses,
        "equal_weight_avg": oracle_eq,
    }
    print(f"\n  Oracle dispatch eq_avg: {oracle_eq:.4f}")
    print(f"  (diagonal of cross-domain matrix: "
          + ", ".join(f"{d}→{d}_spec={oracle_losses[d]:.4f}" for d in DOMAINS) + ")")

    # Fixed dispatch: always route to each specialist and eval ALL domains
    for i, (domain, spec) in enumerate(zip(DOMAINS, specs_42)):
        eq = results["cross_domain_matrix"][f"{domain}_spec_seed42"]["equal_weight_avg"]
        results["dispatch_scenarios"][f"always_{domain}_spec"] = {
            "description": f"Route ALL data to {domain} specialist (fixed dispatch)",
            "equal_weight_avg": eq,
            "per_domain_losses": {d: results["cross_domain_matrix"][f"{domain}_spec_seed42"][d]
                                  for d in DOMAINS},
        }
    print("\n  Fixed-dispatch (always route to one specialist):")
    for d in DOMAINS:
        eq = results["dispatch_scenarios"][f"always_{d}_spec"]["equal_weight_avg"]
        print(f"    always_{d}_spec: eq_avg={eq:.4f}")

    # MoE (soft routing, all specialists run concurrently)
    print("\n[moe seed=42]")
    moe = ThreeExpertMoE_MLP(*specs_42).to(device)
    train_router(moe, train_chunks, device)
    moe.eval()
    moe_eval = eval_all_domains(moe, held_out_sets, device, is_fused=True, label="moe")
    results["moe_results"]["seed42"] = {k: v for k, v in moe_eval.items() if not k.startswith("_")}
    del moe; torch.cuda.empty_cache()

    # Metrics
    moe_eq = moe_eval["equal_weight_avg"]
    best_spec_eq = min(results["cross_domain_matrix"][f"{d}_spec_seed42"]["equal_weight_avg"]
                       for d in DOMAINS)
    results["metrics"] = {
        "moe_eq":           moe_eq,
        "best_spec_eq":     best_spec_eq,
        "oracle_dispatch_eq": oracle_eq,
        "moe_vs_best_spec":   pct_improvement(best_spec_eq, moe_eq),
        "moe_vs_oracle_dispatch": pct_improvement(oracle_eq, moe_eq),
        "best_fixed_dispatch_eq": min(
            results["dispatch_scenarios"][f"always_{d}_spec"]["equal_weight_avg"] for d in DOMAINS
        ),
    }
    results["metrics"]["moe_vs_best_fixed_dispatch"] = pct_improvement(
        results["metrics"]["best_fixed_dispatch_eq"], moe_eq
    )

    del specs_42
    torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Print summary
    print("\n" + "="*70)
    print("CLASSIFIER / DISPATCH SUMMARY (v2, corrected eval)")
    print("="*70)
    m = results["metrics"]
    print(f"  MoE (soft routing):           {m['moe_eq']:.4f}")
    print(f"  Best specialist:              {m['best_spec_eq']:.4f}  (MoE: {m['moe_vs_best_spec']:+.2f}%)")
    print(f"  Oracle dispatch (per-domain): {m['oracle_dispatch_eq']:.4f}  "
          f"(MoE: {m['moe_vs_oracle_dispatch']:+.2f}%)")
    print(f"  Best fixed dispatch:          {m['best_fixed_dispatch_eq']:.4f}  "
          f"(MoE: {m['moe_vs_best_fixed_dispatch']:+.2f}%)")
    print()
    print("  Cross-domain matrix (each specialist on each held-out domain):")
    print(f"  {'':24} {'code':>8} {'science':>10} {'fiction':>10} {'eq_avg':>8}")
    print(f"  {'-'*64}")
    for d in DOMAINS:
        key = f"{d}_spec_seed42"
        row = results["cross_domain_matrix"][key]
        print(f"  {key:<24} {row['code']:>8.4f} {row['science']:>10.4f} "
              f"{row['fiction']:>10.4f} {row['equal_weight_avg']:>8.4f}")

    return results

# ============================================================================
# Experiment: five_domain
# ============================================================================

def run_five_domain(tokenizer, device):
    """
    5-domain specialist scaling: subsets 2→5 experts, 3 seeds each.
    Data is loaded fresh (5 domains). Checkpoints loaded from five_domain dir.
    Equal-weight avg across ALL 5 held-out domains regardless of subset size.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: five_domain (N-specialist scaling)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "five_domain_v2.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_keys = {(r["subset"], r["seed"]) for r in existing.get("results", [])}
        results_list = existing.get("results", [])
        print(f"  Resuming: {len(done_keys)} (subset, seed) combos already done")
    else:
        done_keys = set()
        results_list = []

    # Load 5-domain data once
    train_chunks, held_out_chunks = load_5domain_data(tokenizer)
    held_out_sets_5 = {d: make_dataset(held_out_chunks[d]) for d in DOMAINS_5}

    # Also eval base model once for reference
    base_key = ("_base", 0)
    if base_key not in done_keys:
        print("\n[base model]")
        base = load_base_model(device)
        base_eval = eval_all_domains(base, held_out_sets_5, device)
        results_list.append({"subset": "_base", "seed": 0,
                              "eval": {k: v for k, v in base_eval.items() if not k.startswith("_")}})
        del base; torch.cuda.empty_cache()
        out = {"experiment": "five_domain_v2", "results": results_list}
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for subset_name, domains_in_subset in SUBSETS_5.items():
        for seed in SEEDS_5:
            key = (subset_name, seed)
            if key in done_keys:
                print(f"\n[{subset_name}, seed={seed}] — already done, skipping")
                continue

            # Check all checkpoints exist
            ckpts = [FIVE_CKPT_DIR / f"{d}_seed{seed}.pt" for d in domains_in_subset]
            missing = [c for c in ckpts if not c.exists()]
            if missing:
                print(f"\n[{subset_name}, seed={seed}] MISSING: {[c.name for c in missing]}")
                continue

            print(f"\n[{subset_name}, seed={seed}]")
            specs = [load_5domain_specialist(d, seed, device) for d in domains_in_subset]

            moe = FlexibleMoE(specs).to(device)
            train_chunks_subset = {d: train_chunks[d] for d in domains_in_subset}
            train_flexible_router(moe, train_chunks_subset, device)
            moe.eval()

            print("  Evaluating MoE (all 5 held-out domains)...")
            moe_eval = eval_all_domains(moe, held_out_sets_5, device, is_fused=True)

            spec_evals = {}
            for domain, spec in zip(domains_in_subset, specs):
                spec_evals[domain] = eval_all_domains(spec, held_out_sets_5, device)

            best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
            moe_eq = moe_eval["equal_weight_avg"]

            entry = {
                "subset":    subset_name,
                "n_experts": len(domains_in_subset),
                "domains":   domains_in_subset,
                "seed":      seed,
                "moe_equal_weight":      moe_eq,
                "best_spec_equal_weight": best_spec_eq,
                "improvement_vs_spec":   pct_improvement(best_spec_eq, moe_eq),
                "per_domain_moe":   {k: v for k, v in moe_eval.items() if not k.startswith("_")},
                "per_domain_specs": {d: {k: v for k, v in e.items() if not k.startswith("_")}
                                     for d, e in spec_evals.items()},
                "eval_method": "per-domain-separate-equal-weight-5domains",
                "eval_batch_size": EVAL_BATCH,
            }
            results_list.append(entry)
            print(f"  => {subset_name}/seed={seed}: {entry['improvement_vs_spec']:+.2f}%")

            out = {"experiment": "five_domain_v2", "results": results_list}
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"  Saved: {out_path}")

            del moe
            for s in specs: del s
            torch.cuda.empty_cache()

    # Summary by subset (mean over seeds)
    print("\n" + "="*70)
    print("FIVE_DOMAIN SUMMARY (v2, corrected eval)")
    print("="*70)
    import statistics as st
    by_subset = {}
    for r in results_list:
        if r["subset"].startswith("_"): continue
        by_subset.setdefault(r["subset"], []).append(r)
    print(f"{'subset':>20}  {'n':>4}  {'seeds':>12}  {'vs spec (mean)':>16}  {'std':>8}")
    print("-" * 66)
    for sn in ["2_specialists", "3_specialists", "4_specialists", "5_specialists"]:
        if sn not in by_subset: continue
        entries = by_subset[sn]
        imps = [e["improvement_vs_spec"] for e in entries]
        seeds_str = ",".join(str(e["seed"]) for e in entries)
        n = entries[0]["n_experts"]
        mean_imp = st.mean(imps)
        std_imp  = st.stdev(imps) if len(imps) > 1 else 0.0
        print(f"{sn:>20}  {n:>4}  {seeds_str:>12}  {mean_imp:>+15.2f}%  {std_imp:>7.3f}%")

    return results_list

# ============================================================================
# Experiment: maturity
# ============================================================================

def run_maturity(train_chunks, held_out_sets, device):
    """
    Base model maturity sweep: how fusion gain varies with checkpoint age.
    410M: 6 checkpoints (step5k→143k), multi-seed at step5k/10k/20k.
    1B:  4 checkpoints (step5k/20k/50k/143k), seed=42 only.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: maturity (base model maturity sweep)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path_410m = RESULTS_V2 / "maturity_sweep_410m_v2.json"
    out_path_1b   = RESULTS_V2 / "maturity_sweep_1b_v2.json"

    # --- 410M ---
    print("\n--- 410M maturity sweep ---")
    if out_path_410m.exists():
        existing = json.loads(out_path_410m.read_text())
        done_keys_410m = {(r["step"], r["seed"]) for r in existing.get("results", [])}
        results_410m = existing.get("results", [])
    else:
        done_keys_410m = set()
        results_410m   = []

    for step_rev in MATURITY_410M_STEPS:
        seeds = MATURITY_410M_MULTISEED.get(step_rev, [42])
        for seed in seeds:
            key = (step_rev, seed)
            if key in done_keys_410m:
                print(f"\n[410M {step_rev} seed={seed}] — already done, skipping")
                continue

            ckpts = [MATURITY_410M_DIR / step_rev / f"{d}_seed{seed}.pt" for d in DOMAINS]
            missing = [c for c in ckpts if not c.exists()]
            if missing:
                print(f"\n[410M {step_rev} seed={seed}] MISSING: {[c.name for c in missing]}")
                continue

            print(f"\n[410M {step_rev} seed={seed}]")
            # Base model at this revision
            base = load_maturity_base_410m(step_rev, device)
            base_eval = eval_all_domains(base, held_out_sets, device, label="base")
            del base; torch.cuda.empty_cache()

            specs = [load_maturity_specialist_410m(d, step_rev, seed, device) for d in DOMAINS]
            spec_evals = {d: eval_all_domains(s, held_out_sets, device) for d, s in zip(DOMAINS, specs)}

            moe = ThreeExpertMoE_MLP(*specs).to(device)
            train_router(moe, train_chunks, device)
            moe.eval()
            moe_eval = eval_all_domains(moe, held_out_sets, device, is_fused=True)

            best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
            moe_eq  = moe_eval["equal_weight_avg"]
            base_eq = base_eval["equal_weight_avg"]

            entry = {
                "step": step_rev, "seed": seed,
                "base_equal_weight":     base_eq,
                "best_spec_equal_weight": best_spec_eq,
                "moe_equal_weight":      moe_eq,
                "improvement_vs_spec":   pct_improvement(best_spec_eq, moe_eq),
                "improvement_vs_base":   pct_improvement(base_eq, moe_eq),
                "per_domain_moe":  {k: v for k, v in moe_eval.items() if not k.startswith("_")},
                "per_domain_base": {k: v for k, v in base_eval.items() if not k.startswith("_")},
                "eval_method": "per-domain-separate-equal-weight",
                "eval_batch_size": EVAL_BATCH,
            }
            results_410m.append(entry)
            print(f"  => improvement_vs_spec: {entry['improvement_vs_spec']:+.2f}%")

            out = {"experiment": "maturity_sweep_410m_v2", "results": results_410m}
            out_path_410m.write_text(json.dumps(out, indent=2), encoding="utf-8")

            del moe
            for s in specs: del s
            torch.cuda.empty_cache()

    print(f"  Saved: {out_path_410m}")

    # --- 1B ---
    print("\n--- 1B maturity sweep ---")
    if out_path_1b.exists():
        existing = json.loads(out_path_1b.read_text())
        done_keys_1b = {r["step"] for r in existing.get("results", [])}
        results_1b   = existing.get("results", [])
    else:
        done_keys_1b = set()
        results_1b   = []

    for step_rev in MATURITY_1B_STEPS:
        if step_rev in done_keys_1b:
            print(f"\n[1B {step_rev}] — already done, skipping")
            continue

        ckpts = [MATURITY_1B_DIR / step_rev / f"{d}_specialist_seed42.pt" for d in DOMAINS]
        missing = [c for c in ckpts if not c.exists()]
        if missing:
            print(f"\n[1B {step_rev}] MISSING: {[c.name for c in missing]}")
            continue

        print(f"\n[1B {step_rev}]")
        base_1b = load_maturity_base_1b(step_rev, device)
        base_eval = eval_all_domains(base_1b, held_out_sets, device, label="base_1b")
        del base_1b; torch.cuda.empty_cache()

        specs_1b = [load_maturity_specialist_1b(d, step_rev, device) for d in DOMAINS]
        spec_evals = {d: eval_all_domains(s, held_out_sets, device) for d, s in zip(DOMAINS, specs_1b)}

        # 1B uses single linear router (matches main 1B experiment)
        moe_1b = ThreeExpertMoE_Linear(*specs_1b).to(device)
        train_router(moe_1b, train_chunks, device)
        moe_1b.eval()
        moe_eval = eval_all_domains(moe_1b, held_out_sets, device, is_fused=True)

        best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())
        moe_eq  = moe_eval["equal_weight_avg"]
        base_eq = base_eval["equal_weight_avg"]

        entry = {
            "step": step_rev, "seed": 42,
            "base_equal_weight":     base_eq,
            "best_spec_equal_weight": best_spec_eq,
            "moe_equal_weight":      moe_eq,
            "improvement_vs_spec":   pct_improvement(best_spec_eq, moe_eq),
            "improvement_vs_base":   pct_improvement(base_eq, moe_eq),
            "per_domain_moe":  {k: v for k, v in moe_eval.items() if not k.startswith("_")},
            "per_domain_base": {k: v for k, v in base_eval.items() if not k.startswith("_")},
            "eval_method": "per-domain-separate-equal-weight",
            "eval_batch_size": EVAL_BATCH,
        }
        results_1b.append(entry)
        print(f"  => improvement_vs_spec: {entry['improvement_vs_spec']:+.2f}%")

        out = {"experiment": "maturity_sweep_1b_v2", "results": results_1b}
        out_path_1b.write_text(json.dumps(out, indent=2), encoding="utf-8")

        del moe_1b
        for s in specs_1b: del s
        torch.cuda.empty_cache()

    print(f"  Saved: {out_path_1b}")

    # Print summary
    print("\n" + "="*70)
    print("MATURITY SUMMARY (v2, corrected eval)")
    print("="*70)
    print("410M:")
    import statistics as st
    by_step = {}
    for r in results_410m:
        by_step.setdefault(r["step"], []).append(r)
    print(f"  {'step':>12}  {'seeds':>12}  {'vs spec (mean)':>16}  {'vs base (mean)':>16}")
    print(f"  {'-'*60}")
    for step_rev in MATURITY_410M_STEPS:
        if step_rev not in by_step: continue
        entries = by_step[step_rev]
        imps = [e["improvement_vs_spec"] for e in entries]
        imps_base = [e["improvement_vs_base"] for e in entries]
        seeds_str = ",".join(str(e["seed"]) for e in entries)
        print(f"  {step_rev:>12}  {seeds_str:>12}  "
              f"{st.mean(imps):>+15.2f}%  {st.mean(imps_base):>+15.2f}%")
    print("1B:")
    for r in sorted(results_1b, key=lambda x: x["step"]):
        print(f"  {r['step']:>12}  {'42':>12}  "
              f"{r['improvement_vs_spec']:>+15.2f}%  {r['improvement_vs_base']:>+15.2f}%")

    return results_410m, results_1b

# ============================================================================
# Experiment: hard_routing
# ============================================================================

def run_hard_routing(train_chunks, held_out_sets, device):
    """
    Soft (softmax) vs hard (argmax) routing comparison.
    Uses the same trained router weights for both. Seed=42 specialists.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: hard_routing (soft vs argmax routing)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "hard_routing_v2.json"

    print("\nLoading seed=42 specialists...")
    specs = [load_specialist(d, 42, device) for d in DOMAINS]

    # Train soft MoE router once
    moe_soft = ThreeExpertMoE_MLP(*specs).to(device)
    train_router(moe_soft, train_chunks, device)
    moe_soft.eval()

    # Evaluate soft routing
    print("\n[soft routing]")
    soft_eval = eval_all_domains(moe_soft, held_out_sets, device, is_fused=True, label="soft")

    # Evaluate hard routing (argmax of same router, same weights)
    print("\n[hard routing (argmax)]")
    hard_eval = _eval_hard_routing(moe_soft, held_out_sets, device)

    # Best spec for comparison
    spec_evals = {d: eval_all_domains(s, held_out_sets, device) for d, s in zip(DOMAINS, specs)}
    best_spec_eq = min(v["equal_weight_avg"] for v in spec_evals.values())

    soft_eq = soft_eval["equal_weight_avg"]
    hard_eq = hard_eval["equal_weight_avg"]

    results = {
        "experiment": "hard_routing_v2",
        "eval_method": "per-domain-separate-equal-weight",
        "eval_batch_size": EVAL_BATCH,
        "best_spec_equal_weight":  best_spec_eq,
        "soft_routing": {k: v for k, v in soft_eval.items() if not k.startswith("_")},
        "hard_routing": {k: v for k, v in hard_eval.items() if not k.startswith("_")},
        "metrics": {
            "soft_vs_spec":     pct_improvement(best_spec_eq, soft_eq),
            "hard_vs_spec":     pct_improvement(best_spec_eq, hard_eq),
            "soft_vs_hard":     pct_improvement(hard_eq, soft_eq),
        },
    }

    del moe_soft
    for s in specs: del s
    torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    m = results["metrics"]
    print(f"\n  Best spec:     {best_spec_eq:.4f}")
    print(f"  Soft routing:  {soft_eq:.4f}  (vs spec: {m['soft_vs_spec']:+.2f}%)")
    print(f"  Hard routing:  {hard_eq:.4f}  (vs spec: {m['hard_vs_spec']:+.2f}%)")
    print(f"  Soft vs hard:  {m['soft_vs_hard']:+.2f}%")

    return results


@torch.no_grad()
def _eval_hard_routing(moe, held_out_sets, device, bs=EVAL_BATCH, n_batches=EVAL_BATCHES):
    """Evaluate MoE using argmax (hard) routing with the same trained router."""
    losses = {}
    moe.eval()
    for domain, ds in held_out_sets.items():
        loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True, collate_fn=_collate)
        total, count = 0.0, 0
        for batch in loader:
            if count >= n_batches: break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            # Run all specialists to get logits + hidden states
            la, ha = moe._run(moe.spec_a, input_ids)
            lb, hb = moe._run(moe.spec_b, input_ids)
            lc, hc = moe._run(moe.spec_c, input_ids)
            h_avg = (ha + hb + hc) / 3.0
            raw_gates = moe.router(h_avg)          # (B, 3) logits
            argmax = raw_gates.argmax(dim=-1)       # (B,) — pick one expert per sample
            hard = F.one_hot(argmax, num_classes=3).float()  # (B, 3)
            fused = hard[:, 0:1, None] * la + hard[:, 1:2, None] * lb + hard[:, 2:3, None] * lc

            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
            total += loss.item()
            count += 1

        losses[domain] = round(total / count if count > 0 else float("inf"), 6)
        print(f"    hard {domain:8s}: {losses[domain]:.4f}")

    losses["equal_weight_avg"] = round(sum(losses[d] for d in held_out_sets) / len(held_out_sets), 6)
    return losses

# ============================================================================
# Experiment: benchmarks
# ============================================================================

def run_benchmarks(train_chunks, device):
    """
    Downstream NLP accuracy benchmarks. Uses log-likelihood scoring.
    Fresh router trained on all 3 domains (corrected protocol).
    Evaluates: base, each specialist, weight-avg, MoE.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: benchmarks (downstream NLP accuracy)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "benchmarks_v2.json"

    device_str = str(device)

    print("\nLoading tokenizer + seed=42 specialists...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    specs = {d: load_specialist(d, 42, device) for d in DOMAINS}
    base  = load_base_model(device)

    # Build weight-average
    import copy as _copy
    wa = _copy.deepcopy(specs["code"])
    wa_state = {}
    for k in specs["code"].state_dict():
        wa_state[k] = sum(s.state_dict()[k].float() for s in specs.values()) / 3.0
        wa_state[k] = wa_state[k].to(torch.bfloat16)
    wa.load_state_dict(wa_state)
    wa.eval()

    # Build MoE with fresh router (trained on all 3 domains — corrected)
    moe = ThreeExpertMoE_MLP(specs["code"], specs["science"], specs["fiction"]).to(device)
    train_router(moe, train_chunks, device)
    moe.eval()

    results = {
        "experiment": "benchmarks_v2",
        "protocol":   "log-likelihood accuracy, fresh router trained on all 3 domains",
        "models":     {},
        "n_examples_per_benchmark": 1000,
    }

    models_to_eval = {
        "base":         (base,           False),
        "code_spec":    (specs["code"],  False),
        "science_spec": (specs["science"], False),
        "fiction_spec": (specs["fiction"], False),
        "weight_avg":   (wa,             False),
        "moe":          (moe,            True),
    }

    for model_name, (model, is_moe) in models_to_eval.items():
        print(f"\n[{model_name}]")
        model_results = {}
        for bench in BENCHMARKS:
            print(f"  {bench}...", end=" ", flush=True)
            result = _eval_benchmark(model, tokenizer, bench, device, is_moe=is_moe)
            model_results[bench] = result
            print(f"acc={result['accuracy']:.3f} (n={result['n']})")
        avg_acc = sum(r["accuracy"] for r in model_results.values()) / len(BENCHMARKS)
        model_results["average"] = round(avg_acc, 6)
        results["models"][model_name] = model_results
        print(f"  => average accuracy: {avg_acc:.3f}")

    del moe, wa, base
    for s in specs.values(): del s
    torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Print summary table
    print("\n" + "="*70)
    print("BENCHMARKS SUMMARY (v2)")
    print("="*70)
    header = f"{'Model':<16}" + "".join(f"{b[:8]:>10}" for b in BENCHMARKS) + f"{'Avg':>10}"
    print(header)
    print("-" * len(header))
    for model_name, model_results in results["models"].items():
        row = f"{model_name:<16}"
        for b in BENCHMARKS:
            row += f"{model_results[b]['accuracy']:>10.3f}"
        row += f"{model_results['average']:>10.3f}"
        print(row)

    return results


@torch.no_grad()
def _eval_benchmark(model, tokenizer, benchmark_name, device, is_moe=False, n=1000):
    """Log-likelihood accuracy evaluation for one benchmark."""
    from datasets import load_dataset

    def ll(model, ctx, cont):
        ctx_ids  = tokenizer.encode(ctx,  add_special_tokens=False)
        cont_ids = tokenizer.encode(cont, add_special_tokens=False)
        if not cont_ids: return float("-inf")
        full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
        if is_moe:
            _, logits, _ = model(full_ids)
            logits = logits[0]
        else:
            logits = model(input_ids=full_ids).logits[0]
        start = len(ctx_ids)
        log_p = F.log_softmax(logits, dim=-1)
        return sum(log_p[start + i - 1, tid].item() for i, tid in enumerate(cont_ids))

    try:
        if benchmark_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n: break
                ctx = item["activity_label"] + ": " + item["ctx"]
                scores = [ll(model, ctx, e) for e in item["endings"]]
                if scores.index(max(scores)) == int(item["label"]): correct += 1
                total += 1
        elif benchmark_name == "arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n: break
                choices = item["choices"]["text"]
                labels  = item["choices"]["label"]
                if item["answerKey"] not in labels: total += 1; continue
                ans_idx = labels.index(item["answerKey"])
                scores = [ll(model, item["question"], c) for c in choices]
                if scores.index(max(scores)) == ans_idx: correct += 1
                total += 1
        elif benchmark_name == "piqa":
            ds = load_dataset("piqa", split="validation", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n: break
                scores = [ll(model, item["goal"], item["sol1"]),
                          ll(model, item["goal"], item["sol2"])]
                if scores.index(max(scores)) == int(item["label"]): correct += 1
                total += 1
        elif benchmark_name == "winogrande":
            ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n: break
                choices = [item["option1"], item["option2"]]
                ctxs = [item["sentence"].replace("_", c) for c in choices]
                scores = [ll(model, "", ctx) for ctx in ctxs]
                if scores.index(max(scores)) == int(item["answer"]) - 1: correct += 1
                total += 1
        elif benchmark_name == "lambada_openai":
            ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n: break
                text = item["text"]
                last_space = text.rfind(" ")
                ctx  = text[:last_space]
                cont = text[last_space+1:]
                score = ll(model, ctx, cont)
                # Lambada: "correct" if score > log(1/vocab_size) threshold (just report raw)
                correct += 1 if score > -10 else 0  # generous threshold for early model
                total += 1
        elif benchmark_name == "sciq":
            ds = load_dataset("allenai/sciq", split="test", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n: break
                choices = [item["correct_answer"], item["distractor1"],
                           item["distractor2"], item["distractor3"]]
                scores = [ll(model, item["question"], c) for c in choices]
                if scores.index(max(scores)) == 0: correct += 1  # correct_answer is index 0
                total += 1
        else:
            return {"accuracy": 0.0, "n": 0, "error": f"unknown benchmark: {benchmark_name}"}

        return {"accuracy": round(correct / total, 6) if total else 0.0, "n": total,
                "random_chance": RANDOM_CHANCE.get(benchmark_name, 0.0)}
    except Exception as e:
        print(f"  WARNING: {benchmark_name} failed: {e}")
        return {"accuracy": 0.0, "n": 0, "error": str(e)}

# ============================================================================
# Experiment: inference
# ============================================================================

def run_inference(train_chunks, held_out_sets, device):
    """
    Throughput / VRAM / latency benchmark for various MoE configurations.
    All measurements on seed=42 checkpoints with a freshly trained router.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: inference (throughput / VRAM / latency)")
    print("="*70)

    RESULTS_V2.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_V2 / "inference_v2.json"

    WARMUP = 3
    MEASURE = 10

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    specs = [load_specialist(d, 42, device) for d in DOMAINS]
    base  = load_base_model(device)

    # Train fresh router
    moe = ThreeExpertMoE_MLP(*specs).to(device)
    train_router(moe, train_chunks, device)
    moe.eval()

    # Prompt for throughput testing
    prompt_ids = tokenizer("The quick brown fox jumped over the lazy dog. " * 10,
                           return_tensors="pt")["input_ids"].to(device)
    prompt_ids = prompt_ids[:, :64]  # 64-token context

    results = {
        "experiment": "inference_v2",
        "hardware":   torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "seq_len":    SEQ_LEN,
        "measure_runs": MEASURE,
        "configs": {},
    }

    def measure_config(model_name, model, is_moe=False):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        # Warmup
        for _ in range(WARMUP):
            with torch.no_grad():
                if is_moe:
                    model(prompt_ids)
                else:
                    model(input_ids=prompt_ids)
        torch.cuda.reset_peak_memory_stats()
        # Measure
        times = []
        for _ in range(MEASURE):
            t0 = time.time()
            with torch.no_grad():
                if is_moe:
                    model(prompt_ids)
                else:
                    model(input_ids=prompt_ids)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        import statistics as st
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        mean_ms = st.mean(times) * 1000
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {model_name:<20}: {mean_ms:>7.1f}ms  VRAM={vram_gb:.2f}GB  params={params_m:.0f}M")
        return {
            "latency_ms_mean":   round(mean_ms, 2),
            "latency_ms_std":    round(st.stdev(times) * 1000, 2),
            "vram_gb_peak":      round(vram_gb, 3),
            "params_m":          round(params_m, 1),
        }

    print("\nMeasuring configurations...")
    results["configs"]["base"]          = measure_config("base",          base,        False)
    results["configs"]["code_spec"]     = measure_config("code_spec",     specs[0],    False)
    results["configs"]["moe_full"]      = measure_config("moe_full",      moe,         True)

    # Top-1 sparse MoE (only winning expert's logits used, frozen layers still run once)
    results["configs"]["moe_top1"] = _measure_top1(moe, prompt_ids, WARMUP, MEASURE, device)
    print(f"  {'moe_top1':<20}: {results['configs']['moe_top1']['latency_ms_mean']:>7.1f}ms  "
          f"VRAM={results['configs']['moe_top1']['vram_gb_peak']:.2f}GB")

    # Perplexity loss (corrected eval) for MoE and base on held-out
    print("\nEval loss (corrected protocol):")
    results["eval_losses"] = {
        "base": eval_all_domains(base, held_out_sets, device),
        "moe":  eval_all_domains(moe,  held_out_sets, device, is_fused=True),
    }
    print(f"  base EW: {results['eval_losses']['base']['equal_weight_avg']:.4f}")
    print(f"  moe  EW: {results['eval_losses']['moe']['equal_weight_avg']:.4f}")

    del moe, base
    for s in specs: del s
    torch.cuda.empty_cache()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return results


@torch.no_grad()
def _measure_top1(moe, prompt_ids, warmup, measure, device):
    """Measure top-1 sparse MoE: run all 3 specialists but only use winner's logits."""
    import statistics as st
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        la, ha = moe._run(moe.spec_a, prompt_ids)
        lb, hb = moe._run(moe.spec_b, prompt_ids)
        lc, hc = moe._run(moe.spec_c, prompt_ids)
        h_avg = (ha + hb + hc) / 3.0
        winner = moe.router(h_avg).argmax(dim=-1)  # (B,)
        # Use only winner logits
        logits_stack = torch.stack([la, lb, lc], dim=1)  # (B, 3, T, V)
        _ = logits_stack[torch.arange(prompt_ids.shape[0]), winner]
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(measure):
        t0 = time.time()
        la, ha = moe._run(moe.spec_a, prompt_ids)
        lb, hb = moe._run(moe.spec_b, prompt_ids)
        lc, hc = moe._run(moe.spec_c, prompt_ids)
        h_avg = (ha + hb + hc) / 3.0
        winner = moe.router(h_avg).argmax(dim=-1)
        logits_stack = torch.stack([la, lb, lc], dim=1)
        _ = logits_stack[torch.arange(prompt_ids.shape[0]), winner]
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return {
        "latency_ms_mean": round(st.mean(times) * 1000, 2),
        "latency_ms_std":  round(st.stdev(times) * 1000, 2),
        "vram_gb_peak":    round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "note": "all 3 specialists run; only top-1 logits used",
    }

# ============================================================================
# Main
# ============================================================================

ALL_EXPERIMENTS = [
    "crossover", "router", "freeze", "monolithic", "classifier",
    "five_domain", "maturity", "hard_routing", "benchmarks", "inference",
]


def main():
    parser = argparse.ArgumentParser(description="KALAVAI Phase 1 Ablations v2")
    parser.add_argument("--experiment", choices=ALL_EXPERIMENTS + ["all"], required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nKALAVAI Ablations v2")
    print(f"  experiment:      {args.experiment}")
    print(f"  eval_method:     per-domain equal-weight average (corrected protocol)")
    print(f"  eval_batch_size: {EVAL_BATCH} (consistent for ALL models)")
    print(f"  output dir:      {RESULTS_V2}")
    print(f"  device:          {device}")
    if device == "cuda":
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # Load tokenizer
    print(f"\nLoading tokenizer ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load 3-domain data (shared by most experiments)
    # five_domain loads its own data; benchmarks/inference use tokenizer directly
    needs_3domain = args.experiment in ["crossover", "router", "freeze", "monolithic",
                                        "classifier", "maturity", "hard_routing",
                                        "benchmarks", "inference", "all"]
    train_chunks = held_out_sets = None
    if needs_3domain:
        train_chunks, held_out_chunks = load_data(tokenizer)
        held_out_sets = {d: make_dataset(held_out_chunks[d]) for d in DOMAINS}

    experiments = ALL_EXPERIMENTS if args.experiment == "all" else [args.experiment]

    t_start = time.time()
    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# Starting: {exp}")
        print(f"{'#'*70}")
        t0 = time.time()
        if exp == "crossover":
            run_crossover(train_chunks, held_out_sets, device)
        elif exp == "router":
            run_router(train_chunks, held_out_sets, device)
        elif exp == "freeze":
            run_freeze(train_chunks, held_out_sets, device)
        elif exp == "monolithic":
            run_monolithic(train_chunks, held_out_sets, device)
        elif exp == "classifier":
            run_classifier(train_chunks, held_out_sets, device)
        elif exp == "five_domain":
            run_five_domain(tokenizer, device)
        elif exp == "maturity":
            run_maturity(train_chunks, held_out_sets, device)
        elif exp == "hard_routing":
            run_hard_routing(train_chunks, held_out_sets, device)
        elif exp == "benchmarks":
            run_benchmarks(train_chunks, device)
        elif exp == "inference":
            run_inference(train_chunks, held_out_sets, device)
        print(f"\n  {exp} done in {(time.time()-t0)/60:.1f} min")

    print(f"\n{'='*70}")
    print(f"ALL DONE — total {(time.time()-t_start)/60:.1f} min")
    print(f"Results in: {RESULTS_V2.resolve()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
