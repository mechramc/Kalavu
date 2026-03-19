#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI Phase 2, Experiment 3 — BASE-HIDDEN + LOAD-BALANCE + COSINE ANNEALING
===============================================================================
Standby variant: run ONLY if kalavai_20contributor_base_hidden.py (clean run)
gives PIVOT. Do not run in parallel with the clean variant.

Changes from base_hidden.py (three targeted diffs):
  1. Load-balance auxiliary loss (ROUTER_BALANCE_COEF=0.05):
       L_total = L_nll + 0.05 * ||mean_gate - 1/N||²
     Prevents expert collapse when the router finds it easier to route
     everything to the 2-3 best generalist experts.
  2. CosineAnnealingLR on router optimizer (T_max=ROUTER_STEPS):
     Prevents late-training oscillation by decaying LR after warm-up.
  3. Updated config/logging: records balance_coef and scheduler.

Motivation: if base_hidden clean fails with gate collapse (top-1 expert
receiving >40% weight), these additions force load balance as in
Switch Transformer (Fedus et al. 2021) and are standard MoE practice.
Results saved to results/phase2/twenty_contributor_base_hidden_v2/.

All other logic (router input, specialist list, eval method, data loading,
stop/go thresholds) is IDENTICAL to base_hidden.py.
"""

import argparse
import copy
import json
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

import sys; sys.path.insert(0, str(Path(__file__).parent))
from kalavai_eval_utils import eval_all_domains, eval_loss_domain, PackedChunkDataset, _collate, chunks_to_dataset, SEQ_LEN

# ============================================================================
# Config
# ============================================================================

MODEL_ID    = "EleutherAI/pythia-1b"
REVISION    = "step10000"
HIDDEN_SIZE = 2048

FREEZE_LAYERS   = 0
LR              = 2e-5
WEIGHT_DECAY    = 0.1
MAX_STEPS       = 2000
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
GRADIENT_CLIP   = 1.0
WARMUP_FRACTION = 0.1

ROUTER_STEPS      = 1000
ROUTER_LR         = 2e-4
ROUTER_BATCH      = 4
ROUTER_GRAD_ACCUM = 5
EVAL_BATCH_SIZE   = 4
EVAL_BATCHES      = 50

# ── DIFF 1: load-balance coefficient ─────────────────────────────────────────
# Penalises deviation from uniform gate distribution:  ||mean_gate - 1/N||²
# Range 0.01–0.1; 0.05 is the Switch Transformer default for auxiliary loss.
ROUTER_BALANCE_COEF = 0.05

N_SAMPLES          = 2000
N_SAMPLES_LANGUAGE = 50000

SEEDS = [42]

RESULTS_DIR    = Path(os.environ.get("KALAVAI_RESULTS_DIR",    "results/phase2/twenty_contributor_base_hidden_v2"))
CHECKPOINT_DIR = Path(os.environ.get("KALAVAI_CHECKPOINT_DIR", "checkpoints/phase2/twenty_contributor"))
CACHE_DIR      = Path(os.environ.get("KALAVAI_CACHE_DIR",      "data_cache/phase2"))

# ── Specialist list ──────────────────────────────────────────────────────────
LANGUAGE_SPECIALISTS = [
    "tamil", "yoruba", "welsh", "spanish", "hindi",
    "swahili", "vietnamese", "arabic", "indonesian", "thai",
]
DOMAIN_SPECIALISTS = [
    "code", "medical", "legal", "patent", "math",
    "finance", "chemistry", "fiction", "dialogue", "instructions",
]
SPECIALISTS = LANGUAGE_SPECIALISTS + DOMAIN_SPECIALISTS

_CC100_LANG = {
    "tamil": "ta", "yoruba": "yo", "welsh": "cy", "spanish": "es",
    "hindi": "hi", "swahili": "sw", "vietnamese": "vi", "arabic": "ar",
    "indonesian": "id", "thai": "th",
}

# ============================================================================
# Data loading (with disk caching) — identical to base_hidden.py
# ============================================================================

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_chunks.pt"


def _load_or_build_chunks(name: str, loader_fn, tokenizer) -> list:
    path = _cache_path(name)
    if path.exists():
        print(f"  [{name}] loading cached chunks from {path}")
        return torch.load(path, weights_only=True)
    print(f"  [{name}] building chunks (will cache to {path})...")
    texts = loader_fn(N_SAMPLES)
    ds = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
    chunks = ds.chunks
    torch.save(chunks, path)
    print(f"  [{name}] {len(chunks)} chunks cached")
    return chunks


def _make_cc100_loader(lang_code: str):
    def loader(n):
        from datasets import load_dataset
        ds = load_dataset("cc100", lang=lang_code, split="train", streaming=True,
                          trust_remote_code=True)
        texts = [s["text"][:5000] for _, s in zip(range(N_SAMPLES_LANGUAGE), ds)
                 if s["text"].strip()]
        return texts
    return loader


def load_code_texts(n):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train",
                      filter_languages=True, languages=["Python"])
    texts = [s["code"][:5000] for _, s in zip(range(n * 2), ds) if s.get("code", "").strip()]
    return texts[:n]


def load_medical_texts(n):
    from datasets import load_dataset
    ds = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)
    return [s["article"][:5000] for _, s in zip(range(n), ds)]


def load_legal_texts(n):
    from datasets import load_dataset
    ds = load_dataset("lex_glue", "eurlex", split="train", streaming=True)
    return [s["text"][:5000] for _, s in zip(range(n), ds)]


def load_patent_texts(n):
    from datasets import load_dataset
    ds = load_dataset("big_patent", "a", split="train", streaming=True)
    return [s["description"][:5000] for _, s in zip(range(n), ds)]


def load_math_texts(n):
    from datasets import load_dataset
    try:
        ds = load_dataset("lighteval/MATH", "all", split="train", streaming=True,
                          trust_remote_code=True)
        texts = []
        for _, s in zip(range(n * 2), ds):
            t = s.get("problem", "") + "\n" + s.get("solution", "")
            if t.strip():
                texts.append(t[:5000])
                if len(texts) >= n:
                    break
        if texts:
            return texts
    except Exception:
        pass
    ds = load_dataset("gsm8k", "main", split="train", streaming=True, trust_remote_code=True)
    texts = []
    for _, s in zip(range(n * 2), ds):
        t = s.get("question", "") + "\n" + s.get("answer", "")
        if t.strip():
            texts.append(t[:5000])
            if len(texts) >= n:
                break
    return texts


def load_finance_texts(n):
    from datasets import load_dataset
    try:
        ds = load_dataset("reuters21578", "ModHayes", split="train", streaming=True,
                          trust_remote_code=True)
        texts = []
        for _, s in zip(range(n * 3), ds):
            t = (s.get("title", "") + " " + s.get("text", "")).strip()
            if len(t) > 100:
                texts.append(t[:5000])
                if len(texts) >= n:
                    break
        if len(texts) >= n // 2:
            return texts
    except Exception:
        pass
    ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train",
                      trust_remote_code=True)
    texts = [s["sentence"][:5000] for s in ds if s.get("sentence", "").strip()]
    while len(texts) < n:
        texts = texts + texts
    return texts[:n]


def load_chemistry_texts(n):
    from datasets import load_dataset
    try:
        ds = load_dataset("bigbio/chemdner", name="chemdner_bigbio_kb",
                          split="train", streaming=True)
        texts = []
        for _, s in zip(range(n * 2), ds):
            passages = s.get("passages", [])
            for p in passages:
                for t in p.get("text", []):
                    if t.strip():
                        texts.append(t[:5000])
                        if len(texts) >= n:
                            return texts[:n]
    except Exception:
        pass
    ds = load_dataset("ccdv/pubmed-summarization", split="validation", streaming=True)
    return [s["article"][:5000] for _, s in zip(range(n), ds)]


def load_fiction_texts(n):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    return [s["text"][:5000] for _, s in zip(range(n), ds) if s.get("text", "").strip()]


def load_dialogue_texts(n):
    from datasets import load_dataset
    try:
        ds = load_dataset("blended_skill_talk", split="train", streaming=True,
                          trust_remote_code=True)
        texts = []
        for _, s in zip(range(n * 2), ds):
            turns = s.get("previous_utterance", []) + [s.get("free_messages", [""])[0]]
            t = " ".join(turns).strip()
            if t:
                texts.append(t[:5000])
                if len(texts) >= n:
                    break
        if texts:
            return texts
    except Exception:
        pass
    ds = load_dataset("conv_ai_2", split="train", streaming=True, trust_remote_code=True)
    texts = []
    for _, s in zip(range(n * 2), ds):
        utterances = s.get("utterances", [])
        if utterances:
            history = utterances[-1].get("history", [])
            t = " ".join(history).strip()
            if t:
                texts.append(t[:5000])
                if len(texts) >= n:
                    break
    return texts


def load_instructions_texts(n):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    texts = []
    for _, s in zip(range(n * 2), ds):
        t = (s.get("instruction", "") + " " + s.get("input", "") + " " + s.get("output", "")).strip()
        if t:
            texts.append(t[:5000])
            if len(texts) >= n:
                break
    return texts[:n]


def _make_loader(name: str):
    if name in _CC100_LANG:
        return _make_cc100_loader(_CC100_LANG[name])
    return {
        "code":         load_code_texts,
        "medical":      load_medical_texts,
        "legal":        load_legal_texts,
        "patent":       load_patent_texts,
        "math":         load_math_texts,
        "finance":      load_finance_texts,
        "chemistry":    load_chemistry_texts,
        "fiction":      load_fiction_texts,
        "dialogue":     load_dialogue_texts,
        "instructions": load_instructions_texts,
    }[name]


def load_all_data(tokenizer) -> tuple[dict, dict]:
    print("\n── Loading data (20 specialists) ──────────────────────────────────────")
    train_chunks, held_out_chunks = {}, {}
    for name in SPECIALISTS:
        chunks = _load_or_build_chunks(name, _make_loader(name), tokenizer)
        n = len(chunks)
        a, b = int(n * 0.8), int(n * 0.9)
        train_chunks[name]    = chunks[:a]
        held_out_chunks[name] = chunks[b:]
        print(f"  {name:16s}: total={n:4d}  train={len(train_chunks[name]):4d}  held_out={len(held_out_chunks[name]):4d}")
        if len(train_chunks[name]) < 500:
            print(f"  WARNING: {name} has <500 train chunks — results may be noisy")
    return train_chunks, held_out_chunks


# ============================================================================
# Architecture — identical to base_hidden.py
# ============================================================================

class TwentyExpertMoE(nn.Module):
    """
    Sequence-level MoE over N specialist models — BASE-HIDDEN ROUTER V2.
    Router input: single forward through frozen base checkpoint (undiluted signal).
    V2 additions: load-balance loss + cosine annealing to prevent expert collapse.
    """
    def __init__(self, specialist_state_dicts: list, model_id: str, revision: str,
                 hidden_size: int, device: str):
        super().__init__()
        self.n_experts   = len(specialist_state_dicts)
        self.model_id    = model_id
        self.revision    = revision
        self.device      = device
        self.hidden_size = hidden_size
        self.router = nn.Linear(hidden_size, self.n_experts, bias=False)

        self._gpu_models = None
        self._cpu_sds    = None
        self._base_model = None

        vram_free_gb = 0.0
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram_free_gb = free / 1e9

        vram_needed_gb = self.n_experts * 3.2
        if vram_free_gb >= vram_needed_gb:
            print(f"  [MoE] GPU mode: loading all {self.n_experts} specialists on GPU "
                  f"({vram_free_gb:.0f} GB free, need ~{vram_needed_gb:.0f} GB)")
            try:
                models = []
                for i, sd in enumerate(specialist_state_dicts):
                    m = AutoModelForCausalLM.from_pretrained(
                        model_id, revision=revision,
                        dtype=torch.bfloat16, trust_remote_code=True,
                    ).to(device)
                    m.load_state_dict(sd)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    models.append(m)
                    if (i + 1) % 5 == 0:
                        print(f"    loaded {i+1}/{self.n_experts}")
                self._gpu_models = models
                print(f"  [MoE] GPU mode active — all {self.n_experts} specialists on GPU")
            except torch.cuda.OutOfMemoryError:
                print("  [MoE] GPU mode OOM — falling back to CPU offload")
                for m in models:
                    del m
                torch.cuda.empty_cache()
                self._gpu_models = None

        if self._gpu_models is None:
            print(f"  [MoE] CPU offload mode: {self.n_experts} specialists on CPU "
                  f"({vram_free_gb:.0f} GB VRAM free)")
            self._cpu_sds = specialist_state_dicts

        print(f"  [MoE] Loading frozen base model for router input...")
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
        self._base_model = base
        print(f"  [MoE] Base model ready (v2: balance_coef={ROUTER_BALANCE_COEF}, cosine_lr)")

    def _run_one_cpu(self, sd: dict, input_ids: torch.Tensor):
        m = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision,
            dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)
        m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            out = m(input_ids=input_ids)
        logits = out.logits.float().cpu()
        del m, out
        torch.cuda.empty_cache()
        return logits, None

    def forward(self, input_ids, labels=None):
        input_ids_gpu = input_ids.to(self.device)
        all_logits = []

        if self._gpu_models is not None:
            for m in self._gpu_models:
                with torch.no_grad():
                    out = m(input_ids=input_ids_gpu)
                all_logits.append(out.logits.float().cpu())
        else:
            for sd in self._cpu_sds:
                logits, _ = self._run_one_cpu(sd, input_ids_gpu)
                all_logits.append(logits)

        with torch.no_grad():
            base_out = self._base_model(input_ids=input_ids_gpu, output_hidden_states=True)
        h_router = base_out.hidden_states[-1].float().mean(dim=1)
        gates    = torch.softmax(self.router(h_router), dim=-1)

        fused: torch.Tensor = None
        for i, logit in enumerate(all_logits):
            weighted = gates[:, i, None, None] * logit.to(self.device)
            fused = weighted if fused is None else fused + weighted

        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(self.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss, fused, gates


# ============================================================================
# Training helpers
# ============================================================================

def _batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_specialist(model, name: str, train_chunks: list, seed: int, device: str):
    set_seed(seed)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    dataset = chunks_to_dataset(train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)
    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer    = AdamW([p for p in model.parameters() if p.requires_grad],
                         lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler    = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS:
            break
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(**_batch_to_device(batch, device))
            loss = out.loss / GRAD_ACCUM
        loss.backward()
        accum         += 1
        running_loss  += loss.item() * GRAD_ACCUM

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
            if step % 500 == 0 or step == MAX_STEPS:
                avg = running_loss / step
                print(f"    [{name}] step {step}/{MAX_STEPS} | loss {avg:.4f} | {time.time()-t0:.0f}s")

    model.eval()
    print(f"  [{name}] done in {time.time()-t0:.0f}s")


def train_router(moe: TwentyExpertMoE, train_chunks_by_specialist: dict, device: str):
    """Train router with load-balance auxiliary loss + cosine LR annealing.

    L_total = L_nll + ROUTER_BALANCE_COEF * ||mean_gate - 1/N||²

    The balance term penalises deviation from uniform expert utilisation,
    preventing collapse to the 2-3 most generalist experts (Switch Transformer §5).
    Cosine annealing decays LR from ROUTER_LR to ~0 over ROUTER_STEPS to
    stabilise late-training gate assignments.
    """
    print(f"\n  Training router v2 ({ROUTER_STEPS} steps, {len(SPECIALISTS)} experts)...")
    print(f"  lr={ROUTER_LR} (cosine), physical_bs={ROUTER_BATCH}, grad_accum={ROUTER_GRAD_ACCUM}, "
          f"logical_bs={ROUTER_BATCH * ROUTER_GRAD_ACCUM}, balance_coef={ROUTER_BALANCE_COEF}")
    all_chunks = []
    for name in SPECIALISTS:
        all_chunks.extend(train_chunks_by_specialist[name])
    combined  = chunks_to_dataset(all_chunks)

    # ── DIFF 2: cosine annealing on router optimizer ──────────────────────────
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=ROUTER_STEPS)

    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    t0 = time.time()

    for step in range(1, ROUTER_STEPS + 1):
        optimizer.zero_grad()
        accum_nll     = 0.0
        accum_balance = 0.0

        for _ in range(ROUTER_GRAD_ACCUM):
            batch     = next(it)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            nll_loss, _, gates = moe(input_ids, labels=labels)

            # ── DIFF 1: load-balance auxiliary loss ───────────────────────────
            # gates: (B, N) — penalise deviation of mean gate from uniform 1/N
            balance_loss = (gates.mean(dim=0) - 1.0 / moe.n_experts).pow(2).sum()
            total_loss   = (nll_loss + ROUTER_BALANCE_COEF * balance_loss) / ROUTER_GRAD_ACCUM
            total_loss.backward()

            accum_nll     += nll_loss.item()
            accum_balance += balance_loss.item()

        clip_grad_norm_(moe.router.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # ── DIFF 2: step cosine scheduler each optimizer step

        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: "
                  f"nll={accum_nll/ROUTER_GRAD_ACCUM:.4f}  "
                  f"balance={accum_balance/ROUTER_GRAD_ACCUM:.4f} | "
                  f"{time.time()-t0:.0f}s")
    moe.eval()


@torch.no_grad()
def eval_router_distribution(moe: TwentyExpertMoE, held_out_by_specialist: dict,
                              device, n_batches: int = 10) -> dict:
    moe.eval()
    results = {}
    for name in SPECIALISTS:
        ds     = held_out_by_specialist[name]
        loader = DataLoader(ds, batch_size=ROUTER_BATCH, shuffle=False,
                            drop_last=True, collate_fn=_collate)
        gate_sums = [0.0] * moe.n_experts
        count = 0
        for batch in loader:
            if count >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(moe.n_experts):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        results[name] = [round(g / max(count, 1), 4) for g in gate_sums]
    return results


# ============================================================================
# Run one seed
# ============================================================================

def _load_base(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def _load_checkpoint(path, device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def run_seed(seed: int, tokenizer, device: str,
             train_chunks: dict, held_out_chunks: dict) -> dict:
    print(f"\n{'='*70}")
    print(f"SEED {seed}  [base_hidden_v2: balance_coef={ROUTER_BALANCE_COEF}, cosine_lr]")
    print(f"{'='*70}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    held_out_sets = {name: chunks_to_dataset(held_out_chunks[name]) for name in SPECIALISTS}
    eval_matrix   = {}

    print("\n[base]")
    base = _load_base(device)
    eval_matrix["base"] = eval_all_domains(base, held_out_sets, device,
                                           EVAL_BATCH_SIZE, EVAL_BATCHES)
    del base
    torch.cuda.empty_cache()

    spec_state_dicts = []
    for name in SPECIALISTS:
        ckpt_path = CHECKPOINT_DIR / f"{name}_specialist_seed{seed}.pt"
        if ckpt_path.exists():
            print(f"\n[{name}_spec]  loading {ckpt_path}")
            spec = _load_checkpoint(ckpt_path, device)
        else:
            print(f"\n[{name}_spec]  training (seed={seed}, {MAX_STEPS} steps)...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            train_specialist(spec, name, train_chunks[name], seed, device)
            torch.save(spec.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        print(f"\n[{name}_spec eval]")
        eval_matrix[f"{name}_spec"] = eval_all_domains(
            spec, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)

        spec_state_dicts.append({k: v.cpu() for k, v in spec.state_dict().items()})
        del spec
        torch.cuda.empty_cache()

    print(f"\n[moe]  building 20-expert MoE + training router v2 ({ROUTER_STEPS} steps)...")
    moe = TwentyExpertMoE(
        specialist_state_dicts=spec_state_dicts,
        model_id=MODEL_ID,
        revision=REVISION,
        hidden_size=HIDDEN_SIZE,
        device=device,
    )
    moe.router = moe.router.to(device)
    train_router(moe, train_chunks, device)
    moe.eval()

    print("\n[moe eval]")
    eval_matrix["moe"] = eval_all_domains(moe, held_out_sets, device,
                                          EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=True)

    print("\n[moe router distribution]  (top-3 gates per specialist, 10 batches)")
    router_dist = eval_router_distribution(moe, held_out_sets, device, n_batches=10)
    for name, gates in router_dist.items():
        top3_idx = sorted(range(moe.n_experts), key=lambda i: gates[i], reverse=True)[:3]
        top3_str = "  ".join(f"{SPECIALISTS[i]}={gates[i]:.3f}" for i in top3_idx)
        print(f"  {name:16s}: top3 → {top3_str}")

    del moe, spec_state_dicts
    torch.cuda.empty_cache()

    def eq(k):
        return eval_matrix[k]["equal_weight_avg"]

    base_eq       = eq("base")
    moe_eq        = eq("moe")
    best_spec_eq  = min(eq(f"{n}_spec") for n in SPECIALISTS)
    best_spec_dom = min(SPECIALISTS, key=lambda n: eq(f"{n}_spec"))

    domain_divs = []
    for name in SPECIALISTS:
        base_d = eval_matrix["base"].get(name, base_eq)
        spec_d = eval_matrix[f"{name}_spec"].get(name, eq(f"{name}_spec"))
        div = (base_d - spec_d) / base_d * 100 if base_d > 0 else 0.0
        domain_divs.append(div)
    mean_divergence = round(statistics.mean(domain_divs), 2)
    gain_vs_spec    = round((best_spec_eq - moe_eq) / best_spec_eq * 100, 4)

    metrics = {
        "base_equal_weight":      round(base_eq, 6),
        "best_spec_equal_weight": round(best_spec_eq, 6),
        "best_spec_domain":       best_spec_dom,
        "moe_equal_weight":       round(moe_eq, 6),
        "improvement_vs_spec":    gain_vs_spec,
        "improvement_vs_base":    round((base_eq - moe_eq) / base_eq * 100, 4),
        "mean_divergence":        mean_divergence,
        "per_specialist_divergence": {
            name: round(domain_divs[i], 2)
            for i, name in enumerate(SPECIALISTS)
        },
    }

    if mean_divergence > 15 and gain_vs_spec > 7:
        verdict, reason = "GO", "diverge>15% AND gain>7%"
    elif mean_divergence > 15 and gain_vs_spec <= 7:
        verdict, reason = "PIVOT", "diverge>15% but gain<7% — check router"
    else:
        verdict, reason = "STOP", f"mean divergence {mean_divergence:.2f}% < 10%"

    print(f"\n{'='*60}")
    print(f"STOP/GO [seed={seed}, base_hidden_v2]:")
    print(f"  Mean divergence: {mean_divergence:.2f}%  |  Fusion gain: {gain_vs_spec:+.2f}%")
    print(f"  → {verdict} ({reason})")
    print(f"{'='*60}")

    result = {
        "seed": seed, "model_id": MODEL_ID, "revision": REVISION,
        "n_experts": len(SPECIALISTS), "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_batches": EVAL_BATCHES, "eval_method": "per-domain-separate-then-equal-weight-avg",
        "specialists": SPECIALISTS,
        "eval_matrix": {k: dict(v) for k, v in eval_matrix.items()},
        "metrics": metrics,
        "router_distribution": router_dist,
        "stop_go": {"verdict": verdict, "reason": reason},
        "config": {
            "freeze_layers": FREEZE_LAYERS, "lr": LR, "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "router_steps": ROUTER_STEPS, "router_lr": ROUTER_LR,
            "router_input": "base_hidden",
            # ── DIFF 3: log v2-specific config ───────────────────────────────
            "router_balance_coef": ROUTER_BALANCE_COEF,
            "router_lr_scheduler": "cosine_annealing",
            "n_samples": N_SAMPLES,
        },
    }
    return result


def run_router_only(seed: int, tokenizer, device: str,
                    train_chunks: dict, held_out_chunks: dict) -> dict:
    """Re-train only the router (v2) for an existing seed result."""
    prior_path = RESULTS_DIR / f"result_seed{seed}.json"
    # Also check clean run dir for reusable eval_matrix
    clean_path = Path("results/phase2/twenty_contributor_base_hidden") / f"result_seed{seed}.json"
    for p in [prior_path, clean_path]:
        if p.exists():
            print(f"\n[router-only v2 seed={seed}]  loading prior eval_matrix from {p}")
            with open(p) as f:
                prior = json.load(f)
            eval_matrix = {k: v for k, v in prior["eval_matrix"].items() if k != "moe"}
            break
    else:
        print(f"\n[router-only v2 seed={seed}]  no prior result — will eval specialists from checkpoints")
        eval_matrix = None

    spec_state_dicts = []
    for name in SPECIALISTS:
        ckpt_path = CHECKPOINT_DIR / f"{name}_specialist_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        print(f"  Loading {name} from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        spec_state_dicts.append(sd)

    held_out_sets = {name: chunks_to_dataset(held_out_chunks[name]) for name in SPECIALISTS}

    if eval_matrix is None:
        eval_matrix = {}
        print("\n[base]")
        base = _load_base(device)
        eval_matrix["base"] = eval_all_domains(base, held_out_sets, device,
                                               EVAL_BATCH_SIZE, EVAL_BATCHES)
        del base; torch.cuda.empty_cache()
        for name, sd in zip(SPECIALISTS, spec_state_dicts):
            print(f"\n[{name}_spec eval]")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            spec.load_state_dict(sd)
            spec.eval()
            eval_matrix[f"{name}_spec"] = eval_all_domains(
                spec, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)
            del spec; torch.cuda.empty_cache()

    print(f"\n[moe]  rebuilding 20-expert MoE + retraining router v2 ({ROUTER_STEPS} steps)...")
    moe = TwentyExpertMoE(
        specialist_state_dicts=spec_state_dicts,
        model_id=MODEL_ID, revision=REVISION,
        hidden_size=HIDDEN_SIZE, device=device,
    )
    moe.router = moe.router.to(device)
    train_router(moe, train_chunks, device)
    moe.eval()

    print("\n[moe eval]")
    eval_matrix["moe"] = eval_all_domains(moe, held_out_sets, device,
                                          EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=True)

    print("\n[moe router distribution]  (top-3 gates per specialist, 10 batches)")
    router_dist = eval_router_distribution(moe, held_out_sets, device, n_batches=10)
    for name, gates in router_dist.items():
        top3_idx = sorted(range(moe.n_experts), key=lambda i: gates[i], reverse=True)[:3]
        top3_str = "  ".join(f"{SPECIALISTS[i]}={gates[i]:.3f}" for i in top3_idx)
        print(f"  {name:16s}: top3 → {top3_str}")

    del moe, spec_state_dicts
    torch.cuda.empty_cache()

    def eq(k): return eval_matrix[k]["equal_weight_avg"]

    base_eq       = eq("base")
    moe_eq        = eq("moe")
    best_spec_eq  = min(eq(f"{n}_spec") for n in SPECIALISTS)
    best_spec_dom = min(SPECIALISTS, key=lambda n: eq(f"{n}_spec"))

    domain_divs = []
    for name in SPECIALISTS:
        base_d = eval_matrix["base"].get(name, base_eq)
        spec_d = eval_matrix[f"{name}_spec"].get(name, eq(f"{name}_spec"))
        div = (base_d - spec_d) / base_d * 100 if base_d > 0 else 0.0
        domain_divs.append(div)
    mean_divergence = round(statistics.mean(domain_divs), 2)
    gain_vs_spec    = round((best_spec_eq - moe_eq) / best_spec_eq * 100, 4)

    metrics = {
        "base_equal_weight": round(base_eq, 6),
        "best_spec_equal_weight": round(best_spec_eq, 6),
        "best_spec_domain": best_spec_dom,
        "moe_equal_weight": round(moe_eq, 6),
        "improvement_vs_spec": gain_vs_spec,
        "improvement_vs_base": round((base_eq - moe_eq) / base_eq * 100, 4),
        "mean_divergence": mean_divergence,
        "per_specialist_divergence": {
            name: round(domain_divs[i], 2) for i, name in enumerate(SPECIALISTS)
        },
    }

    if mean_divergence > 15 and gain_vs_spec > 7:
        verdict, reason = "GO", "diverge>15% AND gain>7%"
    elif mean_divergence > 15 and gain_vs_spec <= 7:
        verdict, reason = "PIVOT", "diverge>15% but gain<7% — check router"
    else:
        verdict, reason = "STOP", f"mean divergence {mean_divergence:.2f}% < 10%"

    print(f"\n{'='*60}")
    print(f"STOP/GO [seed={seed}, base_hidden_v2 router-only]:")
    print(f"  Mean divergence: {mean_divergence:.2f}%  |  Fusion gain: {gain_vs_spec:+.2f}%")
    print(f"  → {verdict} ({reason})")
    print(f"{'='*60}")

    result = {
        "seed": seed, "model_id": MODEL_ID, "revision": REVISION,
        "n_experts": len(SPECIALISTS), "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_batches": EVAL_BATCHES, "eval_method": "per-domain-separate-then-equal-weight-avg",
        "router_only_retry": True,
        "specialists": SPECIALISTS,
        "eval_matrix": {k: dict(v) for k, v in eval_matrix.items()},
        "metrics": metrics,
        "router_distribution": router_dist,
        "stop_go": {"verdict": verdict, "reason": reason},
        "config": {
            "freeze_layers": FREEZE_LAYERS, "lr": LR, "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "router_steps": ROUTER_STEPS, "router_lr": ROUTER_LR,
            "router_input": "base_hidden",
            "router_balance_coef": ROUTER_BALANCE_COEF,
            "router_lr_scheduler": "cosine_annealing",
            "n_samples": N_SAMPLES,
        },
    }
    return result


def print_results_summary(result: dict):
    print(f"\n{'─'*70}")
    print(f"Seed {result['seed']} — Summary [base_hidden_v2]")
    print(f"{'─'*70}")
    m = result["metrics"]
    print(f"  Base EW:           {m['base_equal_weight']:.4f}")
    print(f"  Best spec ({m['best_spec_domain']:12s}): {m['best_spec_equal_weight']:.4f}")
    print(f"  MoE EW:            {m['moe_equal_weight']:.4f}")
    print(f"  Gain vs spec:      {m['improvement_vs_spec']:+.2f}%")
    print(f"  Gain vs base:      {m['improvement_vs_base']:+.2f}%")
    print(f"  Mean divergence:   {m['mean_divergence']:.2f}%")
    print(f"\n  Per-specialist divergence:")
    for name, div in m["per_specialist_divergence"].items():
        bar = "█" * int(max(div / 2, 0))
        print(f"    {name:16s}  {div:+6.2f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-only", action="store_true",
                        help="Skip specialist training/eval; retrain only the router v2 "
                             "using saved checkpoints. Reuses eval_matrix from prior result "
                             "if available (checks v2 dir, then clean base_hidden dir).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nModel:         {MODEL_ID} @ {REVISION}")
    print(f"Experts:       {len(SPECIALISTS)} ({len(LANGUAGE_SPECIALISTS)} lang + {len(DOMAIN_SPECIALISTS)} domain)")
    print(f"Router:        base_hidden (v2) — balance_coef={ROUTER_BALANCE_COEF}, cosine_lr")
    print(f"Results dir:   {RESULTS_DIR.absolute()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_chunks, held_out_chunks = load_all_data(tokenizer)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        if args.router_only:
            retry_path = RESULTS_DIR / f"result_seed{seed}_router_retry.json"
            result = run_router_only(seed, tokenizer, device, train_chunks, held_out_chunks)
            print_results_summary(result)
            with open(retry_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n  Result saved: {retry_path}")
            imp = result["metrics"]["improvement_vs_spec"]
            print(f"\n  git commit: [kalavai] phase2 exp3 seed={seed} base_hidden_v2 router-retry: {imp:+.2f}% vs spec")
            continue

        result_path = RESULTS_DIR / f"result_seed{seed}.json"
        if result_path.exists():
            print(f"\n[seed={seed}]  result already exists at {result_path} — skipping")
            with open(result_path) as f:
                result = json.load(f)
            print_results_summary(result)
            continue

        result = run_seed(seed, tokenizer, device, train_chunks, held_out_chunks)
        print_results_summary(result)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result saved: {result_path}")
        imp = result["metrics"]["improvement_vs_spec"]
        print(f"\n  git commit: [kalavai] phase2 exp3 seed={seed} base_hidden_v2: {imp:+.2f}% vs spec")

    print("\n── Phase 2 Experiment 3 (base_hidden_v2) complete ─────────────────────")


if __name__ == "__main__":
    main()
