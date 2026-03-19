#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI Phase 2, Experiment 3 — BASE-HIDDEN ROUTER VARIANT
===========================================================
Same as kalavai_20contributor_experiment.py but with a different router input:
  specialist_mean (original): mean of all 20 specialist last-layer hidden states
  base_hidden     (this file): single forward through frozen base checkpoint

Motivation: specialist_mean oscillated at both lr=1e-3 and lr=2e-4. With N=20
each specialist contributes 5% to the mean pool — domain signal is diluted below
what a linear router can separate. Base-hidden carries the undiluted domain
signature of the input text without averaging out domain structure.

Results saved to results/phase2/twenty_contributor/result_seed{N}_base_hidden.json.
10 languages (cc100) + 10 domains (public HF datasets).
Base: Pythia-1B step10000 (hidden_size=2048).

Design:
  - 20 specialists trained independently (2000 steps each, FREEZE_LAYERS=0)
  - MoE fusion via linear router: nn.Linear(2048, 20) — base hidden state input
  - Specialists run sequentially during MoE forward (CPU offload to fit A100 80GB)
  - Eval: per-domain equal-weight average (Bug A + Bug B fixed)
  - Disk caching of tokenized chunks (pod restart resilience)

Stop/go printed at end:
  → GO if mean_divergence>15% AND gain>7%
  → PIVOT if mean_divergence>15% but gain<7% (router problem)
  → STOP if mean_divergence<10% (insufficient divergence)

RunPod note: run `pip install datasets==2.19.0` if trust_remote_code errors appear.
Target hardware: A100 80GB (specialists offloaded to CPU during MoE forward pass).

Specialists:
  Languages (cc100): tamil, yoruba, welsh, spanish, hindi,
                     swahili, vietnamese, arabic, indonesian, thai
  Domains:           code, medical, legal, patent, math,
                     finance, chemistry, fiction, dialogue, instructions
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

FREEZE_LAYERS   = 0        # CRITICAL — below 5k steps, freeze hurts
LR              = 2e-5
WEIGHT_DECAY    = 0.1
MAX_STEPS       = 2000
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
GRADIENT_CLIP   = 1.0
WARMUP_FRACTION = 0.1

ROUTER_STEPS      = 1000   # optimizer steps
ROUTER_LR         = 2e-4
ROUTER_BATCH      = 4      # physical batch size
ROUTER_GRAD_ACCUM = 5      # logical batch = 4 × 5 = 20 (one sample per domain on average)
EVAL_BATCH_SIZE = 4
EVAL_BATCHES    = 50

N_SAMPLES          = 2000   # per domain (long articles — 2000 texts → thousands of chunks)
N_SAMPLES_LANGUAGE = 50000  # per cc100 language (short sentences ~20-50 tokens each;
                             # need 50k lines to get ~1000+ packed 512-token chunks)

SEEDS           = [42]

# Override via env vars to redirect large checkpoint/cache dirs to local NVMe
# (avoids filling network drive — 20 × Pythia-1B checkpoints = ~56 GB)
# Example: export KALAVAI_CHECKPOINT_DIR=/tmp/kalavai_exp3_checkpoints
RESULTS_DIR    = Path(os.environ.get("KALAVAI_RESULTS_DIR",    "results/phase2/twenty_contributor_base_hidden"))
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
SPECIALISTS = LANGUAGE_SPECIALISTS + DOMAIN_SPECIALISTS   # length 20

# cc100 language codes
_CC100_LANG = {
    "tamil": "ta", "yoruba": "yo", "welsh": "cy", "spanish": "es",
    "hindi": "hi", "swahili": "sw", "vietnamese": "vi", "arabic": "ar",
    "indonesian": "id", "thai": "th",
}

# ============================================================================
# Data loading (with disk caching)
# ============================================================================

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_chunks.pt"


def _load_or_build_chunks(name: str, loader_fn, tokenizer) -> list:
    """Load tokenized chunks from disk cache, or build + cache them."""
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


# ── Language loaders (cc100) ─────────────────────────────────────────────────

def _make_cc100_loader(lang_code: str):
    def loader(n):
        # n is ignored — always pull N_SAMPLES_LANGUAGE lines because cc100 entries
        # are short sentences (~20-50 tokens); need 50k lines to get ~1000+ chunks.
        from datasets import load_dataset
        ds = load_dataset("cc100", lang=lang_code, split="train", streaming=True,
                          trust_remote_code=True)
        texts = [s["text"][:5000] for _, s in zip(range(N_SAMPLES_LANGUAGE), ds)
                 if s["text"].strip()]
        return texts
    return loader


# ── Domain loaders ───────────────────────────────────────────────────────────

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
    # hendrycks/competition_math is disabled (403). Use lighteval/MATH (same data, open mirror)
    # Fallback: gsm8k (grade-school math, smaller but freely available)
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
    # Fallback: gsm8k (question + answer fields)
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
    # Use reuters21578 or financial news with full article text (not just headlines)
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
    # Fallback: financial_phrasebank repeated to fill n (short sentences, but reliable)
    ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train",
                      trust_remote_code=True)
    texts = [s["sentence"][:5000] for s in ds if s.get("sentence", "").strip()]
    while len(texts) < n:
        texts = texts + texts
    return texts[:n]


def load_chemistry_texts(n):
    from datasets import load_dataset
    # Use ChemNLP / CHEMDNER abstracts or fallback to science-adjacent PubMed
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
    # Fallback: PubMed validation split (chemistry content)
    ds = load_dataset("ccdv/pubmed-summarization", split="validation", streaming=True)
    return [s["article"][:5000] for _, s in zip(range(n), ds)]


def load_fiction_texts(n):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    return [s["text"][:5000] for _, s in zip(range(n), ds) if s.get("text", "").strip()]


def load_dialogue_texts(n):
    from datasets import load_dataset
    # daily_dialog zip is corrupt on this env. Use blended_skill_talk (freely available).
    # Fallback: conv_ai_2
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
    # Fallback: conv_ai_2
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


# ── Dispatch map ─────────────────────────────────────────────────────────────

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
    """Load (and cache) tokenized chunks for all 20 specialists. Split 80/10/10."""
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
# Architecture — 20-Expert MoE with linear router
#
# Two modes (auto-selected at init based on available VRAM):
#   GPU mode:  all specialists loaded on GPU simultaneously — fast (~3× faster
#              router training). Requires ~67 GB VRAM (H100 NVL or tight A100).
#   CPU mode:  specialists kept on CPU, moved to GPU one at a time per forward
#              pass. Works on any A100 80GB. Slower but always safe.
# ============================================================================

class TwentyExpertMoE(nn.Module):
    """
    Sequence-level MoE over N specialist models — BASE-HIDDEN ROUTER VARIANT.
    Router input: single forward through frozen base checkpoint → nn.Linear(H, N) → softmax.
    Motivation: mean(specialist_h) diluted at N=20 (each expert contributes 5%);
    base hidden state carries undiluted domain signature. Only the router has requires_grad=True.
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

        # Try to load all specialists onto GPU; fall back to CPU offload
        self._gpu_models = None  # list of nn.Module on GPU (fast path)
        self._cpu_sds    = None  # list of state_dicts on CPU (slow path)
        self._base_model = None  # frozen base model for router input

        vram_free_gb = 0.0
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram_free_gb = free / 1e9

        # Each Pythia-1B is ~2.8 GB bfloat16; need headroom for activations + logits
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

        # Load frozen base model for router input (base-hidden variant)
        print(f"  [MoE] Loading frozen base model for router input...")
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
        self._base_model = base
        print(f"  [MoE] Base model ready — router input: base hidden state (SNR fix for N=20)")

    def _run_one_cpu(self, sd: dict, input_ids: torch.Tensor):
        """CPU offload path: build model, load weights, run, discard. Returns (logits, None)."""
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
            # Fast path: all specialists already on GPU (no hidden states needed)
            for m in self._gpu_models:
                with torch.no_grad():
                    out = m(input_ids=input_ids_gpu)
                all_logits.append(out.logits.float().cpu())
        else:
            # CPU offload path: load one specialist at a time
            for sd in self._cpu_sds:
                logits, _ = self._run_one_cpu(sd, input_ids_gpu)
                all_logits.append(logits)

        # Router input: single forward through frozen base model (undiluted domain signal)
        with torch.no_grad():
            base_out = self._base_model(input_ids=input_ids_gpu, output_hidden_states=True)
        h_router = base_out.hidden_states[-1].float().mean(dim=1)         # (B, H) on device
        gates  = torch.softmax(self.router(h_router), dim=-1)             # (B, N)
        # Accumulate weighted sum one expert at a time to avoid (B, N, T, V) allocation.
        # Peak GPU memory: O(B*T*V) instead of O(B*N*T*V) — critical at N=20.
        fused: torch.Tensor = None
        for i, logit in enumerate(all_logits):
            weighted = gates[:, i, None, None] * logit.to(self.device)    # (B, T, V)
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
    """Train one specialist. FREEZE_LAYERS=0 — all layers trainable."""
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
    """Train only the linear router on mixed data from all specialists.

    Uses gradient accumulation: ROUTER_GRAD_ACCUM physical forward passes per
    optimizer step, so logical batch = ROUTER_BATCH * ROUTER_GRAD_ACCUM = 20.
    Reported 'step' counts optimizer steps.
    """
    print(f"\n  Training router ({ROUTER_STEPS} steps, {len(SPECIALISTS)} experts)...")
    print(f"  lr={ROUTER_LR}, physical_bs={ROUTER_BATCH}, grad_accum={ROUTER_GRAD_ACCUM}, logical_bs={ROUTER_BATCH * ROUTER_GRAD_ACCUM}")
    all_chunks = []
    for name in SPECIALISTS:
        all_chunks.extend(train_chunks_by_specialist[name])
    combined  = chunks_to_dataset(all_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader    = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                           drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    t0 = time.time()
    for step in range(1, ROUTER_STEPS + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(ROUTER_GRAD_ACCUM):
            batch     = next(it)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            loss, _, _ = moe(input_ids, labels=labels)
            (loss / ROUTER_GRAD_ACCUM).backward()
            accum_loss += loss.item()
        clip_grad_norm_(moe.router.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={accum_loss / ROUTER_GRAD_ACCUM:.4f} | {time.time()-t0:.0f}s")
    moe.eval()


@torch.no_grad()
def eval_router_distribution(moe: TwentyExpertMoE, held_out_by_specialist: dict,
                              device, n_batches: int = 10) -> dict:
    """Compute per-specialist gate probabilities for each held-out domain."""
    moe.eval()
    results = {}
    for name in SPECIALISTS:
        ds     = held_out_by_specialist[name]   # already a PackedChunkDataset
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
    print(f"SEED {seed}")
    print(f"{'='*70}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    held_out_sets = {name: chunks_to_dataset(held_out_chunks[name]) for name in SPECIALISTS}
    eval_matrix   = {}

    # ── Base model eval ──────────────────────────────────────────────────────
    print("\n[base]")
    base = _load_base(device)
    eval_matrix["base"] = eval_all_domains(base, held_out_sets, device,
                                           EVAL_BATCH_SIZE, EVAL_BATCHES)
    del base
    torch.cuda.empty_cache()

    # ── Train + eval all 20 specialists ──────────────────────────────────────
    spec_state_dicts = []   # CPU state dicts for MoE
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

        # Keep CPU copy for MoE, free GPU
        spec_state_dicts.append({k: v.cpu() for k, v in spec.state_dict().items()})
        del spec
        torch.cuda.empty_cache()

    # ── Build + train MoE ────────────────────────────────────────────────────
    print(f"\n[moe]  building 20-expert MoE + training router ({ROUTER_STEPS} steps)...")
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

    del moe
    del spec_state_dicts
    torch.cuda.empty_cache()

    # ── Metrics ──────────────────────────────────────────────────────────────
    def eq(k):
        return eval_matrix[k]["equal_weight_avg"]

    base_eq      = eq("base")
    moe_eq       = eq("moe")
    best_spec_eq  = min(eq(f"{n}_spec") for n in SPECIALISTS)
    best_spec_dom = min(SPECIALISTS, key=lambda n: eq(f"{n}_spec"))

    # Per-specialist divergence (how much each specialist improves on its own domain)
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

    # ── Stop / Go ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STOP/GO [seed={seed}]:")
    print(f"  Mean divergence:  {mean_divergence:.2f}%")
    print(f"  Fusion gain:      {gain_vs_spec:+.2f}% vs best specialist")
    print(f"  Best spec domain: {best_spec_dom}")
    if mean_divergence > 15 and gain_vs_spec > 7:
        verdict = "GO"
        reason  = "diverge>15% AND gain>7%"
    elif mean_divergence > 15 and gain_vs_spec <= 7:
        verdict = "PIVOT"
        reason  = "diverge>15% but gain<7% — check router"
    else:
        verdict = "STOP"
        reason  = f"mean divergence {mean_divergence:.2f}% < 10% — insufficient for fusion"
    print(f"  → {verdict} ({reason})")
    print(f"{'='*60}")

    result = {
        "seed":            seed,
        "model_id":        MODEL_ID,
        "revision":        REVISION,
        "n_experts":       len(SPECIALISTS),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_batches":    EVAL_BATCHES,
        "eval_method":     "per-domain-separate-then-equal-weight-avg",
        "specialists":     SPECIALISTS,
        "eval_matrix":     {k: dict(v) for k, v in eval_matrix.items()},
        "metrics":         metrics,
        "router_distribution": router_dist,
        "stop_go":         {"verdict": verdict, "reason": reason},
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr":            LR,
            "max_steps":     MAX_STEPS,
            "batch_size":    BATCH_SIZE,
            "grad_accum":    GRAD_ACCUM,
            "router_steps":  ROUTER_STEPS,
            "router_lr":     ROUTER_LR,
            "router_input":  "base_hidden",
            "n_samples":     N_SAMPLES,
        },
    }
    return result


# ============================================================================
# Main
# ============================================================================

def print_results_summary(result: dict):
    print(f"\n{'─'*70}")
    print(f"Seed {result['seed']} — Summary")
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


def run_router_only(seed: int, tokenizer, device: str,
                    train_chunks: dict, held_out_chunks: dict) -> dict:
    """
    Re-train only the router for an existing seed result.
    Loads specialist checkpoints and eval_matrix from the prior result JSON.
    Use after a router training failure (e.g. lr oscillation) without retraining specialists.
    """
    prior_path = RESULTS_DIR / f"result_seed{seed}.json"
    if prior_path.exists():
        print(f"\n[router-only seed={seed}]  loading prior eval_matrix from {prior_path}")
        with open(prior_path) as f:
            prior = json.load(f)
        eval_matrix = {k: v for k, v in prior["eval_matrix"].items() if k != "moe"}
    else:
        print(f"\n[router-only seed={seed}]  no prior result — will eval specialists from checkpoints")
        eval_matrix = None  # built below after loading specialists

    # Load specialist state dicts from checkpoints
    spec_state_dicts = []
    for name in SPECIALISTS:
        ckpt_path = CHECKPOINT_DIR / f"{name}_specialist_seed{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        print(f"  Loading {name} from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        spec_state_dicts.append(sd)

    held_out_sets = {name: chunks_to_dataset(held_out_chunks[name]) for name in SPECIALISTS}

    # If no prior result, eval base + all specialists now
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

    print(f"\n[moe]  rebuilding 20-expert MoE + retraining router ({ROUTER_STEPS} steps, lr={ROUTER_LR}, bs={ROUTER_BATCH})...")
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

    # Recompute metrics from refreshed eval_matrix
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
    print(f"STOP/GO [seed={seed}, router-only retry]:")
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
            "router_batch": ROUTER_BATCH, "n_samples": N_SAMPLES,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-only", action="store_true",
                        help="Skip specialist training/eval; retrain only the router "
                             "using saved checkpoints and prior eval_matrix.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nModel:      {MODEL_ID} @ {REVISION}")
    print(f"Experts:    {len(SPECIALISTS)} ({len(LANGUAGE_SPECIALISTS)} lang + {len(DOMAIN_SPECIALISTS)} domain)")
    print(f"Cache dir:  {CACHE_DIR.absolute()}")
    if args.router_only:
        print(f"Mode:       router-only retry (lr={ROUTER_LR}, bs={ROUTER_BATCH}, steps={ROUTER_STEPS})")

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
            print(f"\n  git commit: [kalavai] phase2 exp3 seed={seed} router-retry: {imp:+.2f}% vs spec")
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
        print(f"\n  git commit: [kalavai] phase2 exp3 seed={seed}: {imp:+.2f}% vs spec")

    print("\n── Phase 2 Experiment 3 complete ──────────────────────────────────────")


if __name__ == "__main__":
    main()
