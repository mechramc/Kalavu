#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI Phase 2, Experiment 1: Cross-Lingual Fusion — Curriculum Warm-Start
============================================================================
Tamil, Yoruba, Welsh + Code (English control).
Base: Pythia-410M step10000.

This variant adds a two-stage curriculum to the router training to fix router
collapse (seed 42: Yoruba routes to Tamil specialist) caused by byte-fallback
hidden-state geometry similarity at initialisation.

Stage A — domain-pure warm-start:
  Each batch contains tokens from ONE language only, cycling through all four
  languages in round-robin order (lang_idx = step % num_languages). The router
  sees clean, unambiguous signal per domain and learns to separate their
  hidden-state representations before encountering any mixed data.

Stage B — normal mixed training:
  Resumes with the original mixed-batch construction for the remaining steps,
  identical to exp1.py.

CLI arguments (required/defaulted):
  --seed            int  (42, 137, or 2026)
  --stage-a-steps   int  default 100  domain-pure warm-start steps
  --stage-b-steps   int  default 400  normal mixed-training steps

Results are saved to:
  results/phase2/cross_lingual/curriculum/result_seed{seed}.json

After router training a per-domain gate-weight analysis is printed and stored
under the "gate_analysis" key in the result JSON. If any language routes >80%
to a different language's specialist the run is flagged as COLLAPSE.

Design (unchanged from exp1):
  - 4 specialists trained independently (2000 steps, FREEZE_LAYERS=0)
  - MoE fusion via 2-layer MLP router (matches corrected 410M +7.70%)
  - Eval: per-domain equal-weight average (Bug A + Bug B fixed)
  - Also reports raw perplexity (exp(loss)) per language

Datasets:
  - tamil:  cc100 lang=ta  (text_key: text)
  - yoruba: cc100 lang=yo  (text_key: text)
  - welsh:  cc100 lang=cy  (text_key: text)
  - code:   code_search_net python (text_key: whole_func_string)
"""

import argparse
import copy
import json
import math
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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import sys; sys.path.insert(0, str(Path(__file__).parent))
from kalavai_eval_utils import (eval_all_domains, PackedChunkDataset,
                                 _collate, chunks_to_dataset, SEQ_LEN)

# ============================================================================
# Config
# ============================================================================

MODEL_ID    = "EleutherAI/pythia-410m"
REVISION    = "step10000"
HIDDEN_SIZE = 1024

FREEZE_LAYERS   = 0        # CRITICAL — below 5k steps, freeze hurts
LR              = 2e-5
WEIGHT_DECAY    = 0.1
MAX_STEPS       = 2000
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
GRADIENT_CLIP   = 1.0
WARMUP_FRACTION = 0.1

ROUTER_LR       = 1e-3
ROUTER_BATCH    = 4
EVAL_BATCH_SIZE = 4
EVAL_BATCHES    = 50

# Collapse detection threshold: if any domain routes ≥ this fraction to a
# *different* domain's specialist, the run is flagged as COLLAPSE.
COLLAPSE_THRESHOLD = 0.80

N_SAMPLES_LANGUAGE = 50000  # cc100 lines (short ~30 tokens each → need many)
                            # wikipedia fallback uses same n but articles are longer;
                            # streaming stops early once enough text is tokenized
N_SAMPLES_CODE     = 3000   # code_search_net (long functions, 3k → plenty of chunks)

DOMAINS = ["tamil", "yoruba", "welsh", "code"]

RESULTS_DIR    = Path(os.environ.get(
    "KALAVAI_RESULTS_DIR",
    "results/phase2/cross_lingual/curriculum",
))
CHECKPOINT_DIR = Path(os.environ.get(
    "KALAVAI_CHECKPOINT_DIR",
    "checkpoints/phase2/cross_lingual",
))

# ============================================================================
# Data loading (identical to exp1)
# ============================================================================

_WIKI_LANG = {"ta": "20231101.ta", "yo": "20231101.yo", "cy": "20231101.cy"}


def load_cc100_texts(lang_code: str, n: int) -> list[str]:
    from datasets import load_dataset
    # Try cc100 first (works on RunPod with older datasets); fall back to
    # wikimedia/wikipedia (Parquet, no script, works everywhere).
    try:
        print(f"  Loading cc100 lang={lang_code} (n={n})...")
        ds = load_dataset("cc100", lang=lang_code, split="train", streaming=True)
        texts = [s["text"][:5000] for _, s in zip(range(n), ds) if s["text"].strip()]
        if texts:
            print(f"    {len(texts)} raw texts (cc100)")
            return texts
    except Exception:
        pass
    # Fallback: wikimedia/wikipedia — Parquet, no loading script required
    wiki_config = _WIKI_LANG.get(lang_code)
    if wiki_config:
        print(f"  cc100 unavailable — loading wikimedia/wikipedia {wiki_config}...")
        ds = load_dataset("wikimedia/wikipedia", wiki_config,
                          split="train", streaming=True)
        texts = [s["text"][:5000] for _, s in zip(range(n), ds) if s["text"].strip()]
        print(f"    {len(texts)} articles (wikipedia)")
        return texts
    raise RuntimeError(f"No available dataset for lang_code={lang_code}")


def load_code_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading code_search_net python (n={n})...")
    try:
        ds = load_dataset("code_search_net", "python", split="train",
                          streaming=True, trust_remote_code=True)
        texts = [s["whole_func_string"][:5000] for _, s in zip(range(n), ds)
                 if s.get("whole_func_string", "").strip()]
        print(f"    {len(texts)} functions")
        return texts
    except Exception as e:
        print(f"    code_search_net failed ({e}), falling back to codeparrot/github-code")
        ds = load_dataset("codeparrot/github-code", streaming=True, split="train",
                          trust_remote_code=True)
        texts = [s["code"][:5000] for _, s in zip(range(n * 2), ds)
                 if s.get("code", "").strip()]
        return texts[:n]


def load_all_data(tokenizer) -> tuple[dict, dict]:
    print("\nLoading data...")
    loaders = {
        "tamil":  lambda: load_cc100_texts("ta", N_SAMPLES_LANGUAGE),
        "yoruba": lambda: load_cc100_texts("yo", N_SAMPLES_LANGUAGE),
        "welsh":  lambda: load_cc100_texts("cy", N_SAMPLES_LANGUAGE),
        "code":   lambda: load_code_texts(N_SAMPLES_CODE),
    }
    train_chunks, held_out_chunks = {}, {}
    for domain, loader_fn in loaders.items():
        texts = loader_fn()
        ds_full = PackedChunkDataset(texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
        n = len(ds_full.chunks)
        a, b = int(n * 0.8), int(n * 0.9)
        train_chunks[domain]    = ds_full.chunks[:a]
        held_out_chunks[domain] = ds_full.chunks[b:]
        print(f"  {domain:8s}: total={n:5d}  train={len(train_chunks[domain]):5d}"
              f"  held_out={len(held_out_chunks[domain]):4d}")
        if len(train_chunks[domain]) < 500:
            print(f"  WARNING: {domain} has <500 train chunks — results may be noisy")
    return train_chunks, held_out_chunks


# ============================================================================
# Architecture — FourExpertMoE with 2-layer MLP router (identical to exp1)
# ============================================================================

class FourExpertMoE(nn.Module):
    """
    Sequence-level MoE over four specialist models.
    Router: mean of last hidden states → MLP (H→256→ReLU→4) → softmax gates.
    Specialists are frozen; only router is trained.
    """
    def __init__(self, specs: list, hidden_size: int):
        super().__init__()
        assert len(specs) == 4
        self.spec_a, self.spec_b, self.spec_c, self.spec_d = specs
        for p in (list(self.spec_a.parameters()) + list(self.spec_b.parameters()) +
                  list(self.spec_c.parameters()) + list(self.spec_d.parameters())):
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 4, bias=False),
        )

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        la, ha = self._run(self.spec_a, input_ids)
        lb, hb = self._run(self.spec_b, input_ids)
        lc, hc = self._run(self.spec_c, input_ids)
        ld, hd = self._run(self.spec_d, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc + hd) / 4.0), dim=-1)
        fused = (gates[:, 0:1, None] * la + gates[:, 1:2, None] * lb +
                 gates[:, 2:3, None] * lc + gates[:, 3:4, None] * ld)
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))
        return loss, fused, gates


# ============================================================================
# Training helpers
# ============================================================================

def _batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_specialist(model, domain: str, train_chunks: list, seed: int, device: str):
    set_seed(seed)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [{domain}] trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    dataset  = chunks_to_dataset(train_chunks)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
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
        accum        += 1
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
            if step % 500 == 0 or step == MAX_STEPS:
                print(f"    [{domain}] step {step}/{MAX_STEPS} | "
                      f"loss {running_loss/step:.4f} | {time.time()-t0:.0f}s")
    model.eval()
    print(f"  [{domain}] done in {time.time()-t0:.0f}s")


# ============================================================================
# Curriculum router training
# ============================================================================

def _build_domain_loaders(train_chunks_by_domain: dict) -> dict:
    """Build one infinite cycle iterator per domain for Stage A round-robin."""
    loaders = {}
    for domain, chunks in train_chunks_by_domain.items():
        ds = chunks_to_dataset(chunks)
        loader = DataLoader(ds, batch_size=ROUTER_BATCH, shuffle=True,
                            drop_last=True, collate_fn=_collate)
        loaders[domain] = cycle(loader)
    return loaders


def _build_mixed_loader(train_chunks_by_domain: dict):
    """Build a single shuffled mixed iterator for Stage B (identical to exp1)."""
    all_chunks = []
    for chunks in train_chunks_by_domain.values():
        all_chunks.extend(chunks)
    combined = chunks_to_dataset(all_chunks)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    return cycle(loader)


def train_router_curriculum(
    moe: FourExpertMoE,
    train_chunks_by_domain: dict,
    device: str,
    stage_a_steps: int,
    stage_b_steps: int,
) -> dict[str, int]:
    """
    Two-stage curriculum router training.

    Stage A (domain-pure warm-start):
        step % num_languages selects which domain's data to use for that step.
        The router sees only one language per batch, cycling round-robin.
        This prevents early collapse caused by byte-fallback geometry similarity.

    Stage B (normal mixed):
        Identical to exp1's train_router — all domains' chunks pooled and shuffled.

    Returns a dict with step counts for logging.
    """
    num_languages   = len(DOMAINS)
    total_steps     = stage_a_steps + stage_b_steps

    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    moe.train()

    # Pre-build iterators for both stages so we don't rebuild mid-loop.
    domain_loaders = _build_domain_loaders(train_chunks_by_domain)
    mixed_loader   = _build_mixed_loader(train_chunks_by_domain)

    print(f"\n  Training router with curriculum ({total_steps} steps total):")
    print(f"    Stage A (domain-pure): {stage_a_steps} steps, round-robin over {DOMAINS}")
    print(f"    Stage B (mixed):       {stage_b_steps} steps")

    # ── Stage A: domain-pure round-robin ──────────────────────────────────────
    if stage_a_steps > 0:
        print(f"\n  [Stage A] domain-pure warm-start...")
        for step in range(1, stage_a_steps + 1):
            # Round-robin: pick domain by step index (0-based inside stage A)
            domain = DOMAINS[(step - 1) % num_languages]
            batch  = next(domain_loaders[domain])

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            loss, _, _ = moe(input_ids, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0 or step == stage_a_steps:
                current_domain = DOMAINS[(step - 1) % num_languages]
                print(f"    [A] step {step:3d}/{stage_a_steps} | "
                      f"domain={current_domain:7s} | loss={loss.item():.4f}")

    # ── Stage B: normal mixed training ────────────────────────────────────────
    if stage_b_steps > 0:
        print(f"\n  [Stage B] normal mixed training...")
        for step in range(1, stage_b_steps + 1):
            batch     = next(mixed_loader)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            loss, _, _ = moe(input_ids, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = stage_a_steps + step
            if step % 100 == 0 or step == stage_b_steps:
                print(f"    [B] step {step:3d}/{stage_b_steps} "
                      f"(global {global_step}/{total_steps}) | loss={loss.item():.4f}")

    moe.eval()
    return {"stage_a_steps": stage_a_steps, "stage_b_steps": stage_b_steps}


# ============================================================================
# Gate analysis and collapse detection
# ============================================================================

@torch.no_grad()
def eval_router_distribution(moe, held_out_by_domain, device, n_batches=20):
    moe.eval()
    results = {}
    for domain, ds in held_out_by_domain.items():
        loader    = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                               drop_last=True, collate_fn=_collate)
        gate_sums = [0.0] * 4
        count     = 0
        for batch in loader:
            if count >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            _, _, gates = moe(input_ids, labels=labels)
            for i in range(4):
                gate_sums[i] += gates[:, i].mean().item()
            count += 1
        results[domain] = [round(g / max(count, 1), 4) for g in gate_sums]
    return results


def analyze_gate_weights(router_dist: dict) -> dict:
    """
    Examine per-domain gate distributions for collapse.

    For each source domain, checks whether the argmax gate (the specialist that
    receives the most routing weight) is the expected specialist (same index as
    the source domain in DOMAINS). If the argmax is a DIFFERENT specialist AND
    that specialist receives ≥ COLLAPSE_THRESHOLD of the weight, the domain is
    flagged as collapsed.

    Returns a structured dict with per-domain gate breakdowns, collapse flags,
    and an overall COLLAPSE / OK verdict.
    """
    analysis: dict = {
        "domains": {},
        "collapse_detected": False,
        "collapse_domains": [],
        "threshold": COLLAPSE_THRESHOLD,
    }

    for domain_idx, domain in enumerate(DOMAINS):
        gates = router_dist.get(domain, [0.0] * 4)
        argmax_idx   = int(max(range(4), key=lambda i: gates[i]))
        argmax_weight = gates[argmax_idx]
        expected_idx  = domain_idx  # correct specialist for this domain
        is_collapsed  = (argmax_idx != expected_idx and
                         argmax_weight >= COLLAPSE_THRESHOLD)

        domain_info = {
            "gate_weights":    {DOMAINS[i]: gates[i] for i in range(4)},
            "argmax_specialist": DOMAINS[argmax_idx],
            "argmax_weight":     argmax_weight,
            "expected_specialist": DOMAINS[expected_idx],
            "collapsed":          is_collapsed,
        }
        if is_collapsed:
            domain_info["collapse_note"] = (
                f"COLLAPSE: {domain} routes {argmax_weight*100:.1f}% to "
                f"{DOMAINS[argmax_idx]} (expected {DOMAINS[expected_idx]})"
            )
            analysis["collapse_detected"] = True
            analysis["collapse_domains"].append(domain)

        analysis["domains"][domain] = domain_info

    analysis["verdict"] = "COLLAPSE" if analysis["collapse_detected"] else "OK"
    return analysis


def print_gate_analysis(gate_analysis: dict) -> None:
    print("\n  Router gate distribution (per-domain):")
    header = f"  {'Domain':8s}  " + "  ".join(f"{d:>8s}" for d in DOMAINS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for domain in DOMAINS:
        info  = gate_analysis["domains"][domain]
        weights_str = "  ".join(
            f"{info['gate_weights'][d]:>8.4f}" for d in DOMAINS
        )
        collapse_flag = " *** COLLAPSE ***" if info["collapsed"] else ""
        print(f"  {domain:8s}  {weights_str}{collapse_flag}")
    print(f"\n  Gate analysis verdict: {gate_analysis['verdict']}")
    if gate_analysis["collapse_detected"]:
        for d in gate_analysis["collapse_domains"]:
            print(f"    {gate_analysis['domains'][d]['collapse_note']}")


# ============================================================================
# Misc helpers (identical to exp1)
# ============================================================================

def weight_average(models):
    state_dicts = [{k: v.cpu().float() for k, v in m.state_dict().items()} for m in models]
    avg_state   = {k: sum(sd[k] for sd in state_dicts) / len(state_dicts)
                   for k in state_dicts[0]}
    avg_state   = {k: v.to(torch.bfloat16) for k, v in avg_state.items()}
    avg = copy.deepcopy(models[0]).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


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


# ============================================================================
# Run one seed
# ============================================================================

def run_seed(
    seed: int,
    tokenizer,
    device: str,
    train_chunks: dict,
    held_out_chunks: dict,
    stage_a_steps: int,
    stage_b_steps: int,
) -> dict:
    print(f"\n{'='*70}")
    print(f"SEED {seed}  |  Stage-A={stage_a_steps}  Stage-B={stage_b_steps}")
    print(f"{'='*70}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    held_out_sets = {d: chunks_to_dataset(held_out_chunks[d]) for d in DOMAINS}
    eval_matrix   = {}

    # ── Base ─────────────────────────────────────────────────────────────────
    print("\n[base]")
    base = _load_base(device)
    eval_matrix["base"] = eval_all_domains(base, held_out_sets, device,
                                           EVAL_BATCH_SIZE, EVAL_BATCHES)
    del base; torch.cuda.empty_cache()

    # ── Specialists ───────────────────────────────────────────────────────────
    # Reuse checkpoints from exp1 (same CHECKPOINT_DIR, same naming).
    specialists = {}
    for domain in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
        if ckpt.exists():
            print(f"\n[{domain}_spec]  loading {ckpt}")
            spec = _load_checkpoint(ckpt, device)
        else:
            print(f"\n[{domain}_spec]  training (seed={seed}, {MAX_STEPS} steps)...")
            spec = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, dtype=torch.bfloat16, trust_remote_code=True,
            ).to(device)
            train_specialist(spec, domain, train_chunks[domain], seed, device)
            torch.save(spec.state_dict(), ckpt)
            print(f"  Checkpoint saved: {ckpt}")
        specialists[domain] = spec

    for domain, spec in specialists.items():
        print(f"\n[{domain}_spec eval]")
        eval_matrix[f"{domain}_spec"] = eval_all_domains(
            spec, held_out_sets, device, EVAL_BATCH_SIZE, EVAL_BATCHES)

    # ── Weight average ────────────────────────────────────────────────────────
    print("\n[weight_avg]")
    wa = weight_average(list(specialists.values())).to(device)
    eval_matrix["weight_avg"] = eval_all_domains(wa, held_out_sets, device,
                                                  EVAL_BATCH_SIZE, EVAL_BATCHES)
    del wa; torch.cuda.empty_cache()

    # ── MoE (curriculum router) ───────────────────────────────────────────────
    print(f"\n[moe]  building + curriculum router training "
          f"(A={stage_a_steps} + B={stage_b_steps} steps)...")
    moe = FourExpertMoE([specialists[d] for d in DOMAINS], HIDDEN_SIZE).to(device)
    curriculum_info = train_router_curriculum(
        moe, train_chunks, device, stage_a_steps, stage_b_steps
    )
    moe.eval()

    print("\n[moe eval]")
    eval_matrix["moe"] = eval_all_domains(moe, held_out_sets, device,
                                          EVAL_BATCH_SIZE, EVAL_BATCHES, is_fused=True)

    # ── Gate analysis ─────────────────────────────────────────────────────────
    router_dist  = eval_router_distribution(moe, held_out_sets, device)
    gate_analysis = analyze_gate_weights(router_dist)
    print_gate_analysis(gate_analysis)

    del moe
    for spec in specialists.values(): del spec
    torch.cuda.empty_cache()

    # ── Metrics ───────────────────────────────────────────────────────────────
    def eq(k): return eval_matrix[k]["equal_weight_avg"]

    base_eq       = eq("base")
    moe_eq        = eq("moe")
    wa_eq         = eq("weight_avg")
    best_spec_eq  = min(eq(f"{d}_spec") for d in DOMAINS)
    best_spec_dom = min(DOMAINS, key=lambda d: eq(f"{d}_spec"))

    domain_divs = []
    for d in DOMAINS:
        base_d = eval_matrix["base"].get(d, base_eq)
        spec_d = eval_matrix[f"{d}_spec"].get(d, eq(f"{d}_spec"))
        domain_divs.append((base_d - spec_d) / base_d * 100 if base_d > 0 else 0.0)
    mean_divergence = round(statistics.mean(domain_divs), 2)
    gain_vs_spec    = round((best_spec_eq - moe_eq) / best_spec_eq * 100, 4)

    # Perplexity table (key differentiator for cross-lingual)
    perplexity = {}
    for d in DOMAINS:
        perplexity[d] = {
            "base":       round(math.exp(eval_matrix["base"].get(d, 0)), 1),
            "specialist": round(math.exp(eval_matrix[f"{d}_spec"].get(d, 0)), 1),
            "moe":        round(math.exp(eval_matrix["moe"].get(d, 0)), 1),
        }

    metrics = {
        "base_equal_weight":       round(base_eq, 6),
        "best_spec_equal_weight":  round(best_spec_eq, 6),
        "best_spec_domain":        best_spec_dom,
        "weight_avg_equal_weight": round(wa_eq, 6),
        "moe_equal_weight":        round(moe_eq, 6),
        "improvement_vs_spec":     gain_vs_spec,
        "improvement_vs_base":     round((base_eq - moe_eq) / base_eq * 100, 4),
        "mean_divergence":         mean_divergence,
        "per_domain_divergence":   {d: round(domain_divs[i], 2)
                                    for i, d in enumerate(DOMAINS)},
        "perplexity":              perplexity,
    }

    # ── Stop / Go ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"STOP/GO [seed={seed}]:")
    print(f"  Mean divergence: {mean_divergence:.2f}%  |  Fusion gain: {gain_vs_spec:+.2f}%")
    print(f"\n  Perplexity table:")
    print(f"  {'Domain':8s}  {'Base PPL':>9s}  {'Spec PPL':>9s}  {'MoE PPL':>8s}")
    for d in DOMAINS:
        p = perplexity[d]
        print(f"  {d:8s}  {p['base']:>9.1f}  {p['specialist']:>9.1f}  {p['moe']:>8.1f}")

    if mean_divergence > 15 and gain_vs_spec > 7:
        verdict = "GO"
        reason  = "diverge>15% AND gain>7%"
    elif mean_divergence > 15 and gain_vs_spec <= 7:
        verdict = "PIVOT"
        reason  = "diverge>15% but gain<7% — check router"
    else:
        verdict = "STOP"
        reason  = f"mean divergence {mean_divergence:.2f}% < 15%"

    # Append collapse flag to verdict reason when relevant
    if gate_analysis["collapse_detected"]:
        reason += f" | COLLAPSE in {gate_analysis['collapse_domains']}"

    print(f"  → {verdict} ({reason})")
    print(f"{'='*50}")

    result = {
        "seed":             seed,
        "variant":          "curriculum_warm_start",
        "model_id":         MODEL_ID,
        "revision":         REVISION,
        "eval_batch_size":  EVAL_BATCH_SIZE,
        "eval_batches":     EVAL_BATCHES,
        "eval_method":      "per-domain-separate-then-equal-weight-avg",
        "domains":          DOMAINS,
        "eval_matrix":      {k: dict(v) for k, v in eval_matrix.items()},
        "metrics":          metrics,
        "router_distribution": router_dist,
        "gate_analysis":    gate_analysis,
        "stop_go":          {"verdict": verdict, "reason": reason},
        "config": {
            "freeze_layers":      FREEZE_LAYERS,
            "lr":                 LR,
            "max_steps":          MAX_STEPS,
            "batch_size":         BATCH_SIZE,
            "grad_accum":         GRAD_ACCUM,
            "router_stage_a_steps": curriculum_info["stage_a_steps"],
            "router_stage_b_steps": curriculum_info["stage_b_steps"],
            "router_total_steps": curriculum_info["stage_a_steps"] + curriculum_info["stage_b_steps"],
            "router_lr":          ROUTER_LR,
            "n_samples_language": N_SAMPLES_LANGUAGE,
            "n_samples_code":     N_SAMPLES_CODE,
            "collapse_threshold": COLLAPSE_THRESHOLD,
        },
    }
    return result


# ============================================================================
# CLI argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KALAVAI Phase 2 Exp 1 — Cross-Lingual Fusion with Curriculum Warm-Start",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=[42, 137, 2026],
        help="Random seed for specialist training (42, 137, or 2026)",
    )
    parser.add_argument(
        "--stage-a-steps",
        type=int,
        default=100,
        help="Domain-pure warm-start steps (Stage A). Default: 100.",
    )
    parser.add_argument(
        "--stage-b-steps",
        type=int,
        default=400,
        help="Normal mixed-training steps (Stage B). Default: 400.",
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    seed          = args.seed
    stage_a_steps = args.stage_a_steps
    stage_b_steps = args.stage_b_steps
    total_router  = stage_a_steps + stage_b_steps

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"KALAVAI Phase 2 Experiment 1 — Cross-Lingual Fusion (Curriculum Warm-Start)")
    print(f"  model:          {MODEL_ID} @ {REVISION}")
    print(f"  domains:        {DOMAINS}")
    print(f"  freeze:         {FREEZE_LAYERS}")
    print(f"  seed:           {seed}")
    print(f"  stage-a-steps:  {stage_a_steps}  (domain-pure)")
    print(f"  stage-b-steps:  {stage_b_steps}  (mixed)")
    print(f"  total router:   {total_router}")
    print(f"  device:         {device}")
    if device == "cuda":
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_chunks, held_out_chunks = load_all_data(tokenizer)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_path = RESULTS_DIR / f"result_seed{seed}.json"
    if result_path.exists():
        print(f"\n[seed={seed}]  result already exists at {result_path} — skipping.")
        print("  Delete the file to re-run this seed.")
        with open(result_path) as f:
            result = json.load(f)
    else:
        result = run_seed(
            seed, tokenizer, device,
            train_chunks, held_out_chunks,
            stage_a_steps, stage_b_steps,
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {result_path}")

    # ── Single-seed summary ───────────────────────────────────────────────────
    m = result["metrics"]
    ga = result.get("gate_analysis", {})
    print(f"\n{'='*60}")
    print(f"RESULT SUMMARY — seed {seed}")
    print(f"  Gain vs spec:  {m['improvement_vs_spec']:+.2f}%")
    print(f"  Mean div:      {m['mean_divergence']:.2f}%")
    print(f"  Verdict:       {result['stop_go']['verdict']}")
    print(f"  Gate analysis: {ga.get('verdict', 'N/A')}")
    if ga.get("collapse_detected"):
        print(f"  Collapsed:     {ga['collapse_domains']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
