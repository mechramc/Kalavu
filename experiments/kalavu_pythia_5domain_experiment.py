#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Pythia-410M Five-Domain Specialist Fusion Experiment
============================================================
5-specialist fusion experiment on Pythia-410M step10000.
Also computes 2/3/4/5 specialist subset fusions to measure scaling.

Domains: code, science, fiction, math, multilingual
Subsets evaluated:
  2_specialists: code, fiction
  3_specialists: code, science, fiction
  4_specialists: code, science, fiction, math
  5_specialists: code, science, fiction, math, multilingual

For code/science/fiction: reuse main experiment checkpoints (seed=42)
For math/multilingual: always train fresh.

Data split: All domains use a single 80/10/10 split on packed chunks.
ALL reported numbers use held_out_chunks only.
"""

import copy
import json
import statistics
import time
from itertools import combinations, cycle
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
FREEZE_LAYERS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000
BATCH_SIZE = 2
GRAD_ACCUM = 4
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE = 1024
DOMAINS_5 = ["code", "science", "fiction", "math", "multilingual"]
SEEDS = [42, 137, 2026]
N_SAMPLES_PER_DOMAIN = 3000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50

RESULTS_DIR = Path("results/pythia/five_domain")
CHECKPOINT_DIR = Path("checkpoints/pythia/five_domain")
FIGURES_DIR = Path("figures/pythia")

# Main experiment checkpoints for code/science/fiction (seed=42)
MAIN_EXP_CHECKPOINT_DIR = Path("checkpoints/pythia")

SUBSETS = {
    "2_specialists": ["code", "fiction"],
    "3_specialists": ["code", "science", "fiction"],
    "4_specialists": ["code", "science", "fiction", "math"],
    "5_specialists": ["code", "science", "fiction", "math", "multilingual"],
}


# ============================================================================
# Dataset
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
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
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def make_dataset_from_chunks(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


def split_chunks(chunks, train_frac=0.8, indist_frac=0.1):
    n = len(chunks)
    return (chunks[:int(n * train_frac)],
            chunks[int(n * train_frac):int(n * (train_frac + indist_frac))],
            chunks[int(n * (train_frac + indist_frac)):])


# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train", streaming=True,
                      trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) > 200:
            texts.append(content)
        if len(texts) >= n: break
    return texts


def load_science_texts(n):
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("support", "") + "\n" + item.get("question", "") + "\n" + item.get("correct_answer", "")
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n: break
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
    return texts


def load_math_texts(n):
    # gsm8k: question + "\n" + answer
    from datasets import load_dataset
    print(f"  Loading math/gsm8k (n={n})...")
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item["question"] + "\n" + item["answer"]
        if len(content) > 50:
            texts.append(content)
        if len(texts) >= n: break
    return texts


def load_multilingual_texts(n):
    # Spanish/French/German Wikipedia — genuinely OOD for English-trained Pythia
    # Uses wikimedia/wikipedia (standard parquet, no legacy loading script)
    from datasets import load_dataset
    for lang_config in ["20231101.es", "20231101.fr", "20231101.de"]:
        try:
            print(f"  Loading multilingual/wikipedia ({lang_config}) (n={n})...")
            ds = load_dataset("wikimedia/wikipedia", lang_config, split="train", streaming=True)
            texts = []
            for item in ds:
                content = item["text"][:3000]
                if len(content) >= 500:
                    texts.append(content)
                if len(texts) >= n:
                    break
            if texts:
                print(f"  Loaded {len(texts)} multilingual texts ({lang_config})")
                return texts
        except Exception as e:
            print(f"  wikipedia {lang_config} failed ({e}), trying next...")
    raise RuntimeError("All multilingual dataset options exhausted")


# ============================================================================
# Freeze (GPT-NeoX architecture)
# ============================================================================

def freeze_first_n_layers(model, n: int):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"  freeze={n}: trainable={trainable/1e6:.1f}M / {total/1e6:.1f}M ({pct:.1f}%)")
    return pct


# ============================================================================
# MoE classes
# ============================================================================

class FiveExpertMoE(nn.Module):
    """MoE with simple linear router over 5 specialist models."""
    def __init__(self, spec_list, hidden_size: int):
        super().__init__()
        assert len(spec_list) == 5, "FiveExpertMoE requires exactly 5 specialists"
        self.specs = nn.ModuleList(spec_list)
        for spec in self.specs:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, 5, bias=False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        logits_list = []
        hidden_list = []
        for spec in self.specs:
            lg, h = self._run(spec, input_ids)
            logits_list.append(lg)
            hidden_list.append(h)

        h_avg = sum(hidden_list) / len(hidden_list)
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, 5)
        fused = sum(gates[:, i:i+1, None] * logits_list[i] for i in range(5))

        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, gates


class FlexibleMoE(nn.Module):
    """MoE with linear router over any number of specialist models."""
    def __init__(self, specialist_list, hidden_size: int):
        super().__init__()
        self.specs = nn.ModuleList(specialist_list)
        self.n_experts = len(specialist_list)
        for spec in self.specs:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, self.n_experts, bias=False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(dim=1).float()

    def forward(self, input_ids, labels=None):
        logits_list = []
        hidden_list = []
        for spec in self.specs:
            lg, h = self._run(spec, input_ids)
            logits_list.append(lg)
            hidden_list.append(h)

        h_avg = sum(hidden_list) / len(hidden_list)
        gates = torch.softmax(self.router(h_avg), dim=-1)  # (B, n_experts)
        fused = sum(gates[:, i:i+1, None] * logits_list[i] for i in range(self.n_experts))

        loss = None
        if labels is not None:
            shift = fused[:, :-1].contiguous()
            shift_l = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), shift_l.view(-1))
        return loss, fused, gates


# ============================================================================
# Training helpers
# ============================================================================

def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_specialist(model, domain, train_chunks, seed, device):
    """Train one specialist. Assumes freeze already applied."""
    set_seed(seed)
    model.train()
    dataset = make_dataset_from_chunks(train_chunks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    warmup_steps = int(MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_STEPS - warmup_steps)

    step = 0
    accum = 0
    running_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS: break
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
            if step % 500 == 0 or step == MAX_STEPS:
                print(f"    [{domain}] step {step}/{MAX_STEPS} | loss {running_loss/step:.4f} | {time.time()-t0:.0f}s")

    print(f"    {domain} done in {time.time()-t0:.0f}s, final loss={running_loss/MAX_STEPS:.4f}")
    return model


def train_router_flexible(moe, combined_train_chunks, device):
    """Train the router on mixed chunks (works for any FlexibleMoE / FiveExpertMoE)."""
    combined = make_dataset_from_chunks(combined_train_chunks)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
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
            print(f"    Router step {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")


@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=4, is_fused=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        if count >= EVAL_BATCHES: break
        ids = batch["input_ids"].to(device)
        lbl = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(ids, labels=lbl)
        else:
            loss = model(input_ids=ids, labels=lbl).loss
        if loss is not None:
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")


def weight_average_n(specialists_list):
    """Average state_dicts of all specialists in the list."""
    avg = copy.deepcopy(specialists_list[0])
    state_dicts = [s.state_dict() for s in specialists_list]
    n = len(state_dicts)
    avg_state = {
        k: (sum(sd[k].float() for sd in state_dicts) / n).to(torch.bfloat16)
        for k in state_dicts[0]
    }
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Figure
# ============================================================================

def save_specialist_scaling_figure(scaling_results_by_seed, subset_names):
    """fig_specialist_scaling.png: improvement % vs number of specialists."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        counts = [int(k.split("_")[0]) for k in subset_names]

        # Aggregate across seeds
        means = []
        stds = []
        for subset_key in subset_names:
            imps = [scaling_results_by_seed[seed][subset_key]["improvement_pct"]
                    for seed in SEEDS
                    if seed in scaling_results_by_seed
                    and subset_key in scaling_results_by_seed[seed]]
            if imps:
                means.append(statistics.mean(imps))
                stds.append(statistics.stdev(imps) if len(imps) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.errorbar(counts, means, yerr=stds, fmt="o-", color="#3498db",
                    linewidth=2.5, markersize=9, capsize=6, elinewidth=2,
                    label="Mean improvement (3 seeds)")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

        for x, y, s in zip(counts, means, stds):
            ax.annotate(f"{y:+.1f}%", (x, y),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9)

        ax.set_xlabel("Number of Specialists", fontsize=12)
        ax.set_ylabel("Improvement over Best Individual on Mixed Held-Out (%)", fontsize=11)
        ax.set_title("Fusion improvement vs number of cooperative specialists\n(Pythia-410M step10000)", fontsize=12)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(c) for c in counts])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_specialist_scaling.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: Could not save specialist scaling figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU: Pythia-410M Five-Domain Specialist Fusion Experiment")
    print("=" * 70)
    print(f"Model:    {MODEL_ID} @ revision={REVISION}")
    print(f"Domains:  {DOMAINS_5}")
    print(f"Seeds:    {SEEDS}")
    print(f"Subsets:  {list(SUBSETS.keys())}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load tokenizer + data for all 5 domains ONCE
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data for all 5 domains...")
    domain_loaders = {
        "code":          load_code_texts,
        "science":       load_science_texts,
        "fiction":       load_fiction_texts,
        "math":          load_math_texts,
        "multilingual":  load_multilingual_texts,
    }
    domain_texts = {}
    for domain, loader_fn in domain_loaders.items():
        domain_texts[domain] = loader_fn(N_SAMPLES_PER_DOMAIN)

    print("\nPacking and splitting chunks (80/10/10)...")
    all_domain_chunks = {}
    for domain in DOMAINS_5:
        ds_full = PackedChunkDataset(domain_texts[domain], tokenizer,
                                     seq_len=SEQ_LEN, max_chars=5000)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # Build held-out datasets per domain and mixed
    held_out_sets = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                     for d in DOMAINS_5}
    mixed_held_all = []
    for d in DOMAINS_5:
        mixed_held_all.extend(all_domain_chunks[d]["held_out"])
    held_out_sets["mixed_all"] = make_dataset_from_chunks(mixed_held_all)

    # Subset mixed held-out sets (for evaluating subset fusions)
    subset_mixed_held = {}
    for subset_key, subset_domains in SUBSETS.items():
        mixed_held = []
        for d in subset_domains:
            mixed_held.extend(all_domain_chunks[d]["held_out"])
        subset_mixed_held[subset_key] = make_dataset_from_chunks(mixed_held)

    # ------------------------------------------------------------------
    # Eval base model on all held-out domains
    # ------------------------------------------------------------------
    print(f"\nEvaluating base model ({MODEL_ID} @ {REVISION}) on held-out data...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.eval()

    base_losses = {}
    for domain in DOMAINS_5:
        l = eval_loss(base_model, held_out_sets[domain], device)
        base_losses[domain] = round(l, 6)
        print(f"  Base [{domain:14s}]: {l:.4f}")
    base_mixed_all = eval_loss(base_model, held_out_sets["mixed_all"], device)
    base_losses["mixed_all"] = round(base_mixed_all, 6)
    print(f"  Base [mixed_all    ]: {base_mixed_all:.4f}")

    del base_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Per-seed loop
    # ------------------------------------------------------------------
    all_seed_results = {}
    scaling_results_by_seed = {}
    partial_path = RESULTS_DIR / "five_domain_partial.json"

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        specialists = {}

        # --- Load or train all 5 specialists ---
        for domain in DOMAINS_5:
            ckpt_path = CHECKPOINT_DIR / f"{domain}_seed{seed}.pt"

            # For code/science/fiction: try main experiment checkpoints
            if domain in ["code", "science", "fiction"]:
                main_ckpt = MAIN_EXP_CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"
                if main_ckpt.exists() and not ckpt_path.exists():
                    import shutil
                    shutil.copy(main_ckpt, ckpt_path)
                    print(f"  Copied main-exp checkpoint: {main_ckpt.name} -> {ckpt_path.name}")

            if ckpt_path.exists():
                print(f"  Loading existing checkpoint: {ckpt_path}")
                spec = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
                ).to(device)
                spec.load_state_dict(torch.load(ckpt_path, map_location=device))
                spec.eval()
            else:
                print(f"  Training {domain} specialist (seed={seed})...")
                spec = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
                ).to(device)
                freeze_first_n_layers(spec, FREEZE_LAYERS)
                train_specialist(spec, domain, all_domain_chunks[domain]["train"], seed, device)
                spec.eval()
                torch.save(spec.state_dict(), ckpt_path)
                print(f"  Saved: {ckpt_path}")

            specialists[domain] = spec

        # --- Eval each specialist on its own domain ---
        print("\n  Specialist divergence check...")
        spec_losses = {}
        for domain in DOMAINS_5:
            l = eval_loss(specialists[domain], held_out_sets[domain], device)
            spec_losses[domain] = round(l, 6)
            beats = l < base_losses[domain]
            sym = "+" if beats else "-"
            print(f"    [{sym}] {domain}: spec={l:.4f} vs base={base_losses[domain]:.4f} "
                  f"({(l - base_losses[domain]) / base_losses[domain] * 100:+.1f}%)")

        # --- Subset fusion runs ---
        print("\n  Running subset fusions...")
        scaling_results_by_seed[seed] = {}
        subset_names_ordered = list(SUBSETS.keys())

        for subset_key, subset_domains in SUBSETS.items():
            n_exp = len(subset_domains)
            print(f"\n  --- {subset_key} ({subset_domains}) ---")

            # Build specialist list for this subset
            spec_list = [specialists[d] for d in subset_domains]

            # Combined train chunks for this subset
            subset_train = []
            for d in subset_domains:
                subset_train.extend(all_domain_chunks[d]["train"])

            # Best individual on subset mixed held-out
            best_individual = min(
                eval_loss(specialists[d], subset_mixed_held[subset_key], device)
                for d in subset_domains
            )

            # Weight average
            print(f"  Computing {n_exp}-way weight average...")
            wa = weight_average_n(spec_list).to(device)
            wa_loss = eval_loss(wa, subset_mixed_held[subset_key], device)
            del wa
            torch.cuda.empty_cache()

            # Flexible MoE with router
            print(f"  Building FlexibleMoE (n_experts={n_exp}) and training router...")
            # Need fresh model references (specs already on device, frozen)
            moe = FlexibleMoE(spec_list, hidden_size=HIDDEN_SIZE).to(device)
            train_router_flexible(moe, subset_train, device)
            moe.eval()

            moe_loss = eval_loss(moe, subset_mixed_held[subset_key], device,
                                 batch_size=2, is_fused=True)
            improvement = (best_individual - moe_loss) / best_individual * 100

            print(f"  {subset_key}: best_indiv={best_individual:.4f}, "
                  f"wa={wa_loss:.4f}, moe={moe_loss:.4f}, "
                  f"improvement={improvement:+.1f}%")

            scaling_results_by_seed[seed][subset_key] = {
                "subset_domains": subset_domains,
                "n_specialists": n_exp,
                "best_individual_mixed": round(best_individual, 6),
                "weight_avg_mixed": round(wa_loss, 6),
                "moe_mixed_loss": round(moe_loss, 6),
                "improvement_pct": round(improvement, 4),
            }

            del moe
            torch.cuda.empty_cache()

        # --- Free specialists ---
        for s in specialists.values():
            del s
        torch.cuda.empty_cache()

        # Save per-seed results
        seed_result = {
            "seed": seed,
            "base_losses": base_losses,
            "specialist_losses": spec_losses,
            "subset_results": scaling_results_by_seed[seed],
        }
        all_seed_results[seed] = seed_result

        out_path = RESULTS_DIR / f"five_domain_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(seed_result, f, indent=2)
        print(f"\n  Saved: {out_path}")

        with open(partial_path, "w") as f:
            json.dump({"partial_seeds": list(all_seed_results.keys()),
                       "results": {str(k): v for k, v in all_seed_results.items()}}, f, indent=2)

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FIVE-DOMAIN SPECIALIST SCALING — AGGREGATED RESULTS")
    print("=" * 70)

    subset_names_ordered = list(SUBSETS.keys())
    print(f"\n{'Subset':<20} {'N_exp':>6} {'Mean Improvement':>18} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 72)

    aggregate = {}
    for subset_key in subset_names_ordered:
        n_exp = len(SUBSETS[subset_key])
        imps = [scaling_results_by_seed[seed][subset_key]["improvement_pct"]
                for seed in SEEDS if seed in scaling_results_by_seed
                and subset_key in scaling_results_by_seed[seed]]
        if imps:
            mean_imp = statistics.mean(imps)
            std_imp = statistics.stdev(imps) if len(imps) > 1 else 0.0
            aggregate[subset_key] = {
                "n_specialists": n_exp,
                "domains": SUBSETS[subset_key],
                "improvement_mean_pct": round(mean_imp, 4),
                "improvement_std_pct": round(std_imp, 4),
                "improvement_min_pct": round(min(imps), 4),
                "improvement_max_pct": round(max(imps), 4),
                "per_seed": {str(seed): scaling_results_by_seed[seed].get(subset_key, {})
                             for seed in SEEDS},
            }
            print(f"{subset_key:<20} {n_exp:>6} {mean_imp:>+17.1f}% {std_imp:>7.1f}% "
                  f"{min(imps):>+7.1f}% {max(imps):>+7.1f}%")

    # Build scaling commit message
    scaling_str = " | ".join(
        f"{k}: {aggregate[k]['improvement_mean_pct']:+.1f}%"
        for k in subset_names_ordered if k in aggregate
    )
    print(f"\n[kalavu] 5-domain: 2/3/4/5 specialist scaling = {scaling_str}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    print("\nSaving specialist scaling figure...")
    save_specialist_scaling_figure(scaling_results_by_seed, subset_names_ordered)

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "experiment": "five_domain_specialist_scaling",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "domains_5": DOMAINS_5,
        "config": {
            "freeze_layers": FREEZE_LAYERS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "seq_len": SEQ_LEN,
            "warmup_fraction": WARMUP_FRACTION,
            "n_samples_per_domain": N_SAMPLES_PER_DOMAIN,
            "router_steps": ROUTER_STEPS,
            "eval_batches": EVAL_BATCHES,
        },
        "seeds": SEEDS,
        "subsets": SUBSETS,
        "base_losses": base_losses,
        "aggregate_scaling": aggregate,
        "per_seed_results": {str(seed): {
            "specialist_losses": all_seed_results[seed]["specialist_losses"],
            "subset_results": {
                k: scaling_results_by_seed[seed].get(k, {})
                for k in subset_names_ordered
            },
        } for seed in SEEDS},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")

    if partial_path.exists():
        partial_path.unlink()

    print(f"\n[kalavu] 5-domain: 2/3/4/5 specialist scaling = {scaling_str}")


if __name__ == "__main__":
    main()
