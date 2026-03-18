#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-1B Five-Domain Specialist Scaling (Experiment D1)
================================================================
Replicates the 5-domain specialist count scaling experiment at 1B scale.
Tests whether fusion improvement holds at 1B with 3, 4, and 5 specialists.

Domains: code, science, fiction, math (GSM8K), multilingual (Spanish Wikipedia)
Subsets:
  3_specialists: code, science, fiction
  4_specialists: code, science, fiction, math
  5_specialists: code, science, fiction, math, multilingual

Uses existing 1B specialist checkpoints (seed=42, step10000) if available.
Always trains math and multilingual fresh.

Success criterion: 3-5 specialists all achieve ~+14% with near-zero variance,
consistent with the 410M result (which showed 14.12%-14.15% at 3-5 specialists).

Runs on RTX 5090 or A100. ~6 hours.
Resumable: each subset writes a result JSON.

Usage:
  python kalavai_1b_5domain_experiment.py 2>&1 | tee 1b_5domain_log.txt
"""

import copy
import json
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

MODEL_ID    = "EleutherAI/pythia-1b"
REVISION    = "step10000"
HIDDEN_SIZE = 2048
NUM_LAYERS  = 16
FREEZE_LAYERS = 4
LR          = 2e-5
WEIGHT_DECAY = 0.1
MAX_STEPS   = 2000
BATCH_SIZE  = 2
GRAD_ACCUM  = 4
GRADIENT_CLIP = 1.0
SEQ_LEN     = 512
WARMUP_FRACTION = 0.1
N_SAMPLES   = 3000
ROUTER_STEPS = 500
ROUTER_LR   = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
SEEDS       = [42, 137, 2026]

DOMAINS_5 = ["code", "science", "fiction", "math", "multilingual"]
SUBSETS   = {
    "3_specialists": ["code", "science", "fiction"],
    "4_specialists": ["code", "science", "fiction", "math"],
    "5_specialists": ["code", "science", "fiction", "math", "multilingual"],
}

RESULTS_DIR     = Path("results/pythia/pythia_1b_5domain")
FIGURES_DIR     = Path("figures/pythia")
CHECKPOINT_DIR  = Path("checkpoints/pythia_1b_5domain")
# Reuse main 1B checkpoints for code/science/fiction seed=42
MAIN_1B_CKPT_DIR = Path("checkpoints/pythia_1b")

# ============================================================================
# Helpers
# ============================================================================

def result_path(subset: str, seed: int) -> Path:
    return RESULTS_DIR / f"result_{subset}_seed{seed}.json"

def specialist_ckpt(domain: str, seed: int) -> Path:
    if domain in ["code", "science", "fiction"]:
        p = MAIN_1B_CKPT_DIR / f"{domain}_specialist_seed{seed}.pt"
        if p.exists():
            return p
    return CHECKPOINT_DIR / f"{domain}_specialist_seed{seed}.pt"

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
    return {"input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels":    torch.stack([b["labels"]    for b in batch])}

def make_dataset_from_chunks(chunks):
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds

def split_chunks(chunks, train_frac=0.8, indist_frac=0.1):
    n = len(chunks)
    a = int(n * train_frac)
    b = int(n * (train_frac + indist_frac))
    return chunks[:a], chunks[a:b], chunks[b:]

# ============================================================================
# Data loading
# ============================================================================

def load_code_texts(n):
    from datasets import load_dataset
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        c = item.get("whole_func_string","") or item.get("func_code_string","")
        if len(c) >= 200: texts.append(c)
        if len(texts) >= n: break
    print(f"  code: {len(texts)}")
    return texts

def load_science_texts(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("support","") + "\n" + item.get("question","") + "\n" + item.get("correct_answer","")
        if len(c) > 100: texts.append(c)
        if len(texts) >= n: break
    print(f"  science: {len(texts)}")
    return texts

def load_fiction_texts(n):
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("text","")[:5000]
        if len(c) >= 500: texts.append(c)
        if len(texts) >= n: break
    print(f"  fiction: {len(texts)}")
    return texts

def load_math_texts(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("question","") + "\n" + item.get("answer","")
        if len(c) > 50: texts.append(c)
        if len(texts) >= n: break
    print(f"  math: {len(texts)}")
    return texts

def load_multilingual_texts(n):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train",
                      streaming=True)
    texts = []
    for item in ds:
        c = item.get("text","")[:5000]
        if len(c) >= 300: texts.append(c)
        if len(texts) >= n: break
    print(f"  multilingual: {len(texts)}")
    return texts

# ============================================================================
# Model
# ============================================================================

def load_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return model

def load_model_with_weights(ckpt_path: Path, device) -> "AutoModelForCausalLM":
    model = load_model(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def apply_freeze(model, n):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen {n} layers. Trainable: {trainable/1e6:.0f}M/{total/1e6:.0f}M")

# ============================================================================
# MoE (N-expert)
# ============================================================================

class NExpertMoE(nn.Module):
    def __init__(self, specialists: list, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.specialists = nn.ModuleList(specialists)
        for spec in self.specialists:
            for p in spec.parameters():
                p.requires_grad_(False)
        self.router = nn.Linear(hidden_size, len(specialists), bias=False)

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.float(), out.hidden_states[-1].float().mean(dim=1)

    def forward(self, input_ids, labels=None):
        outs  = [self._run(s, input_ids) for s in self.specialists]
        llist = [o[0] for o in outs]
        h_avg = sum(o[1] for o in outs) / len(outs)
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(gates[:, i:i+1, None] * llist[i] for i in range(len(self.specialists)))
        loss = None
        if labels is not None:
            sl = fused[:, :-1].contiguous()
            ll = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), ll.view(-1))
        return loss, fused, gates

# ============================================================================
# Training
# ============================================================================

def train_specialist(model, domain, train_chunks, device, seed, log_every=50):
    set_seed(seed)
    apply_freeze(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, collate_fn=_collate)
    warmup_steps     = int(MAX_STEPS * WARMUP_FRACTION)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, MAX_STEPS - warmup_steps))

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MAX_STEPS: break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss  = model(**batch).loss / GRAD_ACCUM
        loss.backward()
        accum += 1
        running_loss += loss.item() * GRAD_ACCUM

        if accum == GRAD_ACCUM:
            clip_grad_norm_(trainable_params, GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps: scheduler.step()
            optimizer.zero_grad()
            accum = 0; step += 1
            if step % log_every == 0 or step == MAX_STEPS:
                print(f"  [{domain}] {step}/{MAX_STEPS} loss={running_loss/step:.4f} "
                      f"({time.time()-t0:.0f}s)")

    model.eval()

@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=2, is_fused=False):
    g = torch.Generator(); g.manual_seed(999)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, collate_fn=_collate, generator=g)
    total, count = 0.0, 0
    model.eval()
    for batch in loader:
        if count >= EVAL_BATCHES: break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        if is_fused:
            loss, _, _ = model(input_ids, labels=labels)
        else:
            loss = model(input_ids=input_ids, labels=labels).loss
        if loss is not None:
            total += loss.item(); count += 1
    return total / count if count > 0 else float("inf")

def weight_average(specialists):
    states = [{k: v.cpu().float() for k, v in s.state_dict().items()} for s in specialists]
    avg_s  = {k: sum(s[k] for s in states) / len(states) for k in states[0]}
    avg = copy.deepcopy(specialists[0]).cpu()
    avg.load_state_dict(avg_s); avg.eval()
    return avg

def train_router(moe, train_datasets, device):
    all_chunks = []
    for ds in train_datasets.values(): all_chunks.extend(ds.chunks)
    combined  = make_dataset_from_chunks(all_chunks)
    moe.router = moe.router.to(device)
    optimizer  = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    loader     = DataLoader(combined, batch_size=ROUTER_BATCH, shuffle=True,
                            drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    moe.train()
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Run one subset
# ============================================================================

def run_subset(subset_name: str, domains: list, seed: int, device: str,
               tokenizer, all_domain_chunks: dict, base_losses: dict) -> dict:
    rpath = result_path(subset_name, seed)
    if rpath.exists():
        print(f"\n[skip] {subset_name} seed={seed} already done.")
        return json.loads(rpath.read_text(encoding="utf-8"))

    print(f"\n{'='*70}")
    print(f"SUBSET: {subset_name} | domains={domains} | seed={seed}")
    print(f"{'='*70}")

    # ── Train or load specialists ─────────────────────────────────────────
    specialists = []
    for domain in domains:
        ckpt = specialist_ckpt(domain, seed)
        if ckpt.exists():
            print(f"\n  Loading {domain} specialist from {ckpt}...")
            model = load_model_with_weights(ckpt, device)
        else:
            print(f"\n  Training {domain} specialist (seed={seed})...")
            model = load_model(device)
            train_specialist(model, domain, all_domain_chunks[domain]["train"],
                             device, seed)
            # Save for reuse
            if seed == 42:
                ckpt_dir = CHECKPOINT_DIR
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_save = ckpt_dir / f"{domain}_specialist_seed{seed}.pt"
                torch.save(model.state_dict(), ckpt_save)
                print(f"  Saved: {ckpt_save}")
        specialists.append(model)

    # ── Eval datasets — only the subset's domains + mixed ────────────────
    held_out = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                for d in domains}
    mixed_held = []
    for d in domains: mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out["mixed"] = make_dataset_from_chunks(mixed_held)

    # ── Eval specialists ──────────────────────────────────────────────────
    fusion_losses = {}
    fusion_losses["base"] = {d: round(eval_loss(
        AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
                                              trust_remote_code=True,
                                              dtype=torch.bfloat16).to(device),
        ds, device), 6) for d, ds in held_out.items()}
    # Reload base fresh per subset (to keep GPU clean)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True,
        dtype=torch.bfloat16).to(device)
    base_model.eval()
    fusion_losses["base"] = {d: round(eval_loss(base_model, ds, device), 6)
                             for d, ds in held_out.items()}
    del base_model; torch.cuda.empty_cache()

    for i, (domain, spec) in enumerate(zip(domains, specialists)):
        fusion_losses[f"{domain}_spec"] = {
            d: round(eval_loss(spec, ds, device), 6) for d, ds in held_out.items()
        }

    # ── Weight average ────────────────────────────────────────────────────
    avg = weight_average(specialists)
    avg.to(device)
    fusion_losses["weight_avg"] = {d: round(eval_loss(avg, ds, device), 6)
                                   for d, ds in held_out.items()}
    del avg; torch.cuda.empty_cache()

    # ── MoE ───────────────────────────────────────────────────────────────
    train_ds = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"]) for d in domains}
    moe = NExpertMoE(specialists).to(device)
    train_router(moe, train_ds, device)
    fusion_losses["moe"] = {d: round(eval_loss(moe, ds, device, is_fused=True), 6)
                            for d, ds in held_out.items()}
    del moe
    for s in specialists: del s
    torch.cuda.empty_cache()

    # ── Metrics ───────────────────────────────────────────────────────────
    best_spec = min(fusion_losses[f"{d}_spec"]["mixed"] for d in domains)
    moe_mixed  = fusion_losses["moe"]["mixed"]
    base_mixed = fusion_losses["base"]["mixed"]

    imp_vs_spec = round((best_spec - moe_mixed) / best_spec * 100, 4)
    imp_vs_base = round((base_mixed - moe_mixed) / base_mixed * 100, 4)

    print(f"\n  KEY RESULT ({subset_name}, seed={seed}):")
    print(f"    Best spec: {best_spec:.4f}, MoE: {moe_mixed:.4f}, Base: {base_mixed:.4f}")
    print(f"    vs spec: +{imp_vs_spec:.2f}%  vs base: +{imp_vs_base:.2f}%")

    result = {
        "subset":              subset_name,
        "domains":             domains,
        "n_specialists":       len(domains),
        "seed":                seed,
        "eval_heldout":        fusion_losses,
        "best_spec_mixed":     best_spec,
        "moe_mixed":           moe_mixed,
        "base_mixed":          base_mixed,
        "improvement_vs_spec": imp_vs_spec,
        "improvement_vs_base": imp_vs_base,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    git_commit_push(
        f"[kalavai] 1B 5-domain {subset_name} seed={seed}: +{imp_vs_spec:.2f}% vs spec"
    )
    return result

# ============================================================================
# Figures
# ============================================================================

def save_figure(seed_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        subset_names = ["3_specialists", "4_specialists", "5_specialists"]
        n_specs = [3, 4, 5]
        means, stds = [], []
        for s in subset_names:
            vals = [seed_results[s][seed]["improvement_vs_spec"]
                    for seed in SEEDS if seed in seed_results[s]]
            means.append(sum(vals)/len(vals))
            stds.append((sum((v - means[-1])**2 for v in vals)/len(vals))**0.5)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(n_specs, means, yerr=stds, fmt="o-", color="#9b59b6",
                    lw=2, capsize=5, markersize=8)
        for n, m in zip(n_specs, means):
            ax.annotate(f"+{m:.1f}%", (n, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        ax.set_xlabel("Number of Specialists", fontsize=12)
        ax.set_ylabel("MoE Improvement vs Best Specialist (%)", fontsize=11)
        ax.set_title("Pythia-1B: Fusion Improvement vs Specialist Count", fontsize=11)
        ax.set_xticks(n_specs)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = FIGURES_DIR / "fig_1b_5domain_scaling.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: figure failed: {e}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Pythia-1B Five-Domain Scaling (D1)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    texts = {
        "code":         load_code_texts(N_SAMPLES),
        "science":      load_science_texts(N_SAMPLES),
        "fiction":      load_fiction_texts(N_SAMPLES),
        "math":         load_math_texts(N_SAMPLES),
        "multilingual": load_multilingual_texts(N_SAMPLES),
    }

    set_seed(42)
    all_domain_chunks = {}
    for domain, t in texts.items():
        ds_full = PackedChunkDataset(t, tokenizer)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # Base eval (per domain, used by each subset)
    base_path = RESULTS_DIR / "base_eval.json"
    if base_path.exists():
        base_losses = json.loads(base_path.read_text(encoding="utf-8"))
        print(f"\n[skip] Base eval loaded.")
    else:
        base_model = load_model(device)
        all_held = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                    for d in DOMAINS_5}
        base_losses = {d: round(eval_loss(base_model, ds, device), 6)
                       for d, ds in all_held.items()}
        del base_model; torch.cuda.empty_cache()
        base_path.write_text(json.dumps(base_losses, indent=2), encoding="utf-8")
        print(f"  Base losses: {base_losses}")

    # Run all subsets × 3 seeds
    seed_results = {subset: {} for subset in SUBSETS}

    for subset_name, domains in SUBSETS.items():
        for seed in SEEDS:
            result = run_subset(subset_name, domains, seed, device, tokenizer,
                                all_domain_chunks, base_losses)
            seed_results[subset_name][seed] = result

    # Summary
    save_figure(seed_results)

    summary_rows = []
    for subset_name, domains in SUBSETS.items():
        vals = [seed_results[subset_name][s]["improvement_vs_spec"] for s in SEEDS]
        mean = round(sum(vals)/len(vals), 4)
        std  = round((sum((v-mean)**2 for v in vals)/len(vals))**0.5, 4)
        summary_rows.append({
            "subset": subset_name, "n_specialists": len(domains),
            "domains": domains, "mean_improvement_vs_spec": mean,
            "std_improvement_vs_spec": std, "per_seed": vals,
        })
        print(f"  {subset_name}: +{mean:.2f}% ± {std:.2f}%")

    summary = {
        "experiment": "1b_5domain_scaling",
        "model_id": MODEL_ID, "revision": REVISION,
        "results": summary_rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    git_commit_push("[kalavai] 1B 5-domain D1 COMPLETE")

    print("\n" + "=" * 70)
    print("D1 COMPLETE")
    print("=" * 70)
    for row in summary_rows:
        print(f"  {row['subset']}: +{row['mean_improvement_vs_spec']:.2f}% ± {row['std_improvement_vs_spec']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
