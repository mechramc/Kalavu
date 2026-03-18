#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Heterogeneous Cooperative Simulation (Experiment C1)
=============================================================
Tests whether the protocol is robust to realistic variation in training
conditions across contributors. All experiments at Pythia-410M.

Four conditions (each trains 3 specialists + router, evaluates fusion):

  Condition 1 (control):      all specialists: batch=8, lr=2e-5, steps=2000
  Condition 2 (diff_batch):   code: batch=4/4k-steps, science: 8/2k, fiction: 16/1k
                               (total tokens equivalent across specialists)
  Condition 3 (diff_lr):      code: lr=1e-5, science: lr=2e-5, fiction: lr=5e-5
  Condition 4 (diff_steps):   code: 1k steps, science: 2k steps, fiction: 3k steps

Success criterion: fusion remains within 2pp of control across all
heterogeneity conditions. Proves the protocol tolerates realistic variation.

Runs on RTX 5090 (local). ~4 hours total.
Resumable: each condition writes a result JSON; script skips it on re-run.

Usage:
  python kalavai_heterogeneous_cooperative.py 2>&1 | tee hetero_log.txt
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

MODEL_ID    = "EleutherAI/pythia-410m"
REVISION    = "step10000"
HIDDEN_SIZE = 1024
NUM_LAYERS  = 24
FREEZE_LAYERS = 4
SEQ_LEN     = 512
WARMUP_FRACTION = 0.1
GRADIENT_CLIP   = 1.0
WEIGHT_DECAY    = 0.1
DOMAINS     = ["code", "science", "fiction"]
N_SAMPLES   = 3000
ROUTER_STEPS = 500
ROUTER_LR   = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50
SEED        = 42         # single seed for exploration; 3 seeds on any interesting condition

RESULTS_DIR = Path("results/pythia/heterogeneous_cooperative")
FIGURES_DIR = Path("figures/pythia")

# Per-condition, per-specialist training configs
# Keys: batch_size, grad_accum, max_steps, lr
# All "effective batch = batch_size * grad_accum" — kept at ~8 for control
CONDITIONS = {
    "control": {
        "code":    {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
        "fiction": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},
    },
    "diff_batch": {
        # Total tokens per specialist held equivalent (2*4*512*2000 = 8,192,000)
        "code":    {"batch": 1, "accum": 4, "steps": 4000, "lr": 2e-5},  # eff=4, 2x steps
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},  # eff=8, default
        "fiction": {"batch": 4, "accum": 4, "steps": 1000, "lr": 2e-5},  # eff=16, 0.5x steps
    },
    "diff_lr": {
        "code":    {"batch": 2, "accum": 4, "steps": 2000, "lr": 1e-5},  # conservative
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},  # default
        "fiction": {"batch": 2, "accum": 4, "steps": 2000, "lr": 5e-5},  # aggressive
    },
    "diff_steps": {
        "code":    {"batch": 2, "accum": 4, "steps": 1000, "lr": 2e-5},  # early stop
        "science": {"batch": 2, "accum": 4, "steps": 2000, "lr": 2e-5},  # default
        "fiction": {"batch": 2, "accum": 4, "steps": 3000, "lr": 2e-5},  # extended
    },
}

# ============================================================================
# Resume helpers
# ============================================================================

def result_path(condition: str) -> Path:
    return RESULTS_DIR / f"result_{condition}.json"

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
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }

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
        c = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(c) >= 200: texts.append(c)
        if len(texts) >= n: break
    print(f"  code: {len(texts)} samples")
    return texts

def load_science_texts(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("support","") + "\n" + item.get("question","") + "\n" + item.get("correct_answer","")
        if len(c) > 100: texts.append(c)
        if len(texts) >= n: break
    print(f"  science: {len(texts)} samples")
    return texts

def load_fiction_texts(n):
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        c = item.get("text","")[:5000]
        if len(c) >= 500: texts.append(c)
        if len(texts) >= n: break
    print(f"  fiction: {len(texts)} samples")
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

def apply_freeze(model, n):
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(n):
        model.gpt_neox.layers[i].requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen {n} layers. Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

# ============================================================================
# MoE (N-expert, linear router)
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
        outs = [self._run(s, input_ids) for s in self.specialists]
        logits_list = [o[0] for o in outs]
        h_avg = sum(o[1] for o in outs) / len(outs)
        gates = torch.softmax(self.router(h_avg), dim=-1)
        fused = sum(gates[:, i:i+1, None] * logits_list[i] for i in range(len(self.specialists)))
        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1))
        return loss, fused, gates

# ============================================================================
# Training
# ============================================================================

def train_specialist(model, domain, train_chunks, device, seed,
                     batch_size, grad_accum, max_steps, lr,
                     log_every=50):
    set_seed(seed)
    apply_freeze(model, FREEZE_LAYERS)
    model.train()

    dataset = make_dataset_from_chunks(train_chunks)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         drop_last=True, collate_fn=_collate)
    warmup_steps     = int(max_steps * WARMUP_FRACTION)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps))

    step, accum, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= max_steps: break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / grad_accum
        loss.backward()
        accum += 1
        running_loss += loss.item() * grad_accum

        if accum == grad_accum:
            clip_grad_norm_(trainable_params, GRADIENT_CLIP)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * (step + 1) / warmup_steps
            optimizer.step()
            if step >= warmup_steps: scheduler.step()
            optimizer.zero_grad()
            accum  = 0
            step  += 1
            if step % log_every == 0 or step == max_steps:
                print(f"  [{domain}] {step}/{max_steps} loss={running_loss/step:.4f} "
                      f"({time.time()-t0:.0f}s)")

    model.eval()
    print(f"  {domain} done ({time.time()-t0:.0f}s)")

@torch.no_grad()
def eval_loss(model, dataset, device, batch_size=4, is_fused=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=_collate)
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
            total += loss.item()
            count += 1
    return total / count if count > 0 else float("inf")

def weight_average(specialists):
    states = [{k: v.cpu().float() for k, v in s.state_dict().items()}
              for s in specialists]
    avg_state = {k: sum(s[k] for s in states) / len(states) for k in states[0]}
    avg = copy.deepcopy(specialists[0]).cpu()
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg

def train_router(moe, train_datasets, device):
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
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        loss, _, _ = moe(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router {step}/{ROUTER_STEPS}: loss={loss.item():.4f}")
    moe.eval()

# ============================================================================
# Run one condition
# ============================================================================

def run_condition(condition: str, spec_configs: dict, device: str,
                  tokenizer, all_domain_chunks: dict, base_losses: dict) -> dict:
    rpath = result_path(condition)
    if rpath.exists():
        print(f"\n[skip] {condition} already done.")
        return json.loads(rpath.read_text(encoding="utf-8"))

    print(f"\n{'='*70}")
    print(f"CONDITION: {condition}")
    for d, cfg in spec_configs.items():
        print(f"  {d:8s}: batch={cfg['batch']*cfg['accum']}, "
              f"lr={cfg['lr']:.0e}, steps={cfg['steps']}")
    print(f"{'='*70}")

    # ── Train specialists ─────────────────────────────────────────────────
    specialists = []
    for domain in DOMAINS:
        cfg = spec_configs[domain]
        print(f"\n  Training {domain} (batch={cfg['batch']*cfg['accum']}, "
              f"lr={cfg['lr']:.0e}, steps={cfg['steps']})...")
        model = load_model(device)
        train_specialist(model, domain, all_domain_chunks[domain]["train"],
                         device, SEED, cfg["batch"], cfg["accum"],
                         cfg["steps"], cfg["lr"])
        specialists.append(model)

    # ── Eval datasets ─────────────────────────────────────────────────────
    held_out = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                for d in DOMAINS}
    mixed_held = []
    for d in DOMAINS: mixed_held.extend(all_domain_chunks[d]["held_out"])
    held_out["mixed"] = make_dataset_from_chunks(mixed_held)

    # ── Eval specialists ──────────────────────────────────────────────────
    fusion_losses = {"base": base_losses}
    for label, spec in zip(["code_spec","science_spec","fiction_spec"], specialists):
        fusion_losses[label] = {d: round(eval_loss(spec, ds, device), 6)
                                for d, ds in held_out.items()}

    # ── Weight average ────────────────────────────────────────────────────
    avg = weight_average(specialists)
    avg.to(device)
    fusion_losses["weight_avg"] = {d: round(eval_loss(avg, ds, device), 6)
                                   for d, ds in held_out.items()}
    del avg
    torch.cuda.empty_cache()

    # ── MoE fusion ────────────────────────────────────────────────────────
    train_ds = {d: make_dataset_from_chunks(all_domain_chunks[d]["train"]) for d in DOMAINS}
    moe = NExpertMoE(specialists).to(device)
    train_router(moe, train_ds, device)
    fusion_losses["moe"] = {d: round(eval_loss(moe, ds, device, is_fused=True), 6)
                            for d, ds in held_out.items()}
    del moe
    for s in specialists: del s
    torch.cuda.empty_cache()

    # ── Metrics ───────────────────────────────────────────────────────────
    best_spec = min(fusion_losses["code_spec"]["mixed"],
                    fusion_losses["science_spec"]["mixed"],
                    fusion_losses["fiction_spec"]["mixed"])
    moe_mixed  = fusion_losses["moe"]["mixed"]
    base_mixed = base_losses["mixed"]

    imp_vs_spec = round((best_spec - moe_mixed) / best_spec * 100, 4)
    imp_vs_base = round((base_mixed - moe_mixed) / base_mixed * 100, 4)

    print(f"\n  KEY RESULT ({condition}):")
    print(f"    MoE vs best spec: +{imp_vs_spec:.2f}%")
    print(f"    MoE vs base:      +{imp_vs_base:.2f}%")

    result = {
        "condition":           condition,
        "spec_configs":        spec_configs,
        "eval_heldout":        fusion_losses,
        "base_mixed":          base_mixed,
        "best_spec_mixed":     best_spec,
        "moe_mixed":           moe_mixed,
        "improvement_vs_spec": imp_vs_spec,
        "improvement_vs_base": imp_vs_base,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    git_commit_push(
        f"[kalavai] heterogeneous {condition}: +{imp_vs_spec:.2f}% vs spec"
    )
    return result

# ============================================================================
# Figures
# ============================================================================

def save_figure(results: dict, control_imp: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conditions = list(results.keys())
        improvements = [results[c]["improvement_vs_spec"] for c in conditions]
        colors = ["#2ecc71" if abs(v - control_imp) <= 2.0 else "#e74c3c"
                  for v in improvements]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(len(conditions)), improvements, color=colors, alpha=0.85)
        ax.axhline(control_imp, color="gray", linestyle="--", lw=1.5,
                   label=f"Control ({control_imp:.1f}%)")
        ax.axhspan(control_imp - 2, control_imp + 2, alpha=0.1, color="green",
                   label="±2pp tolerance")
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"+{val:.1f}%", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.set_ylabel("MoE Improvement vs Best Specialist (%)", fontsize=11)
        ax.set_title("Pythia-410M: Protocol Robustness to Heterogeneous Training", fontsize=11)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        path = FIGURES_DIR / "fig_heterogeneous_cooperative.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: figure failed: {e}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVAI: Heterogeneous Cooperative Simulation (C1)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading data...")
    code_texts    = load_code_texts(N_SAMPLES)
    science_texts = load_science_texts(N_SAMPLES)
    fiction_texts = load_fiction_texts(N_SAMPLES)

    # Shared data split (same across all conditions for fair comparison)
    set_seed(42)
    all_domain_chunks = {}
    for domain, texts in [("code", code_texts), ("science", science_texts),
                           ("fiction", fiction_texts)]:
        ds_full = PackedChunkDataset(texts, tokenizer)
        train_c, _, held_c = split_chunks(ds_full.chunks)
        all_domain_chunks[domain] = {"train": train_c, "held_out": held_c}
        print(f"  {domain}: train={len(train_c)}, held_out={len(held_c)}")

    # Base eval (once)
    base_path = RESULTS_DIR / "base_eval.json"
    if base_path.exists():
        base_losses = json.loads(base_path.read_text(encoding="utf-8"))
        print(f"\n[skip] Base eval: mixed={base_losses['mixed']:.4f}")
    else:
        print("\nBase eval...")
        base_model = load_model(device)
        held_out   = {d: make_dataset_from_chunks(all_domain_chunks[d]["held_out"])
                      for d in DOMAINS}
        mixed_held = []
        for d in DOMAINS: mixed_held.extend(all_domain_chunks[d]["held_out"])
        held_out["mixed"] = make_dataset_from_chunks(mixed_held)
        base_losses = {d: round(eval_loss(base_model, ds, device), 6)
                       for d, ds in held_out.items()}
        del base_model
        torch.cuda.empty_cache()
        base_path.write_text(json.dumps(base_losses, indent=2), encoding="utf-8")
        print(f"  Base mixed: {base_losses['mixed']:.4f}")

    # Run all conditions
    all_results = {}
    for condition, spec_configs in CONDITIONS.items():
        result = run_condition(condition, spec_configs, device, tokenizer,
                               all_domain_chunks, base_losses)
        all_results[condition] = result

    # Summary
    control_imp = all_results["control"]["improvement_vs_spec"]
    save_figure(all_results, control_imp)

    summary = {
        "experiment":  "heterogeneous_cooperative",
        "model_id":    MODEL_ID,
        "control_improvement_pct": control_imp,
        "conditions": {
            c: {
                "improvement_vs_spec": r["improvement_vs_spec"],
                "delta_vs_control":    round(r["improvement_vs_spec"] - control_imp, 4),
                "within_2pp":          abs(r["improvement_vs_spec"] - control_imp) <= 2.0,
            }
            for c, r in all_results.items()
        },
        "conclusion": (
            "ROBUST" if all(abs(r["improvement_vs_spec"] - control_imp) <= 2.0
                           for c, r in all_results.items() if c != "control")
            else "SENSITIVE — some conditions deviate >2pp"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    git_commit_push("[kalavai] heterogeneous cooperative C1 COMPLETE")

    print("\n" + "=" * 70)
    print("C1 COMPLETE")
    print("=" * 70)
    print(f"  Control:    +{control_imp:.2f}%")
    for c, r in all_results.items():
        if c == "control": continue
        delta = r["improvement_vs_spec"] - control_imp
        status = "OK" if abs(delta) <= 2.0 else "DEVIANT"
        print(f"  {c:15s}: +{r['improvement_vs_spec']:.2f}% (Δ={delta:+.2f}pp) [{status}]")
    print(f"  Conclusion: {summary['conclusion']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
