#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Extract per-step divergence-gain regression points from crossover checkpoints.

For each training step count [50,100,500,1000,2000,5000,10000,20000]:
  1. Load freeze=0 specialist checkpoints for code, science, fiction
  2. Evaluate per-domain EW loss on held-out test data
  3. Compute divergence = (base_loss - spec_loss) / base_loss * 100 per domain, mean
  4. Read MoE EW loss from training_duration_crossover_corrected.json
  5. Compute gain_vs_spec = (best_spec_ew - moe_ew_loss) / best_spec_ew * 100

Output: results/pythia/crossover_regression_points.json
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Paths
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "pythia"
RESULTS_DIR = REPO_ROOT / "results" / "pythia"
CROSSOVER_JSON = RESULTS_DIR / "training_duration_crossover_corrected.json"
OUTPUT_JSON = RESULTS_DIR / "crossover_regression_points.json"

# Add experiments dir to path for kalavai_eval_utils
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from kalavai_eval_utils import eval_all_domains, PackedChunkDataset, chunks_to_dataset

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
STEP_COUNTS = [50, 100, 500, 1000, 2000, 5000, 10000, 20000]
DOMAINS = ["code", "science", "fiction"]
FREEZE = 0
SEED = 42
SEQ_LEN = 512

# Calibrated base losses used throughout the paper
BASE_LOSSES = {
    "code": 2.087168,
    "science": 2.89201,
    "fiction": 2.973911,
}

# base_ew_loss from the crossover JSON
BASE_EW_LOSS = 2.65103

BS = 4
EVAL_BATCHES = 50
N_SAMPLES = 3000  # samples to load; we take test split


# ============================================================================
# Data loading — held-out test splits matching paper protocol
# ============================================================================

def load_held_out_datasets(tokenizer):
    """
    Load held-out test data for each domain.
    Uses 90/10 train/test split of packed chunks (take last 10%).
    Domains:
      code    : CodeSearchNet Python, field "whole_func_string", split="test"
      science : allenai/sciq, field "support" + " " + "question", split="test"
      fiction : pg19, field "text", split="test", truncate to 3000 chars
    """
    from datasets import load_dataset

    held_out = {}

    # --- Code ---
    print("  Loading code from code_search_net python (split=test)...")
    code_ds = load_dataset(
        "code_search_net", "python", split="test",
        streaming=True, trust_remote_code=True
    )
    code_texts = []
    for item in code_ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) > 200:
            code_texts.append(content)
        if len(code_texts) >= N_SAMPLES:
            break
    print(f"    Got {len(code_texts)} code samples")
    full_code_ds = PackedChunkDataset(code_texts, tokenizer, seq_len=SEQ_LEN)
    n = len(full_code_ds.chunks)
    split_idx = int(n * 0.9)
    held_out["code"] = chunks_to_dataset(full_code_ds.chunks[split_idx:])
    print(f"    Held-out code chunks: {len(held_out['code'])}")

    # --- Science ---
    print("  Loading science from allenai/sciq (split=test)...")
    sci_ds = load_dataset("allenai/sciq", split="test", streaming=True)
    sci_texts = []
    for item in sci_ds:
        content = item.get("support", "") + " " + item.get("question", "")
        if len(content) > 50:
            sci_texts.append(content)
        if len(sci_texts) >= N_SAMPLES:
            break
    print(f"    Got {len(sci_texts)} science samples")
    full_sci_ds = PackedChunkDataset(sci_texts, tokenizer, seq_len=SEQ_LEN)
    n = len(full_sci_ds.chunks)
    split_idx = int(n * 0.9)
    held_out["science"] = chunks_to_dataset(full_sci_ds.chunks[split_idx:])
    print(f"    Held-out science chunks: {len(held_out['science'])}")

    # --- Fiction ---
    print("  Loading fiction from pg19 (split=test)...")
    pg_ds = load_dataset("emozilla/pg19", split="test", streaming=True,
                         trust_remote_code=True)
    fic_texts = []
    for item in pg_ds:
        raw = item.get("text", "")
        content = raw[:3000]
        if len(content) >= 500:
            fic_texts.append(content)
        if len(fic_texts) >= N_SAMPLES:
            break
    print(f"    Got {len(fic_texts)} fiction samples")
    full_fic_ds = PackedChunkDataset(fic_texts, tokenizer, seq_len=SEQ_LEN)
    n = len(full_fic_ds.chunks)
    split_idx = int(n * 0.9)
    held_out["fiction"] = chunks_to_dataset(full_fic_ds.chunks[split_idx:])
    print(f"    Held-out fiction chunks: {len(held_out['fiction'])}")

    return held_out


# ============================================================================
# Checkpoint loading
# ============================================================================

def load_specialist_model(domain: str, steps: int, base_model_cache):
    """
    Load a specialist checkpoint by patching the base model state_dict.
    base_model_cache is a dict we populate on first call to avoid repeated downloads.
    """
    ckpt_name = f"crossover_{domain}_freeze{FREEZE}_steps{steps}_seed{SEED}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load or reuse cached base model weights (don't mutate the cache)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load crossover JSON for MoE losses
    with open(CROSSOVER_JSON) as f:
        crossover_data = json.load(f)

    step_to_moe_loss = {
        s: l for s, l in zip(crossover_data["steps"], crossover_data["freeze0_loss"])
    }

    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    # Load held-out datasets once
    print("\nLoading held-out test datasets...")
    held_out = load_held_out_datasets(tokenizer)

    regression_points = []

    for step_idx, steps in enumerate(STEP_COUNTS):
        print(f"\n{'='*60}")
        print(f"Step count: {steps}  ({step_idx+1}/{len(STEP_COUNTS)})")
        print(f"{'='*60}")

        spec_losses = {}  # domain -> ew_loss

        for domain in DOMAINS:
            print(f"  Evaluating {domain} specialist at steps={steps}...")
            model = load_specialist_model(domain, steps, base_model_cache={})
            model = model.to(device)
            model.eval()

            # Evaluate only on this domain's held-out set
            domain_held_out = {domain: held_out[domain]}
            results = eval_all_domains(
                model, domain_held_out, device, bs=BS,
                eval_batches=EVAL_BATCHES, is_fused=False
            )
            spec_losses[domain] = results[domain]
            print(f"    {domain} specialist loss: {spec_losses[domain]:.6f}")

            # Free GPU memory
            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Per-domain divergence
        per_domain_div = {}
        for domain in DOMAINS:
            base_loss_d = BASE_LOSSES[domain]
            spec_loss_d = spec_losses[domain]
            div = (base_loss_d - spec_loss_d) / base_loss_d * 100.0
            per_domain_div[domain] = round(div, 6)

        mean_divergence = round(sum(per_domain_div.values()) / len(DOMAINS), 6)

        # Best specialist EW (min loss = best performance)
        best_spec_ew_loss = round(min(spec_losses.values()), 6)

        # MoE EW loss from JSON
        moe_ew_loss = step_to_moe_loss[steps]

        # Gain vs best specialist
        gain_vs_spec_pct = round(
            (best_spec_ew_loss - moe_ew_loss) / best_spec_ew_loss * 100.0, 6
        )

        point = {
            "label": f"crossover_{steps}steps",
            "steps": steps,
            "mean_divergence": mean_divergence,
            "per_domain_divergence": per_domain_div,
            "best_spec_ew_loss": best_spec_ew_loss,
            "moe_ew_loss": round(moe_ew_loss, 6),
            "gain_vs_spec_pct": gain_vs_spec_pct,
            "base_ew_loss": BASE_EW_LOSS,
            "freeze": FREEZE,
        }
        regression_points.append(point)

        print(f"\n  SUMMARY steps={steps}:")
        print(f"    mean_divergence:   {mean_divergence:.4f}%")
        print(f"    best_spec_ew_loss: {best_spec_ew_loss:.6f}")
        print(f"    moe_ew_loss:       {moe_ew_loss:.6f}")
        print(f"    gain_vs_spec_pct:  {gain_vs_spec_pct:.4f}%")

    # Write output
    output = {
        "regression_points": regression_points,
        "n_points": len(regression_points),
        "model": f"{MODEL_ID}@{REVISION}",
        "eval_protocol": "corrected per-domain bs=4 EW-avg",
        "note": (
            "Derived from training_duration_crossover checkpoints. "
            "Uses freeze=0 specialists."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Output written to: {OUTPUT_JSON}")
    print(f"n_points: {len(regression_points)}")
    print(f"\n{'steps':>8}  {'divergence%':>12}  {'gain_vs_spec%':>14}")
    print("-" * 40)
    for p in regression_points:
        print(f"{p['steps']:>8}  {p['mean_divergence']:>12.4f}  {p['gain_vs_spec_pct']:>14.4f}")


if __name__ == "__main__":
    main()
