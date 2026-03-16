#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVU: Downstream Benchmarks — Pythia-410M Specialist Fusion
=============================================================
Evaluates all model variants on standard NLP benchmarks using lm-eval harness.

Models evaluated (seed=42 only):
  1. Base model (pythia-410m step10000)
  2. Code specialist
  3. Science specialist
  4. Fiction specialist
  5. Weight averaged (3-way)
  6. MoE fused          <- manual log-likelihood (Option B — simpler than lm-eval wrapper)
  7. Monolithic         <- from monolithic baseline experiment

Benchmarks:
  hellaswag, arc_easy, piqa, winogrande, lambada_openai, sciq

MoE evaluation uses manual log-likelihood scoring (no lm-eval wrapper needed):
  For each question: compute log P(answer_i | question) for each choice,
  pick argmax, check if correct.

If all models score near random on all benchmarks:
  Report honestly — model is too early in training for task benchmarks.
  Eval loss is the primary metric.
"""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Config
# ============================================================================

MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "step10000"
SEED = 42
DOMAINS = ["code", "science", "fiction"]

CHECKPOINT_DIR = Path("checkpoints/pythia")
RESULTS_DIR = Path("results/pythia")
FIGURES_DIR = Path("figures/pythia")
MONOLITHIC_CKPT = RESULTS_DIR / "monolithic_baseline_seed42.json"

BENCHMARKS = [
    "hellaswag",
    "arc_easy",
    "piqa",
    "winogrande",
    "lambada_openai",
    "sciq",
]

HIDDEN_SIZE = 1024

# Random-chance baselines for flagging near-random scores
RANDOM_CHANCE = {
    "hellaswag":     0.25,   # 4-choice
    "arc_easy":      0.25,   # 4-choice
    "piqa":          0.50,   # 2-choice
    "winogrande":    0.50,   # 2-choice
    "lambada_openai": 0.00,  # generative — no simple chance baseline
    "sciq":          0.25,   # 4-choice
}


# ============================================================================
# MoE model (identical to main experiment)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in list(self.spec_a.parameters()) + list(self.spec_b.parameters()) + list(self.spec_c.parameters()):
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 3, bias=False),
        )

    def _run(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        return out.logits.detach(), out.hidden_states[-1].detach().mean(1).float()

    def forward(self, input_ids):
        la, ha = self._run(self.spec_a, input_ids)
        lb, hb = self._run(self.spec_b, input_ids)
        lc, hc = self._run(self.spec_c, input_ids)
        gates = torch.softmax(self.router((ha + hb + hc) / 3), dim=-1)
        fused = (gates[:, 0:1, None] * la
                 + gates[:, 1:2, None] * lb
                 + gates[:, 2:3, None] * lc)
        return fused


def load_moe(specialists, device):
    """Load MoE model and restore router from main experiment checkpoint."""
    moe = ThreeExpertMoE(
        specialists["code"], specialists["science"], specialists["fiction"],
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    # Try to load saved router weights from main experiment
    router_path = CHECKPOINT_DIR / "moe_router_seed42.pt"
    if router_path.exists():
        moe.router.load_state_dict(torch.load(router_path, map_location=device))
        print(f"  Loaded router from {router_path}")
    else:
        print(f"  NOTE: No saved router at {router_path} — training fresh router (500 steps)...")
        _train_fresh_router(moe, specialists, device)

    moe.eval()
    return moe


def _train_fresh_router(moe, specialists, device):
    """Train router on mixed data if no saved router exists."""
    from itertools import cycle
    from torch.utils.data import DataLoader, Dataset

    class SimpleTokenDS(Dataset):
        def __init__(self, texts, tok, seq_len=512, max_chars=2000):
            truncated = [t[:max_chars] for t in texts]
            full = tok("\n\n".join(truncated), return_tensors="pt",
                       truncation=False)["input_ids"][0]
            n = len(full) // seq_len
            self.chunks = [full[i*seq_len:(i+1)*seq_len] for i in range(n)]
        def __len__(self): return len(self.chunks)
        def __getitem__(self, i): return self.chunks[i]

    from datasets import load_dataset
    code_ds = load_dataset("code_search_net", "python", split="train",
                           streaming=True, trust_remote_code=True)
    code_texts = [item.get("whole_func_string", "")[:2000]
                  for item in code_ds if len(item.get("whole_func_string", "")) > 200][:500]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = SimpleTokenDS(code_texts, tokenizer)
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(moe.router.parameters(), lr=1e-3)
    moe.train()
    it = cycle(loader)
    for step in range(500):
        ids = next(it).to(device)
        logits = moe(ids)
        shift = logits[:, :-1].contiguous()
        labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift.view(-1, shift.size(-1)), labels.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"    Router step {step+1}/500: loss={loss.item():.4f}")
    moe.eval()


# ============================================================================
# Weight averaging (3-way, matches main experiment)
# ============================================================================

def weight_average_three(spec_a, spec_b, spec_c):
    avg = copy.deepcopy(spec_a)
    sa, sb, sc = spec_a.state_dict(), spec_b.state_dict(), spec_c.state_dict()
    avg_state = {
        k: ((sa[k].float() + sb[k].float() + sc[k].float()) / 3.0).to(torch.bfloat16)
        for k in sa
    }
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Manual log-likelihood evaluation (Option B)
# ============================================================================

@torch.no_grad()
def loglikelihood_hf(model, tokenizer, context: str, continuation: str,
                     device: str) -> float:
    """Compute log P(continuation | context) for a standard HF model."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return float("-inf")

    full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits[0]  # (T, V)

    # Score only the continuation tokens
    start = len(ctx_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    score = 0.0
    for i, tok_id in enumerate(cont_ids):
        score += log_probs[start + i - 1, tok_id].item()
    return score


@torch.no_grad()
def loglikelihood_moe(moe, tokenizer, context: str, continuation: str,
                      device: str) -> float:
    """Compute log P(continuation | context) for the MoE model."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not cont_ids:
        return float("-inf")

    full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = moe(full_ids)[0]  # (T, V)

    start = len(ctx_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    score = 0.0
    for i, tok_id in enumerate(cont_ids):
        score += log_probs[start + i - 1, tok_id].item()
    return score


def evaluate_benchmark(model, tokenizer, benchmark_name: str, device: str,
                        is_moe: bool = False, n_examples: int = 1000) -> dict:
    """
    Evaluate a model on a benchmark using log-likelihood scoring.
    Returns {"accuracy": float, "n_examples": int, "near_random": bool}
    """
    from datasets import load_dataset

    ll_fn = loglikelihood_moe if is_moe else loglikelihood_hf

    try:
        if benchmark_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n_examples: break
                ctx = item["activity_label"] + ": " + item["ctx"]
                endings = item["endings"]
                label = int(item["label"])
                scores = [ll_fn(model, tokenizer, ctx, e, device) for e in endings]
                if scores.index(max(scores)) == label:
                    correct += 1
                total += 1
            return {"accuracy": correct / total if total else 0.0, "n": total}

        elif benchmark_name == "arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation",
                              streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n_examples: break
                ctx = item["question"]
                choices = item["choices"]["text"]
                labels = item["choices"]["label"]
                answer_key = item["answerKey"]
                if answer_key not in labels: total += 1; continue
                answer_idx = labels.index(answer_key)
                scores = [ll_fn(model, tokenizer, ctx, c, device) for c in choices]
                if scores.index(max(scores)) == answer_idx:
                    correct += 1
                total += 1
            return {"accuracy": correct / total if total else 0.0, "n": total}

        elif benchmark_name == "piqa":
            ds = load_dataset("piqa", split="validation", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n_examples: break
                ctx = item["goal"]
                choices = [item["sol1"], item["sol2"]]
                label = int(item["label"])
                scores = [ll_fn(model, tokenizer, ctx, c, device) for c in choices]
                if scores.index(max(scores)) == label:
                    correct += 1
                total += 1
            return {"accuracy": correct / total if total else 0.0, "n": total}

        elif benchmark_name == "winogrande":
            ds = load_dataset("winogrande", "winogrande_xl", split="validation",
                              streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n_examples: break
                sentence = item["sentence"]
                opt1, opt2 = item["option1"], item["option2"]
                label = int(item["answer"]) - 1  # 1-indexed -> 0-indexed
                choices = [opt1, opt2]
                # Replace "_" placeholder with each option
                ctxs = [sentence.replace("_", c) for c in choices]
                # Score full sentence for each option
                scores = [ll_fn(model, tokenizer, "", ctx, device) for ctx in ctxs]
                if scores.index(max(scores)) == label:
                    correct += 1
                total += 1
            return {"accuracy": correct / total if total else 0.0, "n": total}

        elif benchmark_name == "lambada_openai":
            ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n_examples: break
                text = item["text"]
                # Split: context = all but last word, target = last word
                parts = text.rsplit(" ", 1)
                if len(parts) < 2: total += 1; continue
                ctx, target = parts[0], " " + parts[1]
                # Greedy decode: check if top-1 token matches first token of target
                ctx_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                if not target_ids: total += 1; continue
                with torch.no_grad():
                    if is_moe:
                        logits = moe(ctx_ids)[0, -1]
                    else:
                        logits = model(input_ids=ctx_ids).logits[0, -1]
                pred = logits.argmax().item()
                if pred == target_ids[0]:
                    correct += 1
                total += 1
            return {"accuracy": correct / total if total else 0.0, "n": total}

        elif benchmark_name == "sciq":
            ds = load_dataset("allenai/sciq", split="test", streaming=True)
            correct, total = 0, 0
            for item in ds:
                if total >= n_examples: break
                ctx = item.get("support", "") + " " + item["question"]
                correct_ans = item["correct_answer"]
                distractors = [item["distractor1"], item["distractor2"],
                               item["distractor3"]]
                choices = [correct_ans] + distractors
                scores = [ll_fn(model, tokenizer, ctx, c, device) for c in choices]
                if scores.index(max(scores)) == 0:  # correct is always index 0
                    correct += 1
                total += 1
            return {"accuracy": correct / total if total else 0.0, "n": total}

        else:
            return {"accuracy": None, "n": 0, "error": f"Unknown benchmark: {benchmark_name}"}

    except Exception as e:
        print(f"    WARNING: {benchmark_name} failed: {e}")
        return {"accuracy": None, "n": 0, "error": str(e)}


# ============================================================================
# Near-random check
# ============================================================================

def is_near_random(results: dict) -> bool:
    """Returns True if ALL benchmarks with known random chance are within 3pp of chance."""
    near = []
    for bm, res in results.items():
        acc = res.get("accuracy")
        chance = RANDOM_CHANCE.get(bm)
        if acc is not None and chance is not None:
            near.append(abs(acc - chance) < 0.03)
    return all(near) if near else False


# ============================================================================
# Figure
# ============================================================================

def save_benchmark_figure(all_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        model_order = ["base", "code_specialist", "science_specialist",
                       "fiction_specialist", "weight_averaged", "moe_fused", "monolithic"]
        display_names = ["Base", "Code\nspec.", "Science\nspec.", "Fiction\nspec.",
                         "Weight\navg.", "MoE\nfused", "Monolithic"]
        colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#e67e22"]

        # Only plot benchmarks where at least one model exceeds random+3pp
        valid_benchmarks = []
        for bm in BENCHMARKS:
            chance = RANDOM_CHANCE.get(bm, 0.0)
            any_signal = any(
                (all_results.get(mk, {}).get(bm, {}).get("accuracy") or 0) > chance + 0.03
                for mk in model_order
            )
            if any_signal:
                valid_benchmarks.append(bm)

        if not valid_benchmarks:
            print("  No benchmark shows signal above random — skipping figure.")
            return

        n_benchmarks = len(valid_benchmarks)
        n_models = len(model_order)
        x = np.arange(n_benchmarks)
        width = 0.1

        fig, ax = plt.subplots(figsize=(14, 6))
        for i, (mk, name, color) in enumerate(zip(model_order, display_names, colors)):
            vals = []
            for bm in valid_benchmarks:
                acc = all_results.get(mk, {}).get(bm, {}).get("accuracy")
                vals.append((acc or 0) * 100)
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)

        # Random chance lines
        for j, bm in enumerate(valid_benchmarks):
            chance = RANDOM_CHANCE.get(bm, 0.0)
            ax.plot([j - 0.5, j + 0.5], [chance * 100, chance * 100],
                    "k--", alpha=0.3, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in valid_benchmarks])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Downstream Benchmarks — Pythia-410M seed={SEED}\n"
                     f"(dashed lines = random chance)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_benchmarks.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING benchmark figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU: Downstream Benchmarks — Pythia-410M")
    print("=" * 70)
    print(f"Benchmarks: {BENCHMARKS}")
    print(f"Seed: {SEED} | MoE: manual log-likelihood (Option B)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================================================================
    # Load all model variants
    # =========================================================================
    print("\nLoading model variants...")

    def load_base():
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16,
        ).to(device)
        m.eval()
        return m

    def load_specialist(domain):
        m = load_base()
        ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed{SEED}.pt"
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        return m

    model_variants = {}

    print("  Loading base model...")
    model_variants["base"] = load_base()

    print("  Loading specialists...")
    for domain in DOMAINS:
        print(f"    {domain}...")
        model_variants[f"{domain}_specialist"] = load_specialist(domain)

    print("  Computing weight average...")
    model_variants["weight_averaged"] = weight_average_three(
        model_variants["code_specialist"],
        model_variants["science_specialist"],
        model_variants["fiction_specialist"],
    ).to(device)

    print("  Loading MoE...")
    moe_model = load_moe(
        {d: model_variants[f"{d}_specialist"] for d in DOMAINS}, device
    )
    model_variants["moe_fused"] = moe_model

    # Monolithic model: load from saved checkpoint if available
    mono_ckpt = CHECKPOINT_DIR / f"monolithic_seed{SEED}.pt"
    if mono_ckpt.exists():
        print(f"  Loading monolithic from {mono_ckpt}...")
        model_variants["monolithic"] = load_base()
        model_variants["monolithic"].load_state_dict(
            torch.load(mono_ckpt, map_location=device)
        )
        model_variants["monolithic"].eval()
    else:
        print(f"  NOTE: No monolithic checkpoint at {mono_ckpt} — skipping monolithic eval.")
        print(f"  Run kalavu_pythia_monolithic_baseline.py first and save checkpoint.")

    # =========================================================================
    # Run benchmarks
    # =========================================================================
    all_results = {}

    for variant_name, model in model_variants.items():
        is_moe = (variant_name == "moe_fused")
        print(f"\n{'='*55}")
        print(f"Evaluating: {variant_name}")
        print(f"{'='*55}")

        variant_results = {}
        for bm in BENCHMARKS:
            print(f"  {bm}...")
            # Pass moe model as first arg if needed
            eval_model = model if not is_moe else model
            res = evaluate_benchmark(
                eval_model, tokenizer, bm, device,
                is_moe=is_moe, n_examples=500,
            )
            acc_str = f"{res['accuracy']*100:.1f}%" if res.get("accuracy") is not None else "ERROR"
            chance = RANDOM_CHANCE.get(bm, 0.0)
            flag = ""
            if res.get("accuracy") is not None:
                if res["accuracy"] < chance + 0.03:
                    flag = " [near-random]"
            print(f"    {bm}: {acc_str} (n={res.get('n', 0)}){flag}")
            variant_results[bm] = res

        all_results[variant_name] = variant_results

        # Check if near-random overall
        if is_near_random(variant_results):
            print(f"  WARNING: {variant_name} is near-random on all benchmarks.")

    # =========================================================================
    # Results table
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"DOWNSTREAM BENCHMARKS (Pythia-410M, seed={SEED})")
    print("=" * 70)

    bm_labels = [b[:10] for b in BENCHMARKS]
    header = f"{'Model':<24}" + "".join(f"{b:>11}" for b in bm_labels) + f"{'Avg':>8}"
    print(header)
    print("-" * len(header))

    for variant_name in ["base", "code_specialist", "science_specialist",
                         "fiction_specialist", "weight_averaged", "moe_fused", "monolithic"]:
        if variant_name not in all_results:
            continue
        vr = all_results[variant_name]
        accs = [vr.get(bm, {}).get("accuracy") for bm in BENCHMARKS]
        valid_accs = [a for a in accs if a is not None]
        avg = sum(valid_accs) / len(valid_accs) if valid_accs else None

        row = f"{variant_name:<24}"
        for acc in accs:
            row += f"{acc*100:>10.1f}%" if acc is not None else f"{'N/A':>11}"
        row += f"{avg*100:>7.1f}%" if avg is not None else f"{'N/A':>8}"
        print(row)

    # Random chance row
    print("-" * len(header))
    chance_row = f"{'Random chance':<24}"
    for bm in BENCHMARKS:
        c = RANDOM_CHANCE.get(bm)
        chance_row += f"{c*100:>10.1f}%" if c is not None else f"{'---':>11}"
    print(chance_row)

    # Check overall signal
    base_results = all_results.get("base", {})
    near_random_overall = is_near_random(base_results)
    if near_random_overall:
        print("\nCONCLUSION: Base model is near-random on all benchmarks.")
        print("  Pythia-410M at step10000 is too early for meaningful task eval.")
        print("  Eval loss is the primary metric for this paper.")
        print("  Downstream benchmarks reported in appendix only.")
    else:
        print("\nCONCLUSION: Some benchmarks show signal above random chance.")
        print("  See fig_benchmarks.png for visualization.")

    # Figure
    print("\nSaving benchmark figure (if signal exists)...")
    save_benchmark_figure(all_results)

    # Save JSON
    output = {
        "experiment": "downstream_benchmarks",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "seed": SEED,
        "benchmarks": BENCHMARKS,
        "n_examples_per_benchmark": 500,
        "moe_evaluation_method": "manual_loglikelihood_option_b",
        "note": "Pythia-410M at step10000 — scores may be near-random on harder tasks",
        "model_variants": {
            name: {
                bm: {
                    "accuracy": res.get("accuracy"),
                    "n": res.get("n", 0),
                    "error": res.get("error"),
                }
                for bm, res in variant_results.items()
            }
            for name, variant_results in all_results.items()
        },
        "random_chance_baselines": RANDOM_CHANCE,
        "near_random_overall": near_random_overall,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = RESULTS_DIR / "benchmarks_seed42.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
