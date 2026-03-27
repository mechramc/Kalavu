#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
"""
KALAVAI: Pythia-1B Downstream Benchmarks
========================================
Evaluates 7 model variants from the existing Pythia-1B experiment on 5 downstream
benchmarks using manual log-likelihood scoring (no external eval framework).
Uses 2000 evaluation samples per benchmark.

Models evaluated (seed=42):
  1. Base model (pythia-1b step10000)
  2. Code specialist
  3. Science specialist
  4. Fiction specialist
  5. Weight averaged
  6. MoE fused
  7. Monolithic (trained fresh if not already saved)

Benchmarks:
  hellaswag   - 4-choice sentence completion
  arc_easy    - 4-choice science QA (ARC-Easy)
  lambada     - last word prediction
  sciq        - 4-choice science QA
  winogrande  - 2-choice coreference

Output:
  results/pythia/pythia_1b/benchmarks_seed42.json
  figures/pythia/fig_benchmarks_1b.png
"""

import copy
import json
import math
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

MODEL_ID = "EleutherAI/pythia-1b"
REVISION = "step10000"
FREEZE_LAYERS = 4
LR = 2e-5
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
SEQ_LEN = 512
WARMUP_FRACTION = 0.1
HIDDEN_SIZE = 2048
NUM_LAYERS = 16
DOMAINS = ["code", "science", "fiction"]
SEED = 42

RESULTS_DIR = Path("results/pythia/pythia_1b")
CHECKPOINT_DIR = Path("checkpoints/pythia/pythia_1b")
MONOLITHIC_CHECKPOINT = Path("checkpoints/pythia/pythia_1b/monolithic_seed42.pt")
FIGURES_DIR = Path("figures/pythia")

# Monolithic training config (run only if checkpoint not present)
MONO_MAX_STEPS = 6000
MONO_BATCH_SIZE = 2
MONO_GRAD_ACCUM = 4
N_SAMPLES_MONO = 3000

# Benchmark config
N_BENCHMARK_EXAMPLES = 2000
ROUTER_STEPS = 500
ROUTER_LR = 1e-3
ROUTER_BATCH = 4
EVAL_BATCHES = 50

BENCHMARKS = {
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "method": "log_likelihood_completion",
        "random_chance": 0.25,
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "method": "log_likelihood_completion",
        "random_chance": 0.25,
    },
    "lambada": {
        "dataset": "EleutherAI/lambada_openai",
        "method": "log_likelihood_last_word",
        "random_chance": 0.0,
    },
    "sciq": {
        "dataset": "allenai/sciq",
        "method": "log_likelihood_completion",
        "random_chance": 0.25,
    },
    "winogrande": {
        "dataset": "allenai/winogrande",
        "config": "winogrande_xl",
        "method": "log_likelihood_completion",
        "random_chance": 0.50,
    },
}


# ============================================================================
# PackedChunkDataset (for monolithic training)
# ============================================================================

class PackedChunkDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_len: int = SEQ_LEN,
                 max_chars: int = 5000):
        truncated = [t[:max_chars] for t in texts]
        full = tokenizer(
            "\n\n".join(truncated),
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]
        n_chunks = len(full) // seq_len
        self.chunks = [full[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def _collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


def make_dataset_from_chunks(chunks: list) -> PackedChunkDataset:
    ds = PackedChunkDataset.__new__(PackedChunkDataset)
    ds.chunks = chunks
    return ds


# ============================================================================
# ThreeExpertMoE (same as 1B experiment)
# ============================================================================

class ThreeExpertMoE(nn.Module):
    def __init__(self, spec_a, spec_b, spec_c, hidden_size: int):
        super().__init__()
        self.spec_a = spec_a
        self.spec_b = spec_b
        self.spec_c = spec_c
        for p in self.spec_a.parameters():
            p.requires_grad_(False)
        for p in self.spec_b.parameters():
            p.requires_grad_(False)
        for p in self.spec_c.parameters():
            p.requires_grad_(False)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 3, bias=False),
        )

    def _run_specialist(self, model, input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach()
        last_h = out.hidden_states[-1].detach()
        h_pooled = last_h.mean(dim=1).float()
        return logits, h_pooled

    def forward(self, input_ids, labels=None):
        logits_a, h_a = self._run_specialist(self.spec_a, input_ids)
        logits_b, h_b = self._run_specialist(self.spec_b, input_ids)
        logits_c, h_c = self._run_specialist(self.spec_c, input_ids)

        h_avg = (h_a + h_b + h_c) / 3.0
        gates = torch.softmax(self.router(h_avg), dim=-1)

        fused = (
            gates[:, 0:1, None] * logits_a
            + gates[:, 1:2, None] * logits_b
            + gates[:, 2:3, None] * logits_c
        )

        loss = None
        if labels is not None:
            shift_logits = fused[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return loss, fused, gates


# ============================================================================
# Data loading (for monolithic training only)
# ============================================================================

def load_code_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading code (n={n})...")
    ds = load_dataset("code_search_net", "python", split="train",
                      streaming=True, trust_remote_code=True)
    texts = []
    for item in ds:
        content = item.get("whole_func_string", "") or item.get("func_code_string", "")
        if len(content) >= 200:
            texts.append(content)
        if len(texts) >= n:
            break
    return texts


def load_science_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading science (n={n})...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)
    texts = []
    for item in ds:
        content = (item.get("support", "") + "\n"
                   + item.get("question", "") + "\n"
                   + item.get("correct_answer", ""))
        if len(content) > 100:
            texts.append(content)
        if len(texts) >= n:
            break
    return texts


def load_fiction_texts(n: int) -> list[str]:
    from datasets import load_dataset
    print(f"  Loading fiction (n={n})...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    texts = []
    for item in ds:
        content = item.get("text", "")[:5000]
        if len(content) >= 500:
            texts.append(content)
        if len(texts) >= n:
            break
    return texts


# ============================================================================
# Model loading helpers
# ============================================================================

def load_fresh_base(device: str):
    """Load base model (step10000)."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def load_specialist(domain: str, device: str):
    """Load a specialist: fresh base + load saved state_dict."""
    model = load_fresh_base(device)
    ckpt_path = CHECKPOINT_DIR / f"{domain}_specialist_seed{SEED}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing specialist checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def weight_average_three(spec_a, spec_b, spec_c):
    """Three-way weight average of all layer weights."""
    avg = copy.deepcopy(spec_a)
    sa = spec_a.state_dict()
    sb = spec_b.state_dict()
    sc = spec_c.state_dict()
    avg_state = {
        k: ((sa[k].float() + sb[k].float() + sc[k].float()) / 3.0).to(torch.bfloat16)
        for k in sa
    }
    avg.load_state_dict(avg_state)
    avg.eval()
    return avg


# ============================================================================
# Monolithic training (only if checkpoint not present)
# ============================================================================

def train_monolithic(tokenizer, device: str):
    """Train a monolithic 1B model on mixed data, 6000 steps."""
    print("\nTraining monolithic baseline (6000 steps on mixed data)...")
    set_seed(SEED)

    code_texts    = load_code_texts(N_SAMPLES_MONO)
    science_texts = load_science_texts(N_SAMPLES_MONO)
    fiction_texts = load_fiction_texts(N_SAMPLES_MONO)

    all_texts = code_texts + science_texts + fiction_texts
    ds_full = PackedChunkDataset(all_texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
    n = len(ds_full.chunks)
    train_end = int(n * 0.8)
    train_chunks = ds_full.chunks[:train_end]
    print(f"  Monolithic train chunks: {len(train_chunks)}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)

    # Freeze first 4 layers (same as specialists)
    model.gpt_neox.embed_in.requires_grad_(False)
    for i in range(FREEZE_LAYERS):
        model.gpt_neox.layers[i].requires_grad_(False)

    model.train()
    dataset = make_dataset_from_chunks(train_chunks)
    loader = DataLoader(dataset, batch_size=MONO_BATCH_SIZE, shuffle=True,
                        drop_last=True, collate_fn=_collate)

    warmup_steps = int(MONO_MAX_STEPS * WARMUP_FRACTION)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MONO_MAX_STEPS - warmup_steps)

    step, accum = 0, 0
    optimizer.zero_grad()
    t0 = time.time()

    for batch in cycle(loader):
        if step >= MONO_MAX_STEPS:
            break

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**{k: v.to(device) for k, v in batch.items()})
            loss = out.loss / MONO_GRAD_ACCUM

        loss.backward()
        accum += 1

        if accum == MONO_GRAD_ACCUM:
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

            if step % 500 == 0 or step == MONO_MAX_STEPS:
                elapsed = time.time() - t0
                print(f"  [mono] step {step}/{MONO_MAX_STEPS} | {elapsed:.0f}s")

    model.eval()
    MONOLITHIC_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MONOLITHIC_CHECKPOINT)
    print(f"  Saved monolithic checkpoint: {MONOLITHIC_CHECKPOINT}")
    return model


def load_monolithic(tokenizer, device: str):
    """Load monolithic checkpoint, training it first if absent."""
    if MONOLITHIC_CHECKPOINT.exists():
        print(f"  Loading existing monolithic checkpoint: {MONOLITHIC_CHECKPOINT}")
        model = load_fresh_base(device)
        state = torch.load(MONOLITHIC_CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    else:
        return train_monolithic(tokenizer, device)


# ============================================================================
# Router training
# ============================================================================

def train_moe_router(moe: ThreeExpertMoE, tokenizer, device: str):
    """Train the MoE router on mixed training data from the 1B experiment."""
    print("\n  Loading data for router training...")
    code_texts    = load_code_texts(N_SAMPLES_MONO)
    science_texts = load_science_texts(N_SAMPLES_MONO)
    fiction_texts = load_fiction_texts(N_SAMPLES_MONO)

    all_texts = code_texts + science_texts + fiction_texts
    ds_full = PackedChunkDataset(all_texts, tokenizer, seq_len=SEQ_LEN, max_chars=5000)
    n = len(ds_full.chunks)
    train_chunks = ds_full.chunks[:int(n * 0.8)]

    dataset = make_dataset_from_chunks(train_chunks)
    loader = DataLoader(dataset, batch_size=ROUTER_BATCH, shuffle=True,
                        drop_last=True, collate_fn=_collate)
    it = cycle(loader)
    optimizer = AdamW(moe.router.parameters(), lr=ROUTER_LR)
    moe.train()

    print(f"  Training router ({ROUTER_STEPS} steps)...")
    for step in range(1, ROUTER_STEPS + 1):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss, _, _ = moe(input_ids, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == ROUTER_STEPS:
            print(f"    Router step {step:3d}/{ROUTER_STEPS}: loss={loss.item():.4f}")

    moe.eval()


# ============================================================================
# Benchmark evaluation
# ============================================================================

def load_benchmark_data(benchmark_name: str, benchmark_cfg: dict, tokenizer, n: int = 500):
    """
    Load benchmark examples and return a list of dicts.
    Each dict has 'context', 'choices', 'label' (correct choice index),
    or for lambada: 'text', 'label' (int, 1 = correct).
    """
    from datasets import load_dataset

    dataset_name = benchmark_cfg["dataset"]
    config = benchmark_cfg.get("config", None)
    method = benchmark_cfg["method"]

    print(f"  Loading {benchmark_name} ({dataset_name})...")

    kwargs = {"streaming": True}
    # trust_remote_code needed for some HF dataset scripts (e.g. piqa)
    try:
        if config:
            ds = load_dataset(dataset_name, config, split="validation",
                              trust_remote_code=True, **kwargs)
        else:
            try:
                ds = load_dataset(dataset_name, split="validation",
                                  trust_remote_code=True, **kwargs)
            except Exception:
                ds = load_dataset(dataset_name, split="test",
                                  trust_remote_code=True, **kwargs)
    except TypeError:
        # Fallback: some older HF versions don't support trust_remote_code
        if config:
            ds = load_dataset(dataset_name, config, split="validation", **kwargs)
        else:
            try:
                ds = load_dataset(dataset_name, split="validation", **kwargs)
            except Exception:
                ds = load_dataset(dataset_name, split="test", **kwargs)

    examples = []

    if method == "log_likelihood_completion":
        if benchmark_name == "hellaswag":
            for item in ds:
                ctx = item["ctx"]
                choices = item["endings"]
                label = int(item["label"])
                if len(choices) == 4:
                    examples.append({"context": ctx, "choices": choices, "label": label})
                if len(examples) >= n:
                    break

        elif benchmark_name == "arc_easy":
            for item in ds:
                q = item["question"]
                choices_dict = item["choices"]
                labels_list = choices_dict["label"]
                texts_list = choices_dict["text"]
                correct_label = item["answerKey"]
                if correct_label not in labels_list:
                    continue
                label_idx = labels_list.index(correct_label)
                examples.append({
                    "context": q + " ",
                    "choices": texts_list,
                    "label": label_idx,
                })
                if len(examples) >= n:
                    break

        elif benchmark_name == "sciq":
            for item in ds:
                q = item["question"]
                support = item.get("support", "")
                ctx = (support + "\n" + q + " ").strip() + " "
                correct = item["correct_answer"]
                distractors = [item["distractor1"], item["distractor2"], item["distractor3"]]
                choices = [correct] + distractors
                examples.append({
                    "context": ctx,
                    "choices": choices,
                    "label": 0,  # correct_answer is always first
                })
                if len(examples) >= n:
                    break

        elif benchmark_name == "winogrande":
            for item in ds:
                sentence = item["sentence"]
                opt1 = item["option1"]
                opt2 = item["option2"]
                answer = item["answer"]  # "1" or "2"
                label = int(answer) - 1

                # Split on "_" placeholder
                if "_" not in sentence:
                    continue
                parts = sentence.split("_", 1)
                ctx = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""

                choices = [opt1 + suffix, opt2 + suffix]
                examples.append({
                    "context": ctx,
                    "choices": choices,
                    "label": label,
                })
                if len(examples) >= n:
                    break

    elif method == "log_likelihood_last_word":
        # lambada: predict last word
        for item in ds:
            text = item["text"]
            words = text.split()
            if len(words) < 3:
                continue
            examples.append({"text": text})
            if len(examples) >= n:
                break

    print(f"    Loaded {len(examples)} examples")
    return examples


@torch.no_grad()
def evaluate_multiple_choice(model, tokenizer, examples: list, device: str,
                              is_moe: bool = False) -> float:
    """Evaluate multiple-choice benchmark via log-likelihood scoring."""
    correct = 0
    total = 0

    for item in examples:
        context = item["context"]
        choices = item["choices"]
        label = item["label"]

        best_ll = float("-inf")
        best_idx = 0

        for i, choice in enumerate(choices):
            input_text = context + choice
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            context_ids = tokenizer.encode(
                context, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            context_len = context_ids.shape[1]

            if is_moe:
                output = model(input_ids)
                if isinstance(output, tuple):
                    raw_logits = output[1]
                elif hasattr(output, "logits"):
                    raw_logits = output.logits
                else:
                    raw_logits = output
                logits = raw_logits[0]
            else:
                out = model(input_ids)
                logits = out.logits[0]  # (T, V)

            log_probs = torch.log_softmax(logits[:-1], dim=-1)
            target_ids = input_ids[0, 1:]

            start = max(0, context_len - 1)
            if start >= len(target_ids):
                completion_ll = log_probs[-1, target_ids[-1]].item()
            else:
                completion_ll = log_probs[start:, :].gather(
                    1, target_ids[start:].unsqueeze(1)
                ).sum().item()

            if completion_ll > best_ll:
                best_ll = completion_ll
                best_idx = i

        if best_idx == label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_lambada(model, tokenizer, examples: list, device: str,
                     is_moe: bool = False) -> float:
    """Evaluate LAMBADA: predict last token of text."""
    correct = 0
    total = 0

    for item in examples:
        text = item["text"]
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        if len(tokens) < 2:
            continue

        input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
        target = tokens[-1]

        if is_moe:
            output = model(input_ids)
            if isinstance(output, tuple):
                raw_logits = output[1]
            elif hasattr(output, "logits"):
                raw_logits = output.logits
            else:
                raw_logits = output
            last_logits = raw_logits[0, -1]
        else:
            out = model(input_ids)
            last_logits = out.logits[0, -1]

        predicted = last_logits.argmax().item()
        if predicted == target:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def run_benchmarks_on_model(model, tokenizer, benchmark_data: dict,
                             device: str, model_name: str,
                             is_moe: bool = False) -> dict:
    """Run all 5 benchmarks on a single model. Returns {benchmark: accuracy}."""
    results = {}
    model.eval()

    for bname, (examples, cfg) in benchmark_data.items():
        print(f"  [{model_name}] {bname} ({len(examples)} examples)...")
        t0 = time.time()
        try:
            if cfg["method"] == "log_likelihood_last_word":
                acc = evaluate_lambada(model, tokenizer, examples, device, is_moe=is_moe)
            else:
                acc = evaluate_multiple_choice(model, tokenizer, examples, device, is_moe=is_moe)
            results[bname] = round(acc * 100, 2)
            print(f"    {bname}: {acc*100:.1f}% ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    {bname}: ERROR — {e}")
            results[bname] = None

    avg_vals = [v for v in results.values() if v is not None]
    results["average"] = round(sum(avg_vals) / len(avg_vals), 2) if avg_vals else None
    return results


# ============================================================================
# Figures
# ============================================================================

def save_benchmark_figure(all_results: dict):
    """Grouped bar chart: 7 variants x 5 benchmarks."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        bench_names = ["hellaswag", "arc_easy", "lambada", "sciq", "winogrande"]
        random_chance = [25.0, 25.0, 0.0, 25.0, 50.0]
        model_order = ["base", "code_spec", "science_spec", "fiction_spec",
                       "weight_avg", "moe", "monolithic"]
        display_names = ["Base", "Code\nSpec.", "Science\nSpec.", "Fiction\nSpec.",
                         "Weight\nAvg.", "MoE", "Mono-\nlithic"]
        colors = ["#95a5a6", "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

        n_bench = len(bench_names)
        n_models = len(model_order)
        x = np.arange(n_bench)
        width = 0.10

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, (mk, name, color) in enumerate(zip(model_order, display_names, colors)):
            model_res = all_results.get(mk, {})
            vals = [model_res.get(b, 0.0) or 0.0 for b in bench_names]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)

        # Random chance dashed lines
        for j, (b, rc) in enumerate(zip(bench_names, random_chance)):
            if rc > 0:
                ax.plot([j - 0.5, j + 0.5], [rc, rc], color="black",
                        linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", "\n") for b in bench_names])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Downstream Benchmarks — Pythia-1B@step10000 (seed=42)\n"
                     "Dashed lines = random chance")
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, 100)
        fig.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "fig_benchmarks_1b.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"WARNING: Could not save benchmark figure: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KALAVAI: Pythia-1B Downstream Benchmarks")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: results/pythia/pythia_1b/benchmarks_seed42.json)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("KALAVAI: Pythia-1B Downstream Benchmarks")
    print("=" * 70)

    # Check output file
    out_path = Path(args.output) if args.output else RESULTS_DIR / "benchmarks_seed42.json"
    if out_path.exists():
        print(f"\nResults already exist: {out_path}")
        print("Delete to re-run.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: CPU will be extremely slow.")

    # Load tokenizer once
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load all benchmark data upfront (same 500 examples for all models) ─
    print("\n" + "=" * 70)
    print("Loading benchmark datasets...")
    print("=" * 70)
    benchmark_data = {}
    for bname, bcfg in BENCHMARKS.items():
        try:
            examples = load_benchmark_data(bname, bcfg, tokenizer, n=N_BENCHMARK_EXAMPLES)
            benchmark_data[bname] = (examples, bcfg)
        except Exception as e:
            print(f"  ERROR loading {bname}: {e}")
            benchmark_data[bname] = ([], bcfg)

    # ── Verify specialist checkpoints exist ─────────────────────────────────
    for domain in DOMAINS:
        ckpt = CHECKPOINT_DIR / f"{domain}_specialist_seed{SEED}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Missing specialist checkpoint: {ckpt}\n"
                f"Run kalavai_pythia_1b_experiment.py first."
            )

    # ── Evaluate each model variant ─────────────────────────────────────────
    all_results = {}

    # 1. Base model
    print("\n" + "=" * 70)
    print("1/7  Base model")
    print("=" * 70)
    base = load_fresh_base(device)
    all_results["base"] = run_benchmarks_on_model(
        base, tokenizer, benchmark_data, device, "base"
    )
    del base
    torch.cuda.empty_cache()

    # 2. Code specialist
    print("\n" + "=" * 70)
    print("2/7  Code specialist")
    print("=" * 70)
    code_spec = load_specialist("code", device)
    all_results["code_spec"] = run_benchmarks_on_model(
        code_spec, tokenizer, benchmark_data, device, "code_spec"
    )

    # 3. Science specialist
    print("\n" + "=" * 70)
    print("3/7  Science specialist")
    print("=" * 70)
    sci_spec = load_specialist("science", device)
    all_results["science_spec"] = run_benchmarks_on_model(
        sci_spec, tokenizer, benchmark_data, device, "science_spec"
    )

    # 4. Fiction specialist
    print("\n" + "=" * 70)
    print("4/7  Fiction specialist")
    print("=" * 70)
    fic_spec = load_specialist("fiction", device)
    all_results["fiction_spec"] = run_benchmarks_on_model(
        fic_spec, tokenizer, benchmark_data, device, "fiction_spec"
    )

    # 5. Weight averaged
    print("\n" + "=" * 70)
    print("5/7  Weight averaged")
    print("=" * 70)
    avg_model = weight_average_three(code_spec, sci_spec, fic_spec)
    all_results["weight_avg"] = run_benchmarks_on_model(
        avg_model, tokenizer, benchmark_data, device, "weight_avg"
    )
    del avg_model
    torch.cuda.empty_cache()

    # 6. MoE fused — build MoE and train router
    print("\n" + "=" * 70)
    print("6/7  MoE fused (training router...)")
    print("=" * 70)
    moe = ThreeExpertMoE(code_spec, sci_spec, fic_spec, hidden_size=HIDDEN_SIZE).to(device)
    train_moe_router(moe, tokenizer, device)
    all_results["moe"] = run_benchmarks_on_model(
        moe, tokenizer, benchmark_data, device, "moe", is_moe=True
    )
    del moe, code_spec, sci_spec, fic_spec
    torch.cuda.empty_cache()

    # 7. Monolithic
    print("\n" + "=" * 70)
    print("7/7  Monolithic")
    print("=" * 70)
    mono = load_monolithic(tokenizer, device)
    all_results["monolithic"] = run_benchmarks_on_model(
        mono, tokenizer, benchmark_data, device, "monolithic"
    )
    del mono
    torch.cuda.empty_cache()

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DOWNSTREAM BENCHMARKS — Pythia-1B@step10000 (seed=42)")
    print("=" * 70)
    bench_names = ["hellaswag", "arc_easy", "lambada", "sciq", "winogrande", "average"]
    random_chance = {"hellaswag": 25.0, "arc_easy": 25.0, "lambada": 0.0,
                     "sciq": 25.0, "winogrande": 50.0, "average": None}

    header = f"{'Model':<18}" + "".join(f"{b:>12}" for b in bench_names)
    print(header)
    print("-" * len(header))

    model_display = [
        ("base",         "Base"),
        ("code_spec",    "Code spec."),
        ("science_spec", "Science spec."),
        ("fiction_spec", "Fiction spec."),
        ("weight_avg",   "Weight avg"),
        ("moe",          "MoE fused"),
        ("monolithic",   "Monolithic"),
    ]

    for mk, dname in model_display:
        row = f"{dname:<18}"
        for b in bench_names:
            v = all_results.get(mk, {}).get(b)
            row += f"{(str(v)+'%'):>12}" if v is not None else f"{'N/A':>12}"
        print(row)

    print("-" * len(header))
    row = f"{'Random chance':<18}"
    for b in bench_names:
        rc = random_chance.get(b)
        row += f"{(str(rc)+'%' if rc is not None else '—'):>12}"
    print(row)

    # ── Save results ─────────────────────────────────────────────────────────
    moe_avg = all_results.get("moe", {}).get("average")
    base_avg = all_results.get("base", {}).get("average")

    result = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "seed": SEED,
        "n_examples_per_benchmark": N_BENCHMARK_EXAMPLES,
        "benchmarks": list(BENCHMARKS.keys()),
        "random_chance": {k: v["random_chance"] for k, v in BENCHMARKS.items()},
        "results": all_results,
        "moe_avg_accuracy": moe_avg,
        "base_avg_accuracy": base_avg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    save_benchmark_figure(all_results)

    if moe_avg is not None and base_avg is not None:
        print(f"\nGit commit message:")
        print(f"  [kalavai] 1B benchmarks: MoE avg={moe_avg:.1f}% vs base avg={base_avg:.1f}%")


if __name__ == "__main__":
    main()
