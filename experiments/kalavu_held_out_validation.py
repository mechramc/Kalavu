#!/usr/bin/env python3
"""
KALAVU Held-Out Validation Experiment
======================================
Answers: does the fusion mechanism work on data the model has never seen,
or is the 17.1% / 58% result an artifact of in-distribution evaluation?

Training data:  generated with seed = experiment_seed + 100  (same as original)
In-dist eval:   last 10% of training data                    (same as original)
Held-out eval:  generated with seed = 99999                  (completely independent)

Runs 3 seeds [42, 137, 2026]. Reports both eval sets honestly.
Saves to results/synthetic/held_out_validation.json
"""

import json
import math
import copy
import random
import time
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HELD_OUT_SEED = 99999
SEEDS = [42, 137, 2026]
RESULTS_PATH = Path("results/synthetic/held_out_validation.json")


# ============================================================================
# Config — matches 25M param experiment exactly
# ============================================================================

@dataclass
class ExperimentConfig:
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    context_length: int = 256
    vocab_size: int = 512
    dropout: float = 0.0
    freeze_layers: int = 2
    seed: int = 42
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    eval_interval: int = 200
    warmup_steps: int = 100
    data_tokens: int = 2_000_000
    domain_a: str = "code"
    domain_b: str = "stories"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model — identical to kalavu_experiment.py
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(self.dropout(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def freeze_early_layers(self, n):
        self.tok_emb.weight.requires_grad = False
        self.pos_emb.weight.requires_grad = False
        for i in range(n):
            for p in self.blocks[i].parameters():
                p.requires_grad = False

    def count_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# MoE — identical to kalavu_experiment.py
# ============================================================================

class SimpleMoEFusion(nn.Module):
    def __init__(self, config, module_a, module_b):
        super().__init__()
        self.config = config
        self.module_a = module_a
        self.module_b = module_b
        self.router = nn.Linear(config.d_model, 2, bias=False)
        self.tok_emb = module_a.tok_emb
        self.pos_emb = module_a.pos_emb
        self.shared_blocks = nn.ModuleList([module_a.blocks[i] for i in range(config.freeze_layers)])
        self.specialist_a = nn.ModuleList([module_a.blocks[i] for i in range(config.freeze_layers, config.n_layers)])
        self.specialist_b = nn.ModuleList([module_b.blocks[i] for i in range(config.freeze_layers, config.n_layers)])
        self.ln_f = module_a.ln_f
        self.lm_head = module_a.lm_head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.shared_blocks:
            x = block(x)
        weights = F.softmax(self.router(x.detach()), dim=-1)  # (B, T, 2)
        x_a, x_b = x.clone(), x.clone()
        for block in self.specialist_a:
            x_a = block(x_a)
        for block in self.specialist_b:
            x_b = block(x_b)
        x = weights[:, :, 0:1] * x_a + weights[:, :, 1:2] * x_b
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Data — same generators, parameterized by rng
# ============================================================================

def make_generators(rng):
    """Return (gen_code_line, gen_story_line) closures over the given rng."""
    keywords_a = ["def", "return", "if", "else", "for", "in", "range", "import",
                  "class", "self", "print", "while", "try", "except", "with",
                  "as", "from", "None", "True", "False", "lambda", "yield",
                  "break", "continue", "pass", "raise", "global", "not", "and", "or"]
    ops_a = ["=", "==", "!=", "+=", "-=", ":", "(", ")", "[", "]", "{", "}", ",", ".", "+", "-", "*", "/"]

    def gen_code_line():
        k = rng.choice(keywords_a)
        var = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz_", k=rng.randint(2, 8)))
        op = rng.choice(ops_a)
        num = str(rng.randint(0, 1000))
        templates = [
            f"{k} {var}{op} {num}\n",
            f"    {var} {op} {var}({num})\n",
            f"{k} {var} in range({num}):\n",
            f"    return {var}\n",
            f"# {var} {k} {num}\n",
        ]
        return rng.choice(templates)

    subjects = ["the cat", "a dog", "she", "he", "the old man", "the child",
                "the queen", "a bird", "the river", "the moon", "they", "we",
                "the knight", "a wizard", "the forest", "the ship"]
    verbs = ["walked", "ran", "said", "looked", "found", "lost", "saw",
             "heard", "felt", "knew", "wanted", "needed", "loved", "feared",
             "remembered", "forgot", "believed", "hoped", "dreamed", "whispered"]
    places = ["in the garden", "by the river", "through the forest", "at the castle",
              "under the stars", "near the mountain", "along the path", "in the village",
              "across the sea", "into the cave", "beyond the hills", "beside the fire"]

    def gen_story_line():
        s = rng.choice(subjects)
        v = rng.choice(verbs)
        p = rng.choice(places)
        adjs = ["quietly", "slowly", "suddenly", "carefully", "bravely", "gently", "fearfully"]
        templates = [
            f"{s} {v} {rng.choice(adjs)} {p}. ",
            f"once upon a time, {s} {v} {p}. ",
            f"\"{s} {v},\" {rng.choice(subjects)} {rng.choice(verbs)}. ",
            f"{p}, {s} {v}. ",
            f"and then {s} {v} {p}, never to return. ",
        ]
        return rng.choice(templates)

    return gen_code_line, gen_story_line


def generate_domain_texts(data_tokens, seed):
    """Generate training/eval texts for both domains using given seed."""
    rng = random.Random(seed)
    gen_code, gen_story = make_generators(rng)
    text_a = "".join(gen_code() for _ in range(data_tokens // 20))
    text_b = "".join(gen_story() for _ in range(data_tokens // 30))
    return text_a, text_b


def tokenize(text, vocab_size):
    return [min(b, vocab_size - 1) for b in text.encode("utf-8")]


class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.context_length + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def make_mixed_tokens(tokens_a, tokens_b, context_length):
    mixed = []
    for i in range(0, min(len(tokens_a), len(tokens_b)), context_length):
        mixed.extend(tokens_a[i:i + context_length // 2])
        mixed.extend(tokens_b[i:i + context_length // 2])
    return mixed


# ============================================================================
# Training — identical to kalavu_experiment.py
# ============================================================================

def train_module(model, train_dataset, eval_datasets, config, name, device):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    trainable = model.count_params(trainable_only=True)
    total = model.count_params()
    print(f"\n{'='*60}\nTraining {name}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)

        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.max_steps:
            model.eval()
            with torch.no_grad():
                evals = {}
                for ename, eds in eval_datasets.items():
                    el = DataLoader(eds, batch_size=config.batch_size)
                    losses = []
                    for ex, ey in el:
                        _, l = model(ex.to(device), ey.to(device))
                        losses.append(l.item())
                        if len(losses) >= 10:
                            break
                    evals[ename] = sum(losses) / len(losses)
            elapsed = time.time() - t0
            estr = " | ".join(f"{k}: {v:.4f}" for k, v in evals.items())
            print(f"  [{name}] step {step:5d} | train: {loss.item():.4f} | {estr} | {elapsed:.1f}s")
            model.train()


@torch.no_grad()
def evaluate_model(model, eval_datasets, config, device):
    model = model.to(device)
    model.eval()
    results = {}
    for name, ds in eval_datasets.items():
        loader = DataLoader(ds, batch_size=config.batch_size)
        losses = []
        for x, y in loader:
            _, l = model(x.to(device), y.to(device))
            losses.append(l.item())
            if len(losses) >= 20:
                break
        results[name] = sum(losses) / len(losses)
    return results


def fuse_by_averaging(module_a, module_b, config):
    fused = copy.deepcopy(module_a)
    for i in range(config.freeze_layers, config.n_layers):
        for pa, pb in zip(fused.blocks[i].parameters(), module_b.blocks[i].parameters()):
            pa.data = (pa.data + pb.data) / 2.0
    return fused


def fuse_by_moe(module_a, module_b, config, mixed_train, device, router_steps=500):
    moe = SimpleMoEFusion(config, module_a, module_b).to(device)
    for name, p in moe.named_parameters():
        if "router" not in name:
            p.requires_grad = False
    optimizer = torch.optim.Adam(moe.router.parameters(), lr=1e-3)
    loader = DataLoader(mixed_train, batch_size=config.batch_size, shuffle=True, drop_last=True)
    it = iter(loader)
    moe.train()
    print(f"\n  Training MoE router for {router_steps} steps...")
    for step in range(1, router_steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        _, loss = moe(x.to(device), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"    Router step {step}: loss={loss.item():.4f}")
    return moe


# ============================================================================
# Results table printer
# ============================================================================

def print_table(title, models_dict):
    print(f"\nRESULTS: {title}")
    print(f"{'Model':<25} {'Code':>10} {'Stories':>10} {'Mixed':>10} {'Average':>10}")
    print("-" * 65)
    for name, res in models_dict.items():
        avg = sum(res.values()) / len(res)
        print(f"{name:<25} {res['code']:>10.4f} {res['stories']:>10.4f} {res['mixed']:>10.4f} {avg:>10.4f}")


def improvement_pct(results_a, results_b, results_fused):
    best_ind = min(
        sum(results_a.values()) / len(results_a),
        sum(results_b.values()) / len(results_b),
    )
    fused_avg = sum(results_fused.values()) / len(results_fused)
    return (best_ind - fused_avg) / best_ind * 100


# ============================================================================
# Single seed run
# ============================================================================

def run_seed(seed):
    config = ExperimentConfig(seed=seed)
    device = config.device
    print(f"\n{'#'*70}")
    print(f"# SEED {seed}")
    print(f"{'#'*70}")

    # --- Generate training data (matches original exactly) ---
    print(f"\nGenerating training data (seed={seed + 100})...")
    text_a, text_b = generate_domain_texts(config.data_tokens, seed=seed + 100)
    tokens_a = tokenize(text_a, config.vocab_size)
    tokens_b = tokenize(text_b, config.vocab_size)
    print(f"  Code:    {len(tokens_a):,} tokens")
    print(f"  Stories: {len(tokens_b):,} tokens")

    split_a = int(len(tokens_a) * 0.9)
    split_b = int(len(tokens_b) * 0.9)

    train_a = TextDataset(tokens_a[:split_a], config.context_length)
    train_b = TextDataset(tokens_b[:split_b], config.context_length)

    # In-distribution eval (10% of training data — same as original)
    indist_mixed_tokens = make_mixed_tokens(tokens_a[split_a:], tokens_b[split_b:], config.context_length)
    eval_indist = {
        "code":    TextDataset(tokens_a[split_a:], config.context_length),
        "stories": TextDataset(tokens_b[split_b:], config.context_length),
        "mixed":   TextDataset(indist_mixed_tokens, config.context_length),
    }

    # Held-out eval — generated AFTER training data, independent seed
    print(f"\nGenerating held-out eval data (seed={HELD_OUT_SEED})...")
    heldout_a_text, heldout_b_text = generate_domain_texts(
        config.data_tokens // 5,  # 20% of training size is plenty for eval
        seed=HELD_OUT_SEED,
    )
    heldout_tokens_a = tokenize(heldout_a_text, config.vocab_size)
    heldout_tokens_b = tokenize(heldout_b_text, config.vocab_size)
    heldout_mixed_tokens = make_mixed_tokens(heldout_tokens_a, heldout_tokens_b, config.context_length)
    eval_heldout = {
        "code":    TextDataset(heldout_tokens_a, config.context_length),
        "stories": TextDataset(heldout_tokens_b, config.context_length),
        "mixed":   TextDataset(heldout_mixed_tokens, config.context_length),
    }
    print(f"  Held-out code:    {len(heldout_tokens_a):,} tokens")
    print(f"  Held-out stories: {len(heldout_tokens_b):,} tokens")

    # Mixed training data for router
    mixed_train_tokens = make_mixed_tokens(tokens_a[:split_a], tokens_b[:split_b], config.context_length)
    mixed_train = TextDataset(mixed_train_tokens, config.context_length)

    # --- Canonical seed θ₀ ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_model = MiniGPT(config)
    seed_ckpt = copy.deepcopy(seed_model.state_dict())

    # Eval seed model on both eval sets
    seed_indist = evaluate_model(seed_model, eval_indist, config, device)
    seed_heldout = evaluate_model(seed_model, eval_heldout, config, device)

    # --- Train Module A ---
    module_a = MiniGPT(config)
    module_a.load_state_dict(seed_ckpt)
    module_a.freeze_early_layers(config.freeze_layers)
    train_module(module_a, train_a, eval_indist, config, f"Module A (code, seed={seed})", device)

    # --- Train Module B ---
    module_b = MiniGPT(config)
    module_b.load_state_dict(seed_ckpt)
    module_b.freeze_early_layers(config.freeze_layers)
    train_module(module_b, train_b, eval_indist, config, f"Module B (stories, seed={seed})", device)

    # --- Fuse ---
    print(f"\n{'='*60}\nFUSION (seed={seed})\n{'='*60}")

    fused_avg = fuse_by_averaging(module_a, module_b, config)
    fused_moe = fuse_by_moe(module_a, module_b, config, mixed_train, device)

    # --- Evaluate ALL models on BOTH eval sets ---
    results_indist = {
        "seed":    seed_indist,
        "module_a": evaluate_model(module_a, eval_indist, config, device),
        "module_b": evaluate_model(module_b, eval_indist, config, device),
        "fused_avg": evaluate_model(fused_avg, eval_indist, config, device),
        "fused_moe": evaluate_model(fused_moe, eval_indist, config, device),
    }
    results_heldout = {
        "seed":    seed_heldout,
        "module_a": evaluate_model(module_a, eval_heldout, config, device),
        "module_b": evaluate_model(module_b, eval_heldout, config, device),
        "fused_avg": evaluate_model(fused_avg, eval_heldout, config, device),
        "fused_moe": evaluate_model(fused_moe, eval_heldout, config, device),
    }

    # --- Print tables ---
    display_indist = {
        "Seed (untrained)":   results_indist["seed"],
        f"Module A (code)":   results_indist["module_a"],
        f"Module B (stories)": results_indist["module_b"],
        "Fused (averaging)":  results_indist["fused_avg"],
        "Fused (MoE routing)": results_indist["fused_moe"],
    }
    display_heldout = {
        "Seed (untrained)":   results_heldout["seed"],
        f"Module A (code)":   results_heldout["module_a"],
        f"Module B (stories)": results_heldout["module_b"],
        "Fused (averaging)":  results_heldout["fused_avg"],
        "Fused (MoE routing)": results_heldout["fused_moe"],
    }

    print_table(f"In-Distribution Eval (10% train split) — seed={seed}", display_indist)
    imp_i = improvement_pct(results_indist["module_a"], results_indist["module_b"], results_indist["fused_moe"])
    print(f"\nImprovement (in-dist):    {imp_i:+.1f}%")

    print_table(f"Held-Out Eval (fresh data, seed={HELD_OUT_SEED}) — seed={seed}", display_heldout)
    imp_h = improvement_pct(results_heldout["module_a"], results_heldout["module_b"], results_heldout["fused_moe"])
    print(f"\nImprovement (held-out):   {imp_h:+.1f}%")

    return {
        "seed": seed,
        "improvement_indist": round(imp_i, 4),
        "improvement_heldout": round(imp_h, 4),
        "indist": {k: {d: round(v, 6) for d, v in res.items()} for k, res in results_indist.items()},
        "heldout": {k: {d: round(v, 6) for d, v in res.items()} for k, res in results_heldout.items()},
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("KALAVU Held-Out Validation Experiment")
    print("=" * 70)
    print(f"Training seed: experiment_seed + 100 (same as original)")
    print(f"Held-out seed: {HELD_OUT_SEED} (completely independent)")
    print(f"Seeds: {SEEDS}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        result = run_seed(seed)
        all_results.append(result)

    # Aggregate across seeds
    import statistics
    indist_improvements = [r["improvement_indist"] for r in all_results]
    heldout_improvements = [r["improvement_heldout"] for r in all_results]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — ACROSS ALL SEEDS")
    print("=" * 70)
    print(f"\n{'Seed':<8} {'In-Dist':>12} {'Held-Out':>12}")
    print("-" * 34)
    for r in all_results:
        print(f"{r['seed']:<8} {r['improvement_indist']:>+11.1f}% {r['improvement_heldout']:>+11.1f}%")
    print("-" * 34)
    print(f"{'Mean':<8} {statistics.mean(indist_improvements):>+11.1f}% {statistics.mean(heldout_improvements):>+11.1f}%")
    if len(SEEDS) > 1:
        print(f"{'Std':<8} {statistics.stdev(indist_improvements):>11.1f}% {statistics.stdev(heldout_improvements):>11.1f}%")

    # Interpret
    mean_heldout = statistics.mean(heldout_improvements)
    print(f"\nINTERPRETATION:")
    if mean_heldout > 30:
        print(f"  A: Held-out improvement {mean_heldout:.1f}% — fusion genuinely generalizes.")
        print(f"     The Qwen 0% result is explained by insufficient fine-tuning, not a broken mechanism.")
    elif mean_heldout > 5:
        print(f"  B: Held-out improvement {mean_heldout:.1f}% — fusion works but in-dist eval inflated the benefit.")
        print(f"     Paper should report held-out numbers as primary metric.")
    else:
        print(f"  C: Held-out improvement {mean_heldout:.1f}% — fusion relies on memorization, not generalization.")
        print(f"     Thesis needs rethinking.")

    output = {
        "experiment": "held_out_validation",
        "held_out_seed": HELD_OUT_SEED,
        "seeds": SEEDS,
        "summary": {
            "indist_mean": round(statistics.mean(indist_improvements), 4),
            "indist_std": round(statistics.stdev(indist_improvements), 4) if len(SEEDS) > 1 else 0,
            "heldout_mean": round(statistics.mean(heldout_improvements), 4),
            "heldout_std": round(statistics.stdev(heldout_improvements), 4) if len(SEEDS) > 1 else 0,
        },
        "per_seed": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
