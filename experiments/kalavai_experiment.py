#!/usr/bin/env python3
"""
KALAVAI 2-Module Proof-of-Concept Experiment
============================================
The experiment that validates or kills the thesis.

Two tiny transformer LMs, shared seed, shared frozen early layers,
trained on different data domains, then fused. Does the fused model
outperform either individual module?

Requirements:
    pip install torch numpy datasets tiktoken

Hardware: Any GPU with 8GB+ VRAM (tested on RTX 5090, works on 4090/3090/M4)
Time: ~10-15 minutes total

Usage:
    python kalavai_experiment.py

What this tests:
    1. Two modules initialized from identical seed (θ₀)
    2. First K layers frozen (shared backbone)
    3. Remaining layers trained on different data domains
    4. Fusion by: (a) weight averaging unfrozen layers, (b) simple MoE routing
    5. Evaluate all models on both domains + mixed eval set

If fused model outperforms both individual modules → thesis has legs.
If fused model is worse than average of individuals → thesis is dead.
"""

import os
import json
import math
import time
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    # Architecture (both modules share this exactly)
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    context_length: int = 256
    vocab_size: int = 512  # small BPE vocab for speed
    dropout: float = 0.0

    # Alignment protocol
    freeze_layers: int = 2  # first K layers are frozen (shared backbone)
    seed: int = 42          # canonical seed θ₀

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    eval_interval: int = 200
    warmup_steps: int = 100

    # Data
    domain_a: str = "code"       # Module A specializes in code-like text
    domain_b: str = "stories"    # Module B specializes in narrative text
    data_tokens: int = 2_000_000  # tokens per domain

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Output
    output_dir: str = "kalavai_experiment_results"


# ============================================================================
# Minimal GPT Model
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
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
        # Use scaled dot-product attention (Flash Attention if available)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(self.dropout(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU (from Slowrun findings: more data-efficient than GELU)
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
        # Weight tying
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
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def freeze_early_layers(self, n_freeze):
        """Freeze the first n_freeze transformer blocks + embeddings."""
        # Freeze embeddings (shared representation)
        self.tok_emb.weight.requires_grad = False
        self.pos_emb.weight.requires_grad = False
        # Freeze early blocks
        for i in range(n_freeze):
            for param in self.blocks[i].parameters():
                param.requires_grad = False

    def count_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Simple MoE Router for Fusion Backend A
# ============================================================================

class SimpleMoEFusion(nn.Module):
    """
    Given two specialist modules, route each token to one of them
    based on a learned router on the shared frozen backbone output.
    """
    def __init__(self, config, module_a, module_b):
        super().__init__()
        self.config = config
        self.module_a = module_a
        self.module_b = module_b
        # Router operates on the output of shared frozen layers
        self.router = nn.Linear(config.d_model, 2, bias=False)
        # Shared components (identical between a and b due to frozen layers)
        self.tok_emb = module_a.tok_emb  # shared
        self.pos_emb = module_a.pos_emb  # shared
        self.shared_blocks = nn.ModuleList([module_a.blocks[i] for i in range(config.freeze_layers)])
        # Specialist blocks from each module
        self.specialist_a = nn.ModuleList([module_a.blocks[i] for i in range(config.freeze_layers, config.n_layers)])
        self.specialist_b = nn.ModuleList([module_b.blocks[i] for i in range(config.freeze_layers, config.n_layers)])
        # Final layers (use averaged)
        self.ln_f = module_a.ln_f  # RMSNorm has no learned params in this impl
        self.lm_head = module_a.lm_head  # tied to tok_emb

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        # Shared frozen backbone
        for block in self.shared_blocks:
            x = block(x)

        # Router decides weights per token
        router_logits = self.router(x.detach())  # detach: don't backprop routing through backbone
        weights = F.softmax(router_logits, dim=-1)  # (B, T, 2)

        # Run both specialists
        x_a, x_b = x.clone(), x.clone()
        for block in self.specialist_a:
            x_a = block(x_a)
        for block in self.specialist_b:
            x_b = block(x_b)

        # Weighted combination
        x = weights[:, :, 0:1] * x_a + weights[:, :, 1:2] * x_b

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ============================================================================
# Data: Synthetic Domain Split
# ============================================================================

def generate_synthetic_data(config):
    """
    Generate two synthetic text domains with different statistical properties.
    Domain A: code-like (structured, repetitive patterns, special chars)
    Domain B: story-like (natural language patterns, longer sequences)

    For a real experiment, replace with actual domain corpora.
    """
    print("Generating synthetic training data...")
    rng = random.Random(config.seed + 100)  # separate seed for data

    # Domain A: code-like patterns
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

    # Domain B: story-like patterns
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

    # Generate raw text for each domain
    text_a = "".join(gen_code_line() for _ in range(config.data_tokens // 20))
    text_b = "".join(gen_story_line() for _ in range(config.data_tokens // 30))

    return text_a, text_b


class TextDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length - 1)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def tokenize_simple(text, vocab_size):
    """
    Simple byte-level tokenization with vocab_size cap.
    For a real experiment, use minbpe or tiktoken.
    """
    tokens = [min(b, vocab_size - 1) for b in text.encode("utf-8")]
    return tokens


# ============================================================================
# Training Loop
# ============================================================================

def train_module(model, train_dataset, eval_datasets, config, module_name, device):
    """Train a single module and return training history."""
    model = model.to(device)
    model.train()

    # Only optimize trainable params (unfrozen layers)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    train_iter = iter(train_loader)

    history = {"train_loss": [], "eval_losses": {name: [] for name in eval_datasets}, "steps": []}

    total_params = model.count_params()
    trainable_params = model.count_params(trainable_only=True)
    frozen_params = total_params - trainable_params
    print(f"\n{'='*60}")
    print(f"Training {module_name}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen params:    {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"{'='*60}")

    t0 = time.time()
    for step in range(1, config.max_steps + 1):
        # Get batch (cycle through data)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Learning rate schedule: linear warmup then cosine decay
        if step < config.warmup_steps:
            lr = config.learning_rate * step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            lr = config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if step % config.eval_interval == 0 or step == 1:
            model.eval()
            history["steps"].append(step)
            history["train_loss"].append(loss.item())

            with torch.no_grad():
                for eval_name, eval_dataset in eval_datasets.items():
                    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, num_workers=0)
                    eval_losses = []
                    for ex, ey in eval_loader:
                        ex, ey = ex.to(device), ey.to(device)
                        _, el = model(ex, ey)
                        eval_losses.append(el.item())
                        if len(eval_losses) >= 10:  # quick eval
                            break
                    avg_loss = sum(eval_losses) / len(eval_losses)
                    history["eval_losses"][eval_name].append(avg_loss)

            elapsed = time.time() - t0
            evals_str = " | ".join(f"{k}: {v[-1]:.4f}" for k, v in history["eval_losses"].items())
            print(f"  [{module_name}] step {step:5d} | train: {loss.item():.4f} | {evals_str} | {elapsed:.1f}s")
            model.train()

    return history


@torch.no_grad()
def evaluate_model(model, eval_datasets, config, device):
    """Evaluate a model on all eval datasets."""
    model = model.to(device)
    model.eval()
    results = {}
    for eval_name, eval_dataset in eval_datasets.items():
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, num_workers=0)
        losses = []
        for ex, ey in eval_loader:
            ex, ey = ex.to(device), ey.to(device)
            _, el = model(ex, ey)
            losses.append(el.item())
            if len(losses) >= 20:
                break
        results[eval_name] = sum(losses) / len(losses)
    return results


# ============================================================================
# Fusion Methods
# ============================================================================

def fuse_by_averaging(module_a, module_b, config):
    """
    Fusion Backend A (simple): Average the unfrozen layer weights.
    Frozen layers are already identical.
    """
    fused = copy.deepcopy(module_a)
    for i in range(config.freeze_layers, config.n_layers):
        for pa, pb in zip(fused.blocks[i].parameters(), module_b.blocks[i].parameters()):
            pa.data = (pa.data + pb.data) / 2.0
    # Average the final layer norm and lm_head (which is tied to tok_emb)
    # ln_f: RMSNorm has no learnable params in standard impl, skip
    # lm_head is tied to tok_emb which is frozen, so it's already shared
    return fused


def fuse_by_moe(module_a, module_b, config, train_dataset, device, router_steps=500):
    """
    Fusion Backend B: Simple MoE routing with learned router.
    Train the router on mixed-domain data.
    """
    moe = SimpleMoEFusion(config, module_a, module_b).to(device)

    # Freeze everything except the router
    for name, param in moe.named_parameters():
        if "router" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(moe.router.parameters(), lr=1e-3)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=0)
    train_iter = iter(train_loader)

    print(f"\n  Training MoE router for {router_steps} steps...")
    moe.train()
    for step in range(1, router_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        _, loss = moe(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"    Router step {step}: loss={loss.item():.4f}")

    return moe


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    config = ExperimentConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    print("=" * 70)
    print("KALAVAI 2-Module Proof-of-Concept Experiment")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Architecture: {config.n_layers} layers, d={config.d_model}, {config.n_heads} heads")
    print(f"Frozen layers: {config.freeze_layers} (shared backbone)")
    print(f"Training: {config.max_steps} steps, lr={config.learning_rate}")
    print(f"Canonical seed: {config.seed}")

    # ---- Step 1: Generate data ----
    text_a, text_b = generate_synthetic_data(config)
    tokens_a = tokenize_simple(text_a, config.vocab_size)
    tokens_b = tokenize_simple(text_b, config.vocab_size)
    print(f"\nDomain A ({config.domain_a}): {len(tokens_a):,} tokens")
    print(f"Domain B ({config.domain_b}): {len(tokens_b):,} tokens")

    # Split into train/eval
    split_a = int(len(tokens_a) * 0.9)
    split_b = int(len(tokens_b) * 0.9)

    train_a = TextDataset(tokens_a[:split_a], config.context_length)
    eval_a = TextDataset(tokens_a[split_a:], config.context_length)
    train_b = TextDataset(tokens_b[:split_b], config.context_length)
    eval_b = TextDataset(tokens_b[split_b:], config.context_length)

    # Mixed eval: interleave both domains
    mixed_eval_tokens = []
    for i in range(0, min(len(tokens_a[split_a:]), len(tokens_b[split_b:])), config.context_length):
        mixed_eval_tokens.extend(tokens_a[split_a + i: split_a + i + config.context_length // 2])
        mixed_eval_tokens.extend(tokens_b[split_b + i: split_b + i + config.context_length // 2])
    eval_mixed = TextDataset(mixed_eval_tokens, config.context_length)

    eval_datasets = {"code": eval_a, "stories": eval_b, "mixed": eval_mixed}

    # ---- Step 2: Create canonical seed checkpoint ----
    print("\nGenerating canonical seed θ₀...")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    seed_model = MiniGPT(config)
    seed_checkpoint = copy.deepcopy(seed_model.state_dict())

    # Evaluate seed (untrained baseline)
    seed_results = evaluate_model(seed_model, eval_datasets, config, config.device)
    print(f"\nSeed model (untrained) eval:")
    for k, v in seed_results.items():
        print(f"  {k}: {v:.4f}")

    # ---- Step 3: Train Module A (code specialist) ----
    module_a = MiniGPT(config)
    module_a.load_state_dict(seed_checkpoint)  # Initialize from θ₀
    module_a.freeze_early_layers(config.freeze_layers)

    history_a = train_module(
        module_a, train_a, eval_datasets, config,
        module_name=f"Module A ({config.domain_a})", device=config.device,
    )

    # ---- Step 4: Train Module B (stories specialist) ----
    module_b = MiniGPT(config)
    module_b.load_state_dict(seed_checkpoint)  # Initialize from same θ₀
    module_b.freeze_early_layers(config.freeze_layers)

    history_b = train_module(
        module_b, train_b, eval_datasets, config,
        module_name=f"Module B ({config.domain_b})", device=config.device,
    )

    # ---- Step 5: Fuse models ----
    print("\n" + "=" * 60)
    print("FUSION")
    print("=" * 60)

    # Method 1: Simple weight averaging
    print("\nFusion Method 1: Weight Averaging (unfrozen layers)")
    fused_avg = fuse_by_averaging(module_a, module_b, config)
    results_avg = evaluate_model(fused_avg, eval_datasets, config, config.device)
    for k, v in results_avg.items():
        print(f"  {k}: {v:.4f}")

    # Method 2: MoE routing (train router on mixed data)
    print("\nFusion Method 2: MoE Routing (learned router)")
    # Create mixed training data for router
    mixed_train_tokens = []
    for i in range(0, min(split_a, split_b), config.context_length):
        mixed_train_tokens.extend(tokens_a[i:i + config.context_length // 2])
        mixed_train_tokens.extend(tokens_b[i:i + config.context_length // 2])
    mixed_train = TextDataset(mixed_train_tokens, config.context_length)
    fused_moe = fuse_by_moe(module_a, module_b, config, mixed_train, config.device)
    results_moe = evaluate_model(fused_moe, eval_datasets, config, config.device)
    for k, v in results_moe.items():
        print(f"  {k}: {v:.4f}")

    # ---- Step 6: Final evaluation on individual modules ----
    print("\nFinal evaluation of individual modules:")
    results_a = evaluate_model(module_a, eval_datasets, config, config.device)
    results_b = evaluate_model(module_b, eval_datasets, config, config.device)

    # ---- Step 7: Results summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Code':>10} {'Stories':>10} {'Mixed':>10} {'Average':>10}")
    print("-" * 65)

    models = {
        "Seed (untrained)": seed_results,
        f"Module A ({config.domain_a})": results_a,
        f"Module B ({config.domain_b})": results_b,
        "Fused (averaging)": results_avg,
        "Fused (MoE routing)": results_moe,
    }

    for name, results in models.items():
        avg = sum(results.values()) / len(results)
        print(f"{name:<25} {results['code']:>10.4f} {results['stories']:>10.4f} {results['mixed']:>10.4f} {avg:>10.4f}")

    # ---- Step 8: Verdict ----
    best_individual_avg = min(
        sum(results_a.values()) / len(results_a),
        sum(results_b.values()) / len(results_b),
    )
    best_fused_avg = min(
        sum(results_avg.values()) / len(results_avg),
        sum(results_moe.values()) / len(results_moe),
    )
    individual_avg_of_avgs = (
        sum(results_a.values()) / len(results_a) +
        sum(results_b.values()) / len(results_b)
    ) / 2

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Best individual module (avg across domains): {best_individual_avg:.4f}")
    print(f"  Average of both individuals:                 {individual_avg_of_avgs:.4f}")
    print(f"  Best fused model (avg across domains):       {best_fused_avg:.4f}")

    if best_fused_avg < best_individual_avg:
        improvement = (best_individual_avg - best_fused_avg) / best_individual_avg * 100
        print(f"\n  ✅ THESIS HAS LEGS: Fused model outperforms best individual by {improvement:.1f}%")
        print(f"     The fusion of independently trained specialists produces something")
        print(f"     better than either specialist alone.")
    elif best_fused_avg < individual_avg_of_avgs:
        print(f"\n  ⚠️  MIXED RESULTS: Fused model outperforms average of individuals")
        print(f"     but not the best individual. Fusion adds some value but")
        print(f"     doesn't clearly justify the coordination overhead.")
    else:
        print(f"\n  ❌ THESIS IS DEAD: Fused model is worse than average of individuals.")
        print(f"     Independent training + fusion doesn't work at this scale/config.")
        print(f"     Before giving up: try different freeze depths, more training steps,")
        print(f"     or real (non-synthetic) domain data.")

    # ---- Save results ----
    output = {
        "config": {k: v for k, v in config.__dict__.items()},
        "results": {name: {k: round(v, 6) for k, v in res.items()} for name, res in models.items()},
        "verdict": {
            "best_individual": round(best_individual_avg, 6),
            "avg_of_individuals": round(individual_avg_of_avgs, 6),
            "best_fused": round(best_fused_avg, 6),
            "fusion_works": best_fused_avg < best_individual_avg,
        },
        "history_a": {k: [round(x, 6) if isinstance(x, float) else x for x in v] if isinstance(v, list) else
                      {kk: [round(xx, 6) for xx in vv] for kk, vv in v.items()} for k, v in history_a.items()},
        "history_b": {k: [round(x, 6) if isinstance(x, float) else x for x in v] if isinstance(v, list) else
                      {kk: [round(xx, 6) for xx in vv] for kk, vv in v.items()} for k, v in history_b.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    results_path = os.path.join(config.output_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("Done.")


if __name__ == "__main__":
    main()
