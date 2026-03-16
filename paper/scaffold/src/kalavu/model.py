"""
Synthetic transformer model for KALAVU proof-of-concept experiments.

Ported from kalavu_experiment.py — identical architecture and weight-sharing
protocol. Used by kalavu.train for synthetic scale experiments.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model configuration (flat, matches kalavu_experiment.py ExperimentConfig)
# ============================================================================

@dataclass
class ModelConfig:
    n_layers: int = 12
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    context_length: int = 256
    vocab_size: int = 512
    dropout: float = 0.0
    freeze_layers: int = 2
    # Training hyperparams (used by train.py)
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    eval_interval: int = 200
    warmup_steps: int = 100


# ============================================================================
# Transformer building blocks
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(self.dropout(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
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

    def freeze_early_layers(self, n_freeze: int):
        """Freeze embeddings + first n_freeze transformer blocks (shared backbone)."""
        self.tok_emb.weight.requires_grad = False
        self.pos_emb.weight.requires_grad = False
        for i in range(n_freeze):
            for param in self.blocks[i].parameters():
                param.requires_grad = False

    def count_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# SimpleMoEFusion — 2-module MoE for synthetic experiments
# ============================================================================

class SimpleMoEFusion(nn.Module):
    """
    Given two specialist MiniGPT modules, route each token using a learned
    router applied to the shared frozen backbone output.
    """
    def __init__(self, config: ModelConfig, module_a: MiniGPT, module_b: MiniGPT):
        super().__init__()
        self.config = config
        self.module_a = module_a
        self.module_b = module_b
        self.router = nn.Linear(config.d_model, 2, bias=False)
        # Shared components (identical due to frozen layers + same seed init)
        self.tok_emb = module_a.tok_emb
        self.pos_emb = module_a.pos_emb
        self.shared_blocks = nn.ModuleList(
            [module_a.blocks[i] for i in range(config.freeze_layers)]
        )
        self.specialist_a = nn.ModuleList(
            [module_a.blocks[i] for i in range(config.freeze_layers, config.n_layers)]
        )
        self.specialist_b = nn.ModuleList(
            [module_b.blocks[i] for i in range(config.freeze_layers, config.n_layers)]
        )
        self.ln_f = module_a.ln_f
        self.lm_head = module_a.lm_head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        for block in self.shared_blocks:
            x = block(x)

        # Router on backbone output (detached to avoid backprop through backbone)
        router_logits = self.router(x.detach())
        weights = F.softmax(router_logits, dim=-1)  # (B, T, 2)

        x_a, x_b = x.clone(), x.clone()
        for block in self.specialist_a:
            x_a = block(x_a)
        for block in self.specialist_b:
            x_b = block(x_b)

        x = weights[:, :, 0:1] * x_a + weights[:, :, 1:2] * x_b
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
