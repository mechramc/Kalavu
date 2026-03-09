"""Minimal GPT-style transformer model compatible with Kalavu's ArchitectureConfig.

Provides a nanochat-compatible transformer used as the canonical seed model
for cooperative training. Supports extracting hidden representations at
specified probe layers for CKA alignment measurement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from kalavu.core.config import ArchitectureConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class KalavuTransformerBlock(nn.Module):
    """Single transformer block with multi-head self-attention and FFN.

    Args:
        d_model: Hidden dimension size.
        n_heads: Number of attention heads.
        ffn_ratio: Multiplier for FFN hidden dimension.
        norm: Normalization type ("rmsnorm" or "layernorm").
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: float,
        norm: str = "rmsnorm",
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn_up = nn.Linear(d_model, ffn_hidden, bias=False)
        self.ffn_gate = nn.Linear(d_model, ffn_hidden, bias=False)
        self.ffn_down = nn.Linear(ffn_hidden, d_model, bias=False)

        # Norms (pre-norm architecture)
        norm_cls = RMSNorm if norm == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm_cls(d_model)
        self.ffn_norm = norm_cls(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention with residual
        h = self.attn_norm(x)
        h = self._attention(h)
        x = x + h

        # FFN with residual (SwiGLU-style)
        h = self.ffn_norm(x)
        h = self.ffn_down(f.silu(self.ffn_gate(h)) * self.ffn_up(h))
        x = x + h

        return x

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal self-attention
        out = f.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)


class KalavuModel(nn.Module):
    """GPT-style transformer model for Kalavu cooperative training.

    Args:
        depth: Number of transformer layers.
        d_model: Hidden dimension size.
        n_heads: Number of attention heads.
        ffn_ratio: Multiplier for FFN hidden dimension.
        norm: Normalization type ("rmsnorm" or "layernorm").
        vocab_size: Vocabulary size for token embeddings.
        max_seq_len: Maximum sequence length for position embeddings.
    """

    def __init__(
        self,
        depth: int,
        d_model: int,
        n_heads: int,
        ffn_ratio: float,
        norm: str = "rmsnorm",
        vocab_size: int = 4096,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            KalavuTransformerBlock(d_model, n_heads, ffn_ratio, norm)
            for _ in range(depth)
        ])

        # Final norm and output projection
        norm_cls = RMSNorm if norm == "rmsnorm" else nn.LayerNorm
        self.final_norm = norm_cls(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        self.output_proj.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        _bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.output_proj(x)

    def get_probe_representations(
        self,
        input_ids: torch.Tensor,
        probe_layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Extract hidden representations at specified layers for CKA probing.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            probe_layers: List of layer indices (0-based) to extract from.

        Returns:
            Dictionary mapping layer index to hidden state tensor of
            shape (batch, seq_len, d_model).
        """
        _bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        probe_set = set(probe_layers)
        representations: dict[int, torch.Tensor] = {}

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in probe_set:
                representations[i] = x.detach()

        return representations


def create_model_from_config(
    arch_config: ArchitectureConfig,
    vocab_size: int = 4096,
) -> KalavuModel:
    """Create a KalavuModel from an ArchitectureConfig.

    Args:
        arch_config: Architecture configuration specifying model dimensions.
        vocab_size: Vocabulary size for embeddings.

    Returns:
        An initialized KalavuModel instance.
    """
    return KalavuModel(
        depth=arch_config.depth,
        d_model=arch_config.d_model,
        n_heads=arch_config.n_heads,
        ffn_ratio=arch_config.ffn_ratio,
        norm=arch_config.norm,
        vocab_size=vocab_size,
    )
