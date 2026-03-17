"""Canonical seed checkpoint generation for Kalavai cooperatives.

Generates a reproducible seed checkpoint (theta-zero) that all cooperative
members use as their starting point. Same config + same seed always produces
an identical checkpoint file and hash.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from kalavai.core.config import ArchitectureConfig
from kalavai.core.model import create_model_from_config


def generate_seed_checkpoint(
    arch_config: ArchitectureConfig,
    output_path: Path | str,
    vocab_size: int = 4096,
    seed: int = 42,
) -> str:
    """Initialize a model with a fixed seed and save as the canonical seed checkpoint.

    The function is fully deterministic: given the same architecture config
    and seed, it produces a byte-identical checkpoint file every time.

    Args:
        arch_config: Architecture configuration for the model.
        output_path: File path to save the checkpoint (e.g. ``seed_checkpoint.pt``).
        vocab_size: Vocabulary size for the model.
        seed: Random seed for reproducible initialization.

    Returns:
        SHA-256 hex digest of the saved checkpoint file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic initialization
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = create_model_from_config(arch_config, vocab_size=vocab_size)

    # Serialize to an in-memory buffer first to avoid non-deterministic
    # file metadata (e.g. zip timestamps) that torch.save embeds when
    # writing directly to disk.
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    checkpoint_bytes = buf.getvalue()

    output_path.write_bytes(checkpoint_bytes)

    sha = hashlib.sha256(checkpoint_bytes)
    return sha.hexdigest()
