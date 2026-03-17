"""Calibration batch generation for CKA alignment computation.

Tokenizes sequences from a configurable corpus using the trained cooperative
tokenizer.  The resulting tensor is saved once per cooperative and reused by
all participants for alignment checks.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from kalavai.coop.tokenizer import load_tokenizer
from kalavai.core.exceptions import ConfigError


def generate_calibration_batch(
    tokenizer_path: Path,
    corpus_path: Path,
    output_path: Path,
    n_sequences: int = 1024,
    seq_length: int = 128,
) -> None:
    """Tokenize a corpus and save a calibration batch tensor.

    Args:
        tokenizer_path: Path to a saved tokenizer JSON file.
        corpus_path: Path to a UTF-8 text corpus.
        output_path: Destination ``.pt`` file for the calibration tensor.
        n_sequences: Desired number of sequences (rows).  If the corpus is
            too small, fewer sequences are produced.
        seq_length: Number of tokens per sequence (columns).

    Raises:
        ConfigError: If *tokenizer_path* or *corpus_path* do not exist or
            cannot be read.
    """
    # ── Validate inputs ────────────────────────────────────────────────
    tokenizer_path = Path(tokenizer_path)
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)

    if not corpus_path.exists():
        raise ConfigError(f"Corpus file not found: {corpus_path}")

    # load_tokenizer already raises ConfigError for missing/invalid files.
    tokenizer = load_tokenizer(tokenizer_path)

    try:
        text = corpus_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read corpus file: {exc}") from exc

    # ── Tokenize ───────────────────────────────────────────────────────
    token_ids = tokenizer.encode(text)

    if len(token_ids) < seq_length:
        raise ConfigError(
            f"Corpus too small: got {len(token_ids)} tokens, "
            f"need at least {seq_length} for one sequence"
        )

    # ── Chunk into sequences ───────────────────────────────────────────
    # Use as many full sequences as the corpus provides, up to n_sequences.
    available_sequences = len(token_ids) // seq_length
    actual_sequences = min(n_sequences, available_sequences)

    # Truncate to an exact multiple of seq_length.
    total_tokens = actual_sequences * seq_length
    token_ids = token_ids[:total_tokens]

    batch = torch.tensor(token_ids, dtype=torch.long).reshape(actual_sequences, seq_length)

    # ── Save ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, output_path)


def load_calibration_batch(path: Path) -> Tensor:
    """Load a calibration batch tensor from disk.

    Args:
        path: Path to a ``.pt`` file saved by *generate_calibration_batch*.

    Returns:
        A ``torch.Tensor`` of shape ``[n_sequences, seq_length]``.

    Raises:
        ConfigError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Calibration batch not found: {path}")

    return torch.load(path, weights_only=True)
