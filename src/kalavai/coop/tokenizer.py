"""BPE tokenizer training, saving, and loading for Kalavai cooperatives.

Implements a minimal byte-pair encoding tokenizer inspired by Karpathy's minbpe.
Deterministic: same corpus and vocab_size always produce the same tokenizer.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Union


from kalavai.core.exceptions import ConfigError

# Fixed seed for deterministic training.
_SEED = 42


class Tokenizer:
    """Minimal BPE tokenizer with encode/decode support.

    Attributes:
        vocab_size: Total vocabulary size (256 byte tokens + learned merges).
        merges: Ordered list of (pair, new_id) merge rules learned during training.
        vocab: Mapping from token id to bytes.
    """

    def __init__(self) -> None:
        self.merges: list[tuple[tuple[int, int], int]] = []
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size: int = 256

    def train(self, text: str, vocab_size: int = 4096) -> None:
        """Train BPE merges on *text* until *vocab_size* is reached.

        Args:
            text: Training corpus as a single string.
            vocab_size: Desired vocabulary size (must be >= 256).

        Raises:
            ConfigError: If vocab_size < 256.
        """
        if vocab_size < 256:
            raise ConfigError("vocab_size must be >= 256 (the number of byte tokens)")

        random.seed(_SEED)

        tokens: list[int] = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for i in range(num_merges):
            if len(tokens) < 2:
                break

            # Count consecutive pairs.
            pair_counts: Counter[tuple[int, int]] = Counter()
            for a, b in zip(tokens, tokens[1:]):
                pair_counts[(a, b)] += 1

            if not pair_counts:
                break

            # Pick the most frequent pair (deterministic: max by count then pair value).
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] < 2:
                break  # No pair occurs more than once — stop early.

            new_id = 256 + i
            # Merge the best pair in the token list.
            tokens = _merge(tokens, best_pair, new_id)

            self.merges.append((best_pair, new_id))
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        self.vocab_size = 256 + len(self.merges)

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token ids.

        Args:
            text: The string to encode.

        Returns:
            List of integer token ids.
        """
        tokens = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            tokens = _merge(tokens, pair, new_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string.

        Args:
            ids: List of integer token ids.

        Returns:
            The decoded string.
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise tokenizer state to a JSON-safe dictionary."""
        return {
            "merges": [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tokenizer:
        """Reconstruct a Tokenizer from a dictionary produced by *to_dict*."""
        tok = cls()
        for entry in data["merges"]:
            pair = tuple(entry["pair"])
            new_id = entry["new_id"]
            tok.merges.append((pair, new_id))  # type: ignore[arg-type]
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        tok.vocab_size = 256 + len(tok.merges)
        return tok


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


def train_tokenizer(
    corpus_path: Union[str, Path],
    vocab_size: int = 4096,
) -> Tokenizer:
    """Train a BPE tokenizer on a text corpus.

    Args:
        corpus_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256).

    Returns:
        A trained Tokenizer instance.

    Raises:
        ConfigError: If *corpus_path* does not exist or is unreadable.
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise ConfigError(f"Corpus file not found: {corpus_path}")

    try:
        text = corpus_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read corpus file: {exc}") from exc

    tok = Tokenizer()
    tok.train(text, vocab_size=vocab_size)
    return tok


def save_tokenizer(tokenizer: Tokenizer, path: Union[str, Path]) -> None:
    """Save a trained tokenizer to a JSON file.

    Args:
        tokenizer: The Tokenizer to persist.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokenizer.to_dict(), indent=2), encoding="utf-8")


def load_tokenizer(path: Union[str, Path]) -> Tokenizer:
    """Load a tokenizer from a JSON file.

    Args:
        path: Path to a file previously written by *save_tokenizer*.

    Returns:
        A Tokenizer instance with the same merges.

    Raises:
        ConfigError: If *path* does not exist or is unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Tokenizer file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot load tokenizer: {exc}") from exc

    return Tokenizer.from_dict(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of *pair* in *tokens* with *new_id*."""
    out: list[int] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
