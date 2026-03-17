"""Tests for kalavai.coop.calibration — calibration batch generation and loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kalavai.coop.calibration import generate_calibration_batch, load_calibration_batch
from kalavai.coop.tokenizer import Tokenizer, save_tokenizer
from kalavai.core.exceptions import ConfigError

# Repetitive corpus so BPE can learn merges and produce enough tokens.
SAMPLE_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
) * 200


@pytest.fixture()
def trained_tokenizer_path(tmp_path: Path) -> Path:
    """Train a small tokenizer, save it, and return the file path."""
    tok = Tokenizer()
    tok.train(SAMPLE_CORPUS, vocab_size=300)
    p = tmp_path / "tokenizer.json"
    save_tokenizer(tok, p)
    return p


@pytest.fixture()
def corpus_file(tmp_path: Path) -> Path:
    """Write SAMPLE_CORPUS to a temporary file and return its path."""
    p = tmp_path / "corpus.txt"
    p.write_text(SAMPLE_CORPUS, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateCalibrationBatch:
    """Tests for generate_calibration_batch()."""

    def test_creates_file_with_correct_shape(
        self, trained_tokenizer_path: Path, corpus_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "calibration_batch.pt"
        n_seq, seq_len = 8, 32
        generate_calibration_batch(
            trained_tokenizer_path, corpus_file, out,
            n_sequences=n_seq, seq_length=seq_len,
        )
        assert out.exists()
        batch = torch.load(out, weights_only=True)
        assert batch.shape == (n_seq, seq_len)
        assert batch.dtype == torch.long

    def test_small_corpus_fewer_sequences(
        self, trained_tokenizer_path: Path, tmp_path: Path
    ) -> None:
        """When corpus is too small for n_sequences, produce fewer rows."""
        # Tiny corpus — enough for a few sequences but not 1024.
        small_corpus = tmp_path / "small.txt"
        small_corpus.write_text("hello world " * 50, encoding="utf-8")

        out = tmp_path / "calibration_batch.pt"
        generate_calibration_batch(
            trained_tokenizer_path, small_corpus, out,
            n_sequences=1024, seq_length=16,
        )
        batch = torch.load(out, weights_only=True)
        assert batch.shape[0] < 1024
        assert batch.shape[1] == 16

    def test_load_roundtrip(
        self, trained_tokenizer_path: Path, corpus_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "calibration_batch.pt"
        generate_calibration_batch(
            trained_tokenizer_path, corpus_file, out,
            n_sequences=4, seq_length=16,
        )
        loaded = load_calibration_batch(out)
        expected = torch.load(out, weights_only=True)
        assert torch.equal(loaded, expected)

    def test_missing_tokenizer_raises(
        self, corpus_file: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigError, match="Tokenizer file not found"):
            generate_calibration_batch(
                tmp_path / "no_tokenizer.json", corpus_file,
                tmp_path / "out.pt",
            )

    def test_missing_corpus_raises(
        self, trained_tokenizer_path: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigError, match="Corpus file not found"):
            generate_calibration_batch(
                trained_tokenizer_path, tmp_path / "nonexistent.txt",
                tmp_path / "out.pt",
            )

    def test_corpus_too_small_for_one_sequence_raises(
        self, trained_tokenizer_path: Path, tmp_path: Path
    ) -> None:
        tiny = tmp_path / "tiny.txt"
        tiny.write_text("hi", encoding="utf-8")
        with pytest.raises(ConfigError, match="Corpus too small"):
            generate_calibration_batch(
                trained_tokenizer_path, tiny, tmp_path / "out.pt",
                n_sequences=1, seq_length=9999,
            )


class TestLoadCalibrationBatch:
    """Tests for load_calibration_batch()."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Calibration batch not found"):
            load_calibration_batch(tmp_path / "missing.pt")
