"""Tests for kalavai.coop.tokenizer — BPE tokenizer training and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalavai.coop.tokenizer import (
    Tokenizer,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)
from kalavai.core.exceptions import ConfigError

# A small but repetitive corpus so BPE can learn merges.
SAMPLE_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "pack my box with five dozen liquor jugs. "
    "pack my box with five dozen liquor jugs. "
    "pack my box with five dozen liquor jugs. "
) * 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def corpus_file(tmp_path: Path) -> Path:
    """Write SAMPLE_CORPUS to a temporary file and return its path."""
    p = tmp_path / "corpus.txt"
    p.write_text(SAMPLE_CORPUS, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainTokenizer:
    """Tests for train_tokenizer() and the Tokenizer class."""

    def test_encode_decode_roundtrip(self, corpus_file: Path) -> None:
        tok = train_tokenizer(corpus_file, vocab_size=300)
        text = "the quick brown fox"
        assert tok.decode(tok.encode(text)) == text

    def test_encode_decode_roundtrip_unicode(self, corpus_file: Path) -> None:
        tok = train_tokenizer(corpus_file, vocab_size=300)
        # Characters outside the training corpus still work via byte fallback.
        text = "café résumé naïve"
        assert tok.decode(tok.encode(text)) == text

    def test_vocab_size_respected(self, corpus_file: Path) -> None:
        tok = train_tokenizer(corpus_file, vocab_size=280)
        assert tok.vocab_size <= 280

    def test_missing_corpus_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Corpus file not found"):
            train_tokenizer(tmp_path / "nonexistent.txt")

    def test_vocab_size_too_small_raises(self, corpus_file: Path) -> None:
        with pytest.raises(ConfigError, match="vocab_size must be >= 256"):
            tok = Tokenizer()
            tok.train(corpus_file.read_text(), vocab_size=100)


class TestDeterminism:
    """Same corpus + same vocab_size → identical tokenizer."""

    def test_same_merges(self, corpus_file: Path) -> None:
        tok1 = train_tokenizer(corpus_file, vocab_size=300)
        tok2 = train_tokenizer(corpus_file, vocab_size=300)
        assert tok1.merges == tok2.merges

    def test_same_encoding(self, corpus_file: Path) -> None:
        tok1 = train_tokenizer(corpus_file, vocab_size=300)
        tok2 = train_tokenizer(corpus_file, vocab_size=300)
        text = "the quick brown fox jumps over the lazy dog"
        assert tok1.encode(text) == tok2.encode(text)


class TestSaveLoad:
    """Tests for save_tokenizer / load_tokenizer round-trip."""

    def test_save_load_roundtrip(self, corpus_file: Path, tmp_path: Path) -> None:
        tok = train_tokenizer(corpus_file, vocab_size=300)
        model_path = tmp_path / "tokenizer.model"
        save_tokenizer(tok, model_path)

        loaded = load_tokenizer(model_path)
        assert loaded.merges == tok.merges
        assert loaded.vocab_size == tok.vocab_size

    def test_loaded_tokenizer_encodes_same(
        self, corpus_file: Path, tmp_path: Path
    ) -> None:
        tok = train_tokenizer(corpus_file, vocab_size=300)
        model_path = tmp_path / "tokenizer.model"
        save_tokenizer(tok, model_path)
        loaded = load_tokenizer(model_path)

        text = "the quick brown fox jumps over the lazy dog"
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(loaded.encode(text)) == text

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Tokenizer file not found"):
            load_tokenizer(tmp_path / "nope.model")
