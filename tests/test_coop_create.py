"""Tests for end-to-end cooperative creation (kalavai coop create)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kalavai.coop.create import create_cooperative
from kalavai.coop.tokenizer import load_tokenizer
from kalavai.core.config import CooperativeConfig


# ---------------------------------------------------------------------------
# E2E: all 6 files created
# ---------------------------------------------------------------------------


class TestCreateCooperative:
    """End-to-end tests for create_cooperative."""

    def test_creates_all_files(self, tmp_path: Path) -> None:
        """All 6 required files are produced in the output directory."""
        out = create_cooperative(
            name="test-coop",
            modules=5,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        expected_files = [
            "kalavai.yaml",
            "tokenizer.model",
            "seed_checkpoint.pt",
            "calibration_batch.pt",
            "cka_reference.pt",
            "domain_manifest.json",
        ]
        for fname in expected_files:
            assert (out / fname).exists(), f"Missing file: {fname}"

    def test_config_loadable(self, tmp_path: Path) -> None:
        """kalavai.yaml is loadable as a CooperativeConfig."""
        out = create_cooperative(
            name="test-coop",
            modules=5,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        config = CooperativeConfig.from_yaml(out / "kalavai.yaml")
        assert config.name == "test-coop"
        assert config.modules == 5
        assert config.target_params_per_module == "14M"

    def test_manifest_has_correct_slots(self, tmp_path: Path) -> None:
        """domain_manifest.json has the correct number of module slots."""
        out = create_cooperative(
            name="test-coop",
            modules=5,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        manifest = json.loads((out / "domain_manifest.json").read_text(encoding="utf-8"))
        assert manifest["cooperative"] == "test-coop"
        assert len(manifest["slots"]) == 5
        for slot in manifest["slots"]:
            assert slot["status"] == "open"

    def test_tokenizer_loadable(self, tmp_path: Path) -> None:
        """Saved tokenizer can be loaded and used for encoding."""
        out = create_cooperative(
            name="test-coop",
            modules=5,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        tok = load_tokenizer(out / "tokenizer.model")
        assert tok.vocab_size >= 256
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_synthetic_corpus_cleanup(self, tmp_path: Path) -> None:
        """When no corpus is provided, synthetic corpus is cleaned up after creation."""
        out = create_cooperative(
            name="test-coop",
            modules=3,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        # Synthetic corpus file should be cleaned up
        assert not (out / "_synthetic_corpus.txt").exists()

    def test_seed_checkpoint_loadable(self, tmp_path: Path) -> None:
        """seed_checkpoint.pt contains a valid state dict."""
        out = create_cooperative(
            name="test-coop",
            modules=3,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        state_dict = torch.load(out / "seed_checkpoint.pt", weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_calibration_batch_loadable(self, tmp_path: Path) -> None:
        """calibration_batch.pt contains a valid tensor."""
        out = create_cooperative(
            name="test-coop",
            modules=3,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        batch = torch.load(out / "calibration_batch.pt", weights_only=True)
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2

    def test_cka_reference_loadable(self, tmp_path: Path) -> None:
        """cka_reference.pt contains a dict mapping layer indices to tensors."""
        out = create_cooperative(
            name="test-coop",
            modules=3,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            vocab_size=300,
            seed=42,
        )

        ref = torch.load(out / "cka_reference.pt", weights_only=True)
        assert isinstance(ref, dict)
        for key, val in ref.items():
            assert isinstance(key, int)
            assert isinstance(val, torch.Tensor)

    def test_with_custom_corpus(self, tmp_path: Path) -> None:
        """Works with an explicit corpus file."""
        corpus = tmp_path / "corpus.txt"
        text = "The quick brown fox jumps over the lazy dog. " * 2000
        corpus.write_text(text, encoding="utf-8")

        out = create_cooperative(
            name="test-coop",
            modules=3,
            target_params="14M",
            output_dir=tmp_path / "test-coop",
            corpus_path=corpus,
            vocab_size=300,
            seed=42,
        )

        assert (out / "kalavai.yaml").exists()
        assert (out / "tokenizer.model").exists()
