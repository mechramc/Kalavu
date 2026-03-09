"""Tests for seed checkpoint generation and the KalavuModel."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kalavu.coop.seed import generate_seed_checkpoint
from kalavu.core.config import ArchitectureConfig
from kalavu.core.model import create_model_from_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_arch() -> ArchitectureConfig:
    """A small architecture config suitable for fast tests."""
    return ArchitectureConfig(
        depth=2,
        d_model=64,
        n_heads=4,
        ffn_ratio=2.0,
        norm="rmsnorm",
    )


@pytest.fixture()
def alt_arch() -> ArchitectureConfig:
    """An alternative small architecture config (different from small_arch)."""
    return ArchitectureConfig(
        depth=3,
        d_model=128,
        n_heads=8,
        ffn_ratio=2.75,
        norm="rmsnorm",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestKalavuModel:
    """Tests for KalavuModel forward pass and probe extraction."""

    def test_forward_output_shape(self, small_arch: ArchitectureConfig) -> None:
        """Forward pass produces logits of shape (batch, seq_len, vocab_size)."""
        vocab_size = 256
        model = create_model_from_config(small_arch, vocab_size=vocab_size)
        model.eval()

        batch, seq_len = 2, 16
        input_ids = torch.randint(0, vocab_size, (batch, seq_len))

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, vocab_size)

    def test_forward_dtype(self, small_arch: ArchitectureConfig) -> None:
        """Logits should be float32 by default."""
        model = create_model_from_config(small_arch, vocab_size=256)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)

        assert logits.dtype == torch.float32

    def test_probe_representations_returns_correct_layers(
        self, small_arch: ArchitectureConfig
    ) -> None:
        """get_probe_representations returns exactly the requested layers."""
        model = create_model_from_config(small_arch, vocab_size=256)
        model.eval()

        input_ids = torch.randint(0, 256, (2, 8))
        probe_layers = [0, 1]  # depth=2, so layers 0 and 1 exist

        with torch.no_grad():
            reps = model.get_probe_representations(input_ids, probe_layers)

        assert set(reps.keys()) == {0, 1}

    def test_probe_representations_shape(
        self, small_arch: ArchitectureConfig
    ) -> None:
        """Each probe representation has shape (batch, seq_len, d_model)."""
        model = create_model_from_config(small_arch, vocab_size=256)
        model.eval()

        batch, seq_len = 3, 10
        input_ids = torch.randint(0, 256, (batch, seq_len))

        with torch.no_grad():
            reps = model.get_probe_representations(input_ids, [0, 1])

        for layer_idx, tensor in reps.items():
            assert tensor.shape == (batch, seq_len, small_arch.d_model)

    def test_probe_empty_list(self, small_arch: ArchitectureConfig) -> None:
        """Requesting no probe layers returns an empty dict."""
        model = create_model_from_config(small_arch, vocab_size=256)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 4))
        with torch.no_grad():
            reps = model.get_probe_representations(input_ids, [])

        assert reps == {}

    def test_layernorm_variant(self) -> None:
        """Model works with layernorm instead of rmsnorm."""
        arch = ArchitectureConfig(
            depth=2, d_model=64, n_heads=4, ffn_ratio=2.0, norm="layernorm"
        )
        model = create_model_from_config(arch, vocab_size=128)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (1, 8, 128)


# ---------------------------------------------------------------------------
# Seed checkpoint tests
# ---------------------------------------------------------------------------


class TestSeedCheckpoint:
    """Tests for reproducible seed checkpoint generation."""

    def test_checkpoint_file_created(
        self, tmp_path: Path, small_arch: ArchitectureConfig
    ) -> None:
        """generate_seed_checkpoint creates a .pt file."""
        out = tmp_path / "seed_checkpoint.pt"
        generate_seed_checkpoint(small_arch, out, vocab_size=256)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_hash_string(
        self, tmp_path: Path, small_arch: ArchitectureConfig
    ) -> None:
        """Return value is a 64-char hex string (SHA-256)."""
        out = tmp_path / "seed_checkpoint.pt"
        h = generate_seed_checkpoint(small_arch, out, vocab_size=256)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # must be valid hex

    def test_reproducible_same_seed(
        self, tmp_path: Path, small_arch: ArchitectureConfig
    ) -> None:
        """Same config + same seed produces identical hash."""
        out1 = tmp_path / "ckpt1.pt"
        out2 = tmp_path / "ckpt2.pt"

        h1 = generate_seed_checkpoint(small_arch, out1, vocab_size=256, seed=42)
        h2 = generate_seed_checkpoint(small_arch, out2, vocab_size=256, seed=42)

        assert h1 == h2

    def test_different_seed_different_hash(
        self, tmp_path: Path, small_arch: ArchitectureConfig
    ) -> None:
        """Different seeds produce different hashes."""
        out1 = tmp_path / "ckpt_seed42.pt"
        out2 = tmp_path / "ckpt_seed99.pt"

        h1 = generate_seed_checkpoint(small_arch, out1, vocab_size=256, seed=42)
        h2 = generate_seed_checkpoint(small_arch, out2, vocab_size=256, seed=99)

        assert h1 != h2

    def test_different_config_different_hash(
        self,
        tmp_path: Path,
        small_arch: ArchitectureConfig,
        alt_arch: ArchitectureConfig,
    ) -> None:
        """Different architecture configs produce different hashes."""
        out1 = tmp_path / "ckpt_small.pt"
        out2 = tmp_path / "ckpt_alt.pt"

        h1 = generate_seed_checkpoint(small_arch, out1, vocab_size=256, seed=42)
        h2 = generate_seed_checkpoint(alt_arch, out2, vocab_size=256, seed=42)

        assert h1 != h2

    def test_checkpoint_loadable(
        self, tmp_path: Path, small_arch: ArchitectureConfig
    ) -> None:
        """The saved checkpoint can be loaded back as a valid state_dict."""
        out = tmp_path / "seed_checkpoint.pt"
        generate_seed_checkpoint(small_arch, out, vocab_size=256)

        state_dict = torch.load(out, map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify it loads into a fresh model
        model = create_model_from_config(small_arch, vocab_size=256)
        model.load_state_dict(state_dict, strict=False)

    def test_creates_parent_directories(
        self, tmp_path: Path, small_arch: ArchitectureConfig
    ) -> None:
        """Output path with non-existent parent dirs is created automatically."""
        out = tmp_path / "nested" / "dir" / "seed_checkpoint.pt"
        h = generate_seed_checkpoint(small_arch, out, vocab_size=256)
        assert out.exists()
        assert len(h) == 64
