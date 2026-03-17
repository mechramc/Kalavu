"""Tests for CKA reference representation computation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kalavai.coop.reference import compute_cka_reference, load_cka_reference
from kalavai.coop.seed import generate_seed_checkpoint
from kalavai.core.config import ArchitectureConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_arch() -> ArchitectureConfig:
    """A tiny architecture config for fast tests."""
    return ArchitectureConfig(
        depth=4,
        d_model=64,
        n_heads=4,
        ffn_ratio=2.0,
        norm="rmsnorm",
    )


VOCAB_SIZE = 256
PROBE_LAYERS = [1, 3]
BATCH_SIZE = 4
SEQ_LEN = 16


@pytest.fixture()
def seed_checkpoint(tmp_path: Path, tiny_arch: ArchitectureConfig) -> Path:
    """Generate a seed checkpoint and return its path."""
    path = tmp_path / "seed_checkpoint.pt"
    generate_seed_checkpoint(tiny_arch, path, vocab_size=VOCAB_SIZE, seed=42)
    return path


@pytest.fixture()
def calibration_batch(tmp_path: Path) -> Path:
    """Save a calibration batch tensor and return its path."""
    path = tmp_path / "calibration_batch.pt"
    torch.manual_seed(0)
    batch = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    torch.save(batch, path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeCkaReference:
    """Tests for compute_cka_reference."""

    def test_file_created(
        self,
        tmp_path: Path,
        tiny_arch: ArchitectureConfig,
        seed_checkpoint: Path,
        calibration_batch: Path,
    ) -> None:
        """Reference file is created and non-empty."""
        out = tmp_path / "cka_reference.pt"
        compute_cka_reference(
            seed_checkpoint_path=seed_checkpoint,
            calibration_batch_path=calibration_batch,
            arch_config=tiny_arch,
            probe_layers=PROBE_LAYERS,
            output_path=out,
            vocab_size=VOCAB_SIZE,
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_loadable_with_correct_keys(
        self,
        tmp_path: Path,
        tiny_arch: ArchitectureConfig,
        seed_checkpoint: Path,
        calibration_batch: Path,
    ) -> None:
        """Loaded reference has keys matching the requested probe layers."""
        out = tmp_path / "cka_reference.pt"
        compute_cka_reference(
            seed_checkpoint_path=seed_checkpoint,
            calibration_batch_path=calibration_batch,
            arch_config=tiny_arch,
            probe_layers=PROBE_LAYERS,
            output_path=out,
            vocab_size=VOCAB_SIZE,
        )
        ref = load_cka_reference(out)
        assert isinstance(ref, dict)
        assert set(ref.keys()) == set(PROBE_LAYERS)

    def test_tensor_shapes(
        self,
        tmp_path: Path,
        tiny_arch: ArchitectureConfig,
        seed_checkpoint: Path,
        calibration_batch: Path,
    ) -> None:
        """Each reference tensor has shape (batch, seq_len, d_model)."""
        out = tmp_path / "cka_reference.pt"
        compute_cka_reference(
            seed_checkpoint_path=seed_checkpoint,
            calibration_batch_path=calibration_batch,
            arch_config=tiny_arch,
            probe_layers=PROBE_LAYERS,
            output_path=out,
            vocab_size=VOCAB_SIZE,
        )
        ref = load_cka_reference(out)
        for layer_idx, tensor in ref.items():
            assert tensor.shape == (BATCH_SIZE, SEQ_LEN, tiny_arch.d_model), (
                f"Layer {layer_idx}: expected "
                f"({BATCH_SIZE}, {SEQ_LEN}, {tiny_arch.d_model}), "
                f"got {tensor.shape}"
            )

    def test_deterministic(
        self,
        tmp_path: Path,
        tiny_arch: ArchitectureConfig,
        seed_checkpoint: Path,
        calibration_batch: Path,
    ) -> None:
        """Same inputs produce identical reference tensors."""
        out1 = tmp_path / "ref1.pt"
        out2 = tmp_path / "ref2.pt"

        for out in (out1, out2):
            compute_cka_reference(
                seed_checkpoint_path=seed_checkpoint,
                calibration_batch_path=calibration_batch,
                arch_config=tiny_arch,
                probe_layers=PROBE_LAYERS,
                output_path=out,
                vocab_size=VOCAB_SIZE,
            )

        ref1 = load_cka_reference(out1)
        ref2 = load_cka_reference(out2)

        for layer_idx in PROBE_LAYERS:
            assert torch.equal(ref1[layer_idx], ref2[layer_idx]), (
                f"Layer {layer_idx} tensors differ between runs"
            )

    def test_creates_parent_directories(
        self,
        tmp_path: Path,
        tiny_arch: ArchitectureConfig,
        seed_checkpoint: Path,
        calibration_batch: Path,
    ) -> None:
        """Output path with non-existent parent dirs is created automatically."""
        out = tmp_path / "nested" / "dir" / "cka_reference.pt"
        compute_cka_reference(
            seed_checkpoint_path=seed_checkpoint,
            calibration_batch_path=calibration_batch,
            arch_config=tiny_arch,
            probe_layers=PROBE_LAYERS,
            output_path=out,
            vocab_size=VOCAB_SIZE,
        )
        assert out.exists()
