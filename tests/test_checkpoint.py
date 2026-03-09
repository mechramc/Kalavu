"""Tests for kalavu.core.checkpoint — checkpoint format handler."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kalavu.core.checkpoint import (
    AlignmentReport,
    CheckpointMetadata,
    LoadedCheckpoint,
    compute_artifact_hash,
    load_checkpoint,
    save_checkpoint,
    validate_artifact_hashes,
)
from kalavu.core.exceptions import CheckpointValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_model_state() -> dict:
    """Minimal model state_dict for testing."""
    return {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}


@pytest.fixture()
def sample_probe_state() -> dict:
    """Minimal probe state_dict for testing."""
    return {"probe.weight": torch.randn(2, 4)}


@pytest.fixture()
def sample_report() -> AlignmentReport:
    return AlignmentReport(
        cka_scores={"6": 0.91, "12": 0.87, "18": 0.85},
        val_bpb=1.23,
        passed=True,
    )


@pytest.fixture()
def sample_metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        hardware="1xA100-80GB",
        throughput_tokens_per_sec=12500.0,
        total_tokens=1_000_000_000,
        total_time_seconds=80000.0,
    )


@pytest.fixture()
def cooperative_dir(tmp_path: Path) -> Path:
    """Create a cooperative directory with fake shared artifacts."""
    coop = tmp_path / "cooperative"
    coop.mkdir()
    (coop / "tokenizer.model").write_bytes(b"tokenizer-content-v1")
    (coop / "seed_checkpoint.pt").write_bytes(b"seed-content-v1")
    return coop


@pytest.fixture()
def artifact_hashes(cooperative_dir: Path) -> dict[str, str]:
    """Compute real hashes from the cooperative fixtures."""
    return {
        "tokenizer.model": compute_artifact_hash(cooperative_dir / "tokenizer.model"),
        "seed_checkpoint.pt": compute_artifact_hash(
            cooperative_dir / "seed_checkpoint.pt"
        ),
    }


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadCheckpoint:
    """Save a checkpoint and reload it, verifying all fields match."""

    def test_round_trip(
        self,
        tmp_path: Path,
        sample_model_state: dict,
        sample_probe_state: dict,
        sample_report: AlignmentReport,
        sample_metadata: CheckpointMetadata,
        artifact_hashes: dict[str, str],
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(
            ckpt_dir,
            sample_model_state,
            sample_probe_state,
            sample_report,
            sample_metadata,
            artifact_hashes,
        )

        loaded = load_checkpoint(ckpt_dir)

        # Model weights match
        for key in sample_model_state:
            assert torch.equal(loaded.model_state_dict[key], sample_model_state[key])

        # Probe weights match
        for key in sample_probe_state:
            assert torch.equal(loaded.probe_state_dict[key], sample_probe_state[key])

        # Alignment report matches
        assert loaded.alignment_report.cka_scores == sample_report.cka_scores
        assert loaded.alignment_report.val_bpb == sample_report.val_bpb
        assert loaded.alignment_report.passed == sample_report.passed

        # Metadata matches
        assert loaded.metadata.hardware == sample_metadata.hardware
        assert (
            loaded.metadata.throughput_tokens_per_sec
            == sample_metadata.throughput_tokens_per_sec
        )
        assert loaded.metadata.total_tokens == sample_metadata.total_tokens
        assert loaded.metadata.total_time_seconds == sample_metadata.total_time_seconds

        # Artifact hashes match
        assert loaded.artifact_hashes == artifact_hashes

    def test_load_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointValidationError, match="does not exist"):
            load_checkpoint(tmp_path / "nonexistent")

    def test_load_missing_file(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "incomplete"
        ckpt.mkdir()
        with pytest.raises(CheckpointValidationError, match="Missing required"):
            load_checkpoint(ckpt)


# ---------------------------------------------------------------------------
# Artifact hash computation
# ---------------------------------------------------------------------------


class TestComputeArtifactHash:
    def test_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello world")
        h1 = compute_artifact_hash(f)
        h2 = compute_artifact_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content-a")
        f2.write_bytes(b"content-b")
        assert compute_artifact_hash(f1) != compute_artifact_hash(f2)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointValidationError, match="not found"):
            compute_artifact_hash(tmp_path / "nope.bin")


# ---------------------------------------------------------------------------
# Artifact hash validation
# ---------------------------------------------------------------------------


class TestValidateArtifactHashes:
    def test_valid_hashes_pass(
        self,
        tmp_path: Path,
        sample_model_state: dict,
        sample_probe_state: dict,
        sample_report: AlignmentReport,
        sample_metadata: CheckpointMetadata,
        artifact_hashes: dict[str, str],
        cooperative_dir: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(
            ckpt_dir,
            sample_model_state,
            sample_probe_state,
            sample_report,
            sample_metadata,
            artifact_hashes,
        )
        # Should not raise
        validate_artifact_hashes(ckpt_dir, cooperative_dir)

    def test_tampered_file_raises(
        self,
        tmp_path: Path,
        sample_model_state: dict,
        sample_probe_state: dict,
        sample_report: AlignmentReport,
        sample_metadata: CheckpointMetadata,
        artifact_hashes: dict[str, str],
        cooperative_dir: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(
            ckpt_dir,
            sample_model_state,
            sample_probe_state,
            sample_report,
            sample_metadata,
            artifact_hashes,
        )

        # Tamper with the tokenizer file
        (cooperative_dir / "tokenizer.model").write_bytes(b"tampered-content")

        with pytest.raises(CheckpointValidationError, match="hash mismatch"):
            validate_artifact_hashes(ckpt_dir, cooperative_dir)

    def test_missing_artifact_raises(
        self,
        tmp_path: Path,
        sample_model_state: dict,
        sample_probe_state: dict,
        sample_report: AlignmentReport,
        sample_metadata: CheckpointMetadata,
        cooperative_dir: Path,
    ) -> None:
        # Record a hash for a file that won't exist
        hashes = {"nonexistent_file.bin": "abc123"}
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(
            ckpt_dir,
            sample_model_state,
            sample_probe_state,
            sample_report,
            sample_metadata,
            hashes,
        )

        with pytest.raises(CheckpointValidationError, match="not found"):
            validate_artifact_hashes(ckpt_dir, cooperative_dir)
