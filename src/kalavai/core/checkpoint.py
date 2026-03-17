"""Checkpoint format handler for Kalavai module submissions.

Implements save/load for the checkpoint directory structure defined in
spec section 6.2, including artifact hash validation against the
cooperative's shared seed and tokenizer files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from kalavai.core.exceptions import CheckpointValidationError

# Filenames inside a checkpoint directory
_MODEL_FILE = "model.pt"
_PROBES_FILE = "probes.pt"
_ALIGNMENT_REPORT_FILE = "alignment_report.json"
_METADATA_FILE = "metadata.json"
_ARTIFACT_HASHES_FILE = "artifact_hashes.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckpointMetadata:
    """Training metadata recorded at checkpoint time.

    Args:
        hardware: Description of training hardware (e.g. "1xA100-80GB").
        throughput_tokens_per_sec: Measured training throughput.
        total_tokens: Total tokens processed during training.
        total_time_seconds: Wall-clock training time in seconds.
    """

    hardware: str
    throughput_tokens_per_sec: float
    total_tokens: int
    total_time_seconds: float


@dataclass
class AlignmentReport:
    """Final CKA alignment scores from probe evaluation.

    Args:
        cka_scores: Mapping of layer index (as string key) to CKA score.
        val_bpb: Validation bits-per-byte.
        passed: Whether the alignment check passed cooperative thresholds.
    """

    cka_scores: dict[str, float]
    val_bpb: float
    passed: bool


@dataclass
class LoadedCheckpoint:
    """All data loaded from a checkpoint directory.

    Args:
        model_state_dict: Model weights (PyTorch state_dict).
        probe_state_dict: Alignment probe weights.
        alignment_report: CKA alignment report.
        metadata: Training metadata.
        artifact_hashes: Recorded hashes of shared artifacts.
    """

    model_state_dict: dict[str, Any]
    probe_state_dict: dict[str, Any]
    alignment_report: AlignmentReport
    metadata: CheckpointMetadata
    artifact_hashes: dict[str, str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model_state_dict: dict[str, Any],
    probe_state_dict: dict[str, Any],
    alignment_report: AlignmentReport,
    metadata: CheckpointMetadata,
    artifact_hashes: dict[str, str],
) -> Path:
    """Save a module checkpoint to a directory.

    Creates the directory if it does not exist and writes model weights,
    probe weights, alignment report, metadata, and artifact hashes.

    Args:
        path: Destination directory for the checkpoint.
        model_state_dict: PyTorch model state_dict.
        probe_state_dict: PyTorch probe state_dict.
        alignment_report: Alignment CKA report.
        metadata: Training metadata.
        artifact_hashes: SHA-256 hashes of shared artifacts.

    Returns:
        The resolved checkpoint directory path.

    Raises:
        CheckpointValidationError: If the checkpoint cannot be saved.
    """
    ckpt_dir = Path(path)
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointValidationError(
            f"Cannot create checkpoint directory: {exc}"
        ) from exc

    try:
        torch.save(model_state_dict, ckpt_dir / _MODEL_FILE)
        torch.save(probe_state_dict, ckpt_dir / _PROBES_FILE)

        _write_json(ckpt_dir / _ALIGNMENT_REPORT_FILE, asdict(alignment_report))
        _write_json(ckpt_dir / _METADATA_FILE, asdict(metadata))
        _write_json(ckpt_dir / _ARTIFACT_HASHES_FILE, artifact_hashes)
    except Exception as exc:
        raise CheckpointValidationError(
            f"Failed to write checkpoint files: {exc}"
        ) from exc

    return ckpt_dir


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load a module checkpoint from a directory.

    Args:
        path: Path to the checkpoint directory.

    Returns:
        A LoadedCheckpoint containing all saved data.

    Raises:
        CheckpointValidationError: If the directory is missing or any
            required file cannot be read.
    """
    ckpt_dir = Path(path)
    if not ckpt_dir.is_dir():
        raise CheckpointValidationError(
            f"Checkpoint directory does not exist: {ckpt_dir}"
        )

    required = [
        _MODEL_FILE,
        _PROBES_FILE,
        _ALIGNMENT_REPORT_FILE,
        _METADATA_FILE,
        _ARTIFACT_HASHES_FILE,
    ]
    for fname in required:
        if not (ckpt_dir / fname).exists():
            raise CheckpointValidationError(
                f"Missing required checkpoint file: {fname}"
            )

    try:
        model_state_dict = torch.load(
            ckpt_dir / _MODEL_FILE, map_location="cpu", weights_only=True
        )
        probe_state_dict = torch.load(
            ckpt_dir / _PROBES_FILE, map_location="cpu", weights_only=True
        )

        report_raw = _read_json(ckpt_dir / _ALIGNMENT_REPORT_FILE)
        alignment_report = AlignmentReport(
            cka_scores=report_raw["cka_scores"],
            val_bpb=report_raw["val_bpb"],
            passed=report_raw["passed"],
        )

        meta_raw = _read_json(ckpt_dir / _METADATA_FILE)
        metadata = CheckpointMetadata(
            hardware=meta_raw["hardware"],
            throughput_tokens_per_sec=meta_raw["throughput_tokens_per_sec"],
            total_tokens=meta_raw["total_tokens"],
            total_time_seconds=meta_raw["total_time_seconds"],
        )

        artifact_hashes = _read_json(ckpt_dir / _ARTIFACT_HASHES_FILE)
    except CheckpointValidationError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(
            f"Failed to load checkpoint: {exc}"
        ) from exc

    return LoadedCheckpoint(
        model_state_dict=model_state_dict,
        probe_state_dict=probe_state_dict,
        alignment_report=alignment_report,
        metadata=metadata,
        artifact_hashes=artifact_hashes,
    )


def compute_artifact_hash(file_path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest.

    Raises:
        CheckpointValidationError: If the file cannot be read.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise CheckpointValidationError(f"Artifact file not found: {file_path}")

    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
    except OSError as exc:
        raise CheckpointValidationError(
            f"Cannot read artifact file {file_path}: {exc}"
        ) from exc

    return sha.hexdigest()


def validate_artifact_hashes(
    checkpoint_path: str | Path,
    cooperative_dir: str | Path,
) -> None:
    """Validate a checkpoint's artifact hashes against actual files.

    Compares the SHA-256 hashes recorded in the checkpoint's
    ``artifact_hashes.json`` against the corresponding files in the
    cooperative directory (e.g. ``tokenizer.model``, ``seed_checkpoint.pt``).

    Args:
        checkpoint_path: Path to the checkpoint directory.
        cooperative_dir: Path to the cooperative's shared artifacts directory.

    Raises:
        CheckpointValidationError: If any recorded hash does not match the
            actual file, or if a referenced artifact file is missing.
    """
    ckpt_dir = Path(checkpoint_path)
    coop_dir = Path(cooperative_dir)

    hashes_file = ckpt_dir / _ARTIFACT_HASHES_FILE
    if not hashes_file.exists():
        raise CheckpointValidationError(
            f"Missing artifact hashes file in checkpoint: {hashes_file}"
        )

    recorded = _read_json(hashes_file)
    if not isinstance(recorded, dict):
        raise CheckpointValidationError("artifact_hashes.json must be a JSON object")

    mismatches: list[str] = []
    for artifact_name, expected_hash in recorded.items():
        artifact_path = coop_dir / artifact_name
        if not artifact_path.exists():
            raise CheckpointValidationError(
                f"Referenced artifact not found: {artifact_path}"
            )

        actual_hash = compute_artifact_hash(artifact_path)
        if actual_hash != expected_hash:
            mismatches.append(
                f"{artifact_name}: expected {expected_hash}, got {actual_hash}"
            )

    if mismatches:
        detail = "; ".join(mismatches)
        raise CheckpointValidationError(f"Artifact hash mismatch: {detail}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    """Write data as pretty-printed JSON."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"Cannot read JSON file {path}: {exc}"
        ) from exc
