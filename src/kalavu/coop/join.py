"""Join an existing cooperative and claim a domain slot.

Copies shared artifacts to a local working directory, claims a slot in the
domain manifest, and records artifact hashes for later validation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rich.console import Console

from kalavu.coop.manifest import load_manifest, update_slot
from kalavu.core.checkpoint import compute_artifact_hash
from kalavu.core.exceptions import CooperativeError

console = Console()

# Files that must exist in a valid cooperative directory
REQUIRED_FILES = [
    "kalavu.yaml",
    "tokenizer.model",
    "seed_checkpoint.pt",
    "calibration_batch.pt",
    "cka_reference.pt",
    "domain_manifest.json",
]

# Shared artifacts to copy into the work directory
_SHARED_ARTIFACTS = [
    "kalavu.yaml",
    "tokenizer.model",
    "seed_checkpoint.pt",
    "calibration_batch.pt",
    "cka_reference.pt",
]


def join_cooperative(
    cooperative_dir: Path,
    module_id: int,
    contributor_name: str,
    work_dir: Path | None = None,
) -> Path:
    """Join a cooperative by claiming a domain slot and copying shared artifacts.

    Args:
        cooperative_dir: Path to the cooperative directory (created by ``coop create``).
        module_id: The slot ID to claim.
        contributor_name: Name of the contributor claiming the slot.
        work_dir: Local working directory for the module. Defaults to
            ``./module-{module_id}/`` relative to the current directory.

    Returns:
        The resolved working directory path.

    Raises:
        CooperativeError: If the cooperative directory is invalid, the slot
            does not exist, or the slot is already claimed.
    """
    cooperative_dir = Path(cooperative_dir).resolve()

    # ── Validate cooperative directory ────────────────────────────────
    if not cooperative_dir.is_dir():
        raise CooperativeError(
            f"Cooperative directory does not exist: {cooperative_dir}"
        )

    missing = [f for f in REQUIRED_FILES if not (cooperative_dir / f).exists()]
    if missing:
        raise CooperativeError(
            f"Cooperative directory is missing required files: {', '.join(missing)}"
        )

    # ── Load manifest and validate slot ───────────────────────────────
    manifest_path = cooperative_dir / "domain_manifest.json"
    slots = load_manifest(manifest_path)

    slot = _find_slot(slots, module_id)
    if slot is None:
        raise CooperativeError(
            f"Slot with id {module_id} not found in manifest"
        )

    if slot.get("status") != "open":
        raise CooperativeError(
            f"Slot {module_id} is already claimed by {slot.get('contributor', 'unknown')}"
        )

    # ── Resolve work directory ────────────────────────────────────────
    if work_dir is None:
        work_dir = Path.cwd() / f"module-{module_id}"
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy shared artifacts ─────────────────────────────────────────
    console.print("[bold cyan][1/3][/] Copying shared artifacts...")
    for artifact in _SHARED_ARTIFACTS:
        src = cooperative_dir / artifact
        dst = work_dir / artifact
        shutil.copy2(src, dst)

    # ── Compute and store artifact hashes ─────────────────────────────
    console.print("[bold cyan][2/3][/] Computing artifact hashes...")
    hashes: dict[str, str] = {}
    for artifact in _SHARED_ARTIFACTS:
        hashes[artifact] = compute_artifact_hash(work_dir / artifact)

    hashes_path = work_dir / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    # ── Claim slot in manifest ────────────────────────────────────────
    console.print("[bold cyan][3/3][/] Claiming slot in manifest...")
    update_slot(
        manifest_path,
        module_id,
        {"status": "claimed", "contributor": contributor_name},
    )

    console.print(
        f"\n[bold green]Joined cooperative! "
        f"Module {module_id} claimed by {contributor_name}.[/]"
    )
    console.print(f"Working directory: {work_dir}")

    return work_dir


def _find_slot(
    slots: list[dict], slot_id: int
) -> dict | None:
    """Find a slot by ID in the manifest slots list."""
    for slot in slots:
        if slot.get("id") == slot_id:
            return slot
    return None
