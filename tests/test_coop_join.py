"""Tests for cooperative join (kalavu coop join)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalavu.coop.create import create_cooperative
from kalavu.coop.join import join_cooperative
from kalavu.coop.manifest import load_manifest
from kalavu.core.exceptions import CooperativeError


@pytest.fixture()
def cooperative(tmp_path: Path) -> Path:
    """Create a small cooperative for testing."""
    return create_cooperative(
        name="test-coop",
        modules=5,
        target_params="14M",
        output_dir=tmp_path / "test-coop",
        vocab_size=300,
        seed=42,
    )


class TestJoinCooperative:
    """Tests for join_cooperative."""

    def test_join_open_slot(self, cooperative: Path, tmp_path: Path) -> None:
        """Joining an open slot copies files and updates manifest."""
        work = tmp_path / "work" / "module-1"
        result = join_cooperative(
            cooperative_dir=cooperative,
            module_id=1,
            contributor_name="alice",
            work_dir=work,
        )

        assert result == work.resolve()
        # All shared artifacts should be copied
        for fname in [
            "kalavu.yaml",
            "tokenizer.model",
            "seed_checkpoint.pt",
            "calibration_batch.pt",
            "cka_reference.pt",
        ]:
            assert (work / fname).exists(), f"Missing copied file: {fname}"

        # Manifest should be updated
        slots = load_manifest(cooperative / "domain_manifest.json")
        slot = next(s for s in slots if s["id"] == 1)
        assert slot["status"] == "claimed"
        assert slot["contributor"] == "alice"

    def test_join_already_claimed_slot(
        self, cooperative: Path, tmp_path: Path
    ) -> None:
        """Joining a slot that is already claimed raises CooperativeError."""
        join_cooperative(
            cooperative_dir=cooperative,
            module_id=1,
            contributor_name="alice",
            work_dir=tmp_path / "work1",
        )

        with pytest.raises(CooperativeError, match="already claimed"):
            join_cooperative(
                cooperative_dir=cooperative,
                module_id=1,
                contributor_name="bob",
                work_dir=tmp_path / "work2",
            )

    def test_join_nonexistent_slot(
        self, cooperative: Path, tmp_path: Path
    ) -> None:
        """Joining a slot that does not exist raises CooperativeError."""
        with pytest.raises(CooperativeError, match="not found"):
            join_cooperative(
                cooperative_dir=cooperative,
                module_id=999,
                contributor_name="alice",
                work_dir=tmp_path / "work",
            )

    def test_join_invalid_cooperative_dir(self, tmp_path: Path) -> None:
        """Joining with a nonexistent cooperative directory raises CooperativeError."""
        with pytest.raises(CooperativeError, match="does not exist"):
            join_cooperative(
                cooperative_dir=tmp_path / "nonexistent",
                module_id=1,
                contributor_name="alice",
            )

    def test_join_incomplete_cooperative_dir(self, tmp_path: Path) -> None:
        """Joining a directory missing required files raises CooperativeError."""
        incomplete = tmp_path / "incomplete-coop"
        incomplete.mkdir()
        (incomplete / "kalavu.yaml").write_text("placeholder", encoding="utf-8")

        with pytest.raises(CooperativeError, match="missing required files"):
            join_cooperative(
                cooperative_dir=incomplete,
                module_id=1,
                contributor_name="alice",
            )

    def test_artifact_hashes_stored(
        self, cooperative: Path, tmp_path: Path
    ) -> None:
        """Artifact hashes are computed and stored in the work directory."""
        work = tmp_path / "work" / "module-2"
        join_cooperative(
            cooperative_dir=cooperative,
            module_id=2,
            contributor_name="bob",
            work_dir=work,
        )

        hashes_file = work / "artifact_hashes.json"
        assert hashes_file.exists()
        hashes = json.loads(hashes_file.read_text(encoding="utf-8"))

        # Should have hashes for all shared artifacts
        expected_keys = {
            "kalavu.yaml",
            "tokenizer.model",
            "seed_checkpoint.pt",
            "calibration_batch.pt",
            "cka_reference.pt",
        }
        assert set(hashes.keys()) == expected_keys

        # Each hash should be a 64-char hex string (SHA-256)
        for name, h in hashes.items():
            assert len(h) == 64, f"Hash for {name} has wrong length: {len(h)}"
            assert all(c in "0123456789abcdef" for c in h)

    def test_default_work_dir(
        self, cooperative: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no work_dir is given, defaults to ./module-{id}/."""
        monkeypatch.chdir(tmp_path)
        result = join_cooperative(
            cooperative_dir=cooperative,
            module_id=3,
            contributor_name="carol",
        )

        expected = (tmp_path / "module-3").resolve()
        assert result == expected
        assert expected.is_dir()
