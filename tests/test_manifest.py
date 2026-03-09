"""Tests for kalavu.coop.manifest — domain manifest generation and management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalavu.coop.manifest import (
    generate_manifest,
    load_manifest,
    update_slot,
)
from kalavu.core.config import CooperativeConfig, DomainConfig
from kalavu.core.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    modules: int = 20,
    domains: list[DomainConfig] | None = None,
) -> CooperativeConfig:
    """Build a minimal CooperativeConfig for manifest tests."""
    return CooperativeConfig.from_dict(
        {
            "cooperative": {
                "name": "test-coop",
                "modules": modules,
                "target_params_per_module": "125M",
                "architecture": {
                    "depth": 12,
                    "d_model": 768,
                    "n_heads": 12,
                    "ffn_ratio": 2.75,
                },
                "alignment": {
                    "lambda_max": 0.05,
                    "lambda_min": 0.01,
                    "anneal_start": 0.7,
                    "probe_layers": [3, 6, 9],
                },
                "domains": (
                    [{"id": d.id, "name": d.name, "data_hint": d.data_hint} for d in domains]
                    if domains
                    else []
                ),
            }
        }
    )


# ---------------------------------------------------------------------------
# Tests: generate_manifest
# ---------------------------------------------------------------------------


class TestGenerateManifestCustomDomains:
    """Generate manifest with custom domains from config."""

    def test_custom_domains_written(self, tmp_path: Path) -> None:
        custom = [
            DomainConfig(id=1, name="Code", data_hint="github-code"),
            DomainConfig(id=2, name="Math", data_hint="openwebmath"),
        ]
        config = _make_config(modules=2, domains=custom)
        out = tmp_path / "domain_manifest.json"

        generate_manifest(config, out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["cooperative"] == "test-coop"
        assert len(data["slots"]) == 2
        assert data["slots"][0]["name"] == "Code"
        assert data["slots"][1]["data_hint"] == "openwebmath"

    def test_all_slots_open_with_null_contributor(self, tmp_path: Path) -> None:
        custom = [DomainConfig(id=1, name="Bio", data_hint="pubmed")]
        config = _make_config(modules=1, domains=custom)
        out = tmp_path / "domain_manifest.json"

        generate_manifest(config, out)

        slots = json.loads(out.read_text(encoding="utf-8"))["slots"]
        for slot in slots:
            assert slot["status"] == "open"
            assert slot["contributor"] is None


class TestGenerateManifestDefaults:
    """Generate manifest with no domains — uses defaults."""

    def test_defaults_used_when_no_domains(self, tmp_path: Path) -> None:
        config = _make_config(modules=5, domains=None)
        out = tmp_path / "domain_manifest.json"

        generate_manifest(config, out)

        slots = json.loads(out.read_text(encoding="utf-8"))["slots"]
        assert len(slots) == 5
        assert slots[0]["name"] == "Code"
        assert slots[4]["name"] == "History"

    def test_defaults_count_matches_modules(self, tmp_path: Path) -> None:
        config = _make_config(modules=20, domains=None)
        out = tmp_path / "domain_manifest.json"

        generate_manifest(config, out)

        slots = json.loads(out.read_text(encoding="utf-8"))["slots"]
        assert len(slots) == 20

    def test_more_modules_than_defaults_generates_placeholders(
        self, tmp_path: Path
    ) -> None:
        config = _make_config(modules=22, domains=None)
        out = tmp_path / "domain_manifest.json"

        generate_manifest(config, out)

        slots = json.loads(out.read_text(encoding="utf-8"))["slots"]
        assert len(slots) == 22
        assert slots[20]["name"] == "Domain-21"
        assert slots[21]["name"] == "Domain-22"


# ---------------------------------------------------------------------------
# Tests: load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    """Round-trip: generate then load."""

    def test_load_round_trip(self, tmp_path: Path) -> None:
        config = _make_config(modules=3, domains=None)
        out = tmp_path / "domain_manifest.json"

        generate_manifest(config, out)
        slots = load_manifest(out)

        assert len(slots) == 3
        assert slots[0]["id"] == 1
        assert slots[2]["status"] == "open"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_manifest(tmp_path / "nope.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            load_manifest(bad)


# ---------------------------------------------------------------------------
# Tests: update_slot
# ---------------------------------------------------------------------------


class TestUpdateSlot:
    """Update individual slots in the manifest."""

    def test_update_slot_changes_correct_entry(self, tmp_path: Path) -> None:
        config = _make_config(modules=3, domains=None)
        out = tmp_path / "domain_manifest.json"
        generate_manifest(config, out)

        update_slot(out, slot_id=2, updates={"status": "claimed", "contributor": "alice"})

        slots = load_manifest(out)
        slot2 = next(s for s in slots if s["id"] == 2)
        assert slot2["status"] == "claimed"
        assert slot2["contributor"] == "alice"
        # Other slots unchanged
        slot1 = next(s for s in slots if s["id"] == 1)
        assert slot1["status"] == "open"

    def test_update_nonexistent_slot_raises(self, tmp_path: Path) -> None:
        config = _make_config(modules=2, domains=None)
        out = tmp_path / "domain_manifest.json"
        generate_manifest(config, out)

        with pytest.raises(ConfigError, match="not found"):
            update_slot(out, slot_id=999, updates={"status": "claimed"})
