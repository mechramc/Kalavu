"""Tests for cooperative status display (kalavai coop status)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from kalavai.cli import main
from kalavai.coop.status import get_cooperative_status, print_cooperative_status
from kalavai.core.exceptions import ConfigError

SAMPLE_YAML_TEXT = textwrap.dedent("""\
    cooperative:
      name: test-coop
      modules: 3
      target_params_per_module: 125M
      architecture:
        depth: 12
        d_model: 768
        n_heads: 12
        ffn_ratio: 2.75
        norm: rmsnorm
      alignment:
        lambda_max: 0.05
        lambda_min: 0.01
        anneal_start: 0.7
        probe_layers: [3, 6, 9]
        calibration_interval: 500
        thresholds: {layer_3: 0.7, layer_6: 0.6, layer_9: 0.5}
      fusion:
        backend: moe_routing
        n_clusters: 2
""")


@pytest.fixture()
def status_coop(tmp_path: Path) -> Path:
    """Create a minimal cooperative directory suitable for status tests."""
    coop = tmp_path / "test-coop"
    coop.mkdir()

    (coop / "kalavai.yaml").write_text(SAMPLE_YAML_TEXT, encoding="utf-8")

    manifest = {
        "cooperative": "test-coop",
        "slots": [
            {"id": 1, "name": "Code", "data_hint": "github-code", "status": "open", "contributor": None},
            {"id": 2, "name": "Mathematics", "data_hint": "openwebmath", "status": "open", "contributor": None},
            {"id": 3, "name": "Biology", "data_hint": "pubmed-bio", "status": "open", "contributor": None},
        ],
    }
    (coop / "domain_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return coop


class TestGetCooperativeStatus:
    """Unit tests for get_cooperative_status."""

    def test_fresh_cooperative_all_open(self, status_coop: Path) -> None:
        """A fresh cooperative has all modules in 'open' status."""
        result = get_cooperative_status(status_coop)

        assert result["cooperative_name"] == "test-coop"
        assert result["total_modules"] == 3
        assert len(result["modules"]) == 3
        for mod in result["modules"]:
            assert mod["status"] == "open"
            assert mod["contributor"] is None
            assert mod["progress"] == 0
        assert result["summary"]["open"] == 3
        assert result["summary"]["claimed"] == 0

    def test_after_claiming_module(self, status_coop: Path) -> None:
        """After claiming a module, status shows 'claimed' with contributor."""
        # Manually update manifest to simulate a claim
        manifest_path = status_coop / "domain_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["slots"][0]["status"] = "claimed"
        data["slots"][0]["contributor"] = "alice"
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = get_cooperative_status(status_coop)

        mod_1 = result["modules"][0]
        assert mod_1["status"] == "claimed"
        assert mod_1["contributor"] == "alice"
        assert mod_1["progress"] == 10
        assert result["summary"]["claimed"] == 1
        assert result["summary"]["open"] == 2

    def test_mixed_statuses(self, status_coop: Path) -> None:
        """Modules with various statuses are counted correctly."""
        manifest_path = status_coop / "domain_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["slots"][0]["status"] = "submitted"
        data["slots"][0]["contributor"] = "alice"
        data["slots"][1]["status"] = "training"
        data["slots"][1]["contributor"] = "bob"
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = get_cooperative_status(status_coop)

        assert result["summary"]["submitted"] == 1
        assert result["summary"]["training"] == 1
        assert result["summary"]["open"] == 1
        assert result["modules"][0]["progress"] == 100
        assert result["modules"][1]["progress"] == 50

    def test_with_alignment_reports(self, status_coop: Path) -> None:
        """CKA scores are included when alignment reports exist."""
        reports_dir = status_coop / "alignment_reports"
        reports_dir.mkdir()
        report = {"cka_scores": {"layer_3": 0.85, "layer_6": 0.72}}
        (reports_dir / "1.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

        result = get_cooperative_status(status_coop)

        mod_1 = result["modules"][0]
        assert "cka_scores" in mod_1
        assert mod_1["cka_scores"]["layer_3"] == pytest.approx(0.85)
        # Module 2 should not have cka_scores
        assert "cka_scores" not in result["modules"][1]

    def test_invalid_cooperative_dir(self, tmp_path: Path) -> None:
        """Raises ConfigError for a non-existent directory."""
        with pytest.raises(ConfigError, match="not found"):
            get_cooperative_status(tmp_path / "nonexistent")


class TestPrintCooperativeStatus:
    """Tests for the rich table display."""

    def test_prints_table(self, status_coop: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """print_cooperative_status produces output without errors."""
        from rich.console import Console

        console = Console(file=None, force_terminal=False, width=120)
        # Just ensure it doesn't raise
        print_cooperative_status(status_coop, console=console)


class TestCliStatus:
    """Integration tests for the CLI command."""

    def test_json_output_valid(self, status_coop: Path) -> None:
        """--json flag produces valid JSON with correct structure."""
        runner = CliRunner()
        result = runner.invoke(main, ["coop", "status", str(status_coop), "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["cooperative_name"] == "test-coop"
        assert data["total_modules"] == 3
        assert isinstance(data["modules"], list)
        assert isinstance(data["summary"], dict)
        assert "open" in data["summary"]

    def test_table_output(self, status_coop: Path) -> None:
        """Default output contains key information."""
        runner = CliRunner()
        result = runner.invoke(main, ["coop", "status", str(status_coop)])

        assert result.exit_code == 0, result.output
        assert "test-coop" in result.output

    def test_invalid_dir_error(self, tmp_path: Path) -> None:
        """Non-existent directory produces an error exit code."""
        runner = CliRunner()
        result = runner.invoke(main, ["coop", "status", str(tmp_path / "nope")])

        assert result.exit_code != 0
