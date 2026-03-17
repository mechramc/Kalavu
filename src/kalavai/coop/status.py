"""Cooperative status display.

Loads config and manifest to compile and display a rich status table
showing per-module information and a summary line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from kalavai.coop.manifest import load_manifest
from kalavai.core.config import CooperativeConfig
from kalavai.core.exceptions import ConfigError


# Status → colour mapping for rich display
_STATUS_STYLES: dict[str, str] = {
    "open": "dim",
    "claimed": "yellow",
    "training": "blue",
    "submitted": "green",
}


def _load_alignment_reports(cooperative_dir: Path) -> dict[int, dict[str, float]]:
    """Load per-module CKA alignment reports if they exist.

    Looks for ``alignment_reports/<module_id>.json`` files, each expected to
    contain a ``cka_scores`` mapping of layer labels to float scores.

    Returns:
        Mapping from module id to {layer_label: score}.
    """
    reports_dir = cooperative_dir / "alignment_reports"
    results: dict[int, dict[str, float]] = {}
    if not reports_dir.is_dir():
        return results

    for report_file in reports_dir.glob("*.json"):
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            module_id = int(report_file.stem)
            if "cka_scores" in data and isinstance(data["cka_scores"], dict):
                results[module_id] = {
                    str(k): float(v) for k, v in data["cka_scores"].items()
                }
        except (ValueError, json.JSONDecodeError, OSError):
            continue

    return results


def _compute_progress(status: str) -> int:
    """Return a progress percentage based on module status."""
    return {"open": 0, "claimed": 10, "training": 50, "submitted": 100}.get(
        status, 0
    )


def get_cooperative_status(cooperative_dir: Path) -> dict[str, Any]:
    """Load config and manifest, compile cooperative status info.

    Args:
        cooperative_dir: Path to the cooperative directory containing
            ``kalavai.yaml`` and ``domain_manifest.json``.

    Returns:
        Dictionary with keys:
        - ``cooperative_name``: str
        - ``total_modules``: int
        - ``modules``: list of per-module dicts
        - ``summary``: dict of counts by status

    Raises:
        ConfigError: If config or manifest cannot be loaded.
    """
    config_path = cooperative_dir / "kalavai.yaml"
    manifest_path = cooperative_dir / "domain_manifest.json"

    if not cooperative_dir.is_dir():
        raise ConfigError(f"Cooperative directory not found: {cooperative_dir}")

    config = CooperativeConfig.from_yaml(config_path)
    slots = load_manifest(manifest_path)

    # Load CKA alignment reports (optional)
    alignment = _load_alignment_reports(cooperative_dir)

    modules: list[dict[str, Any]] = []
    summary: dict[str, int] = {"open": 0, "claimed": 0, "training": 0, "submitted": 0}

    for slot in slots:
        mid = slot["id"]
        status = slot.get("status", "open")
        summary[status] = summary.get(status, 0) + 1

        module_info: dict[str, Any] = {
            "id": mid,
            "domain": slot.get("name", f"Module-{mid}"),
            "status": status,
            "contributor": slot.get("contributor"),
            "progress": _compute_progress(status),
        }

        if mid in alignment:
            module_info["cka_scores"] = alignment[mid]

        modules.append(module_info)

    return {
        "cooperative_name": config.name,
        "total_modules": len(slots),
        "modules": modules,
        "summary": summary,
    }


def print_cooperative_status(
    cooperative_dir: Path, *, console: Console | None = None
) -> None:
    """Display a rich Table with cooperative status.

    Args:
        cooperative_dir: Path to the cooperative directory.
        console: Optional Rich console (for testing capture).
    """
    if console is None:
        console = Console()

    status = get_cooperative_status(cooperative_dir)

    # Determine if any module has CKA scores
    has_cka = any("cka_scores" in m for m in status["modules"])

    table = Table(
        title=f"Cooperative: {status['cooperative_name']}",
        caption=_summary_line(status["summary"], status["total_modules"]),
    )

    table.add_column("ID", justify="right", style="bold")
    table.add_column("Domain", style="cyan")
    table.add_column("Status")
    table.add_column("Contributor")
    if has_cka:
        table.add_column("CKA", justify="center")
    table.add_column("Progress", justify="right")

    for mod in status["modules"]:
        style = _STATUS_STYLES.get(mod["status"], "")
        status_text = Text(mod["status"], style=style)
        contributor = mod["contributor"] or "-"
        progress = f"{mod['progress']}%"

        row: list[Any] = [
            str(mod["id"]),
            mod["domain"],
            status_text,
            contributor,
        ]
        if has_cka:
            cka = mod.get("cka_scores")
            if cka:
                cka_str = " ".join(f"{k}={v:.2f}" for k, v in cka.items())
            else:
                cka_str = "-"
            row.append(cka_str)
        row.append(progress)

        table.add_row(*row)

    console.print(table)


def _summary_line(summary: dict[str, int], total: int) -> str:
    """Build a summary caption like '3/5 submitted, 1/5 training, 1/5 open'."""
    parts: list[str] = []
    # Show in order: submitted, training, claimed, open
    for key in ("submitted", "training", "claimed", "open"):
        count = summary.get(key, 0)
        if count > 0:
            parts.append(f"{count}/{total} {key}")
    return ", ".join(parts) if parts else "No modules"
