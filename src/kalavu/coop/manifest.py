"""Domain manifest generation and management.

Generates domain_manifest.json with N slots, each containing an id, name,
data_hint, and status. Supports custom domains from kalavu.yaml or defaults
from the Kalavu spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kalavu.core.config import CooperativeConfig
from kalavu.core.exceptions import ConfigError

DEFAULT_DOMAINS: list[dict[str, str]] = [
    {"id": 1, "name": "Code", "data_hint": "github-code"},
    {"id": 2, "name": "Mathematics", "data_hint": "openwebmath"},
    {"id": 3, "name": "Biology", "data_hint": "pubmed-bio"},
    {"id": 4, "name": "Legal", "data_hint": "legal-contracts"},
    {"id": 5, "name": "History", "data_hint": "wiki-history"},
    {"id": 6, "name": "Physics", "data_hint": "arxiv-physics"},
    {"id": 7, "name": "NLP", "data_hint": "arxiv-nlp"},
    {"id": 8, "name": "Logic", "data_hint": "logic-puzzles"},
    {"id": 9, "name": "Medical", "data_hint": "pubmed-clinical"},
    {"id": 10, "name": "Finance", "data_hint": "sec-filings"},
    {"id": 11, "name": "Chemistry", "data_hint": "arxiv-chem"},
    {"id": 12, "name": "Writing", "data_hint": "creative-writing"},
    {"id": 13, "name": "Dialogue", "data_hint": "multi-turn-chat"},
    {"id": 14, "name": "Causal Reasoning", "data_hint": "causal-qa"},
    {"id": 15, "name": "Spatial", "data_hint": "spatial-reasoning"},
    {"id": 16, "name": "Ethics", "data_hint": "ethics-benchmarks"},
    {"id": 17, "name": "News", "data_hint": "news-articles"},
    {"id": 18, "name": "Geography", "data_hint": "geo-knowledge"},
    {"id": 19, "name": "Multilingual-Low", "data_hint": "low-resource-langs"},
    {"id": 20, "name": "Multilingual-High", "data_hint": "high-resource-langs"},
]


def generate_manifest(config: CooperativeConfig, output_path: Path) -> None:
    """Generate domain_manifest.json from cooperative config.

    If the config specifies custom domains, those are used directly (with
    ``status`` and ``contributor`` fields added). Otherwise, the default
    domain list is truncated or extended to match ``config.modules``.

    Args:
        config: The cooperative configuration.
        output_path: File path where the manifest JSON will be written.

    Raises:
        ConfigError: If the output file cannot be written.
    """
    if config.domains:
        slots = [
            {
                "id": d.id,
                "name": d.name,
                "data_hint": d.data_hint,
                "status": "open",
                "contributor": None,
            }
            for d in config.domains
        ]
    else:
        n = config.modules
        base = DEFAULT_DOMAINS[:n]
        # If more modules than defaults, generate extra placeholder domains
        for i in range(len(base), n):
            base.append(
                {"id": i + 1, "name": f"Domain-{i + 1}", "data_hint": ""}
            )
        slots = [
            {
                "id": d["id"],
                "name": d["name"],
                "data_hint": d["data_hint"],
                "status": "open",
                "contributor": None,
            }
            for d in base
        ]

    manifest = {"cooperative": config.name, "slots": slots}

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise ConfigError(f"Cannot write manifest file: {exc}") from exc


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a domain manifest and return the list of slots.

    Args:
        path: Path to the domain_manifest.json file.

    Returns:
        List of slot dictionaries.

    Raises:
        ConfigError: If the file is missing, unreadable, or malformed.
    """
    if not path.exists():
        raise ConfigError(f"Manifest file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read manifest file: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in manifest: {exc}") from exc

    if not isinstance(data, dict) or "slots" not in data:
        raise ConfigError("Manifest missing required 'slots' key")

    return data["slots"]


def update_slot(path: Path, slot_id: int, updates: dict[str, Any]) -> None:
    """Update a specific slot in the manifest file.

    Args:
        path: Path to the domain_manifest.json file.
        slot_id: The ``id`` of the slot to update.
        updates: Dictionary of fields to merge into the slot.

    Raises:
        ConfigError: If the manifest cannot be read/written or the slot
            does not exist.
    """
    if not path.exists():
        raise ConfigError(f"Manifest file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read manifest: {exc}") from exc

    if not isinstance(data, dict) or "slots" not in data:
        raise ConfigError("Manifest missing required 'slots' key")

    for slot in data["slots"]:
        if slot.get("id") == slot_id:
            slot.update(updates)
            break
    else:
        raise ConfigError(f"Slot with id {slot_id} not found in manifest")

    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot write manifest: {exc}") from exc
