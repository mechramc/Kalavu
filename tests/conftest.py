"""Shared pytest fixtures for the Kalavai test suite.

Provides reusable fixtures for cooperative directories, sample configs,
mock checkpoints, and GPU-free test environments.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

SAMPLE_YAML_TEXT = textwrap.dedent("""\
    cooperative:
      name: test-coop
      modules: 4
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
      domains:
        - {id: 1, name: Code, data_hint: 'github-code'}
        - {id: 2, name: Mathematics, data_hint: 'openwebmath'}
""")


@pytest.fixture()
def sample_config_dict() -> dict[str, Any]:
    """Return a valid cooperative config as a plain dictionary."""
    return {
        "cooperative": {
            "name": "test-coop",
            "modules": 4,
            "target_params_per_module": "125M",
            "architecture": {
                "depth": 12,
                "d_model": 768,
                "n_heads": 12,
                "ffn_ratio": 2.75,
                "norm": "rmsnorm",
            },
            "alignment": {
                "lambda_max": 0.05,
                "lambda_min": 0.01,
                "anneal_start": 0.7,
                "probe_layers": [3, 6, 9],
                "calibration_interval": 500,
                "thresholds": {"layer_3": 0.7, "layer_6": 0.6, "layer_9": 0.5},
            },
            "fusion": {
                "backend": "moe_routing",
                "n_clusters": 2,
            },
            "domains": [
                {"id": 1, "name": "Code", "data_hint": "github-code"},
                {"id": 2, "name": "Mathematics", "data_hint": "openwebmath"},
            ],
        }
    }


@pytest.fixture()
def sample_config_yaml(tmp_path: Path) -> Path:
    """Write a valid kalavai.yaml to *tmp_path* and return the file path."""
    p = tmp_path / "kalavai.yaml"
    p.write_text(SAMPLE_YAML_TEXT, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Cooperative directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def cooperative_dir(tmp_path: Path) -> Path:
    """Create a full cooperative directory with all required files.

    Contents:
        kalavai.yaml          — valid cooperative config
        tokenizer.model      — dummy bytes (not a real sentencepiece model)
        seed_checkpoint.pt   — small random state_dict
        calibration_batch.pt — small random tensor batch
        cka_reference.pt     — small random CKA reference activations
        domain_manifest.json — JSON mapping of domain assignments

    Returns:
        Path to the cooperative directory.
    """
    coop = tmp_path / "test-coop"
    coop.mkdir()

    # kalavai.yaml
    (coop / "kalavai.yaml").write_text(SAMPLE_YAML_TEXT, encoding="utf-8")

    # Dummy tokenizer file (just needs to exist for path validation)
    (coop / "tokenizer.model").write_bytes(b"DUMMY_TOKENIZER_DATA")

    # Small seed checkpoint — a tiny state_dict
    seed_state = {
        "model.embed.weight": torch.randn(32, 16),
        "model.layer0.attn.weight": torch.randn(16, 16),
        "model.layer0.ffn.weight": torch.randn(16, 44),
    }
    torch.save(seed_state, coop / "seed_checkpoint.pt")

    # Calibration batch — a small input tensor batch
    calibration = {
        "input_ids": torch.randint(0, 100, (4, 32)),
        "attention_mask": torch.ones(4, 32, dtype=torch.long),
    }
    torch.save(calibration, coop / "calibration_batch.pt")

    # CKA reference activations — one tensor per probe layer
    cka_ref = {
        "layer_3": torch.randn(4, 16),
        "layer_6": torch.randn(4, 16),
        "layer_9": torch.randn(4, 16),
    }
    torch.save(cka_ref, coop / "cka_reference.pt")

    # Domain manifest
    manifest = {
        "cooperative": "test-coop",
        "domains": [
            {"id": 1, "name": "Code", "data_hint": "github-code"},
            {"id": 2, "name": "Mathematics", "data_hint": "openwebmath"},
        ],
    }
    (coop / "domain_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return coop


# ---------------------------------------------------------------------------
# CUDA mock fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_no_cuda():
    """Patch ``torch.cuda.is_available`` to return False for the test scope.

    Ensures tests run without requiring a GPU.
    """
    with patch("torch.cuda.is_available", return_value=False):
        yield
