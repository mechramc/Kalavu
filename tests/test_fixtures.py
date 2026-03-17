"""Tests that verify the shared conftest fixtures produce valid outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from kalavai.core.config import CooperativeConfig


class TestSampleConfigDict:
    """Verify ``sample_config_dict`` fixture."""

    def test_has_cooperative_key(self, sample_config_dict: dict[str, Any]) -> None:
        assert "cooperative" in sample_config_dict

    def test_parses_to_config(self, sample_config_dict: dict[str, Any]) -> None:
        cfg = CooperativeConfig.from_dict(sample_config_dict)
        assert cfg.name == "test-coop"
        assert cfg.modules == 4

    def test_architecture_present(self, sample_config_dict: dict[str, Any]) -> None:
        cfg = CooperativeConfig.from_dict(sample_config_dict)
        assert cfg.architecture.depth == 12
        assert cfg.architecture.d_model == 768

    def test_alignment_present(self, sample_config_dict: dict[str, Any]) -> None:
        cfg = CooperativeConfig.from_dict(sample_config_dict)
        assert cfg.alignment.probe_layers == [3, 6, 9]

    def test_domains_present(self, sample_config_dict: dict[str, Any]) -> None:
        cfg = CooperativeConfig.from_dict(sample_config_dict)
        assert len(cfg.domains) == 2
        assert cfg.domains[0].name == "Code"


class TestSampleConfigYaml:
    """Verify ``sample_config_yaml`` fixture."""

    def test_file_exists(self, sample_config_yaml: Path) -> None:
        assert sample_config_yaml.exists()
        assert sample_config_yaml.name == "kalavai.yaml"

    def test_loads_as_config(self, sample_config_yaml: Path) -> None:
        cfg = CooperativeConfig.from_yaml(sample_config_yaml)
        assert cfg.name == "test-coop"
        assert cfg.modules == 4
        assert cfg.architecture.depth == 12


class TestCooperativeDir:
    """Verify ``cooperative_dir`` fixture creates all required files."""

    EXPECTED_FILES = [
        "kalavai.yaml",
        "tokenizer.model",
        "seed_checkpoint.pt",
        "calibration_batch.pt",
        "cka_reference.pt",
        "domain_manifest.json",
    ]

    def test_directory_exists(self, cooperative_dir: Path) -> None:
        assert cooperative_dir.is_dir()

    def test_all_files_present(self, cooperative_dir: Path) -> None:
        for name in self.EXPECTED_FILES:
            assert (cooperative_dir / name).exists(), f"Missing: {name}"

    def test_config_loads(self, cooperative_dir: Path) -> None:
        cfg = CooperativeConfig.from_yaml(cooperative_dir / "kalavai.yaml")
        assert cfg.name == "test-coop"

    def test_seed_checkpoint_loadable(self, cooperative_dir: Path) -> None:
        state = torch.load(
            cooperative_dir / "seed_checkpoint.pt", weights_only=True
        )
        assert isinstance(state, dict)
        assert "model.embed.weight" in state
        assert isinstance(state["model.embed.weight"], torch.Tensor)

    def test_calibration_batch_loadable(self, cooperative_dir: Path) -> None:
        batch = torch.load(
            cooperative_dir / "calibration_batch.pt", weights_only=True
        )
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert batch["input_ids"].shape == (4, 32)

    def test_cka_reference_loadable(self, cooperative_dir: Path) -> None:
        ref = torch.load(
            cooperative_dir / "cka_reference.pt", weights_only=True
        )
        assert isinstance(ref, dict)
        assert "layer_3" in ref
        assert "layer_6" in ref
        assert "layer_9" in ref

    def test_domain_manifest_loadable(self, cooperative_dir: Path) -> None:
        data = json.loads(
            (cooperative_dir / "domain_manifest.json").read_text(encoding="utf-8")
        )
        assert data["cooperative"] == "test-coop"
        assert len(data["domains"]) == 2

    def test_tokenizer_file_non_empty(self, cooperative_dir: Path) -> None:
        content = (cooperative_dir / "tokenizer.model").read_bytes()
        assert len(content) > 0


class TestMockNoCuda:
    """Verify ``mock_no_cuda`` fixture patches CUDA availability."""

    def test_cuda_is_disabled(self, mock_no_cuda: None) -> None:
        assert torch.cuda.is_available() is False

    def test_cuda_normally_returns_bool(self) -> None:
        # Without the fixture, is_available returns a bool (True or False
        # depending on the machine). We just verify it's callable.
        result = torch.cuda.is_available()
        assert isinstance(result, bool)
