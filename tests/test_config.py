"""Tests for kalavai.core.config — cooperative config schema and parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from kalavai.core.config import (
    AlignmentConfig,
    ArchitectureConfig,
    CooperativeConfig,
    DomainConfig,
    FusionConfig,
)
from kalavai.core.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_YAML = textwrap.dedent("""\
    cooperative:
      name: open-20b
      modules: 20
      target_params_per_module: 1B
      architecture:
        depth: 24
        d_model: 2048
        n_heads: 16
        ffn_ratio: 2.75
        norm: rmsnorm
      alignment:
        lambda_max: 0.05
        lambda_min: 0.01
        anneal_start: 0.7
        probe_layers: [6, 12, 18]
        calibration_interval: 500
        thresholds: {layer_6: 0.7, layer_12: 0.6, layer_18: 0.5}
      fusion:
        backend: moe_routing
        n_clusters: 4
      domains:
        - {id: 1, name: Code, data_hint: 'github-code, stack-v2'}
        - {id: 2, name: Mathematics, data_hint: 'proof-pile, openwebmath'}
""")


@pytest.fixture()
def valid_yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "kalavai.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    return p


@pytest.fixture()
def valid_config(valid_yaml_path: Path) -> CooperativeConfig:
    return CooperativeConfig.from_yaml(valid_yaml_path)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestLoadValid:
    def test_scalar_fields(self, valid_config: CooperativeConfig) -> None:
        assert valid_config.name == "open-20b"
        assert valid_config.modules == 20
        assert valid_config.target_params_per_module == "1B"

    def test_architecture(self, valid_config: CooperativeConfig) -> None:
        arch = valid_config.architecture
        assert isinstance(arch, ArchitectureConfig)
        assert arch.depth == 24
        assert arch.d_model == 2048
        assert arch.n_heads == 16
        assert arch.ffn_ratio == 2.75
        assert arch.norm == "rmsnorm"

    def test_alignment(self, valid_config: CooperativeConfig) -> None:
        a = valid_config.alignment
        assert isinstance(a, AlignmentConfig)
        assert a.lambda_max == 0.05
        assert a.lambda_min == 0.01
        assert a.anneal_start == 0.7
        assert a.probe_layers == [6, 12, 18]
        assert a.calibration_interval == 500
        assert a.thresholds == {"layer_6": 0.7, "layer_12": 0.6, "layer_18": 0.5}

    def test_fusion(self, valid_config: CooperativeConfig) -> None:
        f = valid_config.fusion
        assert isinstance(f, FusionConfig)
        assert f.backend == "moe_routing"
        assert f.n_clusters == 4

    def test_domains(self, valid_config: CooperativeConfig) -> None:
        assert len(valid_config.domains) == 2
        d0 = valid_config.domains[0]
        assert isinstance(d0, DomainConfig)
        assert d0.id == 1
        assert d0.name == "Code"
        assert d0.data_hint == "github-code, stack-v2"

    def test_from_dict(self) -> None:
        raw = yaml.safe_load(VALID_YAML)
        cfg = CooperativeConfig.from_dict(raw)
        assert cfg.name == "open-20b"
        assert cfg.modules == 20


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_yaml_and_back(self, valid_config: CooperativeConfig, tmp_path: Path) -> None:
        out = tmp_path / "out.yaml"
        valid_config.to_yaml(out)
        reloaded = CooperativeConfig.from_yaml(out)
        assert reloaded.name == valid_config.name
        assert reloaded.modules == valid_config.modules
        assert reloaded.architecture.depth == valid_config.architecture.depth
        assert reloaded.alignment.probe_layers == valid_config.alignment.probe_layers
        assert len(reloaded.domains) == len(valid_config.domains)

    def test_to_dict(self, valid_config: CooperativeConfig) -> None:
        d = valid_config.to_dict()
        assert "cooperative" in d
        assert d["cooperative"]["name"] == "open-20b"
        assert d["cooperative"]["architecture"]["depth"] == 24


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Config file not found"):
            CooperativeConfig.from_yaml(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("key: [unclosed\n  - bad:", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            CooperativeConfig.from_yaml(p)

    def test_not_a_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Expected top-level mapping"):
            CooperativeConfig.from_yaml(p)

    def test_missing_cooperative_key(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("foo: bar\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Missing required top-level key"):
            CooperativeConfig.from_yaml(p)

    def test_missing_name(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        del raw["cooperative"]["name"]
        p = tmp_path / "no_name.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="Missing required field: 'name'"):
            CooperativeConfig.from_yaml(p)

    def test_missing_architecture(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        del raw["cooperative"]["architecture"]
        p = tmp_path / "no_arch.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="Missing required field: 'architecture'"):
            CooperativeConfig.from_yaml(p)

    def test_missing_alignment(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        del raw["cooperative"]["alignment"]
        p = tmp_path / "no_align.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="Missing required field: 'alignment'"):
            CooperativeConfig.from_yaml(p)

    def test_wrong_type_modules(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        raw["cooperative"]["modules"] = "twenty"
        p = tmp_path / "bad_modules.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="must be an integer"):
            CooperativeConfig.from_yaml(p)

    def test_domain_missing_id(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        raw["cooperative"]["domains"] = [{"name": "Code"}]
        p = tmp_path / "bad_domain.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="missing required field: 'id'"):
            CooperativeConfig.from_yaml(p)

    def test_domain_missing_name(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        raw["cooperative"]["domains"] = [{"id": 1}]
        p = tmp_path / "bad_domain2.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="missing required field: 'name'"):
            CooperativeConfig.from_yaml(p)


# ---------------------------------------------------------------------------
# Defaults tests
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_fusion_defaults(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        del raw["cooperative"]["fusion"]
        p = tmp_path / "no_fusion.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        cfg = CooperativeConfig.from_yaml(p)
        assert cfg.fusion.backend == "moe_routing"
        assert cfg.fusion.n_clusters == 4

    def test_empty_domains(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        del raw["cooperative"]["domains"]
        p = tmp_path / "no_domains.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        cfg = CooperativeConfig.from_yaml(p)
        assert cfg.domains == []

    def test_norm_default(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(VALID_YAML)
        del raw["cooperative"]["architecture"]["norm"]
        p = tmp_path / "no_norm.yaml"
        p.write_text(yaml.dump(raw), encoding="utf-8")
        cfg = CooperativeConfig.from_yaml(p)
        assert cfg.architecture.norm == "rmsnorm"
