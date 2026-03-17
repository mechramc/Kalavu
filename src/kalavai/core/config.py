"""Cooperative configuration schema and parser for kalavai.yaml.

Loads, validates, and provides typed access to the full cooperative config
defined in spec section 6.1.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kalavai.core.exceptions import ConfigError


@dataclass
class ArchitectureConfig:
    """Transformer architecture parameters."""

    depth: int
    d_model: int
    n_heads: int
    ffn_ratio: float
    norm: str = "rmsnorm"


@dataclass
class AlignmentConfig:
    """CKA alignment constraint parameters."""

    lambda_max: float
    lambda_min: float
    anneal_start: float
    probe_layers: list[int]
    calibration_interval: int = 500
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class FusionConfig:
    """Fusion pipeline parameters."""

    backend: str = "moe_routing"
    n_clusters: int = 4


@dataclass
class DomainConfig:
    """A single domain assignment."""

    id: int
    name: str
    data_hint: str = ""


@dataclass
class CooperativeConfig:
    """Top-level cooperative configuration.

    Args:
        name: Cooperative identifier.
        modules: Number of modules in the cooperative.
        target_params_per_module: Target parameter count per module (e.g. "1B").
        architecture: Transformer architecture settings.
        alignment: CKA alignment constraint settings.
        fusion: Fusion pipeline settings.
        domains: List of domain assignments.
    """

    name: str
    modules: int
    target_params_per_module: str
    architecture: ArchitectureConfig
    alignment: AlignmentConfig
    fusion: FusionConfig = field(default_factory=FusionConfig)
    domains: list[DomainConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CooperativeConfig:
        """Load and validate a cooperative config from a YAML file.

        Args:
            path: Path to the kalavai.yaml file.

        Returns:
            A validated CooperativeConfig instance.

        Raises:
            ConfigError: If the file is missing, unparseable, or has invalid/missing fields.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read config file: {exc}") from exc

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError(f"Expected top-level mapping in {path}, got {type(raw).__name__}")

        if "cooperative" not in raw:
            raise ConfigError("Missing required top-level key: 'cooperative'")

        return cls._from_dict(raw["cooperative"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CooperativeConfig:
        """Create a CooperativeConfig from a raw dictionary.

        Args:
            data: Dictionary with a 'cooperative' key or the cooperative mapping directly.

        Returns:
            A validated CooperativeConfig instance.

        Raises:
            ConfigError: If required fields are missing or invalid.
        """
        if "cooperative" in data:
            data = data["cooperative"]
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, coop: dict[str, Any]) -> CooperativeConfig:
        """Build a CooperativeConfig from the cooperative sub-dict."""
        if not isinstance(coop, dict):
            raise ConfigError(
                f"'cooperative' must be a mapping, got {type(coop).__name__}"
            )

        # --- required scalar fields ---
        name = _require_str(coop, "name")
        modules = _require_int(coop, "modules")
        target = _require(coop, "target_params_per_module")
        target_str = str(target)

        # --- architecture (required) ---
        arch_raw = _require_section(coop, "architecture")
        architecture = ArchitectureConfig(
            depth=_require_int(arch_raw, "depth"),
            d_model=_require_int(arch_raw, "d_model"),
            n_heads=_require_int(arch_raw, "n_heads"),
            ffn_ratio=_require_float(arch_raw, "ffn_ratio"),
            norm=arch_raw.get("norm", "rmsnorm"),
        )

        # --- alignment (required) ---
        align_raw = _require_section(coop, "alignment")
        alignment = AlignmentConfig(
            lambda_max=_require_float(align_raw, "lambda_max"),
            lambda_min=_require_float(align_raw, "lambda_min"),
            anneal_start=_require_float(align_raw, "anneal_start"),
            probe_layers=_require_list(align_raw, "probe_layers", int),
            calibration_interval=int(align_raw.get("calibration_interval", 500)),
            thresholds=_parse_thresholds(align_raw.get("thresholds", {})),
        )

        # --- fusion (optional, has defaults) ---
        fusion_raw = coop.get("fusion", {})
        if not isinstance(fusion_raw, dict):
            raise ConfigError(f"'fusion' must be a mapping, got {type(fusion_raw).__name__}")
        fusion = FusionConfig(
            backend=str(fusion_raw.get("backend", "moe_routing")),
            n_clusters=int(fusion_raw.get("n_clusters", 4)),
        )

        # --- domains (optional list) ---
        domains_raw = coop.get("domains", [])
        if not isinstance(domains_raw, list):
            raise ConfigError(f"'domains' must be a list, got {type(domains_raw).__name__}")
        domains = [_parse_domain(i, d) for i, d in enumerate(domains_raw)]

        return cls(
            name=name,
            modules=modules,
            target_params_per_module=target_str,
            architecture=architecture,
            alignment=alignment,
            fusion=fusion,
            domains=domains,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a plain dictionary."""
        return {
            "cooperative": {
                "name": self.name,
                "modules": self.modules,
                "target_params_per_module": self.target_params_per_module,
                "architecture": dataclasses.asdict(self.architecture),
                "alignment": dataclasses.asdict(self.alignment),
                "fusion": dataclasses.asdict(self.fusion),
                "domains": [dataclasses.asdict(d) for d in self.domains],
            }
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to a YAML file.

        Args:
            path: Destination file path.

        Raises:
            ConfigError: If the file cannot be written.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ConfigError(f"Cannot write config file: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require(data: dict[str, Any], key: str) -> Any:
    """Return data[key] or raise ConfigError."""
    if key not in data:
        raise ConfigError(f"Missing required field: '{key}'")
    return data[key]


def _require_str(data: dict[str, Any], key: str) -> str:
    val = _require(data, key)
    if not isinstance(val, str):
        raise ConfigError(f"Field '{key}' must be a string, got {type(val).__name__}")
    return val


def _require_int(data: dict[str, Any], key: str) -> int:
    val = _require(data, key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise ConfigError(f"Field '{key}' must be an integer, got {type(val).__name__}")
    return val


def _require_float(data: dict[str, Any], key: str) -> float:
    val = _require(data, key)
    if isinstance(val, bool):
        raise ConfigError(f"Field '{key}' must be a number, got bool")
    if not isinstance(val, (int, float)):
        raise ConfigError(f"Field '{key}' must be a number, got {type(val).__name__}")
    return float(val)


def _require_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = _require(data, key)
    if not isinstance(val, dict):
        raise ConfigError(f"Section '{key}' must be a mapping, got {type(val).__name__}")
    return val


def _require_list(data: dict[str, Any], key: str, element_type: type) -> list[Any]:
    val = _require(data, key)
    if not isinstance(val, list):
        raise ConfigError(f"Field '{key}' must be a list, got {type(val).__name__}")
    for i, item in enumerate(val):
        if not isinstance(item, element_type):
            raise ConfigError(
                f"Field '{key}[{i}]' must be {element_type.__name__}, "
                f"got {type(item).__name__}"
            )
    return val


def _parse_thresholds(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ConfigError(f"'thresholds' must be a mapping, got {type(raw).__name__}")
    result: dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ConfigError(f"Threshold '{k}' must be a number, got {type(v).__name__}")
        result[str(k)] = float(v)
    return result


def _parse_domain(index: int, raw: Any) -> DomainConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Domain at index {index} must be a mapping, got {type(raw).__name__}")
    if "id" not in raw:
        raise ConfigError(f"Domain at index {index} missing required field: 'id'")
    if "name" not in raw:
        raise ConfigError(f"Domain at index {index} missing required field: 'name'")
    return DomainConfig(
        id=int(raw["id"]),
        name=str(raw["name"]),
        data_hint=str(raw.get("data_hint", "")),
    )
