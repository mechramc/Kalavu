"""Custom exception hierarchy for Kalavai."""


class KalavaiError(Exception):
    """Base exception for all Kalavai errors."""


class ConfigError(KalavaiError):
    """Raised when configuration is invalid or missing."""


class AlignmentError(KalavaiError):
    """Raised when transaction alignment between books fails."""


class CheckpointValidationError(KalavaiError):
    """Raised when a checkpoint fails validation checks."""


class FusionError(KalavaiError):
    """Raised when fusing multiple data sources fails."""


class CooperativeError(KalavaiError):
    """Raised when cooperative-specific operations fail."""


class HardwareError(KalavaiError):
    """Raised when required hardware (CUDA GPU) is not available."""
