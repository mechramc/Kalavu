"""Tests for Kalavai custom exception hierarchy."""

import pytest

from kalavai.core.exceptions import (
    AlignmentError,
    CheckpointValidationError,
    ConfigError,
    CooperativeError,
    FusionError,
    KalavaiError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from KalavaiError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigError,
            AlignmentError,
            CheckpointValidationError,
            FusionError,
            CooperativeError,
        ],
    )
    def test_subclass_of_kalavai_error(self, exc_class: type) -> None:
        assert issubclass(exc_class, KalavaiError)

    @pytest.mark.parametrize(
        "exc_class",
        [
            KalavaiError,
            ConfigError,
            AlignmentError,
            CheckpointValidationError,
            FusionError,
            CooperativeError,
        ],
    )
    def test_raise_and_catch(self, exc_class: type) -> None:
        with pytest.raises(exc_class, match="test message"):
            raise exc_class("test message")

    def test_catch_all_via_base(self) -> None:
        """Catching KalavaiError catches any subclass."""
        with pytest.raises(KalavaiError):
            raise ConfigError("caught via base")
