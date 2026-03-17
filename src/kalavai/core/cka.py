"""Centered Kernel Alignment (CKA) for representational similarity."""

from __future__ import annotations

import torch
from torch import Tensor


def _center(X: Tensor) -> Tensor:
    """Subtract column means from a matrix."""
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: Tensor, Y: Tensor) -> float:
    """Compute linear CKA similarity between two representation matrices.

    Args:
        X: Activation matrix of shape ``[N, D1]``.
        Y: Activation matrix of shape ``[N, D2]``.

    Returns:
        CKA similarity score in ``[0, 1]``.

    Raises:
        ValueError: If *X* and *Y* have different numbers of samples.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Sample counts must match, got {X.shape[0]} and {Y.shape[0]}"
        )

    X = _center(X)
    Y = _center(Y)

    # ||Y^T X||_F^2
    numerator = torch.norm(Y.T @ X, p="fro") ** 2
    # ||X^T X||_F * ||Y^T Y||_F
    denominator = torch.norm(X.T @ X, p="fro") * torch.norm(Y.T @ Y, p="fro")

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()


def cka_loss(h_module: Tensor, h_reference: Tensor) -> Tensor:
    """Differentiable CKA loss for use as an anchor training term.

    Computes ``1 - CKA(h_module, h_reference)`` so that minimising the loss
    maximises representational alignment.

    Args:
        h_module: Module hidden states, shape ``[N, D1]``.
        h_reference: Reference hidden states, shape ``[N, D2]``.

    Returns:
        Scalar loss tensor suitable for ``.backward()``.

    Raises:
        ValueError: If inputs have different numbers of samples.
    """
    if h_module.shape[0] != h_reference.shape[0]:
        raise ValueError(
            f"Sample counts must match, got {h_module.shape[0]} and {h_reference.shape[0]}"
        )

    X = _center(h_module)
    Y = _center(h_reference)

    numerator = torch.norm(Y.T @ X, p="fro") ** 2
    denominator = torch.norm(X.T @ X, p="fro") * torch.norm(Y.T @ Y, p="fro")

    if denominator == 0:
        return torch.tensor(1.0, dtype=h_module.dtype, device=h_module.device)

    cka = numerator / denominator
    return 1.0 - cka
