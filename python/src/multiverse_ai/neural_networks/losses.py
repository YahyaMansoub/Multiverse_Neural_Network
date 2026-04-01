"""Loss functions for the from-scratch neural-network library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class LossFunction:
    """Container for a loss and its gradient."""

    name: str
    forward_fn: Callable[[Array, Array], float]
    gradient_fn: Callable[[Array, Array], Array]

    def __call__(self, y_true: Array, y_pred: Array) -> float:
        return self.forward_fn(y_true, y_pred)

    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        return self.gradient_fn(y_true, y_pred)


def mean_squared_error(y_true: Array, y_pred: Array) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mean_squared_error_gradient(y_true: Array, y_pred: Array) -> Array:
    return (2.0 / y_true.shape[0]) * (y_pred - y_true)


def binary_cross_entropy(y_true: Array, y_pred: Array) -> float:
    clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    loss = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(np.mean(loss))


def binary_cross_entropy_gradient(y_true: Array, y_pred: Array) -> Array:
    clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    return (-(y_true / clipped) + ((1.0 - y_true) / (1.0 - clipped))) / y_true.shape[0]


_LOSSES = {
    "mean_squared_error": LossFunction(
        "mean_squared_error",
        mean_squared_error,
        mean_squared_error_gradient,
    ),
    "mse": LossFunction(
        "mean_squared_error",
        mean_squared_error,
        mean_squared_error_gradient,
    ),
    "binary_cross_entropy": LossFunction(
        "binary_cross_entropy",
        binary_cross_entropy,
        binary_cross_entropy_gradient,
    ),
    "bce": LossFunction(
        "binary_cross_entropy",
        binary_cross_entropy,
        binary_cross_entropy_gradient,
    ),
}


def get_loss(name: str) -> LossFunction:
    """Return a registered loss by name."""

    key = name.lower()
    if key not in _LOSSES:
        supported = ", ".join(sorted(_LOSSES))
        raise ValueError(f"Unsupported loss '{name}'. Available losses: {supported}.")
    return _LOSSES[key]


def available_losses() -> tuple[str, ...]:
    """Return the currently implemented loss names."""

    return tuple(sorted(set(loss.name for loss in _LOSSES.values())))
