"""Weight and bias initialization helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray
WeightInitializer = Callable[[int, int], Array]
BiasInitializer = Callable[[int], Array]


def zeros(output_dim: int) -> Array:
    """Return a zero bias vector."""

    return np.zeros((1, output_dim), dtype=float)


def xavier_uniform(input_dim: int, output_dim: int) -> Array:
    """Xavier/Glorot uniform initialization."""

    limit = np.sqrt(6.0 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, size=(input_dim, output_dim))


def he_normal(input_dim: int, output_dim: int) -> Array:
    """He normal initialization, useful for ReLU-like layers."""

    std = np.sqrt(2.0 / input_dim)
    return np.random.randn(input_dim, output_dim) * std


_WEIGHT_INITIALIZERS = {
    "xavier_uniform": xavier_uniform,
    "he_normal": he_normal,
}

_DEFAULT_WEIGHT_BY_ACTIVATION = {
    "relu": "he_normal",
}


def get_weight_initializer(
    initializer: str | WeightInitializer | None,
    activation_name: str,
) -> WeightInitializer:
    """Resolve a weight initializer from a string, callable, or activation default."""

    if callable(initializer):
        return initializer

    initializer_name = initializer
    if initializer_name is None:
        initializer_name = _DEFAULT_WEIGHT_BY_ACTIVATION.get(activation_name.lower(), "xavier_uniform")

    key = initializer_name.lower()
    if key not in _WEIGHT_INITIALIZERS:
        supported = ", ".join(sorted(_WEIGHT_INITIALIZERS))
        raise ValueError(f"Unsupported initializer '{initializer_name}'. Available initializers: {supported}.")
    return _WEIGHT_INITIALIZERS[key]


def initialize_bias(
    output_dim: int,
    initializer: str | BiasInitializer = "zeros",
) -> Array:
    """Initialize a bias vector."""

    if callable(initializer):
        return initializer(output_dim)

    if initializer.lower() != "zeros":
        raise ValueError("Only the 'zeros' bias initializer is currently implemented.")
    return zeros(output_dim)
