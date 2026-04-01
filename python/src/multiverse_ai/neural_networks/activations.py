"""Activation functions used by the from-scratch neural-network library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Activation:
    """Container for an activation function and its derivative."""

    name: str
    forward_fn: Callable[[Array], Array]
    derivative_fn: Callable[[Array, Array | None], Array]

    def forward(self, z: Array) -> Array:
        return self.forward_fn(z)

    def derivative(self, z: Array, activated: Array | None = None) -> Array:
        return self.derivative_fn(z, activated)


def linear(z: Array) -> Array:
    return z


def linear_derivative(z: Array, activated: Array | None = None) -> Array:
    return np.ones_like(z)


def sigmoid(z: Array) -> Array:
    clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(z: Array, activated: Array | None = None) -> Array:
    output = sigmoid(z) if activated is None else activated
    return output * (1.0 - output)


def tanh(z: Array) -> Array:
    return np.tanh(z)


def tanh_derivative(z: Array, activated: Array | None = None) -> Array:
    output = tanh(z) if activated is None else activated
    return 1.0 - output**2


def relu(z: Array) -> Array:
    return np.maximum(0.0, z)


def relu_derivative(z: Array, activated: Array | None = None) -> Array:
    return (z > 0.0).astype(float)


_ACTIVATIONS = {
    "linear": Activation("linear", linear, linear_derivative),
    "sigmoid": Activation("sigmoid", sigmoid, sigmoid_derivative),
    "tanh": Activation("tanh", tanh, tanh_derivative),
    "relu": Activation("relu", relu, relu_derivative),
}


def get_activation(name: str) -> Activation:
    """Return a registered activation by name."""

    key = name.lower()
    if key not in _ACTIVATIONS:
        supported = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unsupported activation '{name}'. Available activations: {supported}.")
    return _ACTIVATIONS[key]


def available_activations() -> tuple[str, ...]:
    """Return the currently implemented activation names."""

    return tuple(sorted(_ACTIVATIONS))
