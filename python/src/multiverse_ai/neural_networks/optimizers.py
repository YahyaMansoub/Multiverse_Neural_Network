"""Optimizers for updating trainable parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SGD:
    """Stochastic gradient descent."""

    learning_rate: float = 0.01
    name: str = "sgd"

    def step(self, layers) -> None:
        for layer in layers:
            layer.weights -= self.learning_rate * layer.grad_weights
            layer.bias -= self.learning_rate * layer.grad_bias


def get_optimizer(optimizer=None, learning_rate: float = 0.01):
    """Resolve the optimizer object used by the model."""

    if optimizer is None:
        return SGD(learning_rate=learning_rate)

    if isinstance(optimizer, str):
        if optimizer.lower() != "sgd":
            raise ValueError("Only the 'sgd' optimizer is currently implemented.")
        return SGD(learning_rate=learning_rate)

    if not hasattr(optimizer, "step"):
        raise ValueError("Custom optimizers must provide a step(layers) method.")
    return optimizer
