"""Trainable layer implementations."""

from __future__ import annotations

import numpy as np

from .activations import get_activation
from .initializers import get_weight_initializer, initialize_bias


class Dense:
    """A fully connected layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "linear",
        name: str | None = None,
        weight_initializer: str | None = None,
        bias_initializer: str = "zeros",
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Dense layer dimensions must be positive integers.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        self.activation = get_activation(self.activation_name)
        self.name = name

        weight_init = get_weight_initializer(weight_initializer, self.activation_name)
        self.weights = weight_init(input_dim, output_dim)
        self.bias = initialize_bias(output_dim, bias_initializer)

        self.inputs: np.ndarray | None = None
        self.linear_output: np.ndarray | None = None
        self.output: np.ndarray | None = None

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the layer forward."""

        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected inputs with shape (n_samples, {self.input_dim}), got {inputs.shape}."
            )

        self.inputs = inputs
        self.linear_output = inputs @ self.weights + self.bias
        self.output = self.activation.forward(self.linear_output)
        return self.output

    def backward(self, grad_output: np.ndarray, apply_activation_derivative: bool = True) -> np.ndarray:
        """Backpropagate the gradient through the layer.

        The incoming gradient is expected to already be averaged across the batch.
        """

        if self.inputs is None or self.linear_output is None or self.output is None:
            raise RuntimeError("Cannot call backward before forward.")

        grad_z = grad_output
        if apply_activation_derivative:
            grad_z = grad_output * self.activation.derivative(self.linear_output, self.output)

        self.grad_weights = self.inputs.T @ grad_z
        self.grad_bias = np.sum(grad_z, axis=0, keepdims=True)
        return grad_z @ self.weights.T

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""

        return int(self.weights.size + self.bias.size)

    def parameters(self) -> dict[str, np.ndarray]:
        """Return the layer parameters."""

        return {"weights": self.weights, "bias": self.bias}
