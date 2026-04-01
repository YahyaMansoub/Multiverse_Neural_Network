"""High-level model interfaces."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .layers import Dense
from .losses import get_loss
from .metrics import get_metric
from .optimizers import get_optimizer


class Sequential:
    """A simple Keras-like sequential neural-network container."""

    def __init__(self, layers: Iterable[Dense] | None = None, name: str = "sequential") -> None:
        self.name = name
        self.layers: list[Dense] = []
        self.loss = None
        self.optimizer = None
        self.metrics = []
        self.history: dict[str, list[float]] = {}

        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Dense) -> "Sequential":
        """Append a layer to the model."""

        if self.layers and self.layers[-1].output_dim != layer.input_dim:
            raise ValueError(
                "Layer dimensions do not line up. "
                f"Expected input_dim={self.layers[-1].output_dim}, got {layer.input_dim}."
            )

        if layer.name is None:
            layer.name = f"dense_{len(self.layers) + 1}"

        self.layers.append(layer)
        return self

    def compile(
        self,
        loss: str = "binary_cross_entropy",
        optimizer=None,
        learning_rate: float = 0.01,
        metrics: list[str] | None = None,
    ) -> "Sequential":
        """Configure the model for training."""

        self.loss = get_loss(loss)
        self.optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        self.metrics = [get_metric(metric_name) for metric_name in (metrics or [])]
        self._reset_history()
        return self

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run a full forward pass."""

        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def fit(
        self,
        X,
        y,
        epochs: int = 1000,
        batch_size: int | None = None,
        shuffle: bool = True,
        verbose: bool = True,
        verbose_every: int | None = None,
    ) -> dict[str, list[float]]:
        """Train the model on in-memory numpy data."""

        self._validate_compiled()
        X_array = self._prepare_training_features(X)
        y_array = self._prepare_targets(y)
        self._validate_dataset_shapes(X_array, y_array)

        n_samples = X_array.shape[0]
        effective_batch_size = n_samples if batch_size is None else int(batch_size)
        if effective_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self._reset_history()
        verbose_every = verbose_every or max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, n_samples, effective_batch_size):
                batch_indices = indices[start : start + effective_batch_size]
                X_batch = X_array[batch_indices]
                y_batch = y_array[batch_indices]

                y_pred_batch = self.forward(X_batch)
                self._backward(y_batch, y_pred_batch)
                self.optimizer.step(self.layers)

            epoch_predictions = self.forward(X_array)
            epoch_results = self._evaluate_arrays(y_array, epoch_predictions)
            for key, value in epoch_results.items():
                self.history[key].append(value)

            if verbose and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                print(self._format_progress(epoch, epochs, epoch_results))

        return self.history

    def predict(self, X) -> np.ndarray:
        """Return model outputs for the provided inputs."""

        X_array = self._prepare_inference_features(X)
        return self.forward(X_array)

    def predict_proba(self, X) -> np.ndarray:
        """Alias kept for sigmoid-based binary classification models."""

        return self.predict(X)

    def predict_classes(self, X, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions."""

        predictions = self.predict(X)
        if predictions.shape[1] != 1:
            raise ValueError("predict_classes currently supports only single-output models.")
        return (predictions.reshape(-1) >= threshold).astype(int)

    def evaluate(self, X, y) -> dict[str, float]:
        """Evaluate the model on a dataset."""

        X_array = self._prepare_inference_features(X)
        y_array = self._prepare_targets(y)
        self._validate_dataset_shapes(X_array, y_array)
        predictions = self.forward(X_array)
        return self._evaluate_arrays(y_array, predictions)

    def summary(self, print_fn=print) -> str:
        """Return a readable model summary."""

        lines = [
            f"Model: {self.name}",
            "=" * 72,
            f"{'Layer':<16}{'Shape':<18}{'Activation':<16}{'Params':>10}",
            "=" * 72,
        ]

        total_params = 0
        for layer in self.layers:
            shape = f"{layer.input_dim} -> {layer.output_dim}"
            lines.append(
                f"{layer.name:<16}{shape:<18}{layer.activation_name:<16}{layer.parameter_count:>10}"
            )
            total_params += layer.parameter_count

        lines.append("=" * 72)
        lines.append(f"Total params: {total_params}")
        if self.loss is not None:
            lines.append(f"Loss: {self.loss.name}")
        if self.optimizer is not None:
            learning_rate = getattr(self.optimizer, "learning_rate", "n/a")
            lines.append(f"Optimizer: {self.optimizer.name} (learning_rate={learning_rate})")

        summary_text = "\n".join(lines)
        if print_fn is not None:
            print_fn(summary_text)
        return summary_text

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Run backpropagation through the whole network."""

        if not self.layers:
            raise ValueError("Sequential models need at least one layer before training.")

        last_layer = self.layers[-1]
        if self.loss.name == "binary_cross_entropy" and last_layer.activation_name == "sigmoid":
            grad_output = (y_pred - y_true) / y_true.shape[0]
            grad_output = last_layer.backward(grad_output, apply_activation_derivative=False)
            remaining_layers = reversed(self.layers[:-1])
        else:
            grad_output = self.loss.gradient(y_true, y_pred)
            remaining_layers = reversed(self.layers)

        for layer in remaining_layers:
            grad_output = layer.backward(grad_output)

    def _evaluate_arrays(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        results = {"loss": self.loss(y_true, y_pred)}
        for metric in self.metrics:
            results[metric.name] = metric(y_true, y_pred)
        return results

    def _format_progress(self, epoch: int, epochs: int, metrics: dict[str, float]) -> str:
        metric_parts = [f"{key}: {value:.6f}" for key, value in metrics.items()]
        return f"Epoch {epoch}/{epochs} - " + " - ".join(metric_parts)

    def _reset_history(self) -> None:
        self.history = {"loss": []}
        for metric in self.metrics:
            self.history[metric.name] = []

    def _validate_compiled(self) -> None:
        if not self.layers:
            raise ValueError("Add at least one layer before training the model.")
        if self.loss is None or self.optimizer is None:
            raise ValueError("Call compile() before fit().")

    def _validate_dataset_shapes(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must contain the same number of samples, got {X.shape[0]} and {y.shape[0]}."
            )

        if X.shape[1] != self.layers[0].input_dim:
            raise ValueError(
                f"Expected {self.layers[0].input_dim} input features, got {X.shape[1]}."
            )

        if y.shape[1] != self.layers[-1].output_dim:
            raise ValueError(
                f"Expected {self.layers[-1].output_dim} target columns, got {y.shape[1]}."
            )

    def _prepare_training_features(self, X) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError(
                "Training features must be a 2D array shaped like (n_samples, n_features). "
                "For single-feature data, reshape with X.reshape(-1, 1)."
            )
        return X_array

    def _prepare_inference_features(self, X) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            if not self.layers:
                raise ValueError("The model needs layers before inference can determine input shape.")
            if X_array.shape[0] != self.layers[0].input_dim:
                raise ValueError(
                    f"Expected a single sample with {self.layers[0].input_dim} features, "
                    f"got {X_array.shape[0]}."
                )
            X_array = X_array.reshape(1, -1)
        elif X_array.ndim != 2:
            raise ValueError("Inference features must be a 1D single sample or a 2D batch.")
        return X_array

    def _prepare_targets(self, y) -> np.ndarray:
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        elif y_array.ndim != 2:
            raise ValueError("Targets must be a 1D or 2D numpy-compatible array.")
        return y_array
