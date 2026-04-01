"""Metrics for evaluating neural-network models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Metric:
    """Container for a metric callable."""

    name: str
    function: Callable[[Array, Array], float]

    def __call__(self, y_true: Array, y_pred: Array) -> float:
        return float(self.function(y_true, y_pred))


def binary_accuracy(y_true: Array, y_pred: Array) -> float:
    true_labels = y_true.reshape(-1).astype(int)
    predicted_labels = (y_pred.reshape(-1) >= 0.5).astype(int)
    return float(np.mean(predicted_labels == true_labels))


def mean_squared_error_metric(y_true: Array, y_pred: Array) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def r2_score(y_true: Array, y_pred: Array) -> float:
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(total, 0.0):
        return 1.0 if np.isclose(residual, 0.0) else 0.0
    return float(1.0 - (residual / total))


_METRICS = {
    "accuracy": Metric("accuracy", binary_accuracy),
    "binary_accuracy": Metric("binary_accuracy", binary_accuracy),
    "mean_squared_error": Metric("mean_squared_error", mean_squared_error_metric),
    "mse": Metric("mean_squared_error", mean_squared_error_metric),
    "r2": Metric("r2", r2_score),
}


def get_metric(name: str) -> Metric:
    """Return a registered metric by name."""

    key = name.lower()
    if key not in _METRICS:
        supported = ", ".join(sorted(_METRICS))
        raise ValueError(f"Unsupported metric '{name}'. Available metrics: {supported}.")
    return _METRICS[key]


def available_metrics() -> tuple[str, ...]:
    """Return the currently implemented metric names."""

    return tuple(sorted(set(metric.name for metric in _METRICS.values())))
