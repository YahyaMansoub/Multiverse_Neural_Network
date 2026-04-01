"""Example showing how to train the new Python neural-network library on XOR."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multiverse_ai.neural_networks import Dense, Sequential


def main() -> None:
    np.random.seed(42)

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 1.0, 0.0])

    model = Sequential(
        [
            Dense(2, 4, activation="tanh"),
            Dense(4, 1, activation="sigmoid"),
        ],
        name="xor_network",
    )

    model.compile(
        loss="binary_cross_entropy",
        learning_rate=0.5,
        metrics=["accuracy"],
    )

    model.summary()
    model.fit(X, y, epochs=5000, shuffle=False, verbose_every=500)

    probabilities = model.predict_proba(X).reshape(-1)
    predictions = model.predict_classes(X)
    metrics = model.evaluate(X, y)

    print("\nFinal evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    print("\nPredictions:")
    for sample, probability, prediction, target in zip(X, probabilities, predictions, y):
        print(
            f"  input={sample.tolist()} probability={probability:.4f} "
            f"predicted={int(prediction)} actual={int(target)}"
        )


if __name__ == "__main__":
    main()
