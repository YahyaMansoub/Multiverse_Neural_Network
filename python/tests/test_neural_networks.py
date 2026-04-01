"""Basic tests for the from-scratch neural-network package."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multiverse_ai.neural_networks import Dense, Sequential


class NeuralNetworkLibraryTests(unittest.TestCase):
    def test_summary_reports_total_parameters(self) -> None:
        model = Sequential(
            [
                Dense(2, 4, activation="tanh"),
                Dense(4, 1, activation="sigmoid"),
            ]
        )

        summary = model.summary(print_fn=None)

        self.assertIn("dense_1", summary)
        self.assertIn("dense_2", summary)
        self.assertIn("Total params: 17", summary)

    def test_xor_training_reaches_high_accuracy(self) -> None:
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
                Dense(2, 8, activation="tanh"),
                Dense(8, 1, activation="sigmoid"),
            ]
        )
        model.compile(
            loss="binary_cross_entropy",
            learning_rate=0.5,
            metrics=["accuracy"],
        )

        history = model.fit(X, y, epochs=6000, shuffle=False, verbose=False)
        metrics = model.evaluate(X, y)

        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertGreaterEqual(metrics["accuracy"], 0.99)

    def test_predict_classes_accepts_single_sample(self) -> None:
        np.random.seed(7)

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
            ]
        )
        model.compile(loss="binary_cross_entropy", learning_rate=0.5)
        model.fit(X, y, epochs=3000, shuffle=False, verbose=False)

        prediction = model.predict_classes(np.array([1.0, 0.0]))

        self.assertEqual(prediction.shape, (1,))
        self.assertIn(int(prediction[0]), (0, 1))


if __name__ == "__main__":
    unittest.main()
