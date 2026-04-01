# Python Library

The active Python package lives here and is built to stay modular as the repository grows.

## Package Layout

```text
python/
|-- examples/
|-- src/
|   `-- multiverse_ai/
|       `-- neural_networks/
|           |-- activations.py
|           |-- initializers.py
|           |-- layers.py
|           |-- losses.py
|           |-- metrics.py
|           |-- models.py
|           `-- optimizers.py
`-- tests/
```

## Current API

```python
import numpy as np

from multiverse_ai.neural_networks import Dense, Sequential

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

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

model.fit(X, y, epochs=5000, verbose_every=500)
predictions = model.predict_classes(X)
```

## Design Notes

- `layers.py` contains trainable building blocks
- `activations.py` and `losses.py` keep the math separate and reusable
- `optimizers.py` isolates parameter updates
- `models.py` exposes the simple library-style workflow
- `tests/` checks that the library learns XOR and that the public API works

## Current Scope

This is intentionally a basic first version. The goal right now is not feature-completeness; it is a clean and understandable foundation that can later grow into wider AI modules.
