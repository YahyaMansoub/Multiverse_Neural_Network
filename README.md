# Multiverse Neural Network

This repository is now organized as a Python-first foundation for building neural networks from scratch, while preserving the older multi-language experiments as reference material.

The current focus is intentionally narrow:

- build a basic neural-network library in Python
- keep the mathematical learning path visible through notebooks
- preserve the existing C, C++, Go, Java, and legacy Python scripts
- prepare the repository for future expansion into vision, NLP, and broader AI topics

## Current Direction

The first learning arc already visible in the notebooks is:

1. Linear regression as the simplest one-neuron model
2. Perceptrons and binary classification
3. Multi-layer perceptrons and the XOR problem
4. Activation functions and why they change training behavior

That arc now maps directly to the new Python package under `python/`.

## Repository Layout

```text
Multiverse_Neural_Network/
|-- notebooks/
|   |-- 01_basics/
|   |-- 02_ai_vision/
|   |-- 03_natural_language/
|   |-- 04_deep_learning/
|   `-- 99_archive/
|
|-- python/
|   |-- examples/
|   |-- src/
|   |   `-- multiverse_ai/
|   |       `-- neural_networks/
|   |-- tests/
|   |-- pyproject.toml
|   `-- README.md
|
|-- code/
|   |-- c/
|   |-- cpp/
|   |-- go/
|   |-- java/
|   `-- python/
|
`-- MELANGE/
```

## What Is Active Now

### `python/`

This is the active Python workspace for the new from-scratch library. It currently includes:

- dense layers
- core activations: `linear`, `sigmoid`, `tanh`, `relu`
- losses: mean squared error and binary cross-entropy
- a simple SGD optimizer
- metrics: accuracy, binary accuracy, mean squared error, R2
- a `Sequential` model with `add`, `compile`, `fit`, `predict`, `predict_classes`, `evaluate`, and `summary`

### `code/`

This directory is preserved as legacy/reference code. Nothing there was removed so you can still compare the older single-file implementations across languages.

### `notebooks/`

The notebooks are now grouped by learning area instead of being left flat at the root of the folder:

- `01_basics/` for the current foundations
- `02_ai_vision/` placeholder for future computer vision notebooks
- `03_natural_language/` placeholder for future NLP notebooks
- `04_deep_learning/` placeholder for broader deep-learning work
- `99_archive/` for preserved alternate or older notebook versions

## Quick Start

Run the example:

```powershell
cd python
$env:PYTHONPATH = "src"
python examples\xor_classification.py
```

Run the tests:

```powershell
cd python
$env:PYTHONPATH = "src"
python -m unittest discover tests
```

## Why The Python Package Uses `multiverse_ai`

The repository is still focused on neural networks today, but the broader package namespace leaves room for the direction you described later: a larger `Multiverse Artificial Intelligence` project that can eventually include vision, NLP, and other learning systems without needing another structural reset.
