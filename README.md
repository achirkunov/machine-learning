# ML from Scratch

A machine learning library built from scratch in Rust, based on on the [coding a machine learning library in c from scratch
](https://youtu.be/hL_n_GljC0I?si=5L3SVZsUfHhv0gvI) video.

## Development

Install cargo-watch to auto-rebuild on file changes:

```bash
cargo install cargo-watch
```

Usage:

```bash
cargo watch -x run        # rebuild and run on changes
cargo watch -x check      # just check for errors (faster)
cargo watch -x test       # run tests on changes
```

## MNIST Dataset

Download and prepare the MNIST dataset:

```bash
uv run --with tensorflow --with tensorflow-datasets --with numpy mnist.py
```

This creates four binary files:
- `train_images.mat` — 60k training images (28x28, normalized 0-1)
- `train_labels.mat` — 60k training labels (0-9)
- `test_images.mat` — 10k test images
- `test_labels.mat` — 10k test labels

## Gradient Formulas

Backpropagation computes gradients by applying the chain rule in reverse topological order. For each operation, given the upstream gradient `dL/dz`, we compute gradients for the inputs.

| Op | Forward | Backward |
|----|---------|----------|
| `Add(a, b)` | `z = a + b` | `dL/da += dL/dz`, `dL/db += dL/dz` |
| `Sub(a, b)` | `z = a - b` | `dL/da += dL/dz`, `dL/db -= dL/dz` |
| `ReLU(x)` | `y = max(0, x)` | `dL/dx += dL/dy * (x > 0 ? 1 : 0)` |
| `MatMul(A, B)` | `Z = A × B` | `dL/dA += dL/dZ × Bᵀ`, `dL/dB += Aᵀ × dL/dZ` |
| `Softmax + CrossEntropy` | `L = -Σ(target * ln(softmax(x)))` | `dL/dx = softmax(x) - target` |
