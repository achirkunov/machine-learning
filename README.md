# ML from Scratch

A machine learning library built from scratch in Rust, based on the [coding a machine learning library in C from scratch](https://youtu.be/hL_n_GljC0I?si=5L3SVZsUfHhv0gvI) video.

No external ML dependencies — just raw matrix ops, a computational graph with automatic differentiation, and SGD.

## Benchmark

2-layer MLP (784→128→10, ReLU) on MNIST for 10 epochs, batch size 50:

| Implementation | Time | Test Accuracy |
|---|---|---|
| **Rust (this repo)** | **21.41s** | 96.65% |
| PyTorch CPU | 20.65s | 96.19% |
| PyTorch MPS (Apple GPU) | 39.81s | 96.15% |

On par with PyTorch CPU — PyTorch's optimized BLAS closes the gap on larger matmuls (784×128). GPU is slower due to CPU↔GPU transfer latency with small batch sizes.

Run the PyTorch benchmark: `uv run bench_pytorch.py`

## Differences from the C version

**VarId indices instead of pointers.** The C version uses raw pointers to connect graph nodes. In Rust, `VarId = usize` indices into a `Vec<Var>` sidestep the borrow checker entirely — no `Rc`, no `RefCell`, no unsafe.

**Op enum carries inputs.** Instead of a C-style sentinel enum + separate input array, each `Op` variant holds its own `VarId` references: `MatMul(VarId, VarId)`, `ReLU(VarId)`, etc.

**`Option<Matrix>` for gradients.** Replaces nullable pointers. Inputs have `grad: None`, parameters have `grad: Some(Matrix)`. Pattern matching (`if let Some(ref mut g) = var.grad`) makes it impossible to forget a null check.

**`split_at_mut` for the compute loop.** The borrow checker forbids `&self.vars[a]` and `&mut self.vars[idx]` simultaneously. Splitting the vec at `idx` gives two non-overlapping mutable slices — safe concurrent access without unsafe.

**Transpose flags on matmul.** Instead of materializing transposed matrices, `mul(a, b, out, transpose_a, transpose_b)` changes the indexing pattern. Four private helpers (`mul_nn`, `mul_nt`, `mul_tn`, `mul_tt`) implement each variant.

**Output buffer pattern.** All matrix ops write into caller-provided buffers (`Matrix::add(&a, &b, &mut out)`) to avoid allocations in hot paths. Pre-allocated scratch buffers in the backward pass eliminate per-iteration `Matrix::zeros` calls.

**Topological order by construction.** The `ModelBuilder` only allows referencing existing `VarId`s, so the forward program is already sorted — no explicit toposort needed.

**Cache-friendly matmul loop order.** Inner loops ordered `i,k,j` so both output and source matrices are accessed sequentially, enabling LLVM auto-vectorization.

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

Build with native CPU instructions for best performance:

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
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
