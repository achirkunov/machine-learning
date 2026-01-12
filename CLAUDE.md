# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
cargo build              # Build the project
cargo run                # Run the binary
cargo test               # Run all tests
cargo test test_name     # Run a specific test
cargo watch -x run       # Auto-rebuild on file changes (requires cargo-watch)
cargo watch -x test      # Auto-run tests on changes
```

## Architecture

This is a machine learning library built from scratch in Rust with no external ML dependencies.

### Core Design Patterns

**Matrix operations use output buffers** to avoid allocations in hot paths:
```rust
Matrix::add(&a, &b, &mut out);  // result written to out
Matrix::mul(&a, &b, &mut out, transpose_a, transpose_b);
```

**Transpose flags** on multiplication avoid copying matrices - instead the indexing pattern changes. The `mul_nn`, `mul_nt`, `mul_tn`, `mul_tt` private helpers implement each variant.

**Dimension assertions** panic with descriptive error messages on mismatch - these are programmer errors, not recoverable runtime errors.

### Module Structure

- `matrix.rs` - Core Matrix type with row-major f32 storage. All operations (add, sub, mul, relu, softmax, cross_entropy) implemented here with tests.
- `model.rs` - Computational graph with `ModelBuilder` and `ModelContext`. Forward pass complete, backward pass not yet implemented.
- `tensor.rs`, `activation.rs`, `loss.rs`, `layer.rs`, `optim.rs` - Placeholder modules for future expansion.

### Computational Graph Design

**VarId indexing** instead of pointers - borrow-checker friendly:
```rust
type VarId = u32;
let x = builder.input(1, 784);   // returns VarId 0
let w = builder.parameter(784, 10); // returns VarId 1
let y = builder.matmul(x, w);    // returns VarId 2
```

**Topological order by construction** - builder only allows referencing existing VarIds, so `forward_prog = [0, 1, 2, ...]` is already sorted. No explicit toposort needed.

**split_at_mut pattern** for compute - avoids borrow checker issues when reading inputs and writing outputs from same Vec:
```rust
let (inputs, outputs) = self.vars.split_at_mut(idx);
let src = &inputs[x as usize].val;
let dst = &mut outputs[0].val;
```

### Known TODOs in Code

- `softmax` currently computes over entire matrix, needs per-row computation for batching
- Backward pass not yet implemented (gradients stored in `Var.grad` but never computed)
- Consider `u32` for rows/cols for memory optimization with many small matrices
