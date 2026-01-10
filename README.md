# ML from Scratch

A machine learning library built from scratch in Rust.

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
