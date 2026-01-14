mod matrix;
mod model;

use matrix::Matrix;
use model::ModelBuilder;
use std::time::Instant;

fn draw_mnist_digit(data: &[f32]) {
    for y in 0..28 {
        for x in 0..28 {
            let num = data[x + y * 28];
            let col = 232 + (num * 23.0) as u32;
            print!("\x1b[48;5;{}m  ", col);
        }
        println!();
    }
    print!("\x1b[0m");
}

/// Compute accuracy: fraction of correct predictions
fn compute_accuracy(model: &mut model::ModelContext, images: &Matrix, labels: &Matrix) -> f32 {
    let n = images.rows;
    let mut correct = 0;

    let mut input_row = Matrix::zeros(1, 784);
    let mut label_row = Matrix::zeros(1, 10);

    for i in 0..n {
        // Extract row i
        for j in 0..784 {
            input_row.data[j] = images.data[i * 784 + j];
        }
        for j in 0..10 {
            label_row.data[j] = labels.data[i * 10 + j];
        }

        model.set_input(&input_row);
        model.set_target(&label_row);
        model.forward();

        // Get predicted class (argmax of output)
        let output = model.output();
        let pred = output
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        // Get true class (argmax of label)
        let truth = label_row
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        if pred == truth {
            correct += 1;
        }
    }

    correct as f32 / n as f32
}

fn main() {
    println!("ML from scratch in Rust!\n");

    let train_images = Matrix::load(60000, 784, "train_images.mat");
    let test_images = Matrix::load(10000, 784, "test_images.mat");
    let mut train_labels = Matrix::zeros(60000, 10);
    let mut test_labels = Matrix::zeros(10000, 10);

    {
        let train_labels_file = Matrix::load(60000, 1, "train_labels.mat");
        let test_labels_file = Matrix::load(10000, 1, "test_labels.mat");

        for i in 0..train_labels_file.data.len() {
            let num = train_labels_file.data[i] as usize;
            train_labels.data[i * 10 + num] = 1.0;
        }
        for i in 0..test_labels_file.data.len() {
            let num = test_labels_file.data[i] as usize;
            test_labels.data[i * 10 + num] = 1.0;
        }
    }

    // Visualize first image (28x28)
    let image_idx = 500;
    let offset = image_idx * 784;

    draw_mnist_digit(&train_images.data[offset..offset + 784]);

    for i in 0..10 {
        print!("{} ", train_labels.data[image_idx * 10 + i]);
    }
    println!("\n");

    // Build model: input -> matmul -> softmax -> cross_entropy
    let mut b = ModelBuilder::new();
    let x = b.input(1, 784);
    let w = b.parameter(784, 10);
    let logits = b.matmul(x, w);
    let probs = b.softmax(logits); // Added softmax!
    let y = b.input(1, 10);
    let loss = b.cross_entropy(probs, y);

    let mut m = b.build(x, probs, y, loss);

    // Training hyperparameters
    let learning_rate = 0.01; // Reduced from 0.1 for stability
    let epochs = 5;
    let print_every = 10000;

    println!("Training for {} epochs with lr={}", epochs, learning_rate);
    println!("-------------------------------------------");

    let start = Instant::now();

    let mut input_row = Matrix::zeros(1, 784);
    let mut label_row = Matrix::zeros(1, 10);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for i in 0..train_images.rows {
            // Extract row i from training data
            for j in 0..784 {
                input_row.data[j] = train_images.data[i * 784 + j];
            }
            for j in 0..10 {
                label_row.data[j] = train_labels.data[i * 10 + j];
            }

            // Forward pass
            m.set_input(&input_row);
            m.set_target(&label_row);
            m.forward();

            total_loss += m.loss();

            // Backward pass
            m.backward();

            // SGD update
            m.sgd_step(learning_rate);

            if (i + 1) % print_every == 0 {
                let avg_loss = total_loss / (i + 1) as f32;
                println!(
                    "Epoch {} [{}/{}] avg_loss: {:.4}",
                    epoch + 1,
                    i + 1,
                    train_images.rows,
                    avg_loss
                );
            }
        }

        let avg_loss = total_loss / train_images.rows as f32;
        let test_acc = compute_accuracy(&mut m, &test_images, &test_labels);
        println!(
            "Epoch {} complete - avg_loss: {:.4}, test_acc: {:.2}%",
            epoch + 1,
            avg_loss,
            test_acc * 100.0
        );
        println!("-------------------------------------------");
    }

    let elapsed = start.elapsed();

    // Final evaluation
    let train_acc = compute_accuracy(&mut m, &train_images, &train_labels);
    let test_acc = compute_accuracy(&mut m, &test_images, &test_labels);
    println!("\nFinal Results:");
    println!("  Train accuracy: {:.2}%", train_acc * 100.0);
    println!("  Test accuracy:  {:.2}%", test_acc * 100.0);
    println!("  Training time:  {:.2}s", elapsed.as_secs_f32());
}
