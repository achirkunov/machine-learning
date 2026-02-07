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

/// Compute accuracy: fraction of correct predictions (batched)
fn compute_accuracy(model: &mut model::ModelContext, images: &Matrix, labels: &Matrix, batch_size: usize) -> f32 {
    let n = images.rows;
    let num_batches = n / batch_size;
    let mut correct = 0;
    let mut total = 0;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = batch_start + batch_size;

        Matrix::copy_rows_into(images, batch_start, batch_end, model.input_buffer_mut());
        Matrix::copy_rows_into(labels, batch_start, batch_end, model.target_buffer_mut());
        model.forward();

        // Get predicted classes (argmax per row)
        let output = model.output();
        let preds = Matrix::argmax_per_row(output);

        // Get true classes (argmax per row of target buffer)
        let truths = Matrix::argmax_per_row(model.target_buffer());

        for i in 0..batch_size {
            if preds[i] == truths[i] {
                correct += 1;
            }
        }
        total += batch_size;
    }

    correct as f32 / total as f32
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

    // Training hyperparameters
    let learning_rate = 0.03;
    let batch_size = 250;
    let epochs = 10;
    let print_every = 10000;

    // Build model: input -> matmul -> add_bias -> softmax -> cross_entropy
    // Model built with batch_size rows for batched training
    let mut b = ModelBuilder::new();
    let x = b.input(batch_size, 784);
    let w = b.parameter(784, 10);
    let bias = b.parameter(1, 10);
    let logits = b.matmul(x, w);
    let logits_biased = b.add_bias(logits, bias); // Broadcasting bias addition
    let probs = b.softmax(logits_biased);
    let y = b.input(batch_size, 10);
    let loss = b.cross_entropy(probs, y);

    let mut m = b.build(x, probs, y, loss);

    println!("Training for {} epochs with lr={}, batch_size={}", epochs, learning_rate, batch_size);
    println!("-------------------------------------------");

    let start = Instant::now();

    // Number of complete batches (skip incomplete last batch)
    let num_batches = train_images.rows / batch_size;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut samples_processed = 0;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = batch_start + batch_size;

            // Copy batch data directly into model buffers (zero-copy, no temp)
            Matrix::copy_rows_into(&train_images, batch_start, batch_end, m.input_buffer_mut());
            Matrix::copy_rows_into(&train_labels, batch_start, batch_end, m.target_buffer_mut());
            m.zero_grad();
            m.forward();
            total_loss += m.loss();
            m.backward();

            // SGD update once per batch (scale by batch size)
            m.sgd_step(learning_rate / batch_size as f32);

            samples_processed = batch_end;
            if samples_processed % print_every == 0 {
                let avg_loss = total_loss / samples_processed as f32;
                println!(
                    "Epoch {} [{}/{}] avg_loss: {:.4}",
                    epoch + 1,
                    samples_processed,
                    train_images.rows,
                    avg_loss
                );
            }
        }

        let avg_loss = total_loss / samples_processed as f32;
        let test_acc = compute_accuracy(&mut m, &test_images, &test_labels, batch_size);
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
    let train_acc = compute_accuracy(&mut m, &train_images, &train_labels, batch_size);
    let test_acc = compute_accuracy(&mut m, &test_images, &test_labels, batch_size);
    println!("\nFinal Results:");
    println!("  Train accuracy: {:.2}%", train_acc * 100.0);
    println!("  Test accuracy:  {:.2}%", test_acc * 100.0);
    println!("  Training time:  {:.2}s", elapsed.as_secs_f32());
}
