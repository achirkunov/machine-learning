mod matrix;
mod model;

use matrix::Matrix;
use model::ModelContext;
use model::ModelBuilder;

fn draw_mnist_digit(data: &[f32]) {
    for y in 0..28 {
        for x in 0..28 {
            let num = data[x + y * 28];
            let col = 232 + (num *23.0) as u32;
            print!("\x1b[48;5;{}m  ", col);
        }
        println!();
    }
    print!("\x1b[0m");
}

fn main() {
    println!("ML from scratch in Rust!\n");

    let train_images = Matrix::load(60000, 784, "train_images.mat");
    let test_images = Matrix::load(10000, 784, "test_images.mat");
    let mut train_labels = Matrix::zeros(60000, 10);
    let mut test_labels = Matrix::zeros(10000,10);

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

    let mut b = ModelBuilder::new();
    let x = b.input(1, 784); // 0
    let w = b.parameter(784, 10); // 1
    let logits = b.matmul(x, w); // 2
    let y = b.input(1, 10); // 3
    let loss = b.cross_entropy(logits, y); // 4

    let m = b.build(x, logits, y, loss);

}
