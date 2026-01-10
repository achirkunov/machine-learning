mod matrix;

use matrix::Matrix;

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

    // Visualize first image (28x28)
    let image_idx = 2;
    let offset = image_idx * 784;

    draw_mnist_digit(&train_images.data[offset..offset + 784]);
}
