mod tensor;
mod activation;
mod loss;
mod layer;
mod optim;

use tensor::Tensor;

fn main() {
    println!("ML from scratch in Rust!\n");

    // Create some tensors
    let t1 = Tensor::randn(vec![3, 3]);
    println!("Random 3x3 tensor: {:?}", t1);

    let t2 = Tensor::zeros(vec![2, 4]);
    println!("Zeros 2x4 tensor: {:?}", t2);

    let t3 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    println!("Custom 2x2 tensor: {:?}", t3);
}
