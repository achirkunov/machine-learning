mod matrix;

use matrix::Matrix;

fn main() {
    println!("ML from scratch in Rust!\n");

    let m = Matrix::zeros(2,2);

    println!("{:?}", m);

    let m2 = m.filled(1.0);
    println!("{:?}", m2);

    let mut m3 = Matrix::zeros(2,2);
    m3.fill(2.0);
    println!("{:?}", m3);

    let mut d = Matrix::zeros(2,2);
    Matrix::add(&m2,&m3, &mut d);
    println!("{:?}", d);
}
