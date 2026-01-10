
#[derive(Debug)] // Auto-generates code to print the struct
struct Matrix {
    rows: usize, // TODO: might need to use u32 for memory optimization for millions of small matrices
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    // Advatanges of new method:
    // Validation
    // Encapsulation
    pub fn zeros(rows: usize, cols: usize) -> Self {
        // Names don't match so need to add data
        Self { rows, cols, data: vec![0.0; rows*cols]}
    }

    pub fn clear(&mut self) {
        // Mutates matrix in place
        todo!()
    }

    // The idiomatic Rust pattern is:
    // clear(&mut self) - mutates in place
    pub fn fill(&mut self, x: f32) {
        self.data.fill(x);
    }
    // cleared(&self) -> Matrix returns new
    pub fn filled(&self, x: f32) -> Self {
        // Only use the explicit name (Matrix) when returning a different type
        Self { rows:self.rows, cols:self.cols, data: vec![x; self.rows*self.cols]}
    }

    pub fn add(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        for i in 0..a.data.len() {
            out.data[i] = a.data[i] + b.data[i];
        }
    }
}

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
