#[derive(Debug)] // Auto-generates code to print the struct
pub struct Matrix {
    pub rows: usize, // TODO: might need to use u32 for memory optimization for millions of small matrices
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {


    // Copy into existing buffer (avoid allocation)
    pub fn copy_into(src: &Matrix, dst: &mut Matrix) {
        // Panic if dimensions don't match (programmer error)
        assert!(src.rows == dst.rows && src.cols == dst.cols,
            "Matrix dimensions must match");
        dst.data.copy_from_slice(&src.data);
    }

    pub fn clear(&mut self) {
        // Mutates matrix in place
        todo!()
    }

    pub fn full(rows: usize, cols: usize, value: f32) -> Self {
        Self { rows, cols, data: vec![value; rows * cols] }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::full(rows, cols, 0.0)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::full(rows, cols, 1.0)
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

    // this is equivalent to void mat_add(const Matrix* a, const Matrix* b, Matrix* out)
    pub fn add(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        for i in 0..a.data.len() {
            out.data[i] = a.data[i] + b.data[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_fill() {
        let mut m = Matrix::zeros(2, 2);
        m.fill(5.0);
        assert!(m.data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_add() {
        let a = Matrix::zeros(2, 2).filled(1.0);
        let b = Matrix::zeros(2, 2).filled(2.0);
        let mut out = Matrix::zeros(2, 2);
        Matrix::add(&a, &b, &mut out);
        assert!(out.data.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_copy_into() {
        let src = Matrix::zeros(2, 2).filled(7.0);
        let mut dst = Matrix::zeros(2, 2);
        Matrix::copy_into(&src, &mut dst);
        assert_eq!(src.data, dst.data);
    }

    #[test]
    fn test_full() {
        let m = Matrix::full(2, 3, 7.0);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert!(m.data.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn test_ones() {
        let m = Matrix::ones(2, 3);
        assert!(m.data.iter().all(|&x| x == 1.0));
    }
}
