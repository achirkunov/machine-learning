use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::BufReader;

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

    pub fn full(rows: usize, cols: usize, value: f32) -> Self {
        Self { rows, cols, data: vec![value; rows * cols] }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::full(rows, cols, 0.0)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::full(rows, cols, 1.0)
    }

    /// Random uniform values in [0, 1)
    pub fn rand(rows: usize, cols: usize) -> Self {
        let data: Vec<f32> = (0..rows * cols).map(|_| fastrand::f32()).collect();
        Self { rows, cols, data }
    }

    /// Random values in [-scale, scale] for weight initialization
    pub fn rand_scaled(rows: usize, cols: usize, scale: f32) -> Self {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| (fastrand::f32() - 0.5) * 2.0 * scale)
            .collect();
        Self { rows, cols, data }
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

    pub fn clear(&mut self) {
        // Mutates matrix in place
        self.data.fill(0.0);
    }

    pub fn scale(&mut self, scale: f32) {
        let size = self.rows * self.cols;
        for i in 0..size {
            self.data[i] *= scale;
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    // this is equivalent to void mat_add(const Matrix* a, const Matrix* b, Matrix* out)
    pub fn add(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        assert!(a.rows == b.rows && a.cols == b.cols,
            "a and b dimensions must match: a is {}x{}, b is {}x{}",
            a.rows, a.cols, b.rows, b.cols);
        assert!(a.rows == out.rows && a.cols == out.cols,
            "a and out dimensions must match: a is {}x{}, out is {}x{}",
            a.rows, a.cols, out.rows, out.cols);
        for i in 0..a.data.len() {
            out.data[i] = a.data[i] + b.data[i];
        }
    }

    pub fn sub(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        assert!(a.rows == b.rows && a.cols == b.cols,
            "a and b dimensions must match: a is {}x{}, b is {}x{}",
            a.rows, a.cols, b.rows, b.cols);
        assert!(a.rows == out.rows && a.cols == out.cols,
            "a and out dimensions must match: a is {}x{}, out is {}x{}",
            a.rows, a.cols, out.rows, out.cols);
        for i in 0..a.data.len() {
            out.data[i] = a.data[i] - b.data[i];
        }
    }

    /// Accumulating add: dst[i] += src[i]
    pub fn add_into(src: &Matrix, dst: &mut Matrix) {
        assert!(src.rows == dst.rows && src.cols == dst.cols,
            "src and dst dimensions must match: src is {}x{}, dst is {}x{}",
            src.rows, src.cols, dst.rows, dst.cols);
        for i in 0..src.data.len() {
            dst.data[i] += src.data[i];
        }
    }

    /// Accumulating subtract: dst[i] -= src[i]
    pub fn sub_into(src: &Matrix, dst: &mut Matrix) {
        assert!(src.rows == dst.rows && src.cols == dst.cols,
            "src and dst dimensions must match: src is {}x{}, dst is {}x{}",
            src.rows, src.cols, dst.rows, dst.cols);
        for i in 0..src.data.len() {
            dst.data[i] -= src.data[i];
        }
    }

    /// Dot product: Σ(a[i] * b[i])
    pub fn dot(a: &Matrix, b: &Matrix) -> f32 {
        assert!(a.data.len() == b.data.len(),
            "dot product requires same length: a has {}, b has {}",
            a.data.len(), b.data.len());
        let mut sum = 0.0;
        for i in 0..a.data.len() {
            sum += a.data[i] * b.data[i];
        }
        sum
    }

    pub fn mul(a: &Matrix, b: &Matrix, out: &mut Matrix, transpose_a: bool, transpose_b: bool) {
        // The idea behind tranpose is to avoid transposing the matrix in memory
        // Instead we just change the indexing position
        let a_rows = if transpose_a { a.cols } else { a.rows };
        let a_cols = if transpose_a { a.rows } else { a.cols };

        let b_rows = if transpose_b { b.cols } else { b.rows };
        let b_cols = if transpose_b { b.rows } else { b.cols };

        assert!(a_cols == b_rows,
            "inner dimensions must match for multiplication: a is {}x{}, b is {}x{}",
            a_rows, a_cols, b_rows, b_cols);
        assert!(out.rows == a_rows && out.cols == b_cols,
            "out dimensions must match result: expected {}x{}, got {}x{}",
            a_rows, b_cols, out.rows, out.cols);

        // Clear output buffer since inner loops use += to accumulate.
        // Without this, repeated calls would accumulate on previous results.
        // Note: if you need C = A1*B1 + A2*B2 (accumulating multiple products),
        // you'd need a separate mul_accumulate() that skips this clear.
        out.clear();

        // The match is more idiomatic Rust and makes the intent clearer without bit unpacking
        match (transpose_a, transpose_b) {
            (false, false) => { Self::mul_nn(a, b, out) }
            (false, true)  => { Self::mul_nt(a, b, out) }
            (true, false)  => { Self::mul_tn(a, b, out) }
            (true, true)   => { Self::mul_tt(a, b, out) }
        }
    }

    // Private helpers (no`pub`)
    fn mul_nn(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        for i in 0..out.cols {
            for j in 0..out.rows {
                for k in 0..a.cols {
                    out.data[j + i * out.cols] +=
                        a.data[k + i * a.cols] * 
                        b.data[j + k * b.cols];
                }
            }
        }
    }
    fn mul_nt(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        for i in 0..out.rows {
            for j in 0..out.cols {
                for k in 0..a.cols {
                    out.data[j + i * out.cols] +=
                        a.data[k + i * a.cols] * 
                        b.data[k + j * b.cols];
                }
            }
        }
    }
    fn mul_tn(a: &Matrix, b: &Matrix, out: &mut Matrix) {
       for k in 0..a.rows {
            for i in 0..out.rows {
                for j in 0..out.cols {
                    out.data[j + i * out.cols] +=
                        a.data[i + k * a.cols] * 
                        b.data[j + k * b.cols];
                }
            }
        } 
    }
    fn mul_tt(a: &Matrix, b: &Matrix, out: &mut Matrix) {
        for i in 0..out.rows {
            for j in 0..out.cols {
                for k in 0..a.rows {
                    out.data[j + i * out.cols] +=
                        a.data[i + k * a.cols] * 
                        b.data[k + j * b.cols];
                }
            }
        }
    }

    pub fn relu(a: &Matrix, out: &mut Matrix) {
        assert!(a.rows == out.rows && a.cols == out.cols,
            "a and out dimensions must match: a is {}x{}, out is {}x{}",
            a.rows, a.cols, out.rows, out.cols);
        
        for i in 0..a.data.len() {
            out.data[i] = a.data[i].max(0.0);
        }
    }

    // TODO: compute softmax per row instead of entire matrix
    pub fn softmax(a: &Matrix, out: &mut Matrix) {
        // o_i = e^a_i / sum(e^a_j)
        assert!(a.rows == out.rows && a.cols == out.cols,
            "a and out dimensions must match: a is {}x{}, out is {}x{}",
            a.rows, a.cols, out.rows, out.cols);
        let mut sum = 0.0;
        for i in 0..a.data.len() {
            out.data[i] = a.data[i].exp();
            sum += out.data[i];
        }
        Matrix::scale(out, 1.0 / sum);
    }

    pub fn cross_entropy(p: &Matrix, q: &Matrix, out: &mut Matrix) {
        assert!(p.rows == q.rows && p.cols == q.cols,
            "p and q dimensions must match: p is {}x{}, q is {}x{}",
            p.rows, p.cols, q.rows, q.cols);
        assert!(p.rows == out.rows && p.cols == out.cols,
            "p and out dimensions must match: p is {}x{}, out is {}x{}",
            p.rows, p.cols, out.rows, out.cols);

        // TODO: maybe not technically correct

        // Formula for a single sample
        for i in 0..out.data.len() {
            if p.data[i] == 0.0 {
                out.data[i] = 0.0;
            }
            else {
                out.data[i] = p.data[i] * -q.data[i].ln();
            }
        }
    }

    pub fn load(rows: usize, cols: usize, filename: &str) -> Self {
        // expect unwraps a Result or Option
        // let bytes = fs::read(filename).expect("failed to read file: {filename");

        // assert!(bytes.len() == rows * cols * 4,
        //   "file size mismatch: expected {} bytes, got {}",
        //   rows * cols * 4, bytes.len());

        // let data: Vec<f32> = bytes
        //     .chunks_exact(4)
        //     .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        //     .collect();

        let mut data = vec![0.0_f32; rows * cols];
        let mut file = File::open(filename).expect("failed to open file");

        unsafe {
            let buf = std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                data.len() * 4
            );
            file.read_exact(buf).expect("failed to read file");
        }
        // let mut reader = BufReader::new(File::open(filename).expect("failed to open"));
        // let mut data = Vec::with_capacity(rows * cols);
        // let mut buf = [0u8; 4];

        // while reader.read_exact(&mut buf).is_ok() {
        //     data.push(f32::from_le_bytes(buf));
        // }

        // assert_eq!(data.len(), rows * cols, "file size mismatch");

        Self { rows, cols, data}
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
    fn test_add_into() {
        let src = Matrix::zeros(2, 2).filled(3.0);
        let mut dst = Matrix::zeros(2, 2).filled(5.0);
        Matrix::add_into(&src, &mut dst);
        assert!(dst.data.iter().all(|&x| x == 8.0));

        // Verify accumulation works (call again)
        Matrix::add_into(&src, &mut dst);
        assert!(dst.data.iter().all(|&x| x == 11.0));
    }

    #[test]
    fn test_sub_into() {
        let src = Matrix::zeros(2, 2).filled(3.0);
        let mut dst = Matrix::zeros(2, 2).filled(10.0);
        Matrix::sub_into(&src, &mut dst);
        assert!(dst.data.iter().all(|&x| x == 7.0));

        // Verify accumulation works (call again)
        Matrix::sub_into(&src, &mut dst);
        assert!(dst.data.iter().all(|&x| x == 4.0));
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

    #[test]
    fn test_rand() {
        let m = Matrix::rand(10, 10);
        assert_eq!(m.data.len(), 100);
        // All values should be in [0, 1)
        assert!(m.data.iter().all(|&x| x >= 0.0 && x < 1.0));
        // Should have some variance (not all same value)
        let first = m.data[0];
        assert!(m.data.iter().any(|&x| x != first));
    }

    #[test]
    fn test_rand_scaled() {
        let m = Matrix::rand_scaled(10, 10, 0.1);
        assert_eq!(m.data.len(), 100);
        // All values should be in [-0.1, 0.1]
        assert!(m.data.iter().all(|&x| x >= -0.1 && x <= 0.1));
    }

    #[test]
    fn test_clear() {
        let mut m = Matrix::full(2, 2, 5.0);
        m.clear();
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_scale() {
        let mut m = Matrix::full(2, 2, 3.0);
        m.scale(2.0);
        assert!(m.data.iter().all(|&x| x == 6.0));
    }

    #[test]
    fn test_sum() {
        let m = Matrix::full(2, 2, 3.0);
        assert_eq!(m.sum(), 12.0);
    }

    #[test]
    fn test_dot() {
        let a = Matrix { rows: 1, cols: 4, data: vec![1.0, 2.0, 3.0, 4.0] };
        let b = Matrix { rows: 1, cols: 4, data: vec![2.0, 3.0, 4.0, 5.0] };
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_eq!(Matrix::dot(&a, &b), 40.0);
    }

    #[test]
    #[should_panic(expected = "a and b dimensions must match")]
    fn test_add_mismatched_a_b() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(3, 3);
        let mut out = Matrix::zeros(2, 2);
        Matrix::add(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "a and out dimensions must match")]
    fn test_add_mismatched_a_out() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(2, 2);
        let mut out = Matrix::zeros(3, 3);
        Matrix::add(&a, &b, &mut out);
    }

    #[test]
    fn test_sub() {
        let a = Matrix::full(2, 2, 5.0);
        let b = Matrix::full(2, 2, 2.0);
        let mut out = Matrix::zeros(2, 2);
        Matrix::sub(&a, &b, &mut out);
        assert!(out.data.iter().all(|&x| x == 3.0));
    }

    #[test]
    #[should_panic(expected = "a and b dimensions must match")]
    fn test_sub_mismatched_a_b() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(3, 3);
        let mut out = Matrix::zeros(2, 2);
        Matrix::sub(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "a and out dimensions must match")]
    fn test_sub_mismatched_a_out() {
        let a = Matrix::zeros(2, 2);
        let b = Matrix::zeros(2, 2);
        let mut out = Matrix::zeros(3, 3);
        Matrix::sub(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "inner dimensions must match for multiplication")]
    fn test_mul_mismatched_inner() {
        let a = Matrix::zeros(2, 3);  // 2x3
        let b = Matrix::zeros(4, 2);  // 4x2 - inner dims 3 != 4
        let mut out = Matrix::zeros(2, 2);
        Matrix::mul(&a, &b, &mut out, false, false);
    }

    #[test]
    #[should_panic(expected = "out dimensions must match result")]
    fn test_mul_mismatched_out() {
        let a = Matrix::zeros(2, 3);  // 2x3
        let b = Matrix::zeros(3, 4);  // 3x4 - result should be 2x4
        let mut out = Matrix::zeros(2, 2);  // wrong size
        Matrix::mul(&a, &b, &mut out, false, false);
    }

    #[test]
    fn test_mul_nn() {
        // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
        // [3, 4] x [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
        let a = Matrix { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let b = Matrix { rows: 2, cols: 2, data: vec![5.0, 6.0, 7.0, 8.0] };
        let mut out = Matrix::zeros(2, 2);
        Matrix::mul(&a, &b, &mut out, false, false);
        assert_eq!(out.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_mul_nt() {
        // A (2x3) × Bᵀ (3x2) = (2x2)
        // A = [1, 2, 3]    B = [7,  8,  9]     Bᵀ = [7, 10]
        //     [4, 5, 6]        [10, 11, 12]         [8, 11]
        //                                          [9, 12]
        // Result:
        // [1,2,3]·[7,8,9]   = 50    [1,2,3]·[10,11,12] = 68
        // [4,5,6]·[7,8,9]   = 122   [4,5,6]·[10,11,12] = 167
        let a = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        let b = Matrix { rows: 2, cols: 3, data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0] };
        let mut out = Matrix::zeros(2, 2);
        Matrix::mul(&a, &b, &mut out, false, true);
        assert_eq!(out.data, vec![50.0, 68.0, 122.0, 167.0]);
    }

    #[test]
    fn test_mul_tn() {
        // Aᵀ (3x2) × B (2x2) = (3x2)
        // A = [1, 2, 3]    Aᵀ = [1, 4]    B = [7, 8]
        //     [4, 5, 6]         [2, 5]        [9, 10]
        //                       [3, 6]
        // Result:
        // [1,4]·[7,9]=43   [1,4]·[8,10]=48
        // [2,5]·[7,9]=59   [2,5]·[8,10]=66
        // [3,6]·[7,9]=75   [3,6]·[8,10]=84
        let a = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        let b = Matrix { rows: 2, cols: 2, data: vec![7.0, 8.0, 9.0, 10.0] };
        let mut out = Matrix::zeros(3, 2);
        Matrix::mul(&a, &b, &mut out, true, false);
        assert_eq!(out.data, vec![43.0, 48.0, 59.0, 66.0, 75.0, 84.0]);
    }

    #[test]
    fn test_mul_tt() {
        // Aᵀ (3x2) × Bᵀ (2x3) = (3x3)
        // A = [1, 2, 3]    Aᵀ = [1, 4]    B = [7,  8]     Bᵀ = [7, 9, 11]
        //     [4, 5, 6]         [2, 5]        [9,  10]         [8, 10, 12]
        //                       [3, 6]        [11, 12]
        // Result:
        // [1,4]·[7,8]=39   [1,4]·[9,10]=49   [1,4]·[11,12]=59
        // [2,5]·[7,8]=54   [2,5]·[9,10]=68   [2,5]·[11,12]=82
        // [3,6]·[7,8]=69   [3,6]·[9,10]=87   [3,6]·[11,12]=105
        let a = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        let b = Matrix { rows: 3, cols: 2, data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0] };
        let mut out = Matrix::zeros(3, 3);
        Matrix::mul(&a, &b, &mut out, true, true);
        assert_eq!(out.data, vec![39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0]);
    }

    #[test]
    fn test_relu() {
        let a = Matrix { rows: 2, cols: 3, data: vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] };
        let mut out = Matrix::zeros(2, 3);
        Matrix::relu(&a, &mut out);
        assert_eq!(out.data, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_softmax() {
        // Single row for now (TODO: test per-row softmax later)
        let a = Matrix { rows: 1, cols: 3, data: vec![1.0, 2.0, 3.0] };
        let mut out = Matrix::zeros(1, 3);
        Matrix::softmax(&a, &mut out);

        // Should sum to 1.0
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Larger input → larger probability
        assert!(out.data[2] > out.data[1]);
        assert!(out.data[1] > out.data[0]);
    }

    #[test]
    fn test_cross_entropy() {
        // p = one-hot label (true class is 1)
        // q = predicted probabilities
        let p = Matrix { rows: 1, cols: 3, data: vec![0.0, 1.0, 0.0] };
        let q = Matrix { rows: 1, cols: 3, data: vec![0.1, 0.7, 0.2] };
        let mut out = Matrix::zeros(1, 3);
        Matrix::cross_entropy(&p, &q, &mut out);

        // Element-wise: p[i] * -ln(q[i])
        // out[0] = 0 * -ln(0.1) = 0
        // out[1] = 1 * -ln(0.7) ≈ 0.357
        // out[2] = 0 * -ln(0.2) = 0
        assert_eq!(out.data[0], 0.0);
        assert!((out.data[1] - (-0.7_f32.ln())).abs() < 1e-6);
        assert_eq!(out.data[2], 0.0);

        // Cross-entropy loss = sum ≈ 0.357
        let loss = out.sum();
        assert!((loss - (-0.7_f32.ln())).abs() < 1e-6);
    }
}
