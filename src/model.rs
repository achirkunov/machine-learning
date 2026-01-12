//! Computational graph for automatic differentiation.
//!
//! Design choices:
//! - Op enum carries inputs (no C-style sentinel enum + separate array)
//! - VarId indices instead of pointers (borrow-checker friendly)
//! - Option<Matrix> for grad instead of nullable pointer
//!
//! Two approaches for execution order:
//! 1. Explicit programs: store Vec<VarId> for forward/cost passes
//! 2. Implicit ranges: if vars are in topological order, use 0..=output or 0..len()
//!
//! This implementation uses explicit programs for flexibility.
//! A ModelBuilder (not yet implemented) would construct the graph
//! and guarantee topological ordering.

use crate::matrix::{self, Matrix};

type VarId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarKind {
    Input,
    Parameter,
    Intermediate,
    Output,
    DesiredOutput,
    Cost,
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Input,
    Parameter,
    ReLU(VarId),
    Softmax(VarId),
    Add(VarId, VarId),
    Sub(VarId, VarId),
    MatMul(VarId, VarId),
    CrossEntropy(VarId, VarId),
}

// Variable
struct Var {
    // no index - use position in Vec
    val: Matrix,
    grad: Option<Matrix>, // None if !requires_grad, can you grad.is_some()
    op: Op,               // Op enum carriers inputs as indices
    kind: VarKind,
}

// struct Tape {
//     vars: Vec<Var>,
// }

type Program = Vec<VarId>;

pub struct ModelContext {
    vars: Vec<Var>,

    input: VarId,
    output: VarId,
    target: VarId,
    loss: VarId,

    forward_prog: Program,
    cost_prog: Program,
}

/// Builder for constructing a computational graph.
/// Ensures topological ordering by construction.
pub struct ModelBuilder {
    vars: Vec<Var>,
}

pub struct ModelTrainingDesc {
    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self { vars: Vec::new() }
    }

    /// Add an input variable (data fed from outside, no gradient)
    pub fn input(&mut self, rows: usize, cols: usize) -> VarId {
        self.push(Matrix::zeros(rows, cols), None, Op::Input, VarKind::Input)
    }

    /// Add a trainable parameter (random init, has gradient)
    pub fn parameter(&mut self, rows: usize, cols: usize) -> VarId {
        self.push_with_grad(rows, cols, Op::Parameter, VarKind::Parameter)
    }

    /// Matrix multiplication: result = a @ b
    pub fn matmul(&mut self, a: VarId, b: VarId) -> VarId {
        let a_rows = self.vars[a as usize].val.rows;
        let b_cols = self.vars[b as usize].val.cols;
        self.add_var(
            Matrix::zeros(a_rows, b_cols),
            Some(Matrix::zeros(a_rows, b_cols)),
            Op::MatMul(a, b),
            VarKind::Intermediate,
        )
    }

    /// ReLU activation
    pub fn relu(&mut self, x: VarId) -> VarId {
        self.unary(x, Op::ReLU(x), VarKind::Intermediate)
    }

    /// Softmax activation
    pub fn softmax(&mut self, x: VarId) -> VarId {
        self.unary(x, Op::Softmax(x), VarKind::Intermediate)
    }

    /// Cross-entropy loss
    pub fn cross_entropy(&mut self, pred: VarId, target: VarId) -> VarId {
        self.push_with_grad(1, 1, Op::CrossEntropy(pred, target), VarKind::Cost)
    }

    /// Element-wise addition: result = a + b
    pub fn add(&mut self, a: VarId, b: VarId) -> VarId {
        let rows = self.vars[a as usize].val.rows;
        let cols = self.vars[a as usize].val.cols;
        // TODO: assert a and b have same shape
        self.add_var(
            Matrix::zeros(rows, cols),
            Some(Matrix::zeros(rows, cols)),
            Op::Add(a, b),
            VarKind::Intermediate,
        )
    }

    /// Element-wise subtraction: result = a - b
    pub fn sub(&mut self, a: VarId, b: VarId) -> VarId {
        let rows = self.vars[a as usize].val.rows;
        let cols = self.vars[a as usize].val.cols;
        // TODO: assert a and b have same shape
        self.add_var(
            Matrix::zeros(rows, cols),
            Some(Matrix::zeros(rows, cols)),
            Op::Sub(a, b),
            VarKind::Intermediate,
        )
    }

    fn add_var(&mut self, val: Matrix, grad: Option<Matrix>, op: Op, kind: VarKind) -> VarId {
        let id = self.vars.len() as VarId;
        self.vars.push(Var { val, grad, op, kind });
        id
    }

    fn push(&mut self, val: Matrix, grad: Option<Matrix>, op: Op, kind: VarKind) -> VarId {
        let id = self.vars.len() as VarId;
        self.vars.push(Var { val, grad, op, kind});
        id
    }

    fn push_with_grad(&mut self, rows: usize, cols: usize, op: Op, kind: VarKind) -> VarId {
        self.push(
            Matrix::zeros(rows, cols),
            Some(Matrix::zeros(rows, cols)),
            op,
            kind,
        )
    }

    fn unary(&mut self, x: VarId, op: Op, kind: VarKind) -> VarId {
        // unary ops preserve shape - the output of relu(x) has the same dimension as x
        let rows = self.vars[x as usize].val.rows;
        let cols = self.vars[x as usize].val.cols;
        self.push_with_grad(rows, cols, op, kind)
    }

    /// Build the final model context
    pub fn build(
        self,
        input: VarId,
        output: VarId,
        target: VarId,
        loss: VarId,
    ) -> ModelContext {
        // For now, forward_prog and cost_prog are all vars in order
        // (works because we build in topological order)
        let all_ids: Vec<VarId> = (0..self.vars.len() as VarId).collect();

        ModelContext {
            vars: self.vars,
            input,
            output,
            target,
            loss,
            forward_prog: all_ids.clone(),
            cost_prog: all_ids,
        }
    }


}

impl ModelContext {

    // Copy data into input buffer
    pub fn set_input(&mut self, data: &Matrix) {
        // TODO: avoid copy_into?
        let input_idx = self.input as usize;
        matrix::Matrix::copy_into(&data, &mut self.vars[input_idx].val);
    }

    /// Direct access to input buffer (for zero-copy data loading
    /// Callers who can write directly straight into buffer
    pub fn input_buffer_mut(&mut self) -> &mut Matrix {
        &mut self.vars[self.input as usize].val
    }

    pub fn output(&self) -> &Matrix {
        &self.vars[self.output as usize].val
    }


    pub fn forward(&mut self) {
        // TODO: we will need to compute topological order once at build time
        // TODO: I don't like clone, is there a better way?
        for &id in &self.forward_prog.clone() {
            self.compute(id);
        }
    }

    // Helper function
    fn compute(&mut self, id: VarId) {
        let idx = id as usize;
        match self.vars[idx].op {
            Op::Input | Op::Parameter => {
                // No computation needed
            }
            Op::ReLU(x) => {
                // Topological order ensure inputs < output index
                let (inputs, outputs) = self.vars.split_at_mut(idx);
                let src = &inputs[x as usize].val;
                let dst = &mut outputs[0].val;
                Matrix::relu(src, dst);
            },
            Op::Add(a, b) => {
                let (inputs, outputs) = self.vars.split_at_mut(idx);
                let lhs = &inputs[a as usize].val;
                let rhs = &inputs[b as usize].val;
                let out = &mut outputs[0].val;
                Matrix::add(lhs, rhs, out);
            }
            Op::Softmax(x) => {
                let (inputs, outputs) = self.vars.split_at_mut(idx);
                let src = &inputs[x as usize].val;
                let dst = &mut outputs[0].val;
                Matrix::softmax(src, dst);
            }
            Op::Sub(a, b) => {
                let (inputs, outputs) = self.vars.split_at_mut(idx);
                let lhs = &inputs[a as usize].val;
                let rhs = &inputs[b as usize].val;
                let out = &mut outputs[0].val;
                Matrix::sub(lhs, rhs, out);
            }
            Op::MatMul(a, b) => {
                let (inputs, outputs) = self.vars.split_at_mut(idx);
                let lhs = &inputs[a as usize].val;
                let rhs = &inputs[b as usize].val;
                let out = &mut outputs[0].val;
                Matrix::mul(lhs, rhs, out, false, false);
            }
            Op::CrossEntropy(pred, target) => {
                let (inputs, outputs) = self.vars.split_at_mut(idx);
                let p = &inputs[pred as usize].val;
                let t = &inputs[target as usize].val;
                // Cross entropy output is 1x1 scalar, sum element-wise results
                let mut temp = Matrix::zeros(p.rows, p.cols);
                Matrix::cross_entropy(t, p, &mut temp);
                outputs[0].val.data[0] = temp.sum();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_ids_are_sequential() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 784);
        let w = b.parameter(784, 10);
        let y = b.matmul(x, w);

        assert_eq!(x, 0);
        assert_eq!(w, 1);
        assert_eq!(y, 2);
    }

    #[test]
    fn matmul_computes_output_dimensions() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 784); // 1x784
        let w = b.parameter(784, 10); // 784x10
        let y = b.matmul(x, w); // should be 1x10

        let var = &b.vars[y as usize];
        assert_eq!(var.val.rows, 1);
        assert_eq!(var.val.cols, 10);
    }

    #[test]
    fn ops_reference_correct_inputs() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 10);
        let w = b.parameter(10, 5);
        let y = b.matmul(x, w);

        match b.vars[y as usize].op {
            Op::MatMul(a, b_id) => {
                assert_eq!(a, x);
                assert_eq!(b_id, w);
            }
            _ => panic!("expected MatMul"),
        }
    }

    #[test]
    fn var_kinds_are_correct() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 10);
        let w = b.parameter(10, 5);
        let y = b.matmul(x, w);

        assert_eq!(b.vars[x as usize].kind, VarKind::Input);
        assert_eq!(b.vars[w as usize].kind, VarKind::Parameter);
        assert_eq!(b.vars[y as usize].kind, VarKind::Intermediate);
    }

    #[test]
    fn inputs_have_no_gradient() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 10);
        let w = b.parameter(10, 5);

        assert!(b.vars[x as usize].grad.is_none());
        assert!(b.vars[w as usize].grad.is_some());
    }

    #[test]
    fn build_creates_valid_context() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 784);
        let w = b.parameter(784, 10);
        let logits = b.matmul(x, w);
        let y = b.input(1, 10);
        let loss = b.cross_entropy(logits, y);

        let model = b.build(x, logits, y, loss);

        assert_eq!(model.vars.len(), 5);
        assert_eq!(model.input, 0);
        assert_eq!(model.output, 2);
        assert_eq!(model.target, 3);
       
        assert_eq!(model.loss, 4);
    }

    #[test]
    fn test_compute_relu() {
        let mut b = ModelBuilder::new();
        let x = b.input(2 ,3);
        let y = b.relu(x);
        let mut m = b.build(x, y, x, y); // dummy

        let a = Matrix { rows: 2, cols: 3, data: vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0] };
        let input = m.set_input(&a);
        m.forward();

        let out = m.output();
        assert_eq!(out.data, &[0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    }

    #[test]
    fn test_compute_add() {
        let mut b = ModelBuilder::new();
        let x = b.input(2, 3);
        let y = b.input(2, 3);
        let z = b.add(x, y);
        let mut m = b.build(x, z, x, z); // dummy target/loss

        // Set both inputs
        m.vars[0].val = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        m.vars[1].val = Matrix { rows: 2, cols: 3, data: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0] };
        m.forward();

        let out = m.output();
        assert_eq!(out.data, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn test_compute_softmax() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 3);
        let y = b.softmax(x);
        let mut m = b.build(x, y, x, y);

        m.vars[0].val = Matrix { rows: 1, cols: 3, data: vec![1.0, 2.0, 3.0] };
        m.forward();

        let out = m.output();
        // Softmax: e^x_i / sum(e^x_j)
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1.0");
        // Larger input -> larger probability
        assert!(out.data[2] > out.data[1]);
        assert!(out.data[1] > out.data[0]);
    }

    #[test]
    fn test_compute_sub() {
        let mut b = ModelBuilder::new();
        let x = b.input(2, 3);
        let y = b.input(2, 3);
        let z = b.sub(x, y);
        let mut m = b.build(x, z, x, z);

        m.vars[0].val = Matrix { rows: 2, cols: 3, data: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0] };
        m.vars[1].val = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        m.forward();

        let out = m.output();
        assert_eq!(out.data, vec![9.0, 18.0, 27.0, 36.0, 45.0, 54.0]);
    }

    #[test]
    fn test_compute_matmul() {
        let mut b = ModelBuilder::new();
        let x = b.input(2, 3);
        let w = b.input(3, 2);
        let y = b.matmul(x, w);
        let mut m = b.build(x, y, x, y);

        // [1, 2, 3]   [7,  8]    [1*7+2*9+3*11, 1*8+2*10+3*12]   [58,  64]
        // [4, 5, 6] x [9, 10] =  [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        //             [11,12]
        m.vars[0].val = Matrix { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        m.vars[1].val = Matrix { rows: 3, cols: 2, data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0] };
        m.forward();

        let out = m.output();
        assert_eq!(out.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_compute_cross_entropy() {
        let mut b = ModelBuilder::new();
        let pred = b.input(1, 3);
        let target = b.input(1, 3);
        let loss = b.cross_entropy(pred, target);
        let mut m = b.build(pred, pred, target, loss);

        // pred = softmax-like probabilities
        m.vars[0].val = Matrix { rows: 1, cols: 3, data: vec![0.1, 0.7, 0.2] };
        // target = one-hot (true class is index 1)
        m.vars[1].val = Matrix { rows: 1, cols: 3, data: vec![0.0, 1.0, 0.0] };
        m.forward();

        // Cross entropy = -sum(target * ln(pred)) = -ln(0.7) ≈ 0.357
        let loss_val = m.vars[m.loss as usize].val.data[0];
        let expected = -0.7_f32.ln();
        assert!((loss_val - expected).abs() < 1e-6);
    }
}
