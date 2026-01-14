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

use crate::matrix::Matrix;

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

    /// Add a trainable parameter (Xavier random init, has gradient)
    /// TODO: move this to an init_parameters method with a selectable strategy + bias
    pub fn parameter(&mut self, rows: usize, cols: usize) -> VarId {
        // Xavier initialization: scale = sqrt(6 / (fan_in + fan_out))
        let scale = (6.0 / (rows + cols) as f32).sqrt();
        let val = Matrix::rand_scaled(rows, cols, scale);
        let grad = Matrix::zeros(rows, cols);
        self.push(val, Some(grad), Op::Parameter, VarKind::Parameter)
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
        self.vars.push(Var {
            val,
            grad,
            op,
            kind,
        });
        id
    }

    fn push(&mut self, val: Matrix, grad: Option<Matrix>, op: Op, kind: VarKind) -> VarId {
        let id = self.vars.len() as VarId;
        self.vars.push(Var {
            val,
            grad,
            op,
            kind,
        });
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
    pub fn build(self, input: VarId, output: VarId, target: VarId, loss: VarId) -> ModelContext {
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
        Matrix::copy_into(data, &mut self.vars[input_idx].val);
    }

    /// Set target labels for loss computation
    pub fn set_target(&mut self, data: &Matrix) {
        let target_idx = self.target as usize;
        Matrix::copy_into(data, &mut self.vars[target_idx].val);
    }

    /// Direct access to input buffer (for zero-copy data loading
    /// Callers who can write directly straight into buffer
    pub fn input_buffer_mut(&mut self) -> &mut Matrix {
        &mut self.vars[self.input as usize].val
    }

    pub fn output(&self) -> &Matrix {
        &self.vars[self.output as usize].val
    }

    /// Get the current loss value (scalar)
    pub fn loss(&self) -> f32 {
        self.vars[self.loss as usize].val.data[0]
    }

    /// SGD update: param = param - learning_rate * grad
    pub fn sgd_step(&mut self, learning_rate: f32) {
        for var in &mut self.vars {
            if var.kind == VarKind::Parameter {
                if let Some(ref grad) = var.grad {
                    for i in 0..var.val.data.len() {
                        var.val.data[i] -= learning_rate * grad.data[i];
                    }
                }
            }
        }
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
            }
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

    pub fn zero_grad(&mut self) {
        for var in &mut self.vars {
            // ref mut: creates a mutable reference when pattern matching.
            // Without it, Some(g) would try to move the Matrix out of the Option.
            // With ref mut, g is &mut Matrix and we can modify it in place.
            if let Some(ref mut g) = var.grad {
                g.clear();
            }
        }
    }

    pub fn backward(&mut self) {
        // 0. Zero grads (why?)
        self.zero_grad();

        // 1. Seed: dL/dLoss = 1.0
        // .as_mut(): converts Option<Matrix> to Option<&mut Matrix>
        //            (lets us get a mutable ref without moving ownership)
        // .unwrap(): extracts the inner value, panics if None
        //            (safe here because we know loss has a gradient)
        self.vars[self.loss as usize]
            .grad
            .as_mut()
            .unwrap()
            .fill(1.0);

        // 2. Iterate in reverse topological order
        // We need to use .clone(), because Rust cannot tell than compute_grad doesn't modify forward_prog
        for &id in self.forward_prog.clone().iter().rev() {
            self.compute_grad(id);
        }
    }

    fn compute_grad(&mut self, id: VarId) {
        let idx = id as usize;
        if self.vars[idx].grad.is_none() {
            return; // Skip nodes without gradients
        }
        match self.vars[idx].op {
            Op::Input | Op::Parameter => {
                // Leaf nodes: nothing to propagate backward
            }
            Op::Add(a, b) => {
                // split_at_mut(idx): splits Vec into two mutable slices at position idx.
                // This lets us hold mutable refs to different parts simultaneously,
                // which the borrow checker normally forbids for the same Vec.
                // inputs = &mut vars[0..idx], current_and_rest = &mut vars[idx..]
                let (inputs, current_and_rest) = self.vars.split_at_mut(idx);

                // .as_ref(): converts Option<Matrix> to Option<&Matrix>
                //            (we only need to read current_grad, not modify it)
                // .unwrap(): safe because we checked grad.is_some() at function start
                let current_grad = current_and_rest[0].grad.as_ref().unwrap();

                // ref mut in pattern: creates &mut Matrix without moving out of Option
                if let Some(ref mut grad_a) = inputs[a as usize].grad {
                    Matrix::add_into(current_grad, grad_a);
                }
                if let Some(ref mut grad_b) = inputs[b as usize].grad {
                    Matrix::add_into(current_grad, grad_b);
                }
            }
            Op::Sub(a, b) => {
                let (inputs, current_and_rest) = self.vars.split_at_mut(idx);
                let current_grad = current_and_rest[0].grad.as_ref().unwrap();
                if let Some(ref mut grad_a) = inputs[a as usize].grad {
                    Matrix::add_into(current_grad, grad_a);
                }
                if let Some(ref mut grad_b) = inputs[b as usize].grad {
                    Matrix::sub_into(current_grad, grad_b);
                }
            }
            Op::MatMul(a, b) => {
                let (inputs, current_and_rest) = self.vars.split_at_mut(idx);
                let current_grad = current_and_rest[0].grad.as_ref().unwrap();
                let a_mat = &inputs[a as usize].val; // we need the ref?
                let b_mat = &inputs[b as usize].val;
                if let Some(ref mut grad_a) = inputs[a as usize].grad {
                    let mut temp = Matrix::zeros(grad_a.rows, grad_a.cols);
                    Matrix::mul(current_grad, b_mat, &mut temp, false, true);
                    Matrix::add_into(&temp, grad_a);
                }
                if let Some(ref mut grad_b) = inputs[b as usize].grad {
                    let mut temp = Matrix::zeros(grad_b.rows, grad_b.cols);
                    Matrix::mul(a_mat, current_grad, &mut temp, true, false);
                    Matrix::add_into(&temp, grad_b);
                }
            }
            Op::ReLU(x) => {
                // dL/dx = dL/dy ⊙ (x > 0 ? 1 : 0)
                // Gradient passes through where input was positive, zero otherwise
                let (inputs, current_and_rest) = self.vars.split_at_mut(idx);
                let current_grad = current_and_rest[0].grad.as_ref().unwrap();
                let input_val = &inputs[x as usize].val;
                if let Some(ref mut grad_x) = inputs[x as usize].grad {
                    for i in 0..grad_x.data.len() {
                        if input_val.data[i] > 0.0 {
                            grad_x.data[i] += current_grad.data[i];
                        }
                    }
                }
            }
            Op::Softmax(x) => {
                // y_i = exp(x_i) / Σ exp(x_j)
                // Jacobian: ∂y_i/∂x_j = y_i(δ_ij - y_j)
                // Chain rule: dL/dx = y ⊙ (dL/dy - dot) where dot = Σ(dL/dy_i · y_i)
                let (inputs, current_and_rest) = self.vars.split_at_mut(idx);
                let current_grad = current_and_rest[0].grad.as_ref().unwrap(); // dL/dy
                let y = &current_and_rest[0].val; // softmax output

                if let Some(ref mut grad_x) = inputs[x as usize].grad {
                    let dot = Matrix::dot(current_grad, y); // Σ(dL/dy_i · y_i)

                    // dL/dx_j = y_j · (dL/dy_j - dot)
                    for j in 0..grad_x.data.len() {
                        grad_x.data[j] += y.data[j] * (current_grad.data[j] - dot);
                    }
                }
            }
            Op::CrossEntropy(pred, target) => {
                // L = Σ(-target_i · ln(pred_i))
                // dL/dpred_i = -target_i / pred_i
                let (inputs, current_and_rest) = self.vars.split_at_mut(idx);
                let loss_grad = current_and_rest[0].grad.as_ref().unwrap().data[0]; // scalar
                let pred_val = &inputs[pred as usize].val;
                let target_val = &inputs[target as usize].val;

                if let Some(ref mut grad_pred) = inputs[pred as usize].grad {
                    for i in 0..grad_pred.data.len() {
                        if pred_val.data[i] != 0.0 {
                            grad_pred.data[i] +=
                                loss_grad * (-target_val.data[i] / pred_val.data[i]);
                        }
                    }
                }
                // target is labels, typically no gradient needed
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
        let x = b.input(2, 3);
        let y = b.relu(x);
        let mut m = b.build(x, y, x, y); // dummy

        let a = Matrix {
            rows: 2,
            cols: 3,
            data: vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0],
        };
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
        m.vars[0].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        m.vars[1].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        };
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

        m.vars[0].val = Matrix {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
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

        m.vars[0].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        };
        m.vars[1].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
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
        m.vars[0].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        m.vars[1].val = Matrix {
            rows: 3,
            cols: 2,
            data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        };
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
        m.vars[0].val = Matrix {
            rows: 1,
            cols: 3,
            data: vec![0.1, 0.7, 0.2],
        };
        // target = one-hot (true class is index 1)
        m.vars[1].val = Matrix {
            rows: 1,
            cols: 3,
            data: vec![0.0, 1.0, 0.0],
        };
        m.forward();

        // Cross entropy = -sum(target * ln(pred)) = -ln(0.7) ≈ 0.357
        let loss_val = m.vars[m.loss as usize].val.data[0];
        let expected = -0.7_f32.ln();
        assert!((loss_val - expected).abs() < 1e-6);
    }

    #[test]
    fn test_zero_grad() {
        let mut b = ModelBuilder::new();
        let x = b.input(1, 3);
        let w = b.parameter(3, 2); // has gradient
        let y = b.matmul(x, w); // has gradient
        let mut m = b.build(x, y, x, y);

        // Set non-zero values in gradients
        m.vars[1].grad.as_mut().unwrap().fill(5.0); // w's grad
        m.vars[2].grad.as_mut().unwrap().fill(3.0); // y's grad

        // Verify they're non-zero
        assert!(m.vars[1]
            .grad
            .as_ref()
            .unwrap()
            .data
            .iter()
            .all(|&v| v == 5.0));
        assert!(m.vars[2]
            .grad
            .as_ref()
            .unwrap()
            .data
            .iter()
            .all(|&v| v == 3.0));

        // Zero gradients
        m.zero_grad();

        // Verify all gradients are zeroed
        assert!(m.vars[1]
            .grad
            .as_ref()
            .unwrap()
            .data
            .iter()
            .all(|&v| v == 0.0));
        assert!(m.vars[2]
            .grad
            .as_ref()
            .unwrap()
            .data
            .iter()
            .all(|&v| v == 0.0));

        // Input (x) should have no gradient
        assert!(m.vars[0].grad.is_none());
    }

    #[test]
    fn test_sgd_step() {
        // param = param - learning_rate * grad
        let mut b = ModelBuilder::new();
        let x = b.input(1, 2);
        let p = b.parameter(2, 2);
        let y = b.matmul(x, p);
        let mut m = b.build(x, y, x, y);

        // Set parameter values to known state
        m.vars[1].val.data = vec![1.0, 2.0, 3.0, 4.0];
        // Set gradients
        m.vars[1].grad.as_mut().unwrap().data = vec![0.5, 1.0, 1.5, 2.0];

        let lr = 0.1;
        m.sgd_step(lr);

        // Expected: [1.0 - 0.1*0.5, 2.0 - 0.1*1.0, 3.0 - 0.1*1.5, 4.0 - 0.1*2.0]
        //         = [0.95, 1.9, 2.85, 3.8]
        let expected = vec![0.95, 1.9, 2.85, 3.8];
        for (i, (&actual, &exp)) in m.vars[1].val.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "param[{}]: expected {}, got {}",
                i,
                exp,
                actual
            );
        }

        // Input should be unchanged (no gradient)
        // Intermediate (y) should be unchanged (not a parameter)
    }

    #[test]
    fn test_backward_add() {
        // Graph: z = x + p, where x is input (no grad), p is parameter (has grad)
        // After backward, p.grad should equal 1.0 (the seeded loss gradient)
        let mut b = ModelBuilder::new();
        let x = b.input(2, 3);
        let p = b.parameter(2, 3);
        let z = b.add(x, p);
        let mut m = b.build(x, z, x, z); // z is both output and "loss"

        // Set values
        m.vars[0].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        m.vars[1].val = Matrix {
            rows: 2,
            cols: 3,
            data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        };
        m.forward();

        // backward() seeds z.grad with 1.0, then propagates
        m.backward();

        // x has no gradient (input)
        assert!(m.vars[0].grad.is_none());

        // p.grad should be all 1.0 (gradient passed through unchanged from z)
        let p_grad = m.vars[1].grad.as_ref().unwrap();
        assert!(
            p_grad.data.iter().all(|&v| v == 1.0),
            "expected all 1.0, got {:?}",
            p_grad.data
        );
    }

    #[test]
    fn test_backward_add_two_params() {
        // Graph: z = p1 + p2, both are parameters
        // After backward, both p1.grad and p2.grad should equal 1.0
        let mut b = ModelBuilder::new();
        let x = b.input(1, 1); // dummy input required by build
        let p1 = b.parameter(2, 2);
        let p2 = b.parameter(2, 2);
        let z = b.add(p1, p2);
        let mut m = b.build(x, z, x, z);

        // Set parameter values
        m.vars[1].val = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        m.vars[2].val = Matrix {
            rows: 2,
            cols: 2,
            data: vec![10.0, 20.0, 30.0, 40.0],
        };
        m.forward();

        m.backward();

        // Both parameters should have gradient = 1.0
        let p1_grad = m.vars[1].grad.as_ref().unwrap();
        let p2_grad = m.vars[2].grad.as_ref().unwrap();
        assert!(
            p1_grad.data.iter().all(|&v| v == 1.0),
            "p1 grad: expected all 1.0, got {:?}",
            p1_grad.data
        );
        assert!(
            p2_grad.data.iter().all(|&v| v == 1.0),
            "p2 grad: expected all 1.0, got {:?}",
            p2_grad.data
        );
    }

    #[test]
    fn test_backward_matmul() {
        // Graph: Z = A × B, where A and B are parameters
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // Z = [[19, 22], [43, 50]]
        //
        // With dL/dZ = [[1, 1], [1, 1]] (loss gradient seeded as 1.0):
        // dL/dA = dL/dZ × B^T = [[1,1],[1,1]] × [[5,7],[6,8]] = [[11, 15], [11, 15]]
        // dL/dB = A^T × dL/dZ = [[1,3],[2,4]] × [[1,1],[1,1]] = [[4, 4], [6, 6]]
        let mut b = ModelBuilder::new();
        let x = b.input(1, 1); // dummy input
        let a = b.parameter(2, 2);
        let param_b = b.parameter(2, 2);
        let z = b.matmul(a, param_b);
        let mut m = b.build(x, z, x, z);

        // Set parameter values
        m.vars[1].val = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        m.vars[2].val = Matrix {
            rows: 2,
            cols: 2,
            data: vec![5.0, 6.0, 7.0, 8.0],
        };
        m.forward();

        m.backward();

        // dL/dA = [[11, 15], [11, 15]]
        let grad_a = m.vars[1].grad.as_ref().unwrap();
        assert_eq!(
            grad_a.data,
            vec![11.0, 15.0, 11.0, 15.0],
            "grad_a: expected [11, 15, 11, 15], got {:?}",
            grad_a.data
        );

        // dL/dB = [[4, 4], [6, 6]]
        let grad_b = m.vars[2].grad.as_ref().unwrap();
        assert_eq!(
            grad_b.data,
            vec![4.0, 4.0, 6.0, 6.0],
            "grad_b: expected [4, 4, 6, 6], got {:?}",
            grad_b.data
        );
    }

    #[test]
    fn test_backward_relu() {
        // Graph: y = relu(p), where p is a parameter
        // Input: [-1, 2, -3, 4] -> ReLU -> [0, 2, 0, 4]
        // With dL/dy = [1, 1, 1, 1]:
        // dL/dp = [0, 1, 0, 1] (gradient blocked where input <= 0)
        let mut b = ModelBuilder::new();
        let x = b.input(1, 1); // dummy input
        let p = b.parameter(2, 2);
        let y = b.relu(p);
        let mut m = b.build(x, y, x, y);

        // Set parameter values (mix of positive and negative)
        m.vars[1].val = Matrix {
            rows: 2,
            cols: 2,
            data: vec![-1.0, 2.0, -3.0, 4.0],
        };
        m.forward();

        // Verify forward pass
        assert_eq!(m.output().data, vec![0.0, 2.0, 0.0, 4.0]);

        m.backward();

        // dL/dp: gradient passes through where input > 0
        let grad_p = m.vars[1].grad.as_ref().unwrap();
        assert_eq!(
            grad_p.data,
            vec![0.0, 1.0, 0.0, 1.0],
            "grad_p: expected [0, 1, 0, 1], got {:?}",
            grad_p.data
        );
    }

    #[test]
    fn test_backward_softmax() {
        // Graph: p -> softmax -> y -> matmul(y, w) -> z (scalar)
        // w = [1, 0, 0]^T selects first element, giving dL/dy = [1, 0, 0]
        //
        // With p = [1, 2, 3]:
        //   y = softmax([1,2,3]) ≈ [0.090, 0.245, 0.665]
        //   dot = Σ(dL/dy_i · y_i) = 1·0.090 + 0 + 0 = 0.090
        //   dL/dp_0 = 0.090 · (1 - 0.090) ≈ +0.082
        //   dL/dp_1 = 0.245 · (0 - 0.090) ≈ -0.022
        //   dL/dp_2 = 0.665 · (0 - 0.090) ≈ -0.060
        let mut b = ModelBuilder::new();
        let dummy = b.input(1, 1);
        let p = b.parameter(1, 3); // input to softmax
        let y = b.softmax(p); // softmax output (1x3)
        let w = b.parameter(3, 1); // selector weights (3x1)
        let z = b.matmul(y, w); // scalar output (1x1)
        let mut m = b.build(dummy, z, dummy, z);

        // p = [1, 2, 3]
        m.vars[1].val = Matrix {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        // w = [1, 0, 0]^T selects first softmax output
        m.vars[3].val = Matrix {
            rows: 3,
            cols: 1,
            data: vec![1.0, 0.0, 0.0],
        };

        m.forward();
        m.backward();

        let grad_p = m.vars[1].grad.as_ref().unwrap();

        // Key property: softmax gradients sum to 0
        let sum: f32 = grad_p.data.iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "softmax grads should sum to 0, got {}",
            sum
        );

        // Increasing p[0] increases y[0], so grad should be positive
        // Increasing p[1] or p[2] decreases y[0], so grads should be negative
        assert!(grad_p.data[0] > 0.0, "grad[0] should be positive");
        assert!(grad_p.data[1] < 0.0, "grad[1] should be negative");
        assert!(grad_p.data[2] < 0.0, "grad[2] should be negative");
    }

    #[test]
    fn test_backward_cross_entropy() {
        // pred = [0.1, 0.7, 0.2], target = [0, 1, 0] (one-hot class 1)
        // L = -ln(0.7) ≈ 0.357
        // dL/dpred = -target / pred = [0, -1/0.7, 0] ≈ [0, -1.4286, 0]
        let mut b = ModelBuilder::new();
        let pred = b.parameter(1, 3); // use parameter so it has gradient
        let target = b.input(1, 3); // labels, no gradient
        let loss = b.cross_entropy(pred, target);
        let mut m = b.build(target, loss, target, loss);

        m.vars[0].val = Matrix {
            rows: 1,
            cols: 3,
            data: vec![0.1, 0.7, 0.2],
        };
        m.vars[1].val = Matrix {
            rows: 1,
            cols: 3,
            data: vec![0.0, 1.0, 0.0],
        };

        m.forward();

        // Verify forward: L = -ln(0.7)
        let expected_loss = -(0.7_f32.ln());
        let actual_loss = m.vars[m.loss as usize].val.data[0];
        assert!(
            (actual_loss - expected_loss).abs() < 1e-6,
            "loss: expected {}, got {}",
            expected_loss,
            actual_loss
        );

        m.backward();

        // dL/dpred = [0, -1/0.7, 0]
        let grad_pred = m.vars[0].grad.as_ref().unwrap();
        let expected_grad = vec![0.0, -1.0 / 0.7, 0.0];
        for i in 0..3 {
            assert!(
                (grad_pred.data[i] - expected_grad[i]).abs() < 1e-6,
                "grad[{}]: expected {}, got {}",
                i,
                expected_grad[i],
                grad_pred.data[i]
            );
        }
    }
}
