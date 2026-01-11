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

impl ModelBuilder {
    pub fn new() -> Self {
        Self { vars: Vec::new() }
    }

    /// Add an input variable (data fed from outside, no gradient)
    pub fn input(&mut self, rows: usize, cols: usize) -> VarId {
        self.add_var(
            Matrix::zeros(rows, cols),
            None,
            Op::Input,
            VarKind::Input,
        )
    }

    /// Add a trainable parameter (random init, has gradient)
    pub fn parameter(&mut self, rows: usize, cols: usize) -> VarId {
        self.add_var(
            Matrix::zeros(rows, cols), // TODO: random init
            Some(Matrix::zeros(rows, cols)),
            Op::Parameter,
            VarKind::Parameter,
        )
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
        let rows = self.vars[x as usize].val.rows;
        let cols = self.vars[x as usize].val.cols;
        self.add_var(
            Matrix::zeros(rows, cols),
            Some(Matrix::zeros(rows, cols)),
            Op::ReLU(x),
            VarKind::Intermediate,
        )
    }

    /// Softmax activation
    pub fn softmax(&mut self, x: VarId) -> VarId {
        let rows = self.vars[x as usize].val.rows;
        let cols = self.vars[x as usize].val.cols;
        self.add_var(
            Matrix::zeros(rows, cols),
            Some(Matrix::zeros(rows, cols)),
            Op::Softmax(x),
            VarKind::Intermediate,
        )
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

    /// Cross-entropy loss
    pub fn cross_entropy(&mut self, pred: VarId, target: VarId) -> VarId {
        self.add_var(
            Matrix::zeros(1, 1),
            Some(Matrix::zeros(1, 1)),
            Op::CrossEntropy(pred, target),
            VarKind::Cost,
        )
    }

    fn add_var(&mut self, val: Matrix, grad: Option<Matrix>, op: Op, kind: VarKind) -> VarId {
        let id = self.vars.len() as VarId;
        self.vars.push(Var { val, grad, op, kind });
        id
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
}
