use anyhow::Result;
use genmap::Handle;

/// The TensorSludge engine
pub struct TensorSludge;

/// A handle referring to a matrix
#[derive(Copy, Clone)]
pub struct Matrix(pub(crate) Handle);

/// A handle referring to a pass
#[derive(Copy, Clone)]
pub struct Pass(pub(crate) Handle);

/// An operation to be executed on the GPU
pub enum Operation {
    /// Perform matrix multiplication, dotting `left` and `right` and storing in `dst`. If either
    /// of the `_transpose` flags are set, operate the matrix multiplication as if
    /// the corresponding matrix was transposed.
    MatrixMultiply {
        left: Matrix,
        left_transpose: bool,
        right: Matrix,
        right_transpose: bool,
        dst: Matrix,
    },
    /// Set all values of this matrix to the sigmoid of their current value
    Sigmoid(Matrix),
    /// Set all values of this matrix to the sigmoid derivative of their current value; Note that
    /// this algorithm expects the matrix in question to already have been passed through
    /// Sigmoid().
    SigmoidDerivative(Matrix),
}

impl TensorSludge {
    /// Create a new TensorSludge instance
    pub fn new() -> Result<Self> {
        todo!()
    }

    /// Create a new matrix with the specified dimensions
    pub fn matrix(&mut self, rows: usize, cols: usize) -> Result<Matrix> {
        todo!()
    }

    /// Write data to a matrix in row-major order
    pub fn write(&mut self, matrix: Matrix, data: &[f32]) -> Result<()> {
        todo!()
    }

    /// Read data from a matrix in row-major order
    pub fn read(&mut self, matrix: Matrix, data: &mut [f32]) -> Result<()> {
        todo!()
    }

    /// Create a pass from a sequence of operations
    pub fn create_pass(&mut self, ops: &[Operation]) -> Result<Pass> {
        todo!()
    }

    /// Run the specified pass on the TensorSludge engine
    pub fn flow(&mut self, pass: Pass) -> Result<()> {
        todo!()
    }
}
