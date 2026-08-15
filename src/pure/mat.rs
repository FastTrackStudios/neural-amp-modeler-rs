//! Minimal column-major f32 matrix, mirroring Eigen::MatrixXf semantics
//! as used by NeuralAmpModelerCore.

#[derive(Clone, Debug, Default)]
pub(crate) struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Resize and zero.
    pub fn reset(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.data.clear();
        self.data.resize(rows * cols, 0.0);
    }

    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        debug_assert!(r < self.rows && c < self.cols);
        self.data[c * self.rows + r] = v;
    }

    /// One full column as a slice.
    #[inline]
    pub fn col(&self, c: usize) -> &[f32] {
        &self.data[c * self.rows..(c + 1) * self.rows]
    }

    #[inline]
    pub fn col_mut(&mut self, c: usize) -> &mut [f32] {
        let rows = self.rows;
        &mut self.data[c * rows..(c + 1) * rows]
    }

    /// Contiguous view of the first `n` columns (all rows), like Eigen `leftCols(n)`.
    #[inline]
    pub fn left_cols_mut(&mut self, n: usize) -> &mut [f32] {
        let rows = self.rows;
        &mut self.data[..n * rows]
    }
}
