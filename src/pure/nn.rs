//! NN building blocks: 1x1 convolution (pointwise GEMM) and dilated causal
//! Conv1D with input history, mirroring `NAM/dsp.cpp` (Conv1x1) and
//! `NAM/conv1d.cpp` / `NAM/ring_buffer.cpp` (Conv1D).

use super::mat::Mat;

/// Sequential weight reader mirroring `std::vector<float>::iterator&`.
pub(crate) struct Weights<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> Weights<'a> {
    pub fn new(data: &'a [f32]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    pub fn next(&mut self) -> Result<f32, String> {
        let v = self.data.get(self.pos).copied().ok_or_else(|| {
            format!(
                "weight underflow: model expects more than {} weights",
                self.data.len()
            )
        })?;
        self.pos += 1;
        Ok(v)
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.pos != self.data.len() {
            return Err(format!(
                "weight mismatch: consumed {} of {} weights",
                self.pos,
                self.data.len()
            ));
        }
        Ok(())
    }
}

// Conv1x1 ====================================================================

pub(crate) struct Conv1x1 {
    /// (out_channels x in_channels), block-diagonal when grouped.
    weight: Mat,
    /// Empty when bias is disabled.
    bias: Vec<f32>,
    groups: usize,
    in_ch: usize,
    out_ch: usize,
    pub out: Mat,
}

impl Conv1x1 {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        bias: bool,
        groups: usize,
    ) -> Result<Self, String> {
        if groups == 0 || !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
            return Err(format!(
                "Conv1x1: channels ({in_channels} in / {out_channels} out) must divide evenly by groups ({groups})"
            ));
        }
        Ok(Self {
            weight: Mat::new(out_channels, in_channels),
            bias: if bias {
                vec![0.0; out_channels]
            } else {
                Vec::new()
            },
            groups,
            in_ch: in_channels,
            out_ch: out_channels,
            out: Mat::new(out_channels, 0),
        })
    }

    /// Weight order matches NAM Conv1x1::set_weights_: per group, per out,
    /// per in; then bias.
    pub fn set_weights(&mut self, w: &mut Weights) -> Result<(), String> {
        let out_per_group = self.out_ch / self.groups;
        let in_per_group = self.in_ch / self.groups;
        for g in 0..self.groups {
            for i in 0..out_per_group {
                for j in 0..in_per_group {
                    let v = w.next()?;
                    self.weight
                        .set(g * out_per_group + i, g * in_per_group + j, v);
                }
            }
        }
        for b in self.bias.iter_mut() {
            *b = w.next()?;
        }
        Ok(())
    }

    pub fn set_max_buffer_size(&mut self, max_frames: usize) {
        self.out.reset(self.out_ch, max_frames);
    }

    /// out[:, f] = W * input[0..in_ch, f] (+ bias). Reads the top `in_ch`
    /// rows of each input column (columns may be taller, e.g. gated z).
    pub fn process(&mut self, input: &Mat, num_frames: usize) {
        let in_ch = self.in_ch;
        for f in 0..num_frames {
            let in_col = &input.col(f)[..in_ch];
            let out_col = self.out.col_mut(f);
            // GEMV as a sum of scaled weight columns (weight is column-major,
            // so each weight column is contiguous).
            if self.bias.is_empty() {
                out_col.fill(0.0);
            } else {
                out_col.copy_from_slice(&self.bias);
            }
            for (i, &x) in in_col.iter().enumerate() {
                let w_col = self.weight.col(i);
                for (o, w) in out_col.iter_mut().zip(w_col.iter()) {
                    *o += w * x;
                }
            }
        }
    }
}

// Conv1D =====================================================================

pub(crate) struct Conv1D {
    /// One (out x in) matrix per kernel tap, block-diagonal when grouped.
    weight: Vec<Mat>,
    bias: Vec<f32>,
    dilation: usize,
    groups: usize,
    in_ch: usize,
    out_ch: usize,
    /// Rolling input history + current block: (in_ch x (rf + max_frames)).
    ext: Mat,
    max_frames: usize,
    pub out: Mat,
}

impl Conv1D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, String> {
        if groups == 0 || !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
            return Err(format!(
                "Conv1D: channels ({in_channels} in / {out_channels} out) must divide evenly by groups ({groups})"
            ));
        }
        if kernel_size == 0 || dilation == 0 {
            return Err("Conv1D: kernel_size and dilation must be >= 1".into());
        }
        Ok(Self {
            weight: (0..kernel_size)
                .map(|_| Mat::new(out_channels, in_channels))
                .collect(),
            bias: if bias {
                vec![0.0; out_channels]
            } else {
                Vec::new()
            },
            dilation,
            groups,
            in_ch: in_channels,
            out_ch: out_channels,
            ext: Mat::new(in_channels, 0),
            max_frames: 0,
            out: Mat::new(out_channels, 0),
        })
    }

    #[inline]
    pub fn kernel_size(&self) -> usize {
        self.weight.len()
    }

    #[inline]
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// Receptive-field lookback: (kernel_size - 1) * dilation.
    #[inline]
    pub fn lookback(&self) -> usize {
        (self.kernel_size() - 1) * self.dilation
    }

    /// Weight order matches NAM/conv1d.cpp Conv1D::set_weights_: for each
    /// group, out, in — the kernel index k is the innermost loop; then bias.
    pub fn set_weights(&mut self, w: &mut Weights) -> Result<(), String> {
        let out_per_group = self.out_ch / self.groups;
        let in_per_group = self.in_ch / self.groups;
        let kernel = self.weight.len();
        for g in 0..self.groups {
            for i in 0..out_per_group {
                for j in 0..in_per_group {
                    for k in 0..kernel {
                        let v = w.next()?;
                        self.weight[k].set(g * out_per_group + i, g * in_per_group + j, v);
                    }
                }
            }
        }
        for b in self.bias.iter_mut() {
            *b = w.next()?;
        }
        Ok(())
    }

    /// Reset history to zeros and pre-allocate for `max_frames` per block.
    pub fn set_max_buffer_size(&mut self, max_frames: usize) {
        self.max_frames = max_frames;
        self.ext.reset(self.in_ch, self.lookback() + max_frames);
        self.out.reset(self.out_ch, max_frames);
    }

    /// out[:, f] = bias + sum_k W[k] * x[:, f - (K-1-k)*dilation], where x
    /// includes zero-initialized history from previous blocks.
    pub fn process(&mut self, input: &Mat, num_frames: usize) {
        debug_assert!(num_frames <= self.max_frames);
        let rf = self.lookback();
        let kernel = self.weight.len();
        let in_ch = self.in_ch;

        // Append the new block after the history.
        for f in 0..num_frames {
            let src = &input.col(f)[..in_ch];
            self.ext.col_mut(rf + f).copy_from_slice(src);
        }

        // Convolve. Disjoint field borrows: ext read-only, out mutable.
        let ext = &self.ext;
        let out = &mut self.out;
        for f in 0..num_frames {
            let out_col = out.col_mut(f);
            if self.bias.is_empty() {
                out_col.fill(0.0);
            } else {
                out_col.copy_from_slice(&self.bias);
            }
            for (k, wk) in self.weight.iter().enumerate() {
                let lb = self.dilation * (kernel - 1 - k);
                let x_col = ext.col(rf + f - lb);
                // GEMV as a sum of scaled (contiguous) weight columns.
                for (i, &x) in x_col.iter().enumerate() {
                    let w_col = wk.col(i);
                    for (o, w) in out_col.iter_mut().zip(w_col.iter()) {
                        *o += w * x;
                    }
                }
            }
        }

        // Roll history: keep the last `rf` input columns for the next block.
        if rf > 0 {
            if num_frames >= rf {
                for c in 0..rf {
                    let src_col = num_frames - rf + c;
                    let src = &input.col(src_col)[..in_ch];
                    self.ext.col_mut(c).copy_from_slice(src);
                }
            } else {
                // Shift left by num_frames (front rf + num_frames cols valid).
                for c in 0..rf {
                    let src: Vec<f32> = self.ext.col(c + num_frames).to_vec();
                    self.ext.col_mut(c).copy_from_slice(&src);
                }
            }
        }
    }
}
