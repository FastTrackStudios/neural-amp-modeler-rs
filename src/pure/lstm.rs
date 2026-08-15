//! Pure-Rust LSTM inference, mirroring `NAM/lstm.cpp`.

use serde_json::Value;

use super::activations::sigmoid;
use super::nn::Weights;

struct LstmCell {
    /// (4*hidden x (input+hidden)), row-major as loaded from PyTorch.
    w: Vec<f32>,
    b: Vec<f32>,
    /// Concatenated input + hidden state.
    xh: Vec<f32>,
    ifgo: Vec<f32>,
    /// Cell state.
    c: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
    /// Initial hidden/cell state, kept for reset.
    h0: Vec<f32>,
    c0: Vec<f32>,
}

impl LstmCell {
    fn new(input_size: usize, hidden_size: usize, w: &mut Weights) -> Result<Self, String> {
        let rows = 4 * hidden_size;
        let cols = input_size + hidden_size;
        let mut weight = vec![0.0f32; rows * cols];
        // Row-major load, matching the C++ (PyTorch layout).
        for r in 0..rows {
            for c in 0..cols {
                weight[r * cols + c] = w.next()?;
            }
        }
        let mut b = vec![0.0f32; rows];
        for v in b.iter_mut() {
            *v = w.next()?;
        }
        let mut h0 = vec![0.0f32; hidden_size];
        for v in h0.iter_mut() {
            *v = w.next()?;
        }
        let mut c0 = vec![0.0f32; hidden_size];
        for v in c0.iter_mut() {
            *v = w.next()?;
        }
        let mut xh = vec![0.0f32; cols];
        xh[input_size..].copy_from_slice(&h0);
        Ok(Self {
            w: weight,
            b,
            xh,
            ifgo: vec![0.0f32; rows],
            c: c0.clone(),
            input_size,
            hidden_size,
            h0,
            c0,
        })
    }

    fn reset_state(&mut self) {
        self.xh[self.input_size..].copy_from_slice(&self.h0);
        self.c.copy_from_slice(&self.c0);
    }

    fn hidden_state(&self) -> &[f32] {
        &self.xh[self.input_size..]
    }

    fn process(&mut self, x: &[f32]) {
        let h = self.hidden_size;
        let cols = self.input_size + h;
        self.xh[..self.input_size].copy_from_slice(x);
        // ifgo = W * xh + b
        for r in 0..4 * h {
            let row = &self.w[r * cols..(r + 1) * cols];
            let mut sum = 0.0f32;
            for (wv, xv) in row.iter().zip(self.xh.iter()) {
                sum += wv * xv;
            }
            self.ifgo[r] = sum + self.b[r];
        }
        let (i_off, f_off, g_off, o_off) = (0, h, 2 * h, 3 * h);
        for i in 0..h {
            self.c[i] = sigmoid(self.ifgo[i + f_off]) * self.c[i]
                + sigmoid(self.ifgo[i + i_off]) * self.ifgo[i + g_off].tanh();
        }
        for i in 0..h {
            self.xh[self.input_size + i] = sigmoid(self.ifgo[i + o_off]) * self.c[i].tanh();
        }
    }
}

pub(crate) struct Lstm {
    layers: Vec<LstmCell>,
    /// (out_channels x hidden), row-major.
    head_weight: Vec<f32>,
    head_bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    hidden_size: usize,
    expected_sample_rate: f64,
    input_scratch: Vec<f32>,
}

impl Lstm {
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Matches C++ LSTM::GetPrewarmSamples: half a second at the expected rate.
    pub fn prewarm_samples(&self) -> usize {
        let r = (0.5 * self.expected_sample_rate) as isize;
        if r <= 0 {
            1
        } else {
            r as usize
        }
    }

    pub fn reset_state(&mut self) {
        for l in &mut self.layers {
            l.reset_state();
        }
    }

    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        for (n, &x) in input.iter().enumerate() {
            self.input_scratch[0] = x as f32;
            // Chain layers (first cell consumes the input vector, later cells
            // consume the previous cell's hidden state).
            {
                let first_in = self.layers[0].input_size;
                let (scratch, layers) = (&self.input_scratch, &mut self.layers);
                layers[0].process(&scratch[..first_in]);
            }
            for i in 1..self.layers.len() {
                let (before, rest) = self.layers.split_at_mut(i);
                rest[0].process(before[i - 1].hidden_state());
            }
            // Mono head output (channel 0), matching the mono process API.
            let hidden = self.layers.last().expect("non-empty").hidden_state();
            let row = &self.head_weight[..self.hidden_size];
            let mut sum = 0.0f32;
            for (wv, hv) in row.iter().zip(hidden.iter()) {
                sum += wv * hv;
            }
            output[n] = (sum + self.head_bias[0]) as f64;
        }
    }
}

pub(crate) fn parse(
    config: &Value,
    weights: &[f32],
    expected_sample_rate: f64,
) -> Result<Lstm, String> {
    let geti = |k: &str| -> Result<usize, String> {
        config
            .get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("LSTM config missing '{k}'"))
    };
    let num_layers = geti("num_layers")?;
    let input_size = geti("input_size")?;
    let hidden_size = geti("hidden_size")?;
    let in_channels = config
        .get("in_channels")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let out_channels = config
        .get("out_channels")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    let mut w = Weights::new(weights);
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let cell_in = if i == 0 { input_size } else { hidden_size };
        layers.push(LstmCell::new(cell_in, hidden_size, &mut w)?);
    }
    if layers.is_empty() {
        return Err("LSTM requires at least one layer".into());
    }
    let mut head_weight = vec![0.0f32; out_channels * hidden_size];
    for v in head_weight.iter_mut() {
        *v = w.next()?;
    }
    let mut head_bias = vec![0.0f32; out_channels];
    for v in head_bias.iter_mut() {
        *v = w.next()?;
    }
    w.finish()?;

    Ok(Lstm {
        layers,
        head_weight,
        head_bias,
        in_channels,
        out_channels,
        hidden_size,
        expected_sample_rate,
        input_scratch: vec![0.0f32; input_size.max(1)],
    })
}
