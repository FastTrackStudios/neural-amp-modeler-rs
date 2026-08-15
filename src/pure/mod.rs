//! Pure-Rust NAM inference engine.
//!
//! A dependency-free (serde_json only) port of the NeuralAmpModelerCore
//! forward pass that compiles on any target, including
//! `wasm32-unknown-unknown` where the C++ core cannot be built. On native
//! targets the crate's [`crate::NamModel`] continues to use the C++ core;
//! this engine backs `NamModel` on wasm and is validated against the C++
//! output by the crate's parity tests.
//!
//! Supported architectures: `WaveNet` (legacy 0.5.x, modern 0.7.x, and the
//! full A2 surface — FiLM, head1x1, GATED/BLENDED/NONE gating, nested
//! condition DSPs, and slimmable channel slicing), `LSTM`, and
//! `SlimmableContainer`. Slimmable models default to full size; use
//! [`PureNamModel::set_slimmable_size`] to select a reduced size.

mod activations;
mod lstm;
mod mat;
mod nn;
mod wavenet;

use serde_json::Value;

const DEFAULT_MAX_BUFFER_SIZE: usize = 4096;

enum Dsp {
    WaveNet(wavenet::WaveNet),
    SlimmableWaveNet(Box<wavenet::SlimmableWaveNet>),
    Lstm(lstm::Lstm),
    /// Submodels sorted by ascending max_value; `active` defaults to the
    /// last (full-size) one, matching ContainerModel's default.
    Container {
        subs: Vec<(f64, Box<Dsp>)>,
        active: usize,
    },
}

impl Dsp {
    fn active(&mut self) -> &mut Dsp {
        match self {
            Dsp::Container { subs, active } => subs[*active].1.active(),
            other => other,
        }
    }

    fn active_ref(&self) -> &Dsp {
        match self {
            Dsp::Container { subs, active } => subs[*active].1.active_ref(),
            other => other,
        }
    }

    fn in_channels(&self) -> usize {
        match self.active_ref() {
            Dsp::WaveNet(w) => w.in_channels(),
            Dsp::SlimmableWaveNet(s) => s.current.in_channels(),
            Dsp::Lstm(l) => l.in_channels(),
            Dsp::Container { .. } => unreachable!(),
        }
    }

    fn out_channels(&self) -> usize {
        match self.active_ref() {
            Dsp::WaveNet(w) => w.out_channels(),
            Dsp::SlimmableWaveNet(s) => s.current.out_channels(),
            Dsp::Lstm(l) => l.out_channels(),
            Dsp::Container { .. } => unreachable!(),
        }
    }

    fn prewarm_samples(&self) -> usize {
        match self.active_ref() {
            Dsp::WaveNet(w) => w.prewarm_samples(),
            Dsp::SlimmableWaveNet(s) => s.current.prewarm_samples(),
            Dsp::Lstm(l) => l.prewarm_samples(),
            Dsp::Container { .. } => unreachable!(),
        }
    }

    fn set_max_buffer_size(&mut self, max_frames: usize) {
        match self.active() {
            Dsp::WaveNet(w) => w.set_max_buffer_size(max_frames),
            Dsp::SlimmableWaveNet(s) => s.current.set_max_buffer_size(max_frames),
            Dsp::Lstm(l) => l.reset_state(),
            Dsp::Container { .. } => unreachable!(),
        }
    }

    fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        match self.active() {
            Dsp::WaveNet(w) => w.process_block(input, output),
            Dsp::SlimmableWaveNet(s) => s.current.process_block(input, output),
            Dsp::Lstm(l) => l.process_block(input, output),
            Dsp::Container { .. } => unreachable!(),
        }
    }

    /// Whether this model supports [`Dsp::set_slimmable_size`].
    fn is_slimmable(&self) -> bool {
        matches!(self, Dsp::Container { .. } | Dsp::SlimmableWaveNet(_))
    }

    /// Select the model size for a ratio in [0, 1]. Returns true when the
    /// active model changed (the caller must reset/prewarm before the next
    /// process call). Mirrors `SlimmableModel::SetSlimmableSize`:
    /// containers pick the first submodel with `val < max_value`; slimmable
    /// WaveNets rebuild with a sliced channel subset.
    fn set_slimmable_size(&mut self, val: f64) -> bool {
        match self {
            Dsp::Container { subs, active } => {
                let idx = subs
                    .iter()
                    .position(|(max_value, _)| val < *max_value)
                    .unwrap_or(subs.len() - 1);
                let changed = idx != *active;
                *active = idx;
                changed
            }
            Dsp::SlimmableWaveNet(s) => {
                // Extraction constraints were validated at load time, so
                // this cannot fail; keep the previous model if it ever does.
                s.set_slimmable_size(val).unwrap_or(false)
            }
            _ => false,
        }
    }
}

/// A loaded NAM model backed by the pure-Rust inference engine.
pub struct PureNamModel {
    dsp: Dsp,
    expected_sample_rate: Option<f64>,
    loudness: Option<f64>,
    input_level: Option<f64>,
    output_level: Option<f64>,
    max_buffer_size: usize,
}

impl PureNamModel {
    /// Load from the raw bytes of a `.nam` file (UTF-8 JSON).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let json = std::str::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {e}"))?;
        Self::from_json(json)
    }

    /// Load from `.nam` JSON text.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json).map_err(|e| format!("invalid JSON: {e}"))?;
        Self::from_value(&value)
    }

    fn from_value(value: &Value) -> Result<Self, String> {
        let dsp = build_dsp(value)?;

        let expected_sample_rate = value
            .get("sample_rate")
            .and_then(|v| v.as_f64())
            .filter(|&sr| sr >= 0.0);
        let meta = value.get("metadata").filter(|m| m.is_object());
        let meta_f64 = |key: &str| -> Option<f64> {
            meta.and_then(|m| m.get(key)).and_then(|v| v.as_f64())
        };

        Ok(Self {
            dsp,
            expected_sample_rate,
            loudness: meta_f64("loudness"),
            input_level: meta_f64("input_level_dbu"),
            output_level: meta_f64("output_level_dbu"),
            max_buffer_size: 0,
        })
    }

    /// Reset for a sample rate and max block size, then prewarm — mirrors
    /// the C++ `ResetAndPrewarm`.
    pub fn reset(&mut self, _sample_rate: f64, max_buffer_size: usize) {
        let max_buffer_size = max_buffer_size.max(1);
        self.max_buffer_size = max_buffer_size;
        self.dsp.set_max_buffer_size(max_buffer_size);

        // Prewarm: run zeros through the model in full blocks.
        let prewarm = self.dsp.prewarm_samples();
        if prewarm > 0 {
            let zeros = vec![0.0f64; max_buffer_size];
            let mut sink = vec![0.0f64; max_buffer_size];
            let mut processed = 0usize;
            while processed < prewarm {
                self.dsp.process_block(&zeros, &mut sink);
                processed += max_buffer_size;
            }
        }
    }

    /// Run inference on a block of mono audio. `input` and `output` must be
    /// the same length. Blocks larger than the configured max buffer size
    /// are processed in chunks.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must be the same length"
        );
        if self.max_buffer_size == 0 {
            self.reset(
                self.expected_sample_rate.unwrap_or(-1.0),
                DEFAULT_MAX_BUFFER_SIZE,
            );
        }
        let chunk = self.max_buffer_size;
        for (ic, oc) in input.chunks(chunk).zip(output.chunks_mut(chunk)) {
            self.dsp.process_block(ic, oc);
        }
    }

    /// Convenience f32 block processing (converts through f64, matching the
    /// C++ `NAM_SAMPLE` double pipeline).
    pub fn process_f32(&mut self, input: &[f32], output: &mut [f32]) {
        let in64: Vec<f64> = input.iter().map(|&x| x as f64).collect();
        let mut out64 = vec![0.0f64; output.len()];
        self.process(&in64, &mut out64);
        for (o, v) in output.iter_mut().zip(out64.iter()) {
            *o = *v as f32;
        }
    }

    pub fn expected_sample_rate(&self) -> Option<f64> {
        self.expected_sample_rate
    }

    pub fn loudness(&self) -> Option<f64> {
        self.loudness
    }

    pub fn input_level(&self) -> Option<f64> {
        self.input_level
    }

    pub fn output_level(&self) -> Option<f64> {
        self.output_level
    }

    pub fn input_channels(&self) -> usize {
        match &self.dsp {
            Dsp::Container { .. } => 1,
            d => d.in_channels(),
        }
    }

    pub fn output_channels(&self) -> usize {
        match &self.dsp {
            Dsp::Container { .. } => 1,
            d => d.out_channels(),
        }
    }

    /// Select the slimmable size for models that support dynamic size
    /// reduction (`SlimmableContainer` and slimmable WaveNets): `val` in
    /// [0.0, 1.0], where 1.0 is full size (the default). Mirrors the C++
    /// `SlimmableModel::SetSlimmableSize`. Returns true if the model is
    /// slimmable, false for ordinary models (no-op).
    ///
    /// When the selection changes, the model is reset (and prewarmed) with
    /// its current sample rate / buffer size before the next process call.
    pub fn set_slimmable_size(&mut self, val: f64) -> bool {
        let supported = self.dsp.is_slimmable();
        let changed = self.dsp.set_slimmable_size(val);
        if changed && self.max_buffer_size > 0 {
            let max_buffer_size = self.max_buffer_size;
            self.reset(self.expected_sample_rate.unwrap_or(-1.0), max_buffer_size);
        }
        supported
    }
}

/// Build a DSP from a full `.nam` model JSON (version/architecture/config/
/// weights), recursing for container submodels.
fn build_dsp(value: &Value) -> Result<Dsp, String> {
    let architecture = value
        .get("architecture")
        .and_then(|v| v.as_str())
        .ok_or("model file missing 'architecture'")?;
    let config = value.get("config").ok_or("model file missing 'config'")?;

    match architecture {
        "WaveNet" => {
            let weights = parse_weights(value)?;
            if wavenet::config_is_slimmable(config)? {
                Ok(Dsp::SlimmableWaveNet(Box::new(wavenet::parse_slimmable(
                    config, &weights,
                )?)))
            } else {
                Ok(Dsp::WaveNet(wavenet::parse(config, &weights)?))
            }
        }
        "LSTM" => {
            let weights = parse_weights(value)?;
            let sr = value
                .get("sample_rate")
                .and_then(|v| v.as_f64())
                .unwrap_or(-1.0);
            Ok(Dsp::Lstm(lstm::parse(config, &weights, sr)?))
        }
        "SlimmableContainer" => {
            let submodels = config
                .get("submodels")
                .and_then(|v| v.as_array())
                .ok_or("SlimmableContainer: 'submodels' must be a non-empty array")?;
            if submodels.is_empty() {
                return Err("SlimmableContainer: 'submodels' must be a non-empty array".into());
            }
            let mut subs = Vec::with_capacity(submodels.len());
            let mut prev = f64::NEG_INFINITY;
            for entry in submodels {
                let max_value = entry
                    .get("max_value")
                    .and_then(|v| v.as_f64())
                    .ok_or("SlimmableContainer submodel missing max_value")?;
                if max_value <= prev {
                    return Err(
                        "SlimmableContainer: submodels must be sorted by ascending max_value"
                            .into(),
                    );
                }
                prev = max_value;
                let model = entry
                    .get("model")
                    .ok_or("SlimmableContainer submodel missing model")?;
                subs.push((max_value, Box::new(build_dsp(model)?)));
            }
            // Default to full size (last submodel), matching ContainerModel.
            let active = subs.len() - 1;
            Ok(Dsp::Container { subs, active })
        }
        other => Err(format!(
            "unsupported architecture for the pure-Rust engine: {other}"
        )),
    }
}

pub(crate) fn parse_weights(value: &Value) -> Result<Vec<f32>, String> {
    let arr = value
        .get("weights")
        .and_then(|v| v.as_array())
        .ok_or("corrupted model file is missing weights")?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(
            v.as_f64()
                .ok_or_else(|| format!("non-numeric weight: {v}"))? as f32,
        );
    }
    Ok(out)
}
