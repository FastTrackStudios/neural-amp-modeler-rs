//! Pure-Rust WaveNet inference, structurally mirroring
//! `NAM/wavenet/{model.cpp,detail.h,params.h}` from NeuralAmpModelerCore.
//!
//! Supports the standard trainer output (legacy 0.5.x `kernel_size`/`gated`
//! configs and modern 0.7.x per-layer `kernel_sizes` / activation-object /
//! layer-head configs) plus the full A2 surface: FiLM at all eight
//! application points, head1x1 modules, GATED/BLENDED/NONE gating modes,
//! nested condition DSPs (WaveNet-in-WaveNet), and slimmable channel slicing
//! (`slice_channels_uniform`, see [`SlimmableWaveNet`]).

use serde_json::Value;

use super::activations::Activation;
use super::mat::Mat;
use super::nn::{Conv1D, Conv1x1, Weights};

#[derive(Clone, Copy, PartialEq)]
enum GatingMode {
    None,
    Gated,
    Blended,
}

// FiLM =======================================================================

/// Feature-wise Linear Modulation, mirroring `NAM/film.h`.
///
/// scale, shift = Conv1x1(condition) split across channels (top = scale,
/// bottom = shift); output = input * scale (+ shift).
struct Film {
    /// condition_dim -> (shift ? 2 : 1) * dim, with bias.
    conv: Conv1x1,
    dim: usize,
    shift: bool,
    out: Mat,
}

impl Film {
    fn new(condition_dim: usize, dim: usize, shift: bool, groups: usize) -> Result<Self, String> {
        Ok(Self {
            conv: Conv1x1::new(condition_dim, if shift { 2 * dim } else { dim }, true, groups)?,
            dim,
            shift,
            out: Mat::new(dim, 0),
        })
    }

    fn set_max_buffer_size(&mut self, max_frames: usize) {
        self.conv.set_max_buffer_size(max_frames);
        self.out.reset(self.dim, max_frames);
    }

    fn set_weights(&mut self, w: &mut Weights) -> Result<(), String> {
        self.conv.set_weights(w)
    }

    /// out = input ⊙ scale (+ shift). Reads the top `dim` rows of `input`.
    fn process(&mut self, input: &Mat, condition: &Mat, num_frames: usize) {
        self.conv.process(condition, num_frames);
        for f in 0..num_frames {
            let ss = self.conv.out.col(f);
            let ic = &input.col(f)[..self.dim];
            let oc = self.out.col_mut(f);
            if self.shift {
                for r in 0..self.dim {
                    oc[r] = ic[r] * ss[r] + ss[self.dim + r];
                }
            } else {
                for r in 0..self.dim {
                    oc[r] = ic[r] * ss[r];
                }
            }
        }
    }

    /// In-place modulation of the top `dim` rows of `target` (mirrors the
    /// C++ `Process_`, including the GATED top-rows copy-back path).
    fn process_in_place(&mut self, target: &mut Mat, condition: &Mat, num_frames: usize) {
        self.conv.process(condition, num_frames);
        for f in 0..num_frames {
            let ss = self.conv.out.col(f);
            let tc = &mut target.col_mut(f)[..self.dim];
            if self.shift {
                for (r, t) in tc.iter_mut().enumerate() {
                    *t = *t * ss[r] + ss[self.dim + r];
                }
            } else {
                for (r, t) in tc.iter_mut().enumerate() {
                    *t *= ss[r];
                }
            }
        }
    }
}

// Layer ======================================================================

struct Layer {
    conv: Conv1D,
    input_mixin: Conv1x1,
    layer1x1: Option<Conv1x1>,
    head1x1: Option<Conv1x1>,
    activation: Activation,
    secondary_activation: Option<Activation>,
    gating: GatingMode,
    bottleneck: usize,
    channels: usize,
    // FiLM modules, in weight order (see Layer::set_weights).
    conv_pre_film: Option<Film>,
    conv_post_film: Option<Film>,
    input_mixin_pre_film: Option<Film>,
    input_mixin_post_film: Option<Film>,
    activation_pre_film: Option<Film>,
    activation_post_film: Option<Film>,
    layer1x1_post_film: Option<Film>,
    head1x1_post_film: Option<Film>,
    /// z = conv(input) + input_mixin(condition); rows = bottleneck (or
    /// 2*bottleneck when gated/blended). Post-activation, the top
    /// `bottleneck` rows hold the activated output.
    z: Mat,
    /// Residual output to the next layer (channels rows).
    out_next: Mat,
}

impl Layer {
    fn set_max_buffer_size(&mut self, max_frames: usize) {
        self.conv.set_max_buffer_size(max_frames);
        self.input_mixin.set_max_buffer_size(max_frames);
        let z_channels = match self.gating {
            GatingMode::None => self.bottleneck,
            GatingMode::Gated | GatingMode::Blended => 2 * self.bottleneck,
        };
        self.z.reset(z_channels, max_frames);
        if let Some(l) = &mut self.layer1x1 {
            l.set_max_buffer_size(max_frames);
        }
        if let Some(h) = &mut self.head1x1 {
            h.set_max_buffer_size(max_frames);
        }
        for film in [
            &mut self.conv_pre_film,
            &mut self.conv_post_film,
            &mut self.input_mixin_pre_film,
            &mut self.input_mixin_post_film,
            &mut self.activation_pre_film,
            &mut self.activation_post_film,
            &mut self.layer1x1_post_film,
            &mut self.head1x1_post_film,
        ]
        .into_iter()
        .flatten()
        {
            film.set_max_buffer_size(max_frames);
        }
        self.out_next.reset(self.channels, max_frames);
    }

    /// Weight order matches `detail::Layer::set_weights_`: conv, input_mixin,
    /// layer1x1, head1x1, then the eight FiLMs in declaration order.
    fn set_weights(&mut self, w: &mut Weights) -> Result<(), String> {
        self.conv.set_weights(w)?;
        self.input_mixin.set_weights(w)?;
        if let Some(l) = &mut self.layer1x1 {
            l.set_weights(w)?;
        }
        if let Some(h) = &mut self.head1x1 {
            h.set_weights(w)?;
        }
        for film in [
            &mut self.conv_pre_film,
            &mut self.conv_post_film,
            &mut self.input_mixin_pre_film,
            &mut self.input_mixin_post_film,
            &mut self.activation_pre_film,
            &mut self.activation_post_film,
            &mut self.layer1x1_post_film,
            &mut self.head1x1_post_film,
        ]
        .into_iter()
        .flatten()
        {
            film.set_weights(w)?;
        }
        Ok(())
    }

    fn process(&mut self, input: &Mat, condition: &Mat, num_frames: usize) {
        // Step 1: input convolution (with optional pre/post FiLM).
        match &mut self.conv_pre_film {
            Some(film) => {
                film.process(input, condition, num_frames);
                self.conv.process(&film.out, num_frames);
            }
            None => self.conv.process(input, num_frames),
        }
        if let Some(film) = &mut self.conv_post_film {
            film.process_in_place(&mut self.conv.out, condition, num_frames);
        }

        // Input mixin (with optional pre/post FiLM).
        match &mut self.input_mixin_pre_film {
            Some(film) => {
                film.process(condition, condition, num_frames);
                self.input_mixin.process(&film.out, num_frames);
            }
            None => self.input_mixin.process(condition, num_frames),
        }
        if let Some(film) = &mut self.input_mixin_post_film {
            film.process_in_place(&mut self.input_mixin.out, condition, num_frames);
        }

        // z = conv out + input mixin out
        let z_rows = self.z.rows();
        for f in 0..num_frames {
            let a = self.conv.out.col(f);
            let b = self.input_mixin.out.col(f);
            let z = self.z.col_mut(f);
            for r in 0..z_rows {
                z[r] = a[r] + b[r];
            }
        }

        if let Some(film) = &mut self.activation_pre_film {
            film.process_in_place(&mut self.z, condition, num_frames);
        }

        // Step 2 & 3: activation (with gating/blending) and layer1x1.
        match self.gating {
            GatingMode::None => {
                // Contiguous leftCols apply, matching the C++ flat apply.
                self.activation.apply(self.z.left_cols_mut(num_frames));
                if let Some(film) = &mut self.activation_post_film {
                    film.process_in_place(&mut self.z, condition, num_frames);
                }
                if let Some(l) = &mut self.layer1x1 {
                    l.process(&self.z, num_frames);
                }
            }
            GatingMode::Gated => {
                // Per-column: top = act(top) * secondary(bottom); matches
                // GatingActivation (per-column buffers, so PReLU-style pos
                // indexing restarts each column).
                let bn = self.bottleneck;
                let secondary = self
                    .secondary_activation
                    .as_ref()
                    .unwrap_or(&Activation::Sigmoid);
                let mut top = vec![0.0f32; bn];
                let mut bottom = vec![0.0f32; bn];
                for f in 0..num_frames {
                    {
                        let z = self.z.col(f);
                        top.copy_from_slice(&z[..bn]);
                        bottom.copy_from_slice(&z[bn..2 * bn]);
                    }
                    self.activation.apply(&mut top);
                    secondary.apply(&mut bottom);
                    let z = self.z.col_mut(f);
                    for c in 0..bn {
                        z[c] = top[c] * bottom[c];
                    }
                }
                if let Some(film) = &mut self.activation_post_film {
                    film.process_in_place(&mut self.z, condition, num_frames);
                }
                if let Some(l) = &mut self.layer1x1 {
                    l.process(&self.z, num_frames);
                }
            }
            GatingMode::Blended => {
                // Per-column: alpha = secondary(bottom);
                // top = alpha * act(top) + (1 - alpha) * top_pre; matches
                // BlendingActivation.
                let bn = self.bottleneck;
                let secondary = self
                    .secondary_activation
                    .as_ref()
                    .unwrap_or(&Activation::Sigmoid);
                let mut pre = vec![0.0f32; bn];
                let mut top = vec![0.0f32; bn];
                let mut bottom = vec![0.0f32; bn];
                for f in 0..num_frames {
                    {
                        let z = self.z.col(f);
                        pre.copy_from_slice(&z[..bn]);
                        top.copy_from_slice(&z[..bn]);
                        bottom.copy_from_slice(&z[bn..2 * bn]);
                    }
                    self.activation.apply(&mut top);
                    secondary.apply(&mut bottom);
                    let z = self.z.col_mut(f);
                    for c in 0..bn {
                        z[c] = bottom[c] * top[c] + (1.0 - bottom[c]) * pre[c];
                    }
                }
                if let Some(film) = &mut self.activation_post_film {
                    film.process_in_place(&mut self.z, condition, num_frames);
                }
                if let Some(l) = &mut self.layer1x1 {
                    l.process(&self.z, num_frames);
                    // NOTE: the C++ core applies layer1x1_post_film only in
                    // the BLENDED branch (the weights are consumed in every
                    // mode but the modulation is skipped for NONE/GATED).
                    // Replicated faithfully — the C++ core is the oracle.
                    if let Some(film) = &mut self.layer1x1_post_film {
                        film.process_in_place(&mut l.out, condition, num_frames);
                    }
                }
            }
        }

        // head1x1 reads the activated top `bottleneck` rows of z.
        if let Some(h) = &mut self.head1x1 {
            h.process(&self.z, num_frames);
            if let Some(film) = &mut self.head1x1_post_film {
                film.process_in_place(&mut h.out, condition, num_frames);
            }
        }

        // Residual: out_next = input + layer1x1(z), or input if inactive.
        for f in 0..num_frames {
            let in_col = &input.col(f)[..self.channels];
            let out_col = self.out_next.col_mut(f);
            match &self.layer1x1 {
                Some(l) => {
                    let lc = l.out.col(f);
                    for c in 0..self.channels {
                        out_col[c] = in_col[c] + lc[c];
                    }
                }
                None => out_col.copy_from_slice(in_col),
            }
        }
    }
}

// LayerArray =================================================================

struct LayerArray {
    rechannel: Conv1x1,
    layers: Vec<Layer>,
    /// Accumulated skip connections (head_output_size rows).
    head_inputs: Mat,
    /// Projects accumulated head inputs to head_size (causal Conv1D).
    head_rechannel: Conv1D,
    head_output_size: usize,
}

impl LayerArray {
    fn receptive_field(&self) -> usize {
        let mut rf = 0;
        for layer in &self.layers {
            rf += layer.conv.dilation() * (layer.conv.kernel_size() - 1);
        }
        rf += self.head_rechannel.dilation() * (self.head_rechannel.kernel_size() - 1);
        rf
    }

    fn set_max_buffer_size(&mut self, max_frames: usize) {
        self.rechannel.set_max_buffer_size(max_frames);
        self.head_rechannel.set_max_buffer_size(max_frames);
        for layer in &mut self.layers {
            layer.set_max_buffer_size(max_frames);
        }
        self.head_inputs.reset(self.head_output_size, max_frames);
    }

    fn set_weights(&mut self, w: &mut Weights) -> Result<(), String> {
        self.rechannel.set_weights(w)?;
        for layer in &mut self.layers {
            layer.set_weights(w)?;
        }
        self.head_rechannel.set_weights(w)
    }

    /// `head_in`: None for the first layer array (accumulator zeroed),
    /// Some(prev head outputs) for subsequent ones.
    fn process(&mut self, input: &Mat, condition: &Mat, head_in: Option<&Mat>, num_frames: usize) {
        match head_in {
            None => self.head_inputs.zero(),
            Some(h) => {
                for f in 0..num_frames {
                    let src = &h.col(f)[..self.head_output_size];
                    self.head_inputs.col_mut(f).copy_from_slice(src);
                }
            }
        }

        self.rechannel.process(input, num_frames);

        for i in 0..self.layers.len() {
            let (before, rest) = self.layers.split_at_mut(i);
            let layer = &mut rest[0];
            if i == 0 {
                layer.process(&self.rechannel.out, condition, num_frames);
            } else {
                let prev_out = &before[i - 1].out_next;
                layer.process(prev_out, condition, num_frames);
            }
            // Accumulate skip connection: head1x1 output if active, else the
            // activated top rows of z.
            let src: &Mat = match &layer.head1x1 {
                Some(h) => &h.out,
                None => &layer.z,
            };
            for f in 0..num_frames {
                let s = &src.col(f)[..self.head_output_size];
                let d = self.head_inputs.col_mut(f);
                for r in 0..s.len() {
                    d[r] += s[r];
                }
            }
        }

        self.head_rechannel.process(&self.head_inputs, num_frames);
    }

    fn layer_outputs(&self) -> &Mat {
        &self.layers.last().expect("layer array is non-empty").out_next
    }

    fn head_outputs(&self) -> &Mat {
        &self.head_rechannel.out
    }
}

// Post-stack head ============================================================

struct PostHead {
    convs: Vec<Conv1D>,
    activations: Vec<Activation>,
    in_channels: usize,
    scratch: Mat,
}

impl PostHead {
    fn receptive_field(&self) -> usize {
        let mut rf = 1;
        for c in &self.convs {
            rf += c.kernel_size() - 1;
        }
        rf
    }

    fn set_max_buffer_size(&mut self, max_frames: usize) {
        for c in &mut self.convs {
            c.set_max_buffer_size(max_frames);
        }
        self.scratch.reset(self.in_channels, max_frames);
    }

    fn set_weights(&mut self, w: &mut Weights) -> Result<(), String> {
        for c in &mut self.convs {
            c.set_weights(w)?;
        }
        Ok(())
    }

    /// Input is `scratch` (already scaled by head_scale). Applies
    /// activation → conv per stage; output is the last conv's out.
    fn process(&mut self, num_frames: usize) {
        for i in 0..self.convs.len() {
            if i == 0 {
                self.activations[i].apply(self.scratch.left_cols_mut(num_frames));
                self.convs[i].process(&self.scratch, num_frames);
            } else {
                let (before, rest) = self.convs.split_at_mut(i);
                let prev_out = &mut before[i - 1].out;
                self.activations[i].apply(prev_out.left_cols_mut(num_frames));
                rest[0].process(prev_out, num_frames);
            }
        }
    }
}

// WaveNet ====================================================================

pub(crate) struct WaveNet {
    layer_arrays: Vec<LayerArray>,
    head_scale: f32,
    post_head: Option<PostHead>,
    /// Optional nested condition DSP (WaveNet-in-WaveNet). The raw input is
    /// still the layer input; the condition fed to input mixins / FiLMs is
    /// this DSP's output.
    condition_dsp: Option<Box<WaveNet>>,
    in_channels: usize,
    out_channels: usize,
    /// Raw input (in_channels x frames).
    condition_input: Mat,
    /// Final scaled output (out_channels x frames) so nested WaveNets can be
    /// consumed as multi-channel condition signals.
    output: Mat,
    prewarm_samples: usize,
    max_frames: usize,
}

impl WaveNet {
    pub fn prewarm_samples(&self) -> usize {
        self.prewarm_samples
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn set_max_buffer_size(&mut self, max_frames: usize) {
        self.max_frames = max_frames;
        self.condition_input.reset(self.in_channels, max_frames);
        self.output.reset(self.out_channels, max_frames);
        if let Some(cd) = &mut self.condition_dsp {
            cd.set_max_buffer_size(max_frames);
        }
        for la in &mut self.layer_arrays {
            la.set_max_buffer_size(max_frames);
        }
        if let Some(h) = &mut self.post_head {
            h.set_max_buffer_size(max_frames);
        }
    }

    /// Core block processing over the internal `condition_input` /
    /// `output` matrices (multi-channel capable, used for nesting).
    fn process_frames(&mut self, num_frames: usize) {
        let out_channels = self.out_channels;
        let head_scale = self.head_scale;
        let WaveNet {
            layer_arrays,
            post_head,
            condition_dsp,
            condition_input,
            output,
            ..
        } = self;

        // Condition: the nested DSP's output if present, else the raw input.
        // (The f32→f64→f32 round trips of the C++ buffer plumbing are exact,
        // so the all-f32 path here is bit-identical.)
        if let Some(cd) = condition_dsp.as_deref_mut() {
            for f in 0..num_frames {
                let src = &condition_input.col(f)[..cd.in_channels];
                cd.condition_input.col_mut(f).copy_from_slice(src);
            }
            cd.process_frames(num_frames);
        }
        let condition: &Mat = match condition_dsp.as_deref() {
            Some(cd) => &cd.output,
            None => condition_input,
        };

        for i in 0..layer_arrays.len() {
            let (before, rest) = layer_arrays.split_at_mut(i);
            let la = &mut rest[0];
            if i == 0 {
                la.process(condition_input, condition, None, num_frames);
            } else {
                // `before` and `la` are disjoint borrows from split_at_mut.
                let prev = &before[i - 1];
                la.process(
                    prev.layer_outputs(),
                    condition,
                    Some(prev.head_outputs()),
                    num_frames,
                );
            }
        }

        let final_head = layer_arrays.last().expect("non-empty").head_outputs();

        match post_head {
            Some(head) => {
                // scratch = head_scale * final head outputs
                for f in 0..num_frames {
                    let src = final_head.col(f);
                    let dst = head.scratch.col_mut(f);
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d = head_scale * s;
                    }
                }
                head.process(num_frames);
                let out_mat = &head.convs.last().expect("non-empty").out;
                for f in 0..num_frames {
                    let src = &out_mat.col(f)[..out_channels];
                    output.col_mut(f).copy_from_slice(src);
                }
            }
            None => {
                for f in 0..num_frames {
                    let src = &final_head.col(f)[..out_channels];
                    let dst = output.col_mut(f);
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d = head_scale * s;
                    }
                }
            }
        }
    }

    /// Process one block (mono). `num_frames` must be <= max buffer size.
    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        let num_frames = input.len();
        debug_assert!(num_frames <= self.max_frames);

        for (f, &x) in input.iter().enumerate() {
            self.condition_input.col_mut(f)[0] = x as f32;
        }
        self.process_frames(num_frames);
        for (f, o) in output.iter_mut().enumerate().take(num_frames) {
            *o = self.output.col(f)[0] as f64;
        }
    }
}

// Parsing ====================================================================

fn as_usize(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("expected non-negative integer for {what}, got {v}"))
}

fn get<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    obj.as_object().and_then(|o| o.get(key)).filter(|v| !v.is_null())
}

// FiLM application points, in weight order.
const FILM_CONV_PRE: usize = 0;
const FILM_CONV_POST: usize = 1;
const FILM_INPUT_MIXIN_PRE: usize = 2;
const FILM_INPUT_MIXIN_POST: usize = 3;
const FILM_ACTIVATION_PRE: usize = 4;
const FILM_ACTIVATION_POST: usize = 5;
const FILM_LAYER1X1_POST: usize = 6;
const FILM_HEAD1X1_POST: usize = 7;
const FILM_KEYS: [&str; 8] = [
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
];

#[derive(Clone, Copy, Default)]
struct FilmParams {
    active: bool,
    shift: bool,
    groups: usize,
}

/// Everything needed to construct one layer array — the analogue of the C++
/// `LayerArrayParams`. Kept around by [`SlimmableWaveNet`] so slimmed
/// variants can be rebuilt without re-parsing JSON.
#[derive(Clone)]
struct LayerArraySpec {
    input_size: usize,
    condition_size: usize,
    head_size: usize,
    head_dilation: usize,
    head_kernel_size: usize,
    channels: usize,
    bottleneck: usize,
    kernel_sizes: Vec<usize>,
    dilations: Vec<usize>,
    activations: Vec<Activation>,
    gating_modes: Vec<GatingMode>,
    secondary_activations: Vec<Option<Activation>>,
    head_bias: bool,
    groups_input: usize,
    groups_input_mixin: usize,
    layer1x1_active: bool,
    layer1x1_groups: usize,
    head1x1_active: bool,
    head1x1_out: usize,
    head1x1_groups: usize,
    films: [FilmParams; 8],
}

impl LayerArraySpec {
    fn head_output_size(&self) -> usize {
        if self.head1x1_active {
            self.head1x1_out
        } else {
            self.bottleneck
        }
    }
}

/// Post-stack head parameters (top-level "head" object).
#[derive(Clone)]
struct HeadSpec {
    in_channels: usize,
    channels: usize,
    out_channels: usize,
    kernel_sizes: Vec<usize>,
    activation: Activation,
}

fn parse_film(lc: &Value, key: &str, err_ctx: &dyn Fn(&str) -> String) -> Result<FilmParams, String> {
    match get(lc, key) {
        None | Some(Value::Bool(false)) => Ok(FilmParams::default()),
        Some(Value::Object(o)) => Ok(FilmParams {
            active: o.get("active").and_then(|v| v.as_bool()).unwrap_or(true),
            shift: o.get("shift").and_then(|v| v.as_bool()).unwrap_or(true),
            groups: match o.get("groups").filter(|v| !v.is_null()) {
                Some(v) => as_usize(v, &format!("{key}.groups"))?,
                None => 1,
            },
        }),
        Some(_) => Err(err_ctx(&format!("'{key}' must be an object or false"))),
    }
}

fn parse_layer_array_spec(lc: &Value, idx: usize) -> Result<LayerArraySpec, String> {
    let err_ctx = |m: &str| format!("layer array {idx}: {m}");

    let channels = as_usize(
        get(lc, "channels").ok_or_else(|| err_ctx("missing channels"))?,
        "channels",
    )?;
    let bottleneck = match get(lc, "bottleneck") {
        Some(v) => as_usize(v, "bottleneck")?,
        None => channels,
    };
    let input_size = as_usize(
        get(lc, "input_size").ok_or_else(|| err_ctx("missing input_size"))?,
        "input_size",
    )?;
    let condition_size = as_usize(
        get(lc, "condition_size").ok_or_else(|| err_ctx("missing condition_size"))?,
        "condition_size",
    )?;
    let groups_input = match get(lc, "groups_input") {
        Some(v) => as_usize(v, "groups_input")?,
        None => 1,
    };
    let groups_input_mixin = match get(lc, "groups_input_mixin") {
        Some(v) => as_usize(v, "groups_input_mixin")?,
        None => 1,
    };

    // layer1x1: defaults to active with groups 1.
    let (layer1x1_active, layer1x1_groups) = match get(lc, "layer1x1") {
        Some(v) => (
            v.get("active").and_then(|b| b.as_bool()).unwrap_or(true),
            v.get("groups")
                .and_then(|g| g.as_u64())
                .map(|g| g as usize)
                .unwrap_or(1),
        ),
        None => (true, 1),
    };
    // head1x1: defaults to inactive.
    let (head1x1_active, head1x1_out, head1x1_groups) = match get(lc, "head1x1") {
        Some(v) => (
            v.get("active").and_then(|b| b.as_bool()).unwrap_or(false),
            v.get("out_channels")
                .and_then(|g| g.as_u64())
                .map(|g| g as usize)
                .unwrap_or(channels),
            v.get("groups")
                .and_then(|g| g.as_u64())
                .map(|g| g as usize)
                .unwrap_or(1),
        ),
        None => (false, channels, 1),
    };

    // Layer-array head (rechannel to head_size).
    let (head_size, head_kernel_size, head_bias, head_dilation) =
        if let Some(hj) = get(lc, "head") {
            if !hj.is_object() {
                return Err(err_ctx("'head' must be a JSON object"));
            }
            (
                as_usize(
                    hj.get("out_channels").ok_or_else(|| err_ctx("head missing out_channels"))?,
                    "head.out_channels",
                )?,
                as_usize(
                    hj.get("kernel_size").ok_or_else(|| err_ctx("head missing kernel_size"))?,
                    "head.kernel_size",
                )?,
                hj.get("bias")
                    .and_then(|v| v.as_bool())
                    .ok_or_else(|| err_ctx("head missing bias"))?,
                match hj.get("head_dilation").filter(|v| !v.is_null()) {
                    Some(v) => as_usize(v, "head.head_dilation")?,
                    None => 1,
                },
            )
        } else if let Some(hs) = get(lc, "head_size") {
            (
                as_usize(hs, "head_size")?,
                1,
                get(lc, "head_bias")
                    .and_then(|v| v.as_bool())
                    .ok_or_else(|| err_ctx("missing head_bias"))?,
                1,
            )
        } else {
            return Err(err_ctx("expected 'head' object or legacy 'head_size'/'head_bias'"));
        };

    let dilations: Vec<usize> = get(lc, "dilations")
        .and_then(|v| v.as_array())
        .ok_or_else(|| err_ctx("missing dilations"))?
        .iter()
        .map(|v| as_usize(v, "dilation"))
        .collect::<Result<_, _>>()?;
    let num_layers = dilations.len();

    // Kernel sizes: per-layer array or single legacy value.
    let kernel_sizes: Vec<usize> = if let Some(ks) = get(lc, "kernel_sizes") {
        let arr = ks
            .as_array()
            .ok_or_else(|| err_ctx("kernel_sizes must be an array"))?;
        if arr.len() != num_layers {
            return Err(err_ctx("kernel_sizes size must match dilations size"));
        }
        arr.iter()
            .map(|v| as_usize(v, "kernel_size"))
            .collect::<Result<_, _>>()?
    } else if let Some(k) = get(lc, "kernel_size") {
        vec![as_usize(k, "kernel_size")?; num_layers]
    } else {
        return Err(err_ctx("either kernel_size or kernel_sizes must be provided"));
    };

    // Activations: single config or per-layer array.
    let activation_json = get(lc, "activation").ok_or_else(|| err_ctx("missing activation"))?;
    let activations: Vec<Activation> = if let Some(arr) = activation_json.as_array() {
        if arr.len() != num_layers {
            return Err(err_ctx("activation array size must match dilations size"));
        }
        arr.iter().map(Activation::from_json).collect::<Result<_, _>>()?
    } else {
        vec![Activation::from_json(activation_json)?; num_layers]
    };

    // Gating: gating_mode (string or array) or legacy "gated" bool.
    let parse_mode = |s: &str| -> Result<GatingMode, String> {
        match s {
            "none" => Ok(GatingMode::None),
            "gated" => Ok(GatingMode::Gated),
            "blended" => Ok(GatingMode::Blended),
            other => Err(format!("invalid gating_mode: {other}")),
        }
    };
    let secondary_json = get(lc, "secondary_activation");
    let (gating_modes, secondary_activations): (Vec<GatingMode>, Vec<Option<Activation>>) =
        if let Some(gm) = get(lc, "gating_mode") {
            if let Some(arr) = gm.as_array() {
                if arr.len() != num_layers {
                    return Err(err_ctx("gating_mode array size must match dilations size"));
                }
                let mut modes = Vec::new();
                let mut secs = Vec::new();
                for (li, m) in arr.iter().enumerate() {
                    let mode = parse_mode(m.as_str().unwrap_or("")).map_err(|e| err_ctx(&e))?;
                    modes.push(mode);
                    let sec = if mode == GatingMode::None {
                        None
                    } else if let Some(sa) = secondary_json {
                        if let Some(sarr) = sa.as_array() {
                            match sarr.get(li).filter(|v| !v.is_null()) {
                                Some(v) => Some(Activation::from_json(v)?),
                                None => Some(Activation::Sigmoid),
                            }
                        } else {
                            Some(Activation::from_json(sa)?)
                        }
                    } else {
                        Some(Activation::Sigmoid)
                    };
                    secs.push(sec);
                }
                (modes, secs)
            } else {
                let mode = parse_mode(gm.as_str().unwrap_or("")).map_err(|e| err_ctx(&e))?;
                let sec = if mode == GatingMode::None {
                    None
                } else if let Some(sa) = secondary_json {
                    Some(Activation::from_json(sa)?)
                } else {
                    Some(Activation::Sigmoid)
                };
                (vec![mode; num_layers], vec![sec; num_layers])
            }
        } else if let Some(g) = get(lc, "gated") {
            let gated = g.as_bool().unwrap_or(false);
            if gated {
                (
                    vec![GatingMode::Gated; num_layers],
                    vec![Some(Activation::Sigmoid); num_layers],
                )
            } else {
                (vec![GatingMode::None; num_layers], vec![None; num_layers])
            }
        } else {
            (vec![GatingMode::None; num_layers], vec![None; num_layers])
        };

    let mut films = [FilmParams::default(); 8];
    for (i, key) in FILM_KEYS.iter().enumerate() {
        films[i] = parse_film(lc, key, &err_ctx)?;
    }

    if !layer1x1_active {
        if bottleneck != channels {
            return Err(err_ctx("when layer1x1 is inactive, bottleneck must equal channels"));
        }
        if films[FILM_LAYER1X1_POST].active {
            return Err(err_ctx(
                "layer1x1_post_film cannot be active when layer1x1 is not active",
            ));
        }
    }
    if !head1x1_active && films[FILM_HEAD1X1_POST].active {
        return Err(err_ctx(
            "head1x1_post_film cannot be active when head1x1 is not active",
        ));
    }

    Ok(LayerArraySpec {
        input_size,
        condition_size,
        head_size,
        head_dilation,
        head_kernel_size,
        channels,
        bottleneck,
        kernel_sizes,
        dilations,
        activations,
        gating_modes,
        secondary_activations,
        head_bias,
        groups_input,
        groups_input_mixin,
        layer1x1_active,
        layer1x1_groups,
        head1x1_active,
        head1x1_out,
        head1x1_groups,
        films,
    })
}

fn parse_layer_array_specs(config: &Value) -> Result<Vec<LayerArraySpec>, String> {
    let layers_json = get(config, "layers")
        .and_then(|v| v.as_array())
        .ok_or("WaveNet config missing 'layers' array")?;
    if layers_json.is_empty() {
        return Err("WaveNet requires at least one layer array".into());
    }
    layers_json
        .iter()
        .enumerate()
        .map(|(idx, lc)| parse_layer_array_spec(lc, idx))
        .collect()
}

fn parse_head_spec(config: &Value, specs: &[LayerArraySpec]) -> Result<Option<HeadSpec>, String> {
    let Some(hj) = get(config, "head") else {
        return Ok(None);
    };
    let hp_in = specs.last().expect("non-empty").head_size;
    if let Some(legacy_in) = hj.get("in_channels").filter(|v| !v.is_null()) {
        if as_usize(legacy_in, "head.in_channels")? != hp_in {
            return Err("WaveNet head.in_channels must equal last layer's head_size".into());
        }
    }
    let channels = as_usize(
        hj.get("channels").ok_or("head missing channels")?,
        "head.channels",
    )?;
    let out_channels = as_usize(
        hj.get("out_channels").ok_or("head missing out_channels")?,
        "head.out_channels",
    )?;
    let kernel_sizes: Vec<usize> = hj
        .get("kernel_sizes")
        .and_then(|v| v.as_array())
        .ok_or("head missing kernel_sizes")?
        .iter()
        .map(|v| as_usize(v, "head kernel_size"))
        .collect::<Result<_, _>>()?;
    if kernel_sizes.is_empty() {
        return Err("head.kernel_sizes must be non-empty".into());
    }
    let activation = Activation::from_json(hj.get("activation").ok_or("head missing activation")?)?;
    Ok(Some(HeadSpec {
        in_channels: hp_in,
        channels,
        out_channels,
        kernel_sizes,
        activation,
    }))
}

fn parse_condition_dsp(config: &Value) -> Result<Option<Box<WaveNet>>, String> {
    let Some(cd) = get(config, "condition_dsp") else {
        return Ok(None);
    };
    let arch = cd
        .get("architecture")
        .and_then(|v| v.as_str())
        .ok_or("condition_dsp missing 'architecture'")?;
    if arch != "WaveNet" {
        return Err(format!(
            "condition_dsp architecture '{arch}' is not supported by the pure-Rust engine"
        ));
    }
    let cfg = cd.get("config").ok_or("condition_dsp missing 'config'")?;
    let weights = super::parse_weights(cd)?;
    Ok(Some(Box::new(parse(cfg, &weights)?)))
}

fn build_layer(spec: &LayerArraySpec, li: usize) -> Result<Layer, String> {
    let gated = spec.gating_modes[li] != GatingMode::None;
    let z_channels = if gated { 2 * spec.bottleneck } else { spec.bottleneck };
    let film = |idx: usize, dim: usize| -> Result<Option<Film>, String> {
        let p = spec.films[idx];
        if p.active {
            Ok(Some(Film::new(spec.condition_size, dim, p.shift, p.groups)?))
        } else {
            Ok(None)
        }
    };
    Ok(Layer {
        conv: Conv1D::new(
            spec.channels,
            z_channels,
            spec.kernel_sizes[li],
            true,
            spec.dilations[li],
            spec.groups_input,
        )?,
        input_mixin: Conv1x1::new(spec.condition_size, z_channels, false, spec.groups_input_mixin)?,
        layer1x1: if spec.layer1x1_active {
            Some(Conv1x1::new(spec.bottleneck, spec.channels, true, spec.layer1x1_groups)?)
        } else {
            None
        },
        head1x1: if spec.head1x1_active {
            Some(Conv1x1::new(spec.bottleneck, spec.head1x1_out, true, spec.head1x1_groups)?)
        } else {
            None
        },
        activation: spec.activations[li].clone(),
        secondary_activation: spec.secondary_activations[li].clone(),
        gating: spec.gating_modes[li],
        bottleneck: spec.bottleneck,
        channels: spec.channels,
        conv_pre_film: film(FILM_CONV_PRE, spec.channels)?,
        conv_post_film: film(FILM_CONV_POST, z_channels)?,
        input_mixin_pre_film: film(FILM_INPUT_MIXIN_PRE, spec.condition_size)?,
        input_mixin_post_film: film(FILM_INPUT_MIXIN_POST, z_channels)?,
        activation_pre_film: film(FILM_ACTIVATION_PRE, z_channels)?,
        activation_post_film: film(FILM_ACTIVATION_POST, spec.bottleneck)?,
        layer1x1_post_film: if spec.layer1x1_active {
            film(FILM_LAYER1X1_POST, spec.channels)?
        } else {
            None
        },
        head1x1_post_film: if spec.head1x1_active {
            film(FILM_HEAD1X1_POST, spec.head1x1_out)?
        } else {
            None
        },
        z: Mat::new(z_channels, 0),
        out_next: Mat::new(spec.channels, 0),
    })
}

fn build_layer_array(spec: &LayerArraySpec) -> Result<LayerArray, String> {
    let mut layers = Vec::with_capacity(spec.dilations.len());
    for li in 0..spec.dilations.len() {
        layers.push(build_layer(spec, li)?);
    }
    let head_output_size = spec.head_output_size();
    Ok(LayerArray {
        rechannel: Conv1x1::new(spec.input_size, spec.channels, false, 1)?,
        layers,
        head_inputs: Mat::new(head_output_size, 0),
        head_rechannel: Conv1D::new(
            head_output_size,
            spec.head_size,
            spec.head_kernel_size,
            spec.head_bias,
            spec.head_dilation,
            1,
        )?,
        head_output_size,
    })
}

/// Construct a WaveNet from parsed specs and the flat weight vector.
fn build(
    in_channels: usize,
    specs: &[LayerArraySpec],
    head_scale: f32,
    head_spec: Option<&HeadSpec>,
    condition_dsp: Option<Box<WaveNet>>,
    weights: &[f32],
) -> Result<WaveNet, String> {
    if let Some(cd) = &condition_dsp {
        if cd.in_channels() != in_channels {
            return Err(format!(
                "input channels of WaveNet ({in_channels}) don't match input channels of condition DSP ({})",
                cd.in_channels()
            ));
        }
        for (idx, spec) in specs.iter().enumerate() {
            if spec.condition_size != cd.out_channels() {
                return Err(format!(
                    "condition_size of layer array {idx} ({}) doesn't match output channels of condition DSP ({})",
                    spec.condition_size,
                    cd.out_channels()
                ));
            }
        }
    }

    let mut layer_arrays = Vec::with_capacity(specs.len());
    let mut prev_head_size: Option<usize> = None;
    for (idx, spec) in specs.iter().enumerate() {
        // Head-input chaining requires size match with the previous array.
        if let Some(prev) = prev_head_size {
            if prev != spec.head_output_size() {
                return Err(format!(
                    "layer array {idx}: head chaining size mismatch: previous head_size {prev} vs head accumulator {}",
                    spec.head_output_size()
                ));
            }
        }
        prev_head_size = Some(spec.head_size);
        layer_arrays.push(build_layer_array(spec)?);
    }

    let post_head = match head_spec {
        Some(hs) => {
            let n = hs.kernel_sizes.len();
            let mut convs = Vec::new();
            let mut acts = Vec::new();
            let mut cin = hs.in_channels;
            for (i, &k) in hs.kernel_sizes.iter().enumerate() {
                let cout = if i + 1 == n { hs.out_channels } else { hs.channels };
                acts.push(hs.activation.clone());
                convs.push(Conv1D::new(cin, cout, k, true, 1, 1)?);
                cin = cout;
            }
            Some(PostHead {
                convs,
                activations: acts,
                in_channels: hs.in_channels,
                scratch: Mat::new(hs.in_channels, 0),
            })
        }
        None => None,
    };

    let out_channels = match head_spec {
        Some(hs) => hs.out_channels,
        None => specs.last().expect("non-empty").head_size,
    };

    let mut net = WaveNet {
        layer_arrays,
        head_scale,
        post_head,
        condition_dsp,
        in_channels,
        out_channels,
        condition_input: Mat::new(in_channels, 0),
        output: Mat::new(out_channels, 0),
        prewarm_samples: 0,
        max_frames: 0,
    };

    // Set weights: layer arrays, post head, then the trailing head_scale.
    // (The condition DSP has its own nested weight array.)
    let mut w = Weights::new(weights);
    for la in &mut net.layer_arrays {
        la.set_weights(&mut w)?;
    }
    if let Some(h) = &mut net.post_head {
        h.set_weights(&mut w)?;
    }
    net.head_scale = w.next()?;
    w.finish()?;

    // Prewarm samples: condition DSP prewarm (or 1) + sum of receptive
    // fields (+ post head rf - 1) — mirrors the C++ WaveNet constructor.
    let mut prewarm = match &net.condition_dsp {
        Some(cd) => cd.prewarm_samples(),
        None => 1,
    };
    for la in &net.layer_arrays {
        prewarm += la.receptive_field();
    }
    if let Some(h) = &net.post_head {
        prewarm += h.receptive_field() - 1;
    }
    net.prewarm_samples = prewarm;

    Ok(net)
}

/// Parse a WaveNet `config` object (the value of the top-level "config" key)
/// and build the model with `weights`.
pub(crate) fn parse(config: &Value, weights: &[f32]) -> Result<WaveNet, String> {
    let specs = parse_layer_array_specs(config)?;
    let head_scale = get(config, "head_scale")
        .and_then(|v| v.as_f64())
        .ok_or("WaveNet config missing head_scale")? as f32;
    let in_channels = match get(config, "in_channels") {
        Some(v) => as_usize(v, "in_channels")?,
        None => 1,
    };
    let head_spec = parse_head_spec(config, &specs)?;
    let condition_dsp = parse_condition_dsp(config)?;
    build(in_channels, &specs, head_scale, head_spec.as_ref(), condition_dsp, weights)
}

// Slimmable WaveNet ==========================================================

/// True when the WaveNet config carries a `slimmable` block using
/// `slice_channels_uniform` on any layer array (the `SlimmableWavenet`
/// routing in the C++ core's `create_config`).
pub(crate) fn config_is_slimmable(config: &Value) -> Result<bool, String> {
    let Some(layers) = get(config, "layers").and_then(|v| v.as_array()) else {
        return Ok(false);
    };
    for lc in layers {
        let Some(s) = get(lc, "slimmable").filter(|s| s.is_object()) else {
            continue;
        };
        let method = s.get("method").and_then(|v| v.as_str()).unwrap_or("");
        if method == "slice_channels_uniform" {
            return Ok(true);
        }
        if !method.is_empty() {
            return Err(format!("SlimmableWavenet: unsupported slimmable method '{method}'"));
        }
    }
    Ok(false)
}

/// A WaveNet with per-layer-array dynamic channel reduction, mirroring
/// `NAM/wavenet/slimmable.{h,cpp}` (`slice_channels_uniform`). Holds the
/// full-size specs + weights; [`SlimmableWaveNet::set_slimmable_size`] maps
/// a ratio in [0, 1] to per-array channel counts, slices the weight subset,
/// and rebuilds the active model. Defaults to full size.
pub(crate) struct SlimmableWaveNet {
    specs: Vec<LayerArraySpec>,
    /// Per-array sorted allowed channel counts (empty = non-slimmable array).
    allowed: Vec<Vec<usize>>,
    in_channels: usize,
    head_scale: f32,
    condition_json: Option<Value>,
    full_weights: Vec<f32>,
    pub current: WaveNet,
    current_channels: Vec<usize>,
}

impl SlimmableWaveNet {
    /// Ratio [0,1] → per-array target channel counts.
    /// Matches the C++/Python: idx = min(floor(ratio * len), len - 1).
    fn channels_for_size(&self, val: f64) -> Vec<usize> {
        self.specs
            .iter()
            .zip(self.allowed.iter())
            .map(|(spec, allowed)| {
                if allowed.is_empty() {
                    spec.channels
                } else {
                    let idx = ((val * allowed.len() as f64).floor() as isize)
                        .clamp(0, allowed.len() as isize - 1) as usize;
                    allowed[idx]
                }
            })
            .collect()
    }

    /// Select the model size. Returns true when the active model changed
    /// (the caller should then reset/prewarm before processing).
    pub fn set_slimmable_size(&mut self, val: f64) -> Result<bool, String> {
        let targets = self.channels_for_size(val);
        if targets == self.current_channels {
            return Ok(false);
        }
        let full = self
            .specs
            .iter()
            .zip(targets.iter())
            .all(|(spec, &t)| t == spec.channels);
        let condition_dsp = match &self.condition_json {
            Some(cd) => parse_condition_dsp_value(cd)?,
            None => None,
        };
        let net = if full {
            build(
                self.in_channels,
                &self.specs,
                self.head_scale,
                None,
                condition_dsp,
                &self.full_weights,
            )?
        } else {
            let slim_specs = slim_specs_for_channels(&self.specs, &targets);
            let slim_weights = extract_slimmed_weights(&self.specs, &self.full_weights, &targets)?;
            build(
                self.in_channels,
                &slim_specs,
                self.head_scale,
                None,
                condition_dsp,
                &slim_weights,
            )?
        };
        self.current = net;
        self.current_channels = targets;
        Ok(true)
    }
}

fn parse_condition_dsp_value(cd: &Value) -> Result<Option<Box<WaveNet>>, String> {
    // Wrap in a synthetic config so parse_condition_dsp's logic is reused.
    let arch = cd
        .get("architecture")
        .and_then(|v| v.as_str())
        .ok_or("condition_dsp missing 'architecture'")?;
    if arch != "WaveNet" {
        return Err(format!(
            "condition_dsp architecture '{arch}' is not supported by the pure-Rust engine"
        ));
    }
    let cfg = cd.get("config").ok_or("condition_dsp missing 'config'")?;
    let weights = super::parse_weights(cd)?;
    Ok(Some(Box::new(parse(cfg, &weights)?)))
}

/// Parse a slimmable WaveNet config, validating the constraints the C++
/// implementation requires (checked here at load time so that
/// `set_slimmable_size` cannot fail on the audio path).
pub(crate) fn parse_slimmable(config: &Value, weights: &[f32]) -> Result<SlimmableWaveNet, String> {
    let specs = parse_layer_array_specs(config)?;
    let head_scale = get(config, "head_scale")
        .and_then(|v| v.as_f64())
        .ok_or("WaveNet config missing head_scale")? as f32;
    let in_channels = match get(config, "in_channels") {
        Some(v) => as_usize(v, "in_channels")?,
        None => 1,
    };
    if get(config, "head").is_some() {
        return Err("SlimmableWavenet: post-stack head is not supported".into());
    }

    // Per-array allowed channel lists.
    let layers_json = get(config, "layers").and_then(|v| v.as_array()).expect("validated");
    let mut allowed: Vec<Vec<usize>> = Vec::with_capacity(specs.len());
    for (idx, lc) in layers_json.iter().enumerate() {
        let mut arr: Vec<usize> = Vec::new();
        if let Some(s) = get(lc, "slimmable").filter(|s| s.is_object()) {
            let method = s.get("method").and_then(|v| v.as_str()).unwrap_or("");
            if method == "slice_channels_uniform" {
                if let Some(ac) = s
                    .get("kwargs")
                    .and_then(|k| k.get("allowed_channels"))
                    .and_then(|v| v.as_array())
                {
                    for ch in ac {
                        arr.push(as_usize(ch, "allowed_channels entry")?);
                    }
                } else {
                    // Missing allowed_channels: assume 1..=channels.
                    arr.extend(1..=specs[idx].channels);
                }
                for w in arr.windows(2) {
                    if w[1] <= w[0] {
                        return Err(
                            "SlimmableWavenet: allowed_channels must be sorted ascending".into()
                        );
                    }
                }
                if arr.last() != Some(&specs[idx].channels) {
                    return Err("SlimmableWavenet: last allowed_channels entry must equal the full channel count for that array".into());
                }
            }
        }
        allowed.push(arr);
    }
    if allowed.iter().all(|a| a.is_empty()) {
        return Err("SlimmableWavenet: at least one layer array must have allowed_channels".into());
    }

    // Constraints of the C++ weight extraction, validated up front.
    for spec in &specs {
        if spec.head_kernel_size != 1 {
            return Err("SlimmableWavenet: head rechannel kernel_size must be 1 (slimming with head kernel_size > 1 is not implemented)".into());
        }
        if spec.groups_input != 1 {
            return Err("SlimmableWavenet: groups_input > 1 not supported".into());
        }
        if spec.groups_input_mixin != 1 {
            return Err("SlimmableWavenet: groups_input_mixin > 1 not supported".into());
        }
        if spec.layer1x1_active && spec.layer1x1_groups != 1 {
            return Err("SlimmableWavenet: layer1x1 groups > 1 not supported".into());
        }
        if spec.head1x1_active && spec.head1x1_groups != 1 {
            return Err("SlimmableWavenet: head1x1 groups > 1 not supported".into());
        }
        for p in &spec.films {
            if p.active && p.groups != 1 {
                return Err("SlimmableWavenet: FiLM groups > 1 not supported".into());
            }
        }
    }

    let condition_json = get(config, "condition_dsp").cloned();
    let condition_dsp = match &condition_json {
        Some(cd) => parse_condition_dsp_value(cd)?,
        None => None,
    };

    // Build with full channel counts as default.
    let current = build(in_channels, &specs, head_scale, None, condition_dsp, weights)?;
    let current_channels = specs.iter().map(|s| s.channels).collect();

    Ok(SlimmableWaveNet {
        specs,
        allowed,
        in_channels,
        head_scale,
        condition_json,
        full_weights: weights.to_vec(),
        current,
        current_channels,
    })
}

/// Compute the slim bottleneck (mirrors `compute_slim_bottleneck`).
fn slim_bottleneck(spec: &LayerArraySpec, new_channels: usize) -> usize {
    if !spec.layer1x1_active {
        new_channels // bottleneck must equal channels when layer1x1 inactive
    } else {
        (spec.bottleneck * new_channels / spec.channels).max(1)
    }
}

/// Build modified specs for the target per-array channel counts
/// (mirrors `modify_params_for_channels`).
fn slim_specs_for_channels(specs: &[LayerArraySpec], targets: &[usize]) -> Vec<LayerArraySpec> {
    let n = specs.len();
    specs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            let mut s = spec.clone();
            s.channels = targets[i];
            s.bottleneck = slim_bottleneck(spec, targets[i]);
            if i > 0 {
                s.input_size = targets[i - 1];
            }
            if i < n - 1 {
                s.head_size = targets[i + 1];
            }
            s
        })
        .collect()
}

/// Sequential reader over the full weight vector for slim extraction.
struct Src<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> Src<'a> {
    fn take(&mut self) -> Result<f32, String> {
        let v = self
            .data
            .get(self.pos)
            .copied()
            .ok_or("SlimmableWavenet: weight underflow during extraction")?;
        self.pos += 1;
        Ok(v)
    }
}

/// Take the first `slim_out` rows / `slim_in` cols of a Conv1x1 weight block
/// (groups=1 layout: row-major out×in, then optional bias(out)).
fn extract_conv1x1(
    src: &mut Src,
    full_in: usize,
    full_out: usize,
    slim_in: usize,
    slim_out: usize,
    bias: bool,
    dst: &mut Vec<f32>,
) -> Result<(), String> {
    for i in 0..full_out {
        for j in 0..full_in {
            let w = src.take()?;
            if i < slim_out && j < slim_in {
                dst.push(w);
            }
        }
    }
    if bias {
        for i in 0..full_out {
            let b = src.take()?;
            if i < slim_out {
                dst.push(b);
            }
        }
    }
    Ok(())
}

/// Conv1D layout (groups=1): per out, per in, per kernel tap; then bias(out).
fn extract_conv1d(
    src: &mut Src,
    full_in: usize,
    full_out: usize,
    slim_in: usize,
    slim_out: usize,
    kernel_size: usize,
    dst: &mut Vec<f32>,
) -> Result<(), String> {
    for i in 0..full_out {
        for j in 0..full_in {
            for _ in 0..kernel_size {
                let w = src.take()?;
                if i < slim_out && j < slim_in {
                    dst.push(w);
                }
            }
        }
    }
    // Bias is always present for conv in WaveNet layers.
    for i in 0..full_out {
        let b = src.take()?;
        if i < slim_out {
            dst.push(b);
        }
    }
    Ok(())
}

fn copy_weights(src: &mut Src, n: usize, dst: &mut Vec<f32>) -> Result<(), String> {
    for _ in 0..n {
        dst.push(src.take()?);
    }
    Ok(())
}

/// Walk the full weight vector in `set_weights` order and extract the subset
/// for the target channel counts — a faithful port of the C++
/// `extract_slimmed_weights` (including its take-the-first-rows treatment of
/// gated conv blocks and shift-FiLM blocks).
fn extract_slimmed_weights(
    specs: &[LayerArraySpec],
    full_weights: &[f32],
    targets: &[usize],
) -> Result<Vec<f32>, String> {
    let mut slim = Vec::new();
    let mut src = Src {
        data: full_weights,
        pos: 0,
    };
    let n = specs.len();

    for (arr, spec) in specs.iter().enumerate() {
        let full_ch = spec.channels;
        let full_bn = spec.bottleneck;
        let slim_ch = targets[arr];
        let slim_bn = slim_bottleneck(spec, slim_ch);
        let cond = spec.condition_size;

        // Input size: first array keeps original, others follow prev target.
        let slim_input_size = if arr == 0 { spec.input_size } else { targets[arr - 1] };
        // Head size: intermediate arrays match next array's channels.
        let slim_head_size = if arr < n - 1 { targets[arr + 1] } else { spec.head_size };

        let full_head_out = if spec.head1x1_active { spec.head1x1_out } else { full_bn };
        let slim_head_out = if spec.head1x1_active { spec.head1x1_out } else { slim_bn };

        // rechannel: Conv1x1(input_size -> channels, no bias)
        extract_conv1x1(&mut src, spec.input_size, full_ch, slim_input_size, slim_ch, false, &mut slim)?;

        for li in 0..spec.dilations.len() {
            let gated = spec.gating_modes[li] != GatingMode::None;
            let full_bg = if gated { 2 * full_bn } else { full_bn };
            let slim_bg = if gated { 2 * slim_bn } else { slim_bn };

            // conv: Conv1D(channels -> B_g, K, bias=true)
            extract_conv1d(&mut src, full_ch, full_bg, slim_ch, slim_bg, spec.kernel_sizes[li], &mut slim)?;
            // input_mixin: Conv1x1(condition_size -> B_g, no bias)
            extract_conv1x1(&mut src, cond, full_bg, cond, slim_bg, false, &mut slim)?;
            // layer1x1 (optional): Conv1x1(B -> C, bias=true)
            if spec.layer1x1_active {
                extract_conv1x1(&mut src, full_bn, full_ch, slim_bn, slim_ch, true, &mut slim)?;
            }
            // head1x1 (optional): Conv1x1(B -> head1x1_out, bias=true)
            if spec.head1x1_active {
                extract_conv1x1(&mut src, full_bn, spec.head1x1_out, slim_bn, spec.head1x1_out, true, &mut slim)?;
            }

            // FiLM objects, in set_weights order. Each is a
            // Conv1x1(cond -> (shift?2:1)*dim, bias=true).
            let film_dims: [(usize, usize, usize); 8] = [
                (FILM_CONV_PRE, full_ch, slim_ch),
                (FILM_CONV_POST, full_bg, slim_bg),
                (FILM_INPUT_MIXIN_PRE, cond, cond),
                (FILM_INPUT_MIXIN_POST, full_bg, slim_bg),
                (FILM_ACTIVATION_PRE, full_bg, slim_bg),
                (FILM_ACTIVATION_POST, full_bn, slim_bn),
                (FILM_LAYER1X1_POST, full_ch, slim_ch),
                (FILM_HEAD1X1_POST, spec.head1x1_out, spec.head1x1_out),
            ];
            for (idx, full_dim, slim_dim) in film_dims {
                let p = spec.films[idx];
                let module_active = match idx {
                    FILM_LAYER1X1_POST => p.active && spec.layer1x1_active,
                    FILM_HEAD1X1_POST => p.active && spec.head1x1_active,
                    _ => p.active,
                };
                if !module_active {
                    continue;
                }
                let mult = if p.shift { 2 } else { 1 };
                if full_dim == slim_dim {
                    // Unchanged: straight copy (weights + bias).
                    let dim = mult * full_dim;
                    copy_weights(&mut src, cond * dim + dim, &mut slim)?;
                } else {
                    extract_conv1x1(&mut src, cond, mult * full_dim, cond, mult * slim_dim, true, &mut slim)?;
                }
            }
        }

        // head_rechannel: kernel_size 1 (validated) — Conv1x1 layout.
        extract_conv1x1(&mut src, full_head_out, spec.head_size, slim_head_out, slim_head_size, spec.head_bias, &mut slim)?;
    }

    // head_scale: 1 float, copy as-is.
    slim.push(src.take()?);
    if src.pos != full_weights.len() {
        return Err(format!(
            "SlimmableWavenet: extraction consumed {} of {} weights",
            src.pos,
            full_weights.len()
        ));
    }
    Ok(slim)
}
