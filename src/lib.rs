//! Safe Rust bindings to [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).
//!
//! Loads `.nam` model files and runs inference for guitar amp/pedal modeling.
//!
//! [`NamModel`] runs on one of three engines behind the same API:
//!
//! - [valverig-nam](https://github.com/ValveRig/valverig-nam)'s, the
//!   default on every target, wasm32 included: pure Rust, allocation-free in
//!   the audio callback, about twice as fast as the C++ core on WaveNet
//!   captures and within ~1e-6 of it (tests/parity.rs);
//! - this crate's own port, [`pure`], for what valverig doesn't do: a
//!   reduced size of a slimmable WaveNet;
//! - the vendored C++ core over FFI, with the `cpp` feature (native only).
//!   It is the oracle the parity tests hold both Rust engines to; building
//!   it costs a C++20 compile of the core with Eigen.
//!
//! # Example
//!
//! ```no_run
//! use neural_amp_modeler::NamModel;
//!
//! let mut model = NamModel::load("model.nam").unwrap();
//! model.reset(48000.0, 512);
//!
//! let input = vec![0.0f64; 512];
//! let mut output = vec![0.0f64; 512];
//! model.process(&input, &mut output);
//! ```

use std::path::Path;

pub mod pure;

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Sample rate the model was trained at, or `None` if unknown.
    pub expected_sample_rate: Option<f64>,
    /// Loudness in dB, if the model provides it.
    pub loudness: Option<f64>,
    /// Input level in dBu, if the model provides it.
    pub input_level: Option<f64>,
    /// Output level in dBu, if the model provides it.
    pub output_level: Option<f64>,
    /// Number of input channels.
    pub input_channels: usize,
    /// Number of output channels.
    pub output_channels: usize,
}

// ============================================================================
// Native implementation: FFI over the vendored C++ NeuralAmpModelerCore.
// ============================================================================

#[cfg(all(feature = "cpp", not(target_arch = "wasm32")))]
mod ffi {
    use std::os::raw::{c_char, c_double, c_int};

    #[repr(C)]
    pub(crate) struct NamModel {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub(crate) struct NamLoadResult {
        pub model: *mut NamModel,
        pub error: *mut c_char,
    }

    unsafe extern "C" {
        pub fn nam_load(path: *const c_char) -> NamLoadResult;
        pub fn nam_load_from_json(json: *const c_char) -> NamLoadResult;
        pub fn nam_free(model: *mut NamModel);
        pub fn nam_free_error_string(error: *mut c_char);
        pub fn nam_process(
            model: *mut NamModel,
            input: *const c_double,
            output: *mut c_double,
            num_frames: c_int,
        );
        pub fn nam_reset(model: *mut NamModel, sample_rate: c_double, max_buffer_size: c_int);
        pub fn nam_get_expected_sample_rate(model: *const NamModel) -> c_double;
        pub fn nam_has_loudness(model: *const NamModel) -> c_int;
        pub fn nam_get_loudness(model: *const NamModel) -> c_double;
        pub fn nam_num_input_channels(model: *const NamModel) -> c_int;
        pub fn nam_num_output_channels(model: *const NamModel) -> c_int;
        pub fn nam_has_input_level(model: *const NamModel) -> c_int;
        pub fn nam_get_input_level(model: *const NamModel) -> c_double;
        pub fn nam_has_output_level(model: *const NamModel) -> c_int;
        pub fn nam_get_output_level(model: *const NamModel) -> c_double;
        pub fn nam_set_slimmable_size(model: *mut NamModel, val: c_double) -> c_int;
    }
}

/// A loaded Neural Amp Modeler model ready for inference.
///
/// Thread safety: `NamModel` is `Send` but not `Sync`. You can move it
/// between threads, but must not share it across threads without external
/// synchronisation (the underlying DSP object is not thread-safe).
#[cfg(all(feature = "cpp", not(target_arch = "wasm32")))]
pub struct NamModel {
    ptr: *mut ffi::NamModel,
}

// NAM models can be moved between threads safely.
// The underlying C++ DSP object is single-threaded but not tied to a thread.
#[cfg(all(feature = "cpp", not(target_arch = "wasm32")))]
unsafe impl Send for NamModel {}

#[cfg(all(feature = "cpp", not(target_arch = "wasm32")))]
impl NamModel {
    fn from_load_result(result: ffi::NamLoadResult) -> Result<Self, String> {
        if !result.error.is_null() {
            let msg = unsafe {
                let s = std::ffi::CStr::from_ptr(result.error)
                    .to_string_lossy()
                    .into_owned();
                ffi::nam_free_error_string(result.error);
                s
            };
            return Err(msg);
        }

        if result.model.is_null() {
            return Err("nam_load returned null model without error".to_string());
        }

        Ok(Self { ptr: result.model })
    }

    /// Load a `.nam` model file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| "Path contains invalid UTF-8".to_string())?;
        let c_path =
            std::ffi::CString::new(path_str).map_err(|_| "Path contains null byte".to_string())?;

        let result = unsafe { ffi::nam_load(c_path.as_ptr()) };
        Self::from_load_result(result)
    }

    /// Load a model from the raw bytes of a `.nam` file (UTF-8 JSON).
    ///
    /// This avoids any filesystem access — useful when models arrive over
    /// the network or are embedded in the binary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let json =
            std::str::from_utf8(bytes).map_err(|e| format!("Model is not valid UTF-8: {e}"))?;
        Self::from_json(json)
    }

    /// Load a model from `.nam` JSON text.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let c_json = std::ffi::CString::new(json)
            .map_err(|_| "Model JSON contains null byte".to_string())?;
        let result = unsafe { ffi::nam_load_from_json(c_json.as_ptr()) };
        Self::from_load_result(result)
    }

    /// Reset the model for a given sample rate and buffer size, then prewarm.
    ///
    /// Must be called before the first `process()` call, and whenever the
    /// host sample rate or buffer size changes.
    pub fn reset(&mut self, sample_rate: f64, max_buffer_size: usize) {
        unsafe {
            ffi::nam_reset(self.ptr, sample_rate, max_buffer_size as i32);
        }
    }

    /// Run inference on a block of mono audio.
    ///
    /// `input` and `output` must be the same length. In-place processing
    /// (same slice for both) is supported.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must be the same length"
        );
        unsafe {
            ffi::nam_process(
                self.ptr,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as i32,
            );
        }
    }

    /// Get the sample rate the model was trained at, or `None` if unknown.
    pub fn expected_sample_rate(&self) -> Option<f64> {
        let sr = unsafe { ffi::nam_get_expected_sample_rate(self.ptr) };
        if sr < 0.0 { None } else { Some(sr) }
    }

    /// Get the model's loudness in dB, if available.
    pub fn loudness(&self) -> Option<f64> {
        if unsafe { ffi::nam_has_loudness(self.ptr) } != 0 {
            Some(unsafe { ffi::nam_get_loudness(self.ptr) })
        } else {
            None
        }
    }

    /// Get the input level in dBu, if available.
    pub fn input_level(&self) -> Option<f64> {
        if unsafe { ffi::nam_has_input_level(self.ptr) } != 0 {
            Some(unsafe { ffi::nam_get_input_level(self.ptr) })
        } else {
            None
        }
    }

    /// Get the output level in dBu, if available.
    pub fn output_level(&self) -> Option<f64> {
        if unsafe { ffi::nam_has_output_level(self.ptr) } != 0 {
            Some(unsafe { ffi::nam_get_output_level(self.ptr) })
        } else {
            None
        }
    }

    /// Number of input channels (typically 1 for guitar models).
    pub fn input_channels(&self) -> usize {
        unsafe { ffi::nam_num_input_channels(self.ptr) as usize }
    }

    /// Number of output channels (typically 1 for guitar models).
    pub fn output_channels(&self) -> usize {
        unsafe { ffi::nam_num_output_channels(self.ptr) as usize }
    }

    /// Get all available metadata for this model.
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            expected_sample_rate: self.expected_sample_rate(),
            loudness: self.loudness(),
            input_level: self.input_level(),
            output_level: self.output_level(),
            input_channels: self.input_channels(),
            output_channels: self.output_channels(),
        }
    }

    /// Select the slimmable size for models that support dynamic size
    /// reduction (`SlimmableContainer` and slimmable WaveNets): `val` in
    /// [0.0, 1.0], where 1.0 is full size (the default). Returns `true`
    /// if the model is slimmable, `false` for ordinary models (no-op).
    /// Not real-time safe; the new size takes effect on the next
    /// `process()` call.
    pub fn set_slimmable_size(&mut self, val: f64) -> bool {
        unsafe { ffi::nam_set_slimmable_size(self.ptr, val) != 0 }
    }
}

#[cfg(all(feature = "cpp", not(target_arch = "wasm32")))]
impl Drop for NamModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::nam_free(self.ptr) };
        }
    }
}

// ============================================================================
// The Rust engines: wasm32, or native without the `cpp` feature.
//
// valverig-nam (MIT, github.com/ValveRig/valverig-nam) runs the model: on
// WaveNet captures it is about twice as fast as the C++ core, and it agrees
// with the C++ core to ~1e-6 (tests/parity.rs). Two things it does not do
// fall back to this crate's own port, `pure`: the pre-A2 `Linear` and
// `ConvNet` architectures (which `pure` does not run either, so the load
// still fails, with valverig's error), and selecting a reduced size of a
// slimmable *WaveNet* (valverig slims only a `SlimmableContainer`).
// ============================================================================

#[cfg(not(all(feature = "cpp", not(target_arch = "wasm32"))))]
const DEFAULT_MAX_BUFFER_SIZE: usize = 4096;

#[cfg(not(all(feature = "cpp", not(target_arch = "wasm32"))))]
enum Engine {
    /// valverig's engine, with the f32 blocks it reads and writes: our API
    /// is f64, and the copies happen through these, sized in `reset`, so
    /// `process` never allocates.
    Valverig {
        model: Box<valverig_nam::loader::Model>,
        input: Vec<f32>,
        output: Vec<f32>,
    },
    Pure(Box<pure::PureNamModel>),
}

/// A loaded Neural Amp Modeler model ready for inference.
///
/// Backed by valverig-nam's engine, or this crate's [`pure`] port where
/// valverig cannot run the model (see the module notes). With the `cpp`
/// feature on a native target it is the C++ core instead. On
/// `wasm32-unknown-unknown` load models with [`NamModel::from_bytes`] /
/// [`NamModel::from_json`]: there is no filesystem.
#[cfg(not(all(feature = "cpp", not(target_arch = "wasm32"))))]
pub struct NamModel {
    engine: Engine,
    /// The model's JSON, kept only for a slimmable WaveNet: selecting a
    /// reduced size rebuilds it on the `pure` engine.
    slimmable_source: Option<String>,
    sample_rate: f64,
    max_buffer_size: usize,
    expected_sample_rate: Option<f64>,
    loudness: Option<f64>,
    input_level: Option<f64>,
    output_level: Option<f64>,
}

#[cfg(not(all(feature = "cpp", not(target_arch = "wasm32"))))]
impl NamModel {
    /// Load a `.nam` model file from disk.
    ///
    /// Fails at runtime on `wasm32-unknown-unknown` (no filesystem); use
    /// [`NamModel::from_bytes`] there.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| format!("Failed to read model file: {e}"))?;
        Self::from_bytes(&bytes)
    }

    /// Load a model from the raw bytes of a `.nam` file (UTF-8 JSON).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let json =
            std::str::from_utf8(bytes).map_err(|e| format!("Model is not valid UTF-8: {e}"))?;
        Self::from_json(json)
    }

    /// Load a model from `.nam` JSON text.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let mut model = match valverig_nam::loader::Model::from_json(json) {
            Ok(model) => Self::from_valverig(model, json),
            Err(valverig) => {
                let pure = pure::PureNamModel::from_json(json).map_err(|e| {
                    format!("Failed to load NAM model: {valverig} (and the fallback engine: {e})")
                })?;
                Self::from_pure(pure)
            }
        };
        // Usable straight away, as the C++ core is: sized for the default
        // block and settled. A host calls `reset` with its own rate and size.
        model.reset(model.expected_sample_rate.unwrap_or(48_000.0), DEFAULT_MAX_BUFFER_SIZE);
        Ok(model)
    }

    fn from_valverig(model: valverig_nam::loader::Model, json: &str) -> Self {
        let meta = &model.metadata;
        let (loudness, input_level, output_level) =
            (meta.loudness, meta.input_level_dbu, meta.output_level_dbu);
        let expected_sample_rate = model.expected_sample_rate;
        Self {
            slimmable_source: is_slimmable_wavenet(json).then(|| json.to_owned()),
            engine: Engine::Valverig {
                model: Box::new(model),
                input: Vec::new(),
                output: Vec::new(),
            },
            sample_rate: expected_sample_rate.unwrap_or(48_000.0),
            max_buffer_size: 0,
            expected_sample_rate,
            loudness,
            input_level,
            output_level,
        }
    }

    fn from_pure(model: pure::PureNamModel) -> Self {
        Self {
            expected_sample_rate: model.expected_sample_rate(),
            loudness: model.loudness(),
            input_level: model.input_level(),
            output_level: model.output_level(),
            sample_rate: model.expected_sample_rate().unwrap_or(48_000.0),
            max_buffer_size: 0,
            slimmable_source: None,
            engine: Engine::Pure(Box::new(model)),
        }
    }

    /// Reset the model for a given sample rate and buffer size, then prewarm.
    ///
    /// Allocating: call it before the audio thread starts, and whenever the
    /// host sample rate or buffer size changes.
    pub fn reset(&mut self, sample_rate: f64, max_buffer_size: usize) {
        let max_buffer_size = max_buffer_size.max(1);
        self.sample_rate = sample_rate;
        self.max_buffer_size = max_buffer_size;
        match &mut self.engine {
            Engine::Valverig { model, input, output } => {
                model.reset(sample_rate, max_buffer_size);
                input.resize(max_buffer_size, 0.0);
                output.resize(max_buffer_size, 0.0);
            }
            Engine::Pure(model) => model.reset(sample_rate, max_buffer_size),
        }
    }

    /// Run inference on a block of mono audio.
    ///
    /// `input` and `output` must be the same length. Allocation-free once
    /// [`NamModel::reset`] has run; a block longer than the size `reset`
    /// was given is processed in pieces of that size.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) {
        assert_eq!(
            input.len(),
            output.len(),
            "input and output must be the same length"
        );
        match &mut self.engine {
            Engine::Valverig {
                model,
                input: scratch_in,
                output: scratch_out,
            } => {
                for (dry, wet) in input
                    .chunks(scratch_in.len().max(1))
                    .zip(output.chunks_mut(scratch_out.len().max(1)))
                {
                    let n = dry.len();
                    for (s, &x) in scratch_in[..n].iter_mut().zip(dry) {
                        *s = x as f32;
                    }
                    model.process_mono(&scratch_in[..n], &mut scratch_out[..n]);
                    for (y, &s) in wet.iter_mut().zip(&scratch_out[..n]) {
                        *y = f64::from(s);
                    }
                }
            }
            Engine::Pure(model) => model.process(input, output),
        }
    }

    /// Get the sample rate the model was trained at, or `None` if unknown.
    pub fn expected_sample_rate(&self) -> Option<f64> {
        self.expected_sample_rate
    }

    /// Get the model's loudness in dB, if available.
    pub fn loudness(&self) -> Option<f64> {
        self.loudness
    }

    /// Get the input level in dBu, if available.
    pub fn input_level(&self) -> Option<f64> {
        self.input_level
    }

    /// Get the output level in dBu, if available.
    pub fn output_level(&self) -> Option<f64> {
        self.output_level
    }

    /// Number of input channels (typically 1 for guitar models).
    pub fn input_channels(&self) -> usize {
        match &self.engine {
            Engine::Valverig { model, .. } => model.in_channels(),
            Engine::Pure(model) => model.input_channels(),
        }
    }

    /// Number of output channels (typically 1 for guitar models).
    pub fn output_channels(&self) -> usize {
        match &self.engine {
            Engine::Valverig { model, .. } => model.out_channels(),
            Engine::Pure(model) => model.output_channels(),
        }
    }

    /// Get all available metadata for this model.
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            expected_sample_rate: self.expected_sample_rate(),
            loudness: self.loudness(),
            input_level: self.input_level(),
            output_level: self.output_level(),
            input_channels: self.input_channels(),
            output_channels: self.output_channels(),
        }
    }

    /// Select the slimmable size for models that support dynamic size
    /// reduction (`SlimmableContainer` and slimmable WaveNets): `val` in
    /// [0.0, 1.0], where 1.0 is full size (the default). Returns `true`
    /// if the model is slimmable, `false` for ordinary models (no-op).
    ///
    /// Allocating (not real-time safe). A reduced size of a slimmable
    /// WaveNet moves the model onto the `pure` engine, which can slim one.
    pub fn set_slimmable_size(&mut self, val: f64) -> bool {
        // A SlimmableContainer slims on valverig; an ordinary model on
        // valverig is not slimmable.
        if let Engine::Valverig { model, .. } = &mut self.engine {
            if model.set_slimmable_size(val).is_ok() {
                return true;
            }
            if self.slimmable_source.is_none() {
                return false;
            }
        }
        // A slimmable WaveNet: valverig runs it at full size, the port at
        // any size. Swap engines only once the new one has taken the size.
        let Some(json) = self.slimmable_source.as_deref() else {
            // On the port without a source: a model valverig never loaded.
            return match &mut self.engine {
                Engine::Pure(model) => model.set_slimmable_size(val),
                Engine::Valverig { .. } => false,
            };
        };
        if val >= 1.0 {
            if matches!(self.engine, Engine::Pure(_)) {
                let Ok(model) = valverig_nam::loader::Model::from_json(json) else {
                    return false;
                };
                self.engine = Engine::Valverig {
                    model: Box::new(model),
                    input: Vec::new(),
                    output: Vec::new(),
                };
                self.reset(self.sample_rate, self.max_buffer_size);
            }
            return true;
        }
        let Ok(mut pure) = pure::PureNamModel::from_json(json) else {
            return false;
        };
        if !pure.set_slimmable_size(val) {
            return false;
        }
        pure.reset(self.sample_rate, self.max_buffer_size.max(1));
        self.engine = Engine::Pure(Box::new(pure));
        true
    }
}

/// Whether a model is a WaveNet with slimmable layers (`slimmable` on a
/// layer array of its config): the size-switchable shape valverig runs at
/// full size only.
#[cfg(not(all(feature = "cpp", not(target_arch = "wasm32"))))]
fn is_slimmable_wavenet(json: &str) -> bool {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return false;
    };
    v.get("architecture").and_then(|a| a.as_str()) == Some("WaveNet")
        && v.pointer("/config/layers")
            .and_then(|l| l.as_array())
            .is_some_and(|layers| layers.iter().any(|l| l.get("slimmable").is_some()))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_returns_error() {
        let result = NamModel::load("/nonexistent/path.nam");
        assert!(result.is_err());
    }

    #[test]
    fn model_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NamModel>();
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(NamModel::from_bytes(b"not json").is_err());
    }
}
