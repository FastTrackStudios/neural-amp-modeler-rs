//! Safe Rust bindings to [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).
//!
//! Loads `.nam` model files and runs inference for guitar amp/pedal modeling.
//!
//! On native targets, [`NamModel`] wraps the vendored C++ core (compiled by
//! `build.rs`). On `wasm32` targets — where no C++ toolchain/stdlib is
//! available — the same API is backed by the [`pure`] Rust inference engine,
//! which is validated against the C++ core by this crate's parity tests.
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

#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
pub struct NamModel {
    ptr: *mut ffi::NamModel,
}

// NAM models can be moved between threads safely.
// The underlying C++ DSP object is single-threaded but not tied to a thread.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for NamModel {}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NamModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::nam_free(self.ptr) };
        }
    }
}

// ============================================================================
// wasm32 implementation: pure-Rust inference engine.
// ============================================================================

/// A loaded Neural Amp Modeler model ready for inference.
///
/// On wasm32 this is backed by the pure-Rust engine in [`pure`]; load models
/// with [`NamModel::from_bytes`] / [`NamModel::from_json`] (there is no
/// filesystem on `wasm32-unknown-unknown`).
#[cfg(target_arch = "wasm32")]
pub struct NamModel {
    inner: pure::PureNamModel,
}

#[cfg(target_arch = "wasm32")]
impl NamModel {
    /// Load a `.nam` model file from disk.
    ///
    /// Compiles on wasm for API parity but fails at runtime on
    /// `wasm32-unknown-unknown` (no filesystem); use [`NamModel::from_bytes`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| format!("Failed to read model file: {e}"))?;
        Self::from_bytes(&bytes)
    }

    /// Load a model from the raw bytes of a `.nam` file (UTF-8 JSON).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Ok(Self {
            inner: pure::PureNamModel::from_bytes(bytes)
                .map_err(|e| format!("Failed to load NAM model: {e}"))?,
        })
    }

    /// Load a model from `.nam` JSON text.
    pub fn from_json(json: &str) -> Result<Self, String> {
        Ok(Self {
            inner: pure::PureNamModel::from_json(json)
                .map_err(|e| format!("Failed to load NAM model: {e}"))?,
        })
    }

    /// Reset the model for a given sample rate and buffer size, then prewarm.
    pub fn reset(&mut self, sample_rate: f64, max_buffer_size: usize) {
        self.inner.reset(sample_rate, max_buffer_size);
    }

    /// Run inference on a block of mono audio.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) {
        self.inner.process(input, output);
    }

    /// Get the sample rate the model was trained at, or `None` if unknown.
    pub fn expected_sample_rate(&self) -> Option<f64> {
        self.inner.expected_sample_rate()
    }

    /// Get the model's loudness in dB, if available.
    pub fn loudness(&self) -> Option<f64> {
        self.inner.loudness()
    }

    /// Get the input level in dBu, if available.
    pub fn input_level(&self) -> Option<f64> {
        self.inner.input_level()
    }

    /// Get the output level in dBu, if available.
    pub fn output_level(&self) -> Option<f64> {
        self.inner.output_level()
    }

    /// Number of input channels (typically 1 for guitar models).
    pub fn input_channels(&self) -> usize {
        self.inner.input_channels()
    }

    /// Number of output channels (typically 1 for guitar models).
    pub fn output_channels(&self) -> usize {
        self.inner.output_channels()
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
    pub fn set_slimmable_size(&mut self, val: f64) -> bool {
        self.inner.set_slimmable_size(val)
    }
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
