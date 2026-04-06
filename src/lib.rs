//! Safe Rust bindings to [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).
//!
//! Loads `.nam` model files and runs inference for guitar amp/pedal modeling.
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

use std::ffi::CString;
use std::path::Path;

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
    }
}

/// A loaded Neural Amp Modeler model ready for inference.
///
/// Thread safety: `NamModel` is `Send` but not `Sync`. You can move it
/// between threads, but must not share it across threads without external
/// synchronisation (the underlying C++ object is not thread-safe).
pub struct NamModel {
    ptr: *mut ffi::NamModel,
}

// NAM models can be moved between threads safely.
// The underlying C++ DSP object is single-threaded but not tied to a thread.
unsafe impl Send for NamModel {}

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

impl NamModel {
    /// Load a `.nam` model file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| "Path contains invalid UTF-8".to_string())?;
        let c_path = CString::new(path_str).map_err(|_| "Path contains null byte".to_string())?;

        let result = unsafe { ffi::nam_load(c_path.as_ptr()) };

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
        if sr < 0.0 {
            None
        } else {
            Some(sr)
        }
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
}

impl Drop for NamModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::nam_free(self.ptr) };
        }
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
}
