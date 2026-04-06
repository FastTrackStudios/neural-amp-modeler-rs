/// C shim for NeuralAmpModelerCore — exposes the C++ DSP API through
/// extern "C" functions suitable for Rust FFI.

#ifndef NAM_SHIM_H
#define NAM_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a loaded NAM model (wraps std::unique_ptr<nam::DSP>).
typedef struct NamModel NamModel;

/// Result of a model load operation.
typedef struct {
    /// Non-null on success. Caller owns (must call nam_free).
    NamModel* model;
    /// Non-null on failure. Caller must call nam_free_error_string.
    char* error;
} NamLoadResult;

/// Load a .nam model file from disk.
NamLoadResult nam_load(const char* path);

/// Free a loaded model.
void nam_free(NamModel* model);

/// Free an error string returned by nam_load.
void nam_free_error_string(char* error);

/// Process a block of mono audio (in-place is fine: input == output).
/// input and output are flat arrays of `num_frames` doubles.
void nam_process(NamModel* model, const double* input, double* output, int num_frames);

/// Reset the model state for a new sample rate and max buffer size.
/// Also prewarms the model.
void nam_reset(NamModel* model, double sample_rate, int max_buffer_size);

/// Get the sample rate the model was trained at (-1 if unknown).
double nam_get_expected_sample_rate(const NamModel* model);

/// Check if the model has loudness metadata.
int nam_has_loudness(const NamModel* model);

/// Get the model's loudness in dB (only valid if nam_has_loudness returns 1).
double nam_get_loudness(const NamModel* model);

/// Get the number of input channels.
int nam_num_input_channels(const NamModel* model);

/// Get the number of output channels.
int nam_num_output_channels(const NamModel* model);

/// Check if the model has input level metadata.
int nam_has_input_level(const NamModel* model);

/// Get the input level in dBu (only valid if nam_has_input_level returns 1).
double nam_get_input_level(const NamModel* model);

/// Check if the model has output level metadata.
int nam_has_output_level(const NamModel* model);

/// Get the output level in dBu (only valid if nam_has_output_level returns 1).
double nam_get_output_level(const NamModel* model);

#ifdef __cplusplus
}
#endif

#endif // NAM_SHIM_H
