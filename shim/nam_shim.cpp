/// C shim implementation — wraps NeuralAmpModelerCore's C++ API for FFI.

#include "nam_shim.h"
#include "../NeuralAmpModelerCore/NAM/get_dsp.h"
#include "../NeuralAmpModelerCore/NAM/dsp.h"

#include <cstring>
#include <memory>
#include <string>

struct NamModel {
    std::unique_ptr<nam::DSP> dsp;
};

static char* make_error(const std::string& msg) {
    char* err = static_cast<char*>(std::malloc(msg.size() + 1));
    if (err) {
        std::memcpy(err, msg.c_str(), msg.size() + 1);
    }
    return err;
}

extern "C" {

NamLoadResult nam_load(const char* path) {
    NamLoadResult result = {nullptr, nullptr};
    try {
        auto dsp = nam::get_dsp(std::filesystem::path(path));
        if (!dsp) {
            result.error = make_error("Failed to load NAM model (unknown error)");
            return result;
        }
        auto* model = new NamModel{std::move(dsp)};
        result.model = model;
    } catch (const std::exception& e) {
        result.error = make_error(std::string("Failed to load NAM model: ") + e.what());
    } catch (...) {
        result.error = make_error("Failed to load NAM model: unknown exception");
    }
    return result;
}

void nam_free(NamModel* model) {
    delete model;
}

void nam_free_error_string(char* error) {
    std::free(error);
}

void nam_process(NamModel* model, const double* input, double* output, int num_frames) {
    if (!model || !model->dsp) return;

    // NAM expects double** (channel pointers). Models are mono (1 in, 1 out).
    const double* in_ptrs[1] = {input};
    double* out_ptrs[1] = {output};

    model->dsp->process(
        const_cast<double**>(in_ptrs),
        out_ptrs,
        num_frames
    );
}

void nam_reset(NamModel* model, double sample_rate, int max_buffer_size) {
    if (!model || !model->dsp) return;
    model->dsp->ResetAndPrewarm(sample_rate, max_buffer_size);
}

double nam_get_expected_sample_rate(const NamModel* model) {
    if (!model || !model->dsp) return -1.0;
    return model->dsp->GetExpectedSampleRate();
}

int nam_has_loudness(const NamModel* model) {
    if (!model || !model->dsp) return 0;
    return model->dsp->HasLoudness() ? 1 : 0;
}

double nam_get_loudness(const NamModel* model) {
    if (!model || !model->dsp) return 0.0;
    try {
        return model->dsp->GetLoudness();
    } catch (...) {
        return 0.0;
    }
}

int nam_num_input_channels(const NamModel* model) {
    if (!model || !model->dsp) return 0;
    return model->dsp->NumInputChannels();
}

int nam_num_output_channels(const NamModel* model) {
    if (!model || !model->dsp) return 0;
    return model->dsp->NumOutputChannels();
}

int nam_has_input_level(const NamModel* model) {
    if (!model || !model->dsp) return 0;
    // HasInputLevel is non-const in the C++ API
    return const_cast<nam::DSP*>(model->dsp.get())->HasInputLevel() ? 1 : 0;
}

double nam_get_input_level(const NamModel* model) {
    if (!model || !model->dsp) return 0.0;
    return const_cast<nam::DSP*>(model->dsp.get())->GetInputLevel();
}

int nam_has_output_level(const NamModel* model) {
    if (!model || !model->dsp) return 0;
    return const_cast<nam::DSP*>(model->dsp.get())->HasOutputLevel() ? 1 : 0;
}

double nam_get_output_level(const NamModel* model) {
    if (!model || !model->dsp) return 0.0;
    return const_cast<nam::DSP*>(model->dsp.get())->GetOutputLevel();
}

} // extern "C"
