/// Reference renderer — processes a 440 Hz sine through a NAM model using
/// the C++ API directly, writing raw f64 samples to stdout.
///
/// Usage: reference_render <model.nam> <num_samples> <sample_rate> <buffer_size>
///
/// Output: raw little-endian f64 samples on stdout (num_samples * 8 bytes).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <model.nam> <num_samples> <sample_rate> <buffer_size>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const int num_samples = atoi(argv[2]);
    const double sample_rate = atof(argv[3]);
    const int buffer_size = atoi(argv[4]);

    auto model = nam::get_dsp(std::filesystem::path(model_path));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    model->ResetAndPrewarm(sample_rate, buffer_size);

    // Generate 440 Hz sine input
    std::vector<double> input(num_samples);
    for (int i = 0; i < num_samples; i++) {
        input[i] = std::sin(2.0 * M_PI * 440.0 * i / sample_rate) * 0.5;
    }

    std::vector<double> output(num_samples);

    // Process in blocks (matching our Rust test pattern)
    for (int pos = 0; pos < num_samples; pos += buffer_size) {
        int block = std::min(buffer_size, num_samples - pos);
        double* in_ptr = &input[pos];
        double* out_ptr = &output[pos];
        double* in_ptrs[1] = {in_ptr};
        double* out_ptrs[1] = {out_ptr};
        model->process(in_ptrs, out_ptrs, block);
    }

    // Write raw f64 to stdout
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    fwrite(output.data(), sizeof(double), num_samples, stdout);
    return 0;
}
