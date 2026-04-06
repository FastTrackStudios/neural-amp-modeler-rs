//! Integration tests — load real .nam models and process audio through them.
//!
//! Includes bit-exact comparison tests against the C++ NeuralAmpModelerCore
//! reference renderer to verify our Rust bindings produce identical output.

use neural_amp_modeler::NamModel;
use std::f64::consts::PI;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("NeuralAmpModelerCore")
        .join("example_models")
}

/// Path to the C++ reference renderer (compiled by build.rs).
fn reference_render() -> PathBuf {
    PathBuf::from(env!("NAM_REFERENCE_RENDER"))
}

/// Run the C++ reference renderer and return its raw f64 output.
fn run_reference(
    model: &str,
    num_samples: usize,
    sample_rate: f64,
    buffer_size: usize,
) -> Vec<f64> {
    let output = std::process::Command::new(reference_render())
        .arg(models_dir().join(model))
        .arg(num_samples.to_string())
        .arg(sample_rate.to_string())
        .arg(buffer_size.to_string())
        .output()
        .expect("Failed to run reference renderer");

    assert!(
        output.status.success(),
        "Reference renderer failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(
        output.stdout.len(),
        num_samples * 8,
        "Expected {} bytes, got {}",
        num_samples * 8,
        output.stdout.len()
    );

    output
        .stdout
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Process through Rust bindings using the same parameters as the reference renderer.
fn run_rust(model: &str, num_samples: usize, sample_rate: f64, buffer_size: usize) -> Vec<f64> {
    let mut m = NamModel::load(models_dir().join(model)).unwrap();
    m.reset(sample_rate, buffer_size);

    let input = sine_440(sample_rate, num_samples, 0.5);
    let mut output = vec![0.0; num_samples];

    for pos in (0..num_samples).step_by(buffer_size) {
        let end = (pos + buffer_size).min(num_samples);
        m.process(&input[pos..end], &mut output[pos..end]);
    }
    output
}

fn sine_440(sample_rate: f64, num_samples: usize, amplitude: f64) -> Vec<f64> {
    (0..num_samples)
        .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin() * amplitude)
        .collect()
}

// ── Model loading ──────────────────────────────────────────────────

#[test]
fn load_lstm() {
    let path = models_dir().join("lstm.nam");
    let model = NamModel::load(&path).expect("Failed to load LSTM model");
    let meta = model.metadata();
    assert_eq!(meta.input_channels, 1);
    assert_eq!(meta.output_channels, 1);
}

#[test]
fn load_wavenet_nano() {
    let path = models_dir().join("wavenet.nam");
    let model = NamModel::load(&path).expect("Failed to load WaveNet nano model");
    assert_eq!(model.input_channels(), 1);
}

#[test]
fn load_wavenet_standard() {
    let path = models_dir().join("wavenet_a1_standard.nam");
    let model = NamModel::load(&path).expect("Failed to load WaveNet standard model");
    assert_eq!(model.input_channels(), 1);
}

#[test]
fn load_wavenet_max() {
    let path = models_dir().join("wavenet_a2_max.nam");
    let model = NamModel::load(&path).expect("Failed to load WaveNet max model");
    assert_eq!(model.input_channels(), 1);
}

#[test]
fn load_my_model() {
    let path = models_dir().join("my_model.nam");
    let model = NamModel::load(&path).expect("Failed to load my_model");
    assert_eq!(model.input_channels(), 1);
}

#[test]
fn load_slimmable_wavenet() {
    let path = models_dir().join("slimmable_wavenet.nam");
    let model = NamModel::load(&path).expect("Failed to load slimmable WaveNet");
    assert_eq!(model.input_channels(), 1);
}

#[test]
fn load_slimmable_container() {
    let path = models_dir().join("slimmable_container.nam");
    let model = NamModel::load(&path).expect("Failed to load slimmable container");
    assert_eq!(model.input_channels(), 1);
}

#[test]
fn load_wavenet_condition_dsp() {
    let path = models_dir().join("wavenet_condition_dsp.nam");
    let model = NamModel::load(&path).expect("Failed to load conditional DSP model");
    assert_eq!(model.input_channels(), 1);
}

// ── Processing ─────────────────────────────────────────────────────

#[test]
fn lstm_process_sine() {
    let mut model = NamModel::load(models_dir().join("lstm.nam")).unwrap();
    model.reset(48000.0, 512);

    let input = sine_440(48000.0, 4800, 0.5);
    let mut output = vec![0.0; 4800];

    // Process in blocks of 512 like a real audio host
    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut output[chunk_start..chunk_end],
        );
    }

    // Output should be finite
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "LSTM output NaN/Inf at sample {i}");
    }

    // Output should be non-trivial (model actually did something)
    let energy: f64 = output.iter().map(|s| s * s).sum::<f64>() / output.len() as f64;
    assert!(energy > 1e-10, "LSTM output is silent: energy={energy}");
}

#[test]
fn wavenet_nano_process_sine() {
    let mut model = NamModel::load(models_dir().join("wavenet.nam")).unwrap();
    model.reset(48000.0, 512);

    let input = sine_440(48000.0, 4800, 0.5);
    let mut output = vec![0.0; 4800];

    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut output[chunk_start..chunk_end],
        );
    }

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "WaveNet nano NaN/Inf at {i}");
    }

    let energy: f64 = output.iter().map(|s| s * s).sum::<f64>() / output.len() as f64;
    assert!(
        energy > 1e-10,
        "WaveNet nano output is silent: energy={energy}"
    );
}

#[test]
fn wavenet_standard_process_sine() {
    let mut model = NamModel::load(models_dir().join("wavenet_a1_standard.nam")).unwrap();
    model.reset(48000.0, 512);

    let input = sine_440(48000.0, 4800, 0.5);
    let mut output = vec![0.0; 4800];

    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut output[chunk_start..chunk_end],
        );
    }

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "WaveNet standard NaN/Inf at {i}");
    }

    let energy: f64 = output.iter().map(|s| s * s).sum::<f64>() / output.len() as f64;
    assert!(
        energy > 1e-10,
        "WaveNet standard output is silent: energy={energy}"
    );
}

#[test]
fn wavenet_max_process_sine() {
    let mut model = NamModel::load(models_dir().join("wavenet_a2_max.nam")).unwrap();
    model.reset(48000.0, 512);

    let input = sine_440(48000.0, 4800, 0.5);
    let mut output = vec![0.0; 4800];

    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut output[chunk_start..chunk_end],
        );
    }

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "WaveNet max NaN/Inf at {i}");
    }

    let energy: f64 = output.iter().map(|s| s * s).sum::<f64>() / output.len() as f64;
    assert!(
        energy > 1e-10,
        "WaveNet max output is silent: energy={energy}"
    );
}

#[test]
fn my_model_process_sine() {
    let mut model = NamModel::load(models_dir().join("my_model.nam")).unwrap();
    model.reset(48000.0, 512);

    let input = sine_440(48000.0, 4800, 0.5);
    let mut output = vec![0.0; 4800];

    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut output[chunk_start..chunk_end],
        );
    }

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "my_model NaN/Inf at {i}");
    }

    let energy: f64 = output.iter().map(|s| s * s).sum::<f64>() / output.len() as f64;
    assert!(energy > 1e-10, "my_model output is silent: energy={energy}");
}

// ── Behavioral tests ───────────────────────────────────────────────

#[test]
fn silence_produces_near_silence() {
    let mut model = NamModel::load(models_dir().join("wavenet.nam")).unwrap();
    model.reset(48000.0, 512);

    // Prewarm with a few blocks of silence to let model settle
    let silence = vec![0.0; 512];
    let mut discard = vec![0.0; 512];
    for _ in 0..20 {
        model.process(&silence, &mut discard);
    }

    // Now process silence and check output is low
    let mut output = vec![0.0; 512];
    model.process(&silence, &mut output);

    let peak = output.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
    // Neural models may have a small DC offset or noise floor, but it should be small
    assert!(
        peak < 0.1,
        "Silence input should produce near-silence, got peak={peak}"
    );
}

#[test]
fn different_models_produce_different_output() {
    let input = sine_440(48000.0, 4800, 0.5);

    let process_model = |path: PathBuf| -> Vec<f64> {
        let mut model = NamModel::load(&path).unwrap();
        model.reset(48000.0, 512);
        let mut output = vec![0.0; 4800];
        for chunk_start in (0..4800).step_by(512) {
            let chunk_end = (chunk_start + 512).min(4800);
            model.process(
                &input[chunk_start..chunk_end],
                &mut output[chunk_start..chunk_end],
            );
        }
        output
    };

    let out_lstm = process_model(models_dir().join("lstm.nam"));
    let out_wavenet = process_model(models_dir().join("wavenet.nam"));

    // They should produce meaningfully different output
    let diff: f64 = out_lstm
        .iter()
        .zip(out_wavenet.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 4800.0;

    assert!(
        diff > 0.001,
        "LSTM and WaveNet should produce different output: avg_diff={diff}"
    );
}

#[test]
fn reset_clears_state() {
    let mut model = NamModel::load(models_dir().join("wavenet.nam")).unwrap();
    model.reset(48000.0, 512);

    let input = sine_440(48000.0, 4800, 0.5);
    let mut out1 = vec![0.0; 4800];
    let mut out2 = vec![0.0; 4800];

    // First pass
    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut out1[chunk_start..chunk_end],
        );
    }

    // Reset and process same input
    model.reset(48000.0, 512);
    for chunk_start in (0..4800).step_by(512) {
        let chunk_end = (chunk_start + 512).min(4800);
        model.process(
            &input[chunk_start..chunk_end],
            &mut out2[chunk_start..chunk_end],
        );
    }

    // Output should be identical (deterministic)
    let diff: f64 = out1
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 4800.0;

    assert!(
        diff < 1e-10,
        "Reset should produce identical output: avg_diff={diff}"
    );
}

#[test]
fn metadata_sample_rate() {
    let model = NamModel::load(models_dir().join("wavenet_a1_standard.nam")).unwrap();
    let meta = model.metadata();
    // Standard models are trained at 48kHz
    if let Some(sr) = meta.expected_sample_rate {
        assert!(sr > 0.0, "Expected positive sample rate, got {sr}");
    }
}

#[test]
fn process_varying_buffer_sizes() {
    let mut model = NamModel::load(models_dir().join("lstm.nam")).unwrap();
    model.reset(48000.0, 2048);

    let input = sine_440(48000.0, 9600, 0.5);
    let mut output = vec![0.0; 9600];

    // Process with varying buffer sizes like a real DAW might
    let buffer_sizes = [64, 128, 256, 512, 1024, 128, 64, 512, 256, 1024, 512];
    let mut pos = 0;
    for &bs in buffer_sizes.iter().cycle() {
        if pos >= 9600 {
            break;
        }
        let end = (pos + bs).min(9600);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "Varying buffer NaN/Inf at {i}");
    }
}

#[test]
fn process_single_sample() {
    let mut model = NamModel::load(models_dir().join("wavenet.nam")).unwrap();
    model.reset(48000.0, 1);

    // Process sample-by-sample (worst case but must work)
    let mut output = [0.0f64; 1];
    for i in 0..480 {
        let input = [(2.0 * PI * 440.0 * i as f64 / 48000.0).sin() * 0.5];
        model.process(&input, &mut output);
        assert!(output[0].is_finite(), "Single-sample NaN at {i}");
    }
}

// ── Bit-exact comparison against C++ reference ─────────────────────

/// Compare Rust bindings output against the C++ NeuralAmpModelerCore
/// reference renderer sample-by-sample. They should be bit-identical
/// since our shim calls the exact same C++ code.
fn assert_bit_exact(model: &str, num_samples: usize, buffer_size: usize) {
    let sample_rate = 48000.0;

    let cpp_output = run_reference(model, num_samples, sample_rate, buffer_size);
    let rust_output = run_rust(model, num_samples, sample_rate, buffer_size);

    assert_eq!(cpp_output.len(), rust_output.len());

    let mut max_diff: f64 = 0.0;
    let mut max_diff_idx: usize = 0;

    for (i, (cpp, rust)) in cpp_output.iter().zip(rust_output.iter()).enumerate() {
        let diff = (cpp - rust).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    // Allow tiny floating-point tolerance (compiler optimization differences)
    // but in practice these should be bit-identical since it's the same code.
    assert!(
        max_diff < 1e-12,
        "{model}: Rust/C++ mismatch at sample {max_diff_idx}: \
         cpp={}, rust={}, diff={max_diff:.2e}",
        cpp_output[max_diff_idx],
        rust_output[max_diff_idx],
    );

    eprintln!(
        "  {model}: {num_samples} samples, buffer_size={buffer_size} — \
         max_diff={max_diff:.2e} (bit-exact: {})",
        max_diff == 0.0
    );
}

#[test]
fn bitexact_lstm() {
    assert_bit_exact("lstm.nam", 4800, 512);
}

#[test]
fn bitexact_wavenet_nano() {
    assert_bit_exact("wavenet.nam", 4800, 512);
}

#[test]
fn bitexact_wavenet_standard() {
    assert_bit_exact("wavenet_a1_standard.nam", 4800, 512);
}

#[test]
fn bitexact_wavenet_max() {
    assert_bit_exact("wavenet_a2_max.nam", 4800, 512);
}

#[test]
fn bitexact_my_model() {
    assert_bit_exact("my_model.nam", 4800, 512);
}

#[test]
fn bitexact_slimmable_wavenet() {
    assert_bit_exact("slimmable_wavenet.nam", 4800, 512);
}

#[test]
fn bitexact_slimmable_container() {
    assert_bit_exact("slimmable_container.nam", 4800, 512);
}

#[test]
fn bitexact_wavenet_condition_dsp() {
    assert_bit_exact("wavenet_condition_dsp.nam", 4800, 512);
}

#[test]
fn bitexact_small_buffer() {
    // Test with a tiny buffer size (64 samples) — more block boundaries
    assert_bit_exact("wavenet.nam", 4800, 64);
}

#[test]
fn bitexact_large_buffer() {
    // Test with a large buffer size (2048 samples)
    assert_bit_exact("lstm.nam", 4800, 2048);
}

#[test]
fn bitexact_long_render() {
    // 1 second at 48kHz — tests sustained processing
    assert_bit_exact("wavenet.nam", 48000, 512);
}
