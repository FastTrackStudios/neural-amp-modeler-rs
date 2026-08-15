//! Parity tests: run the same .nam model + input through the C++ core (via
//! the FFI `NamModel`) and the pure-Rust engine (`pure::PureNamModel`), and
//! require the outputs to agree within a small tolerance.
//!
//! Both engines compute in f32 but with different accumulation orders
//! (Eigen GEMM vs naive loops), so exact bit equality is not expected.

#![cfg(not(target_arch = "wasm32"))]

use neural_amp_modeler::{pure::PureNamModel, NamModel};
use std::path::PathBuf;

const SAMPLE_RATE: f64 = 48000.0;
const BUFFER_SIZE: usize = 512;
const NUM_SAMPLES: usize = 8192;

/// Deterministic test signal: a few sine partials with an amplitude ramp and
/// a touch of deterministic "noise" so the nonlinearity is well exercised.
fn test_signal(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x243F6A8885A308D3;
    (0..n)
        .map(|i| {
            let t = i as f64 / SAMPLE_RATE;
            // xorshift for deterministic noise
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            let ramp = (i as f64 / n as f64).min(1.0);
            ramp * (0.4 * (2.0 * std::f64::consts::PI * 110.0 * t).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.05 * noise)
        })
        .collect()
}

fn process_native(path: &PathBuf, input: &[f64]) -> Vec<f64> {
    let mut m = NamModel::load(path).unwrap_or_else(|e| panic!("native load {path:?}: {e}"));
    m.reset(SAMPLE_RATE, BUFFER_SIZE);
    let mut out = vec![0.0f64; input.len()];
    for (ic, oc) in input.chunks(BUFFER_SIZE).zip(out.chunks_mut(BUFFER_SIZE)) {
        m.process(ic, oc);
    }
    out
}

fn process_pure(path: &PathBuf, input: &[f64]) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    let mut m =
        PureNamModel::from_bytes(&bytes).unwrap_or_else(|e| panic!("pure load {path:?}: {e}"));
    m.reset(SAMPLE_RATE, BUFFER_SIZE);
    let mut out = vec![0.0f64; input.len()];
    m.process(input, &mut out);
    out
}

fn compare(name: &str, a: &[f64], b: &[f64]) {
    let rms = (a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt();
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    let rel = if rms > 0.0 { max_abs / rms } else { max_abs };
    println!("{name}: output rms {rms:.6}, max abs diff {max_abs:.3e}, rel {rel:.3e}");
    // f32 accumulation error scales with signal level, so scale the absolute
    // tolerance for hot models (the generated A2 test models have rms >> 1).
    let abs_tol = 5e-4 * rms.max(1.0);
    assert!(
        max_abs < abs_tol && rel < 2e-3,
        "{name}: parity failure — max abs diff {max_abs:.3e} (rel {rel:.3e}) exceeds tolerance"
    );
    // Sanity: the model actually did something.
    assert!(rms > 1e-6, "{name}: output is silent");
}

/// Parity-check every .nam in `dir`; returns the number of models the pure
/// engine refused to load (parity of loadable models is asserted).
fn parity_for_dir(dir: PathBuf) -> usize {
    let mut checked = 0;
    let mut skipped = 0;
    let input = test_signal(NUM_SAMPLES);
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {dir:?}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "nam"))
        .collect();
    entries.sort();
    for path in entries {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // Skip models the pure engine explicitly does not support.
        let bytes = std::fs::read(&path).unwrap();
        let pure = match PureNamModel::from_bytes(&bytes) {
            Ok(_) => process_pure(&path, &input),
            Err(e) => {
                println!("{name}: skipped (unsupported by pure engine: {e})");
                skipped += 1;
                continue;
            }
        };
        let native = process_native(&path, &input);
        compare(&name, &native, &pure);
        checked += 1;
    }
    assert!(checked > 0, "no models were parity-checked in {dir:?}");
    skipped
}

/// The 11 real rig models (WaveNet 0.5.4 + SlimmableContainer/WaveNet 0.7.0).
#[test]
fn parity_rig_models() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../features/rigs/guitar/default-config/models");
    if !dir.exists() {
        eprintln!("rig models dir not found; skipping");
        return;
    }
    parity_for_dir(dir);
}

/// The NeuralAmpModelerCore example models — including the full A2 surface
/// (wavenet_a2_max, condition_dsp, slimmable). The pure engine must load
/// every one of them (no skips).
#[test]
fn parity_example_models() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("NeuralAmpModelerCore")
        .join("example_models");
    let skipped = parity_for_dir(dir);
    assert_eq!(
        skipped, 0,
        "the pure engine must support every example model (A2 included)"
    );
}

/// Slimmable size selection: both engines must agree at reduced sizes as
/// well as full size, for the slimmable WaveNet (channel slicing) and the
/// SlimmableContainer models (submodel selection).
#[test]
fn parity_slimmable_sizes() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("NeuralAmpModelerCore")
        .join("example_models");
    let input = test_signal(NUM_SAMPLES);

    // slimmable_wavenet.nam: allowed_channels [1, 2, 3];
    // 0.0 -> 1ch, 0.4 -> 2ch, 1.0 -> 3ch (full).
    // Container models (A2.nam, slimmable_container.nam): submodel selection
    // by max_value.
    for name in ["slimmable_wavenet.nam", "slimmable_container.nam", "A2.nam"] {
        let path = dir.join(name);
        for val in [0.0, 0.4, 1.0] {
            let mut native = NamModel::load(&path).unwrap();
            native.reset(SAMPLE_RATE, BUFFER_SIZE);
            assert!(
                native.set_slimmable_size(val),
                "{name}: C++ core did not accept SetSlimmableSize"
            );
            let mut native_out = vec![0.0f64; input.len()];
            for (ic, oc) in input
                .chunks(BUFFER_SIZE)
                .zip(native_out.chunks_mut(BUFFER_SIZE))
            {
                native.process(ic, oc);
            }

            let bytes = std::fs::read(&path).unwrap();
            let mut pure = PureNamModel::from_bytes(&bytes).unwrap();
            pure.reset(SAMPLE_RATE, BUFFER_SIZE);
            assert!(
                pure.set_slimmable_size(val),
                "{name}: pure engine did not accept set_slimmable_size"
            );
            let mut pure_out = vec![0.0f64; input.len()];
            pure.process(&input, &mut pure_out);

            compare(&format!("{name} @ size {val}"), &native_out, &pure_out);
        }
    }

    // Ordinary models must report non-slimmable in both engines.
    let plain = dir.join("wavenet.nam");
    let mut native = NamModel::load(&plain).unwrap();
    assert!(!native.set_slimmable_size(0.5));
    let mut pure = PureNamModel::from_bytes(&std::fs::read(&plain).unwrap()).unwrap();
    assert!(!pure.set_slimmable_size(0.5));
}

/// Freshly generated A2 models (via the vendored generate_weights_a2.py)
/// exercising the FULL A2 surface — FiLM everywhere, head1x1, condition_dsp,
/// GATED/BLENDED/NONE gating, complex activations — with weights the
/// checked-in example doesn't have. Skips gracefully if python3 is missing.
#[test]
fn parity_generated_a2_models() {
    let core = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("NeuralAmpModelerCore");
    let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let template = core.join("example_models/wavenet_a2_max.nam");
    let input = test_signal(NUM_SAMPLES);

    for seed in [7u64, 1234, 987654321] {
        let out = out_dir.join(format!("wavenet_a2_gen_{seed}.nam"));
        let status = std::process::Command::new("python3")
            .arg(core.join("generate_weights_a2.py"))
            .arg("--input")
            .arg(&template)
            .arg("--output")
            .arg(&out)
            .arg("--seed")
            .arg(seed.to_string())
            .stdout(std::process::Stdio::null())
            .status();
        let Ok(status) = status else {
            eprintln!("python3 not available; skipping generated-A2 parity");
            return;
        };
        assert!(status.success(), "generate_weights_a2.py failed (seed {seed})");

        let native = process_native(&out, &input);
        let pure = process_pure(&out, &input);
        compare(&format!("wavenet_a2_gen seed {seed}"), &native, &pure);
    }
}

/// Native from_bytes must match native load-from-path exactly.
#[test]
fn native_from_bytes_matches_load() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("NeuralAmpModelerCore/example_models/wavenet.nam");
    let input = test_signal(4096);

    let from_path = process_native(&path, &input);

    let bytes = std::fs::read(&path).unwrap();
    let mut m = NamModel::from_bytes(&bytes).unwrap();
    m.reset(SAMPLE_RATE, BUFFER_SIZE);
    let mut from_bytes = vec![0.0f64; input.len()];
    for (ic, oc) in input
        .chunks(BUFFER_SIZE)
        .zip(from_bytes.chunks_mut(BUFFER_SIZE))
    {
        m.process(ic, oc);
    }

    assert_eq!(from_path, from_bytes, "from_bytes must be bit-identical to load()");
}
