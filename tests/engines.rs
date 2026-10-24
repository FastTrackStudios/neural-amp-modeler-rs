//! `NamModel` on the Rust engines (no `cpp` feature): valverig's engine by
//! default, this crate's port where valverig can't run the model, and the
//! f64 <-> f32 block handling between our API and valverig's.

#![cfg(not(feature = "cpp"))]
#![cfg(not(target_arch = "wasm32"))]

use neural_amp_modeler::{NamModel, pure::PureNamModel};
use std::path::PathBuf;

const RATE: f64 = 48_000.0;

fn model(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("NeuralAmpModelerCore/example_models")
        .join(name)
}

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.3 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / RATE).sin())
        .collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn run(m: &mut NamModel, input: &[f64], block: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    for (i, o) in input.chunks(block).zip(out.chunks_mut(block)) {
        m.process(i, o);
    }
    out
}

fn run_pure(name: &str, size: Option<f64>, input: &[f64], block: usize) -> Vec<f64> {
    let mut p = PureNamModel::from_bytes(&std::fs::read(model(name)).unwrap()).unwrap();
    if let Some(v) = size {
        assert!(p.set_slimmable_size(v));
    }
    p.reset(RATE, block);
    let mut out = vec![0.0; input.len()];
    p.process(input, &mut out);
    out
}

/// The default engine agrees with this crate's port (itself held to the
/// C++ core by tests/parity.rs) on every example model the two treat alike.
#[test]
fn default_engine_matches_the_port() {
    let input = signal(8192);
    for name in [
        "wavenet.nam",
        "wavenet_a1_standard.nam",
        "lstm.nam",
        "A2.nam",
        "slimmable_container.nam",
        "slimmable_wavenet.nam",
        "wavenet_condition_dsp.nam",
        "my_model.nam",
    ] {
        let mut m = NamModel::load(model(name)).unwrap();
        m.reset(RATE, 128);
        let ours = run(&mut m, &input, 128);
        let port = run_pure(name, None, &input, 128);
        let d = max_diff(&ours, &port);
        assert!(d < 1e-4, "{name}: max diff {d:e}");
        assert!(ours.iter().any(|x| x.abs() > 1e-6), "{name}: silent");
    }
}

/// A block longer than `reset`'s size goes through valverig in pieces.
#[test]
fn a_long_block_is_processed_in_pieces() {
    let input = signal(4096);
    let mut a = NamModel::load(model("wavenet.nam")).unwrap();
    a.reset(RATE, 64);
    let mut whole = vec![0.0; input.len()];
    a.process(&input, &mut whole);
    let mut b = NamModel::load(model("wavenet.nam")).unwrap();
    b.reset(RATE, 64);
    let blocks = run(&mut b, &input, 64);
    assert_eq!(whole, blocks);
}

/// A reduced slimmable WaveNet moves onto the port (valverig slims only a
/// SlimmableContainer) and sounds as the port does at that size.
#[test]
fn a_reduced_slimmable_wavenet_runs_on_the_port() {
    let input = signal(8192);
    for size in [0.0, 0.4] {
        let mut m = NamModel::load(model("slimmable_wavenet.nam")).unwrap();
        assert!(m.set_slimmable_size(size));
        m.reset(RATE, 128);
        let ours = run(&mut m, &input, 128);
        let port = run_pure("slimmable_wavenet.nam", Some(size), &input, 128);
        assert_eq!(ours, port, "size {size}");
    }
}

/// A SlimmableContainer slims on valverig, and agrees with the port there.
#[test]
fn a_slimmable_container_slims_on_valverig() {
    let input = signal(8192);
    for size in [0.0, 0.4, 1.0] {
        let mut m = NamModel::load(model("slimmable_container.nam")).unwrap();
        assert!(m.set_slimmable_size(size));
        m.reset(RATE, 128);
        let ours = run(&mut m, &input, 128);
        let port = run_pure("slimmable_container.nam", Some(size), &input, 128);
        let d = max_diff(&ours, &port);
        assert!(d < 1e-4, "size {size}: max diff {d:e}");
    }
}

/// Back at full size, a slimmable WaveNet returns to valverig: the same
/// output as a model that was never slimmed.
#[test]
fn a_slimmable_wavenet_back_at_full_size_is_on_valverig_again() {
    let input = signal(8192);
    let mut fresh = NamModel::load(model("slimmable_wavenet.nam")).unwrap();
    fresh.reset(RATE, 128);
    let want = run(&mut fresh, &input, 128);
    let mut m = NamModel::load(model("slimmable_wavenet.nam")).unwrap();
    assert!(m.set_slimmable_size(0.4));
    assert!(m.set_slimmable_size(1.0));
    m.reset(RATE, 128);
    assert_eq!(run(&mut m, &input, 128), want);
}

#[test]
fn an_ordinary_model_is_not_slimmable() {
    let mut m = NamModel::load(model("wavenet.nam")).unwrap();
    assert!(!m.set_slimmable_size(0.5));
    assert!(!m.set_slimmable_size(1.0));
}

/// The word in a model's metadata doesn't make it slimmable.
#[test]
fn slimmable_in_the_metadata_is_not_a_slimmable_model() {
    let text = std::fs::read_to_string(model("wavenet.nam")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
    v["metadata"]["slimmable"] = serde_json::json!("not a config");
    let mut m = NamModel::from_json(&v.to_string()).unwrap();
    assert!(!m.set_slimmable_size(0.5));
    assert!(!m.set_slimmable_size(1.0));
}

#[test]
fn metadata_comes_from_the_file() {
    let m = NamModel::load(model("wavenet_a1_standard.nam")).unwrap();
    let p = PureNamModel::from_bytes(&std::fs::read(model("wavenet_a1_standard.nam")).unwrap())
        .unwrap();
    assert_eq!(m.expected_sample_rate(), p.expected_sample_rate());
    assert_eq!(m.loudness(), p.loudness());
    assert_eq!(m.input_level(), p.input_level());
    assert_eq!(m.output_level(), p.output_level());
    assert_eq!((m.input_channels(), m.output_channels()), (1, 1));
}
