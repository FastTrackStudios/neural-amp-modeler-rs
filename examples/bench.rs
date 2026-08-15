//! Deterministic per-block timing for a NAM model — measure how much of the
//! audio budget a model eats, so you can compute the lowest reliable buffer
//! instead of guessing.
//!
//! ```text
//! cargo run --release --example bench -- <model.nam> [block_frames=128] [sample_rate=48000] [iters=20000]
//! ```
//!
//! Reports per-block process time (avg/p50/p99/max) and that as a % of the
//! block budget (block/rate). Run in --release (debug is ~10-50x slower and
//! meaningless here). Add cyclictest's worst-case jitter on top to get the real
//! headroom; if p99/max load is near or over ~70-80% you'll xrun at that buffer.

use std::time::Instant;

use neural_amp_modeler::NamModel;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: bench <model.nam> [block=128] [rate=48000] [iters=20000]");
        std::process::exit(2);
    };
    let block: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let rate: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48_000.0);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    let mut model = match NamModel::load(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load {path}: {e}");
            std::process::exit(1);
        }
    };
    model.reset(rate, block);

    // A fixed, non-silent input block (guitar-ish level). Silence can be
    // cheaper for some architectures, so drive it.
    let input: Vec<f64> = (0..block).map(|i| (i as f64 * 0.04).sin() * 0.3).collect();
    let mut out = vec![0.0f64; block];

    // Warm up (allocations, caches, model prewarm settling).
    for _ in 0..200 {
        model.process(&input, &mut out);
    }

    let mut us: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        model.process(&input, &mut out);
        us.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pick = |q: f64| us[((iters as f64 * q) as usize).min(iters - 1)];
    let avg = us.iter().sum::<f64>() / iters as f64;
    let budget_us = block as f64 / rate * 1e6;
    let load = |t: f64| t / budget_us * 100.0;

    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path);

    println!("model: {name}");
    println!(
        "block {} @ {:.0} Hz  →  budget {:.0} µs/block   ({} iters)",
        block, rate, budget_us, iters
    );
    println!(
        "process µs   avg {:.0}   p50 {:.0}   p99 {:.0}   max {:.0}",
        avg,
        pick(0.50),
        pick(0.99),
        us[iters - 1]
    );
    println!(
        "DSP load     avg {:.0}%   p50 {:.0}%   p99 {:.0}%   max {:.0}%",
        load(avg),
        load(pick(0.50)),
        load(pick(0.99)),
        load(us[iters - 1])
    );
}
