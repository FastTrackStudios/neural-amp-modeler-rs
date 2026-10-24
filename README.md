# neural-amp-modeler

Rust bindings for [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore)
inference, with **two engines behind one `NamModel` API**:

- **C++ (default on native, the `cpp` feature)**: FFI over the vendored
  upstream C++ core (`NeuralAmpModelerCore/`, unmodified snapshot;
  `shim/nam_shim.cpp` is our thin C ABI). Upstream behavior, bit-for-bit,
  and the faster engine (1.5-3.8x on WaveNet models, `examples/bench.rs`).
  Costs a C++20 compile of the core (with Eigen) in `build.rs`.
- **Rust (wasm32; native with `default-features = false`)**: a pure-Rust
  port of the forward pass (`src/pure/` — WaveNet 0.5.x/0.7.x and the A2
  surface, LSTM, all NAM activations). No C++ anywhere in the build.

`NamModel::from_bytes` / `from_json` load models from memory on every
platform (browsers have no filesystem); native path-loading is
unchanged.

## How the Rust engine tracks upstream

The C++ core is the **oracle**. `tests/parity.rs` (`cargo test --features
cpp`; `--features reference-render` adds `tests/integration.rs`, which
compares against a standalone C++ renderer) runs every shipped rig
model (`features/rigs/guitar/default-config/models/`) plus the core's
example models through both engines and fails on divergence (tolerance
5e-4; actual agreement is ~1e-6, LSTM bit-exact). Model configs the Rust
engine doesn't implement (slimmable channel slicing, `condition_dsp`,
active FiLM) fail to **load** with a descriptive error — they can never
produce silently-wrong audio.

## Updating the core

```bash
just nam-update   # re-vendors upstream HEAD, then runs the test suite
```

Outcomes:
- tests green → commit the vendor bump, done;
- parity fails → upstream changed the math; port the change in
  `src/pure/` until parity is green again;
- new architecture in upstream models → shows up as a load error in
  whatever model uses it; implement it in `src/pure/` when we care.
