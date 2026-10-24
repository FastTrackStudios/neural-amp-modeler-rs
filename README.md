# neural-amp-modeler

Rust bindings for [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore)
inference, with **three engines behind one `NamModel` API**:

- **valverig (default, every target)**: [valverig-nam](https://github.com/ValveRig/valverig-nam)
  (MIT, pure Rust, allocation-free in the audio callback). About twice as
  fast as the C++ core on WaveNet captures (`examples/bench.rs`) and within
  ~1e-6 of it. Pinned exactly (`=0.0.1`): it is young.
- **Rust port (`src/pure/`)**: this crate's own port of the forward pass,
  used where valverig can't run the model — a reduced size of a slimmable
  WaveNet (valverig slims only a `SlimmableContainer`).
- **C++ (`cpp` feature, native only)**: FFI over the vendored upstream C++
  core (`NeuralAmpModelerCore/`, unmodified snapshot; `shim/nam_shim.cpp`
  is our thin C ABI), and the **oracle** the parity tests hold both Rust
  engines to. Costs a C++20 compile of the core (with Eigen) in `build.rs`;
  without the feature, `build.rs` compiles nothing.

`NamModel::from_bytes` / `from_json` load models from memory on every
platform (browsers have no filesystem); native path-loading is
unchanged.

## How the Rust engine tracks upstream

The C++ core is the **oracle**. `tests/parity.rs` (`cargo test --features
cpp`, both Rust engines against it; `--features reference-render` adds
`tests/integration.rs`, which compares against a standalone C++ renderer)
runs every shipped rig
model (`features/rigs/guitar/default-config/models/`) plus the core's
example models through both engines and fails on divergence (tolerance
5e-4; actual agreement is ~1e-6, LSTM bit-exact). One known gap: the
vendored core predates upstream's "Fix layer1x1 post FiLM for non-blended
gating" (#322), which valverig implements and `src/pure` does not; on
`wavenet_a2_max` valverig therefore differs from this oracle, and the test
says so instead of comparing. Bump the core (and port the fix to
`src/pure`) to close it. `tests/engines.rs` (default features) checks
`NamModel`'s engine choice and fallback. Model configs the Rust
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
