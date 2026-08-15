/// Unity build — includes all NeuralAmpModelerCore sources into a single
/// translation unit. This ensures C++ static initializers (the
/// ConfigParserHelper registrations) are never stripped by the linker.
///
/// File set tracks NeuralAmpModelerCore `main` (v0.5.x + Slimmable interface /
/// SlimmableContainer, PR #242) — the architecture the A2 `.nam` models use.
/// Quoted includes inside each source resolve relative to that source's own
/// directory (GCC/Clang behaviour), so the `wavenet/` sub-directory files work
/// unmodified in the unity TU.

// Core
#include "../NeuralAmpModelerCore/NAM/dsp.cpp"
#include "../NeuralAmpModelerCore/NAM/get_dsp.cpp"
#include "../NeuralAmpModelerCore/NAM/activations.cpp"
#include "../NeuralAmpModelerCore/NAM/util.cpp"
#include "../NeuralAmpModelerCore/NAM/ring_buffer.cpp"
#include "../NeuralAmpModelerCore/NAM/linear.cpp"
#include "../NeuralAmpModelerCore/NAM/conv1d.cpp"

// Architectures (each registers via static ConfigParserHelper)
#include "../NeuralAmpModelerCore/NAM/lstm.cpp"
#include "../NeuralAmpModelerCore/NAM/convnet.cpp"
#include "../NeuralAmpModelerCore/NAM/container.cpp"          // registers "SlimmableContainer"
#include "../NeuralAmpModelerCore/NAM/wavenet/model.cpp"      // registers "WaveNet"
#include "../NeuralAmpModelerCore/NAM/wavenet/slimmable.cpp"  // slimmable WaveNet submodels
// a2_fast.cpp is an optional NEON fast-path, no-op unless NAM_ENABLE_A2_FAST
// is defined (its symbols in model.cpp are under the same guard); omitted.
