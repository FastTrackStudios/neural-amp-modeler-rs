/// Unity build — includes all NeuralAmpModelerCore sources into a single
/// translation unit. This ensures C++ static initializers (the
/// ConfigParserHelper registrations) are never stripped by the linker.

// Core
#include "../NeuralAmpModelerCore/NAM/dsp.cpp"
#include "../NeuralAmpModelerCore/NAM/get_dsp.cpp"
#include "../NeuralAmpModelerCore/NAM/activations.cpp"
#include "../NeuralAmpModelerCore/NAM/util.cpp"
#include "../NeuralAmpModelerCore/NAM/ring_buffer.cpp"

// Architectures (each registers via static ConfigParserHelper)
#include "../NeuralAmpModelerCore/NAM/lstm.cpp"
#include "../NeuralAmpModelerCore/NAM/wavenet.cpp"
#include "../NeuralAmpModelerCore/NAM/convnet.cpp"
#include "../NeuralAmpModelerCore/NAM/conv1d.cpp"
#include "../NeuralAmpModelerCore/NAM/container.cpp"
#include "../NeuralAmpModelerCore/NAM/slimmable_wavenet.cpp"
