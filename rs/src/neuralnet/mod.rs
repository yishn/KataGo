/// CPU-based neural network evaluator for KataGo.
///
/// A pure-Rust, WASM-compatible translation of `cpp/neuralnet/eigenbackend.cpp`.
/// Uses float32 arithmetic in NHWC memory layout (matching the Eigen backend).
///
/// Layout conventions (matching C++ Eigen backend NHWC):
///   4-D tensors: [C, X, Y, N] in column-major == [N][Y][X][C] row-major.
///   In Rust we store data flat in `[N * H * W * C]` row-major order so that
///   index `(n, y, x, c)` → `n*H*W*C + y*W*C + x*C + c`.

pub mod eval;
pub mod layers;
