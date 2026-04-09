/// Neural network evaluator with pluggable backend support.
///
/// Backends available:
/// * [`backend_cpu`] — pure-Rust CPU backend (always available, WASM-compatible)
/// * [`backend_wgpu`] — WebGPU backend via the `wgpu` crate (native + WASM)
///
/// Layout conventions (matching C++ Eigen backend NHWC):
///   4-D tensors: [C, X, Y, N] in column-major == [N][Y][X][C] row-major.
///   In Rust we store data flat in `[N * H * W * C]` row-major order so that
///   index `(n, y, x, c)` → `n*H*W*C + y*W*C + x*C + c`.
pub mod backend;
pub mod backend_cpu;
pub mod backend_wgpu;
pub mod eval;
pub mod layers;
pub mod nninputs;
pub mod nnoutput;
