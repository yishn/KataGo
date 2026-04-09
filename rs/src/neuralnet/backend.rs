/// Backend abstraction for neural network inference.
///
/// [`Backend`] is the single trait every inference engine must implement.
/// The [`super::eval::Evaluator`] holds a `Box<dyn Backend>` and delegates
/// every inference call through it, so the caller is completely agnostic of
/// whether the computation runs on the CPU or on a GPU via WebGPU.
///
/// # Implementing a new backend
///
/// 1. Create a struct that holds all compiled/uploaded weight data.
/// 2. Implement [`Backend`].
/// 3. Build it from a [`crate::model::ModelDesc`] using whichever initialiser
///    your struct exposes.
/// 4. Pass it to [`super::eval::Evaluator::from_backend`].

use crate::model::ModelDesc;

// ---------------------------------------------------------------------------
// EvalOutput
// ---------------------------------------------------------------------------

/// Raw per-position outputs from a neural network forward pass.
///
/// All tensors use batch size 1.  Layout conventions match the NHWC row-major
/// scheme used throughout the Rust codebase.
///
/// | field            | length                        | layout               |
/// |------------------|-------------------------------|----------------------|
/// | `policy_pass`    | `policy_ch`                   | `[ch]`               |
/// | `policy_spatial` | `H * W * policy_ch`           | NHWC `[hw * ch]`     |
/// | `value`          | `value_ch`                    | `[ch]`               |
/// | `score_value`    | `score_ch`                    | `[ch]`               |
/// | `ownership`      | `H * W * ownership_ch`        | NHWC `[hw * ch]`     |
#[derive(Debug, Clone)]
pub struct EvalOutput {
  pub policy_pass: Vec<f32>,
  pub policy_spatial: Vec<f32>,
  pub value: Vec<f32>,
  pub score_value: Vec<f32>,
  pub ownership: Vec<f32>,

  pub nn_x: usize,
  pub nn_y: usize,
  pub policy_ch: usize,
  pub value_ch: usize,
  pub score_ch: usize,
  pub ownership_ch: usize,
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Synchronous inference backend.
///
/// All methods receive the board dimensions because the same compiled model
/// can service different board sizes (the CPU backend pre-computes Winograd
/// tiles per size; GPU backends may need to rebuild pipelines lazily).
pub trait Backend: Send + Sync {
  /// Run a single-position (batch = 1) forward pass.
  ///
  /// * `spatial`  – NHWC spatial features `[H * W * C_spatial]`
  /// * `global`   – global features `[C_global]`
  /// * `meta`     – optional SGF metadata features `[C_meta]`
  /// * `nn_x`, `nn_y` – board width / height the evaluator was built for
  fn run(
    &self,
    spatial: &[f32],
    global: &[f32],
    meta: Option<&[f32]>,
    nn_x: usize,
    nn_y: usize,
  ) -> EvalOutput;

  // Metadata that callers may query without running inference.
  fn model_version(&self) -> i32;
  fn num_input_channels(&self) -> usize;
  fn num_input_global_channels(&self) -> usize;
  fn num_input_meta_channels(&self) -> usize;
  fn num_policy_channels(&self) -> usize;
  fn num_value_channels(&self) -> usize;
  fn num_score_value_channels(&self) -> usize;
  fn num_ownership_channels(&self) -> usize;
}

// ---------------------------------------------------------------------------
// BackendKind — named variants consumers can request
// ---------------------------------------------------------------------------

/// Which backend to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendKind {
  /// Pure-Rust CPU backend (always available).
  #[default]
  Cpu,
  /// WebGPU backend via the `wgpu` crate.
  Wgpu,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build a [`Box<dyn Backend>`] from a parsed model descriptor.
///
/// Falls back silently to [`BackendKind::Cpu`] when the requested backend is
/// unavailable (e.g. no GPU adapter found on the current machine).
pub fn build(desc: &ModelDesc, nn_x: usize, nn_y: usize, kind: BackendKind)
  -> Box<dyn Backend>
{
  match kind {
    BackendKind::Cpu => {
      Box::new(crate::neuralnet::backend_cpu::CpuBackend::new(desc, nn_x, nn_y))
    }
    BackendKind::Wgpu => {
      #[cfg(not(target_arch = "wasm32"))]
      {
        match crate::neuralnet::backend_wgpu::WgpuBackend::new(desc, nn_x, nn_y) {
          Ok(b) => return Box::new(b),
          Err(e) => {
            eprintln!(
              "[katago-rs] WebGPU backend unavailable ({e}), falling back to CPU"
            );
          }
        }
      }
      #[cfg(target_arch = "wasm32")]
      {
        eprintln!(
          "[katago-rs] WebGPU backend not yet wired for wasm32, using CPU"
        );
      }
      Box::new(crate::neuralnet::backend_cpu::CpuBackend::new(desc, nn_x, nn_y))
    }
  }
}
