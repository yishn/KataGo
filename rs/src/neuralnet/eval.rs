/// Backend-agnostic neural network evaluator.
///
/// [`Evaluator`] wraps a `Box<dyn Backend>` and exposes the same public API
/// as before.  The concrete backend (CPU or WebGPU) is chosen at construction
/// time via [`BackendKind`].
///
/// ```ignore
/// // CPU backend (default, backward-compatible, sync)
/// let ev = Evaluator::new(&desc, 19, 19);
///
/// // WebGPU backend (async constructor)
/// let ev = Evaluator::with_backend(&desc, 19, 19, BackendKind::Wgpu).await;
/// ```

use crate::neuralnet::model::ModelDesc;
use crate::neuralnet::backend::{self, BackendKind};
use crate::neuralnet::backend_cpu::CpuBackend;

// Re-export EvalOutput from the backend module so existing import paths keep working.
pub use crate::neuralnet::backend::EvalOutput;

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// A loaded, inference-ready neural network.
///
/// Create via [`Evaluator::new`] (CPU, sync) or [`Evaluator::with_backend`]
/// (explicit backend, async).
pub struct Evaluator {
  backend: Box<dyn backend::Backend>,
  pub nn_x: usize,
  pub nn_y: usize,
  pub model_version: i32,
  pub num_input_channels: usize,
  pub num_input_global_channels: usize,
  pub num_policy_channels: usize,
  pub num_value_channels: usize,
  pub num_score_value_channels: usize,
  pub num_ownership_channels: usize,
}

impl Evaluator {
  /// Build a CPU-backed evaluator synchronously (backward-compatible).
  pub fn new(desc: &ModelDesc, nn_x: usize, nn_y: usize) -> Self {
    let b: Box<dyn backend::Backend> =
      Box::new(CpuBackend::new(desc, nn_x, nn_y));
    Self::from_backend(b, nn_x, nn_y)
  }

  /// Build an evaluator with an explicit [`BackendKind`] (async).
  ///
  /// If the requested backend is unavailable the factory falls back to CPU.
  pub async fn with_backend(
    desc: &ModelDesc,
    nn_x: usize,
    nn_y: usize,
    kind: BackendKind,
  ) -> Self {
    let b = backend::build(desc, nn_x, nn_y, kind).await;
    Self::from_backend(b, nn_x, nn_y)
  }

  /// Wrap a pre-built backend directly.
  pub fn from_backend(b: Box<dyn backend::Backend>, nn_x: usize, nn_y: usize) -> Self {
    Evaluator {
      nn_x,
      nn_y,
      model_version: b.model_version(),
      num_input_channels: b.num_input_channels(),
      num_input_global_channels: b.num_input_global_channels(),
      num_policy_channels: b.num_policy_channels(),
      num_value_channels: b.num_value_channels(),
      num_score_value_channels: b.num_score_value_channels(),
      num_ownership_channels: b.num_ownership_channels(),
      backend: b,
    }
  }

  /// Run the network for a single board position (batch size 1) — async.
  pub async fn run(&self, spatial: &[f32], global: &[f32]) -> EvalOutput {
    let hw = self.nn_x * self.nn_y;
    debug_assert_eq!(spatial.len(), hw * self.num_input_channels, "spatial length mismatch");
    debug_assert_eq!(global.len(), self.num_input_global_channels, "global length mismatch");
    self.backend.run(spatial, global, None, self.nn_x, self.nn_y).await
  }

  /// Run the network with optional SGF metadata features — async.
  pub async fn run_with_meta(
    &self,
    spatial: &[f32],
    global: &[f32],
    meta: Option<&[f32]>,
  ) -> EvalOutput {
    let hw = self.nn_x * self.nn_y;
    debug_assert_eq!(spatial.len(), hw * self.num_input_channels);
    debug_assert_eq!(global.len(), self.num_input_global_channels);
    self.backend.run(spatial, global, meta, self.nn_x, self.nn_y).await
  }
}
