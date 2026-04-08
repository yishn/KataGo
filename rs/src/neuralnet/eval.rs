/// CPU evaluator — the public interface consumed by the search engine.
///
/// Mirrors the role of `NeuralNet::getOutput` in eigenbackend.cpp, but
/// adapted to the pure-Rust, WASM-compatible layer stack in `layers.rs`.
///
/// The [`Evaluator`] wraps a [`layers::Model`] and exposes a single
/// [`Evaluator::run`] method that takes NHWC spatial features and global
/// features for a **single board position** (batch size 1) and returns an
/// [`EvalOutput`] containing raw network logits.
///
/// All data is kept on the heap as `Vec<f32>` so no GPU/WASM barriers apply.

use crate::model::ModelDesc;
use crate::neuralnet::layers::Model;

// ---------------------------------------------------------------------------
// EvalOutput
// ---------------------------------------------------------------------------

/// Raw per-position outputs from the neural network (logits, not probabilities).
///
/// Layout notes (batch size 1; all Vecs have length == the relevant channel
/// count *except* `policy_spatial` which is `H * W * policy_ch`):
///
/// | field           | length                         | layout              |
/// |-----------------|-------------------------------|---------------------|
/// | `policy_pass`   | `policy_ch`                   | `[ch]`              |
/// | `policy_spatial`| `H * W * policy_ch`           | NHWC `[hw * ch]`    |
/// | `value`         | `value_ch`                    | `[ch]`              |
/// | `score_value`   | `score_ch`                    | `[ch]`              |
/// | `ownership`     | `H * W * ownership_ch`        | NHWC `[hw * ch]`    |
///
/// (All lengths are for a single batch element, i.e. N=1.)
#[derive(Debug, Clone)]
pub struct EvalOutput {
  /// Policy logits for the pass move: `[policy_ch]`.
  pub policy_pass: Vec<f32>,
  /// Policy logits for board positions: NHWC `[hw * policy_ch]`.
  pub policy_spatial: Vec<f32>,
  /// Win/loss/noResult logits: `[value_ch]` (typically 3).
  pub value: Vec<f32>,
  /// Score distribution logits: `[score_ch]` (typically 6).
  pub score_value: Vec<f32>,
  /// Ownership map logits: `[hw * ownership_ch]` (typically 1 channel).
  pub ownership: Vec<f32>,

  // Metadata (useful for the search layer)
  pub nn_x: usize,
  pub nn_y: usize,
  pub policy_ch: usize,
  pub value_ch: usize,
  pub score_ch: usize,
  pub ownership_ch: usize,
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// A loaded, inference-ready neural network.
///
/// Create from a parsed [`ModelDesc`] via [`Evaluator::new`], then call
/// [`Evaluator::run`] for each position to evaluate.
pub struct Evaluator {
  model: Model,
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
  /// Build an evaluator from a parsed model descriptor.
  ///
  /// `nn_x` / `nn_y` are the board dimensions (e.g. 19 × 19).
  pub fn new(desc: &ModelDesc, nn_x: usize, nn_y: usize) -> Self {
    let model = Model::new(desc, nn_x, nn_y);
    Evaluator {
      nn_x,
      nn_y,
      model_version: desc.model_version,
      num_input_channels: desc.num_input_channels as usize,
      num_input_global_channels: desc.num_input_global_channels as usize,
      num_policy_channels: desc.num_policy_channels as usize,
      num_value_channels: desc.num_value_channels as usize,
      num_score_value_channels: desc.num_score_value_channels as usize,
      num_ownership_channels: desc.num_ownership_channels as usize,
      model,
    }
  }

  /// Run the network for a single board position (batch size 1).
  ///
  /// # Parameters
  /// * `spatial`  – NHWC spatial features: `[H * W * num_input_channels]`
  ///   in row-major order (position p, channel c → `spatial[p * C + c]`).
  /// * `global`   – Global features: `[num_input_global_channels]`.
  ///
  /// # Panics
  /// Panics in debug builds if the slice lengths do not match the model's
  /// expected input sizes.
  pub fn run(&self, spatial: &[f32], global: &[f32]) -> EvalOutput {
    let hw = self.nn_x * self.nn_y;
    debug_assert_eq!(
      spatial.len(),
      hw * self.num_input_channels,
      "spatial input length mismatch"
    );
    debug_assert_eq!(
      global.len(),
      self.num_input_global_channels,
      "global input length mismatch"
    );

    // Wrap global in [C * N] layout (N=1, so layout == [C])
    let (policy_pass, policy_spatial, value, score_value, ownership) =
      self.model.apply(
        spatial,
        global,
        None, // no SGF metadata
        1,    // batch size
        self.nn_x,
        self.nn_y,
      );

    EvalOutput {
      policy_pass,
      policy_spatial,
      value,
      score_value,
      ownership,
      nn_x: self.nn_x,
      nn_y: self.nn_y,
      policy_ch: self.num_policy_channels,
      value_ch: self.num_value_channels,
      score_ch: self.num_score_value_channels,
      ownership_ch: self.num_ownership_channels,
    }
  }

  /// Run the network with optional SGF metadata features.
  ///
  /// `meta` – metadata features: `[num_input_meta_channels]` (or `None`).
  pub fn run_with_meta(
    &self,
    spatial: &[f32],
    global: &[f32],
    meta: Option<&[f32]>,
  ) -> EvalOutput {
    let hw = self.nn_x * self.nn_y;
    debug_assert_eq!(spatial.len(), hw * self.num_input_channels);
    debug_assert_eq!(global.len(), self.num_input_global_channels);

    let (policy_pass, policy_spatial, value, score_value, ownership) =
      self.model.apply(spatial, global, meta, 1, self.nn_x, self.nn_y);

    EvalOutput {
      policy_pass,
      policy_spatial,
      value,
      score_value,
      ownership,
      nn_x: self.nn_x,
      nn_y: self.nn_y,
      policy_ch: self.num_policy_channels,
      value_ch: self.num_value_channels,
      score_ch: self.num_score_value_channels,
      ownership_ch: self.num_ownership_channels,
    }
  }
}
