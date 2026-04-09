/// CPU backend — wraps the existing [`layers::Model`] scalar implementation.
///
/// This is the reference backend and is always available.  All computation
/// happens on the CPU using the same pure-Rust NHWC layer stack that was
/// previously embedded directly in [`super::eval::Evaluator`].

use crate::neuralnet::model::ModelDesc;
use crate::neuralnet::backend::{Backend, EvalOutput, RunFuture};
use crate::neuralnet::layers::Model;

// ---------------------------------------------------------------------------
// CpuBackend
// ---------------------------------------------------------------------------

pub struct CpuBackend {
  model: Model,
}

impl CpuBackend {
  pub fn new(desc: &ModelDesc, nn_x: usize, nn_y: usize) -> Self {
    CpuBackend { model: Model::new(desc, nn_x, nn_y) }
  }
}

impl Backend for CpuBackend {
  fn run<'a>(
    &'a self,
    spatial: &'a [f32],
    global: &'a [f32],
    meta: Option<&'a [f32]>,
    nn_x: usize,
    nn_y: usize,
  ) -> RunFuture<'a> {
    let (policy_pass, policy_spatial, value, score_value, ownership) =
      self.model.apply(spatial, global, meta, 1, nn_x, nn_y);

    let result = EvalOutput {
      policy_pass,
      policy_spatial,
      value,
      score_value,
      ownership,
      nn_x,
      nn_y,
      policy_ch: self.model.num_policy_channels,
      value_ch: self.model.num_value_channels,
      score_ch: self.model.num_score_value_channels,
      ownership_ch: self.model.num_ownership_channels,
    };
    Box::pin(std::future::ready(result))
  }

  fn model_version(&self) -> i32 { self.model.model_version }
  fn num_input_channels(&self) -> usize { self.model.num_input_channels }
  fn num_input_global_channels(&self) -> usize { self.model.num_input_global_channels }
  fn num_input_meta_channels(&self) -> usize { self.model.num_input_meta_channels }
  fn num_policy_channels(&self) -> usize { self.model.num_policy_channels }
  fn num_value_channels(&self) -> usize { self.model.num_value_channels }
  fn num_score_value_channels(&self) -> usize { self.model.num_score_value_channels }
  fn num_ownership_channels(&self) -> usize { self.model.num_ownership_channels }
}
