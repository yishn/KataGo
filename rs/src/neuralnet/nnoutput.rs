/// NNOutput struct and output extraction from raw model output.
///
/// Mirrors the NNOutput struct from `cpp/neuralnet/nninputs.h`.
use crate::neuralnet::eval::EvalOutput;

// -----------------------------------------------------------------------
// NNOutput struct (mirrors C++ NNOutput from cpp/neuralnet/nninputs.h)
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NNOutput {
  pub white_win_prob: f32,
  pub white_loss_prob: f32,
  pub white_no_result_prob: f32,
  pub white_score_mean: f32,
  pub white_score_mean_sq: f32,
  pub white_lead: f32,
  pub shortterm_winloss_error: f32,
  pub shortterm_score_error: f32,
  /// Length = nn_len² + 1. Pass move is at index [nn_len²]. Illegal moves have value < 0.
  pub policy_probs: Vec<f32>,
  /// Optional nn_len² ownership map (white positive).
  pub owner_map: Option<Vec<f32>>,
  /// Policy after Dirichlet noise / temperature (set at root).
  pub noised_policy: Option<Vec<f32>>,
}

impl NNOutput {
  /// Returns the policy slice to use: noised if available, else raw.
  pub fn get_policy(&self) -> &[f32] {
    if let Some(np) = &self.noised_policy {
      np
    } else {
      &self.policy_probs
    }
  }
}

// -----------------------------------------------------------------------
// extract_nn_output — read from Rust EvalOutput for one batch slot
// -----------------------------------------------------------------------

/// Extract `NNOutput` from a `EvalOutput` for batch position `batch_idx`.
///
/// `policy_size` = nn_len² + 1 (pass), `value_ch` = number of value channels,
/// `score_ch` = number of score channels.
pub fn extract_nn_output(
  eval: &EvalOutput,
  batch_idx: usize,
  nn_len: u32,
  value_ch: usize,
  score_ch: usize,
  policy_ch: usize,
  ownership_ch: usize,
  include_owner: bool,
) -> NNOutput {
  let nl = nn_len as usize;
  let policy_size = nl * nl + 1;

  // ---- Policy logits → softmax probabilities ----
  // policy_spatial: [batch * policy_ch * H * W], channels=0
  // policy_pass:    [policy_ch * batch], channels=0
  // We use policy channel 0.
  let hw = nl * nl;
  let mut raw_policy = vec![0.0f32; policy_size];
  for p in 0..hw {
    // spatial: layout is [batch, policy_ch, H*W] flattened
    let idx = batch_idx * policy_ch * hw + 0 * hw + p;
    if idx < eval.policy_spatial.len() {
      raw_policy[p] = eval.policy_spatial[idx];
    }
  }
  // pass logit: layout [policy_ch, batch], channel 0
  {
    let idx = 0 * /* batches via stride */ batch_idx + batch_idx;
    // Actually layout is [policy_ch * batch], so ch=0, batch_idx:
    let idx2 = 0 * /* batch_count */ (eval.policy_pass.len() / policy_ch.max(1))
      + batch_idx;
    if idx2 < eval.policy_pass.len() {
      raw_policy[hw] = eval.policy_pass[idx2];
    }
    let _ = idx;
  }

  // Softmax
  let max_logit = raw_policy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let mut exp_sum = 0.0f32;
  let mut policy_probs: Vec<f32> = raw_policy
    .iter()
    .map(|&x| {
      let e = (x - max_logit).exp();
      exp_sum += e;
      e
    })
    .collect();
  if exp_sum > 0.0 {
    for p in policy_probs.iter_mut() {
      *p /= exp_sum;
    }
  }

  // ---- Value head ----
  // value: [value_ch * batch], i.e. channel-major
  let get_val = |ch: usize| -> f32 {
    // Layout: [value_ch, batch] → idx = ch * batch + batch_idx
    let batch_count = eval.value.len() / value_ch.max(1);
    let i = ch * batch_count + batch_idx;
    if i < eval.value.len() {
      eval.value[i]
    } else {
      0.0
    }
  };
  // Softmax over win/loss/noResult (first 3 value channels)
  let wl_logits = [
    get_val(0),
    get_val(1),
    if value_ch >= 3 { get_val(2) } else { 0.0 },
  ];
  let wl_max = wl_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let wl_exp: [f32; 3] = [
    (wl_logits[0] - wl_max).exp(),
    (wl_logits[1] - wl_max).exp(),
    (wl_logits[2] - wl_max).exp(),
  ];
  let wl_sum = wl_exp[0] + wl_exp[1] + wl_exp[2];
  let (white_win_prob, white_loss_prob, white_no_result_prob) = if wl_sum > 0.0
  {
    (wl_exp[0] / wl_sum, wl_exp[1] / wl_sum, wl_exp[2] / wl_sum)
  } else {
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
  };

  // Shortterm errors (value channels 4 and 5 if they exist)
  let shortterm_winloss_error =
    if value_ch >= 5 { get_val(3).abs() } else { 0.0 };
  let shortterm_score_error =
    if value_ch >= 6 { get_val(4).abs() } else { 0.0 };

  // ---- Score value head ----
  let get_sv = |ch: usize| -> f32 {
    let batch_count = eval.score_value.len() / score_ch.max(1);
    let i = ch * batch_count + batch_idx;
    if i < eval.score_value.len() {
      eval.score_value[i]
    } else {
      0.0
    }
  };
  let white_score_mean = if score_ch >= 1 { get_sv(0) } else { 0.0 };
  let white_score_mean_sq = if score_ch >= 2 {
    get_sv(1)
  } else {
    white_score_mean * white_score_mean
  };
  let white_lead = if score_ch >= 3 {
    get_sv(2)
  } else {
    white_score_mean
  };

  // ---- Ownership ----
  let owner_map =
    if include_owner && ownership_ch > 0 && !eval.ownership.is_empty() {
      let offset = batch_idx * ownership_ch * hw;
      let len = ownership_ch * hw;
      if offset + len <= eval.ownership.len() {
        Some(eval.ownership[offset..offset + hw].to_vec())
      } else {
        None
      }
    } else {
      None
    };

  NNOutput {
    white_win_prob,
    white_loss_prob,
    white_no_result_prob,
    white_score_mean,
    white_score_mean_sq,
    white_lead,
    shortterm_winloss_error,
    shortterm_score_error,
    policy_probs,
    owner_map,
    noised_policy: None,
  }
}
