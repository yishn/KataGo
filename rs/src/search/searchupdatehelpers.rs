/// Node statistics update after playouts (backpropagation).
/// Mirrors `cpp/search/searchupdatehelpers.cpp`.
use crate::search::distributiontable::DistributionTable;
use crate::neuralnet::nnoutput::NNOutput;
use crate::search::node::{NodeStats, SearchNode};
use crate::search::params::SearchParams;
use crate::search::scorevalue;

// -----------------------------------------------------------------------
// compute_weight
// -----------------------------------------------------------------------

/// Compute playout weight from NN output.
/// Returns 1.0 unless `use_uncertainty` is enabled.
pub fn compute_weight(nn: &NNOutput, params: &SearchParams) -> f64 {
  if !params.use_uncertainty {
    return 1.0;
  }
  let win_loss_uncert =
    params.win_loss_utility_factor * nn.shortterm_winloss_error as f64;
  let score_uncert = nn.shortterm_score_error as f64; // approximation
  let util_uncert = win_loss_uncert + score_uncert;
  let powered = if params.uncertainty_exponent == 1.0 {
    util_uncert
  } else {
    util_uncert.powf(params.uncertainty_exponent)
  };
  let baseline = params.uncertainty_coeff / params.uncertainty_max_weight;
  (params.uncertainty_coeff / (powered + baseline))
    .max(1.0 / params.uncertainty_max_weight)
}

// -----------------------------------------------------------------------
// add_leaf_value
// -----------------------------------------------------------------------

/// Update `node.stats` with a direct leaf NN evaluation (running weighted average).
///
/// `assume_fresh = true` on the first visit (no existing weight to average with).
pub fn add_leaf_value(
  node: &mut SearchNode,
  params: &SearchParams,
  recent_score_center: f64,
  sqrt_board_area: f64,
  assume_fresh: bool,
) {
  let nn = match &node.nn_output {
    Some(n) => n,
    None => return,
  };
  let win_loss = (nn.white_win_prob - nn.white_loss_prob) as f64;
  let no_result = nn.white_no_result_prob as f64;
  let score_mean = nn.white_score_mean as f64;
  let score_mean_sq = nn.white_score_mean_sq as f64;
  let lead = nn.white_lead as f64;
  let utility = scorevalue::get_utility(
    win_loss,
    no_result,
    score_mean,
    score_mean_sq,
    recent_score_center,
    sqrt_board_area,
    params,
  );
  let weight = compute_weight(nn, params);
  let utility_sq = utility * utility;
  let weight_sq = weight * weight;

  let s = &mut node.stats;
  if assume_fresh {
    s.win_loss_value_avg = win_loss;
    s.no_result_value_avg = no_result;
    s.score_mean_avg = score_mean;
    s.score_mean_sq_avg = score_mean_sq;
    s.lead_avg = lead;
    s.utility_avg = utility;
    s.utility_sq_avg = utility_sq;
    s.weight_sq_sum = weight_sq;
    s.weight_sum = weight;
    s.visits = 1;
  } else {
    let old_w = s.weight_sum;
    let new_w = old_w + weight;
    s.win_loss_value_avg =
      (s.win_loss_value_avg * old_w + win_loss * weight) / new_w;
    s.no_result_value_avg =
      (s.no_result_value_avg * old_w + no_result * weight) / new_w;
    s.score_mean_avg = (s.score_mean_avg * old_w + score_mean * weight) / new_w;
    s.score_mean_sq_avg =
      (s.score_mean_sq_avg * old_w + score_mean_sq * weight) / new_w;
    s.lead_avg = (s.lead_avg * old_w + lead * weight) / new_w;
    s.utility_avg = (s.utility_avg * old_w + utility * weight) / new_w;
    s.utility_sq_avg = (s.utility_sq_avg * old_w + utility_sq * weight) / new_w;
    s.weight_sq_sum += weight_sq;
    s.weight_sum = new_w;
    s.visits += 1;
  }
}

// -----------------------------------------------------------------------
// downweight_bad_children
// -----------------------------------------------------------------------

/// Downweight children whose utility is bad relative to the best child,
/// as in C++ `downweightBadChildrenAndNormalizeWeight`.
fn downweight_bad_children(
  weights: &mut Vec<f64>,
  utilities: &[f64],
  total_weight: f64,
  params: &SearchParams,
  dist: &DistributionTable,
) {
  if params.value_weight_exponent == 0.0 || weights.is_empty() {
    return;
  }
  // Find mean and stdev of utilities
  let mut w_sum = 0.0f64;
  let mut u_sum = 0.0f64;
  for (i, &w) in weights.iter().enumerate() {
    w_sum += w;
    u_sum += utilities[i] * w;
  }
  if w_sum <= 0.0 {
    return;
  }
  let u_mean = u_sum / w_sum;
  let mut u_sq_sum = 0.0f64;
  for (i, &w) in weights.iter().enumerate() {
    let diff = utilities[i] - u_mean;
    u_sq_sum += diff * diff * w;
  }
  let u_stdev = (u_sq_sum / w_sum).sqrt().max(1e-4);

  for (i, w) in weights.iter_mut().enumerate() {
    let z = (utilities[i] - u_mean) / u_stdev;
    let cdf = dist.cdf(z);
    // Weight by cdf^value_weight_exponent — downweights low-utility children
    let factor = cdf.powf(params.value_weight_exponent);
    *w *= factor;
  }
  // Renormalize to preserve total_weight
  let new_sum: f64 = weights.iter().sum();
  if new_sum > 0.0 {
    let scale = total_weight / new_sum;
    for w in weights.iter_mut() {
      *w *= scale;
    }
  }
}

// -----------------------------------------------------------------------
// recompute_node_stats
// -----------------------------------------------------------------------

/// Fully recompute `node.stats` from children + own NN leaf.
/// Called on each node along the path from leaf to root after a playout.
pub fn recompute_node_stats(
  node: &mut SearchNode,
  params: &SearchParams,
  dist: &DistributionTable,
  recent_score_center: f64,
  sqrt_board_area: f64,
) {
  let nn = match &node.nn_output {
    Some(n) => n,
    None => return,
  };

  // Collect child stats with edge-visit-weighted downscaling
  let mut child_weights: Vec<f64> = Vec::new();
  let mut child_utilities: Vec<f64> = Vec::new();
  let mut child_stats: Vec<NodeStats> = Vec::new();

  for edge in &node.children {
    let cs = &edge.child.stats;
    if cs.visits <= 0 || cs.weight_sum <= 0.0 || edge.edge_visits <= 0 {
      continue;
    }
    let w =
      cs.weight_sum * (edge.edge_visits as f64) / (cs.visits.max(1) as f64);
    child_weights.push(w);
    child_utilities.push(cs.utility_avg);
    child_stats.push(cs.clone());
  }

  let total_child_weight: f64 = child_weights.iter().sum();

  // Optionally downweight bad children
  if !child_weights.is_empty() {
    downweight_bad_children(
      &mut child_weights,
      &child_utilities,
      total_child_weight,
      params,
      dist,
    );
  }

  // Aggregate weighted sum from children
  let mut win_loss_sum = 0.0f64;
  let mut no_result_sum = 0.0f64;
  let mut score_mean_sum = 0.0f64;
  let mut score_mean_sq_sum = 0.0f64;
  let mut lead_sum = 0.0f64;
  let mut utility_sum = 0.0f64;
  let mut utility_sq_sum = 0.0f64;
  let mut weight_sq_sum = 0.0f64;
  let mut weight_sum = 0.0f64;

  for (i, &dw) in child_weights.iter().enumerate() {
    let cs = &child_stats[i];
    let scale = dw / cs.weight_sum.max(1e-30);
    win_loss_sum += dw * cs.win_loss_value_avg;
    no_result_sum += dw * cs.no_result_value_avg;
    score_mean_sum += dw * cs.score_mean_avg;
    score_mean_sq_sum += dw * cs.score_mean_sq_avg;
    lead_sum += dw * cs.lead_avg;
    utility_sum += dw * cs.utility_avg;
    utility_sq_sum += dw * cs.utility_sq_avg;
    weight_sq_sum += scale * scale * cs.weight_sq_sum;
    weight_sum += dw;
  }

  // Add own NN leaf contribution
  let win_loss = (nn.white_win_prob - nn.white_loss_prob) as f64;
  let no_result = nn.white_no_result_prob as f64;
  let score_mean = nn.white_score_mean as f64;
  let score_mean_sq = nn.white_score_mean_sq as f64;
  let lead = nn.white_lead as f64;
  let utility = scorevalue::get_utility(
    win_loss,
    no_result,
    score_mean,
    score_mean_sq,
    recent_score_center,
    sqrt_board_area,
    params,
  );
  let weight = compute_weight(nn, params);

  win_loss_sum += win_loss * weight;
  no_result_sum += no_result * weight;
  score_mean_sum += score_mean * weight;
  score_mean_sq_sum += score_mean_sq * weight;
  lead_sum += lead * weight;
  utility_sum += utility * weight;
  utility_sq_sum += utility * utility * weight;
  weight_sq_sum += weight * weight;
  weight_sum += weight;

  if weight_sum <= 0.0 {
    return;
  }

  let visits = node.children.iter().map(|e| e.edge_visits).sum::<i64>() + 1;
  let s = &mut node.stats;
  s.win_loss_value_avg = win_loss_sum / weight_sum;
  s.no_result_value_avg = no_result_sum / weight_sum;
  s.score_mean_avg = score_mean_sum / weight_sum;
  s.score_mean_sq_avg = score_mean_sq_sum / weight_sum;
  s.lead_avg = lead_sum / weight_sum;
  s.utility_avg = utility_sum / weight_sum;
  s.utility_sq_avg = utility_sq_sum / weight_sum;
  s.weight_sq_sum = weight_sq_sum;
  s.weight_sum = weight_sum;
  s.visits = visits;
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::board::Player;
  use crate::neuralnet::nnoutput::NNOutput;
  use crate::search::node::SearchNode;
  use crate::search::params::SearchParams;

  fn make_nn(win: f32, loss: f32, score_mean: f32) -> NNOutput {
    NNOutput {
      white_win_prob: win,
      white_loss_prob: loss,
      white_no_result_prob: 1.0 - win - loss,
      white_score_mean: score_mean,
      white_score_mean_sq: score_mean * score_mean,
      white_lead: score_mean,
      shortterm_winloss_error: 0.0,
      shortterm_score_error: 0.0,
      policy_probs: vec![1.0 / 82.0; 82],
      owner_map: None,
      noised_policy: None,
    }
  }

  #[test]
  fn add_leaf_value_first_visit() {
    let mut node = SearchNode::new(Player::Black);
    node.nn_output = Some(make_nn(0.6, 0.3, 2.0));
    let p = SearchParams::default();
    add_leaf_value(&mut node, &p, 0.0, 3.0, true);
    assert_eq!(node.stats.visits, 1);
    assert!(
      (node.stats.win_loss_value_avg - 0.3).abs() < 1e-6,
      "win_loss = win - loss = 0.6 - 0.3 = 0.3"
    );
    assert!(node.stats.weight_sum > 0.0);
  }

  #[test]
  fn add_leaf_value_running_average() {
    let mut node = SearchNode::new(Player::Black);
    node.nn_output = Some(make_nn(0.6, 0.3, 2.0));
    let p = SearchParams::default();
    add_leaf_value(&mut node, &p, 0.0, 3.0, true);
    // Second update with same NN value
    add_leaf_value(&mut node, &p, 0.0, 3.0, false);
    assert_eq!(node.stats.visits, 2);
    // The average should still be the same since values are equal
    assert!((node.stats.win_loss_value_avg - 0.3).abs() < 1e-5);
  }
}
