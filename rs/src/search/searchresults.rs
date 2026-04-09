/// Search result extraction — move selection, values, LCB.
/// Mirrors `cpp/search/searchresults.cpp` (subset).
use crate::game::board::{Loc, PASS_LOC, Player};
use crate::search::node::SearchNode;
use crate::search::params::SearchParams;
use crate::search::scorevalue;

// -----------------------------------------------------------------------
// ReportedSearchValues
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ReportedSearchValues {
  pub win_value: f64,
  pub loss_value: f64,
  pub no_result_value: f64,
  pub win_loss_value: f64,
  pub expected_score: f64,
  pub expected_score_stdev: f64,
  pub lead: f64,
  pub utility: f64,
  pub weight: f64,
  pub visits: i64,
}

// -----------------------------------------------------------------------
// get_root_values
// -----------------------------------------------------------------------

pub fn get_root_values(
  root: &SearchNode,
  params: &SearchParams,
  recent_score_center: f64,
  sqrt_board_area: f64,
) -> Option<ReportedSearchValues> {
  let s = &root.stats;
  if s.visits <= 0 {
    return None;
  }
  let stdev =
    scorevalue::get_score_stdev(s.score_mean_avg, s.score_mean_sq_avg);
  Some(ReportedSearchValues {
    win_value: (s.win_loss_value_avg + 1.0) / 2.0,
    loss_value: (1.0 - s.win_loss_value_avg) / 2.0,
    no_result_value: s.no_result_value_avg,
    win_loss_value: s.win_loss_value_avg,
    expected_score: s.score_mean_avg,
    expected_score_stdev: stdev,
    lead: s.lead_avg,
    utility: s.utility_avg,
    weight: s.weight_sum,
    visits: s.visits,
  })
}

// -----------------------------------------------------------------------
// LCB (Lower Confidence Bound) for move selection
// -----------------------------------------------------------------------

/// LCB radius: `stdev(utility) / sqrt(visits) * lcb_stdevs`
fn lcb_radius(
  child: &SearchNode,
  edge_visits: i64,
  params: &SearchParams,
) -> f64 {
  if child.stats.visits <= 0
    || child.stats.weight_sum <= 0.0
    || edge_visits <= 0
  {
    return 1e30;
  }
  let stdev = child.stats.utility_stdev();
  // Effective visits taking edge_visits weighting into account
  let eff_visits = (child.stats.weight_sum * edge_visits as f64)
    / (child.stats.visits.max(1) as f64);
  if eff_visits <= 0.0 {
    return 1e30;
  }
  stdev / eff_visits.sqrt() * params.lcb_stdevs
}

// -----------------------------------------------------------------------
// get_play_selection_values
// -----------------------------------------------------------------------

/// Returns `(loc, selection_value)` pairs, sorted descending by selection value.
/// Applies subtract/prune and optionally LCB boosting.
pub fn get_play_selection_values(
  root: &SearchNode,
  params: &SearchParams,
  recent_score_center: f64,
  sqrt_board_area: f64,
) -> Vec<(Loc, f64)> {
  if root.stats.visits <= 0 {
    return Vec::new();
  }

  let mut results: Vec<(Loc, f64)> = Vec::new();

  for edge in &root.children {
    if edge.edge_visits <= 0 {
      continue;
    }
    let child = &edge.child;
    if child.stats.visits <= 0 {
      continue;
    }

    // Base value: edge visits
    let mut val = edge.edge_visits as f64;
    results.push((edge.move_loc, val));
    let _ = val; // val used via last push
  }

  if results.is_empty() {
    return results;
  }

  // Apply chosenMoveSubtract and chosenMovePrune
  let max_val = results
    .iter()
    .map(|&(_, v)| v)
    .fold(f64::NEG_INFINITY, f64::max);
  let subtract = params.chosen_move_subtract.min(max_val / 64.0);
  let prune = params.chosen_move_prune.min(max_val / 64.0);

  results.retain(|&(_, v)| v - subtract >= prune);
  for (_, v) in results.iter_mut() {
    *v = (*v - subtract).max(0.0);
  }

  // LCB boosting: if use_lcb_for_selection, apply LCB adjustment
  if params.use_lcb_for_selection && max_val > 0.0 {
    let min_visits_for_lcb = (max_val * params.min_visit_prop_for_lcb) as i64;
    // Find best move by raw visits
    if let Some(&(best_loc, best_visits)) =
      results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
      // Find best child's utility for LCB reference
      let best_utility = root
        .children
        .iter()
        .find(|e| e.move_loc == best_loc)
        .map(|e| {
          if params.use_non_buggy_lcb {
            e.child.stats.utility_avg
              - lcb_radius(&e.child, e.edge_visits, params)
          } else {
            e.child.stats.utility_avg
          }
        })
        .unwrap_or(0.0);
      let pla = root.next_pla;
      let sign = if pla == Player::White { 1.0 } else { -1.0 };

      let best_lcb = best_utility * sign;

      for (loc, val) in results.iter_mut() {
        if *loc == best_loc {
          continue;
        }
        let edge = root.children.iter().find(|e| e.move_loc == *loc);
        if let Some(edge) = edge {
          if edge.edge_visits < min_visits_for_lcb {
            continue;
          }
          let child_utility = edge.child.stats.utility_avg;
          let child_lcb = child_utility * sign
            - lcb_radius(&edge.child, edge.edge_visits, params);
          // If child's LCB > best's LCB, boost it above best_visits
          if child_lcb > best_lcb {
            *val = best_visits + (child_lcb - best_lcb);
          }
        }
      }
    }
  }

  results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
  results
}

// -----------------------------------------------------------------------
// get_chosen_move_loc
// -----------------------------------------------------------------------

/// Sample a move using `chosen_move_temperature`.
pub fn get_chosen_move_loc(
  root: &SearchNode,
  params: &SearchParams,
  recent_score_center: f64,
  sqrt_board_area: f64,
  rng: &mut impl rand::Rng,
) -> Loc {
  let mut vals = get_play_selection_values(
    root,
    params,
    recent_score_center,
    sqrt_board_area,
  );
  if vals.is_empty() {
    return PASS_LOC;
  }

  let temp = params.chosen_move_temperature;
  if temp <= 0.0 {
    // Deterministic: return best
    return vals[0].0;
  }

  // Apply temperature
  let below_prob = params.chosen_move_temperature_only_below_prob;
  let max_val = vals
    .iter()
    .map(|&(_, v)| v)
    .fold(f64::NEG_INFINITY, f64::max);
  for (_, v) in vals.iter_mut() {
    if max_val <= 0.0 || (*v / max_val) < below_prob {
      *v = v.powf(1.0 / temp);
    }
  }
  let sum: f64 = vals.iter().map(|&(_, v)| v).sum();
  if sum <= 0.0 {
    return vals[0].0;
  }

  let mut r: f64 = rng.r#gen::<f64>() * sum;
  for &(loc, v) in &vals {
    r -= v;
    if r <= 0.0 {
      return loc;
    }
  }
  vals.last().unwrap().0
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::board::{Board, Player};
  use crate::neuralnet::nnoutput::NNOutput;
  use crate::search::node::{ChildEdge, SearchNode};
  use crate::search::params::SearchParams;

  fn make_evaluated_node(pla: Player, win: f32, loss: f32) -> SearchNode {
    let mut n = SearchNode::new(pla);
    n.nn_output = Some(NNOutput {
      white_win_prob: win,
      white_loss_prob: loss,
      white_no_result_prob: 1.0 - win - loss,
      white_score_mean: 0.0,
      white_score_mean_sq: 0.0,
      white_lead: 0.0,
      shortterm_winloss_error: 0.0,
      shortterm_score_error: 0.0,
      policy_probs: vec![1.0 / 82.0; 82],
      owner_map: None,
      noised_policy: None,
    });
    n.stats.visits = 1;
    n.stats.weight_sum = 1.0;
    n.stats.win_loss_value_avg = (win - loss) as f64;
    n.stats.utility_avg = (win - loss) as f64;
    n
  }

  #[test]
  fn root_values_with_visits() {
    let mut root = make_evaluated_node(Player::Black, 0.7, 0.2);
    root.stats.score_mean_avg = 5.0;
    root.stats.score_mean_sq_avg = 25.0;
    let p = SearchParams::default();
    let rv = get_root_values(&root, &p, 0.0, 4.0).unwrap();
    assert!((rv.win_loss_value - 0.5).abs() < 1e-6);
    assert_eq!(rv.visits, 1);
  }

  #[test]
  fn root_values_no_visits_returns_none() {
    let root = SearchNode::new(Player::Black);
    let p = SearchParams::default();
    assert!(get_root_values(&root, &p, 0.0, 4.0).is_none());
  }
}
