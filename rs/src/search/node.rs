/// MCTS tree node and related types.
/// Mirrors `cpp/search/searchnode.h` (simplified for single-threaded use).
use crate::game::board::{Loc, PASS_LOC, Player};
use crate::neuralnet::nnoutput::NNOutput;

// -----------------------------------------------------------------------
// NodeStats — per-node aggregated statistics
// -----------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct NodeStats {
  pub visits: i64,
  pub weight_sum: f64,
  pub weight_sq_sum: f64,
  pub win_loss_value_avg: f64,
  pub no_result_value_avg: f64,
  pub score_mean_avg: f64,
  pub score_mean_sq_avg: f64,
  pub lead_avg: f64,
  pub utility_avg: f64,
  pub utility_sq_avg: f64,
}

impl NodeStats {
  /// Utility stdev estimate: sqrt(max(0, E[U²] - E[U]²))
  pub fn utility_stdev(&self) -> f64 {
    let var = self.utility_sq_avg - self.utility_avg * self.utility_avg;
    if var > 0.0 { var.sqrt() } else { 0.0 }
  }
}

// -----------------------------------------------------------------------
// ChildEdge — pointer from parent to a child node with edge metadata
// -----------------------------------------------------------------------

pub struct ChildEdge {
  pub move_loc: Loc,
  /// Number of times this edge has been traversed.
  pub edge_visits: i64,
  pub child: Box<SearchNode>,
}

// -----------------------------------------------------------------------
// SearchNode — a node in the MCTS tree
// -----------------------------------------------------------------------

pub struct SearchNode {
  pub next_pla: Player,
  /// Set for root to prevent premature terminal detection.
  pub force_non_terminal: bool,
  /// NN evaluation output (None until first visit).
  pub nn_output: Option<NNOutput>,
  pub stats: NodeStats,
  /// Children sorted by policy probability (descending) after first expansion.
  pub children: Vec<ChildEdge>,
  /// Policy probabilities parallel to children (all legal moves, sorted desc).
  pub policy_probs: Vec<f32>,
  /// Move locs parallel to policy_probs (all legal moves, sorted desc by policy).
  pub move_locs: Vec<Loc>,
}

impl SearchNode {
  pub fn new(pla: Player) -> Self {
    SearchNode {
      next_pla: pla,
      force_non_terminal: false,
      nn_output: None,
      stats: NodeStats::default(),
      children: Vec::new(),
      policy_probs: Vec::new(),
      move_locs: Vec::new(),
    }
  }

  /// Sum of edge_visits across all children — the "N_parent" for PUCT.
  pub fn total_child_weight(&self) -> f64 {
    self.children.iter().map(|c| c.edge_visits as f64).sum()
  }

  /// True if the node has been evaluated by the NN.
  pub fn is_evaluated(&self) -> bool {
    self.nn_output.is_some()
  }

  /// True if children have been expanded from the policy.
  pub fn is_expanded(&self) -> bool {
    !self.move_locs.is_empty()
  }

  /// The pass-move position index in policy_probs / move_locs.
  pub fn pass_pos(&self) -> Option<usize> {
    self.move_locs.iter().position(|&l| l == PASS_LOC)
  }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn new_node_defaults() {
    let n = SearchNode::new(Player::Black);
    assert_eq!(n.stats.visits, 0);
    assert!(n.nn_output.is_none());
    assert!(n.children.is_empty());
    assert!(!n.is_evaluated());
    assert!(!n.is_expanded());
  }

  #[test]
  fn total_child_weight() {
    let mut n = SearchNode::new(Player::Black);
    let child1 = Box::new(SearchNode::new(Player::White));
    let child2 = Box::new(SearchNode::new(Player::White));
    n.children.push(ChildEdge {
      move_loc: 10,
      edge_visits: 3,
      child: child1,
    });
    n.children.push(ChildEdge {
      move_loc: 20,
      edge_visits: 7,
      child: child2,
    });
    assert_eq!(n.total_child_weight(), 10.0);
  }
}
