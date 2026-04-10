/// Simple PUCT-based Monte Carlo Tree Search.
///
/// [`mcts`] runs `num_playouts` simulations from the root position and returns
/// an [`MctsOutput`] whose fields mirror [`NNOutput`]:
/// - `white_win_prob`, `white_loss_prob`, `white_no_result_prob` are the visit-
///   weighted average of all leaf evaluations (from White's perspective).
/// - `white_score_mean`, `white_score_mean_sq`, `white_lead` are similarly
///   averaged over visits.
/// - `policy_probs` is the **visit-count distribution** over moves (the MCTS
///   improved policy), normalised to sum to 1. Pass is at index `nn_x*nn_y`.
///
/// The tree uses PUCT exploration (the same formula as KataGo / AlphaZero):
///
/// ```text
/// score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
/// ```
///
/// where Q is the mean value, P is the prior (NN policy), and N the visit count.
use crate::game::{
  board::{Board, Loc, PASS_LOC, Player, location},
  boardhistory::BoardHistory,
};
use crate::neuralnet::{eval::Evaluator, nnoutput::NNOutput};

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// MCTS search result — parallel to [`NNOutput`] but derived from tree search.
#[derive(Debug, Clone)]
pub struct MctsOutput {
  /// Visit-weighted average White win probability.
  pub white_win_prob: f32,
  /// Visit-weighted average White loss probability.
  pub white_loss_prob: f32,
  /// Visit-weighted average no-result probability.
  pub white_no_result_prob: f32,
  /// Visit-weighted average White score lead.
  pub white_score_mean: f32,
  /// Visit-weighted average White score mean-square.
  pub white_score_mean_sq: f32,
  /// Visit-weighted average White lead (score-head channel 2).
  pub white_lead: f32,
  /// Visit-count distribution over moves (sums to 1). Length = nn_x*nn_y + 1.
  /// Pass is at index nn_x*nn_y. Unvisited or illegal moves have value 0.
  pub policy_probs: Vec<f32>,
  /// Best move according to visit counts (highest-visited child).
  pub best_move: Loc,
}

// ---------------------------------------------------------------------------
// PUCT constant
// ---------------------------------------------------------------------------

const C_PUCT: f32 = 1.5;

// ---------------------------------------------------------------------------
// Internal tree node
// ---------------------------------------------------------------------------

struct Node {
  /// Move that led to this node (PASS_LOC for the root).
  loc: Loc,
  /// Total visit count.
  visits: u32,
  /// Sum of value estimates seen in subtree (White's perspective: win - loss).
  total_value: f32,
  /// Sum of `white_score_mean` seen in subtree.
  total_score_mean: f32,
  /// Sum of `white_score_mean_sq` seen in subtree.
  total_score_mean_sq: f32,
  /// Sum of `white_lead` seen in subtree.
  total_lead: f32,
  /// Sum of `white_win_prob`, `white_loss_prob`, `white_no_result_prob`.
  total_win: f32,
  total_loss: f32,
  total_no_result: f32,
  /// Prior probability from the parent's NN policy.
  prior: f32,
  /// Child nodes (empty until this node is expanded).
  children: Vec<Node>,
}

impl Node {
  fn new(loc: Loc, prior: f32) -> Self {
    Node {
      loc,
      visits: 0,
      total_value: 0.0,
      total_score_mean: 0.0,
      total_score_mean_sq: 0.0,
      total_lead: 0.0,
      total_win: 0.0,
      total_loss: 0.0,
      total_no_result: 0.0,
      prior,
      children: Vec::new(),
    }
  }

  fn mean_value(&self) -> f32 {
    if self.visits == 0 {
      0.0
    } else {
      self.total_value / self.visits as f32
    }
  }
}

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

/// Run PUCT MCTS for `num_playouts` simulations from `(board, hist, next_player)`.
///
/// Returns an [`MctsOutput`] summarising the search.
pub async fn mcts(
  evaluator: &Evaluator,
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
  num_playouts: u32,
) -> MctsOutput {
  // Evaluate the root position once to get the NN prior.
  let root_nn =
    crate::genmove::position_eval(evaluator, board, hist, next_player).await;

  let nn_x = evaluator.nn_x;
  let nn_y = evaluator.nn_y;
  let policy = root_nn.get_policy();

  // Build the root node and expand it immediately.
  let mut root = Node::new(PASS_LOC, 1.0);
  expand_node(&mut root, board, hist, next_player, policy, nn_x, nn_y);

  // Seed the root with one virtual visit from the NN evaluation.
  let root_value = value_from_nn(&root_nn, next_player);
  backprop_node(&mut root, root_value, &root_nn);

  for _ in 0..num_playouts {
    let mut search_board = board.clone();
    let mut search_hist = hist.clone();

    run_playout(
      evaluator,
      &mut root,
      &mut search_board,
      &mut search_hist,
      next_player,
      nn_x,
      nn_y,
    )
    .await;
  }

  // Build the output from the final tree.
  build_output(&root, nn_x, nn_y, board.x_size)
}

// ---------------------------------------------------------------------------
// Single playout
// ---------------------------------------------------------------------------

/// Traverse the tree, expand a leaf, evaluate it, and backpropagate.
async fn run_playout(
  evaluator: &Evaluator,
  root: &mut Node,
  board: &mut Board,
  hist: &mut BoardHistory,
  root_player: Player,
  nn_x: usize,
  nn_y: usize,
) {
  // Phase 1 — Selection.
  //
  // Walk down the tree using Rust's reborrow rule: assigning
  // `node = &mut node.children[idx]` drops the parent borrow and creates a
  // child borrow, so each step is safe.  We record the chosen child index at
  // every level so we can re-navigate for backpropagation.
  let mut index_path: Vec<usize> = Vec::new();
  let mut current_player = root_player;
  {
    let mut node: &mut Node = root;
    while !node.children.is_empty() {
      let idx = select_child(node, node.visits);
      let child_loc = node.children[idx].loc;
      hist.make_board_move(board, child_loc, current_player, None);
      current_player = current_player.opponent();
      index_path.push(idx);
      node = &mut node.children[idx]; // reborrow: parent borrow released
    }
    // `node` (the leaf) is dropped here, releasing the borrow on `root`.
  }

  // Phase 2 — Evaluation (no borrow of the tree is held across the await).
  let nn =
    crate::genmove::position_eval(evaluator, board, hist, current_player).await;
  let policy = nn.get_policy().to_vec();

  // Phase 3 — Expansion: re-navigate to the leaf and add its children.
  {
    let leaf = navigate_mut(root, &index_path);
    expand_node(leaf, board, hist, current_player, &policy, nn_x, nn_y);
  }

  // Phase 4 — Backpropagation: update every node from root down to the leaf.
  // Each `navigate_mut` call is an independent fresh borrow, released before
  // the next iteration.
  let leaf_value = value_from_nn(&nn, current_player);
  for prefix_len in 0..=index_path.len() {
    let node = navigate_mut(root, &index_path[..prefix_len]);
    backprop_node(node, leaf_value, &nn);
  }
}

/// Walk `path` of child indices from `root`, returning a mutable reference to
/// the node at the end of the path.
fn navigate_mut<'a>(root: &'a mut Node, path: &[usize]) -> &'a mut Node {
  let mut node = root;
  for &idx in path {
    node = &mut node.children[idx];
  }
  node
}

// ---------------------------------------------------------------------------
// Node expansion
// ---------------------------------------------------------------------------

/// Populate `node.children` with all legal moves using `policy` as priors.
fn expand_node(
  node: &mut Node,
  board: &Board,
  hist: &BoardHistory,
  pla: Player,
  policy: &[f32],
  nn_x: usize,
  nn_y: usize,
) {
  let hw = nn_x * nn_y;

  // Collect legal board moves.
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = location::get_loc(x, y, board.x_size);
      if hist.is_legal(board, loc, pla) {
        let policy_idx = y * nn_x + x;
        let prior = if policy_idx < policy.len() {
          policy[policy_idx].max(0.0)
        } else {
          0.0
        };
        node.children.push(Node::new(loc, prior));
      }
    }
  }

  // Pass move.
  let pass_prior = if hw < policy.len() {
    policy[hw].max(0.0)
  } else {
    0.0
  };
  node.children.push(Node::new(PASS_LOC, pass_prior));
}

// ---------------------------------------------------------------------------
// PUCT child selection
// ---------------------------------------------------------------------------

fn select_child(node: &Node, parent_visits: u32) -> usize {
  let sqrt_parent = (parent_visits as f32).sqrt();
  let mut best_idx = 0;
  let mut best_score = f32::NEG_INFINITY;

  for (i, child) in node.children.iter().enumerate() {
    let q = child.mean_value();
    let u = C_PUCT * child.prior * sqrt_parent / (1.0 + child.visits as f32);
    let score = q + u;
    if score > best_score {
      best_score = score;
      best_idx = i;
    }
  }
  best_idx
}

// ---------------------------------------------------------------------------
// Value conversion and backpropagation
// ---------------------------------------------------------------------------

/// Convert NN output to a scalar value from `pla`'s perspective
/// (positive = good for `pla`).
fn value_from_nn(nn: &NNOutput, pla: Player) -> f32 {
  // Use win - loss as value signal. White-perspective, so flip for Black.
  let white_value = nn.white_win_prob - nn.white_loss_prob;
  match pla {
    Player::White => white_value,
    Player::Black => -white_value,
  }
}

/// Accumulate one evaluation into a node (always from the root player's POV,
/// stored as White-perspective internals so backprop is consistent).
fn backprop_node(node: &mut Node, leaf_value: f32, nn: &NNOutput) {
  node.visits += 1;
  node.total_value += leaf_value;
  node.total_win += nn.white_win_prob;
  node.total_loss += nn.white_loss_prob;
  node.total_no_result += nn.white_no_result_prob;
  node.total_score_mean += nn.white_score_mean;
  node.total_score_mean_sq += nn.white_score_mean_sq;
  node.total_lead += nn.white_lead;
}

// ---------------------------------------------------------------------------
// Build MctsOutput from the finished tree
// ---------------------------------------------------------------------------

fn build_output(
  root: &Node,
  nn_x: usize,
  nn_y: usize,
  x_size: usize,
) -> MctsOutput {
  let hw = nn_x * nn_y;
  let policy_len = hw + 1;
  let n = root.visits as f32;

  // Visit-weighted averages (accumulated over all playouts via backprop).
  let (win, loss, no_result, score_mean, score_mean_sq, lead) =
    if root.visits > 0 {
      (
        root.total_win / n,
        root.total_loss / n,
        root.total_no_result / n,
        root.total_score_mean / n,
        root.total_score_mean_sq / n,
        root.total_lead / n,
      )
    } else {
      (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0)
    };

  // Build the visit-count policy (improved policy).
  let mut policy_probs = vec![0.0f32; policy_len];
  let mut best_visits = 0u32;
  let mut best_move = PASS_LOC;
  let total_child_visits: u32 = root.children.iter().map(|c| c.visits).sum();

  for child in &root.children {
    if child.loc == PASS_LOC {
      policy_probs[hw] = child.visits as f32;
    } else {
      let x = location::get_x(child.loc, x_size);
      let y = location::get_y(child.loc, x_size);
      if x < nn_x && y < nn_y {
        policy_probs[y * nn_x + x] = child.visits as f32;
      }
    }
    if child.visits > best_visits {
      best_visits = child.visits;
      best_move = child.loc;
    }
  }
  // Normalise to a probability distribution.
  if total_child_visits > 0 {
    let denom = total_child_visits as f32;
    for p in &mut policy_probs {
      *p /= denom;
    }
  }

  MctsOutput {
    white_win_prob: win,
    white_loss_prob: loss,
    white_no_result_prob: no_result,
    white_score_mean: score_mean,
    white_score_mean_sq: score_mean_sq,
    white_lead: lead,
    policy_probs,
    best_move,
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::{
    board::{Board, PASS_LOC, Player},
    boardhistory::BoardHistory,
  };
  use crate::neuralnet::eval::Evaluator;

  fn try_load_wgpu_evaluator(nn_x: usize, nn_y: usize) -> Option<Evaluator> {
    let desc = crate::neuralnet::model::ModelDesc::load_from_gz_bytes(
      include_bytes!("../../.network.bin.gz"),
      true,
    )
    .expect(".network.bin.gz should be present and parseable");
    match pollster::block_on(crate::neuralnet::backend_wgpu::WgpuBackend::new(
      &desc,
    )) {
      Ok(b) => Some(Evaluator::from_backend(Box::new(b), nn_x, nn_y)),
      Err(e) => {
        eprintln!("[wgpu-test] WebGPU unavailable: {e}. Skipping.");
        None
      }
    }
  }

  /// MCTS on empty 9×9 returns a non-pass best move.
  #[test]
  fn mcts_best_move_not_pass() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, 7.5);
      let out = mcts(&ev, &board, &hist, Player::Black, 10).await;
      assert_ne!(out.best_move, PASS_LOC, "MCTS returned pass as best move");
    });
  }

  /// Best move must be legal.
  #[test]
  fn mcts_best_move_is_legal() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, 7.5);
      let out = mcts(&ev, &board, &hist, Player::Black, 10).await;
      assert!(
        hist.is_legal(&board, out.best_move, Player::Black),
        "best_move {} is illegal",
        out.best_move
      );
    });
  }

  /// Value probabilities sum to ~1.
  #[test]
  fn mcts_value_probs_sum_to_one() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, 7.5);
      let out = mcts(&ev, &board, &hist, Player::Black, 10).await;
      let sum =
        out.white_win_prob + out.white_loss_prob + out.white_no_result_prob;
      assert!((sum - 1.0).abs() < 1e-3, "value probs sum = {sum}");
    });
  }

  /// Visit-count policy sums to ~1 and has the right length.
  #[test]
  fn mcts_policy_probs_valid() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, 7.5);
      let out = mcts(&ev, &board, &hist, Player::Black, 20).await;
      assert_eq!(out.policy_probs.len(), 9 * 9 + 1);
      let sum: f32 = out.policy_probs.iter().sum();
      assert!((sum - 1.0).abs() < 1e-3, "policy_probs sum = {sum}");
    });
  }

  /// More playouts should not break anything.
  #[test]
  fn mcts_many_playouts() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, 7.5);
      let out = mcts(&ev, &board, &hist, Player::Black, 50).await;
      assert!(out.white_win_prob.is_finite());
      assert!(out.white_score_mean.is_finite());
    });
  }
}
