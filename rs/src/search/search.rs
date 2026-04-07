/// Main MCTS search engine.
/// Mirrors `cpp/search/search.h/cpp` for single-threaded tree search.
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Dirichlet, Distribution};

use crate::game::board::{Board, Loc, PASS_LOC, Player};
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::eval::{EvalOutput, Evaluator};
use crate::search::distributiontable::DistributionTable;
use crate::search::nnoutput::{
  NNOutput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES, extract_nn_output,
  fill_row_v7, loc_to_pos,
};
use crate::search::node::{ChildEdge, SearchNode};
use crate::search::params::SearchParams;
use crate::search::searchpuct;
use crate::search::searchresults::{
  ReportedSearchValues, get_chosen_move_loc, get_play_selection_values,
  get_root_values,
};
use crate::search::searchupdatehelpers::{
  add_leaf_value, recompute_node_stats,
};

// -----------------------------------------------------------------------
// Search struct
// -----------------------------------------------------------------------

pub struct Search {
  pub root_pla: Player,
  pub root_board: Board,
  pub root_history: BoardHistory,
  pub params: SearchParams,
  /// Board length (in cells per side, e.g. 19).
  pub nn_len: u32,

  root_node: Option<Box<SearchNode>>,
  dist: DistributionTable,
  recent_score_center: f64,
  // Model metadata for extract_nn_output
  value_ch: usize,
  score_ch: usize,
  policy_ch: usize,
  ownership_ch: usize,
  rng: StdRng,
}

impl Search {
  pub fn new(
    board: Board,
    hist: BoardHistory,
    pla: Player,
    params: SearchParams,
    nn_len: u32,
    value_ch: usize,
    score_ch: usize,
    policy_ch: usize,
    ownership_ch: usize,
  ) -> Self {
    Search {
      root_pla: pla,
      root_board: board,
      root_history: hist,
      params,
      nn_len,
      root_node: None,
      dist: DistributionTable::default(),
      recent_score_center: 0.0,
      value_ch,
      score_ch,
      policy_ch,
      ownership_ch,
      rng: StdRng::from_entropy(),
    }
  }

  pub fn clear_search(&mut self) {
    self.root_node = None;
  }

  pub fn set_position(
    &mut self,
    pla: Player,
    board: Board,
    hist: BoardHistory,
  ) {
    self.root_pla = pla;
    self.root_board = board;
    self.root_history = hist;
    self.root_node = None;
  }

  /// Run `n` playouts from the current root. Returns the number of playouts completed.
  pub fn run_playouts(&mut self, n: usize, evaluator: &Evaluator) -> i64 {
    let sqrt_board_area =
      ((self.root_board.x_size * self.root_board.y_size) as f64).sqrt();

    for _ in 0..n {
      if self.root_node.is_none() {
        let mut node = Box::new(SearchNode::new(self.root_pla));
        node.force_non_terminal = true;
        self.root_node = Some(node);
      }
      let board = self.root_board.clone();
      let hist = self.root_history.clone();
      let pla = self.root_pla;
      // We use a raw pointer trick to call playout_descend recursively;
      // for single-threaded use this is safe.
      let root_ptr: *mut SearchNode = self.root_node.as_mut().unwrap().as_mut();
      let mut path: Vec<*mut SearchNode> = Vec::new();
      self.playout_descend(
        root_ptr,
        board,
        hist,
        pla,
        true,
        &mut path,
        evaluator,
        sqrt_board_area,
      );
    }

    self.root_visits()
  }

  /// Returns the number of visits at the root.
  pub fn root_visits(&self) -> i64 {
    self.root_node.as_ref().map(|n| n.stats.visits).unwrap_or(0)
  }

  /// Choose the best move (deterministic, using temperature).
  pub fn get_chosen_move_loc(&mut self) -> Loc {
    let root = match &self.root_node {
      Some(r) => r,
      None => return PASS_LOC,
    };
    let sqrt_board_area =
      ((self.root_board.x_size * self.root_board.y_size) as f64).sqrt();
    get_chosen_move_loc(
      root,
      &self.params,
      self.recent_score_center,
      sqrt_board_area,
      &mut self.rng,
    )
  }

  /// Get aggregated root values.
  pub fn get_root_values(&self) -> Option<ReportedSearchValues> {
    let root = self.root_node.as_ref()?;
    let sqrt_board_area =
      ((self.root_board.x_size * self.root_board.y_size) as f64).sqrt();
    get_root_values(
      root,
      &self.params,
      self.recent_score_center,
      sqrt_board_area,
    )
  }

  /// Get (loc, selection_value) pairs sorted descending.
  pub fn get_play_selection_values(&self) -> Vec<(Loc, f64)> {
    let root = match &self.root_node {
      Some(r) => r,
      None => return Vec::new(),
    };
    let sqrt_board_area =
      ((self.root_board.x_size * self.root_board.y_size) as f64).sqrt();
    get_play_selection_values(
      root,
      &self.params,
      self.recent_score_center,
      sqrt_board_area,
    )
  }

  // -----------------------------------------------------------------------
  // NN evaluation helper
  // -----------------------------------------------------------------------

  fn evaluate_position(
    &self,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    evaluator: &Evaluator,
  ) -> NNOutput {
    let nl = self.nn_len as usize;
    let hw = nl * nl;
    let mut spatial = vec![0.0f32; hw * NUM_SPATIAL_FEATURES];
    let mut global = vec![0.0f32; NUM_GLOBAL_FEATURES];
    fill_row_v7(board, hist, pla, self.nn_len, &mut spatial, &mut global);

    // The Evaluator expects NHWC spatial and global shaped for a batch of 1
    let eval_out: EvalOutput = evaluator.run(&spatial, &global);

    extract_nn_output(
      &eval_out,
      0,
      self.nn_len,
      self.value_ch,
      self.score_ch,
      self.policy_ch,
      self.ownership_ch,
      false,
    )
  }

  // -----------------------------------------------------------------------
  // Expand children from policy
  // -----------------------------------------------------------------------

  fn expand_children_from_policy(
    node: &mut SearchNode,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    nn_len: u32,
    multi_stone_suicide_legal: bool,
  ) {
    let nn = match &node.nn_output {
      Some(n) => n,
      None => return,
    };
    let policy = nn.get_policy();
    let nl = nn_len as usize;
    let pass_pos = nl * nl;

    // Collect all legal moves with their policy prob
    let mut moves: Vec<(Loc, f32)> = Vec::new();

    // Pass
    let pass_prob = *policy.get(pass_pos).unwrap_or(&0.0);
    if pass_prob >= 0.0 {
      moves.push((PASS_LOC, pass_prob));
    }

    // Board moves
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = crate::game::board::location::get_loc(x, y, board.x_size);
        if hist.is_legal(board, loc, pla) {
          let pos = loc_to_pos(loc, board.x_size, nn_len);
          let prob = if pos < policy.len() {
            policy[pos]
          } else {
            -1.0
          };
          if prob >= 0.0 {
            moves.push((loc, prob));
          }
        }
      }
    }

    // Sort descending by probability
    moves.sort_by(|a, b| {
      b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let (locs, probs): (Vec<Loc>, Vec<f32>) = moves.into_iter().unzip();
    node.move_locs = locs;
    node.policy_probs = probs;
  }

  // -----------------------------------------------------------------------
  // Apply Dirichlet noise + policy temperature at root
  // -----------------------------------------------------------------------

  fn apply_root_noise(&mut self) {
    let root = match self.root_node.as_mut() {
      Some(r) => r,
      None => return,
    };
    if root.move_locs.is_empty() {
      return;
    }
    let n = root.move_locs.len();

    // Policy temperature
    let temp = self.params.root_policy_temperature;
    if (temp - 1.0).abs() > 1e-6 {
      for p in root.policy_probs.iter_mut() {
        if *p > 0.0 {
          *p = (*p as f64).powf(1.0 / temp) as f32;
        }
      }
      let sum: f32 = root.policy_probs.iter().cloned().sum();
      if sum > 0.0 {
        for p in root.policy_probs.iter_mut() {
          *p /= sum;
        }
      }
    }

    // Dirichlet noise
    if !self.params.root_noise_enabled {
      return;
    }
    let alpha = self.params.root_dirichlet_noise_total_concentration / n as f64;
    if alpha <= 0.0 {
      return;
    }

    // Sample Dirichlet(alpha, ..., alpha) by sampling n Gamma(alpha, 1) values
    let noise: Vec<f64> =
      if let Ok(dirichlet) = Dirichlet::new_with_size(alpha, n) {
        dirichlet.sample(&mut self.rng)
      } else {
        return;
      };

    let w = self.params.root_dirichlet_noise_weight;
    let old_probs = root.policy_probs.clone();
    for (i, p) in root.policy_probs.iter_mut().enumerate() {
      let noised = (1.0 - w) * old_probs[i] as f64 + w * noise[i];
      *p = noised as f32;
    }
  }

  // -----------------------------------------------------------------------
  // playout_descend — recursive MCTS descent
  // -----------------------------------------------------------------------

  fn playout_descend(
    &mut self,
    node_ptr: *mut SearchNode,
    mut board: Board,
    mut hist: BoardHistory,
    pla: Player,
    is_root: bool,
    path: &mut Vec<*mut SearchNode>,
    evaluator: &Evaluator,
    sqrt_board_area: f64,
  ) {
    // Safety: node_ptr always points to a live SearchNode owned by the tree.
    let node: &mut SearchNode = unsafe { &mut *node_ptr };

    // ---- 1. Terminal check ----
    if hist.is_game_finished && !node.force_non_terminal {
      // Build a synthetic NNOutput from the final score.
      let (win_prob, loss_prob, no_result) = if hist.is_no_result {
        (0.0f32, 0.0f32, 1.0f32)
      } else {
        let score = hist.final_white_minus_black_score;
        if score > 0.0 {
          (1.0, 0.0, 0.0)
        } else if score < 0.0 {
          (0.0, 1.0, 0.0)
        } else {
          (0.5, 0.5, 0.0)
        }
      };
      let score_mean = hist.final_white_minus_black_score;
      let policy_size = (self.nn_len * self.nn_len + 1) as usize;
      if node.nn_output.is_none() {
        node.nn_output = Some(NNOutput {
          white_win_prob: win_prob,
          white_loss_prob: loss_prob,
          white_no_result_prob: no_result,
          white_score_mean: score_mean,
          white_score_mean_sq: score_mean * score_mean,
          white_lead: score_mean,
          shortterm_winloss_error: 0.0,
          shortterm_score_error: 0.0,
          policy_probs: vec![-1.0; policy_size],
          owner_map: None,
          noised_policy: None,
        });
      }
      let assume_fresh = node.stats.visits == 0;
      add_leaf_value(
        node,
        &self.params,
        self.recent_score_center,
        sqrt_board_area,
        assume_fresh,
      );
      self.backprop_path(path, sqrt_board_area);
      return;
    }

    // ---- 2. Unevaluated node: run NN ----
    if !node.is_evaluated() {
      let nn_out = self.evaluate_position(&board, &hist, pla, evaluator);
      node.nn_output = Some(nn_out);

      let multi_stone_suicide_legal = hist.rules.multi_stone_suicide_legal;
      Self::expand_children_from_policy(
        node,
        &board,
        &hist,
        pla,
        self.nn_len,
        multi_stone_suicide_legal,
      );

      if is_root {
        self.apply_root_noise();
      }

      let assume_fresh = node.stats.visits == 0;
      // Re-borrow node after apply_root_noise potentially mutated self.root_node
      let node: &mut SearchNode = unsafe { &mut *node_ptr };
      add_leaf_value(
        node,
        &self.params,
        self.recent_score_center,
        sqrt_board_area,
        assume_fresh,
      );
      self.backprop_path(path, sqrt_board_area);
      return;
    }

    // ---- 3. Select best child ----
    let (child_idx, move_loc) = searchpuct::select_best_child(
      node,
      pla,
      is_root,
      self.recent_score_center,
      sqrt_board_area,
      &self.params,
    );

    if move_loc == crate::game::board::NULL_LOC {
      // No legal moves — treat as pass
      return;
    }

    // ---- 4. Make the move ----
    let actual_move_loc = move_loc;
    hist.make_board_move(&mut board, actual_move_loc, pla, None);
    let next_pla = pla.opponent();

    // ---- 5. New or existing child ----
    let child_ptr: *mut SearchNode = if child_idx < node.children.len() {
      // Existing child: increment edge visits
      node.children[child_idx].edge_visits += 1;
      node.children[child_idx].child.as_mut() as *mut SearchNode
    } else {
      // New child: create and push
      let child = Box::new(SearchNode::new(next_pla));
      node.children.push(ChildEdge {
        move_loc: actual_move_loc,
        edge_visits: 1,
        child,
      });
      node.children.last_mut().unwrap().child.as_mut() as *mut SearchNode
    };

    // ---- 6. Recurse ----
    path.push(node_ptr);
    self.playout_descend(
      child_ptr,
      board,
      hist,
      next_pla,
      false,
      path,
      evaluator,
      sqrt_board_area,
    );
  }

  // -----------------------------------------------------------------------
  // Backpropagation — recompute stats on the path root→leaf
  // -----------------------------------------------------------------------

  fn backprop_path(
    &mut self,
    path: &mut Vec<*mut SearchNode>,
    sqrt_board_area: f64,
  ) {
    // Walk path in reverse (leaf → root), recomputing stats at each node.
    while let Some(node_ptr) = path.pop() {
      let node: &mut SearchNode = unsafe { &mut *node_ptr };
      recompute_node_stats(
        node,
        &self.params,
        &self.dist,
        self.recent_score_center,
        sqrt_board_area,
      );
    }
  }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::board::Board;
  use crate::game::boardhistory::BoardHistory;
  use crate::game::rules::Rules;
  use crate::search::params::SearchParams;

  fn make_search(board_size: usize) -> (Board, BoardHistory, SearchParams) {
    let board = Board::new(board_size, board_size);
    let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
    let params = SearchParams::default();
    (board, hist, params)
  }

  #[test]
  fn search_new_has_zero_visits() {
    let (board, hist, params) = make_search(9);
    let search = Search::new(board, hist, Player::Black, params, 9, 3, 6, 2, 1);
    assert_eq!(search.root_visits(), 0);
  }

  #[test]
  fn clear_search_resets_visits() {
    let (board, hist, params) = make_search(9);
    let mut search =
      Search::new(board, hist, Player::Black, params, 9, 3, 6, 2, 1);
    // Manually insert a root node with some visits
    let mut root = Box::new(SearchNode::new(Player::Black));
    root.stats.visits = 5;
    search.root_node = Some(root);
    search.clear_search();
    assert_eq!(search.root_visits(), 0);
  }

  // Integration test (GPU-dependent) is gated behind #[cfg(feature = "gpu_test")]
  // so it doesn't run in CI without a GPU.
}
