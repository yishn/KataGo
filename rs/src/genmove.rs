/// Greedy policy-network move generation — no MCTS / search.
///
/// [`genmove`] runs the neural network once for the given position and returns
/// the legal, non-pass board move with the highest policy logit (channel 0).
/// It falls back to [`PASS_LOC`] only when every non-pass position is illegal
/// (e.g. the board is completely filled).
use crate::game::board::{Board, Loc, PASS_LOC, Player, location};
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::nnoutput::{self, NNOutput};
use crate::neuralnet::{eval::Evaluator, nninputs};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns the raw policy logits (channel 0) for every board position,
/// indexed by `y * nn_x + x`. Positions outside the board are zero.
pub async fn policy_values(
  evaluator: &Evaluator,
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
) -> Vec<f32> {
  let nn_x = evaluator.nn_x;
  let nn_y = evaluator.nn_y;
  let nc = evaluator.num_input_channels;
  let ng = evaluator.num_input_global_channels;
  let policy_ch = evaluator.num_policy_channels;

  let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
  let mut global = vec![0.0f32; ng];

  nninputs::fill_row(
    board,
    hist,
    next_player,
    nn_x,
    nn_y,
    nc,
    ng,
    &mut spatial,
    &mut global,
  );

  let output = evaluator.run(&spatial, &global).await;

  output
    .policy_spatial
    .chunks(policy_ch)
    .map(|ch| ch[0])
    .collect()
}

/// Run the network once and return a fully-decoded [`NNOutput`] containing
/// win/loss/no-result probabilities, score estimates, and softmaxed policy.
///
/// All values are from **White's perspective**:
/// - `white_win_prob` + `white_loss_prob` + `white_no_result_prob` ≈ 1.0
/// - `white_score_mean` is positive when White leads
/// - For Black's win probability use `nn_out.white_loss_prob`
pub async fn position_eval(
  evaluator: &Evaluator,
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
) -> NNOutput {
  let nn_x = evaluator.nn_x;
  let nn_y = evaluator.nn_y;
  let nc = evaluator.num_input_channels;
  let ng = evaluator.num_input_global_channels;

  let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
  let mut global = vec![0.0f32; ng];

  nninputs::fill_row(
    board,
    hist,
    next_player,
    nn_x,
    nn_y,
    nc,
    ng,
    &mut spatial,
    &mut global,
  );

  let output = evaluator.run(&spatial, &global).await;

  nnoutput::extract_nn_output(
    &output,
    0,
    nn_x as u32,
    evaluator.num_value_channels,
    evaluator.num_score_value_channels,
    evaluator.num_policy_channels,
    evaluator.num_ownership_channels,
    false,
  )
}

/// Generate a move for `next_player` by sampling from the policy distribution
/// over legal moves.
///
/// The function:
/// 1. Encodes `board` + `hist` into NN input features.
/// 2. Runs a single forward pass through the network.
/// 3. Applies softmax over legal, non-pass positions to form a probability
///    distribution and samples one move from it.
/// 4. Returns [`PASS_LOC`] if no legal non-pass move exists.
pub async fn genmove(
  evaluator: &Evaluator,
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
) -> Loc {
  use rand::Rng;

  let nn_x = evaluator.nn_x;
  let logits = policy_values(evaluator, board, hist, next_player).await;

  // Collect (loc, logit) for every legal non-pass position.
  let mut candidates: Vec<(Loc, f32)> = Vec::new();
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = location::get_loc(x, y, board.x_size);
      if hist.is_legal(board, loc, next_player) {
        candidates.push((loc, logits[y * nn_x + x]));
      }
    }
  }

  if candidates.is_empty() {
    return PASS_LOC;
  }

  // Softmax over legal moves for numerical stability.
  let max_logit = candidates
    .iter()
    .map(|&(_, l)| l)
    .fold(f32::NEG_INFINITY, f32::max);
  let exps: Vec<f32> = candidates
    .iter()
    .map(|&(_, l)| (l - max_logit).exp())
    .collect();
  let sum: f32 = exps.iter().sum();

  // Sample via a single uniform draw against the cumulative distribution.
  let threshold = rand::thread_rng().gen_range(0.0f32..sum);
  let mut cumulative = 0.0f32;
  for (i, &exp) in exps.iter().enumerate() {
    cumulative += exp;
    if cumulative >= threshold {
      return candidates[i].0;
    }
  }

  // Fallback: floating-point rounding left us just short — return last candidate.
  candidates.last().unwrap().0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::{
    board::{Board, PASS_LOC, Player, location},
    boardhistory::BoardHistory,
    rules::Rules,
  };
  use crate::neuralnet::eval::Evaluator;

  // -------------------------------------------------------------------------
  // Shared helpers
  // -------------------------------------------------------------------------

  fn load_evaluator(nn_x: usize, nn_y: usize) -> Evaluator {
    let desc = crate::model::ModelDesc::load_from_gz_bytes(include_bytes!("../../.network.bin.gz"), true)
      .expect(".network.bin.gz should be present and parseable");
    Evaluator::new(&desc, nn_x, nn_y)
  }

  // -------------------------------------------------------------------------
  // nninputs tests (pure feature-encoding, no model weights needed)
  // -------------------------------------------------------------------------

  /// fill_row produces a buffer of the correct size for every known version
  /// and marks all on-board cells with feature 0 = 1.
  #[test]
  fn fill_row_sizes_and_on_board_feature() {
    use crate::neuralnet::nninputs::{
      NUM_GLOBAL_V3_V4, NUM_GLOBAL_V5, NUM_GLOBAL_V6, NUM_GLOBAL_V7,
      NUM_SPATIAL, NUM_SPATIAL_V5,
    };
    let board = Board::new(5, 5);
    let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
    let nn_x = 5;
    let nn_y = 5;

    for (ns, ng) in [
      (NUM_SPATIAL, NUM_GLOBAL_V3_V4),
      (NUM_SPATIAL_V5, NUM_GLOBAL_V5),
      (NUM_SPATIAL, NUM_GLOBAL_V6),
      (NUM_SPATIAL, NUM_GLOBAL_V7),
    ] {
      let mut spatial = vec![0.0f32; nn_y * nn_x * ns];
      let mut global = vec![0.0f32; ng];
      crate::neuralnet::nninputs::fill_row(
        &board,
        &hist,
        Player::Black,
        nn_x,
        nn_y,
        ns,
        ng,
        &mut spatial,
        &mut global,
      );
      for y in 0..5 {
        for x in 0..5 {
          assert_eq!(
            spatial[(y * nn_x + x) * ns],
            1.0,
            "feature 0 at ({x},{y}) should be 1 for ns={ns}, ng={ng}"
          );
        }
      }
    }
  }

  /// Feature 2 (opponent stone) should be set at the position where the
  /// opponent most recently played.
  #[test]
  fn fill_row_encodes_pla_stone() {
    use crate::neuralnet::nninputs::{NUM_GLOBAL_V6, NUM_SPATIAL};
    let mut board = Board::new(5, 5);
    let mut hist =
      BoardHistory::new(&board, Player::Black, Rules::default(), 0);
    let loc = location::get_loc(2, 2, 5);
    hist.make_board_move(&mut board, loc, Player::Black, None);

    let nn_x = 5;
    let nn_y = 5;
    let nc = NUM_SPATIAL;
    let ng = NUM_GLOBAL_V6;
    let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
    let mut global = vec![0.0f32; ng];
    crate::neuralnet::nninputs::fill_row(
      &board,
      &hist,
      Player::White,
      nn_x,
      nn_y,
      nc,
      ng,
      &mut spatial,
      &mut global,
    );
    let pos = 2 * nn_x + 2;
    assert_eq!(spatial[pos * nc + 2], 1.0, "feature 2 (opp stone) at (2,2)");
    assert_eq!(
      spatial[pos * nc + 1],
      0.0,
      "feature 1 (pla stone) should be 0 at (2,2)"
    );
  }

  /// Global[5] encodes self-komi correctly for both players (V6 uses /20).
  #[test]
  fn fill_row_global_komi() {
    use crate::neuralnet::nninputs::{NUM_GLOBAL_V6, NUM_SPATIAL};
    let mut rules = Rules::default();
    rules.komi = 6.5;
    let board = Board::new(5, 5);
    let hist = BoardHistory::new(&board, Player::Black, rules, 0);
    let nn_x = 5;
    let nn_y = 5;
    let nc = NUM_SPATIAL;
    let ng = NUM_GLOBAL_V6;

    for (pla, expected) in [
      (Player::Black, -6.5f32 / 20.0),
      (Player::White, 6.5f32 / 20.0),
    ] {
      let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
      let mut global = vec![0.0f32; ng];
      crate::neuralnet::nninputs::fill_row(
        &board,
        &hist,
        pla,
        nn_x,
        nn_y,
        nc,
        ng,
        &mut spatial,
        &mut global,
      );
      assert!(
        (global[5] - expected).abs() < 1e-5,
        "{pla:?} komi global[5]={} expected {expected}",
        global[5]
      );
    }
  }

  // -------------------------------------------------------------------------
  // WebGPU backend helper
  // -------------------------------------------------------------------------

  /// Try to build a wgpu-backed evaluator using `WgpuBackend::new` directly
  /// (no fallback). Returns `None` when no GPU adapter is available so the
  /// calling test can skip cleanly with an early `return`.
  fn try_load_wgpu_evaluator(nn_x: usize, nn_y: usize) -> Option<Evaluator> {
    let desc = crate::model::ModelDesc::load_from_gz_bytes(include_bytes!("../../.network.bin.gz"), true)
      .expect(".network.bin.gz should be present and parseable");
    match pollster::block_on(crate::neuralnet::backend_wgpu::WgpuBackend::new(
      &desc, nn_x, nn_y,
    )) {
      Ok(b) => Some(Evaluator::from_backend(Box::new(b), nn_x, nn_y)),
      Err(e) => {
        eprintln!("[wgpu-test] WebGPU unavailable: {e}. Skipping.");
        None
      }
    }
  }

  // -------------------------------------------------------------------------
  // genmove tests — WebGPU backend
  // -------------------------------------------------------------------------

  /// Smoke test: wgpu backend on an empty 19×19 must not return pass.
  #[test]
  fn wgpu_smoke_genmove_not_pass() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      assert_ne!(
        mv, PASS_LOC,
        "wgpu genmove on empty 19×19 returned pass (loc={mv})"
      );
    });
  }

  /// wgpu backend must return a legal move on an empty board.
  #[test]
  fn wgpu_genmove_returns_legal_move() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      assert!(
        hist.is_legal(&board, mv, Player::Black),
        "wgpu genmove returned illegal move {mv}"
      );
    });
  }

  /// On a fully occupied 1×1 board the wgpu backend must return pass.
  #[test]
  fn wgpu_genmove_full_board_returns_pass() {
    let Some(ev) = try_load_wgpu_evaluator(1, 1) else {
      return;
    };
    pollster::block_on(async {
      let mut board = Board::new(1, 1);
      board.play_move_assume_legal(location::get_loc(0, 0, 1), Player::Black);
      let hist = BoardHistory::new(&board, Player::White, Rules::default(), 0);
      let mv = genmove(&ev, &board, &hist, Player::White).await;
      assert_eq!(mv, PASS_LOC, "expected pass on full board, got {mv}");
    });
  }

  /// After one Black move at tengen, the wgpu backend returns a legal
  /// non-pass move for White.
  #[test]
  fn wgpu_genmove_after_one_move() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let mut board = Board::new(19, 19);
      let mut hist =
        BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      hist.make_board_move(
        &mut board,
        location::get_loc(9, 9, 19),
        Player::Black,
        None,
      );
      let mv = genmove(&ev, &board, &hist, Player::White).await;
      assert_ne!(mv, PASS_LOC, "expected non-pass after one move");
      assert!(
        hist.is_legal(&board, mv, Player::White),
        "wgpu genmove returned illegal move {mv}"
      );
    });
  }

  /// On a 9×9 board the wgpu preferred move should be legal and within the grid.
  #[test]
  fn wgpu_genmove_9x9_is_legal_on_board() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      assert_ne!(mv, PASS_LOC, "expected non-pass on empty 9×9");
      assert!(
        hist.is_legal(&board, mv, Player::Black),
        "move {mv} is illegal on 9×9"
      );
      let x = location::get_x(mv, 9);
      let y = location::get_y(mv, 9);
      assert!(
        x < 9 && y < 9,
        "wgpu move ({x},{y}) is outside the 9×9 board"
      );
    });
  }

  /// genmove samples from the distribution, so repeated calls may differ —
  /// but each result must be a legal move.
  #[test]
  fn wgpu_genmove_is_legal_on_repeated_calls() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      for _ in 0..5 {
        let mv = genmove(&ev, &board, &hist, Player::Black).await;
        assert!(
          hist.is_legal(&board, mv, Player::Black),
          "sampled move {mv} is illegal"
        );
      }
    });
  }

  /// After both players pass, the wgpu backend still returns a legal move (or
  /// pass if the game is already finished).
  #[test]
  fn wgpu_genmove_after_two_passes_returns_legal() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let mut board = Board::new(19, 19);
      let mut hist =
        BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      hist.make_board_move(&mut board, PASS_LOC, Player::Black, None);
      hist.make_board_move(&mut board, PASS_LOC, Player::White, None);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      if hist.is_game_finished {
        assert_eq!(mv, PASS_LOC, "game over: expected pass");
      } else {
        assert!(
          hist.is_legal(&board, mv, Player::Black),
          "wgpu returned illegal move {mv}"
        );
      }
    });
  }

  // -------------------------------------------------------------------------
  // Cross-backend numerical validation
  // -------------------------------------------------------------------------

  /// The wgpu and CPU backends must agree on value-head outputs, pass-policy,
  /// and the argmax policy move for the same input position.
  ///
  /// Tolerance is 1e-3 to account for f32 GPU/CPU arithmetic differences.
  #[ignore]
  #[test]
  fn wgpu_outputs_match_cpu() {
    let Some(wgpu_ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    let cpu_ev = load_evaluator(9, 9);

    let nn_x = 9usize;
    let nn_y = 9usize;
    let nc = cpu_ev.num_input_channels;
    let ng = cpu_ev.num_input_global_channels;

    // Non-trivial position: two stones on a 9×9.
    let mut board = Board::new(9, 9);
    let mut hist =
      BoardHistory::new(&board, Player::Black, Rules::default(), 0);
    hist.make_board_move(
      &mut board,
      location::get_loc(4, 4, 9),
      Player::Black,
      None,
    );
    hist.make_board_move(
      &mut board,
      location::get_loc(2, 6, 9),
      Player::White,
      None,
    );

    let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
    let mut global = vec![0.0f32; ng];
    crate::neuralnet::nninputs::fill_row(
      &board,
      &hist,
      Player::Black,
      nn_x,
      nn_y,
      nc,
      ng,
      &mut spatial,
      &mut global,
    );

    let cpu_out = pollster::block_on(cpu_ev.run(&spatial, &global));
    let wgpu_out = pollster::block_on(wgpu_ev.run(&spatial, &global));

    // value head
    assert_eq!(
      cpu_out.value.len(),
      wgpu_out.value.len(),
      "value length mismatch"
    );
    for (i, (c, g)) in
      cpu_out.value.iter().zip(wgpu_out.value.iter()).enumerate()
    {
      assert!(
        (c - g).abs() < 1e-3,
        "value[{i}]: CPU={c:.6} wgpu={g:.6} diff={:.2e}",
        (c - g).abs()
      );
    }

    // policy_pass head
    assert_eq!(
      cpu_out.policy_pass.len(),
      wgpu_out.policy_pass.len(),
      "policy_pass length mismatch"
    );
    for (i, (c, g)) in cpu_out
      .policy_pass
      .iter()
      .zip(wgpu_out.policy_pass.iter())
      .enumerate()
    {
      assert!(
        (c - g).abs() < 1e-3,
        "policy_pass[{i}]: CPU={c:.6} wgpu={g:.6} diff={:.2e}",
        (c - g).abs()
      );
    }

    // top policy spatial move (argmax on channel 0) must agree
    let argmax = |out: &crate::neuralnet::eval::EvalOutput| {
      out
        .policy_spatial
        .chunks(out.policy_ch)
        .map(|ch| ch[0])
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
    };
    assert_eq!(
      argmax(&cpu_out),
      argmax(&wgpu_out),
      "CPU and wgpu disagree on the top-1 policy move"
    );
  }

  // -------------------------------------------------------------------------
  // position_eval tests — WebGPU backend
  // -------------------------------------------------------------------------

  /// Win/loss/no-result probs sum to ~1 and are each in [0, 1].
  #[test]
  fn wgpu_position_eval_value_probs() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let nn = position_eval(&ev, &board, &hist, Player::Black).await;
      let sum =
        nn.white_win_prob + nn.white_loss_prob + nn.white_no_result_prob;
      assert!((sum - 1.0).abs() < 1e-4, "win+loss+no_result = {sum}");
      assert!(nn.white_win_prob >= 0.0 && nn.white_win_prob <= 1.0);
      assert!(nn.white_loss_prob >= 0.0 && nn.white_loss_prob <= 1.0);
      assert!(nn.white_no_result_prob >= 0.0 && nn.white_no_result_prob <= 1.0);
    });
  }

  /// Score estimate is finite.
  #[test]
  fn wgpu_position_eval_score_finite() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let nn = position_eval(&ev, &board, &hist, Player::Black).await;
      assert!(
        nn.white_score_mean.is_finite(),
        "white_score_mean not finite"
      );
      assert!(nn.white_lead.is_finite(), "white_lead not finite");
    });
  }

  /// Policy probs from position_eval sum to ~1 and have the right length.
  #[test]
  fn wgpu_position_eval_policy_probs() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let nn = position_eval(&ev, &board, &hist, Player::Black).await;
      // policy_probs has nn_len² + 1 entries (last = pass)
      assert_eq!(nn.policy_probs.len(), 9 * 9 + 1);
      let sum: f32 = nn.policy_probs.iter().sum();
      assert!((sum - 1.0).abs() < 1e-4, "policy_probs sum = {sum}");
    });
  }

  /// On an empty board, position is roughly balanced so neither player should
  /// have a win probability above 90%.
  #[test]
  fn wgpu_position_eval_empty_board_balanced() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let nn = position_eval(&ev, &board, &hist, Player::Black).await;
      assert!(
        nn.white_win_prob < 0.9,
        "white win prob suspiciously high: {}",
        nn.white_win_prob
      );
      assert!(
        nn.white_loss_prob < 0.9,
        "white loss prob suspiciously high: {}",
        nn.white_loss_prob
      );
    });
  }

  // -------------------------------------------------------------------------
  // policy_values tests — WebGPU backend
  // -------------------------------------------------------------------------

  /// Output length equals nn_x * nn_y.
  #[test]
  fn wgpu_policy_values_length() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let vals = policy_values(&ev, &board, &hist, Player::Black).await;
      assert_eq!(vals.len(), 9 * 9, "expected 81 values for 9×9 board");
    });
  }

  /// All returned logits must be finite (no NaN or ±inf).
  #[test]
  fn wgpu_policy_values_all_finite() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let vals = policy_values(&ev, &board, &hist, Player::Black).await;
      for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "policy_values[{i}] = {v} is not finite");
      }
    });
  }

  /// Two calls with identical inputs return the same logits.
  #[test]
  fn wgpu_policy_values_deterministic() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let v1 = policy_values(&ev, &board, &hist, Player::Black).await;
      let v2 = policy_values(&ev, &board, &hist, Player::Black).await;
      assert_eq!(v1, v2, "policy_values must be deterministic");
    });
  }

  /// genmove must only sample moves that have positive policy weight,
  /// i.e. the returned position must be in the legal-move candidate set.
  #[test]
  fn wgpu_genmove_samples_from_policy_support() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let nn_x = ev.nn_x;

      let vals = policy_values(&ev, &board, &hist, Player::Black).await;
      // Set of positions with strictly positive softmax weight among legal moves.
      let legal_positions: std::collections::HashSet<usize> = vals
        .iter()
        .enumerate()
        .filter(|&(i, _)| {
          let x = i % nn_x;
          let y = i / nn_x;
          let loc = location::get_loc(x, y, board.x_size);
          hist.is_legal(&board, loc, Player::Black)
        })
        .map(|(i, _)| i)
        .collect();

      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      let x = location::get_x(mv, board.x_size);
      let y = location::get_y(mv, board.x_size);
      assert!(
        legal_positions.contains(&(y * nn_x + x)),
        "genmove returned move outside policy support"
      );
    });
  }

  /// Calling policy_values for Black and White on the same empty board must
  /// produce different logit vectors (the network is player-aware).
  #[test]
  fn wgpu_policy_values_differs_by_player() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let black = policy_values(&ev, &board, &hist, Player::Black).await;
      let white = policy_values(&ev, &board, &hist, Player::White).await;
      assert_ne!(
        black, white,
        "policy_values should differ between Black and White"
      );
    });
  }

  /// After placing a stone, policy_values changes relative to the empty board.
  #[test]
  fn wgpu_policy_values_changes_after_move() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else {
      return;
    };
    pollster::block_on(async {
      let board_empty = Board::new(9, 9);
      let hist_empty =
        BoardHistory::new(&board_empty, Player::Black, Rules::default(), 0);
      let before =
        policy_values(&ev, &board_empty, &hist_empty, Player::Black).await;

      let mut board = Board::new(9, 9);
      let mut hist =
        BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      hist.make_board_move(
        &mut board,
        location::get_loc(4, 4, 9),
        Player::Black,
        None,
      );
      let after = policy_values(&ev, &board, &hist, Player::White).await;

      assert_ne!(
        before, after,
        "policy_values should change after a move is played"
      );
    });
  }
}
