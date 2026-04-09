/// Greedy policy-network move generation — no MCTS / search.
///
/// [`genmove`] runs the neural network once for the given position and returns
/// the legal, non-pass board move with the highest policy logit (channel 0).
/// It falls back to [`PASS_LOC`] only when every non-pass position is illegal
/// (e.g. the board is completely filled).
use crate::game::board::{Board, Loc, Player, PASS_LOC, location};
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::{eval::Evaluator, nninputs};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a move for `next_player` using the policy head of `evaluator`.
///
/// The function:
/// 1. Encodes `board` + `hist` into NN input features.
/// 2. Runs a single forward pass through the network.
/// 3. Among all legal, non-pass board positions returns the one with the
///    highest policy logit (channel 0 of `policy_spatial`).
/// 4. Returns [`PASS_LOC`] if no legal non-pass move exists.
pub async fn genmove(
  evaluator: &Evaluator,
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
) -> Loc {
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

  // Pick the legal non-pass position with the highest policy logit (channel 0).
  let mut best_loc = PASS_LOC;
  let mut best_logit = f32::NEG_INFINITY;

  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = location::get_loc(x, y, board.x_size);
      if !hist.is_legal(board, loc, next_player) {
        continue;
      }
      let pos = y * nn_x + x;
      let logit = output.policy_spatial[pos * policy_ch]; // channel 0
      if logit > best_logit {
        best_logit = logit;
        best_loc = loc;
      }
    }
  }

  best_loc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::{
    board::{Board, Player, PASS_LOC, location},
    boardhistory::BoardHistory,
    rules::Rules,
  };
  use crate::neuralnet::eval::Evaluator;

  // -------------------------------------------------------------------------
  // Shared helpers
  // -------------------------------------------------------------------------

  fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .parent()
      .expect("workspace root")
      .to_path_buf()
  }

  /// Load the real `.network.bin.gz` from the workspace root and wrap it in
  /// an `Evaluator` sized for a `nn_x × nn_y` grid.
  fn load_evaluator(nn_x: usize, nn_y: usize) -> Evaluator {
    let path = workspace_root().join(".network.bin.gz");
    let desc = crate::model::ModelDesc::load_from_file(&path)
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
        &board, &hist, Player::Black, nn_x, nn_y, ns, ng,
        &mut spatial, &mut global,
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
    let mut hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
    let loc = location::get_loc(2, 2, 5);
    hist.make_board_move(&mut board, loc, Player::Black, None);

    let nn_x = 5;
    let nn_y = 5;
    let nc = NUM_SPATIAL;
    let ng = NUM_GLOBAL_V6;
    let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
    let mut global = vec![0.0f32; ng];
    crate::neuralnet::nninputs::fill_row(
      &board, &hist, Player::White, nn_x, nn_y, nc, ng,
      &mut spatial, &mut global,
    );
    let pos = 2 * nn_x + 2;
    assert_eq!(spatial[pos * nc + 2], 1.0, "feature 2 (opp stone) at (2,2)");
    assert_eq!(spatial[pos * nc + 1], 0.0, "feature 1 (pla stone) should be 0 at (2,2)");
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
      (Player::White,  6.5f32 / 20.0),
    ] {
      let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
      let mut global = vec![0.0f32; ng];
      crate::neuralnet::nninputs::fill_row(
        &board, &hist, pla, nn_x, nn_y, nc, ng,
        &mut spatial, &mut global,
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
  #[cfg(not(target_arch = "wasm32"))]
  fn try_load_wgpu_evaluator(nn_x: usize, nn_y: usize) -> Option<Evaluator> {
    let path = workspace_root().join(".network.bin.gz");
    let desc = crate::model::ModelDesc::load_from_file(&path)
      .expect(".network.bin.gz should be present and parseable");
    match pollster::block_on(crate::neuralnet::backend_wgpu::WgpuBackend::new(&desc, nn_x, nn_y)) {
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
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_smoke_genmove_not_pass() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else { return };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      assert_ne!(mv, PASS_LOC, "wgpu genmove on empty 19×19 returned pass (loc={mv})");
    });
  }

  /// wgpu backend must return a legal move on an empty board.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_genmove_returns_legal_move() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else { return };
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
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_genmove_full_board_returns_pass() {
    let Some(ev) = try_load_wgpu_evaluator(1, 1) else { return };
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
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_genmove_after_one_move() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else { return };
    pollster::block_on(async {
      let mut board = Board::new(19, 19);
      let mut hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      hist.make_board_move(&mut board, location::get_loc(9, 9, 19), Player::Black, None);
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
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_genmove_9x9_is_legal_on_board() {
    let Some(ev) = try_load_wgpu_evaluator(9, 9) else { return };
    pollster::block_on(async {
      let board = Board::new(9, 9);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      assert_ne!(mv, PASS_LOC, "expected non-pass on empty 9×9");
      assert!(hist.is_legal(&board, mv, Player::Black), "move {mv} is illegal on 9×9");
      let x = location::get_x(mv, 9);
      let y = location::get_y(mv, 9);
      assert!(x < 9 && y < 9, "wgpu move ({x},{y}) is outside the 9×9 board");
    });
  }

  /// Two consecutive wgpu forward passes with identical inputs must agree.
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_genmove_is_deterministic() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else { return };
    pollster::block_on(async {
      let board = Board::new(19, 19);
      let hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      let mv1 = genmove(&ev, &board, &hist, Player::Black).await;
      let mv2 = genmove(&ev, &board, &hist, Player::Black).await;
      assert_eq!(mv1, mv2, "wgpu genmove must be deterministic");
    });
  }

  /// After both players pass, the wgpu backend still returns a legal move (or
  /// pass if the game is already finished).
  #[test]
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_genmove_after_two_passes_returns_legal() {
    let Some(ev) = try_load_wgpu_evaluator(19, 19) else { return };
    pollster::block_on(async {
      let mut board = Board::new(19, 19);
      let mut hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
      hist.make_board_move(&mut board, PASS_LOC, Player::Black, None);
      hist.make_board_move(&mut board, PASS_LOC, Player::White, None);
      let mv = genmove(&ev, &board, &hist, Player::Black).await;
      if hist.is_game_finished {
        assert_eq!(mv, PASS_LOC, "game over: expected pass");
      } else {
        assert!(hist.is_legal(&board, mv, Player::Black), "wgpu returned illegal move {mv}");
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
  #[cfg(not(target_arch = "wasm32"))]
  fn wgpu_outputs_match_cpu() {
    let Some(wgpu_ev) = try_load_wgpu_evaluator(9, 9) else { return };
    let cpu_ev = load_evaluator(9, 9);

    let nn_x = 9usize;
    let nn_y = 9usize;
    let nc = cpu_ev.num_input_channels;
    let ng = cpu_ev.num_input_global_channels;

    // Non-trivial position: two stones on a 9×9.
    let mut board = Board::new(9, 9);
    let mut hist = BoardHistory::new(&board, Player::Black, Rules::default(), 0);
    hist.make_board_move(&mut board, location::get_loc(4, 4, 9), Player::Black, None);
    hist.make_board_move(&mut board, location::get_loc(2, 6, 9), Player::White, None);

    let mut spatial = vec![0.0f32; nn_y * nn_x * nc];
    let mut global  = vec![0.0f32; ng];
    crate::neuralnet::nninputs::fill_row(
      &board, &hist, Player::Black, nn_x, nn_y, nc, ng,
      &mut spatial, &mut global,
    );

    let cpu_out  = pollster::block_on(cpu_ev.run(&spatial, &global));
    let wgpu_out = pollster::block_on(wgpu_ev.run(&spatial, &global));

    // value head
    assert_eq!(cpu_out.value.len(), wgpu_out.value.len(), "value length mismatch");
    for (i, (c, g)) in cpu_out.value.iter().zip(wgpu_out.value.iter()).enumerate() {
      assert!(
        (c - g).abs() < 1e-3,
        "value[{i}]: CPU={c:.6} wgpu={g:.6} diff={:.2e}", (c - g).abs()
      );
    }

    // policy_pass head
    assert_eq!(cpu_out.policy_pass.len(), wgpu_out.policy_pass.len(), "policy_pass length mismatch");
    for (i, (c, g)) in cpu_out.policy_pass.iter().zip(wgpu_out.policy_pass.iter()).enumerate() {
      assert!(
        (c - g).abs() < 1e-3,
        "policy_pass[{i}]: CPU={c:.6} wgpu={g:.6} diff={:.2e}", (c - g).abs()
      );
    }

    // top policy spatial move (argmax on channel 0) must agree
    let argmax = |out: &crate::neuralnet::eval::EvalOutput| {
      out.policy_spatial
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
}
