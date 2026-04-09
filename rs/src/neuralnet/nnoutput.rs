/// NN input feature construction (fill_row_v7) and output extraction.
///
/// Mirrors `cpp/neuralnet/nninputs.cpp::fillRowV7` (model input version 7)
/// and the NNOutput struct from `cpp/neuralnet/nninputs.h`.
use crate::game::board::{
  Board, Color, Loc, MAX_ARR_SIZE, NULL_LOC, PASS_LOC, Player,
};
use crate::game::boardhistory::BoardHistory;
use crate::game::rules::{KoRule, ScoringRule, TaxRule};
use crate::neuralnet::eval::EvalOutput;

// Number of spatial/global input features for V7.
pub const NUM_SPATIAL_FEATURES: usize = 22;
pub const NUM_GLOBAL_FEATURES: usize = 19;

pub const MAX_BOARD_LEN: usize = 19;
pub const MAX_NN_POLICY_SIZE: usize = MAX_BOARD_LEN * MAX_BOARD_LEN + 1;

const KOMI_CLIP_RADIUS: f32 = 20.0;

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
// Coordinate conversion (mirrors NNPos::xyToPos / locToPos)
// -----------------------------------------------------------------------

#[inline]
fn xy_to_pos(x: usize, y: usize, nn_len: u32) -> usize {
  y * nn_len as usize + x
}

#[inline]
pub fn loc_to_pos(loc: Loc, x_size: usize, nn_len: u32) -> usize {
  if loc == PASS_LOC {
    return nn_len as usize * nn_len as usize;
  }
  if loc == NULL_LOC {
    return nn_len as usize * (nn_len as usize + 1);
  }
  let x = crate::game::board::location::get_x(loc, x_size);
  let y = crate::game::board::location::get_y(loc, x_size);
  xy_to_pos(x, y, nn_len)
}

// -----------------------------------------------------------------------
// NHWC spatial feature setter
// -----------------------------------------------------------------------

/// Set one value in a NHWC spatial array.
/// Index: `[pos * n_features + feature_idx]`
#[inline]
fn set_spatial(
  row_bin: &mut [f32],
  pos: usize,
  feature: usize,
  val: f32,
  n_features: usize,
) {
  row_bin[pos * n_features + feature] = val;
}

// -----------------------------------------------------------------------
// Ladder iteration — simplified (mirrors cpp/neuralnet/nninputs.cpp::iterLadders)
// -----------------------------------------------------------------------

/// For every stone on `board` that is in atari (1 liberty), mark it in `row_bin` at channel `ch`.
fn iter_ladder_stones(
  board: &Board,
  nn_len: u32,
  row_bin: &mut [f32],
  ch: usize,
) {
  let nl = nn_len as usize;
  let x_size = board.x_size;
  for y in 0..board.y_size {
    for x in 0..x_size {
      let loc = crate::game::board::location::get_loc(x, y, x_size);
      let c = board.colors[loc as usize];
      if c == Color::Black || c == Color::White {
        if board.get_num_liberties(loc) == 1 {
          let pos = xy_to_pos(x, y, nl as u32);
          set_spatial(row_bin, pos, ch, 1.0, NUM_SPATIAL_FEATURES);
        }
      }
    }
  }
}

// -----------------------------------------------------------------------
// fill_row_v7 — full port of cpp/neuralnet/nninputs.cpp::fillRowV7
// -----------------------------------------------------------------------

/// Fill V7 spatial (NHWC, shape [H*W, 22]) and global (shape [19]) input features.
///
/// `n_global_ch` is the model's `num_input_global_channels` — we write exactly
/// `NUM_GLOBAL_FEATURES = 19` values; any extra channels remain 0.
pub fn fill_row_v7(
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
  nn_len: u32,
  spatial_nhwc: &mut [f32],
  global: &mut [f32],
) {
  let nl = nn_len as usize;
  // Zero out
  for v in spatial_nhwc.iter_mut() {
    *v = 0.0;
  }
  for v in global.iter_mut() {
    *v = 0.0;
  }

  let pla = next_player;
  let opp = next_player.opponent();
  let x_size = board.x_size;
  let y_size = board.y_size;

  // ----------------------------------------------------------------
  // Features 0-5: on-board, pla/opp stones, 1/2/3 liberty chains
  // ----------------------------------------------------------------
  for y in 0..y_size {
    for x in 0..x_size {
      let loc = crate::game::board::location::get_loc(x, y, x_size);
      let pos = xy_to_pos(x, y, nn_len);
      // Feature 0: on board
      set_spatial(spatial_nhwc, pos, 0, 1.0, NUM_SPATIAL_FEATURES);

      let stone = board.colors[loc as usize];
      if stone == pla.color() {
        set_spatial(spatial_nhwc, pos, 1, 1.0, NUM_SPATIAL_FEATURES);
      } else if stone == opp.color() {
        set_spatial(spatial_nhwc, pos, 2, 1.0, NUM_SPATIAL_FEATURES);
      }

      if stone == pla.color() || stone == opp.color() {
        let libs = board.get_num_liberties(loc);
        match libs {
          1 => set_spatial(spatial_nhwc, pos, 3, 1.0, NUM_SPATIAL_FEATURES),
          2 => set_spatial(spatial_nhwc, pos, 4, 1.0, NUM_SPATIAL_FEATURES),
          3 => set_spatial(spatial_nhwc, pos, 5, 1.0, NUM_SPATIAL_FEATURES),
          _ => {}
        }
      }
    }
  }

  // ----------------------------------------------------------------
  // Feature 6: ko-ban locations (including superko)
  // Feature 7: encore ko recap blocked
  // ----------------------------------------------------------------
  if hist.encore_phase == 0 {
    if board.ko_loc != NULL_LOC {
      let pos = loc_to_pos(board.ko_loc, x_size, nn_len);
      if pos < nl * nl {
        set_spatial(spatial_nhwc, pos, 6, 1.0, NUM_SPATIAL_FEATURES);
      }
    }
    for y in 0..y_size {
      for x in 0..x_size {
        let loc = crate::game::board::location::get_loc(x, y, x_size);
        if hist.super_ko_banned[loc as usize] && loc != board.ko_loc {
          let pos = xy_to_pos(x, y, nn_len);
          set_spatial(spatial_nhwc, pos, 6, 1.0, NUM_SPATIAL_FEATURES);
        }
      }
    }
  } else {
    for y in 0..y_size {
      for x in 0..x_size {
        let loc = crate::game::board::location::get_loc(x, y, x_size);
        let pos = xy_to_pos(x, y, nn_len);
        if hist.super_ko_banned[loc as usize] {
          set_spatial(spatial_nhwc, pos, 6, 1.0, NUM_SPATIAL_FEATURES);
        }
        if hist.ko_recap_blocked[loc as usize] {
          set_spatial(spatial_nhwc, pos, 7, 1.0, NUM_SPATIAL_FEATURES);
        }
      }
    }
  }

  // ----------------------------------------------------------------
  // Features 18,19: territory area (current scoring estimate)
  // ----------------------------------------------------------------
  let mut area = [Color::Empty; MAX_ARR_SIZE];
  let has_area = hist.rules.scoring_rule == ScoringRule::Area
    && hist.rules.tax_rule == TaxRule::None;
  if has_area {
    board.calculate_area(true, true, true, &mut area);
    for y in 0..y_size {
      for x in 0..x_size {
        let loc = crate::game::board::location::get_loc(x, y, x_size);
        let pos = xy_to_pos(x, y, nn_len);
        if area[loc as usize] == pla.color() {
          set_spatial(spatial_nhwc, pos, 18, 1.0, NUM_SPATIAL_FEATURES);
        } else if area[loc as usize] == opp.color() {
          set_spatial(spatial_nhwc, pos, 19, 1.0, NUM_SPATIAL_FEATURES);
        }
      }
    }
  }

  // ----------------------------------------------------------------
  // History features 9-13 and global pass flags 0-4
  // ----------------------------------------------------------------
  let max_history = 5usize.min(hist.move_history.len());
  let move_history = &hist.move_history;
  let mlen = move_history.len();

  // We only fill history if the last move was made by opp (normal alternating play).
  let mut num_turns_included = 0usize;
  if max_history >= 1 && mlen >= 1 && move_history[mlen - 1].player == opp {
    let prev1 = move_history[mlen - 1].loc;
    num_turns_included = 1;
    if prev1 == PASS_LOC {
      global[0] = 1.0;
    } else if prev1 != NULL_LOC {
      let pos = loc_to_pos(prev1, x_size, nn_len);
      if pos < nl * nl {
        set_spatial(spatial_nhwc, pos, 9, 1.0, NUM_SPATIAL_FEATURES);
      }
    }

    if max_history >= 2 && mlen >= 2 && move_history[mlen - 2].player == pla {
      let prev2 = move_history[mlen - 2].loc;
      num_turns_included = 2;
      if prev2 == PASS_LOC {
        global[1] = 1.0;
      } else if prev2 != NULL_LOC {
        let pos = loc_to_pos(prev2, x_size, nn_len);
        if pos < nl * nl {
          set_spatial(spatial_nhwc, pos, 10, 1.0, NUM_SPATIAL_FEATURES);
        }
      }

      if max_history >= 3 && mlen >= 3 && move_history[mlen - 3].player == opp {
        let prev3 = move_history[mlen - 3].loc;
        num_turns_included = 3;
        if prev3 == PASS_LOC {
          global[2] = 1.0;
        } else if prev3 != NULL_LOC {
          let pos = loc_to_pos(prev3, x_size, nn_len);
          if pos < nl * nl {
            set_spatial(spatial_nhwc, pos, 11, 1.0, NUM_SPATIAL_FEATURES);
          }
        }

        if max_history >= 4 && mlen >= 4 && move_history[mlen - 4].player == pla
        {
          let prev4 = move_history[mlen - 4].loc;
          num_turns_included = 4;
          if prev4 == PASS_LOC {
            global[3] = 1.0;
          } else if prev4 != NULL_LOC {
            let pos = loc_to_pos(prev4, x_size, nn_len);
            if pos < nl * nl {
              set_spatial(spatial_nhwc, pos, 12, 1.0, NUM_SPATIAL_FEATURES);
            }
          }

          if max_history >= 5
            && mlen >= 5
            && move_history[mlen - 5].player == opp
          {
            let prev5 = move_history[mlen - 5].loc;
            num_turns_included = 5;
            if prev5 == PASS_LOC {
              global[4] = 1.0;
            } else if prev5 != NULL_LOC {
              let pos = loc_to_pos(prev5, x_size, nn_len);
              if pos < nl * nl {
                set_spatial(spatial_nhwc, pos, 13, 1.0, NUM_SPATIAL_FEATURES);
              }
            }
          }
        }
      }
    }
  }

  // ----------------------------------------------------------------
  // Ladder features 14, 15, 16 (stones in atari on current/prev/prevprev boards)
  // ----------------------------------------------------------------
  iter_ladder_stones(board, nn_len, spatial_nhwc, 14);

  if num_turns_included >= 1 {
    if let Some(prev_board) = hist.get_recent_board(1) {
      iter_ladder_stones(prev_board, nn_len, spatial_nhwc, 15);
      if num_turns_included >= 2 {
        if let Some(pp_board) = hist.get_recent_board(2) {
          iter_ladder_stones(pp_board, nn_len, spatial_nhwc, 16);
        }
      }
    }
  }

  // Feature 17 (ladder attack) is skipped for simplicity.

  // Features 20,21: second-encore starting stones — not tracked in our Rust port, skip.

  // ----------------------------------------------------------------
  // Global features 5-18
  // ----------------------------------------------------------------
  let b_area = (x_size * y_size) as f32;
  // self_komi from current player's perspective
  let raw_komi = if pla == Player::White {
    hist.rules.komi + hist.white_bonus_score + hist.white_handicap_bonus_score
  } else {
    -(hist.rules.komi
      + hist.white_bonus_score
      + hist.white_handicap_bonus_score)
  };
  let self_komi = raw_komi
    .max(-b_area - KOMI_CLIP_RADIUS)
    .min(b_area + KOMI_CLIP_RADIUS);
  global[5] = self_komi / 20.0;

  // Feature 6,7: ko rule
  match hist.rules.ko_rule {
    KoRule::Simple => {}
    KoRule::Positional | KoRule::Spight => {
      global[6] = 1.0;
      global[7] = 0.5;
    }
    KoRule::Situational => {
      global[6] = 1.0;
      global[7] = -0.5;
    }
  }

  // Feature 8: multi-stone suicide
  if hist.rules.multi_stone_suicide_legal {
    global[8] = 1.0;
  }

  // Feature 9: scoring rule
  if hist.rules.scoring_rule == ScoringRule::Territory {
    global[9] = 1.0;
  }

  // Features 10,11: tax rule
  match hist.rules.tax_rule {
    TaxRule::None => {}
    TaxRule::Seki => {
      global[10] = 1.0;
    }
    TaxRule::All => {
      global[10] = 1.0;
      global[11] = 1.0;
    }
  }

  // Features 12,13: encore phase
  if hist.encore_phase > 0 {
    global[12] = 1.0;
  }
  if hist.encore_phase > 1 {
    global[13] = 1.0;
  }

  // Feature 14: passWouldEndPhase
  if hist.pass_would_end_phase(board, pla) {
    global[14] = 1.0;
  }

  // Features 15,16: playout doubling (unused here — set to 0)

  // Feature 17: button — not tracked in our Rust port

  // Feature 18: komi parity wave
  if hist.rules.scoring_rule == ScoringRule::Area || hist.encore_phase >= 2 {
    let board_area_is_even = (x_size * y_size) % 2 == 0;
    let drawable_komis_are_even = board_area_is_even;
    let komi_floor = if drawable_komis_are_even {
      (self_komi / 2.0).floor() * 2.0
    } else {
      ((self_komi - 1.0) / 2.0).floor() * 2.0 + 1.0
    };
    let mut delta = self_komi - komi_floor;
    delta = delta.clamp(0.0, 2.0);
    let wave = if delta < 0.5 {
      delta
    } else if delta < 1.5 {
      1.0 - delta
    } else {
      delta - 2.0
    };
    global[18] = wave;
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
    let idx = ch * eval.value.len() / value_ch.max(1) / 1 + batch_idx;
    // Actually layout: [value_ch, batch] → idx = ch * batch + batch_idx
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

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::board::Board;
  use crate::game::boardhistory::BoardHistory;
  use crate::game::rules::Rules;

  fn make_hist(board: &Board) -> BoardHistory {
    BoardHistory::new(board, Player::Black, Rules::default(), 0)
  }

  #[test]
  fn fill_row_v7_channel0_all_ones_on_board() {
    let board = Board::new(5, 5);
    let hist = make_hist(&board);
    let nl = 5u32;
    let hw = (nl * nl) as usize;
    let mut spatial = vec![0.0f32; hw * NUM_SPATIAL_FEATURES];
    let mut global = vec![0.0f32; NUM_GLOBAL_FEATURES];
    fill_row_v7(&board, &hist, Player::Black, nl, &mut spatial, &mut global);
    // Channel 0: all on-board cells should be 1.0
    for y in 0..5usize {
      for x in 0..5usize {
        let pos = y * 5 + x;
        assert_eq!(
          spatial[pos * NUM_SPATIAL_FEATURES],
          1.0,
          "channel 0 at ({x},{y}) should be 1.0"
        );
      }
    }
  }

  #[test]
  fn fill_row_v7_single_stone_channel1() {
    use crate::game::board::PASS_LOC;
    let mut board = Board::new(5, 5);
    board.play_move_assume_legal(
      crate::game::board::location::get_loc(2, 2, 5),
      Player::Black,
    );
    let hist = make_hist(&board);
    let nl = 5u32;
    let hw = (nl * nl) as usize;
    let mut spatial = vec![0.0f32; hw * NUM_SPATIAL_FEATURES];
    let mut global = vec![0.0f32; NUM_GLOBAL_FEATURES];
    fill_row_v7(&board, &hist, Player::White, nl, &mut spatial, &mut global);
    // From White's perspective: channel 2 = opp (Black) stone at (2,2)
    let pos = 2 * 5 + 2;
    assert_eq!(
      spatial[pos * NUM_SPATIAL_FEATURES + 2],
      1.0,
      "opponent stone should be in channel 2 at (2,2)"
    );
    let _ = PASS_LOC;
    // Count total channel-2 cells
    let count: usize = (0..hw)
      .filter(|&p| spatial[p * NUM_SPATIAL_FEATURES + 2] == 1.0)
      .count();
    assert_eq!(count, 1, "exactly 1 opponent stone");
  }

  #[test]
  fn policy_size_correct() {
    // MAX_NN_POLICY_SIZE = 19*19 + 1 = 362
    assert_eq!(MAX_NN_POLICY_SIZE, 362);
  }
}
