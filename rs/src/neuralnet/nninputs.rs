/// Neural network input feature encoding.
///
/// Translates `cpp/neuralnet/nninputs.cpp` (functions `fillRowV3`–`fillRowV7`)
/// into Rust.  All output buffers use **NHWC** flat layout:
///
/// ```text
///   spatial[pos * num_spatial_ch + feature]   where pos = y * nn_x + x
///   global[feature]
/// ```
///
/// Complex features (ladder read-out, territory calculation from scratch) are
/// left as zero for now; the basic stone/liberties/ko/history/komi/rules
/// features are faithfully encoded, which is sufficient for inference with a
/// real model or for unit testing with synthetic weights.
use crate::game::{
  board::{Board, Color, Loc, Player, NULL_LOC, PASS_LOC, location},
  boardhistory::BoardHistory,
  rules::{KoRule, ScoringRule, TaxRule},
};

// ---------------------------------------------------------------------------
// Public constants (spatial channel counts are all 22 for input versions 3–7
// except V5 which uses 13).
// ---------------------------------------------------------------------------

pub const NUM_SPATIAL_V5: usize = 13;
pub const NUM_SPATIAL: usize = 22; // V3/V4/V6/V7

pub const NUM_GLOBAL_V3_V4: usize = 14;
pub const NUM_GLOBAL_V5: usize = 12;
pub const NUM_GLOBAL_V6: usize = 16;
pub const NUM_GLOBAL_V7: usize = 19;

// ---------------------------------------------------------------------------
// Position helpers
// ---------------------------------------------------------------------------

/// Board `(x, y)` → NN grid position index.  Pass → `nn_x * nn_y`.
#[inline]
pub fn xy_to_pos(x: usize, y: usize, nn_x: usize) -> usize {
  y * nn_x + x
}

/// Board `Loc` → NN grid position index.
/// Returns `nn_x * nn_y` for `PASS_LOC` and `nn_x * (nn_y + 1)` for `NULL_LOC`.
#[inline]
pub fn loc_to_pos(loc: Loc, board_x: usize, nn_x: usize, nn_y: usize) -> usize {
  if loc == PASS_LOC {
    return nn_x * nn_y;
  }
  if loc == NULL_LOC {
    return nn_x * (nn_y + 1);
  }
  let x = location::get_x(loc, board_x);
  let y = location::get_y(loc, board_x);
  y * nn_x + x
}

/// NN grid position index → board `Loc`.
/// Returns `PASS_LOC` for `pos == nn_x * nn_y`, `NULL_LOC` for out-of-board.
#[inline]
pub fn pos_to_loc(
  pos: usize,
  board_x: usize,
  board_y: usize,
  nn_x: usize,
  nn_y: usize,
) -> Loc {
  if pos == nn_x * nn_y {
    return PASS_LOC;
  }
  let x = pos % nn_x;
  let y = pos / nn_x;
  if x >= board_x || y >= board_y {
    return NULL_LOC;
  }
  location::get_loc(x, y, board_x)
}

// ---------------------------------------------------------------------------
// Main encoding entry point
// ---------------------------------------------------------------------------

/// Encode `board` / `hist` / `next_player` into flat NHWC input buffers.
///
/// # Parameters
/// * `nn_x`, `nn_y` — NN grid dimensions (≥ board dimensions).
/// * `num_spatial`  — number of spatial input channels expected by the model.
/// * `num_global`   — number of global input channels expected by the model.
/// * `out_spatial`  — filled in NHWC order: index `pos * num_spatial + f`.
/// * `out_global`   — filled as a flat slice of length `num_global`.
///
/// The function dispatches to the correct feature version based on
/// `(num_spatial, num_global)`:
/// * 13 spatial / 12 global → V5
/// * 22 spatial / 14 global → V3 / V4
/// * 22 spatial / 16 global → V6
/// * 22 spatial / 19 global → V7 (same as V6 + 3 reserved zeros)
/// * other                  → best-effort V6 encoding
pub fn fill_row(
  board: &Board,
  hist: &BoardHistory,
  next_player: Player,
  nn_x: usize,
  nn_y: usize,
  num_spatial: usize,
  num_global: usize,
  out_spatial: &mut [f32],
  out_global: &mut [f32],
) {
  debug_assert_eq!(out_spatial.len(), nn_y * nn_x * num_spatial);
  debug_assert_eq!(out_global.len(), num_global);

  for v in out_spatial.iter_mut() {
    *v = 0.0;
  }
  for v in out_global.iter_mut() {
    *v = 0.0;
  }

  let pla = next_player;
  let opp = pla.opponent();
  let nc = num_spatial;

  // ------------------------------------------------------------------
  // Spatial features 0-8: board state + ko

  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = location::get_loc(x, y, board.x_size);
      let base = (y * nn_x + x) * nc;

      // Feature 0: on board
      out_spatial[base] = 1.0;

      let stone = board.colors[loc as usize];
      if stone == pla.color() {
        out_spatial[base + 1] = 1.0;
        if nc > 3 {
          encode_liberties(board, loc, base, nc, out_spatial);
        }
      } else if stone == opp.color() {
        out_spatial[base + 2] = 1.0;
        if nc > 3 {
          encode_liberties(board, loc, base, nc, out_spatial);
        }
      }
    }
  }

  // Feature 6: ko bans (normal phase)
  if nc > 6 && hist.encore_phase == 0 {
    // Simple ko loc
    if board.ko_loc != NULL_LOC {
      let x = location::get_x(board.ko_loc, board.x_size);
      let y = location::get_y(board.ko_loc, board.x_size);
      if x < nn_x && y < nn_y {
        out_spatial[(y * nn_x + x) * nc + 6] = 1.0;
      }
    }
    // Superko bans
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = location::get_loc(x, y, board.x_size);
        if hist.super_ko_banned[loc as usize] && loc != board.ko_loc {
          out_spatial[(y * nn_x + x) * nc + 6] = 1.0;
        }
      }
    }
  }
  // Features 7-8 (encore ko recap / phase): left as zero for non-encore play

  // ------------------------------------------------------------------
  // Spatial features for previous moves + global pass flags 0-4.
  //
  // V5 puts previous moves at features 6-10; all others at 9-13.

  let prev_feat = if nc == NUM_SPATIAL_V5 { 6usize } else { 9usize };
  let hide_history = hist.is_game_finished;

  if !hide_history {
    encode_prev_moves(
      board, hist, pla, opp, nn_x, nn_y, nc, prev_feat,
      out_spatial, out_global,
    );
  }

  // Features 14-17: ladder features → left as 0 (complex read-out omitted)
  // Features 18-19: territory features → left as 0 (heavy computation omitted)
  // Features 20-21: second encore start colors → 0 in normal play

  // ------------------------------------------------------------------
  // Global features

  let board_area = (board.x_size * board.y_size) as f32;
  let self_komi = {
    let raw = hist.rules.komi
      + hist.white_handicap_bonus_score
      + hist.white_bonus_score;
    if pla == Player::White { raw } else { -raw }
  };
  let self_komi = self_komi
    .max(-(board_area + 1.0))
    .min(board_area + 1.0);

  // Global[5]: komi (scale by /15 for V3/V4, /20 for V5/V6/V7)
  if num_global > 5 {
    let div = if num_global == NUM_GLOBAL_V3_V4 { 15.0f32 } else { 20.0f32 };
    out_global[5] = self_komi / div;
  }

  // Global[6,7]: ko rule encoding
  if num_global > 7 {
    match hist.rules.ko_rule {
      KoRule::Simple => {}
      KoRule::Positional | KoRule::Spight => {
        out_global[6] = 1.0;
        out_global[7] = 0.5;
      }
      KoRule::Situational => {
        out_global[6] = 1.0;
        out_global[7] = -0.5;
      }
    }
  }

  // Global[8]: multi-stone suicide legal
  if num_global > 8 && hist.rules.multi_stone_suicide_legal {
    out_global[8] = 1.0;
  }

  // Global[9]: territory scoring
  if num_global > 9 && hist.rules.scoring_rule == ScoringRule::Territory {
    out_global[9] = 1.0;
  }

  // Version-specific tail features
  match num_global {
    14 => {
      // V3 / V4: encore=10,11; pass-end=12; komi-parity=13
      if hist.encore_phase > 0 { out_global[10] = 1.0; }
      if hist.encore_phase > 1 { out_global[11] = 1.0; }
      let pass_end = !hide_history && hist.pass_would_end_game(board, pla);
      out_global[12] = if pass_end { 1.0 } else { 0.0 };
      set_komi_parity(self_komi, &hist.rules, hist.encore_phase, board, 13, out_global);
    }
    12 => {
      // V5: encore=10,11 only
      if hist.encore_phase > 0 { out_global[10] = 1.0; }
      if hist.encore_phase > 1 { out_global[11] = 1.0; }
    }
    16 | 19 => {
      // V6 (16) and V7 (19, extras at 16-18 left as zero)
      //   10,11: tax rule
      //   12,13: encore phase
      //   14: pass would end phase
      //   15: komi parity
      match hist.rules.tax_rule {
        TaxRule::None => {}
        TaxRule::Seki => { out_global[10] = 1.0; }
        TaxRule::All => { out_global[10] = 1.0; out_global[11] = 1.0; }
      }
      if hist.encore_phase > 0 { out_global[12] = 1.0; }
      if hist.encore_phase > 1 { out_global[13] = 1.0; }
      let pass_end = !hide_history && hist.pass_would_end_game(board, pla);
      out_global[14] = if pass_end { 1.0 } else { 0.0 };
      set_komi_parity(self_komi, &hist.rules, hist.encore_phase, board, 15, out_global);
      // V7 extras [16,17,18] remain 0
    }
    _ => {
      // Unknown version: best effort — V6-style if enough room
      if num_global >= 16 {
        match hist.rules.tax_rule {
          TaxRule::None => {}
          TaxRule::Seki => { if num_global > 10 { out_global[10] = 1.0; } }
          TaxRule::All => {
            if num_global > 10 { out_global[10] = 1.0; }
            if num_global > 11 { out_global[11] = 1.0; }
          }
        }
        if num_global > 12 && hist.encore_phase > 0 { out_global[12] = 1.0; }
        if num_global > 13 && hist.encore_phase > 1 { out_global[13] = 1.0; }
      } else if num_global >= 12 {
        if num_global > 10 && hist.encore_phase > 0 { out_global[10] = 1.0; }
        if num_global > 11 && hist.encore_phase > 1 { out_global[11] = 1.0; }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn encode_liberties(
  board: &Board,
  loc: Loc,
  base: usize,
  nc: usize,
  spatial: &mut [f32],
) {
  let libs = board.get_num_liberties(loc);
  let f = match libs {
    1 => 3,
    2 => 4,
    3 => 5,
    _ => return,
  };
  if f < nc {
    spatial[base + f] = 1.0;
  }
}

fn encode_prev_moves(
  board: &Board,
  hist: &BoardHistory,
  pla: Player,
  opp: Player,
  nn_x: usize,
  _nn_y: usize,
  nc: usize,
  prev_feat: usize,
  spatial: &mut [f32],
  global: &mut [f32],
) {
  // C++ interleaves: prev1=opp, prev2=pla, prev3=opp, prev4=pla, prev5=opp
  // Global 0-4 flag passes; spatial marks board position.
  let moves = &hist.move_history;
  let n = moves.len();

  let pairs: &[(usize, Player, usize)] = &[
    (1, opp, 0),  // prev1: opp, global flag 0
    (2, pla, 1),  // prev2: pla, global flag 1
    (3, opp, 2),  // prev3: opp, global flag 2
    (4, pla, 3),  // prev4: pla, global flag 3
    (5, opp, 4),  // prev5: opp, global flag 4
  ];

  for &(dist, expected_pla, gidx) in pairs {
    if n < dist { break; }
    let m = &moves[n - dist];
    if m.player != expected_pla { break; }
    let feat_idx = prev_feat + (dist - 1);
    if feat_idx >= nc { break; }
    if m.loc == PASS_LOC {
      if gidx < global.len() {
        global[gidx] = 1.0;
      }
    } else if m.loc != NULL_LOC {
      let x = location::get_x(m.loc, board.x_size);
      let y = location::get_y(m.loc, board.x_size);
      if x < nn_x {
        spatial[(y * nn_x + x) * nc + feat_idx] = 1.0;
      }
    }
  }
}

fn set_komi_parity(
  self_komi: f32,
  rules: &crate::game::rules::Rules,
  encore_phase: i32,
  board: &Board,
  idx: usize,
  global: &mut [f32],
) {
  if global.len() <= idx {
    return;
  }
  if rules.scoring_rule != ScoringRule::Area && encore_phase < 2 {
    return;
  }

  // Triangular wave in komi with period = 2, peaks at drawable-komi + 0.5
  let board_area_is_even = (board.x_size * board.y_size) % 2 == 0;
  let drawable_komis_are_even = board_area_is_even;

  let komi_floor = if drawable_komis_are_even {
    (self_komi / 2.0).floor() * 2.0
  } else {
    ((self_komi - 1.0) / 2.0).floor() * 2.0 + 1.0
  };

  let delta = (self_komi - komi_floor).clamp(0.0, 2.0);
  let wave = if delta < 0.5 {
    delta
  } else if delta < 1.5 {
    1.0 - delta
  } else {
    delta - 2.0
  };

  global[idx] = wave;
}
