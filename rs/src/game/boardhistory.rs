/// Game history and superko enforcement — mirrors `cpp/game/boardhistory.h/cpp`.
///
/// `BoardHistory` tracks the sequence of moves played, enforces ko/superko rules,
/// and handles game-end detection and scoring.
use crate::game::{
  Hash128,
  board::{
    Board, Color, KO_MARK_HASH, Loc, MAX_ARR_SIZE, Move, NULL_LOC, PASS_LOC,
    Player, ZOBRIST_PLAYER_HASH, location,
  },
};

// ---------------------------------------------------------------------------
// EncoreKoCapture
// ---------------------------------------------------------------------------

/// Records a ko-capture during the encore phase so it can be forbidden from
/// immediate recapture.
#[derive(Clone, Debug)]
pub struct EncoreKoCapture {
  pub pos_hash_before_move: Hash128,
  pub move_loc: Loc,
  pub move_pla: Player,
}

// ---------------------------------------------------------------------------
// KoHashTable
// ---------------------------------------------------------------------------

pub struct KoHashTable {
  idx_table: [u32; Self::TABLE_SIZE],
  sorted: Vec<Hash128>,
  pub first_turn_idx: usize,
}

impl KoHashTable {
  pub const TABLE_SIZE: usize = 1024;
  const TABLE_MASK: u64 = (Self::TABLE_SIZE - 1) as u64;

  pub fn new() -> Self {
    Self {
      idx_table: [0; Self::TABLE_SIZE],
      sorted: Vec::new(),
      first_turn_idx: 0,
    }
  }

  /// Rebuild the lookup table from the history's ko-hash sequence.
  pub fn recompute(&mut self, history: &BoardHistory) {
    self.sorted = history.ko_hash_history.clone();
    self.first_turn_idx = history.first_turn_idx_with_ko_history;

    self.sorted.sort_by(|a, b| {
      let a_bits = a.hash0 & Self::TABLE_MASK;
      let b_bits = b.hash0 & Self::TABLE_MASK;
      a_bits.cmp(&b_bits).then(a.cmp(b))
    });

    let size = self.sorted.len() as u32;
    let mut idx = 0u32;
    for bits in 0..Self::TABLE_SIZE {
      while idx < size
        && (self.sorted[idx as usize].hash0 & Self::TABLE_MASK) < bits as u64
      {
        idx += 1;
      }
      self.idx_table[bits] = idx;
    }
  }

  pub fn contains_hash(&self, hash: Hash128) -> bool {
    self.occurrences_of_hash(hash) > 0
  }

  pub fn occurrences_of_hash(&self, hash: Hash128) -> usize {
    let bits = (hash.hash0 & Self::TABLE_MASK) as usize;
    let mut idx = self.idx_table[bits] as usize;
    let size = self.sorted.len();
    let mut count = 0;
    while idx < size
      && (self.sorted[idx].hash0 & Self::TABLE_MASK) as usize == bits
    {
      if self.sorted[idx] == hash {
        count += 1;
      }
      idx += 1;
    }
    count
  }
}

impl Default for KoHashTable {
  fn default() -> Self {
    Self::new()
  }
}

// ---------------------------------------------------------------------------
// BoardHistory
// ---------------------------------------------------------------------------

/// Complete game state including history of moves, ko bans, and scoring.
/// Rules are fixed to Tromp-Taylor: positional superko, area scoring, suicide legal.
#[derive(Clone)]
pub struct BoardHistory {
  pub komi: f32,
  pub move_history: Vec<Move>,
  pub ko_hash_history: Vec<Hash128>,
  pub first_turn_idx_with_ko_history: usize,
  pub super_ko_banned: [bool; MAX_ARR_SIZE],
  pub was_ever_occupied_or_played: [bool; MAX_ARR_SIZE],
  pub consecutive_ending_passes: i32,
  pub encore_phase: i32,
  pub ko_recap_blocked: [bool; MAX_ARR_SIZE],
  pub ko_captures_in_encore: Vec<EncoreKoCapture>,
}

impl BoardHistory {
  // -----------------------------------------------------------------------
  // Construction
  // -----------------------------------------------------------------------

  pub fn new(board: &Board, komi: f32) -> Self {
    let mut h = BoardHistory {
      komi,
      move_history: Vec::new(),
      ko_hash_history: Vec::new(),
      first_turn_idx_with_ko_history: 0,
      super_ko_banned: [false; MAX_ARR_SIZE],
      was_ever_occupied_or_played: [false; MAX_ARR_SIZE],
      consecutive_ending_passes: 0,
      encore_phase: 0,
      ko_recap_blocked: [false; MAX_ARR_SIZE],
      ko_captures_in_encore: Vec::new(),
    };
    h.clear(board);
    h
  }

  /// Reset history to the initial state for the given board position.
  pub fn clear(&mut self, board: &Board) {
    self.move_history.clear();
    self.ko_hash_history.clear();
    self.first_turn_idx_with_ko_history = 0;
    self.super_ko_banned = [false; MAX_ARR_SIZE];
    self.was_ever_occupied_or_played = [false; MAX_ARR_SIZE];
    self.consecutive_ending_passes = 0;
    self.encore_phase = 0;
    self.ko_recap_blocked = [false; MAX_ARR_SIZE];
    self.ko_captures_in_encore.clear();

    // Mark existing stones as ever-occupied.
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = location::get_loc(x, y, board.x_size);
        if board.colors[loc as usize].is_stone() {
          self.was_ever_occupied_or_played[loc as usize] = true;
        }
      }
    }

    // Push the initial ko hash.
    let ko_hash = Self::get_ko_hash(board, 0, Hash128::ZERO);
    self.ko_hash_history.push(ko_hash);
  }

  // -----------------------------------------------------------------------
  // Ko hash computation
  // -----------------------------------------------------------------------

  // Tromp-Taylor: positional superko — ko hash is just board position hash.
  fn get_ko_hash(
    board: &Board,
    encore_phase: i32,
    ko_recap_block_hash: Hash128,
  ) -> Hash128 {
    if encore_phase > 0 {
      let player_hash = ZOBRIST_PLAYER_HASH[0]; // unused in practice
      board.pos_hash ^ player_hash ^ ko_recap_block_hash
    } else {
      board.pos_hash
    }
  }

  fn get_ko_hash_after_move_non_encore(pos_hash_after: Hash128) -> Hash128 {
    // Positional superko: no player-hash mixing.
    pos_hash_after
  }

  // -----------------------------------------------------------------------
  // Superko check helpers
  // -----------------------------------------------------------------------

  fn ko_hash_occurs_in_history(
    &self,
    hash: Hash128,
    table: &KoHashTable,
  ) -> bool {
    // Fast path via KoHashTable.
    if table.contains_hash(hash) {
      return true;
    }
    // Fall back to linear scan (when table is not yet built).
    let start = self.first_turn_idx_with_ko_history;
    for i in start..self.ko_hash_history.len() {
      if self.ko_hash_history[i] == hash {
        return true;
      }
    }
    false
  }

  // -----------------------------------------------------------------------
  // Move legality
  // -----------------------------------------------------------------------

  /// Returns true if `loc` is legal for `pla` under the current history rules.
  pub fn is_legal(&self, board: &Board, loc: Loc, pla: Player) -> bool {
    if loc == PASS_LOC {
      return true;
    }
    // Tromp-Taylor: multi-stone suicide is legal.
    if !board.is_legal(loc, pla, true) {
      return false;
    }
    if self.super_ko_banned[loc as usize] {
      return false;
    }
    true
  }

  // -----------------------------------------------------------------------
  // Phase transitions
  // -----------------------------------------------------------------------

  fn phase_has_spightlike_ending(&self) -> bool {
    // Tromp-Taylor: positional superko — no Spight/Simple-like clearing.
    self.encore_phase > 0
  }

  pub fn pass_would_end_phase(&self, _board: &Board, _pla: Player) -> bool {
    // Simplified: two consecutive passes end the game in non-encore.
    if self.encore_phase == 0 && self.consecutive_ending_passes >= 1 {
      return true;
    }
    false
  }

  pub fn pass_would_end_game(&self, board: &Board, pla: Player) -> bool {
    self.pass_would_end_phase(board, pla)
  }

  // -----------------------------------------------------------------------
  // Playing a move
  // -----------------------------------------------------------------------

  /// Apply `loc` for `pla` to `board`, updating all history state.
  ///
  /// Returns `true` if the move was played (always true; panics on illegal move
  /// when compiled with debug assertions).
  pub fn make_board_move(
    &mut self,
    board: &mut Board,
    loc: Loc,
    pla: Player,
    root_ko_hash_table: Option<&KoHashTable>,
  ) -> bool {
    debug_assert!(self.is_legal(board, loc, pla), "illegal move");

    // Track ever-played.
    if loc != PASS_LOC {
      self.was_ever_occupied_or_played[loc as usize] = true;
    }

    // Play on the board.
    let pos_hash_before = board.pos_hash;
    let _rec = board.play_move_assume_legal(loc, pla);

    // Record the move.
    self.move_history.push(Move { loc, player: pla });

    // Ko recap blocked management (encore).
    let ko_recap_block_hash = if self.encore_phase > 0 {
      // Encode the blocked set as a hash (simplified: single Hash128 XOR of blocked locs).
      // In full C++, this is maintained incrementally via ZOBRIST_KO_MARK_HASH.
      // For now use a running XOR approach.
      self.compute_ko_recap_block_hash()
    } else {
      Hash128::ZERO
    };

    // Compute and record the new ko hash.
    let opp = pla.opponent();
    let new_ko_hash = if self.encore_phase == 0 {
      Self::get_ko_hash_after_move_non_encore(board.pos_hash)
    } else {
      Self::get_ko_hash(board, self.encore_phase, ko_recap_block_hash)
    };

    // Clear ko history on Spight/Simple/encore passes.
    if self.phase_has_spightlike_ending() && loc == PASS_LOC {
      self.ko_hash_history.clear();
      self.first_turn_idx_with_ko_history = self.move_history.len();
    }
    self.ko_hash_history.push(new_ko_hash);

    // Consecutive ending passes.
    if loc == PASS_LOC {
      self.consecutive_ending_passes += 1;
    } else {
      self.consecutive_ending_passes = 0;
    }

    // Update superko bans.
    self.update_super_ko_banned(board, opp, root_ko_hash_table);

    // Encore ko capture tracking.
    if self.encore_phase > 0 {
      if board.ko_loc != NULL_LOC {
        self.ko_captures_in_encore.push(EncoreKoCapture {
          pos_hash_before_move: pos_hash_before,
          move_loc: loc,
          move_pla: pla,
        });
      }
    }

    true
  }

  fn compute_ko_recap_block_hash(&self) -> Hash128 {
    let mut h = Hash128::ZERO;
    for (i, &blocked) in self.ko_recap_blocked.iter().enumerate() {
      if blocked {
        h ^= KO_MARK_HASH[i][1]; // color index 1 = black (arbitrary non-zero)
      }
    }
    h
  }

  /// Recompute `super_ko_banned` for `next_player` after a move has been made.
  fn update_super_ko_banned(
    &mut self,
    board: &Board,
    next_player: Player,
    root_table: Option<&KoHashTable>,
  ) {
    self.super_ko_banned = [false; MAX_ARR_SIZE];

    if self.encore_phase == 0 {
      // Tromp-Taylor positional superko.
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc = location::get_loc(x, y, board.x_size);
          let idx = loc as usize;
          if board.colors[idx] != Color::Empty || board.ko_loc == loc {
            continue;
          }
          if !self.was_ever_occupied_or_played[idx]
            && !board.would_be_ko_capture(loc, next_player)
          {
            continue;
          }
          let pos_hash_after = board.get_pos_hash_after_move(loc, next_player);
          let ko_hash_after =
            Self::get_ko_hash_after_move_non_encore(pos_hash_after);
          if let Some(table) = root_table {
            if self.ko_hash_occurs_in_history(ko_hash_after, table) {
              self.super_ko_banned[idx] = true;
            }
          } else {
            let start = self.first_turn_idx_with_ko_history;
            if self.ko_hash_history[start..].contains(&ko_hash_after) {
              self.super_ko_banned[idx] = true;
            }
          }
        }
      }
    } else {
      // Encore: forbidden only if this exact (pos_hash, loc, pla) was seen before.
      for ekc in &self.ko_captures_in_encore {
        if ekc.pos_hash_before_move == board.pos_hash
          && ekc.move_pla == next_player
        {
          self.super_ko_banned[ekc.move_loc as usize] = true;
        }
      }
    }
    // Simple ko: only board.ko_loc is banned — already enforced by board.is_legal.
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;
  use crate::game::board::{Board, Player, location};

  fn make_game() -> (Board, BoardHistory) {
    let b = Board::new(9, 9);
    let h = BoardHistory::new(&b, 7.5);
    (b, h)
  }

  #[test]
  fn playing_a_stone_resets_passes() {
    let (mut b, mut h) = make_game();
    h.make_board_move(&mut b, PASS_LOC, Player::Black, None);
    h.make_board_move(&mut b, location::get_loc(3, 3, 9), Player::White, None);
    assert_eq!(h.consecutive_ending_passes, 0);
  }

  #[test]
  fn legal_move_check() {
    let (mut b, h) = make_game();
    // Pass is always legal.
    assert!(h.is_legal(&b, PASS_LOC, Player::Black));
    // Playing on an empty interior cell is legal.
    assert!(h.is_legal(&b, location::get_loc(4, 4, 9), Player::Black));
    // Playing on a stone is illegal.
    b.play_move_assume_legal(location::get_loc(3, 3, 9), Player::Black);
    assert!(!h.is_legal(&b, location::get_loc(3, 3, 9), Player::White));
  }

  #[test]
  fn ko_hash_history_grows_with_moves() {
    let (mut b, mut h) = make_game();
    let initial_len = h.ko_hash_history.len();
    h.make_board_move(&mut b, location::get_loc(4, 4, 9), Player::Black, None);
    assert_eq!(h.ko_hash_history.len(), initial_len + 1);
  }
}
