/// Game history and superko enforcement — mirrors `cpp/game/boardhistory.h/cpp`.
///
/// `BoardHistory` tracks the sequence of moves played, enforces ko/superko rules,
/// and handles game-end detection and scoring.
use crate::game::{
  Hash128,
  board::{Board, Color, Loc, Move, Player, NULL_LOC, PASS_LOC, MAX_ARR_SIZE, location,
          ZOBRIST_PLAYER_HASH, KO_MARK_HASH},
  rules::{KoRule, Rules, ScoringRule},
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const NUM_RECENT_BOARDS: usize = 6;

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
    while idx < size && (self.sorted[idx].hash0 & Self::TABLE_MASK) as usize == bits {
      if self.sorted[idx] == hash {
        count += 1;
      }
      idx += 1;
    }
    count
  }
}

impl Default for KoHashTable {
  fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// BoardHistory
// ---------------------------------------------------------------------------

/// Complete game state including history of moves, ko bans, and scoring.
#[derive(Clone)]
pub struct BoardHistory {
  pub rules: Rules,
  pub move_history: Vec<Move>,
  pub ko_hash_history: Vec<Hash128>,
  pub first_turn_idx_with_ko_history: usize,
  /// Ring buffer of the last `NUM_RECENT_BOARDS` board positions.
  pub recent_boards: Vec<Board>, // index 0 = most recent
  pub super_ko_banned: [bool; MAX_ARR_SIZE],
  pub was_ever_occupied_or_played: [bool; MAX_ARR_SIZE],
  pub consecutive_ending_passes: i32,
  pub hashes_before_black_pass: Vec<Hash128>,
  pub hashes_before_white_pass: Vec<Hash128>,
  pub encore_phase: i32,
  pub ko_recap_blocked: [bool; MAX_ARR_SIZE],
  pub ko_captures_in_encore: Vec<EncoreKoCapture>,
  pub is_game_finished: bool,
  pub winner: Option<Player>,
  pub final_white_minus_black_score: f32,
  pub is_scored: bool,
  pub is_no_result: bool,
  pub is_resignation: bool,
  pub white_bonus_score: f32,
  pub white_handicap_bonus_score: f32,
  /// Number of handicap stones assumed at game start.
  pub assume_multistep_starting_black_moves_are_handicap: bool,
  pub override_num_handicap_stones: Option<i32>,
}

impl BoardHistory {
  // -----------------------------------------------------------------------
  // Construction
  // -----------------------------------------------------------------------

  pub fn new(board: &Board, next_player: Player, rules: Rules, handicap_stones: i32) -> Self {
    let mut h = BoardHistory {
      rules,
      move_history: Vec::new(),
      ko_hash_history: Vec::new(),
      first_turn_idx_with_ko_history: 0,
      recent_boards: Vec::new(),
      super_ko_banned: [false; MAX_ARR_SIZE],
      was_ever_occupied_or_played: [false; MAX_ARR_SIZE],
      consecutive_ending_passes: 0,
      hashes_before_black_pass: Vec::new(),
      hashes_before_white_pass: Vec::new(),
      encore_phase: 0,
      ko_recap_blocked: [false; MAX_ARR_SIZE],
      ko_captures_in_encore: Vec::new(),
      is_game_finished: false,
      winner: None,
      final_white_minus_black_score: 0.0,
      is_scored: false,
      is_no_result: false,
      is_resignation: false,
      white_bonus_score: 0.0,
      white_handicap_bonus_score: 0.0,
      assume_multistep_starting_black_moves_are_handicap: false,
      override_num_handicap_stones: None,
    };
    h.clear(board, next_player, handicap_stones);
    h
  }

  /// Reset history to the initial state for the given board position.
  pub fn clear(&mut self, board: &Board, next_player: Player, handicap_stones: i32) {
    self.move_history.clear();
    self.ko_hash_history.clear();
    self.first_turn_idx_with_ko_history = 0;
    self.recent_boards.clear();
    self.recent_boards.push(board.clone());
    self.super_ko_banned = [false; MAX_ARR_SIZE];
    self.was_ever_occupied_or_played = [false; MAX_ARR_SIZE];
    self.consecutive_ending_passes = 0;
    self.hashes_before_black_pass.clear();
    self.hashes_before_white_pass.clear();
    self.encore_phase = 0;
    self.ko_recap_blocked = [false; MAX_ARR_SIZE];
    self.ko_captures_in_encore.clear();
    self.is_game_finished = false;
    self.winner = None;
    self.final_white_minus_black_score = 0.0;
    self.is_scored = false;
    self.is_no_result = false;
    self.is_resignation = false;
    self.white_bonus_score = 0.0;
    self.white_handicap_bonus_score = 0.0;

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
    let ko_hash = Self::get_ko_hash(&self.rules, board, next_player, 0, Hash128::ZERO);
    self.ko_hash_history.push(ko_hash);

    // Handicap bonus.
    self.white_handicap_bonus_score =
      Self::compute_white_handicap_bonus(&self.rules, handicap_stones);
  }

  // -----------------------------------------------------------------------
  // Ko hash computation
  // -----------------------------------------------------------------------

  fn get_ko_hash(
    rules: &Rules,
    board: &Board,
    pla: Player,
    encore_phase: i32,
    ko_recap_block_hash: Hash128,
  ) -> Hash128 {
    let player_hash = ZOBRIST_PLAYER_HASH[pla as usize];
    if rules.ko_rule == KoRule::Situational
      || rules.ko_rule == KoRule::Simple
      || encore_phase > 0
    {
      board.pos_hash ^ player_hash ^ ko_recap_block_hash
    } else {
      board.pos_hash ^ ko_recap_block_hash
    }
  }

  fn get_ko_hash_after_move_non_encore(
    rules: &Rules,
    pos_hash_after: Hash128,
    pla: Player,
  ) -> Hash128 {
    let player_hash = ZOBRIST_PLAYER_HASH[pla as usize];
    if rules.ko_rule == KoRule::Situational || rules.ko_rule == KoRule::Simple {
      pos_hash_after ^ player_hash
    } else {
      pos_hash_after
    }
  }

  // -----------------------------------------------------------------------
  // Superko check helpers
  // -----------------------------------------------------------------------

  fn ko_hash_occurs_in_history(&self, hash: Hash128, table: &KoHashTable) -> bool {
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

  fn number_of_ko_hash_occurrences(&self, hash: Hash128, table: &KoHashTable) -> usize {
    if !table.sorted.is_empty() {
      return table.occurrences_of_hash(hash);
    }
    let start = self.first_turn_idx_with_ko_history;
    self.ko_hash_history[start..].iter().filter(|&&h| h == hash).count()
  }

  // -----------------------------------------------------------------------
  // Move legality
  // -----------------------------------------------------------------------

  /// Returns true if `loc` is legal for `pla` under the current history rules.
  pub fn is_legal(&self, board: &Board, loc: Loc, pla: Player) -> bool {
    if self.is_game_finished {
      return false;
    }
    if loc == PASS_LOC {
      return true;
    }
    if !board.is_legal(loc, pla, self.rules.multi_stone_suicide_legal) {
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
    self.encore_phase > 0
      || self.rules.ko_rule == KoRule::Simple
      || self.rules.ko_rule == KoRule::Spight
  }

  pub fn pass_would_end_phase(&self, _board: &Board, _pla: Player) -> bool {
    if self.is_game_finished { return false; }
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

    let ko_hash_before = *self.ko_hash_history.last().unwrap_or(&Hash128::ZERO);

    // Record pass history for Spight-like endings.
    if self.phase_has_spightlike_ending() {
      if loc == PASS_LOC {
        match pla {
          Player::Black => self.hashes_before_black_pass.push(ko_hash_before),
          Player::White => self.hashes_before_white_pass.push(ko_hash_before),
        }
      }
    }

    // Track ever-played.
    if loc != PASS_LOC {
      self.was_ever_occupied_or_played[loc as usize] = true;
    }

    // Play on the board.
    let pos_hash_before = board.pos_hash;
    let rec = board.play_move_assume_legal(loc, pla);

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
      Self::get_ko_hash_after_move_non_encore(&self.rules, board.pos_hash, opp)
    } else {
      Self::get_ko_hash(&self.rules, board, opp, self.encore_phase, ko_recap_block_hash)
    };

    // Clear ko history on Spight/Simple/encore passes.
    if self.phase_has_spightlike_ending() && loc == PASS_LOC {
      self.ko_hash_history.clear();
      self.first_turn_idx_with_ko_history = self.move_history.len();
    }
    self.ko_hash_history.push(new_ko_hash);

    // Update recent_boards ring buffer.
    if self.recent_boards.len() >= NUM_RECENT_BOARDS {
      self.recent_boards.remove(0);
    }
    self.recent_boards.push(board.clone());

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

    // Check game end.
    self.check_game_end(board, opp);

    let _ = rec;
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

    if self.is_game_finished { return; }

    if self.encore_phase == 0 && self.rules.ko_rule != KoRule::Simple {
      // Positional/situational/Spight superko.
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc = location::get_loc(x, y, board.x_size);
          let idx = loc as usize;
          if board.colors[idx] != Color::Empty
            || board.is_suicide(loc, next_player)
              && !self.rules.multi_stone_suicide_legal
            || board.ko_loc == loc
          {
            // Not empty or already illegal — not ko-banned specifically.
            continue;
          }
          // Only check superko for locations that were ever occupied/played
          // OR could be a ko capture.
          if !self.was_ever_occupied_or_played[idx]
            && !board.would_be_ko_capture(loc, next_player)
          {
            continue;
          }
          let pos_hash_after = board.get_pos_hash_after_move(loc, next_player);
          let ko_hash_after =
            Self::get_ko_hash_after_move_non_encore(&self.rules, pos_hash_after, next_player.opponent());
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
    } else if self.encore_phase > 0 {
      // Encore: forbidden only if this exact (pos_hash, loc, pla) was seen before.
      for ekc in &self.ko_captures_in_encore {
        if ekc.pos_hash_before_move == board.pos_hash && ekc.move_pla == next_player {
          self.super_ko_banned[ekc.move_loc as usize] = true;
        }
      }
    }
    // Simple ko: only board.ko_loc is banned — already enforced by board.is_legal.
  }

  // -----------------------------------------------------------------------
  // Game end detection
  // -----------------------------------------------------------------------

  fn check_game_end(&mut self, board: &Board, _next_player: Player) {
    // Two consecutive passes end the game.
    if self.consecutive_ending_passes >= 2 && self.encore_phase == 0 {
      self.end_and_score_game_now(board);
    }
    // 3-fold repetition under Simple/encore rules → no result.
    if self.encore_phase > 0 || self.rules.ko_rule == KoRule::Simple {
      if let Some(&last_hash) = self.ko_hash_history.last() {
        let start = self.first_turn_idx_with_ko_history;
        let occurrences =
          self.ko_hash_history[start..].iter().filter(|&&h| h == last_hash).count();
        if occurrences >= 3 {
          self.is_no_result = true;
          self.is_game_finished = true;
        }
      }
    }
  }

  pub fn end_and_score_game_now(&mut self, board: &Board) {
    self.is_game_finished = true;
    self.is_scored = true;
    let (white_score, black_score) = self.compute_final_score(board);
    self.final_white_minus_black_score = white_score - black_score;
    self.winner = match self.final_white_minus_black_score.partial_cmp(&0.0) {
      Some(std::cmp::Ordering::Greater) => Some(Player::White),
      Some(std::cmp::Ordering::Less) => Some(Player::Black),
      _ => None,
    };
  }

  pub fn set_winner_by_resignation(&mut self, winner: Player) {
    self.is_game_finished = true;
    self.is_resignation = true;
    self.winner = Some(winner);
  }

  /// End the game if all stones are pass-alive.
  pub fn end_game_if_all_pass_alive(&mut self, board: &Board) {
    let mut area = [Color::Empty; MAX_ARR_SIZE];
    board.calculate_area(true, false, false, &mut area);
    // If no empties are unowned, the game is effectively over.
    let all_owned = (0..board.y_size).all(|y| {
      (0..board.x_size).all(|x| {
        let loc = location::get_loc(x, y, board.x_size) as usize;
        area[loc] != Color::Empty
      })
    });
    if all_owned {
      self.end_and_score_game_now(board);
    }
  }

  // -----------------------------------------------------------------------
  // Scoring
  // -----------------------------------------------------------------------

  fn compute_final_score(&self, board: &Board) -> (f32, f32) {
    let mut area = [Color::Empty; MAX_ARR_SIZE];

    match self.rules.scoring_rule {
      ScoringRule::Area => {
        board.calculate_area(true, false, false, &mut area);
        let mut black_score = 0.0f32;
        let mut white_score = self.rules.komi + self.white_bonus_score + self.white_handicap_bonus_score;
        for y in 0..board.y_size {
          for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size) as usize;
            match area[loc] {
              Color::Black => black_score += 1.0,
              Color::White => white_score += 1.0,
              _ => {}
            }
          }
        }
        (white_score, black_score)
      }
      ScoringRule::Territory => {
        board.calculate_area(true, false, false, &mut area);
        let mut black_score = 0.0f32;
        let mut white_score = self.rules.komi + self.white_bonus_score + self.white_handicap_bonus_score;
        // Territory: empty cells owned + opponent captures.
        for y in 0..board.y_size {
          for x in 0..board.x_size {
            let loc = location::get_loc(x, y, board.x_size) as usize;
            if board.colors[loc] == Color::Empty {
              match area[loc] {
                Color::Black => black_score += 1.0,
                Color::White => white_score += 1.0,
                _ => {}
              }
            }
          }
        }
        black_score += board.num_white_captures as f32;
        white_score += board.num_black_captures as f32;
        (white_score, black_score)
      }
    }
  }

  // -----------------------------------------------------------------------
  // Handicap helpers
  // -----------------------------------------------------------------------

  fn compute_white_handicap_bonus(rules: &Rules, handicap: i32) -> f32 {
    use crate::game::rules::WhbRule;
    match rules.white_handicap_bonus_rule {
      WhbRule::Zero => 0.0,
      WhbRule::N => handicap as f32,
      WhbRule::NMinusOne => (handicap - 1).max(0) as f32,
    }
  }

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------

  pub fn get_recent_board(&self, steps_ago: usize) -> Option<&Board> {
    let n = self.recent_boards.len();
    if steps_ago >= n { return None; }
    Some(&self.recent_boards[n - 1 - steps_ago])
  }

  pub fn current_turn_number(&self) -> usize {
    self.move_history.len()
  }

  /// Compute the combined situation+rules+ko hash used in GraphHash.
  ///
  /// Matches C++ `BoardHistory::getSituationRulesAndKoHash`:
  /// `pos_hash ^ player_hash ^ rules_hash ^ komi_hash`
  pub fn get_situation_rules_and_ko_hash(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    draw_equivalent_wins_for_white: f64,
  ) -> Hash128 {
    let mut h = board.pos_hash;
    h ^= ZOBRIST_PLAYER_HASH[next_player as usize];
    h ^= hist.rules.rules_hash();
    h.hash0 ^= komi_hash(hist.rules.komi, draw_equivalent_wins_for_white);
    h
  }
}

/// Deterministic hash contribution from komi (matches C++ komi hashing).
fn komi_hash(komi: f32, draw_equivalent_wins_for_white: f64) -> u64 {
  // Simple deterministic: bit-mix komi bits and draw equivalent.
  let k = komi.to_bits() as u64;
  let d = (draw_equivalent_wins_for_white * 1e9) as u64;
  let mut x = k.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(d);
  x ^= x >> 30;
  x = x.wrapping_mul(0xbf58476d1ce4e5b9);
  x ^= x >> 27;
  x = x.wrapping_mul(0x94d049bb133111eb);
  x ^ (x >> 31)
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
    let h = BoardHistory::new(&b, Player::Black, Rules::default(), 0);
    (b, h)
  }

  #[test]
  fn new_game_is_not_finished() {
    let (_, h) = make_game();
    assert!(!h.is_game_finished);
    assert_eq!(h.consecutive_ending_passes, 0);
  }

  #[test]
  fn two_passes_end_game() {
    let (mut b, mut h) = make_game();
    h.make_board_move(&mut b, PASS_LOC, Player::Black, None);
    assert!(!h.is_game_finished);
    h.make_board_move(&mut b, PASS_LOC, Player::White, None);
    assert!(h.is_game_finished);
    assert!(h.is_scored);
  }

  #[test]
  fn playing_a_stone_resets_passes() {
    let (mut b, mut h) = make_game();
    h.make_board_move(&mut b, PASS_LOC, Player::Black, None);
    h.make_board_move(&mut b, location::get_loc(3, 3, 9), Player::White, None);
    assert_eq!(h.consecutive_ending_passes, 0);
  }

  #[test]
  fn resign_marks_winner() {
    let (_, mut h) = make_game();
    h.set_winner_by_resignation(Player::White);
    assert!(h.is_game_finished);
    assert_eq!(h.winner, Some(Player::White));
    assert!(h.is_resignation);
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

  #[test]
  fn area_scoring_gives_reasonable_result_on_empty_board() {
    let (mut b, mut h) = make_game();
    // Play two passes to end the game on an empty board.
    h.make_board_move(&mut b, PASS_LOC, Player::Black, None);
    h.make_board_move(&mut b, PASS_LOC, Player::White, None);
    // On an empty board with komi 7.5, white wins.
    assert!(h.is_scored);
    assert!(h.final_white_minus_black_score > 0.0,
      "white should win on empty board with positive komi");
  }
}
