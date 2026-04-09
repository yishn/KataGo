/// Go rules — mirrors `cpp/game/rules.h` and `cpp/game/rules.cpp`.
///
/// All Zobrist hash constants are hardcoded from the C++ source.
use crate::game::Hash128;

// ---------------------------------------------------------------------------
// Rule enums
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum KoRule {
  Simple = 0,
  Positional = 1,
  Situational = 2,
  Spight = 3,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ScoringRule {
  Area = 0,
  Territory = 1,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TaxRule {
  None = 0,
  Seki = 1,
  All = 2,
}

/// White handicap bonus rule.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum WhbRule {
  Zero = 0,
  N = 1,
  NMinusOne = 2,
}

// ---------------------------------------------------------------------------
// Rules struct
// ---------------------------------------------------------------------------

/// The complete ruleset used for one game.
#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
  pub ko_rule: KoRule,
  pub scoring_rule: ScoringRule,
  pub tax_rule: TaxRule,
  pub multi_stone_suicide_legal: bool,
  pub has_button: bool,
  pub white_handicap_bonus_rule: WhbRule,
  pub friendly_pass_ok: bool,
  pub komi: f32,
}

impl Default for Rules {
  /// Closest to Tromp-Taylor: area scoring, positional ko, no tax,
  /// multi-stone suicide legal, komi 7.5.  Matches C++ `Rules()`.
  fn default() -> Self {
    Self {
      ko_rule: KoRule::Positional,
      scoring_rule: ScoringRule::Area,
      tax_rule: TaxRule::None,
      multi_stone_suicide_legal: true,
      has_button: false,
      white_handicap_bonus_rule: WhbRule::Zero,
      friendly_pass_ok: false,
      komi: 7.5,
    }
  }
}

impl Rules {
  /// Tromp-Taylor-ish: area scoring, positional ko, komi 7.5.
  pub fn tromp_taylorish() -> Self {
    Self::default()
  }

  /// Simple territory rules: territory scoring, simple ko, TAX_SEKI, komi 6.5.
  pub fn simple_territory() -> Self {
    Self {
      ko_rule: KoRule::Simple,
      scoring_rule: ScoringRule::Territory,
      tax_rule: TaxRule::Seki,
      multi_stone_suicide_legal: false,
      has_button: false,
      white_handicap_bonus_rule: WhbRule::Zero,
      friendly_pass_ok: false,
      komi: 6.5,
    }
  }

  /// Whether the game result will be an integer (no half-point komi).
  pub fn game_result_will_be_integer(&self) -> bool {
    (self.komi - self.komi.round()).abs() < 0.0001
      && !(self.scoring_rule == ScoringRule::Area && self.has_button)
  }

  /// Compute the Zobrist hash contribution of these rules (used in BoardHistory).
  pub fn rules_hash(&self) -> Hash128 {
    let mut h = ZOBRIST_KO_RULE_HASH[self.ko_rule as usize];
    h ^= ZOBRIST_SCORING_RULE_HASH[self.scoring_rule as usize];
    h ^= ZOBRIST_TAX_RULE_HASH[self.tax_rule as usize];
    if self.multi_stone_suicide_legal {
      h ^= ZOBRIST_MULTI_STONE_SUICIDE_HASH;
    }
    if self.has_button {
      h ^= ZOBRIST_BUTTON_HASH;
    }
    if self.friendly_pass_ok {
      h ^= ZOBRIST_FRIENDLY_PASS_OK_HASH;
    }
    h
  }
}

// ---------------------------------------------------------------------------
// Zobrist constants — hardcoded from cpp/game/rules.cpp
// ---------------------------------------------------------------------------

pub const ZOBRIST_KO_RULE_HASH: [Hash128; 4] = [
  Hash128::new(0x3cc7e0bf846820f6, 0x1fb7fbde5fc6ba4e), // KO_SIMPLE
  Hash128::new(0xcc18f5d47188554a, 0x3a63152c23e4128d), // KO_POSITIONAL
  Hash128::new(0x3bc55e42b23b35bf, 0xc75fa1e615621dcd), // KO_SITUATIONAL
  Hash128::new(0x5b2096e48241d21b, 0x23cc18d4e85cd67f), // KO_SPIGHT
];

pub const ZOBRIST_SCORING_RULE_HASH: [Hash128; 2] = [
  Hash128::new(
    0x8b3ed7598f901494 ^ 0x72eeccc72c82a5e7,
    0x1dfd47ac77bce5f8 ^ 0x0d1265e413623e2b,
  ), // SCORING_AREA
  Hash128::new(
    0x381345dc357ec982 ^ 0x125bfe48a41042d5,
    0x03ba55c026026b56 ^ 0x061866b5f2b98a79,
  ), // SCORING_TERRITORY
];

pub const ZOBRIST_TAX_RULE_HASH: [Hash128; 3] = [
  Hash128::new(0x72eeccc72c82a5e7, 0x0d1265e413623e2b), // TAX_NONE
  Hash128::new(0x125bfe48a41042d5, 0x061866b5f2b98a79), // TAX_SEKI
  Hash128::new(0xa384ece9d8ee713c, 0xfdc9f3b5d1f3732b), // TAX_ALL
];

pub const ZOBRIST_MULTI_STONE_SUICIDE_HASH: Hash128 =
  Hash128::new(0xf9b475b3bbf35e37, 0xefa19d8b1e5b3e5a);

pub const ZOBRIST_BUTTON_HASH: Hash128 =
  Hash128::new(0xb8b914c9234ece84, 0x3d759cddebe29c14);

pub const ZOBRIST_FRIENDLY_PASS_OK_HASH: Hash128 =
  Hash128::new(0x0113655998ef0a25, 0x99c9d04ecd964874);

// Komi hashes are computed numerically from komi value in BoardHistory.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_rules_are_tromp_taylorish() {
    let r = Rules::default();
    assert_eq!(r.ko_rule, KoRule::Positional);
    assert_eq!(r.scoring_rule, ScoringRule::Area);
    assert_eq!(r.tax_rule, TaxRule::None);
    assert!(r.multi_stone_suicide_legal);
    assert!((r.komi - 7.5).abs() < 0.001);
  }

  #[test]
  fn rules_hash_differs_by_ko_rule() {
    let mut r1 = Rules::default();
    let mut r2 = Rules::default();
    r1.ko_rule = KoRule::Simple;
    r2.ko_rule = KoRule::Situational;
    assert_ne!(r1.rules_hash(), r2.rules_hash());
  }

  #[test]
  fn rules_hash_differs_by_scoring() {
    let mut r1 = Rules::default();
    let mut r2 = Rules::default();
    r1.scoring_rule = ScoringRule::Area;
    r2.scoring_rule = ScoringRule::Territory;
    assert_ne!(r1.rules_hash(), r2.rules_hash());
  }

  #[test]
  fn ko_rule_hash_constants_are_distinct() {
    let hashes: Vec<_> = ZOBRIST_KO_RULE_HASH.iter().collect();
    for i in 0..4 {
      for j in (i + 1)..4 {
        assert_ne!(
          hashes[i], hashes[j],
          "KO_RULE_HASH[{i}] == KO_RULE_HASH[{j}]"
        );
      }
    }
  }
}
