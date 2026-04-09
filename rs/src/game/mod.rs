/// Core game types shared across all game submodules.
///
/// This module mirrors the C++ `game/` directory and is entirely WASM-compatible
/// (no filesystem I/O).

pub mod board;
pub mod boardhistory;
pub mod rules;

// ---------------------------------------------------------------------------
// Hash128 — 128-bit Zobrist hash, matches Hash128 in cpp/core/hash.h
// ---------------------------------------------------------------------------

/// A 128-bit hash value composed of two independent 64-bit halves.
///
/// `hash0` is the low half, `hash1` the high half — matching the C++ layout.
/// The default value is `{0, 1}` to match `Hash128::Hash128()` in C++.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Hash128 {
  pub hash0: u64,
  pub hash1: u64,
}

impl Default for Hash128 {
  /// Matches the C++ `Hash128()` default constructor: `{hash0: 0, hash1: 1}`.
  fn default() -> Self {
    Self { hash0: 0, hash1: 1 }
  }
}

impl Hash128 {
  pub const fn new(hash0: u64, hash1: u64) -> Self {
    Self { hash0, hash1 }
  }

  pub const ZERO: Self = Self { hash0: 0, hash1: 0 };
}

impl std::ops::BitXor for Hash128 {
  type Output = Self;
  fn bitxor(self, rhs: Self) -> Self {
    Self {
      hash0: self.hash0 ^ rhs.hash0,
      hash1: self.hash1 ^ rhs.hash1,
    }
  }
}

impl std::ops::BitXorAssign for Hash128 {
  fn bitxor_assign(&mut self, rhs: Self) {
    self.hash0 ^= rhs.hash0;
    self.hash1 ^= rhs.hash1;
  }
}
