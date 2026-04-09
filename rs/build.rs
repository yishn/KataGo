/// Build script: generates `src/game/zobrist_tables.rs` with all Zobrist hash
/// constants used by the Go board.  The generation algorithm exactly replicates
/// `Board::initHash()` from `cpp/game/board.cpp`.
///
/// Seeding algorithm (from cpp/core/rand.cpp `Rand::init`):
///   1. MD5(seed)  →  4×u32 digest
///   2. Build string  `"|{digest[0]}|{seed}"`
///   3. Loop with counter (starts 0, step 37):
///      hash `"{counter}{s}"` with SHA-256  →  4×u64
///      emit non-zero u64s; when exhausted advance counter and rehash
///   4. Collect 16 non-zero u64s for XorShift1024Mult  +  1 for PCG32
///
/// `nextUInt()` = pcg32.next() + xorm.next()   (u32 + u32, wrapping)
/// `nextUInt64()` = lower-32 | (upper-32 << 32)   (two consecutive u32 calls)
use md5::{Digest as _, Md5};
use sha2::Sha256;
use std::{env, fs, path::Path};

// ---------------------------------------------------------------------------
// PRNG — mirrors C++ exactly
// ---------------------------------------------------------------------------

struct XorShift1024Mult {
  a: [u64; 16],
  idx: usize,
}

impl XorShift1024Mult {
  fn new(a: [u64; 16]) -> Self {
    Self { a, idx: 0 }
  }

  fn next_u32(&mut self) -> u32 {
    let a0 = self.a[self.idx];
    self.idx = (self.idx + 1) & 15;
    let mut a1 = self.a[self.idx];
    a1 ^= a1 << 31;
    a1 ^= a1 >> 11;
    let a0x = a0 ^ (a0 >> 30);
    self.a[self.idx] = a0x ^ a1;
    ((self.a[self.idx].wrapping_mul(1181783497276652981u64)) >> 32) as u32
  }
}

struct Pcg32 {
  s: u64,
}

impl Pcg32 {
  fn new(s: u64) -> Self {
    Self { s }
  }

  fn next_u32(&mut self) -> u32 {
    self.s = self
      .s
      .wrapping_mul(6364136223846793005u64)
      .wrapping_add(1442695040888963407u64);
    let x = ((self.s >> 18) ^ self.s) >> 27;
    let rot = (self.s >> 59) as u32;
    let x32 = x as u32;
    if rot == 0 {
      x32
    } else {
      (x32 >> rot) | (x32 << (32u32.wrapping_sub(rot)))
    }
  }
}

struct Rand {
  xorm: XorShift1024Mult,
  pcg: Pcg32,
}

impl Rand {
  fn from_seed(seed: &str) -> Self {
    // MD5(seed)
    let mut h = Md5::new();
    h.update(seed.as_bytes());
    let digest: [u8; 16] = h.finalize().into();
    let d0 = u32::from_le_bytes(digest[0..4].try_into().unwrap());

    let s = format!("|{}|{}", d0, seed);

    let mut counter: i64 = 0;
    let mut hash_buf = [0u64; 4];
    let mut hash_idx = 4usize; // start exhausted so we hash immediately

    let mut get_nonzero = move || -> u64 {
      loop {
        if hash_idx < 4 {
          let v = hash_buf[hash_idx];
          hash_idx += 1;
          if v != 0 {
            return v;
          }
        } else {
          // Hash "{counter}{s}" with SHA-256
          let input = format!("{}{}", counter, s);
          counter += 37;
          let digest: [u8; 32] = Sha256::digest(input.as_bytes()).into();
          for i in 0..4 {
            hash_buf[i] = u64::from_le_bytes(
              digest[i * 8..(i + 1) * 8].try_into().unwrap(),
            );
          }
          hash_idx = 0;
        }
      }
    };

    let mut init_a = [0u64; 16];
    for x in &mut init_a {
      *x = get_nonzero();
    }
    let pcg_seed = get_nonzero();

    Self {
      xorm: XorShift1024Mult::new(init_a),
      pcg: Pcg32::new(pcg_seed),
    }
  }

  fn next_u32(&mut self) -> u32 {
    self.pcg.next_u32().wrapping_add(self.xorm.next_u32())
  }

  fn next_u64(&mut self) -> u64 {
    let lo = self.next_u32() as u64;
    let hi = self.next_u32() as u64;
    lo | (hi << 32)
  }

  /// Re-initialise with a new seed string (matches `rand.init(newSeed)` in C++).
  fn reinit(&mut self, seed: &str) {
    *self = Self::from_seed(seed);
  }
}

// ---------------------------------------------------------------------------
// Main generation
// ---------------------------------------------------------------------------

const MAX_LEN: usize = 19;
const MAX_ARR_SIZE: usize = (MAX_LEN + 1) * (MAX_LEN + 2) + 1; // 421

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();
  let dest = Path::new(&out_dir).join("zobrist_tables.rs");

  // -----------------------------------------------------------------------
  // Phase 1 — seed "Board::initHash()"
  // -----------------------------------------------------------------------
  let mut rand = Rand::from_seed("Board::initHash()");

  let mut player_hash = [[0u64; 2]; 4];
  for h in &mut player_hash {
    h[0] = rand.next_u64();
    h[1] = rand.next_u64();
  }

  // board_hash[loc][color]:  empty(0) and wall(3) → zero, black(1) and white(2) → random
  let mut board_hash = [[[0u64; 2]; 4]; MAX_ARR_SIZE];
  let mut ko_mark_hash = [[[0u64; 2]; 4]; MAX_ARR_SIZE];
  let mut ko_loc_hash = [[0u64; 2]; MAX_ARR_SIZE];
  for i in 0..MAX_ARR_SIZE {
    for j in 0usize..4 {
      if j == 0 || j == 3 {
        // C_EMPTY=0, C_WALL=3 → zero (Hash128 default)
        board_hash[i][j] = [0, 0];
        ko_mark_hash[i][j] = [0, 0];
      } else {
        board_hash[i][j] = [rand.next_u64(), rand.next_u64()];
        ko_mark_hash[i][j] = [rand.next_u64(), rand.next_u64()];
      }
    }
    ko_loc_hash[i] = [rand.next_u64(), rand.next_u64()];
  }

  // -----------------------------------------------------------------------
  // Phase 3 — size hashes (reseed)
  // -----------------------------------------------------------------------
  rand.reinit("Board::initHash() for ZOBRIST_SIZE hashes");
  let mut size_x_hash = [[0u64; 2]; MAX_LEN + 1];
  let mut size_y_hash = [[0u64; 2]; MAX_LEN + 1];
  for i in 0..=MAX_LEN {
    size_x_hash[i] = [rand.next_u64(), rand.next_u64()];
    size_y_hash[i] = [rand.next_u64(), rand.next_u64()];
  }

  // -----------------------------------------------------------------------
  // Emit Rust source
  // -----------------------------------------------------------------------
  let mut src = String::new();
  src.push_str("// AUTO-GENERATED by build.rs — do not edit by hand.\n");
  src.push_str(
    "// Zobrist hash tables matching Board::initHash() from board.cpp.\n\n",
  );
  src.push_str("use crate::game::Hash128;\n\n");

  fn emit_arr1(src: &mut String, name: &str, arr: &[[u64; 2]]) {
    src.push_str(&format!(
      "pub static {}: [Hash128; {}] = [\n",
      name,
      arr.len()
    ));
    for h in arr {
      src.push_str(&format!(
        "  Hash128 {{ hash0: 0x{:016x}, hash1: 0x{:016x} }},\n",
        h[0], h[1]
      ));
    }
    src.push_str("];\n\n");
  }

  fn emit_arr2(src: &mut String, name: &str, arr: &[[[u64; 2]; 4]]) {
    src.push_str(&format!(
      "pub static {}: [[Hash128; 4]; {}] = [\n",
      name,
      arr.len()
    ));
    for row in arr {
      src.push_str("  [\n");
      for h in row {
        src.push_str(&format!(
          "    Hash128 {{ hash0: 0x{:016x}, hash1: 0x{:016x} }},\n",
          h[0], h[1]
        ));
      }
      src.push_str("  ],\n");
    }
    src.push_str("];\n\n");
  }

  emit_arr1(&mut src, "ZOBRIST_PLAYER_HASH", &player_hash);
  emit_arr2(&mut src, "ZOBRIST_BOARD_HASH", &board_hash);
  emit_arr2(&mut src, "ZOBRIST_KO_MARK_HASH", &ko_mark_hash);
  emit_arr1(&mut src, "ZOBRIST_KO_LOC_HASH", &ko_loc_hash);
  emit_arr1(&mut src, "ZOBRIST_SIZE_X_HASH", &size_x_hash);
  emit_arr1(&mut src, "ZOBRIST_SIZE_Y_HASH", &size_y_hash);

  fs::write(&dest, src).unwrap();

  println!("cargo:rerun-if-changed=build.rs");
}
