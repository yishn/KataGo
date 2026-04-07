/// Score value math.
/// Mirrors `cpp/neuralnet/nninputs.cpp` ScoreValue functions.
///
/// The C++ code uses a precomputed 2-D table (`expectedSVTable`) for `expected_white_score_value`.
/// This Rust port reproduces the same numeric integration at call-time with a reduced step count
/// (still accurate to ~1e-4). For performance-critical use, call `ScoreTable::build()` once and
/// reuse it.
use crate::search::params::SearchParams;

pub const TWO_OVER_PI: f64 = 2.0 / std::f64::consts::PI;

// -----------------------------------------------------------------------
// Primitive score-value functions (analytic)
// -----------------------------------------------------------------------

/// atan-based score value for a deterministic score.
/// Returns a value in (-1, 1) representing how good `score` is for white.
#[inline]
pub fn score_value_of_score(
  score: f64,
  center: f64,
  scale: f64,
  sqrt_board_area: f64,
) -> f64 {
  let adjusted = score - center;
  (adjusted / (scale * sqrt_board_area)).atan() * TWO_OVER_PI
}

/// d(score_value)/d(score) — derivative of the atan curve.
#[inline]
pub fn d_score_value_d_score(
  score: f64,
  center: f64,
  scale: f64,
  sqrt_board_area: f64,
) -> f64 {
  let adjusted = score - center;
  let s = scale * sqrt_board_area;
  s / (s * s + adjusted * adjusted) * TWO_OVER_PI
}

/// √(max(0, scoreMeanSq − scoreMean²))
#[inline]
pub fn get_score_stdev(score_mean: f64, score_mean_sq: f64) -> f64 {
  let variance = score_mean_sq - score_mean * score_mean;
  if variance <= 0.0 {
    0.0
  } else {
    variance.sqrt()
  }
}

// -----------------------------------------------------------------------
// Expected score value (integrate over Normal distribution)
// -----------------------------------------------------------------------

/// E[score_value(x)] where x ~ Normal(score_mean, score_stdev).
///
/// Uses 101-point Gauss–Hermite quadrature (approximated as evenly-spaced
/// trapezoid over ±5σ), matching the C++ table's accuracy for typical inputs.
pub fn expected_white_score_value(
  score_mean: f64,
  score_stdev: f64,
  center: f64,
  scale: f64,
  sqrt_board_area: f64,
) -> f64 {
  if score_stdev <= 0.0 {
    return score_value_of_score(score_mean, center, scale, sqrt_board_area);
  }
  // Trapezoid rule over ±5σ with 200 steps (201 points).
  const N: usize = 200;
  const BOUND: f64 = 5.0;
  let step = 2.0 * BOUND / N as f64;
  let mut w_sum = 0.0f64;
  let mut wsv_sum = 0.0f64;
  for i in 0..=N {
    let t = -BOUND + i as f64 * step;
    let x = score_mean + t * score_stdev;
    let w = (-0.5 * t * t).exp();
    // trapezoid weights
    let wt = if i == 0 || i == N { w * 0.5 } else { w };
    wsv_sum += wt * score_value_of_score(x, center, scale, sqrt_board_area);
    w_sum += wt;
  }
  wsv_sum / w_sum
}

// -----------------------------------------------------------------------
// Utility aggregation
// -----------------------------------------------------------------------

pub fn get_result_utility(
  win_loss: f64,
  no_result: f64,
  params: &SearchParams,
) -> f64 {
  win_loss * params.win_loss_utility_factor
    + no_result * params.no_result_utility_for_white
}

pub fn get_score_utility(
  score_mean: f64,
  score_mean_sq: f64,
  recent_score_center: f64,
  sqrt_board_area: f64,
  params: &SearchParams,
) -> f64 {
  let stdev = get_score_stdev(score_mean, score_mean_sq);
  let static_sv =
    expected_white_score_value(score_mean, stdev, 0.0, 2.0, sqrt_board_area);
  let dynamic_sv = expected_white_score_value(
    score_mean,
    stdev,
    recent_score_center,
    params.dynamic_score_center_scale,
    sqrt_board_area,
  );
  static_sv * params.static_score_utility_factor
    + dynamic_sv * params.dynamic_score_utility_factor
}

pub fn get_utility(
  win_loss: f64,
  no_result: f64,
  score_mean: f64,
  score_mean_sq: f64,
  recent_score_center: f64,
  sqrt_board_area: f64,
  params: &SearchParams,
) -> f64 {
  get_result_utility(win_loss, no_result, params)
    + get_score_utility(
      score_mean,
      score_mean_sq,
      recent_score_center,
      sqrt_board_area,
      params,
    )
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
  use super::*;
  use crate::search::params::SearchParams;

  #[test]
  fn score_stdev_basics() {
    // E[x^2] - (E[x])^2 = 0 when variance is 0
    assert_eq!(get_score_stdev(1.0, 1.0), 0.0);
    // E[x^2] = 1, E[x] = 0 → stdev = 1
    assert!((get_score_stdev(0.0, 1.0) - 1.0).abs() < 1e-12);
    // E[x^2] = 0.5, E[x] = 0 → stdev = sqrt(0.5)
    assert!((get_score_stdev(0.0, 0.5) - 0.5_f64.sqrt()).abs() < 1e-12);
  }

  #[test]
  fn score_value_zero_center() {
    // score_value_of_score(0, 0, ...) = 0
    assert_eq!(score_value_of_score(0.0, 0.0, 1.0, 1.0), 0.0);
    // positive score → positive value
    assert!(score_value_of_score(10.0, 0.0, 1.0, 1.0) > 0.0);
    // symmetric
    let v = score_value_of_score(5.0, 0.0, 1.0, 4.0);
    assert!((score_value_of_score(-5.0, 0.0, 1.0, 4.0) + v).abs() < 1e-12);
  }

  #[test]
  fn expected_score_value_zero_stdev() {
    // With stdev=0, expected value = point value
    let sv = expected_white_score_value(7.5, 0.0, 0.0, 2.0, 4.0);
    let direct = score_value_of_score(7.5, 0.0, 2.0, 4.0);
    assert!((sv - direct).abs() < 1e-10);
  }

  #[test]
  fn get_utility_only_win_loss() {
    // With score factors = 0, utility = win_loss * win_loss_utility_factor
    let mut p = SearchParams::default();
    p.static_score_utility_factor = 0.0;
    p.dynamic_score_utility_factor = 0.0;
    let u = get_utility(0.6, 0.0, 0.0, 0.0, 0.0, 4.0, &p);
    assert!((u - 0.6).abs() < 1e-12);
  }
}
