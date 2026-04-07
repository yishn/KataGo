/// Precomputed standard-normal PDF/CDF lookup table.
/// Used when `value_weight_exponent > 0` in `recompute_node_stats` to downweight
/// children with unusually bad/good values relative to the mean.
///
/// Mirrors `cpp/search/distributiontable.h/cpp`.

/// Standard normal PDF/CDF lookup table over [-8, 8].
pub struct DistributionTable {
  pdf: Vec<f64>,
  cdf: Vec<f64>,
  min_z: f64,
  max_z: f64,
  step: f64,
}

impl DistributionTable {
  /// Build the table with `n_steps` equally-spaced points over `[min_z, max_z]`.
  pub fn new(min_z: f64, max_z: f64, n_steps: usize) -> Self {
    assert!(n_steps >= 2);
    let step = (max_z - min_z) / (n_steps - 1) as f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let inv_sqrt_2 = 1.0 / std::f64::consts::SQRT_2;

    let mut pdf = Vec::with_capacity(n_steps);
    let mut cdf = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
      let z = min_z + i as f64 * step;
      pdf.push(inv_sqrt_2pi * (-0.5 * z * z).exp());
      // CDF via erfc: Φ(z) = 0.5 * erfc(-z/√2)
      cdf.push(0.5 * erfc(-z * inv_sqrt_2));
    }
    DistributionTable {
      pdf,
      cdf,
      min_z,
      max_z,
      step,
    }
  }

  /// Returns the standard normal PDF at `z`, clamped to table range.
  pub fn pdf(&self, z: f64) -> f64 {
    let idx = self.clamp_idx(z);
    self.pdf[idx]
  }

  /// Returns the standard normal CDF at `z`, clamped to table range.
  pub fn cdf(&self, z: f64) -> f64 {
    let idx = self.clamp_idx(z);
    self.cdf[idx]
  }

  fn clamp_idx(&self, z: f64) -> usize {
    let raw = ((z - self.min_z) / self.step).round() as i64;
    raw.clamp(0, (self.pdf.len() - 1) as i64) as usize
  }
}

impl Default for DistributionTable {
  fn default() -> Self {
    DistributionTable::new(-8.0, 8.0, 1601)
  }
}

/// Approximate complementary error function via the Horner-form rational approximation
/// (error < 1.5e-7 for |x| ≤ 3.75, standard for normal CDF computation).
fn erfc(x: f64) -> f64 {
  // Use the standard library if available — Rust's f64 doesn't expose erfc directly,
  // so we use an explicit series approximation.
  // Abramowitz & Stegun 7.1.26
  if x < 0.0 {
    return 2.0 - erfc(-x);
  }
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let poly = t
    * (0.254829592
      + t
        * (-0.284496736
          + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  poly * (-x * x).exp()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cdf_at_zero_is_half() {
    let dt = DistributionTable::default();
    assert!((dt.cdf(0.0) - 0.5).abs() < 1e-4);
  }

  #[test]
  fn pdf_peak_at_zero() {
    let dt = DistributionTable::default();
    let peak = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    assert!((dt.pdf(0.0) - peak).abs() < 1e-4);
  }

  #[test]
  fn cdf_monotone() {
    let dt = DistributionTable::default();
    assert!(dt.cdf(-2.0) < dt.cdf(0.0));
    assert!(dt.cdf(0.0) < dt.cdf(2.0));
  }
}
