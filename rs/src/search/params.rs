/// Search configuration parameters.
/// Mirrors `cpp/search/searchparams.h` with Rust naming conventions.
/// Defaults match the C++ constructor in `cpp/search/searchparams.cpp`.
#[derive(Debug, Clone)]
pub struct SearchParams {
  // -----------------------------------------------------------------------
  // Utility function parameters
  // -----------------------------------------------------------------------
  pub win_loss_utility_factor: f64,
  pub static_score_utility_factor: f64,
  pub dynamic_score_utility_factor: f64,
  pub dynamic_score_center_zero_weight: f64,
  pub dynamic_score_center_scale: f64,
  pub no_result_utility_for_white: f64,
  pub draw_equivalent_wins_for_white: f64,

  // -----------------------------------------------------------------------
  // CPUCT exploration
  // -----------------------------------------------------------------------
  pub cpuct_exploration: f64,
  pub cpuct_exploration_log: f64,
  pub cpuct_exploration_base: f64,
  pub cpuct_utility_stdev_prior: f64,
  pub cpuct_utility_stdev_prior_weight: f64,
  pub cpuct_utility_stdev_scale: f64,

  // -----------------------------------------------------------------------
  // FPU (First-Play Urgency)
  // -----------------------------------------------------------------------
  pub fpu_reduction_max: f64,
  pub fpu_loss_prop: f64,
  pub fpu_parent_weight_by_visited_policy: bool,
  pub fpu_parent_weight_by_visited_policy_pow: f64,
  pub fpu_parent_weight: f64,
  pub policy_optimism: f64,

  // -----------------------------------------------------------------------
  // Tree value aggregation
  // -----------------------------------------------------------------------
  pub value_weight_exponent: f64,
  pub use_noise_pruning: bool,
  pub noise_prune_utility_scale: f64,
  pub noise_pruning_cap: f64,

  // -----------------------------------------------------------------------
  // Uncertainty weighting
  // -----------------------------------------------------------------------
  pub use_uncertainty: bool,
  pub uncertainty_coeff: f64,
  pub uncertainty_exponent: f64,
  pub uncertainty_max_weight: f64,

  // -----------------------------------------------------------------------
  // Root exploration parameters
  // -----------------------------------------------------------------------
  pub root_noise_enabled: bool,
  pub root_dirichlet_noise_total_concentration: f64,
  pub root_dirichlet_noise_weight: f64,
  pub root_policy_temperature: f64,
  pub root_policy_temperature_early: f64,
  pub root_fpu_reduction_max: f64,
  pub root_fpu_loss_prop: f64,
  pub root_policy_optimism: f64,

  // -----------------------------------------------------------------------
  // Move selection
  // -----------------------------------------------------------------------
  pub chosen_move_temperature: f64,
  pub chosen_move_temperature_early: f64,
  pub chosen_move_temperature_halflife: f64,
  pub chosen_move_temperature_only_below_prob: f64,
  pub chosen_move_subtract: f64,
  pub chosen_move_prune: f64,
  pub use_lcb_for_selection: bool,
  pub lcb_stdevs: f64,
  pub min_visit_prop_for_lcb: f64,
  pub use_non_buggy_lcb: bool,

  // -----------------------------------------------------------------------
  // Search limits
  // -----------------------------------------------------------------------
  pub max_visits: i64,
  pub max_playouts: i64,

  // -----------------------------------------------------------------------
  // Endgame / pass behaviour
  // -----------------------------------------------------------------------
  pub root_ending_bonus_points: f64,
  pub root_prune_useless_moves: bool,
  pub conservative_pass: bool,
  pub fill_dame_before_pass: bool,

  // -----------------------------------------------------------------------
  // Misc
  // -----------------------------------------------------------------------
  pub wide_root_noise: f64,
  pub nn_policy_temperature: f32,
  pub playout_doubling_advantage: f64,
  pub avoid_repeated_pattern_utility: f64,
}

impl Default for SearchParams {
  fn default() -> Self {
    SearchParams {
      // Utility
      win_loss_utility_factor: 1.0,
      static_score_utility_factor: 0.3,
      dynamic_score_utility_factor: 0.0,
      dynamic_score_center_zero_weight: 0.0,
      dynamic_score_center_scale: 1.0,
      no_result_utility_for_white: 0.0,
      draw_equivalent_wins_for_white: 0.5,

      // CPUCT
      cpuct_exploration: 1.0,
      cpuct_exploration_log: 0.0,
      cpuct_exploration_base: 500.0,
      cpuct_utility_stdev_prior: 0.25,
      cpuct_utility_stdev_prior_weight: 1.0,
      cpuct_utility_stdev_scale: 0.0,

      // FPU
      fpu_reduction_max: 0.2,
      fpu_loss_prop: 0.0,
      fpu_parent_weight_by_visited_policy: false,
      fpu_parent_weight_by_visited_policy_pow: 1.0,
      fpu_parent_weight: 0.0,
      policy_optimism: 0.0,

      // Value aggregation
      value_weight_exponent: 0.5,
      use_noise_pruning: false,
      noise_prune_utility_scale: 0.15,
      noise_pruning_cap: 1e50,

      // Uncertainty
      use_uncertainty: false,
      uncertainty_coeff: 0.2,
      uncertainty_exponent: 1.0,
      uncertainty_max_weight: 8.0,

      // Root
      root_noise_enabled: false,
      root_dirichlet_noise_total_concentration: 10.83,
      root_dirichlet_noise_weight: 0.25,
      root_policy_temperature: 1.0,
      root_policy_temperature_early: 1.0,
      root_fpu_reduction_max: 0.2,
      root_fpu_loss_prop: 0.0,
      root_policy_optimism: 0.0,

      // Move selection
      chosen_move_temperature: 0.0,
      chosen_move_temperature_early: 0.0,
      chosen_move_temperature_halflife: 19.0,
      chosen_move_temperature_only_below_prob: 1.0,
      chosen_move_subtract: 0.0,
      chosen_move_prune: 1.0,
      use_lcb_for_selection: false,
      lcb_stdevs: 4.0,
      min_visit_prop_for_lcb: 0.05,
      use_non_buggy_lcb: false,

      // Limits
      max_visits: 1 << 50,
      max_playouts: 1 << 50,

      // Endgame
      root_ending_bonus_points: 0.0,
      root_prune_useless_moves: false,
      conservative_pass: false,
      fill_dame_before_pass: false,

      // Misc
      wide_root_noise: 0.0,
      nn_policy_temperature: 1.0,
      playout_doubling_advantage: 0.0,
      avoid_repeated_pattern_utility: 0.0,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_params_compile() {
    let p = SearchParams::default();
    assert_eq!(p.win_loss_utility_factor, 1.0);
    assert_eq!(p.cpuct_exploration, 1.0);
    assert_eq!(p.fpu_reduction_max, 0.2);
    assert_eq!(p.value_weight_exponent, 0.5);
    assert!(!p.use_lcb_for_selection);
    assert_eq!(p.max_visits, 1 << 50);
  }
}
