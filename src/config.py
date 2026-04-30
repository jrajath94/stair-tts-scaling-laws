"""Central configuration for the information-theoretic test-time compute scaling experiment.

Contains dataset generation parameters, method hyperparameters, evaluation settings,
compute budget guards, and hypothesis thresholds. Single source of truth for all constants.
"""

import numpy as np


class ExperimentConfig:
    """All hyperparameters and derived values for the experiment."""

    def __init__(self):
        # ── Compute budget ──
        self.total_time_budget_seconds = 600
        self.hard_cap_fraction = 0.8
        self.seeds = [0, 1, 2, 3, 4]
        self.n_seeds = 5

        # ── Synthetic dataset ──
        self.n_problems_per_cell = 200
        self.total_problems = 800  # 4 cells x 200
        self.reasoning_budgets = [4, 8, 16, 32, 64, 128, 256, 512]
        self.n_budgets = 8
        self.temperatures = [0.1, 0.5, 1.0]
        self.n_temperatures = 3
        self.samples_per_cell = 32
        self.planted_staircase_fraction = 0.6

        # ── Regime factors (2x2 factorial) ──
        self.complexity_levels = ["low", "high"]
        self.k_depth_correlations = ["correlated", "decorrelated"]
        self.low_dc_range = (8, 64)
        self.high_dc_range = (64, 512)
        self.correlated_k_slope = 0.8
        self.correlated_k_noise_frac = 0.1
        self.decorrelated_k_range = (5, 100)

        # ── Transition sharpness ──
        self.sharp_gamma_range = (1, 4)
        self.smooth_gamma_range = (8, 20)

        # ── Model configurations ──
        self.model_configs = {
            "model_A": {
                "noise_scale": 0.15,
                "depth_sensitivity": 1.0,
                "mi_nonmonotone_probability": 0.3,
            },
            "model_B": {
                "noise_scale": 0.25,
                "depth_sensitivity": 0.7,
                "mi_nonmonotone_probability": 0.45,
            },
        }

        # ── Accuracy simulation ──
        self.acc_low_range = (0.05, 0.15)
        self.acc_high_range = (0.85, 0.95)
        self.accuracy_clip_min = 0.001
        self.accuracy_clip_max = 0.999
        self.eps = 1e-8

        # ── GSM8K ecological validation ──
        self.gsm8k_n_problems = 500
        self.gsm8k_gzip_level = 9

        # ── SmoothLogconcaveBound hyperparams ──
        self.slb_a_init = 1.0
        self.slb_b_init = 0.1
        self.slb_c_init = 0.01
        self.slb_bounds_a = (0.01, 10.0)
        self.slb_bounds_b = (0.001, 1.0)
        self.slb_bounds_c = (0.0001, 0.1)
        self.slb_optimizer = "L-BFGS-B"

        # ── EmpiricalPowerLawFit hyperparams ──
        self.eplf_accuracy_clip = 0.999
        self.eplf_marginal_gain_epsilon = 0.01
        self.eplf_min_budget_for_fit = 4

        # ── StaircaseTemperatureAllocator hyperparams ──
        self.sta_bic_evidence_threshold = 2.0
        self.sta_holdout_fraction = 0.2
        self.sta_n_complexity_buckets = 3
        self.sta_safety_margin = 1.2
        self.sta_temperature_grid = [0.1, 0.5, 1.0]  # constrained to simulated temps

        # ── MINonmonotoneDetector hyperparams ──
        self.mnd_smoothing_window = 3
        self.mnd_constructive_threshold = 0.01
        self.mnd_corrective_threshold = -0.005  # initial; overridden by adaptive estimation
        self.mnd_adaptive_noise_sigmas = 3.5  # corrective threshold = -N * noise_std
        self.mnd_saturation_threshold = 0.005
        self.mnd_holdout_fraction = 0.2

        # ── CircuitDepthPredictor hyperparams ──
        self.cdp_alpha_candidates = [0.01, 0.1, 1.0, 10.0]
        self.cdp_cv_folds = 5
        self.cdp_train_fraction = 0.8

        # ── StaircaseNoTemperature hyperparams (H1 ablation) ──
        self.snt_fixed_temperature = 0.5
        self.snt_bic_evidence_threshold = 2.0
        self.snt_safety_margin = 1.2

        # ── KolmogorovGzipPredictor hyperparams (H3 ablation) ──
        self.kgp_gzip_compression_level = 9
        self.kgp_alpha_candidates = [0.01, 0.1, 1.0, 10.0]
        self.kgp_cv_folds = 5

        # ── Evaluation / statistical analysis ──
        self.bootstrap_n_resamples = 10000
        self.bootstrap_ci_level = 0.95
        self.cross_seed_cv_target = 0.15
        self.success_rate_target = 0.95

        # ── Hypothesis thresholds ──
        self.h1_bic_win_rate_threshold = 0.55
        self.h1_bootstrap_cv_threshold = 0.20
        self.h1_temp_complexity_spearman_threshold = 0.60
        self.h2_elbow_divergence_threshold = 0.35
        self.h2_mi_nonmonotonicity_threshold = 0.25
        self.h3_depth_elbow_corr_threshold = 0.65
        self.h3_gzip_elbow_corr_threshold = 0.35
        self.h3_mape_gap_threshold = 10.0  # percentage points

        # ── Scaling fallback thresholds ──
        self.fallback_time_fraction = 0.7  # trigger at 70% budget consumed
        self.fallback_reduced_problems = 100
        self.fallback_reduced_bootstrap = 5000
        self.fallback_reduced_seeds = 3

        # ── Derived values ──
        self.hard_cap_seconds = self.total_time_budget_seconds * self.hard_cap_fraction  # 480
        self.budget_array = np.array(self.reasoning_budgets, dtype=float)  # shape [8]
        self.n_regime_cells = len(self.complexity_levels) * len(self.k_depth_correlations)  # 4

        # Validate consistency
        assert self.total_problems == self.n_problems_per_cell * self.n_regime_cells, (
            f"total_problems ({self.total_problems}) != "
            f"n_problems_per_cell ({self.n_problems_per_cell}) * "
            f"n_regime_cells ({self.n_regime_cells})"
        )
        assert self.n_budgets == len(self.reasoning_budgets), (
            f"n_budgets ({self.n_budgets}) != len(reasoning_budgets) ({len(self.reasoning_budgets)})"
        )
        assert self.n_temperatures == len(self.temperatures), (
            f"n_temperatures ({self.n_temperatures}) != len(temperatures) ({len(self.temperatures)})"
        )
        assert self.n_seeds == len(self.seeds), (
            f"n_seeds ({self.n_seeds}) != len(seeds) ({len(self.seeds)})"
        )

    def get_condition_names(self):
        """Return ordered list of all 7 experimental condition names.

        Order: baselines first, then proposed methods, then ablations.
        This ordering ensures baselines are computed before methods that
        may reference their results in hypothesis testing.
        """
        return [
            "smooth_logconcave_bound",
            "empirical_power_law_fit",
            "staircase_temperature_allocator",
            "mi_nonmonotone_detector",
            "circuit_depth_predictor",
            "staircase_no_temperature",
            "kolmogorov_gzip_predictor",
        ]
