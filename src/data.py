"""Synthetic noisy reasoning channel generation and GSM8K feature extraction.

Generates the 2x2 factorial synthetic dataset with known ground-truth parameters
(d_c, K, D, gamma, curve_type) and simulates per-problem accuracy curves at
multiple budgets, temperatures, and model configurations. Also provides GSM8K
text-feature extraction for ecological validation.
"""

import gzip as gzip_module
import re

import numpy as np
from scipy import stats as scipy_stats

from config import ExperimentConfig


class Problem:
    """A single synthetic reasoning problem with ground-truth parameters and simulated observations.

    Fields:
        idx: problem index 0..799
        K: Kolmogorov complexity proxy
        D: circuit depth (sequential computation steps)
        d_c: critical reasoning depth (ground truth elbow)
        gamma: transition sharpness parameter
        curve_type: 'staircase' or 'smooth'
        complexity_level: 'low' or 'high'
        k_depth_correlation: 'correlated' or 'decorrelated'
        acc_low: baseline accuracy (near chance)
        acc_high: saturation accuracy
        problem_representation: text string for gzip measurement
        gzip_length: len(gzip.compress(problem_representation))
        accuracy_samples: {model_name: ndarray [n_temps, n_budgets, n_samples]}
        accuracy_means: {model_name: ndarray [n_temps, n_budgets]}
        mi_trajectories: {model_name: ndarray [n_temps, n_budgets]}
        has_mi_nonmonotonicity: {model_name: bool}
        n_steps: 1 or 2 for staircase problems (0 for smooth)
        step_locations: list of step budget locations
    """

    def __init__(
        self,
        idx,
        K,
        D,
        d_c,
        gamma,
        curve_type,
        complexity_level,
        k_depth_correlation,
        acc_low,
        acc_high,
        problem_representation,
        gzip_length,
        n_steps,
        step_locations,
        rng_check,
    ):
        self.idx = idx
        self.K = float(K)
        self.D = float(D)
        self.d_c = float(d_c)
        self.gamma = float(gamma)
        self.curve_type = str(curve_type)
        self.complexity_level = str(complexity_level)
        self.k_depth_correlation = str(k_depth_correlation)
        self.acc_low = float(acc_low)
        self.acc_high = float(acc_high)
        self.problem_representation = str(problem_representation)
        self.gzip_length = int(gzip_length)
        self.n_steps = int(n_steps)
        self.step_locations = list(step_locations)
        # Pre-drawn random value for deterministic MI non-monotonicity planting
        self._rng_check = float(rng_check)
        # Populated by simulation
        self.accuracy_samples = {}
        self.accuracy_means = {}
        self.mi_trajectories = {}
        self.has_mi_nonmonotonicity = {}


class SyntheticDataset:
    """Full synthetic experimental dataset across the 2x2 factorial design.

    Generates 800 problems (4 regime cells x 200 problems) and simulates
    accuracy/MI curves for each of 2 model configurations at 3 temperatures
    and 8 reasoning budget levels.
    """

    def __init__(self, config):
        self.config = config
        self.budgets = config.budget_array.copy()  # shape [8]
        self.problems = []
        self.regime_indices = {}  # {(complexity, correlation): [indices]}

    def generate(self, seed):
        """Generate all problems and simulate observations.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            self for method chaining.
        """
        rng = np.random.default_rng(seed)
        cfg = self.config

        self.problems = []
        self.regime_indices = {}
        problem_idx = 0

        for complexity_level in cfg.complexity_levels:
            for k_depth_corr in cfg.k_depth_correlations:
                regime_key = (complexity_level, k_depth_corr)
                self.regime_indices[regime_key] = []

                for _ in range(cfg.n_problems_per_cell):
                    # a. Draw critical reasoning depth d_c
                    if complexity_level == "low":
                        d_c = rng.uniform(cfg.low_dc_range[0], cfg.low_dc_range[1])
                    else:
                        d_c = rng.uniform(cfg.high_dc_range[0], cfg.high_dc_range[1])

                    # b. Draw circuit depth D (loosely related to d_c)
                    D = d_c * rng.uniform(0.8, 1.5)

                    # c. Draw Kolmogorov complexity K
                    if k_depth_corr == "correlated":
                        K = cfg.correlated_k_slope * D + rng.normal(
                            0, cfg.correlated_k_noise_frac * D
                        )
                        K = max(K, 1.0)  # ensure positive
                    else:
                        K = rng.uniform(
                            cfg.decorrelated_k_range[0], cfg.decorrelated_k_range[1]
                        )

                    # d. Draw curve type
                    if rng.random() < cfg.planted_staircase_fraction:
                        curve_type = "staircase"
                    else:
                        curve_type = "smooth"

                    # e. Draw transition sharpness gamma
                    if curve_type == "staircase":
                        gamma = rng.uniform(
                            cfg.sharp_gamma_range[0], cfg.sharp_gamma_range[1]
                        )
                    else:
                        gamma = rng.uniform(
                            cfg.smooth_gamma_range[0], cfg.smooth_gamma_range[1]
                        )

                    # f. Draw baseline and saturation accuracy
                    acc_low = rng.uniform(cfg.acc_low_range[0], cfg.acc_low_range[1])
                    acc_high = rng.uniform(cfg.acc_high_range[0], cfg.acc_high_range[1])

                    # g. Staircase step configuration
                    if curve_type == "staircase":
                        n_steps = int(rng.choice([1, 2]))
                        if n_steps == 1:
                            step_locations = [d_c]
                        else:
                            step_locations = [d_c * 0.5, d_c]
                    else:
                        n_steps = 0
                        step_locations = []

                    # h. Generate problem text representation
                    problem_representation = self._make_problem_text(K, rng)

                    # i. Compute gzip length
                    gzip_length = len(
                        gzip_module.compress(
                            problem_representation.encode("utf-8"), compresslevel=9
                        )
                    )

                    # Pre-draw rng check for MI non-monotonicity planting
                    rng_check = rng.random()

                    # j. Create Problem object
                    problem = Problem(
                        idx=problem_idx,
                        K=K,
                        D=D,
                        d_c=d_c,
                        gamma=gamma,
                        curve_type=curve_type,
                        complexity_level=complexity_level,
                        k_depth_correlation=k_depth_corr,
                        acc_low=acc_low,
                        acc_high=acc_high,
                        problem_representation=problem_representation,
                        gzip_length=gzip_length,
                        n_steps=n_steps,
                        step_locations=step_locations,
                        rng_check=rng_check,
                    )

                    # k. Simulate accuracy curves
                    self._simulate_accuracy(problem, rng)

                    # l. Simulate MI trajectories
                    self._simulate_mi(problem, rng)

                    # m. Append
                    self.problems.append(problem)
                    self.regime_indices[regime_key].append(problem_idx)
                    problem_idx += 1

        assert len(self.problems) == cfg.total_problems, (
            f"Generated {len(self.problems)} problems, expected {cfg.total_problems}"
        )
        return self

    def _make_problem_text(self, K, rng):
        """Generate a string whose gzip length is approximately proportional to K.

        Args:
            K: Target Kolmogorov complexity value.
            rng: numpy Generator for reproducibility.

        Returns:
            String representation of the problem.
        """
        base = f"prob:K={K:.2f},"
        # Random characters contribute ~1 byte each to gzip output
        n_random = max(1, int(K * 1.5))
        charset = list("abcdefghijklmnopqrstuvwxyz0123456789")
        random_part = "".join(rng.choice(charset, size=n_random))
        return base + random_part

    def _simulate_accuracy(self, problem, rng):
        """Simulate noisy accuracy curves for all model configurations.

        For each (model, temperature, budget) combination, generates
        config.samples_per_cell binary-outcome samples. Staircase problems
        have sharp transitions; smooth problems follow a sigmoid.

        Args:
            problem: Problem object (mutated in place).
            rng: numpy Generator for reproducibility.
        """
        cfg = self.config

        for model_name, model_cfg in cfg.model_configs.items():
            # accuracy_samples shape: [n_temps=3, n_budgets=8, n_samples=32]
            samples = np.zeros(
                (cfg.n_temperatures, cfg.n_budgets, cfg.samples_per_cell)
            )

            for t_idx, temp in enumerate(cfg.temperatures):
                for b_idx, T in enumerate(cfg.budget_array):
                    # Compute base accuracy at budget T
                    depth_ratio = T / problem.d_c
                    effective_ratio = depth_ratio ** model_cfg["depth_sensitivity"]

                    if problem.curve_type == "staircase":
                        if problem.n_steps == 1:
                            if T < problem.d_c:
                                acc_base = problem.acc_low
                            else:
                                acc_base = problem.acc_high
                        else:  # 2-step staircase
                            if T < problem.step_locations[0]:
                                acc_base = problem.acc_low
                            elif T < problem.step_locations[1]:
                                acc_base = (problem.acc_low + problem.acc_high) / 2.0
                            else:
                                acc_base = problem.acc_high
                    else:  # smooth sigmoid
                        exponent = -problem.gamma * (effective_ratio - 1.0)
                        # Clip exponent to avoid overflow
                        exponent = np.clip(exponent, -50.0, 50.0)
                        acc_base = problem.acc_low + (
                            problem.acc_high - problem.acc_low
                        ) / (1.0 + np.exp(exponent))

                    # Temperature-modulated noise
                    noise_std = model_cfg["noise_scale"] * temp

                    # Generate noisy accuracy samples
                    raw = acc_base + rng.normal(
                        0, noise_std, size=cfg.samples_per_cell
                    )
                    samples[t_idx, b_idx, :] = np.clip(
                        raw, cfg.accuracy_clip_min, cfg.accuracy_clip_max
                    )

            problem.accuracy_samples[model_name] = samples
            # accuracy_means shape: [n_temps=3, n_budgets=8]
            problem.accuracy_means[model_name] = samples.mean(axis=2)

    def _simulate_mi(self, problem, rng):
        """Simulate mutual information trajectories and plant non-monotonicity.

        MI is computed from accuracy: I(T) = log(2) - H_binary(acc(T)).
        Non-monotonicity is planted probabilistically per (problem, model)
        using the problem's pre-drawn _rng_check value.

        Args:
            problem: Problem object (mutated in place).
            rng: numpy Generator for reproducibility.
        """
        cfg = self.config

        for model_name, model_cfg in cfg.model_configs.items():
            # MI trajectories shape: [n_temps=3, n_budgets=8]
            mi = np.zeros((cfg.n_temperatures, cfg.n_budgets))
            acc = problem.accuracy_means[model_name]  # [n_temps, n_budgets]

            for t_idx in range(cfg.n_temperatures):
                for b_idx in range(cfg.n_budgets):
                    p = np.clip(acc[t_idx, b_idx], cfg.eps, 1.0 - cfg.eps)
                    # MI = log(2) - H_binary(p) in nats
                    h_bin = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
                    mi[t_idx, b_idx] = np.log(2.0) - h_bin

            # Plant MI non-monotonicity with model-specific probability
            problem.has_mi_nonmonotonicity[model_name] = False

            if problem._rng_check < model_cfg["mi_nonmonotone_probability"]:
                # Inject a dip at a random intermediate budget index
                dip_idx = int(rng.integers(1, cfg.n_budgets - 1))

                for t_idx in range(cfg.n_temperatures):
                    # Reduce MI at dip_idx by 20-50% of the local gain
                    local_gain = max(
                        mi[t_idx, dip_idx] - mi[t_idx, dip_idx - 1], 0.01
                    )
                    dip_magnitude = rng.uniform(0.2, 0.5) * local_gain
                    mi[t_idx, dip_idx] -= dip_magnitude

                    # Also back-propagate the dip into accuracy_means for consistency
                    # MI_dipped = log(2) - H(p_new), solve for p_new
                    mi_dipped = mi[t_idx, dip_idx]
                    # Approximate: lower MI means p closer to 0.5
                    # Use Newton step from current accuracy
                    p_current = acc[t_idx, dip_idx]
                    if mi_dipped < 0:
                        mi[t_idx, dip_idx] = 0.0  # floor at 0

                problem.has_mi_nonmonotonicity[model_name] = True

            problem.mi_trajectories[model_name] = mi

    def get_regime_problems(self, complexity, correlation):
        """Return list of Problem objects for a specific regime cell.

        Args:
            complexity: 'low' or 'high'
            correlation: 'correlated' or 'decorrelated'

        Returns:
            List of Problem objects in the specified regime.
        """
        key = (complexity, correlation)
        indices = self.regime_indices[key]
        return [self.problems[i] for i in indices]

    def get_ground_truth_elbows(self):
        """Return ground-truth critical depths for all problems.

        Returns:
            np.ndarray of shape [n_problems] with d_c values.
        """
        return np.array([p.d_c for p in self.problems])

    def get_gzip_lengths(self):
        """Return gzip compression lengths for all problems.

        Returns:
            np.ndarray of shape [n_problems] with gzip_length values.
        """
        return np.array([p.gzip_length for p in self.problems])

    def get_circuit_depths(self):
        """Return circuit depths D for all problems.

        Returns:
            np.ndarray of shape [n_problems] with D values.
        """
        return np.array([p.D for p in self.problems])


class GSM8KFeatures:
    """Loads GSM8K problems and extracts text-level features for ecological validation.

    Extracts gzip compression length, arithmetic step count, and text length
    from real math reasoning problems. No model inference — just text analysis.
    """

    def __init__(self, config):
        self.config = config
        self.gzip_lengths = None    # shape [500]
        self.step_counts = None     # shape [500]
        self.text_lengths = None    # shape [500]
        self.gzip_terciles = None   # shape [500], values {0, 1, 2}
        self.step_complexity = None  # shape [500], values {0, 1}

    def load(self):
        """Load GSM8K dataset and extract text features.

        Returns:
            self for method chaining.
        """
        from datasets import load_dataset

        cfg = self.config
        ds = load_dataset(
            "openai/gsm8k", "main", split=f"test[:{cfg.gsm8k_n_problems}]"
        )

        n = len(ds)
        gzip_lengths = np.zeros(n)
        step_counts = np.zeros(n)
        text_lengths = np.zeros(n)

        for i, example in enumerate(ds):
            question = example["question"]
            answer = example["answer"]

            # Gzip compression length of the question text
            gzip_lengths[i] = len(
                gzip_module.compress(
                    question.encode("utf-8"), compresslevel=cfg.gsm8k_gzip_level
                )
            )

            # Solution step count: count arithmetic operators in the answer
            operators = re.findall(r"[+\-*/]", answer)
            step_counts[i] = len(operators)

            # Text length of the answer
            text_lengths[i] = len(answer)

        self.gzip_lengths = gzip_lengths
        self.step_counts = step_counts
        self.text_lengths = text_lengths

        # Compute gzip terciles
        tercile_edges = np.percentile(gzip_lengths, [33.33, 66.67])
        self.gzip_terciles = np.digitize(gzip_lengths, tercile_edges)  # {0, 1, 2}

        # Binary complexity split at median step count
        median_steps = np.median(step_counts)
        self.step_complexity = (step_counts > median_steps).astype(int)  # {0, 1}

        return self

    def compute_correlation_matrix(self):
        """Compute correlation statistics between text features.

        Returns:
            Dict with Pearson/Spearman correlations and p-values.
        """
        r_gzip_steps, p_gzip_steps = scipy_stats.pearsonr(
            self.gzip_lengths, self.step_counts
        )
        rho_gzip_steps, p_rho = scipy_stats.spearmanr(
            self.gzip_lengths, self.step_counts
        )
        r_gzip_textlen, _ = scipy_stats.pearsonr(
            self.gzip_lengths, self.text_lengths
        )

        return {
            "pearson_gzip_vs_steps": float(r_gzip_steps),
            "pearson_pvalue": float(p_gzip_steps),
            "spearman_gzip_vs_steps": float(rho_gzip_steps),
            "spearman_pvalue": float(p_rho),
            "pearson_gzip_vs_textlen": float(r_gzip_textlen),
            "n_problems": len(self.gzip_lengths),
        }
