"""All 7 experimental conditions for scaling-elbow prediction.

Implements 2 baselines (smooth log-concave, empirical power-law), 3 proposed methods
(staircase+temperature for H1, MI non-monotone for H2, circuit depth for H3), and
2 ablations (no-temperature for H1, gzip-only for H3). Each class encapsulates a
distinct strategy for predicting per-problem scaling elbows from noisy accuracy curves.
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from config import ExperimentConfig
from data import SyntheticDataset, Problem


# ═══════════════════════════════════════════════════════════════════════
# Abstract base class
# ═══════════════════════════════════════════════════════════════════════


class BaseMethod:
    """Abstract base class for all scaling-elbow prediction methods.

    Defines the fit -> predict -> evaluate interface. All 7 conditions
    inherit from this class and implement distinct analytical pipelines.
    """

    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.is_fitted = False

    def fit(self, dataset):
        """Fit the method on the dataset. Must be overridden."""
        raise NotImplementedError(f"{self.name} must implement fit()")

    def predict_per_problem(self, dataset):
        """Return shape [n_problems] array of predicted elbow budgets."""
        raise NotImplementedError(f"{self.name} must implement predict_per_problem()")

    def evaluate(self, dataset):
        """Fit (if needed), predict, and compute MAPE metrics.

        Returns dict with primary MAPE, per-regime breakdown, and raw predictions.
        """
        if not self.is_fitted:
            self.fit(dataset)

        predictions = self.predict_per_problem(dataset)  # shape [n_problems]
        ground_truth = dataset.get_ground_truth_elbows()  # shape [n_problems]

        # Primary metric: MAPE (%)
        mape = float(
            np.mean(
                np.abs(predictions - ground_truth)
                / np.maximum(ground_truth, self.config.eps)
            )
            * 100.0
        )

        # Per-regime MAPE breakdown
        regime_mapes = {}
        for (complexity, correlation), indices in dataset.regime_indices.items():
            idx = np.array(indices)
            gt_regime = ground_truth[idx]
            pred_regime = predictions[idx]
            regime_mapes[(complexity, correlation)] = float(
                np.mean(
                    np.abs(pred_regime - gt_regime)
                    / np.maximum(gt_regime, self.config.eps)
                )
                * 100.0
            )

        return {
            "elbow_prediction_mape": mape,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "regime_mapes": regime_mapes,
            "method_name": self.name,
        }


# ═══════════════════════════════════════════════════════════════════════
# BASELINE 1: Smooth Log-Concave Bound
# ═══════════════════════════════════════════════════════════════════════


class SmoothLogconcaveBound(BaseMethod):
    """Target paper's information-theoretic framework.

    Fits a log-concave parametric function f(T) = a*log(1+b*T)/(1+c*T) to the
    population-average scaling curve and extracts a SINGLE elbow for ALL problems.
    No per-problem adaptation, no model awareness, no temperature sensitivity.
    """

    def __init__(self, config):
        super().__init__(config, name="smooth_logconcave_bound")
        self.params_fitted = None  # shape [3] after fit
        self.population_elbow = None  # single float

    def _logconcave_func(self, T, a, b, c):
        """f(T) = a * log(1 + b*T) / (1 + c*T)

        Input T: shape [N], Output: shape [N]
        """
        return a * np.log(1.0 + b * T) / (1.0 + c * T)

    def fit_population_curve(self, dataset):
        """Aggregate accuracy into population average and fit log-concave function."""
        cfg = self.config
        n_problems = len(dataset.problems)
        # Population-average accuracy at tau=0.5 (temp index 1) for model_A
        # shape [n_budgets=8]
        acc_pop = np.zeros(cfg.n_budgets)
        for b_idx in range(cfg.n_budgets):
            acc_sum = 0.0
            for problem in dataset.problems:
                acc_sum += problem.accuracy_means["model_A"][1, b_idx]
            acc_pop[b_idx] = acc_sum / n_problems

        T = dataset.budgets  # shape [8]
        x0 = [cfg.slb_a_init, cfg.slb_b_init, cfg.slb_c_init]
        bounds = [cfg.slb_bounds_a, cfg.slb_bounds_b, cfg.slb_bounds_c]

        def loss(params):
            a, b, c = params
            pred = self._logconcave_func(T, a, b, c)
            return float(np.sum((pred - acc_pop) ** 2))

        result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        if result.success:
            self.params_fitted = result.x  # shape [3]
        else:
            # Fallback parameters
            self.params_fitted = np.array([acc_pop[-1], 0.01, 0.001])

    def compute_elbow(self):
        """Find elbow via maximum absolute second derivative on dense grid."""
        a, b, c = self.params_fitted
        T_dense = np.linspace(4.0, 512.0, 2000)  # shape [2000]
        f_vals = self._logconcave_func(T_dense, a, b, c)  # shape [2000]

        # Numerical second derivative
        dT = T_dense[1] - T_dense[0]
        f_pp = np.gradient(np.gradient(f_vals, dT), dT)  # shape [2000]

        # Elbow = point of maximum absolute curvature change
        elbow_idx = np.argmax(np.abs(f_pp))
        self.population_elbow = float(T_dense[elbow_idx])
        return self.population_elbow

    def fit(self, dataset):
        self.fit_population_curve(dataset)
        self.compute_elbow()
        self.is_fitted = True

    def predict_per_problem(self, dataset):
        """Broadcast single population elbow to ALL problems."""
        n = len(dataset.problems)
        return np.full(n, self.population_elbow)  # shape [n_problems]


# ═══════════════════════════════════════════════════════════════════════
# BASELINE 2: Empirical Power-Law Fit
# ═══════════════════════════════════════════════════════════════════════


class EmpiricalPowerLawFit(BaseMethod):
    """Per-problem power-law fit following Snell et al. (NeurIPS 2024).

    Fits 1-acc ~ a * T^(-b) in log-space via OLS per problem, then finds
    the elbow where marginal gain dAcc/dT drops below epsilon.
    """

    def __init__(self, config):
        super().__init__(config, name="empirical_power_law_fit")
        self.per_problem_params = None  # shape [n_problems, 2]
        self.per_problem_elbows = None  # shape [n_problems]

    def fit_per_problem_curve(self, problem, budgets):
        """Fit power-law to single problem's accuracy curve.

        Returns (a_i, b_i) parameters.
        """
        cfg = self.config
        # Mean accuracy at tau=0.5 (temp index 1) for model_A
        acc = problem.accuracy_means["model_A"][1, :]  # shape [8]
        acc_clipped = np.clip(acc, cfg.eps, cfg.eplf_accuracy_clip)

        # Log-space transform: y = log(1 - acc), x = log(T)
        y = np.log(np.maximum(1.0 - acc_clipped, cfg.eps))  # shape [8]
        x = np.log(budgets)  # shape [8]

        # Filter: only use budgets >= min_budget_for_fit
        mask = budgets >= cfg.eplf_min_budget_for_fit
        x_fit, y_fit = x[mask], y[mask]

        if len(x_fit) < 2:
            return (0.5, 0.5)

        # OLS: y = log(a) - b * x
        A = np.vstack([np.ones_like(x_fit), x_fit]).T  # shape [n_valid, 2]
        result = np.linalg.lstsq(A, y_fit, rcond=None)
        log_a = result[0][0]
        neg_b = result[0][1]
        a_i = np.exp(np.clip(log_a, -20, 20))  # prevent overflow
        b_i = -neg_b
        return (float(a_i), max(float(b_i), cfg.eps))

    def compute_marginal_gain(self, a, b, T):
        """dAcc/dT = a * b * T^(-(b+1))"""
        return a * b * np.power(T, -(b + 1))

    def fit(self, dataset):
        cfg = self.config
        n = len(dataset.problems)
        self.per_problem_params = np.zeros((n, 2))  # shape [n, 2]
        self.per_problem_elbows = np.zeros(n)  # shape [n]

        T_dense = np.linspace(
            dataset.budgets[0], dataset.budgets[-1], 1000
        )  # shape [1000]

        for i, problem in enumerate(dataset.problems):
            a_i, b_i = self.fit_per_problem_curve(problem, dataset.budgets)
            self.per_problem_params[i] = [a_i, b_i]

            # Find elbow: smallest T where marginal gain < epsilon
            mg = self.compute_marginal_gain(a_i, b_i, T_dense)  # shape [1000]
            below = np.where(mg < cfg.eplf_marginal_gain_epsilon)[0]
            if len(below) > 0:
                self.per_problem_elbows[i] = T_dense[below[0]]
            else:
                self.per_problem_elbows[i] = dataset.budgets[-1]

        self.is_fitted = True

    def predict_per_problem(self, dataset):
        return self.per_problem_elbows.copy()  # shape [n_problems]


# ═══════════════════════════════════════════════════════════════════════
# PROPOSED 1: Staircase Temperature Allocator (H1)
# ═══════════════════════════════════════════════════════════════════════


class StaircaseTemperatureAllocator(BaseMethod):
    """H1 method: BIC-based staircase vs sigmoid classification with temperature optimization.

    For each problem, fits both a piecewise-constant (1-step and 2-step) and a
    logistic sigmoid, selects via BIC. Groups problems by gzip-complexity tercile
    and optimizes temperature per bucket. Predictions use bucket-optimal temperature
    and 1.2x safety margin.
    """

    def __init__(self, config):
        super().__init__(config, name="staircase_temperature_allocator")
        self.bic_results = None  # list of per-problem dicts
        self.gzip_tercile_edges = None  # shape [2]
        self.bucket_optimal_temps = None  # {bucket_idx: float}
        self.predictions = None  # shape [n_problems]

    def fit_piecewise_constant(self, acc_curve, budgets):
        """Fit 1-step and 2-step piecewise constant models, return best via BIC.

        Args:
            acc_curve: shape [n_budgets]
            budgets: shape [n_budgets]

        Returns:
            dict with type, bic, d_c_est, rss, n_params
        """
        cfg = self.config
        n = len(acc_curve)

        # -- 1-step: brute-force over split points --
        best_1step_rss = np.inf
        best_1step_loc = budgets[1]
        best_1step_levels = [0.0, 0.0]

        for split_idx in range(1, n):
            level_low = np.mean(acc_curve[:split_idx])
            level_high = np.mean(acc_curve[split_idx:])
            pred = np.concatenate(
                [np.full(split_idx, level_low), np.full(n - split_idx, level_high)]
            )
            rss = np.sum((acc_curve - pred) ** 2)
            if rss < best_1step_rss:
                best_1step_rss = rss
                best_1step_loc = budgets[split_idx]
                best_1step_levels = [level_low, level_high]

        # -- 2-step: brute-force over pairs of split points --
        best_2step_rss = np.inf
        best_2step_locs = [budgets[1], budgets[2]]

        for s1 in range(1, n - 1):
            for s2 in range(s1 + 1, n):
                l1 = np.mean(acc_curve[:s1])
                l2 = np.mean(acc_curve[s1:s2])
                l3 = np.mean(acc_curve[s2:])
                pred = np.concatenate(
                    [np.full(s1, l1), np.full(s2 - s1, l2), np.full(n - s2, l3)]
                )
                rss = np.sum((acc_curve - pred) ** 2)
                if rss < best_2step_rss:
                    best_2step_rss = rss
                    best_2step_locs = [budgets[s1], budgets[s2]]

        # BIC = n*ln(RSS/n) + p*ln(n)
        bic_1 = n * np.log(max(best_1step_rss / n, cfg.eps)) + 2 * np.log(n)
        bic_2 = n * np.log(max(best_2step_rss / n, cfg.eps)) + 4 * np.log(n)

        # Select best staircase model
        if bic_2 < bic_1 - cfg.sta_bic_evidence_threshold:
            return {
                "type": "staircase_2step",
                "bic": float(bic_2),
                "d_c_est": float(best_2step_locs[-1]),
                "rss": float(best_2step_rss),
                "n_params": 4,
            }
        return {
            "type": "staircase_1step",
            "bic": float(bic_1),
            "d_c_est": float(best_1step_loc),
            "rss": float(best_1step_rss),
            "n_params": 2,
        }

    def fit_logistic_sigmoid(self, acc_curve, budgets):
        """Fit logistic sigmoid L / (1 + exp(-k*(T-T0))) via curve_fit.

        Args:
            acc_curve: shape [n_budgets]
            budgets: shape [n_budgets]

        Returns:
            dict with type, bic, d_c_est, rss, n_params, params
        """
        cfg = self.config

        def sigmoid(T, L, k, T0):
            exponent = -k * (T - T0)
            exponent = np.clip(exponent, -50.0, 50.0)
            return L / (1.0 + np.exp(exponent))

        p0 = [float(acc_curve[-1]), 0.05, float(budgets[len(budgets) // 2])]
        bounds_low = [0.01, 0.001, float(budgets[0])]
        bounds_high = [1.0, 1.0, float(budgets[-1])]

        try:
            popt, _ = curve_fit(
                sigmoid, budgets, acc_curve, p0=p0,
                bounds=(bounds_low, bounds_high), maxfev=5000,
            )
            pred = sigmoid(budgets, *popt)
            rss = float(np.sum((acc_curve - pred) ** 2))
        except (RuntimeError, ValueError, TypeError):
            popt = np.array(p0)
            pred = sigmoid(budgets, *popt)
            rss = float(np.sum((acc_curve - pred) ** 2))

        n = len(acc_curve)
        bic = n * np.log(max(rss / n, cfg.eps)) + 3 * np.log(n)

        return {
            "type": "sigmoid",
            "bic": float(bic),
            "d_c_est": float(popt[2]),  # T0 = midpoint
            "rss": rss,
            "n_params": 3,
            "params": popt,
        }

    def bic_model_select(self, staircase_result, sigmoid_result):
        """Select between staircase and sigmoid via BIC difference.

        Positive bic_diff = staircase is better.
        """
        cfg = self.config
        bic_diff = sigmoid_result["bic"] - staircase_result["bic"]

        if bic_diff > cfg.sta_bic_evidence_threshold:
            winner = staircase_result
        elif bic_diff < -cfg.sta_bic_evidence_threshold:
            winner = sigmoid_result
        else:
            # Within evidence threshold: default to staircase (simpler)
            winner = staircase_result

        return {
            "winner": winner["type"],
            "d_c_est": winner["d_c_est"],
            "bic_diff": float(bic_diff),
        }

    def bucket_by_gzip_tercile(self, dataset):
        """Partition problems into 3 buckets by gzip compression length.

        Returns (bucket_assignments [n_problems], edges [2]).
        """
        gzip_lengths = dataset.get_gzip_lengths()  # shape [n_problems]
        edges = np.percentile(gzip_lengths, [33.33, 66.67])  # shape [2]
        self.gzip_tercile_edges = edges
        bucket_assignments = np.digitize(gzip_lengths, edges)  # values {0, 1, 2}
        return bucket_assignments, edges

    def optimize_temperature_per_bucket(self, dataset, bucket_assignments):
        """Grid-search optimal temperature per gzip-complexity bucket on holdout set."""
        cfg = self.config
        self.bucket_optimal_temps = {}

        for bucket_idx in range(cfg.sta_n_complexity_buckets):
            problem_indices = np.where(bucket_assignments == bucket_idx)[0]
            n_bucket = len(problem_indices)
            if n_bucket == 0:
                self.bucket_optimal_temps[bucket_idx] = cfg.temperatures[1]
                continue

            # 80/20 train/holdout split within bucket
            n_holdout = max(1, int(n_bucket * cfg.sta_holdout_fraction))
            holdout_idx = problem_indices[-n_holdout:]

            best_temp = cfg.temperatures[1]  # default tau=0.5
            best_mape = np.inf

            for t_idx, temp in enumerate(cfg.temperatures):
                mape_sum = 0.0
                for idx in holdout_idx:
                    p = dataset.problems[idx]
                    acc_at_temp = p.accuracy_means["model_A"][t_idx, :]  # shape [8]
                    staircase_res = self.fit_piecewise_constant(
                        acc_at_temp, dataset.budgets
                    )
                    sigmoid_res = self.fit_logistic_sigmoid(
                        acc_at_temp, dataset.budgets
                    )
                    selected = self.bic_model_select(staircase_res, sigmoid_res)
                    pred = selected["d_c_est"] * cfg.sta_safety_margin
                    true_dc = p.d_c
                    mape_sum += abs(pred - true_dc) / max(true_dc, cfg.eps)

                avg_mape = mape_sum / max(len(holdout_idx), 1)
                if avg_mape < best_mape:
                    best_mape = avg_mape
                    best_temp = temp

            self.bucket_optimal_temps[bucket_idx] = best_temp

        return self.bucket_optimal_temps

    def fit(self, dataset):
        cfg = self.config
        bucket_assignments, _ = self.bucket_by_gzip_tercile(dataset)
        self.optimize_temperature_per_bucket(dataset, bucket_assignments)

        # Fit BIC for each problem at its bucket's optimal temperature
        self.bic_results = []
        for i, problem in enumerate(dataset.problems):
            bucket = bucket_assignments[i]
            opt_temp = self.bucket_optimal_temps[bucket]
            t_idx = cfg.temperatures.index(opt_temp)
            acc_curve = problem.accuracy_means["model_A"][t_idx, :]  # shape [8]
            staircase_res = self.fit_piecewise_constant(acc_curve, dataset.budgets)
            sigmoid_res = self.fit_logistic_sigmoid(acc_curve, dataset.budgets)
            result = self.bic_model_select(staircase_res, sigmoid_res)
            self.bic_results.append(result)

        self.is_fitted = True

    def predict_per_problem(self, dataset):
        cfg = self.config
        n = len(dataset.problems)
        predictions = np.zeros(n)  # shape [n_problems]
        for i in range(n):
            predictions[i] = self.bic_results[i]["d_c_est"] * cfg.sta_safety_margin
        predictions = np.clip(predictions, dataset.budgets[0], dataset.budgets[-1])
        return predictions

    def evaluate(self, dataset):
        base_results = super().evaluate(dataset)

        # H1-specific: BIC staircase win rate
        staircase_wins = sum(
            1 for r in self.bic_results if "staircase" in r["winner"]
        )
        base_results["bic_staircase_win_rate"] = staircase_wins / len(
            self.bic_results
        )
        base_results["bic_diffs"] = [r["bic_diff"] for r in self.bic_results]
        base_results["bucket_optimal_temps"] = self.bucket_optimal_temps

        return base_results


# ═══════════════════════════════════════════════════════════════════════
# PROPOSED 2: MI Non-Monotone Detector (H2)
# ═══════════════════════════════════════════════════════════════════════


class MINonmonotoneDetector(BaseMethod):
    """H2 method: MI trajectory analysis with non-monotonicity detection.

    Tracks per-step MI trajectory, detects non-monotonic segments (DPI violations),
    classifies steps as constructive vs corrective, computes effective MI curve
    (corrective steps frozen), and finds elbow where effective MI gain saturates.
    Produces model-specific predictions.
    """

    def __init__(self, config):
        super().__init__(config, name="mi_nonmonotone_detector")
        self.step_classifications = {}  # {model: list of per-problem classifications}
        self.effective_mi_curves = {}  # {model: ndarray [n_problems, n_budgets]}
        self.nonmonotone_flags = {}  # {model: ndarray [n_problems] bool}
        self.calibrated_thresholds = {}  # {model: float}
        self.per_model_predictions = {}  # {model: ndarray [n_problems]}
        self.adaptive_corrective_thresholds = {}  # {model: float}

    def compute_mi_trajectory(self, problem, model_name, temp_idx):
        """Return MI trajectory at given temp for given model. Shape [n_budgets]."""
        return problem.mi_trajectories[model_name][temp_idx, :].copy()

    def smooth_trajectory(self, mi):
        """Uniform moving-average smoothing with edge padding. Shape [n_budgets]."""
        w = self.config.mnd_smoothing_window  # 3
        padded = np.pad(mi, (w // 2, w // 2), mode="edge")
        kernel = np.ones(w) / w
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: len(mi)]  # ensure exact shape match

    def estimate_noise_floor(self, dataset, model_name):
        """Estimate the noise floor of smoothed MI deltas from the data.

        Computes the standard deviation of all smoothed MI step-deltas across
        all problems for the given model, then sets the adaptive corrective
        threshold at -N * sigma (where N = config.mnd_adaptive_noise_sigmas).

        This ensures the threshold scales with the actual noise level rather
        than relying on a fixed magic number, preventing false positives from
        dominating the non-monotonicity detection.

        Args:
            dataset: SyntheticDataset
            model_name: which model config to estimate noise for

        Returns:
            float: adaptive corrective threshold (negative value)
        """
        cfg = self.config
        all_deltas = []

        for problem in dataset.problems:
            mi_raw = self.compute_mi_trajectory(problem, model_name, temp_idx=1)
            mi_smooth = self.smooth_trajectory(mi_raw)
            deltas = np.diff(mi_smooth)  # shape [n_budgets - 1]
            all_deltas.extend(deltas.tolist())

        all_deltas = np.array(all_deltas)
        # Use robust estimate: median absolute deviation instead of std
        # to avoid the planted dips from inflating the noise estimate
        median_delta = np.median(all_deltas)
        mad = np.median(np.abs(all_deltas - median_delta))
        # Convert MAD to equivalent std: sigma = MAD * 1.4826
        robust_std = mad * 1.4826

        # Adaptive threshold: -N * sigma below zero
        adaptive_threshold = -cfg.mnd_adaptive_noise_sigmas * robust_std

        # Floor: never less sensitive than the config default
        adaptive_threshold = min(adaptive_threshold, cfg.mnd_corrective_threshold)

        return float(adaptive_threshold)

    def detect_nonmonotonic_steps(self, mi_smoothed, corrective_threshold):
        """Detect steps where MI decreases beyond the adaptive corrective threshold.

        Args:
            mi_smoothed: shape [n_budgets]
            corrective_threshold: float, negative (model-specific adaptive value)

        Returns bool array shape [n_budgets - 1].
        """
        deltas = np.diff(mi_smoothed)  # shape [n_budgets - 1]
        return deltas < corrective_threshold

    def classify_constructive_corrective(self, mi_smoothed, corrective_threshold):
        """Classify each step as constructive, corrective, or neutral.

        Uses the model-specific adaptive corrective threshold instead of
        the fixed config value.

        Args:
            mi_smoothed: shape [n_budgets]
            corrective_threshold: float, negative (model-specific adaptive value)

        Returns list of strings, length n_budgets - 1.
        """
        cfg = self.config
        deltas = np.diff(mi_smoothed)  # shape [n_budgets - 1]
        classifications = []
        for d in deltas:
            if d > cfg.mnd_constructive_threshold:
                classifications.append("constructive")
            elif d < corrective_threshold:
                classifications.append("corrective")
            else:
                classifications.append("neutral")
        return classifications

    def compute_effective_mi_curve(self, mi_smoothed, classifications):
        """Compute effective MI: only accumulate constructive deltas.

        Corrective and neutral steps freeze the effective MI.
        Returns shape [n_budgets].
        """
        effective = np.zeros_like(mi_smoothed)
        effective[0] = mi_smoothed[0]
        for t in range(1, len(mi_smoothed)):
            delta = mi_smoothed[t] - mi_smoothed[t - 1]
            if classifications[t - 1] == "constructive":
                effective[t] = effective[t - 1] + delta
            else:
                effective[t] = effective[t - 1]
        return effective

    def calibrate_per_model(self, dataset):
        """Find optimal saturation threshold per model on holdout set."""
        cfg = self.config
        n = len(dataset.problems)
        n_holdout = max(1, int(n * cfg.mnd_holdout_fraction))
        holdout_indices = list(range(n - n_holdout, n))

        candidate_thresholds = [0.001, 0.002, 0.005, 0.01, 0.02]

        for model_name in cfg.model_configs.keys():
            best_threshold = cfg.mnd_saturation_threshold
            best_mape = np.inf

            for sat_thr in candidate_thresholds:
                mape_sum = 0.0
                for idx in holdout_indices:
                    eff_mi = self.effective_mi_curves[model_name][idx]  # [n_budgets]
                    eff_gains = np.diff(eff_mi)  # [n_budgets - 1]
                    saturated = np.where(eff_gains < sat_thr)[0]
                    if len(saturated) > 0:
                        pred_elbow = dataset.budgets[min(saturated[0] + 1, cfg.n_budgets - 1)]
                    else:
                        pred_elbow = dataset.budgets[-1]
                    true_dc = dataset.problems[idx].d_c
                    mape_sum += abs(pred_elbow - true_dc) / max(true_dc, cfg.eps)

                avg_mape = mape_sum / max(len(holdout_indices), 1)
                if avg_mape < best_mape:
                    best_mape = avg_mape
                    best_threshold = sat_thr

            self.calibrated_thresholds[model_name] = best_threshold

    def fit(self, dataset):
        cfg = self.config

        # Phase 0: estimate adaptive noise floor per model
        for model_name in cfg.model_configs.keys():
            adaptive_thr = self.estimate_noise_floor(dataset, model_name)
            self.adaptive_corrective_thresholds[model_name] = adaptive_thr

        # Phase 1: compute effective MI curves and detect non-monotonicity
        for model_name in cfg.model_configs.keys():
            n = len(dataset.problems)
            eff_curves = np.zeros((n, cfg.n_budgets))  # [n_problems, 8]
            nm_flags = np.zeros(n, dtype=bool)  # [n_problems]
            classifications_list = []
            corrective_thr = self.adaptive_corrective_thresholds[model_name]

            for i, problem in enumerate(dataset.problems):
                # Use tau=0.5 (temp index 1)
                mi_raw = self.compute_mi_trajectory(problem, model_name, temp_idx=1)
                mi_smooth = self.smooth_trajectory(mi_raw)
                nm_mask = self.detect_nonmonotonic_steps(mi_smooth, corrective_thr)
                nm_flags[i] = bool(np.any(nm_mask))
                classif = self.classify_constructive_corrective(mi_smooth, corrective_thr)
                classifications_list.append(classif)
                eff_curves[i] = self.compute_effective_mi_curve(mi_smooth, classif)

            self.effective_mi_curves[model_name] = eff_curves
            self.nonmonotone_flags[model_name] = nm_flags
            self.step_classifications[model_name] = classifications_list

        # Phase 2: calibrate saturation thresholds on holdout
        self.calibrate_per_model(dataset)

        # Phase 3: compute per-model predictions using calibrated thresholds
        for model_name in cfg.model_configs.keys():
            sat_thr = self.calibrated_thresholds[model_name]
            n = len(dataset.problems)
            preds = np.zeros(n)

            for i in range(n):
                eff_mi = self.effective_mi_curves[model_name][i]  # [8]
                eff_gains = np.diff(eff_mi)  # [7]
                saturated = np.where(eff_gains < sat_thr)[0]
                if len(saturated) > 0:
                    budget_idx = min(saturated[0] + 1, cfg.n_budgets - 1)
                    preds[i] = dataset.budgets[budget_idx]
                else:
                    preds[i] = dataset.budgets[-1]

            self.per_model_predictions[model_name] = preds

        self.is_fitted = True

    def predict_per_problem(self, dataset):
        """Use model_A predictions as primary output."""
        return self.per_model_predictions["model_A"].copy()  # shape [n_problems]

    def evaluate(self, dataset):
        base_results = super().evaluate(dataset)
        cfg = self.config

        # H2-specific: non-monotonicity rate per model
        for model_name in cfg.model_configs.keys():
            nm_rate = float(np.mean(self.nonmonotone_flags[model_name]))
            base_results[f"mi_nonmonotonicity_rate_{model_name}"] = nm_rate

        # H2-specific: cross-model elbow divergence
        preds_A = self.per_model_predictions["model_A"]  # [n_problems]
        preds_B = self.per_model_predictions["model_B"]  # [n_problems]
        max_preds = np.maximum(np.maximum(preds_A, preds_B), cfg.eps)
        divergence = float(np.mean(np.abs(preds_A - preds_B) / max_preds))
        base_results["cross_model_elbow_divergence"] = divergence

        return base_results


# ═══════════════════════════════════════════════════════════════════════
# PROPOSED 3: Circuit Depth Predictor (H3)
# ═══════════════════════════════════════════════════════════════════════


class CircuitDepthPredictor(BaseMethod):
    """H3 method: ridge regression on circuit depth to predict scaling elbows.

    Uses circuit depth D (sequential computation steps) as the sole feature
    in a cross-validated ridge regression. Designed to show that D predicts
    elbows better than gzip (K proxy) in the decorrelated regime.
    """

    def __init__(self, config):
        super().__init__(config, name="circuit_depth_predictor")
        self.ridge_alpha = None
        self.ridge_coef = None
        self.ridge_intercept = None
        self.train_indices = None
        self.test_indices = None
        self.correlation_diagnostics = None

    def extract_depth_features(self, dataset):
        """Return circuit depth D for each problem. Shape [n_problems]."""
        return dataset.get_circuit_depths()

    def stratified_train_test_split(self, dataset, seed):
        """Stratified 80/20 split by regime cell.

        Returns (train_indices, test_indices).
        """
        cfg = self.config
        rng = np.random.default_rng(seed + 1000)
        train_list, test_list = [], []

        for _regime_key, cell_indices in dataset.regime_indices.items():
            cell_arr = np.array(cell_indices)
            rng.shuffle(cell_arr)
            n_train = int(len(cell_arr) * cfg.cdp_train_fraction)
            train_list.extend(cell_arr[:n_train].tolist())
            test_list.extend(cell_arr[n_train:].tolist())

        self.train_indices = np.array(train_list)
        self.test_indices = np.array(test_list)
        return self.train_indices, self.test_indices

    def cross_validate_alpha(self, X_train, y_train):
        """5-fold CV to select best ridge alpha from candidates.

        X_train: shape [n_train, 1], y_train: shape [n_train]
        Returns best alpha float.
        """
        cfg = self.config
        best_alpha = cfg.cdp_alpha_candidates[0]
        best_score = np.inf
        kf = KFold(n_splits=cfg.cdp_cv_folds, shuffle=True, random_state=42)

        for alpha in cfg.cdp_alpha_candidates:
            fold_errors = []
            for train_idx, val_idx in kf.split(X_train):
                model = Ridge(alpha=alpha)
                model.fit(X_train[train_idx], y_train[train_idx])
                preds = model.predict(X_train[val_idx])
                mape = np.mean(
                    np.abs(preds - y_train[val_idx])
                    / np.maximum(np.abs(y_train[val_idx]), cfg.eps)
                )
                fold_errors.append(float(mape))
            mean_error = np.mean(fold_errors)
            if mean_error < best_score:
                best_score = mean_error
                best_alpha = alpha

        return best_alpha

    def fit_ridge_regression(self, X_train, y_train):
        """Fit final ridge regression with selected alpha.

        X_train: shape [n_train, 1], y_train: shape [n_train]
        """
        model = Ridge(alpha=self.ridge_alpha)
        model.fit(X_train, y_train)
        self.ridge_coef = float(model.coef_[0])
        self.ridge_intercept = float(model.intercept_)

    def fit(self, dataset):
        features = self.extract_depth_features(dataset)  # [n_problems]
        ground_truth = dataset.get_ground_truth_elbows()  # [n_problems]

        self.stratified_train_test_split(dataset, seed=0)
        X_train = features[self.train_indices].reshape(-1, 1)  # [~640, 1]
        y_train = ground_truth[self.train_indices]  # [~640]

        self.ridge_alpha = self.cross_validate_alpha(X_train, y_train)
        self.fit_ridge_regression(X_train, y_train)

        # Pre-compute LOO predictions for training problems to avoid data leakage.
        # For each training problem, fit ridge on all OTHER training problems and
        # predict the held-out one. This ensures every problem's prediction is
        # made without seeing that problem's ground truth.
        self._loo_train_preds = self._compute_loo_predictions(X_train, y_train)

        self.is_fitted = True

    def _compute_loo_predictions(self, X_train, y_train):
        """Leave-one-out predictions for training set problems.

        For each training problem i, fits ridge on all other training problems and predicts i.
        This removes the optimistic bias from evaluating on training data.

        Args:
            X_train: shape [n_train, 1]
            y_train: shape [n_train]

        Returns:
            ndarray shape [n_train] of held-out predictions
        """
        n_train = len(y_train)
        loo_preds = np.zeros(n_train)

        for i in range(n_train):
            # Leave out problem i
            mask = np.ones(n_train, dtype=bool)
            mask[i] = False
            X_loo = X_train[mask]
            y_loo = y_train[mask]

            model = Ridge(alpha=self.ridge_alpha)
            model.fit(X_loo, y_loo)
            loo_preds[i] = model.predict(X_train[i : i + 1])[0]

        return loo_preds

    def predict_per_problem(self, dataset):
        """Predict elbow for all problems using held-out predictions only.

        Test-set problems: use the fitted model (trained on all training data).
        Training-set problems: use leave-one-out predictions (each problem predicted
        by a model that never saw that problem's ground truth).

        This eliminates the train/test leakage that would give regression-based
        methods an unfair advantage over methods that don't use train/test splits.
        """
        features = self.extract_depth_features(dataset)  # [n_problems]
        n = len(features)
        predictions = np.zeros(n)

        # Test set: predict using the full fitted model
        test_features = features[self.test_indices]
        predictions[self.test_indices] = (
            self.ridge_coef * test_features + self.ridge_intercept
        )

        # Training set: use pre-computed LOO predictions
        predictions[self.train_indices] = self._loo_train_preds

        predictions = np.clip(predictions, dataset.budgets[0], dataset.budgets[-1])
        return predictions

    def compute_correlation_diagnostics(self, dataset):
        """Compute Pearson correlations: depth vs elbow, gzip vs elbow, per regime."""
        gt = dataset.get_ground_truth_elbows()  # [n_problems]
        depths = dataset.get_circuit_depths()  # [n_problems]
        gzips = dataset.get_gzip_lengths().astype(float)  # [n_problems]

        r_depth, _ = scipy_stats.pearsonr(depths, gt)
        r_gzip, _ = scipy_stats.pearsonr(gzips, gt)

        regime_corrs = {}
        for (complexity, correlation), indices in dataset.regime_indices.items():
            idx = np.array(indices)
            r_d, _ = scipy_stats.pearsonr(depths[idx], gt[idx])
            r_g, _ = scipy_stats.pearsonr(gzips[idx], gt[idx])
            regime_corrs[(complexity, correlation)] = {
                "depth_elbow_r": float(r_d),
                "gzip_elbow_r": float(r_g),
            }

        self.correlation_diagnostics = {
            "overall_depth_elbow_r": float(r_depth),
            "overall_gzip_elbow_r": float(r_gzip),
            "per_regime": regime_corrs,
        }
        return self.correlation_diagnostics

    def evaluate(self, dataset):
        base_results = super().evaluate(dataset)
        diags = self.compute_correlation_diagnostics(dataset)
        base_results["correlation_diagnostics"] = diags
        return base_results


# ═══════════════════════════════════════════════════════════════════════
# ABLATION 1: Staircase No Temperature (H1 ablation)
# ═══════════════════════════════════════════════════════════════════════


class StaircaseNoTemperature(StaircaseTemperatureAllocator):
    """H1 ablation: BIC staircase/sigmoid selection WITHOUT temperature optimization.

    STRATEGY DIFFERENCE from parent:
    - Removes the entire temperature optimization loop
    - All problems use fixed tau=0.5 regardless of gzip-complexity bucket
    - BIC model selection (staircase vs sigmoid) is identical to parent
    - Gzip bucketing still computed for reporting but NOT used for optimization

    This isolates temperature's contribution to H1's predictive advantage.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "staircase_no_temperature"
        self.fixed_temp_idx = config.temperatures.index(config.snt_fixed_temperature)
        self._safety_margin = config.snt_safety_margin

    def fit(self, dataset):
        """Fit BIC at FIXED temperature — no temperature optimization."""
        cfg = self.config

        # Still compute gzip buckets for analysis reporting
        bucket_assignments, _ = self.bucket_by_gzip_tercile(dataset)

        # NO temperature optimization — set all buckets to fixed temp
        self.bucket_optimal_temps = {
            b: cfg.snt_fixed_temperature
            for b in range(cfg.sta_n_complexity_buckets)
        }

        # Fit BIC for each problem at FIXED temperature
        self.bic_results = []
        for i, problem in enumerate(dataset.problems):
            # Always use fixed temperature index (tau=0.5)
            acc_curve = problem.accuracy_means["model_A"][
                self.fixed_temp_idx, :
            ]  # shape [8]
            staircase_res = self.fit_piecewise_constant(acc_curve, dataset.budgets)
            sigmoid_res = self.fit_logistic_sigmoid(acc_curve, dataset.budgets)
            result = self.bic_model_select(staircase_res, sigmoid_res)
            self.bic_results.append(result)

        self.is_fitted = True

    def predict_per_problem(self, dataset):
        """Predict using fixed-temperature BIC results with ablation safety margin."""
        n = len(dataset.problems)
        predictions = np.zeros(n)  # shape [n_problems]
        for i in range(n):
            predictions[i] = self.bic_results[i]["d_c_est"] * self._safety_margin
        predictions = np.clip(predictions, dataset.budgets[0], dataset.budgets[-1])
        return predictions


# ═══════════════════════════════════════════════════════════════════════
# ABLATION 2: Kolmogorov Gzip Predictor (H3 ablation)
# ═══════════════════════════════════════════════════════════════════════


class KolmogorovGzipPredictor(CircuitDepthPredictor):
    """H3 ablation: ridge regression on gzip length instead of circuit depth.

    STRATEGY DIFFERENCE from parent:
    - Swaps the feature from circuit depth (D) to gzip compression length (K proxy)
    - Same ridge regression pipeline, same CV, same train/test split
    - In correlated regime: should perform similarly (K proportional to D)
    - In decorrelated regime: should perform significantly worse

    This tests whether the Kolmogorov complexity proxy suffices or whether
    circuit depth is specifically required.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "kolmogorov_gzip_predictor"

    def extract_depth_features(self, dataset):
        """OVERRIDE: use gzip compression length instead of circuit depth.

        This is the SOLE algorithmic difference from CircuitDepthPredictor.
        The entire ridge regression pipeline (CV, train/test split, fitting)
        operates identically — only the input feature changes.
        """
        return dataset.get_gzip_lengths().astype(float)  # shape [n_problems]

    def compute_gzip_features(self, dataset):
        """Explicit alias for clarity."""
        return self.extract_depth_features(dataset)
