"""Evaluation metrics, bootstrap inference, hypothesis testing, and cross-seed aggregation.

Computes primary MAPE, bootstrap confidence intervals, Wilcoxon signed-rank tests,
all 9 hypothesis-specific quantitative predictions (H1-H3), and the overall
discrimination score. Handles per-regime breakdowns and cross-seed statistics.
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import minimize

from config import ExperimentConfig
from data import SyntheticDataset


def compute_mape(predictions, ground_truth, eps):
    """Mean Absolute Percentage Error.

    Args:
        predictions: shape [N] predicted elbow budgets
        ground_truth: shape [N] true critical depths
        eps: small constant to prevent division by zero

    Returns:
        MAPE as a percentage (float).
    """
    return float(
        np.mean(
            np.abs(predictions - ground_truth)
            / np.maximum(ground_truth, eps)
        )
        * 100.0
    )


def bootstrap_ci(predictions, ground_truth, config, rng):
    """Bootstrap confidence interval for MAPE.

    Args:
        predictions: shape [N]
        ground_truth: shape [N]
        config: ExperimentConfig with bootstrap_n_resamples, bootstrap_ci_level, eps
        rng: np.random.Generator for reproducibility

    Returns:
        dict with mean, std, ci_low, ci_high
    """
    n = len(predictions)
    n_resamples = config.bootstrap_n_resamples
    mapes = np.zeros(n_resamples)

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        mapes[b] = compute_mape(predictions[idx], ground_truth[idx], config.eps)

    alpha = 1.0 - config.bootstrap_ci_level  # 0.05
    ci_low = float(np.percentile(mapes, 100.0 * alpha / 2.0))
    ci_high = float(np.percentile(mapes, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": float(np.mean(mapes)),
        "std": float(np.std(mapes)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def bootstrap_mape_difference_ci(preds_a, preds_b, gt, config, rng):
    """Paired bootstrap CI for the difference in MAPE between two methods.

    Args:
        preds_a: shape [N] predictions from method A
        preds_b: shape [N] predictions from method B
        gt: shape [N] ground truth
        config: ExperimentConfig
        rng: np.random.Generator

    Returns:
        dict with mean_diff, ci_low, ci_high (negative = A is better)
    """
    n = len(preds_a)
    n_resamples = config.bootstrap_n_resamples
    diffs = np.zeros(n_resamples)

    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        mape_a = compute_mape(preds_a[idx], gt[idx], config.eps)
        mape_b = compute_mape(preds_b[idx], gt[idx], config.eps)
        diffs[b] = mape_a - mape_b

    alpha = 1.0 - config.bootstrap_ci_level
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 100.0 * alpha / 2.0)),
        "ci_high": float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0))),
    }


def wilcoxon_paired_test(scores_a, scores_b):
    """Wilcoxon signed-rank test for paired MAPE scores across seeds.

    Args:
        scores_a: shape [n_seeds] MAPE values from method A
        scores_b: shape [n_seeds] MAPE values from method B

    Returns:
        dict with statistic, p_value, effect_size_rank_biserial
    """
    differences = scores_a - scores_b

    if np.all(np.abs(differences) < 1e-12):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_rank_biserial": 0.0,
        }

    # Need at least some non-zero differences for Wilcoxon
    nonzero = np.sum(np.abs(differences) > 1e-12)
    if nonzero < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_rank_biserial": 0.0,
        }

    try:
        stat, p_val = scipy_stats.wilcoxon(
            scores_a, scores_b, alternative="two-sided"
        )
    except ValueError:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_rank_biserial": 0.0,
        }

    # Rank-biserial effect size: r = 1 - (2*W) / (n*(n+1)/2)
    n = len(scores_a)
    denominator = n * (n + 1) / 2.0
    r_rb = 1.0 - (2.0 * stat) / denominator if denominator > 0 else 0.0

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "effect_size_rank_biserial": float(r_rb),
    }


def _fit_logconcave_elbow_on_acc_pop(acc_pop, budgets, config):
    """Fit log-concave function and extract elbow from a population accuracy curve.

    Used internally for bootstrap CV of population elbow.

    Args:
        acc_pop: shape [n_budgets] population-average accuracy
        budgets: shape [n_budgets] budget values
        config: ExperimentConfig

    Returns:
        elbow float value
    """
    def logconcave_func(T, a, b, c):
        return a * np.log(1.0 + b * T) / (1.0 + c * T)

    x0 = [config.slb_a_init, config.slb_b_init, config.slb_c_init]
    bounds = [config.slb_bounds_a, config.slb_bounds_b, config.slb_bounds_c]

    def loss(params):
        a, b, c = params
        pred = logconcave_func(budgets, a, b, c)
        return float(np.sum((pred - acc_pop) ** 2))

    result = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
    if result.success:
        a, b, c = result.x
    else:
        a, b, c = acc_pop[-1], 0.01, 0.001

    # Compute elbow via max |f''(T)|
    T_dense = np.linspace(budgets[0], budgets[-1], 500)
    f_vals = logconcave_func(T_dense, a, b, c)
    dT = T_dense[1] - T_dense[0]
    f_pp = np.gradient(np.gradient(f_vals, dT), dT)
    elbow_idx = np.argmax(np.abs(f_pp))
    return float(T_dense[elbow_idx])


def compute_hypothesis_metrics(dataset, all_results, config):
    """Compute all 9 hypothesis-specific metrics from a single seed's results.

    Args:
        dataset: SyntheticDataset from the seed being analyzed
        all_results: {condition_name: evaluate() dict} from all 7 conditions
        config: ExperimentConfig

    Returns:
        dict mapping metric names to float values
    """
    metrics = {}
    gt = dataset.get_ground_truth_elbows()  # shape [n_problems]

    # ── H1: Staircase BIC win rate ──
    sta_results = all_results.get("staircase_temperature_allocator", {})
    metrics["H1_staircase_bic_win_rate"] = float(
        sta_results.get("bic_staircase_win_rate", 0)
    )

    # ── H1: Population elbow bootstrap CV ──
    # Re-fit the log-concave bound on 50 bootstrap resamples of the population
    # to measure how unstable the single-elbow estimate is
    rng = np.random.default_rng(999)
    n_problems = len(dataset.problems)
    n_budgets = config.n_budgets
    budgets = dataset.budgets

    bootstrap_elbows = []
    for _ in range(50):
        bootstrap_idx = rng.integers(0, n_problems, size=n_problems)
        # Compute population-average accuracy on bootstrap sample
        acc_pop_boot = np.zeros(n_budgets)
        for b_idx in range(n_budgets):
            acc_sum = 0.0
            for prob_idx in bootstrap_idx:
                acc_sum += dataset.problems[prob_idx].accuracy_means["model_A"][
                    1, b_idx
                ]
            acc_pop_boot[b_idx] = acc_sum / len(bootstrap_idx)

        elbow = _fit_logconcave_elbow_on_acc_pop(acc_pop_boot, budgets, config)
        bootstrap_elbows.append(elbow)

    bootstrap_elbows = np.array(bootstrap_elbows)
    elbow_mean = np.mean(bootstrap_elbows)
    elbow_std = np.std(bootstrap_elbows)
    metrics["H1_population_elbow_bootstrap_cv"] = float(
        elbow_std / max(elbow_mean, config.eps)
    )

    # ── H1: Temperature-complexity Spearman ──
    # Correlation between gzip tercile and the optimal temperature assigned
    # by the staircase allocator to each problem's bucket
    gzip_lengths = dataset.get_gzip_lengths().astype(float)
    gzip_tercile_edges = np.percentile(gzip_lengths, [33.33, 66.67])
    gzip_terciles = np.digitize(gzip_lengths, gzip_tercile_edges)  # {0, 1, 2}

    bucket_optimal_temps = sta_results.get("bucket_optimal_temps", None)
    if bucket_optimal_temps is not None and len(bucket_optimal_temps) > 0:
        # Map each problem to its bucket's optimal temperature
        per_problem_opt_temps = np.array(
            [bucket_optimal_temps.get(int(t), 0.5) for t in gzip_terciles]
        )
        # Spearman correlation between tercile index and optimal temperature
        rho, _p = scipy_stats.spearmanr(gzip_terciles, per_problem_opt_temps)
        metrics["H1_temperature_complexity_spearman"] = float(rho)
    else:
        metrics["H1_temperature_complexity_spearman"] = 0.0

    # ── H2: MI non-monotonicity rate ──
    mi_results = all_results.get("mi_nonmonotone_detector", {})
    for model_name in config.model_configs.keys():
        key = f"mi_nonmonotonicity_rate_{model_name}"
        metrics[f"H2_{key}"] = float(mi_results.get(key, 0))

    # ── H2: Cross-model elbow divergence ──
    metrics["H2_cross_model_elbow_divergence"] = float(
        mi_results.get("cross_model_elbow_divergence", 0)
    )

    # ── H3: Correlations in decorrelated regime ──
    # Aggregate across both complexity levels for the decorrelated K-D regime
    decor_indices = []
    for (c, k), idx_list in dataset.regime_indices.items():
        if k == "decorrelated":
            decor_indices.extend(idx_list)
    decor_idx = np.array(decor_indices)

    depths = dataset.get_circuit_depths()
    gzips = gzip_lengths  # already computed above

    gt_decor = gt[decor_idx]
    depths_decor = depths[decor_idx]
    gzips_decor = gzips[decor_idx]

    # Pearson correlations in decorrelated regime
    if len(decor_idx) > 2:
        r_depth_decor, _ = scipy_stats.pearsonr(depths_decor, gt_decor)
        r_gzip_decor, _ = scipy_stats.pearsonr(gzips_decor, gt_decor)
    else:
        r_depth_decor = 0.0
        r_gzip_decor = 0.0

    metrics["H3_depth_elbow_correlation"] = float(r_depth_decor)
    metrics["H3_gzip_elbow_correlation"] = float(r_gzip_decor)

    # ── H3: MAPE gap in decorrelated regime ──
    cdp_results = all_results.get("circuit_depth_predictor", {})
    cdp_preds = cdp_results.get("predictions", np.zeros(len(gt)))
    kgp_results = all_results.get("kolmogorov_gzip_predictor", {})
    kgp_preds = kgp_results.get("predictions", np.zeros(len(gt)))

    mape_depth_decor = compute_mape(cdp_preds[decor_idx], gt_decor, config.eps)
    mape_gzip_decor = compute_mape(kgp_preds[decor_idx], gt_decor, config.eps)
    metrics["H3_depth_vs_gzip_mape_gap"] = float(
        mape_gzip_decor - mape_depth_decor
    )

    return metrics


def compute_discrimination_score(hypothesis_metrics, config):
    """Count how many hypothesis predictions FAIL to meet their threshold.

    0 = all confirmed (best), 9 = none confirmed (worst).

    Args:
        hypothesis_metrics: dict from compute_hypothesis_metrics
        config: ExperimentConfig with threshold values

    Returns:
        int count of unconfirmed predictions
    """
    checks = [
        ("H1_staircase_bic_win_rate", ">", config.h1_bic_win_rate_threshold),
        ("H1_population_elbow_bootstrap_cv", ">", config.h1_bootstrap_cv_threshold),
        (
            "H1_temperature_complexity_spearman",
            ">",
            config.h1_temp_complexity_spearman_threshold,
        ),
        (
            "H2_cross_model_elbow_divergence",
            ">",
            config.h2_elbow_divergence_threshold,
        ),
        (
            "H2_mi_nonmonotonicity_rate_model_A",
            ">",
            config.h2_mi_nonmonotonicity_threshold,
        ),
        (
            "H2_mi_nonmonotonicity_rate_model_B",
            ">",
            config.h2_mi_nonmonotonicity_threshold,
        ),
        ("H3_depth_elbow_correlation", ">", config.h3_depth_elbow_corr_threshold),
        ("H3_gzip_elbow_correlation", "<", config.h3_gzip_elbow_corr_threshold),
        ("H3_depth_vs_gzip_mape_gap", ">", config.h3_mape_gap_threshold),
    ]

    unconfirmed = 0
    for metric_name, direction, threshold in checks:
        value = hypothesis_metrics.get(metric_name, 0)
        if direction == ">":
            if not (value > threshold):
                unconfirmed += 1
        elif direction == "<":
            if not (value < threshold):
                unconfirmed += 1

    return unconfirmed


def compute_cross_seed_stats(per_seed_mapes):
    """Aggregate MAPE values across seeds for each condition.

    Args:
        per_seed_mapes: {condition_name: list of MAPE values (one per seed)}

    Returns:
        dict of {condition: {mean, std, cv, per_seed, cv_ok}}
    """
    stats_out = {}

    for condition, mapes in per_seed_mapes.items():
        # Filter NaN values for statistics
        arr = np.array(mapes, dtype=float)
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            stats_out[condition] = {
                "mean": float("nan"),
                "std": float("nan"),
                "cv": float("nan"),
                "per_seed": arr.tolist(),
                "cv_ok": False,
            }
            continue

        mean_mape = float(np.mean(valid))
        std_mape = float(np.std(valid))
        cv = std_mape / max(mean_mape, 1e-8)

        stats_out[condition] = {
            "mean": mean_mape,
            "std": std_mape,
            "cv": cv,
            "per_seed": arr.tolist(),
            "cv_ok": cv < 0.15,
        }

    return stats_out
