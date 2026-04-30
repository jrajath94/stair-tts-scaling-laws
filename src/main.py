"""Entry point for the information-theoretic test-time compute scaling experiment.

Runs ALL 7 conditions x 5 seeds with time budget enforcement, per-seed random
state isolation, scaling fallback, GSM8K ecological validation, hypothesis testing,
and structured output printing. No CLI arguments -- all conditions iterated internally.
"""

import math
import time
import traceback

import numpy as np

from config import ExperimentConfig
from data import SyntheticDataset, GSM8KFeatures
from methods import (
    BaseMethod,
    SmoothLogconcaveBound,
    EmpiricalPowerLawFit,
    StaircaseTemperatureAllocator,
    MINonmonotoneDetector,
    CircuitDepthPredictor,
    StaircaseNoTemperature,
    KolmogorovGzipPredictor,
)
from evaluation import (
    compute_mape,
    bootstrap_ci,
    bootstrap_mape_difference_ci,
    wilcoxon_paired_test,
    compute_hypothesis_metrics,
    compute_discrimination_score,
    compute_cross_seed_stats,
)

# Try to import experiment_harness (available in sandbox, may not exist locally)
try:
    from experiment_harness import ExperimentHarness

    HAS_HARNESS = True
except ImportError:
    HAS_HARNESS = False


# ======================================================================
# Method registry and helpers
# ======================================================================

METHOD_REGISTRY = {
    "smooth_logconcave_bound": SmoothLogconcaveBound,
    "empirical_power_law_fit": EmpiricalPowerLawFit,
    "staircase_temperature_allocator": StaircaseTemperatureAllocator,
    "mi_nonmonotone_detector": MINonmonotoneDetector,
    "circuit_depth_predictor": CircuitDepthPredictor,
    "staircase_no_temperature": StaircaseNoTemperature,
    "kolmogorov_gzip_predictor": KolmogorovGzipPredictor,
}


def create_method(condition_name, config):
    """Instantiate a method class by condition name."""
    cls = METHOD_REGISTRY[condition_name]
    return cls(config)


def run_single_seed(condition_name, config, seed):
    """Run a single condition on a single seed with isolated random state.

    The paired design ensures identical datasets across conditions for the same seed.

    Returns:
        (results_dict, dataset) tuple
    """
    np.random.seed(seed)  # legacy compat for any library using np.random

    # Generate fresh dataset with this seed
    dataset = SyntheticDataset(config)
    dataset.generate(seed)

    # Create and evaluate method
    method = create_method(condition_name, config)
    results = method.evaluate(dataset)
    results["seed"] = seed

    return results, dataset


def run_gsm8k_analysis(config):
    """Run GSM8K ecological validation (text-feature correlation analysis)."""
    print("-- GSM8K Ecological Validation --")
    gsm8k = GSM8KFeatures(config)
    try:
        gsm8k.load()
    except Exception as e:
        print(f"  WARNING: GSM8K loading failed: {e}")
        print("  (This is expected if running without internet or HuggingFace access)")
        return {"status": "failed", "error": str(e)}

    corr_matrix = gsm8k.compute_correlation_matrix()
    print(
        f"  Pearson r(gzip, steps): {corr_matrix['pearson_gzip_vs_steps']:.3f} "
        f"(p={corr_matrix['pearson_pvalue']:.4f})"
    )
    print(
        f"  Spearman rho(gzip, steps): {corr_matrix['spearman_gzip_vs_steps']:.3f}"
    )
    print(
        f"  Pearson r(gzip, textlen): {corr_matrix['pearson_gzip_vs_textlen']:.3f}"
    )
    return corr_matrix


def check_scaling_fallback(elapsed_seconds, conditions_done, total_conditions, config):
    """Activate scaling fallback if approaching time budget with work remaining."""
    fraction_time = elapsed_seconds / config.total_time_budget_seconds
    fraction_work = conditions_done / total_conditions

    if fraction_time > config.fallback_time_fraction and fraction_work < 0.5:
        print("WARNING: Approaching time budget. Activating scaling fallback.")
        config.n_problems_per_cell = config.fallback_reduced_problems
        config.total_problems = config.n_problems_per_cell * config.n_regime_cells
        config.bootstrap_n_resamples = config.fallback_reduced_bootstrap
        # Recompute budget array (unchanged but ensures consistency)
        config.budget_array = np.array(config.reasoning_budgets, dtype=float)
        print(
            f"  Reduced to {config.n_problems_per_cell} problems/cell, "
            f"{config.bootstrap_n_resamples} bootstrap resamples"
        )

    return config


def run_pilot_estimate(config):
    """Run 1 seed of the fastest condition to estimate total runtime."""
    pilot_start = time.time()
    # Use circuit_depth_predictor (fastest) as pilot
    try:
        _results, _ds = run_single_seed("circuit_depth_predictor", config, seed=99)
        pilot_time = time.time() - pilot_start
    except Exception:
        pilot_time = 2.0  # fallback estimate

    # Estimate: 7 conditions x 5 seeds, but staircase methods are ~10x slower
    # Weight: 2 fast baselines + 2 fast methods + 1 staircase + 2 staircase-derived
    weighted_per_seed = pilot_time * (4 + 3 * 10)  # 4 fast + 3 slow
    estimated_total = weighted_per_seed * config.n_seeds / 7  # per condition avg
    estimated_total = estimated_total * 7  # all conditions
    estimated_total += 30  # overhead for bootstrap CIs and hypothesis metrics

    return estimated_total, pilot_time


# ======================================================================
# Main function
# ======================================================================


def main():
    start_time = time.time()
    config = ExperimentConfig()

    # Initialize experiment harness if available
    harness = None
    if HAS_HARNESS:
        harness = ExperimentHarness(time_budget=config.total_time_budget_seconds)

    # -- Print METRIC_DEF --
    print("METRIC_DEF: elbow_prediction_mape (MAPE %) | minimize")

    # -- Print REGISTERED_CONDITIONS --
    conditions = config.get_condition_names()
    print(f"REGISTERED_CONDITIONS: {', '.join(conditions)}")

    # -- GSM8K ecological validation (one-time cost) --
    gsm8k_results = run_gsm8k_analysis(config)

    # -- Pilot time estimate --
    estimated_total, pilot_time = run_pilot_estimate(config)
    print(f"TIME_ESTIMATE: {estimated_total:.0f}s (pilot={pilot_time:.2f}s)")

    # -- Main experiment loop --
    per_seed_mapes = {c: [] for c in conditions}
    all_seed_results = {c: [] for c in conditions}
    last_seed_dataset = None
    last_seed_results = {}

    for cond_idx, condition_name in enumerate(conditions):
        print(f"\n{'=' * 60}")
        print(f"CONDITION: {condition_name} ({cond_idx + 1}/{len(conditions)})")
        print(f"{'=' * 60}")

        # Time budget guard
        elapsed = time.time() - start_time
        config = check_scaling_fallback(
            elapsed, cond_idx, len(conditions), config
        )

        if elapsed > config.hard_cap_seconds:
            print(f"HARD CAP REACHED at {elapsed:.0f}s. Stopping.")
            break

        # Harness time check
        if harness is not None and harness.should_stop():
            print(f"HARNESS STOP at {elapsed:.0f}s. Saving partial results.")
            break

        condition_mapes = []
        for seed in config.seeds:
            seed_start = time.time()

            # Check time budget before each seed
            elapsed_now = time.time() - start_time
            if elapsed_now > config.hard_cap_seconds:
                print(
                    f"  HARD CAP at {elapsed_now:.0f}s mid-condition. "
                    f"Stopping seeds."
                )
                break

            if harness is not None and harness.should_stop():
                print(f"  HARNESS STOP at {elapsed_now:.0f}s mid-condition.")
                break

            try:
                results, dataset = run_single_seed(condition_name, config, seed)
                mape = results["elbow_prediction_mape"]

                # NaN/divergence guard
                if math.isnan(mape) or math.isinf(mape):
                    print(
                        f"  FAIL: NaN/divergence detected for "
                        f"condition={condition_name} seed={seed}"
                    )
                    condition_mapes.append(float("nan"))
                    continue

                # Harness metric validation
                if harness is not None:
                    if not harness.check_value(mape, "elbow_prediction_mape"):
                        print(f"  SKIP: harness rejected value {mape}")
                        condition_mapes.append(float("nan"))
                        continue
                    harness.report_metric("elbow_prediction_mape", mape)

                condition_mapes.append(mape)
                all_seed_results[condition_name].append(results)
                seed_time = time.time() - seed_start

                print(
                    f"  RESULT: condition={condition_name} seed={seed} "
                    f"elbow_prediction_mape={mape:.2f} time={seed_time:.1f}s"
                )

                # Per-regime breakdown
                for regime_key, regime_mape in results["regime_mapes"].items():
                    print(f"    regime={regime_key} MAPE={regime_mape:.2f}")

                # Store last seed for hypothesis analysis
                if seed == config.seeds[-1]:
                    last_seed_dataset = dataset
                    last_seed_results[condition_name] = results

            except Exception as e:
                print(
                    f"  FAILED: condition={condition_name} seed={seed} "
                    f"error={e}"
                )
                traceback.print_exc()
                condition_mapes.append(float("nan"))

        per_seed_mapes[condition_name] = condition_mapes

        valid_mapes = [m for m in condition_mapes if not math.isnan(m)]
        if len(valid_mapes) > 0:
            arr = np.array(valid_mapes)
            print(
                f"  AGGREGATE: {condition_name} "
                f"mean={np.mean(arr):.2f} std={np.std(arr):.2f}"
            )

    # -- Cross-seed statistics --
    print(f"\n{'=' * 60}")
    print("CROSS-SEED STATISTICS")
    print(f"{'=' * 60}")
    cross_seed = compute_cross_seed_stats(per_seed_mapes)
    for condition, stats_dict in cross_seed.items():
        mean_val = stats_dict["mean"]
        if math.isnan(mean_val):
            print(f"  {condition}: NO VALID RESULTS")
            continue
        print(
            f"  {condition}: mean={stats_dict['mean']:.2f} "
            f"std={stats_dict['std']:.2f} CV={stats_dict['cv']:.3f} "
            f"CV_OK={'YES' if stats_dict['cv_ok'] else 'NO'}"
        )
        print(f"    per_seed: {stats_dict['per_seed']}")

    # -- Success rate --
    total_runs = sum(len(v) for v in all_seed_results.values())
    n_nan = sum(
        1
        for c in per_seed_mapes
        for m in per_seed_mapes[c]
        if math.isnan(m)
    )
    total_attempted = total_runs + n_nan
    success_rate = 1.0 - n_nan / max(total_attempted, 1)
    print(
        f"\nSUCCESS_RATE: {success_rate:.3f} "
        f"(target >= {config.success_rate_target})"
    )

    # -- Pairwise Wilcoxon tests --
    print(f"\n{'=' * 60}")
    print("PAIRWISE COMPARISONS (Wilcoxon signed-rank)")
    print(f"{'=' * 60}")
    baseline_name = "smooth_logconcave_bound"
    for condition in conditions:
        if condition == baseline_name:
            continue
        a = np.array(per_seed_mapes.get(baseline_name, []))
        b = np.array(per_seed_mapes.get(condition, []))

        if len(a) == 0 or len(b) == 0:
            continue

        # Pad shorter array with NaN to match lengths
        max_len = max(len(a), len(b))
        if len(a) < max_len:
            a = np.concatenate([a, np.full(max_len - len(a), float("nan"))])
        if len(b) < max_len:
            b = np.concatenate([b, np.full(max_len - len(b), float("nan"))])

        valid = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid) >= 3:
            test = wilcoxon_paired_test(a[valid], b[valid])
            print(
                f"  {baseline_name} vs {condition}: "
                f"p={test['p_value']:.4f} "
                f"r_rb={test['effect_size_rank_biserial']:.3f}"
            )

    # -- Bootstrap CIs on last seed --
    print(f"\n{'=' * 60}")
    print("BOOTSTRAP 95% CI (last seed)")
    print(f"{'=' * 60}")
    rng_boot = np.random.default_rng(42)
    if last_seed_dataset is not None:
        gt = last_seed_dataset.get_ground_truth_elbows()
        for condition, results in last_seed_results.items():
            preds = results.get("predictions", np.zeros_like(gt))
            ci = bootstrap_ci(preds, gt, config, rng_boot)
            print(
                f"  {condition}: MAPE={ci['mean']:.2f} "
                f"95%CI=[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"
            )
    else:
        print("  (no completed seeds available)")

    # -- Hypothesis-specific metrics --
    print(f"\n{'=' * 60}")
    print("HYPOTHESIS METRICS")
    print(f"{'=' * 60}")
    if last_seed_dataset is not None and len(last_seed_results) > 0:
        h_metrics = compute_hypothesis_metrics(
            last_seed_dataset, last_seed_results, config
        )
        for key, val in sorted(h_metrics.items()):
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

        disc_score = compute_discrimination_score(h_metrics, config)
        print(
            f"\n  hypothesis_discrimination_score: {disc_score} / 9 "
            f"(0=all confirmed)"
        )
    else:
        print("  (no completed results for hypothesis testing)")

    # -- SUMMARY table --
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Condition':<40} {'Mean MAPE':>10} {'Std':>8} {'CV':>8}")
    print("-" * 66)
    for condition in conditions:
        s = cross_seed.get(condition, {})
        mean_val = s.get("mean", float("nan"))
        if math.isnan(mean_val):
            print(f"{condition:<40} {'N/A':>10} {'N/A':>8} {'N/A':>8}")
        else:
            print(
                f"{condition:<40} {mean_val:>10.2f} "
                f"{s.get('std', 0):>8.2f} {s.get('cv', 0):>8.3f}"
            )

    # -- Total time --
    elapsed_total = time.time() - start_time
    print(
        f"\nTOTAL_TIME: {elapsed_total:.1f}s / "
        f"{config.total_time_budget_seconds}s budget"
    )

    # -- Final answer: best proposed method --
    proposed = [
        "staircase_temperature_allocator",
        "mi_nonmonotone_detector",
        "circuit_depth_predictor",
    ]
    best_proposed = min(
        proposed,
        key=lambda c: cross_seed.get(c, {}).get("mean", float("inf")),
    )
    best_mape = cross_seed.get(best_proposed, {}).get("mean", float("nan"))

    if not math.isnan(best_mape):
        print(
            f"\nFINAL_ANSWER: best_proposed={best_proposed} "
            f"elbow_prediction_mape={best_mape:.2f}"
        )
    else:
        print("\nFINAL_ANSWER: no valid results obtained")

    # -- Finalize harness --
    if harness is not None:
        harness.finalize()


if __name__ == "__main__":
    main()
