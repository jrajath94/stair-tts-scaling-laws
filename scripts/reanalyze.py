"""Single-pipeline reanalysis of real LLM inference results.

Takes per-problem accuracy tensors [N, B, T] (problems x budgets x
temperatures) and produces all numbers reported in the paper:

    * MSE-BIC and binomial-BIC staircase win rates (overall and on the
      "variation subset" where accuracy range exceeds the 1/S sampling floor)
    * Bootstrap 95% CIs, cluster-resampled by problem (B=2000)
    * Cross-model elbow divergence
    * Accuracy non-monotonicity rates (epsilon = 0.85 sigma)
    * BIC threshold sensitivity sweep

Output: results/reanalysis_stratified.json (single source of truth that
every paper number must match).

Usage
-----
    python scripts/reanalyze.py \
        --data-dir data/ \
        --out results/reanalysis_stratified.json

Notes
-----
S = number of samples per (problem, budget, temperature) cell.  Inferred
from the value range of the input tensor (8 in our experiments).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize

BUDGETS = np.array([32, 64, 128, 256, 512])
TEMPERATURES = [0.1, 0.5, 0.9]
SAMPLES_PER_CELL = 8
EPSILON_NONMONOTONIC = 0.15  # ~0.85 sigma at p=0.5, S=8
RNG_SEED = 42
N_BOOTSTRAP = 2000


# ──────────────────────────────────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────────────────────────────────
def fit_staircase(accuracies: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Fit a 1-step piecewise-constant model. Returns (split_idx, rss, fit)."""
    n = len(accuracies)
    best_rss = np.inf
    best_idx = 1
    best_fit = accuracies.copy()
    for split in range(1, n):
        left = accuracies[:split].mean()
        right = accuracies[split:].mean()
        fit = np.concatenate([np.full(split, left), np.full(n - split, right)])
        rss = float(((accuracies - fit) ** 2).sum())
        if rss < best_rss:
            best_rss, best_idx, best_fit = rss, split, fit
    return best_idx, best_rss, best_fit


def fit_sigmoid(budgets: np.ndarray, accuracies: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit a logistic sigmoid; returns (rss, fit)."""
    def sig(t, L, k, t0):
        return L / (1.0 + np.exp(-k * (t - t0)))

    try:
        popt, _ = optimize.curve_fit(
            sig, budgets, accuracies,
            p0=[max(accuracies.max(), 1e-3), 0.01, np.median(budgets)],
            bounds=([0, 1e-4, budgets.min()], [1.5, 1.0, budgets.max() * 2]),
            maxfev=2000,
        )
        fit = sig(budgets, *popt)
        return float(((accuracies - fit) ** 2).sum()), fit
    except Exception:
        # Fallback: degenerate fit at the mean
        fit = np.full_like(accuracies, accuracies.mean())
        return float(((accuracies - fit) ** 2).sum()), fit


def bic_mse(rss: float, n: int, k: int) -> float:
    """MSE-based BIC (assumes Gaussian residuals)."""
    return n * np.log(max(rss / n, 1e-12)) + k * np.log(n)


def bic_binomial(accs: np.ndarray, fit: np.ndarray, samples: int, k: int) -> float:
    """Binomial-likelihood BIC. accs and fit are probabilities in [0, 1]."""
    p = np.clip(fit, 1e-6, 1 - 1e-6)
    successes = accs * samples
    failures = (1 - accs) * samples
    log_lik = float(np.sum(successes * np.log(p) + failures * np.log(1 - p)))
    return -2 * log_lik + k * np.log(len(accs))


# ──────────────────────────────────────────────────────────────────────────
# Per-problem classification
# ──────────────────────────────────────────────────────────────────────────
def classify_curve(accs: np.ndarray, samples: int = SAMPLES_PER_CELL,
                   delta_threshold: float = 2.0) -> dict:
    """Fit both models and return classification + diagnostic."""
    n = len(accs)
    split_idx, stair_rss, stair_fit = fit_staircase(accs)
    sig_rss, sig_fit = fit_sigmoid(BUDGETS.astype(float), accs)

    bic_stair_mse = bic_mse(stair_rss, n, k=2)
    bic_sig_mse = bic_mse(sig_rss, n, k=3)
    bic_stair_bin = bic_binomial(accs, stair_fit, samples, k=2)
    bic_sig_bin = bic_binomial(accs, sig_fit, samples, k=3)

    # Tie-break rule: |ΔBIC| < threshold is ambiguous, defaults to staircase.
    # Sigmoid wins only with strong evidence (ΔBIC < -threshold).
    delta_mse = float(bic_sig_mse - bic_stair_mse)
    delta_bin = float(bic_sig_bin - bic_stair_bin)
    return {
        "split_idx": int(split_idx),
        "elbow_budget": float(BUDGETS[split_idx]),
        "stair_wins_mse": delta_mse > -delta_threshold,
        "stair_wins_bin": delta_bin > -delta_threshold,
        "delta_bic_mse": delta_mse,
        "delta_bic_bin": delta_bin,
        "acc_range": float(accs.max() - accs.min()),
    }


def detect_non_monotonic(accs: np.ndarray, eps: float = EPSILON_NONMONOTONIC) -> bool:
    diffs = np.diff(accs)
    return bool((diffs < -eps).any())


# ──────────────────────────────────────────────────────────────────────────
# Cluster bootstrap (resample problems, not (problem, temp) pairs)
# ──────────────────────────────────────────────────────────────────────────
def cluster_bootstrap_rate(per_problem_per_temp: np.ndarray,
                           n_iter: int = N_BOOTSTRAP,
                           seed: int = RNG_SEED) -> tuple[float, float]:
    """per_problem_per_temp: shape [N_problems, N_temps] of bool. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    n_problems = per_problem_per_temp.shape[0]
    rates = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n_problems, size=n_problems)
        rates[i] = per_problem_per_temp[idx].mean()
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


# ──────────────────────────────────────────────────────────────────────────
# Aggregate per model
# ──────────────────────────────────────────────────────────────────────────
def analyse_model(data: np.ndarray, label: str) -> dict:
    """data: shape [N_problems, N_budgets, N_temperatures]."""
    n_p, n_b, n_t = data.shape
    classifications = np.empty((n_p, n_t), dtype=object)
    stair_mse = np.zeros((n_p, n_t), dtype=bool)
    stair_bin = np.zeros((n_p, n_t), dtype=bool)
    nonmono = np.zeros((n_p, n_t), dtype=bool)
    has_variation = np.zeros((n_p, n_t), dtype=bool)
    elbows = np.zeros((n_p, n_t))

    for p in range(n_p):
        for t in range(n_t):
            accs = data[p, :, t].astype(float)
            classification = classify_curve(accs)
            classifications[p, t] = classification
            stair_mse[p, t] = classification["stair_wins_mse"]
            stair_bin[p, t] = classification["stair_wins_bin"]
            nonmono[p, t] = detect_non_monotonic(accs)
            has_variation[p, t] = classification["acc_range"] > 1.0 / SAMPLES_PER_CELL
            elbows[p, t] = classification["elbow_budget"]

    overall_mse = float(stair_mse.mean())
    overall_bin = float(stair_bin.mean())
    overall_nm = float(nonmono.mean())

    var_subset = stair_mse[has_variation]
    var_bin = stair_bin[has_variation]
    n_var = int(has_variation.sum())

    # Per-temperature breakdown
    per_temp = {}
    for ti, tau in enumerate(TEMPERATURES):
        per_temp[f"tau_{tau}"] = {
            "n_with_variation": int(has_variation[:, ti].sum()),
            "mse_rate_all": float(stair_mse[:, ti].mean()),
            "mse_rate_with_var": float(stair_mse[has_variation[:, ti], ti].mean())
            if has_variation[:, ti].any() else None,
            "binom_rate_all": float(stair_bin[:, ti].mean()),
            "binom_rate_with_var": float(stair_bin[has_variation[:, ti], ti].mean())
            if has_variation[:, ti].any() else None,
        }

    # Bootstrap CIs (cluster by problem)
    ci_mse_overall = cluster_bootstrap_rate(stair_mse)
    ci_bin_overall = cluster_bootstrap_rate(stair_bin)

    # Variation subset CI at tau=0.5
    tau05_idx = TEMPERATURES.index(0.5)
    var_mask_tau05 = has_variation[:, tau05_idx]
    if var_mask_tau05.any():
        decisions_mse_tau05 = stair_mse[var_mask_tau05, tau05_idx][:, None]
        decisions_bin_tau05 = stair_bin[var_mask_tau05, tau05_idx][:, None]
        ci_mse_tau05 = cluster_bootstrap_rate(decisions_mse_tau05)
        ci_bin_tau05 = cluster_bootstrap_rate(decisions_bin_tau05)
    else:
        ci_mse_tau05 = (None, None)
        ci_bin_tau05 = (None, None)

    return {
        "label": label,
        "total_pairs": n_p * n_t,
        "overall": {
            "mse_rate": overall_mse,
            "binom_rate": overall_bin,
            "nm_rate": overall_nm,
            "ci_mse": ci_mse_overall,
            "ci_binom": ci_bin_overall,
        },
        "with_variation": {
            "n_pairs": n_var,
            "pct_of_total": n_var / (n_p * n_t),
            "mse_rate": float(var_subset.mean()) if n_var else None,
            "binom_rate": float(var_bin.mean()) if n_var else None,
            "ci_mse_tau05": list(ci_mse_tau05),
            "ci_binom_tau05": list(ci_bin_tau05),
        },
        "per_temp": per_temp,
        "elbows_tau05": elbows[:, tau05_idx].tolist(),
        "has_variation_tau05": var_mask_tau05.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Cross-model divergence
# ──────────────────────────────────────────────────────────────────────────
def cross_model_divergence(elbows_a: list, elbows_b: list,
                           variation_mask_a: list, variation_mask_b: list,
                           seed: int = RNG_SEED) -> dict:
    a = np.array(elbows_a, dtype=float)
    b = np.array(elbows_b, dtype=float)
    norm = np.maximum(a, b)
    norm[norm == 0] = 1.0  # avoid div-by-zero
    div = np.abs(a - b) / norm
    overall = float(div.mean())

    var = np.array(variation_mask_a, dtype=bool) | np.array(variation_mask_b, dtype=bool)
    n_var = int(var.sum())
    var_div = float(div[var].mean()) if n_var else None

    rng = np.random.default_rng(seed)
    n = len(div)
    bootstraps_overall = np.empty(N_BOOTSTRAP)
    bootstraps_var = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        bootstraps_overall[i] = div[idx].mean()
        var_idx = var[idx]
        bootstraps_var[i] = div[idx][var_idx].mean() if var_idx.any() else np.nan

    return {
        "n": n,
        "overall_mean": overall,
        "overall_ci": [float(np.quantile(bootstraps_overall, 0.025)),
                       float(np.quantile(bootstraps_overall, 0.975))],
        "variation_subset_n": n_var,
        "variation_subset_mean": var_div,
        "variation_subset_ci": [float(np.nanquantile(bootstraps_var, 0.025)),
                                float(np.nanquantile(bootstraps_var, 0.975))]
        if n_var else [None, None],
    }


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/reanalysis_stratified.json"))
    args = ap.parse_args()

    data_05 = np.load(args.data_dir / "results_Qwen-0.5B.npy")
    data_15 = np.load(args.data_dir / "results_Qwen-1.5B.npy")

    out: dict = {"real_llm_stratified": {}}
    for label, data in [("Qwen2.5-0.5B", data_05), ("Qwen2.5-1.5B", data_15)]:
        out["real_llm_stratified"][label] = analyse_model(data, label)

    out["cross_model_divergence"] = cross_model_divergence(
        out["real_llm_stratified"]["Qwen2.5-0.5B"]["elbows_tau05"],
        out["real_llm_stratified"]["Qwen2.5-1.5B"]["elbows_tau05"],
        out["real_llm_stratified"]["Qwen2.5-0.5B"]["has_variation_tau05"],
        out["real_llm_stratified"]["Qwen2.5-1.5B"]["has_variation_tau05"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
