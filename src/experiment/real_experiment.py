#!/usr/bin/env python3
"""STAIR Real LLM Experiment + Figure Generation.

Runs actual LLM inference on GSM8K with varying token budgets and temperatures,
then generates publication-quality figures for NeurIPS submission.

Hardware: 1x RTX A5000 (24GB VRAM)
Models: Qwen2.5-0.5B-Instruct + Qwen2.5-1.5B-Instruct
Time: ~25 minutes total
"""

import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.linear_model import RidgeCV

# Config
N_PROBLEMS = 40
TOKEN_BUDGETS = [32, 64, 128, 256, 512]
TEMPERATURES = [0.1, 0.5, 1.0]
SAMPLES_PER_CELL = 4
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen-0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-1.5B"),
]

# NeurIPS figure style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "text.usetex": False, "axes.spines.top": False, "axes.spines.right": False,
})
C = {"blue": "#2563EB", "red": "#DC2626", "green": "#059669",
     "amber": "#D97706", "gray": "#6B7280", "lblue": "#93C5FD"}


def load_gsm8k(n=N_PROBLEMS):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    problems = []
    for row in ds:
        match = re.search(r"####\s*(-?[\d,]+)", row["answer"])
        final_answer = match.group(1).replace(",", "") if match else ""
        problems.append({
            "question": row["question"], "answer": final_answer,
            "step_count": row["answer"].count("\n") + 1,
            "gzip_len": len(gzip.compress(row["question"].encode(), compresslevel=9)),
            "text_len": len(row["question"]),
        })
    return problems


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_real_experiment(problems, model, tokenizer, model_label, out_dir):
    import torch
    n = len(problems)
    results = np.zeros((n, len(TOKEN_BUDGETS), len(TEMPERATURES)))
    total_cells = n * len(TOKEN_BUDGETS) * len(TEMPERATURES)
    cell_idx = 0
    t0 = time.time()

    for b_idx, budget in enumerate(TOKEN_BUDGETS):
        for t_idx, temp in enumerate(TEMPERATURES):
            # Batch all problems x samples for this budget+temp
            for p_idx, prob in enumerate(problems):
                cell_idx += 1
                prompt = (f"Solve step by step. End with 'The answer is [number]'.\n\n"
                          f"Question: {prob['question']}\n\nSolution:")
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate all samples at once via num_return_sequences
                with torch.no_grad():
                    outs = model.generate(
                        **{k: v.expand(SAMPLES_PER_CELL, -1) for k, v in inputs.items()},
                        max_new_tokens=budget,
                        temperature=max(temp, 0.01),
                        do_sample=temp > 0.05,
                        top_p=0.95 if temp > 0.05 else 1.0,
                        pad_token_id=tokenizer.pad_token_id)

                correct = 0
                for s in range(SAMPLES_PER_CELL):
                    resp = tokenizer.decode(outs[s][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    m = re.search(r"(?:answer is|=)\s*\$?(-?[\d,.]+)", resp, re.IGNORECASE)
                    pred = m.group(1).replace(",", "").rstrip(".") if m else ""
                    if not pred:
                        nums = re.findall(r"-?\d+", resp)
                        pred = nums[-1] if nums else ""
                    if pred and pred.strip() == prob["answer"].strip():
                        correct += 1
                results[p_idx, b_idx, t_idx] = correct / SAMPLES_PER_CELL

            elapsed = time.time() - t0
            done_frac = cell_idx / total_cells
            eta = (elapsed / max(done_frac, 0.001)) * (1 - done_frac)
            print(f"    [{model_label}] budget={budget} tau={temp} "
                  f"({cell_idx}/{total_cells}, {elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)

    np.save(out_dir / f"results_{model_label}.npy", results)
    return results


# Analysis
def fit_piecewise(curve):
    n = len(curve)
    best_rss, best_i = np.inf, n // 2
    for i in range(1, n - 1):
        pred = np.where(np.arange(n) < i, curve[:i].mean(), curve[i:].mean())
        rss = np.sum((curve - pred) ** 2)
        if rss < best_rss:
            best_rss = rss
            best_i = i
    bic = n * np.log(max(best_rss / n, 1e-12)) + 2 * np.log(n)
    return best_i, bic


def fit_sigmoid(curve):
    n = len(curve)
    try:
        fn = lambda t, L, k, t0: L / (1 + np.exp(-k * (t - t0)))
        popt, _ = optimize.curve_fit(fn, np.arange(n, dtype=float), curve,
            p0=[max(curve.max(), 0.1), 1.0, n/2],
            bounds=([0.01, 0.001, 0], [1.0, 10.0, n]), maxfev=5000)
        pred = fn(np.arange(n, dtype=float), *popt)
        rss = np.sum((curve - pred) ** 2)
        bic = n * np.log(max(rss / n, 1e-12)) + 3 * np.log(n)
        return popt, bic
    except Exception:
        return None, 1e6


def analyze_all(all_results, problems):
    model_key = list(all_results.keys())[-1]
    results = all_results[model_key]
    n = results.shape[0]

    # BIC analysis
    staircase_wins = np.zeros((n, len(TEMPERATURES)))
    for t_idx in range(len(TEMPERATURES)):
        for p_idx in range(n):
            _, bic_pw = fit_piecewise(results[p_idx, :, t_idx])
            _, bic_sig = fit_sigmoid(results[p_idx, :, t_idx])
            staircase_wins[p_idx, t_idx] = 1 if bic_pw < bic_sig - 2 else 0

    # Model divergence
    divergence = 0
    if len(all_results) >= 2:
        keys = list(all_results.keys())
        r1, r2 = all_results[keys[0]], all_results[keys[1]]
        divs = []
        for p in range(n):
            e1 = np.argmax(r1[p, :, 1] > 0.3) if r1[p, :, 1].max() > 0.3 else len(TOKEN_BUDGETS)-1
            e2 = np.argmax(r2[p, :, 1] > 0.3) if r2[p, :, 1].max() > 0.3 else len(TOKEN_BUDGETS)-1
            if max(e1, e2) > 0:
                divs.append(abs(e1 - e2) / max(e1, e2))
        divergence = np.mean(divs) if divs else 0

    # MI non-monotonicity
    nonmono = {}
    for ml, res in all_results.items():
        cnt = sum(1 for p in range(n) for t in range(len(TEMPERATURES))
                  if np.any(np.diff(res[p, :, t]) < -0.15))
        nonmono[ml] = cnt / (n * len(TEMPERATURES))

    return staircase_wins, divergence, nonmono


# Figures
def make_all_figures(all_results, problems, staircase_wins, fig_dir):
    model_key = list(all_results.keys())[-1]
    results = all_results[model_key]
    n = len(problems)
    budgets = TOKEN_BUDGETS

    # Fig 1: Per-problem staircase curves
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0), sharex=True)
    gz = np.array([p["gzip_len"] for p in problems])
    order = np.argsort(gz)
    terciles = [order[:n//3], order[n//3:2*n//3], order[2*n//3:]]
    labels = ["Easy (low gzip)", "Medium", "Hard (high gzip)"]
    for col, (subset, lab) in enumerate(zip(terciles, labels)):
        for row in range(2):
            ax = axes[row, col]
            if row >= len(subset):
                ax.axis("off"); continue
            p_idx = subset[row * (len(subset)//2)]
            curve = results[p_idx, :, 1]
            ax.plot(budgets, curve, "o-", color=C["blue"], markersize=4, linewidth=1.5)
            si, bpw = fit_piecewise(curve)
            spopt, bsig = fit_sigmoid(curve)
            if bpw < bsig:
                left, right = curve[:si].mean(), curve[si:].mean()
                ax.step(budgets, [left]*si + [right]*(len(budgets)-si),
                       "--", color=C["red"], linewidth=1, alpha=0.8)
                ax.annotate("Staircase", (0.95, 0.05), xycoords="axes fraction",
                           ha="right", fontsize=6, color=C["red"])
            elif spopt is not None:
                fn = lambda t, L, k, t0: L / (1 + np.exp(-k * (t - t0)))
                xs = np.linspace(0, len(budgets)-1, 50)
                ys = fn(xs, *spopt)
                bx = [budgets[min(int(x), len(budgets)-1)] for x in np.linspace(0, len(budgets)-1, 50)]
                ax.plot(bx, ys, "--", color=C["green"], linewidth=1, alpha=0.8)
                ax.annotate("Sigmoid", (0.95, 0.05), xycoords="axes fraction",
                           ha="right", fontsize=6, color=C["green"])
            ax.set_ylim(-0.05, 1.05)
            if col == 0: ax.set_ylabel("Accuracy")
            if row == 0: ax.set_title(lab, fontsize=9)
            if row == 1: ax.set_xlabel("Token Budget")
    fig.suptitle("Per-Problem Scaling Curves ($\\tau$=0.5, Qwen-1.5B, GSM8K)", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_staircase.pdf"); fig.savefig(fig_dir / "fig1_staircase.png"); plt.close()
    print("  Fig 1 saved")

    # Fig 2: Bootstrap violin
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for t_idx, temp in enumerate(TEMPERATURES):
        elbows = []
        for _ in range(200):
            idx = np.random.choice(n, n, replace=True)
            pop = results[idx, :, t_idx].mean(axis=0)
            d2 = np.gradient(np.gradient(pop))
            elbows.append(budgets[np.argmax(np.abs(d2))])
        bp = ax.boxplot([elbows], positions=[t_idx], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(C["lblue"])
        cv = np.std(elbows) / max(np.mean(elbows), 1)
        ax.annotate(f"CV={cv:.2f}", (t_idx, max(elbows)+10), ha="center", fontsize=7, color=C["gray"])
    ax.set_xticks(range(len(TEMPERATURES)))
    ax.set_xticklabels([f"$\\tau$={t}" for t in TEMPERATURES])
    ax.set_ylabel("Estimated Elbow (tokens)")
    ax.set_title("Bootstrap Elbow Stability", fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_bootstrap.pdf"); fig.savefig(fig_dir / "fig2_bootstrap.png"); plt.close()
    print("  Fig 2 saved")

    # Fig 3: Model heatmap
    if len(all_results) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)
        for m_idx, (ml, res) in enumerate(all_results.items()):
            mat = np.zeros((3, len(budgets)))
            for ti, terc in enumerate(terciles):
                for bi in range(len(budgets)):
                    mat[ti, bi] = res[terc, bi, 1].mean()
            im = axes[m_idx].imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.6)
            axes[m_idx].set_xticks(range(len(budgets)))
            axes[m_idx].set_xticklabels(budgets, fontsize=7)
            axes[m_idx].set_xlabel("Token Budget")
            axes[m_idx].set_title(ml, fontsize=10)
            if m_idx == 0:
                axes[m_idx].set_yticks(range(3))
                axes[m_idx].set_yticklabels(["Easy", "Medium", "Hard"], fontsize=8)
            for i in range(3):
                for j in range(len(budgets)):
                    axes[m_idx].text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                                    fontsize=6, color="white" if mat[i,j]<0.2 or mat[i,j]>0.5 else "black")
        fig.colorbar(im, ax=axes, label="Accuracy", shrink=0.8)
        fig.suptitle("Accuracy: Model x Complexity x Budget ($\\tau$=0.5)", fontsize=10, y=1.02)
        plt.tight_layout()
        fig.savefig(fig_dir / "fig3_heatmap.pdf"); fig.savefig(fig_dir / "fig3_heatmap.png"); plt.close()
        print("  Fig 3 saved")

    # Fig 4: Complexity proxy scatter
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    elbows = []
    for p in range(n):
        curve = results[p, :, 1]
        above = np.where(curve > 0.2)[0]
        elbows.append(budgets[above[0]] if len(above) > 0 else budgets[-1])
    elbows = np.array(elbows)
    for ax, (feat, lab, col) in zip(axes, [
        ([p["gzip_len"] for p in problems], "Gzip Length (bytes)", C["blue"]),
        ([p["step_count"] for p in problems], "Solution Steps", C["red"]),
        ([p["text_len"] for p in problems], "Question Length", C["green"]),
    ]):
        feat = np.array(feat)
        ax.scatter(feat, elbows, s=12, alpha=0.5, color=col, edgecolors="none")
        v = np.isfinite(feat) & (elbows < budgets[-1])
        if v.sum() > 5:
            r = np.corrcoef(feat[v], elbows[v])[0, 1]
            z = np.polyfit(feat[v], elbows[v], 1)
            xl = np.linspace(feat.min(), feat.max(), 50)
            ax.plot(xl, np.polyval(z, xl), "--", color=col, alpha=0.7)
            ax.annotate(f"$\\rho$={r:.2f}", (0.05, 0.9), xycoords="axes fraction",
                       fontsize=8, color=col, fontweight="bold")
        ax.set_xlabel(lab)
        if ax == axes[0]: ax.set_ylabel("Elbow (tokens)")
    fig.suptitle("Complexity Proxy vs Scaling Elbow", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_proxy.pdf"); fig.savefig(fig_dir / "fig4_proxy.png"); plt.close()
    print("  Fig 4 saved")

    # Fig 5: Pareto curve
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fixed_c, fixed_a = [], []
    for bi, b in enumerate(budgets):
        fixed_c.append(b * n)
        fixed_a.append(results[:, bi, 1].mean())
    ax.plot(fixed_c, fixed_a, "o-", color=C["gray"], markersize=5, linewidth=1.5, label="Fixed Budget")
    for thresh, col, mk in [(0.2, C["blue"], "s"), (0.4, C["red"], "D")]:
        cost, corr = 0, 0
        for p in range(n):
            curve = results[p, :, 1]
            assigned = False
            for bi, b in enumerate(budgets):
                if curve[bi] >= thresh:
                    cost += b; corr += curve[bi]; assigned = True; break
            if not assigned:
                cost += budgets[-1]; corr += curve[-1]
        ax.plot(cost, corr/n, mk, color=col, markersize=8, zorder=4,
               label=f"STAIR ($\\theta$={thresh})")
    for bi, b in enumerate(budgets):
        ax.annotate(f"t={b}", (fixed_c[bi], fixed_a[bi]), textcoords="offset points",
                   xytext=(5,5), fontsize=6, color=C["gray"])
    ax.set_xlabel("Total Tokens"); ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Inference Cost", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig5_pareto.pdf"); fig.savefig(fig_dir / "fig5_pareto.png"); plt.close()
    print("  Fig 5 saved")

    # Fig 6: Temperature effect
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, (subset, lab) in zip(axes, [(terciles[0], "Easy"), (terciles[2], "Hard")]):
        for ti, (temp, col) in enumerate(zip(TEMPERATURES, [C["blue"], C["amber"], C["red"]])):
            pop = results[subset, :, ti].mean(axis=0)
            ax.plot(budgets, pop, "o-", color=col, markersize=4, linewidth=1.5, label=f"$\\tau$={temp}")
        ax.set_xlabel("Token Budget"); ax.set_ylabel("Accuracy" if ax==axes[0] else "")
        ax.set_title(lab, fontsize=10); ax.set_ylim(-0.05, 0.8); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Temperature Effect on Scaling (Qwen-1.5B, GSM8K)", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig6_temperature.pdf"); fig.savefig(fig_dir / "fig6_temperature.png"); plt.close()
    print("  Fig 6 saved")

    # Fig 7: BIC rates bar chart
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    rates = staircase_wins.mean(axis=0)
    bars = ax.bar(range(len(TEMPERATURES)), rates, color=C["blue"], alpha=0.8, width=0.5)
    ax.axhline(0.55, color=C["red"], linestyle="--", linewidth=1, label="Threshold (0.55)")
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.8, alpha=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
               f"{r:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(TEMPERATURES)))
    ax.set_xticklabels([f"$\\tau$={t}" for t in TEMPERATURES])
    ax.set_ylabel("Staircase BIC Win Rate"); ax.set_ylim(0, 1.0)
    ax.set_title("BIC Win Rate by Temperature", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig7_bic.pdf"); fig.savefig(fig_dir / "fig7_bic.png"); plt.close()
    print("  Fig 7 saved")


def main(output_dir="/workspace/results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STAIR Real LLM Experiment")
    print("=" * 60)

    print(f"\n[1/4] Loading GSM8K...", flush=True)
    problems = load_gsm8k(N_PROBLEMS)
    print(f"  {len(problems)} problems loaded")
    gz = [p["gzip_len"] for p in problems]
    sc = [p["step_count"] for p in problems]
    print(f"  Gzip-step corr: {np.corrcoef(gz, sc)[0,1]:.3f}")

    print(f"\n[2/4] Running inference...", flush=True)
    all_results = {}
    for mname, mlabel in MODELS:
        print(f"\n  === {mlabel} ===")
        model, tok = load_model(mname)
        res = run_real_experiment(problems, model, tok, mlabel, out)
        all_results[mlabel] = res
        del model, tok
        import torch; torch.cuda.empty_cache()

    print(f"\n[3/4] Analysis...", flush=True)
    sw, div, nm = analyze_all(all_results, problems)
    print(f"  BIC win rates: {sw.mean(axis=0).round(3)}")
    print(f"  Model divergence: {div:.3f}")
    print(f"  MI non-mono: {nm}")

    summary = {
        "n_problems": len(problems), "models": [m[1] for m in MODELS],
        "bic_win_rates": {f"tau_{t}": float(r) for t, r in zip(TEMPERATURES, sw.mean(axis=0))},
        "overall_bic_rate": float(sw.mean()),
        "model_divergence": float(div),
        "mi_nonmono": {k: float(v) for k, v in nm.items()},
        "gzip_step_corr": float(np.corrcoef(gz, sc)[0, 1]),
        "accuracy": {ml: {str(b): float(r[:, bi, 1].mean()) for bi, b in enumerate(TOKEN_BUDGETS)}
                     for ml, r in all_results.items()},
    }
    with open(out / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[4/4] Generating figures...", flush=True)
    make_all_figures(all_results, problems, sw, fig_dir)

    print(f"\n{'='*60}\nDONE. Results: {out}\n{'='*60}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/results")
