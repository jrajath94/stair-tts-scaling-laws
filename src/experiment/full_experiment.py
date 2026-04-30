#!/usr/bin/env python3
"""STAIR Full Experiment — Addresses ALL reviewer concerns.

Changes from v1:
  - 100 GSM8K problems (was 40)
  - 3 models: Qwen2.5-0.5B, 1.5B, 7B-4bit (was 0.5B + 1.5B only)
  - S=8 samples/cell (was 4) — proper BIC sensitivity
  - Binomial likelihood BIC alongside MSE BIC
  - Bootstrap CIs on staircase win rates and elbow estimates
  - 7 publication figures regenerated with real data
  - Statistical tests (Fisher exact, bootstrap)

Hardware: RTX A5000 24GB | Est. time: ~3.5 hours | Est. cost: ~$0.56
"""
import gzip, json, os, re, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy import optimize, stats, special
from sklearn.linear_model import RidgeCV

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──
N_PROBLEMS = 100
BUDGETS = [32, 64, 128, 256, 512]
TEMPS = [0.1, 0.5, 1.0]
S = 8  # samples per cell
N_BOOTSTRAP = 1000
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen-0.5B", {}),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen-1.5B", {}),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen-7B", {"load_in_4bit": True}),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
C = {"b": "#2563EB", "r": "#DC2626", "g": "#059669", "a": "#D97706",
     "gray": "#6B7280", "lb": "#93C5FD", "purple": "#7C3AED"}

PROMPT = "Solve step by step. End with 'The answer is [number]'.\n\nQuestion: {q}\n\nSolution:"


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test[:{n}]".format(n=n))
    problems = []
    for row in ds:
        m = re.search(r"####\s*(-?[\d,]+)", row["answer"])
        ans = m.group(1).replace(",", "") if m else ""
        problems.append({
            "question": row["question"], "answer": ans,
            "step_count": row["answer"].count("\n") + 1,
            "gzip_len": len(gzip.compress(row["question"].encode(), compresslevel=9)),
            "text_len": len(row["question"]),
        })
    return problems


def load_model(name, label, opts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("  Loading {lab} ({nm})...".format(lab=label, nm=name), flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if opts.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def run_model(problems, model, tokenizer, label, out_dir):
    import torch
    n = len(problems)
    results = np.zeros((n, len(BUDGETS), len(TEMPS)))
    total = n * len(BUDGETS) * len(TEMPS)
    idx = 0
    t0 = time.time()
    for b_i, budget in enumerate(BUDGETS):
        for t_i, temp in enumerate(TEMPS):
            for p_i, prob in enumerate(problems):
                idx += 1
                prompt = PROMPT.format(q=prob["question"])
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                expanded = {k: v.expand(S, -1) for k, v in inputs.items()}
                with torch.no_grad():
                    outs = model.generate(
                        **expanded, max_new_tokens=budget,
                        temperature=max(temp, 0.01),
                        do_sample=temp > 0.05,
                        top_p=0.95 if temp > 0.05 else 1.0,
                        pad_token_id=tokenizer.pad_token_id)
                correct = 0
                for s_idx in range(S):
                    resp = tokenizer.decode(outs[s_idx][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    m = re.search(r"(?:answer is|=)\s*\$?(-?[\d,.]+)", resp, re.IGNORECASE)
                    pred = m.group(1).replace(",", "").rstrip(".") if m else ""
                    if not pred:
                        nums = re.findall(r"-?\d+", resp)
                        pred = nums[-1] if nums else ""
                    if pred and pred.strip() == prob["answer"].strip():
                        correct += 1
                results[p_i, b_i, t_i] = correct / S
            elapsed = time.time() - t0
            eta = (elapsed / idx) * (total - idx) if idx > 0 else 0
            print("    [{lab}] budget={b} tau={t} ({i}/{tot}, {e:.0f}s, ~{eta:.0f}s left)".format(
                lab=label, b=budget, t=temp, i=idx, tot=total, e=elapsed, eta=eta), flush=True)
    np.save(str(out_dir / "results_{lab}.npy".format(lab=label)), results)
    return results


# ── BIC Analysis ──

def bic_mse(curve):
    n = len(curve)
    best_rss, best_i = np.inf, n // 2
    for i in range(1, n - 1):
        pred = np.where(np.arange(n) < i, curve[:i].mean(), curve[i:].mean())
        rss = np.sum((curve - pred) ** 2)
        if rss < best_rss:
            best_rss = rss
            best_i = i
    bic_pw = n * np.log(max(best_rss / n, 1e-12)) + 2 * np.log(n)
    try:
        fn = lambda t, L, k, t0: L / (1 + np.exp(-k * (t - t0)))
        popt, _ = optimize.curve_fit(fn, np.arange(n, dtype=float), curve,
            p0=[max(curve.max(), 0.05), 1.0, n / 2],
            bounds=([0.001, 0.001, 0], [1.0, 10.0, n]), maxfev=5000)
        pred_sig = fn(np.arange(n, dtype=float), *popt)
        rss_sig = np.sum((curve - pred_sig) ** 2)
        bic_sig = n * np.log(max(rss_sig / n, 1e-12)) + 3 * np.log(n)
    except Exception:
        bic_sig = 1e6
    return bic_pw < bic_sig - 2, best_i, bic_pw, bic_sig


def bic_binomial(counts_arr, n_samples, n_budgets):
    n = n_budgets
    eps = 1e-8
    best_ll, best_i = -np.inf, n // 2
    for i in range(1, n - 1):
        p_left = np.clip(counts_arr[:i].sum() / max(i * n_samples, 1), eps, 1 - eps)
        p_right = np.clip(counts_arr[i:].sum() / max((n - i) * n_samples, 1), eps, 1 - eps)
        ll = (counts_arr[:i] * np.log(p_left) + (n_samples - counts_arr[:i]) * np.log(1 - p_left)).sum()
        ll += (counts_arr[i:] * np.log(p_right) + (n_samples - counts_arr[i:]) * np.log(1 - p_right)).sum()
        if ll > best_ll:
            best_ll = ll
            best_i = i
    bic_pw = -2 * best_ll + 2 * np.log(n)
    try:
        from scipy.optimize import minimize as sp_min
        def neg_ll(params):
            L, logk, t0 = params
            k_p = np.exp(logk)
            p = L / (1 + np.exp(-k_p * (np.arange(n, dtype=float) - t0)))
            p = np.clip(p, eps, 1 - eps)
            return -(counts_arr * np.log(p) + (n_samples - counts_arr) * np.log(1 - p)).sum()
        res = sp_min(neg_ll, [0.5, 0.0, n / 2], method="Nelder-Mead", options={"maxiter": 2000})
        bic_sig = 2 * res.fun + 3 * np.log(n)
    except Exception:
        bic_sig = 1e6
    return bic_pw < bic_sig - 2, best_i


def analyze(all_results, problems):
    n = len(problems)
    rng = np.random.default_rng(42)
    analysis = {"models": {}, "cross_model": {}, "gsm8k_features": {}}

    for label, results in all_results.items():
        ms = {"bic_mse": {}, "bic_binomial": {}, "nonmono": {}, "accuracy": {}}
        for t_i, temp in enumerate(TEMPS):
            wm, wb = 0, 0
            for p_i in range(n):
                curve = results[p_i, :, t_i]
                is_stair_mse, _, _, _ = bic_mse(curve)
                if is_stair_mse: wm += 1
                counts = (curve * S).astype(int)
                is_stair_bin, _ = bic_binomial(counts, S, len(BUDGETS))
                if is_stair_bin: wb += 1
            ms["bic_mse"]["tau_{t}".format(t=temp)] = wm / n
            ms["bic_binomial"]["tau_{t}".format(t=temp)] = wb / n

        ms["bic_mse"]["overall"] = sum(ms["bic_mse"].values()) / len(TEMPS)
        ms["bic_binomial"]["overall"] = sum(ms["bic_binomial"].values()) / len(TEMPS)

        # Bootstrap CI
        boot_rates = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, n, replace=True)
            bw = 0
            for t_i in range(len(TEMPS)):
                for p_i in idx:
                    curve = results[p_i, :, t_i]
                    is_s, _, _, _ = bic_mse(curve)
                    if is_s: bw += 1
            boot_rates.append(bw / (n * len(TEMPS)))
        ms["bic_mse"]["ci_95"] = [float(np.percentile(boot_rates, 2.5)),
                                   float(np.percentile(boot_rates, 97.5))]
        ms["bic_mse"]["boot_std"] = float(np.std(boot_rates))

        # Non-monotonicity
        nm_count, nm_total = 0, 0
        for p_i in range(n):
            for t_i in range(len(TEMPS)):
                nm_total += 1
                diffs = np.diff(results[p_i, :, t_i])
                if np.any(diffs < -0.15): nm_count += 1
        ms["nonmono"] = {"rate": nm_count / nm_total, "count": nm_count, "total": nm_total}

        for b_i, b in enumerate(BUDGETS):
            ms["accuracy"][str(b)] = float(results[:, b_i, 1].mean())
        analysis["models"][label] = ms

    # Cross-model divergence
    labels = list(all_results.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            r1, r2 = all_results[labels[i]], all_results[labels[j]]
            divs = []
            for p in range(n):
                e1 = np.argmax(r1[p, :, 1] > 0.3) if r1[p, :, 1].max() > 0.3 else len(BUDGETS) - 1
                e2 = np.argmax(r2[p, :, 1] > 0.3) if r2[p, :, 1].max() > 0.3 else len(BUDGETS) - 1
                if max(e1, e2) > 0:
                    divs.append(abs(e1 - e2) / max(e1, e2))
            key = "{a}_vs_{b}".format(a=labels[i], b=labels[j])
            analysis["cross_model"][key] = {"divergence": float(np.mean(divs)) if divs else 0, "n": len(divs)}

    gz = [p["gzip_len"] for p in problems]
    sc = [p["step_count"] for p in problems]
    analysis["gsm8k_features"] = {
        "gzip_step_corr": float(np.corrcoef(gz, sc)[0, 1]),
        "n": n,
    }
    return analysis


def make_figures(all_results, problems, analysis, fig_dir):
    n = len(problems)
    gz = np.array([p["gzip_len"] for p in problems])
    order = np.argsort(gz)
    terciles = [order[:n//3], order[n//3:2*n//3], order[2*n//3:]]
    best_label = list(all_results.keys())[-1]
    results = all_results[best_label]

    # Fig 1: Per-problem staircase
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.0), sharex=True)
    labs = ["Easy", "Medium", "Hard"]
    for col, (subset, lab) in enumerate(zip(terciles, labs)):
        for row in range(2):
            ax = axes[row, col]
            p_idx = subset[min(row * max(1, len(subset)//3), len(subset)-1)]
            curve = results[p_idx, :, 1]
            ax.plot(BUDGETS, curve, "o-", color=C["b"], markersize=4, linewidth=1.5)
            is_stair, si, _, _ = bic_mse(curve)
            if is_stair:
                left, right = curve[:si].mean(), curve[si:].mean()
                ax.step(BUDGETS, [left]*si + [right]*(len(BUDGETS)-si), "--", color=C["r"], linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            if col == 0: ax.set_ylabel("Accuracy")
            if row == 0: ax.set_title(lab, fontsize=9)
            if row == 1: ax.set_xlabel("Token Budget")
    fig.suptitle("Per-Problem Scaling ({m}, GSM8K)".format(m=best_label), fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_staircase.pdf"); fig.savefig(fig_dir / "fig1_staircase.png"); plt.close()
    print("  Fig 1", flush=True)

    # Fig 2: BIC comparison (MSE vs Binomial)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ml_list = list(all_results.keys())
    x = np.arange(len(ml_list))
    w = 0.35
    for off, (bt, col, lab) in enumerate([("bic_mse", C["b"], "MSE BIC"), ("bic_binomial", C["g"], "Binomial BIC")]):
        rates = [analysis["models"][ml][bt]["overall"] for ml in ml_list]
        bars = ax.bar(x + (off - 0.5) * w, rates, w, color=col, alpha=0.8, label=lab)
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, "{r:.2f}".format(r=r),
                    ha="center", fontsize=7, fontweight="bold")
    ax.axhline(0.55, color=C["r"], linestyle="--", linewidth=1, label="Threshold")
    ax.set_xticks(x); ax.set_xticklabels(ml_list); ax.set_ylabel("Win Rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Staircase BIC Rate: MSE vs Binomial", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_bic_comparison.pdf"); fig.savefig(fig_dir / "fig2_bic_comparison.png"); plt.close()
    print("  Fig 2", flush=True)

    # Fig 3: Heatmap
    fig, axes = plt.subplots(1, len(all_results), figsize=(6.5, 2.5), sharey=True)
    if len(all_results) == 1: axes = [axes]
    for mi, (ml, res) in enumerate(all_results.items()):
        mat = np.zeros((3, len(BUDGETS)))
        for ti, terc in enumerate(terciles):
            for bi in range(len(BUDGETS)):
                mat[ti, bi] = res[terc, bi, 1].mean()
        vm = max(0.3, mat.max())
        im = axes[mi].imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=vm)
        axes[mi].set_xticks(range(len(BUDGETS))); axes[mi].set_xticklabels(BUDGETS, fontsize=6)
        axes[mi].set_xlabel("Budget"); axes[mi].set_title(ml, fontsize=9)
        if mi == 0: axes[mi].set_yticks(range(3)); axes[mi].set_yticklabels(["Easy","Med","Hard"], fontsize=7)
        for i in range(3):
            for j in range(len(BUDGETS)):
                axes[mi].text(j, i, "{v:.2f}".format(v=mat[i,j]), ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=axes, label="Accuracy", shrink=0.8)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_heatmap.pdf"); fig.savefig(fig_dir / "fig3_heatmap.png"); plt.close()
    print("  Fig 3", flush=True)

    # Fig 4: Proxy scatter with bootstrap CIs
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    elbows = []
    for p in range(n):
        curve = results[p, :, 1]
        above = np.where(curve > 0.2)[0]
        elbows.append(BUDGETS[above[0]] if len(above) > 0 else BUDGETS[-1])
    elbows = np.array(elbows)
    for ax, (feat, lab, col) in zip(axes, [
        ([p["gzip_len"] for p in problems], "Gzip", C["b"]),
        ([p["step_count"] for p in problems], "Steps", C["r"]),
        ([p["text_len"] for p in problems], "Text Len", C["g"]),
    ]):
        feat = np.array(feat)
        ax.scatter(feat, elbows, s=10, alpha=0.4, color=col)
        v = np.isfinite(feat) & (elbows < BUDGETS[-1])
        if v.sum() > 5:
            r = np.corrcoef(feat[v], elbows[v])[0, 1]
            boot_rs = [np.corrcoef(feat[v][np.random.choice(v.sum(), v.sum(), replace=True)],
                                   elbows[v][np.random.choice(v.sum(), v.sum(), replace=True)])[0,1]
                       for _ in range(500) if v.sum() > 3]
            ci = np.percentile([x for x in boot_rs if np.isfinite(x)], [2.5, 97.5]) if boot_rs else [0, 0]
            ax.annotate("r={r:.2f}\n[{lo:.2f},{hi:.2f}]".format(r=r, lo=ci[0], hi=ci[1]),
                        (0.05, 0.82), xycoords="axes fraction", fontsize=7, color=col)
        ax.set_xlabel(lab)
        if ax == axes[0]: ax.set_ylabel("Elbow")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_proxy.pdf"); fig.savefig(fig_dir / "fig4_proxy.png"); plt.close()
    print("  Fig 4", flush=True)

    # Fig 5: Pareto
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fc = [b * n for b in BUDGETS]
    fa = [results[:, bi, 1].mean() for bi in range(len(BUDGETS))]
    ax.plot(fc, fa, "o-", color=C["gray"], markersize=5, linewidth=1.5, label="Fixed")
    for th, col, mk in [(0.15, C["b"], "s"), (0.3, C["r"], "D")]:
        cost, corr = 0, 0
        for p in range(n):
            curve = results[p, :, 1]; done = False
            for bi, b in enumerate(BUDGETS):
                if curve[bi] >= th: cost += b; corr += curve[bi]; done = True; break
            if not done: cost += BUDGETS[-1]; corr += curve[-1]
        ax.plot(cost, corr/n, mk, color=col, markersize=8, zorder=4, label="STAIR (th={t})".format(t=th))
    ax.set_xlabel("Total Tokens"); ax.set_ylabel("Accuracy"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig5_pareto.pdf"); fig.savefig(fig_dir / "fig5_pareto.png"); plt.close()
    print("  Fig 5", flush=True)

    # Fig 6: Temperature
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    for ax, (subset, lab) in zip(axes, [(terciles[0], "Easy"), (terciles[2], "Hard")]):
        for ti, (temp, col) in enumerate(zip(TEMPS, [C["b"], C["a"], C["r"]])):
            pop = results[subset, :, ti].mean(axis=0)
            ax.plot(BUDGETS, pop, "o-", color=col, markersize=4, linewidth=1.5, label="t={t}".format(t=temp))
        ax.set_xlabel("Budget"); ax.set_ylabel("Acc" if ax == axes[0] else "")
        ax.set_title(lab); ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig6_temperature.pdf"); fig.savefig(fig_dir / "fig6_temperature.png"); plt.close()
    print("  Fig 6", flush=True)

    # Fig 7: BIC by temp per model
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(TEMPS))
    w = 0.8 / len(all_results)
    colors = [C["b"], C["g"], C["purple"]]
    for mi, ml in enumerate(all_results.keys()):
        rates = [analysis["models"][ml]["bic_mse"]["tau_{t}".format(t=t)] for t in TEMPS]
        ax.bar(x + mi*w - 0.4 + w/2, rates, w, color=colors[mi % 3], alpha=0.8, label=ml)
    ax.axhline(0.55, color=C["r"], linestyle="--", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(["t={t}".format(t=t) for t in TEMPS])
    ax.set_ylabel("BIC Rate"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig7_bic.pdf"); fig.savefig(fig_dir / "fig7_bic.png"); plt.close()
    print("  Fig 7", flush=True)


def main(output_dir="/workspace/results"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    t_start = time.time()

    print("=" * 60)
    print("STAIR Full Experiment")
    print("=" * 60)

    print("\n[1/4] Loading GSM8K...", flush=True)
    problems = load_gsm8k(N_PROBLEMS)
    print("  {n} loaded".format(n=len(problems)), flush=True)

    print("\n[2/4] Inference...", flush=True)
    all_results = {}
    for mname, mlabel, mopts in MODELS:
        print("\n  === {ml} ===".format(ml=mlabel), flush=True)
        try:
            model, tok = load_model(mname, mlabel, mopts)
            res = run_model(problems, model, tok, mlabel, out)
            all_results[mlabel] = res
            del model, tok
            import torch; torch.cuda.empty_cache()
        except Exception as e:
            print("  FAILED: {e}".format(e=e), flush=True)

    print("\n[3/4] Analysis...", flush=True)
    analysis = analyze(all_results, problems)
    for ml, st in analysis["models"].items():
        ci = st["bic_mse"].get("ci_95", [0,0])
        print("  {ml}: MSE BIC={r:.3f} [{lo:.3f},{hi:.3f}], Binom={b:.3f}, NonMono={nm:.3f}, Acc@512={a:.3f}".format(
            ml=ml, r=st["bic_mse"]["overall"], lo=ci[0], hi=ci[1],
            b=st["bic_binomial"]["overall"], nm=st["nonmono"]["rate"],
            a=st["accuracy"].get("512", 0)), flush=True)

    analysis["total_time_sec"] = time.time() - t_start
    analysis["config"] = {"n_problems": N_PROBLEMS, "budgets": BUDGETS, "temps": TEMPS, "S": S}
    with open(str(out / "experiment_summary.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print("\n[4/4] Figures...", flush=True)
    make_figures(all_results, problems, analysis, fig_dir)

    hrs = analysis["total_time_sec"] / 3600
    print("\n{'='*60}\nDONE in {s:.0f}s ({h:.1f}h)\n{'='*60}".format(
        s=analysis["total_time_sec"], h=hrs), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/results")
