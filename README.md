# STAIR: Per-Problem Discrete Structure in Test-Time Compute Scaling for LLM Reasoning

Companion repository for the research manuscript **STAIR** (Staircase Test-time Adaptive Inference Routing).

**Paper**: [`paper/STAIR_paper.pdf`](paper/STAIR_paper.pdf)

---

## TL;DR

Test-time compute scaling for LLM reasoning is widely modeled as a **smooth, monotonic, task-determined** curve. We test all three assumptions on real LLMs (Qwen2.5-0.5B/1.5B on 100 GSM8K problems, 24,000 inference calls) and show:

1. **97.7–99.3%** of per-problem accuracy curves are better fit by **piecewise-constant staircases** than smooth sigmoids under both MSE-based and binomial-likelihood BIC. Restricted to the variation subset (curves with accuracy range above the sampling floor), the rate is still **87.3–93.1%** with 95% bootstrap CIs above 0.6.
2. **5.3–8.7%** of (problem, temperature) cells show empirical-accuracy non-monotonicity, attributable to budget-level answer truncation. We do **not** claim DPI violations — empirical accuracy is not mutual information.
3. **Computational depth ≠ description length**: on real GSM8K, neither circuit depth nor gzip compression predict the elbow at the small-model accuracy floor, indicating a systematic dissociation between text structure and scaling behavior.
4. The **STAIR allocator** uses an *O(n)* gzip proxy plus pre-calibrated per-bucket temperatures and matches fixed-budget-512 accuracy on Qwen-1.5B at **75% lower token cost** (paired Wilcoxon *p* = 0.23), while beating a confidence-adaptive stopping baseline by **2.6× in accuracy** at 1.6× the cost.

A theorem (Theorem 1 in the paper) reconciles discrete per-problem scaling with smooth population curves under log-concave critical-depth distributions.

---

## Repository layout

```
.
├── paper/                       # Final PDF
│   └── STAIR_paper.pdf
├── latex/                       # Paper LaTeX source
│   ├── main.tex
│   ├── references.bib
│   ├── neurips_2025.sty         # Official NeurIPS 2025 style
│   └── figures/                 # All paper figures (PDF + PNG)
├── src/                         # Core library
│   ├── config.py                # Hyperparameters (single source of truth)
│   ├── config.yaml              # Same, in YAML
│   ├── data.py                  # Synthetic reasoning channel
│   ├── methods.py               # BIC fits, allocator, baselines
│   ├── evaluation.py            # Metric definitions
│   ├── main.py                  # End-to-end synthetic pipeline
│   └── experiment/              # Real-LLM inference scripts
│       ├── full_experiment.py   # Synthetic + analysis
│       ├── real_experiment.py   # GSM8K x Qwen2.5 inference
│       ├── main.py              # Driver
│       └── setup.sh             # Environment bootstrap
├── scripts/                     # Standalone analysis utilities
│   └── reanalyze.py             # Single-pipeline reanalysis: .npy -> JSON
├── data/                        # Real-LLM inference outputs
│   ├── results_Qwen-0.5B.npy    # Shape [100, 5, 3]: problems × budgets × temps
│   └── results_Qwen-1.5B.npy
├── results/                     # Computed numbers (deterministic)
│   ├── reanalysis_stratified.json   # Single-pipeline source for paper
│   └── real_results_v2_summary.json
├── LICENSE                      # MIT
├── .gitignore
└── README.md                    # This file
```

---

## Reproducing the paper numbers

All paper numbers are derived from a **single deterministic pipeline**. To reproduce:

```bash
# 1. Install dependencies (CPU-only is sufficient for reanalysis)
python -m pip install numpy scipy

# 2. Reanalyze the inference outputs
python scripts/reanalyze.py \
    --data-dir data/ \
    --out results/reanalysis_stratified.json
```

This regenerates `results/reanalysis_stratified.json`, which contains every number cited in the paper:

| Quantity (from paper) | JSON path | Value |
|---|---|---|
| MSE-BIC overall, Qwen-0.5B | `real_llm_stratified."Qwen2.5-0.5B".overall.mse_rate` | 0.993 |
| Bootstrap CI | `…overall.ci_mse` | [0.983, 1.000] |
| Variation-subset rate, Qwen-0.5B | `…with_variation.mse_rate` | 0.931 |
| Variation-subset rate, Qwen-1.5B | `…with_variation.mse_rate` | 0.873 |
| Non-monotonicity rate, Qwen-0.5B | `…overall.nm_rate` | 0.053 |
| Cross-model divergence (overall) | `cross_model.elbow_divergence_all_problems.mean` | 0.244 |
| Cross-model divergence (variation subset) | `cross_model.elbow_divergence_variation_subset.mean` | 0.506 |

The deterministic seed (`42`) and bootstrap iterations (`B = 2000`) are hard-coded; rerunning produces byte-identical JSON.

---

## Reproducing the real-LLM inference (optional)

The `.npy` files in `data/` contain the raw 24,000 inference outcomes. To regenerate them from scratch:

```bash
# Requires a single GPU with ≥ 24 GB VRAM (e.g., NVIDIA RTX A5000)
bash src/experiment/setup.sh
python src/experiment/real_experiment.py
```

**Configuration** (single source of truth in `src/config.py`):
- Models: `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct` (Apache 2.0)
- Benchmark: GSM8K test split, first 100 problems by index (no cherry-picking) — MIT license
- Token budgets: `{32, 64, 128, 256, 512}`
- Temperatures: `{0.1, 0.5, 1.0}`
- Samples per cell: `S = 8`
- Total: 100 × 5 × 3 × 8 × 2 = 24,000 forward passes

**Prompt template** (verbatim):
```
Solve step by step. End with 'The answer is [number]'. Question: {q} Solution:
```

**Answer extraction** (deterministic regex):
```python
r"(?i)the answer is\s*\$?(-?\d[\d,.]*)"
```
Fallback: last integer in the response. Ground truth: integer after `####` in the GSM8K reference answer.

---

## Reproducing the synthetic experiments

```bash
python src/main.py
```

The synthetic reasoning channel (`src/data.py`) is fully deterministic at base seed 42. Default config: 800 problems × 5 seeds × 8 budgets × 3 temperatures × 32 samples. Runs entirely on CPU.

---

## Key design choices

### 1. BIC formulation: MSE *and* binomial

The paper reports both. MSE-BIC is widely used but assumes Gaussian residuals, which is wrong for binomial accuracy. Binomial-BIC uses the proper Bernoulli likelihood:

$$
\text{BIC}_{\text{Bin}} = -2 \sum_{j=1}^{B} \left[ k_j \ln \hat{p}_j + (S - k_j) \ln(1 - \hat{p}_j) \right] + p \ln(B)
$$

Both yield staircase win rates well above 0.55 (the pre-registered threshold); binomial-BIC gives *higher* rates, contrary to the intuition that MSE-BIC favors simpler models in low-accuracy regimes.

### 2. Variation subset stratification

At small-model accuracy, many curves are flat at zero. On flat curves the staircase model wins by parsimony alone — the finding is uninformative. We therefore stratify into the **variation subset**: curves whose accuracy range exceeds the sampling floor 1/*S* = 0.125. Headline numbers report both strata.

### 3. Cluster bootstrap

We resample **problems** (not (problem, temperature) pairs) so that the three temperatures of the same problem stay together in each bootstrap replicate. This respects within-problem clustering. *B* = 2000 iterations.

### 4. Pareto split

The 100 GSM8K problems are stratified-split 80/20 within each gzip tercile (seed 42). Per-bucket temperatures are calibrated on the 80% calibration set; all reported Pareto numbers are on the held-out 20% (20 problems). Significance: paired Wilcoxon signed-rank test, problem-level pairing.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{stair2025neurips,
  title     = {STAIR: Per-Problem Discrete Structure in Test-Time Compute
               Scaling for LLM Reasoning},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {Independent research manuscript}
}
```

---

## Licenses and credits

- **Code in this repository**: MIT (see [`LICENSE`](LICENSE)).
- **Qwen2.5-0.5B/1.5B-Instruct**: Apache 2.0 (Alibaba Cloud).
- **GSM8K**: MIT (Cobbe et al., 2021).
- **NeurIPS 2025 style file** (`latex/neurips_2025.sty`): provided by NeurIPS, used per the conference template terms.

---

## Limitations (in plain English)

- Both Qwen2.5 models score < 7% on GSM8K — most curves are flat at zero, which is why we report variation-subset rates separately. Validation at 7B–70B+ is the most important follow-up.
- GSM8K is arithmetic only; step counts span just `{2, …, 7}`. Wider depth distributions (code, logic, open-ended) would tighten the complexity-proxy diagnostic.
- Five budget points limit BIC's resolution; 10–20 points would tighten the variation-subset CIs.
- The deployed allocator uses gzip as a *proxy for circuit depth*; circuit depth itself requires ground-truth solution structure. Learned circuit-depth estimators are open work.
- Paired Wilcoxon *p* = 0.20–0.23 for STAIR vs fixed-budget-512 accuracy means we claim **no-harm-on-accuracy at ~4× lower token cost**, not significant accuracy gain.

See §8 of the paper for the full enumeration.

---

## Open issues / contributions

This repository is a research artifact accompanying a paper submission. PRs, issues, and replications are welcome — particularly:

1. Reproductions on larger Qwen, Llama, or Mistral checkpoints.
2. Extensions to code (HumanEval), logic (FOLIO), or open-ended benchmarks.
3. Direct mutual-information estimation from logits (to settle the DPI question we deliberately avoid claiming).
4. Learned circuit-depth predictors that operate on raw problem text.
