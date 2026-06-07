# A Graph-Attention Variable-Support Front-End for Symbolic Regression of Physical Laws

Companion code, models, data artifacts and manuscript for the paper
**"A Graph-Attention Variable-Support Front-End that Accelerates Symbolic
Regression of Physical Laws"** (submitted to *Communications Physics*).

A learned **front-end** reads a small set of input–output observations and
predicts a compact **support prior** — the variables (and optionally operators)
likely to appear in the governing law — which is handed to a conventional
symbolic-regression backend (PySR) for the final, verifiable search.

The neural model proposes **where** to look; the symbolic backend decides
**what** the law is.

---

## Headline results

| Result | Number |
|---|---|
| Variable-support **recall** (AI-Feynman 118 / OOD 91 / extended 1085 / **SRSD 120**) | **1.000 / 0.996 / 0.999 / 0.967** (≥92% perfect-recall per suite) |
| PySR exact-law recovery, fixed 10 s budget (30 Feynman, ~20 distractors, **3 seeds**) | **42 ± 13% → 60 ± 0%** (seed-stable) |
| Mean held-out R² (3 seeds) | 0.906 ± 0.045 → 0.980 ± 0.011 |
| Column-space reduction | **85%** (23.2 → 3.4 cols) |
| Search-vocabulary reduction | 40 → 20 (variables), → 13 (+operators) |
| Speedup vs. full PySR (single seed, 10 formulas) | **up to 6×** (median ~1.5×) |
| Front-end inference overhead | ~40 ms / task |
| Second backend (gplearn, 3 seeds) | full 3.3 ± 3.3% → DenoisedSR 17.8 ± 1.9% exact |
| Classical-selector baselines (Lasso CV with oracle k, Feynman 118) | 0.905 recall (DenoisedSR: 1.000) |

A **same-size random-support** control performs far worse, showing the gain is
search guidance, not dimensionality reduction. Five **classical feature
selectors** (Pearson / Spearman / mutual info / RF importance / Lasso CV at oracle
*k*) reach 0.63–0.91 recall on Feynman; DenoisedSR is 1.000 on all 118. **SRSD-Feynman**
(Matsubara 2024) provides an external benchmark the model never saw during training,
and gplearn provides a second backend confirming the gain is solver-agnostic.

All numbers are computed by the scripts in `frontend/eval/` from the recorded
artifacts in `data/results/`.

---

## Repository layout

```
.
├── paper/                     manuscript (Communications Physics)
│   ├── main.tex / main.pdf
│   ├── references.bib
│   ├── code/make_figures.py   regenerates every figure from data/results/
│   └── figures/               12 figures (PDF+PNG, 300 dpi)
├── src/                       shared helpers (vendored from the training tree)
│   ├── run_pysr_pmlb_feynman_learned_prior.py
│   └── evaluate_stage8g_open_generation.py
├── frontend/                  front-end code
│   ├── data_prep/             formula-pool construction (training only)
│   ├── train/                 train_gat.py, train_support_predictor_v2.py, …
│   ├── eval/                  reproduction scripts for paper figures + tables
│   │     eval_pysr_frontend.py         PySR 3-way + Table 1 case studies
│   │     eval_baselines_feynman.py     5 classical selectors vs. DenoisedSR
│   │     eval_srsd_recall.py           external SRSD-Feynman 120 suite
│   │     eval_noise_sweep.py           noise robustness η ∈ {0,0.01,0.05,0.10}
│   │     eval_gplearn_backend.py       second-backend (gplearn) test
│   │     eval_feynman_suite.py         AI-Feynman 118 recall
│   │     aggregate_sweep.py            multi-seed aggregation
│   │     fetch_srsd.py                 download SRSD-Feynman dummy from HF
│   └── analysis/              diagnostics, ablations (some training-only)
├── models/                    trained weights (see "Models")
│   ├── gat_best.pt            GAT variable-support predictor (PRIMARY)
│   ├── gat_operator.pt        GAT operator predictor (ablation)
│   └── cooc_graph.joblib      variable co-occurrence graph (GAT edges)
│   (support_predictor_v2_40k.joblib + operator_predictor_full.joblib — RF
│    partners, ~213 MB and ~358 MB — fetched separately via scripts/fetch_weights.sh)
├── data/
│   ├── benchmarks/            task manifests (AI-Feynman, PMLB, SRSD)
│   └── results/               every figure/table's backing JSON
├── requirements.txt
└── LICENSE
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```
PySR additionally installs a Julia backend on first use (`python -c "import pysr"`).

## Reproduce the figures

```bash
cd paper/code
python make_figures.py            # reads ../../data/results/*.json, writes ../figures/
```

## Reproduce the headline numbers

The five core paper experiments. Each reads model weights + benchmark data from
this repo and writes a JSON under `data/results/`. From the repo root:

```bash
# 1. Variable-support recall on AI-Feynman 118 (DenoisedSR vs. 5 classical selectors)
python frontend/eval/eval_baselines_feynman.py

# 2. External SRSD-Feynman 120 (first run fetches the SRSD-dummy txt files)
python frontend/eval/fetch_srsd.py        # ~30 MB download
python frontend/eval/eval_srsd_recall.py

# 3. Label-noise robustness sweep (η ∈ {0, 0.01, 0.05, 0.10})
python frontend/eval/eval_noise_sweep.py

# 4. PySR 3-way comparison (full | +var prior | +var+op prior). One seed ≈ 15 min on CPU.
SEED=42 N_DIST=20 N_TASKS=30 TIMEOUT=10 OUT_PATH=data/results/pysr_frontend_3way.json \
  python frontend/eval/eval_pysr_frontend.py
# For multi-seed variance: repeat with SEED=43, 44 and aggregate:
python frontend/eval/aggregate_sweep.py

# 5. Second backend: gplearn
SEED=42 python frontend/eval/eval_gplearn_backend.py
```

## Use the front-end (variable support → PySR)

```python
# pseudocode of the deployed interface (see frontend/eval/eval_pysr_frontend.py)
import joblib, torch
rf   = joblib.load("models/support_predictor_v2_40k.joblib")  # large RF (see below)
gat  = torch.load("models/gat_best.pt")
cooc = joblib.load("models/cooc_graph.joblib")

# score each column from observations (X, y); keep columns with combined score >= 0.10
score = rf_score(rf, X, y) + sigmoid(gat_score(gat, cooc, X, y))
support = [c for c in columns if score[c] >= 0.10]

# run PySR restricted to `support` instead of all columns
```

## Models

The three **small** weights needed for GAT inference are included directly:
`models/gat_best.pt`, `models/gat_operator.pt`, `models/cooc_graph.joblib`.

The ensemble's random-forest partners are **large** (212 MB – 601 MB) and are not
committed to git. To run the full RF+GAT ensemble, obtain them via Git LFS or the
release assets:

| File | Size | Role |
|---|---|---|
| `support_predictor_v2_40k.joblib` | 212 MB | RF variable-support (ensemble partner) |
| `operator_predictor_full.joblib`  | 357 MB | RF operator predictor (ablation baseline) |

Track them with Git LFS (`git lfs track "*.joblib"`) or attach them as release
assets / deposit on Zenodo. The GAT-only path runs from the committed weights.

## Citation

```bibtex
@article{srfrontend_commphys,
  title   = {A Graph-Attention Variable-Support Front-End that Accelerates
             Symbolic Regression of Physical Laws},
  author  = {Anonymous Author(s)},
  journal = {submitted to Communications Physics},
  year    = {2026}
}
```

## License

Code and artifacts released under the MIT License (see `LICENSE`). External
benchmarks (AI-Feynman, PMLB/SRBench) retain their original licenses.
