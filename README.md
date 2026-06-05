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
| Variable-support **recall** (AI-Feynman / real591 / extended set) | **1.000 / 1.000 / 0.999** (≈100% perfect-recall) |
| PySR exact-law recovery, fixed 10 s budget (30 Feynman, ~20 distractors) | **30% → 60%** (+30 pp) |
| Mean held-out R² | 0.919 → 0.967 |
| Column-space reduction | **86%** (23.2 → 3.2 cols) |
| Search-vocabulary reduction | 40 → 20 (variables), → 13 (+operators) |
| Time-to-solution speedup (typical formulas) | **3–6×** |
| Front-end inference overhead | ~40 ms / task |

A **same-size random-support** control performs far worse, showing the gain is
search guidance, not dimensionality reduction. Operator restriction is net-neutral
on average but a useful **adaptive safeguard** that rescues pathological high-R²
fits (e.g. R² 0.19 → 0.92).

All numbers are computed by the scripts in `frontend/eval/` from the recorded
artifacts in `data/results/`.

---

## Repository layout

```
.
├── paper/                     manuscript (Communications Physics)
│   ├── main.tex / main.pdf
│   ├── references.bib
│   ├── make_figures.py        regenerates all figures from data/results/
│   └── figures/               fig1_concept … fig5_operator_safeguard (PDF+PNG)
├── frontend/                  front-end code
│   ├── data_prep/             formula-pool construction (knowledge graph → tasks)
│   ├── train/                 train_gat.py, train_support_predictor_v2.py, …
│   ├── eval/                  evaluation + benchmarks (support, PySR, speed, ops)
│   └── analysis/              diagnostics, ablations, case studies
├── models/                    small, ship-ready model weights (see "Models")
│   ├── gat_best.pt            GAT variable-support predictor (PRIMARY)
│   ├── gat_operator.pt        GAT operator predictor (ablation)
│   └── cooc_graph.joblib      variable co-occurrence graph (GAT edges)
├── data/
│   ├── benchmarks/            task manifests (AI-Feynman, PMLB/SRBench)
│   └── results/               recorded result JSONs (tables + figures derive from these)
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
cd paper
python make_figures.py            # reads ../data/results/*.json, writes figures/
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
