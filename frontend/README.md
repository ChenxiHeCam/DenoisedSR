# Neural SR Front-End — Code

Graph-attention variable-support front-end for symbolic regression.
Companion code for the paper in `../paper/main.pdf`. See `../README.md` for
top-level overview and reproduction commands.

## Two kinds of scripts in this directory

Scripts here fall into **two categories**:

- **paper-reproducing**: self-contained, run from the release with the shipped
  benchmarks + models, write outputs into `../data/results/`. These are the
  only scripts a reviewer needs to verify any number in the paper.
- **training-only**: reproduce the training pipeline that *produced* the
  shipped models. They reference internal training data
  (`_FOUNDATION_TRAINING_DATASET_20260522/`,
  `stage9_opensidr_expert_route_expansion_manifest_*`,
  `stage8g_real_for_sure_*`) that is **not** included in this release because
  of size; those files still contain hard-coded `D:/Physics Fundation model/...`
  paths from the authors' tree. They are kept for transparency about how the
  models were trained but cannot be run from a fresh clone without recreating
  the training corpus.

## Paper-reproducing scripts

In `eval/` (each writes a JSON under `../data/results/` that the corresponding
figure consumes):

| Script | Reproduces | Output JSON |
|---|---|---|
| `eval_pysr_frontend.py`       | PySR 3-way + Table 1 cases | `pysr_frontend_3way.json` (+ `_formulas.json`) |
| `eval_baselines_feynman.py`   | Fig fig_baselines: 5 classical selectors | `baselines_feynman_recall.json` |
| `eval_srsd_recall.py`         | SRSD-Feynman external suite | `srsd_dummy_recall.json` |
| `eval_noise_sweep.py`         | Fig fig_noise: η ∈ {0,0.01,0.05,0.10} | `noise_sweep.json` |
| `eval_gplearn_backend.py`     | Fig fig_gplearn: second backend | `gplearn_backend_3way.json` |
| `eval_feynman_suite.py`       | AI-Feynman 118 recall + ablation | (stdout summary) |
| `aggregate_sweep.py`          | multi-seed PySR + distractor + gplearn aggregate | `multiseed_distractor_aggregate.json` |
| `fetch_srsd.py`               | one-off: pull SRSD-Feynman dummy files from Hugging Face | `../data/benchmarks/srsd/` |

In `eval/` other:

| Script | Purpose |
|---|---|
| `eval_speed_vars.py`          | per-formula time-to-solution (Fig fig6_speed) |
| `measure_frontend_overhead.py`| front-end inference time (~40 ms / task) |
| `eval_gat_operator.py`        | operator-predictor recall (ablation) |
| `eval_numeric_equiv.py`       | numeric-equivalence judging vs sympy |

## Training-only (NOT runnable from a fresh clone)

These all import / read data paths from the authors' training tree:

```
analysis/analyze_recall_failures.py
analysis/eval_noise_comparison.py
analysis/eval_pretrain.py
analysis/score_dist.py
analysis/verify_experiment.py
data_prep/build_formula_pool.py
eval/eval_ensemble.py
eval/eval_final_ensemble.py
eval/eval_support_predictor.py
eval/eval_v5_suite.py
eval/run_real591_fullvars.py
eval/run_real591_verify.py
train/train_support_predictor_v2.py
train/train_support_predictor_large.py
train/finetune_support_predictor.py
train/train_gan_gat.py        (abandoned experiment, kept for transparency)
```

To re-run any of these you would have to recreate the training corpus
(`_FOUNDATION_TRAINING_DATASET_20260522/nodes_full.jsonl` etc.) and the
intermediate stage8g / stage9 manifests. The shipped models are the artifacts
of these scripts; the artifacts in `data/results/` are what consume the models.

## Key models for inference

```
models/gat_best.pt                          1.6 MB   GAT variable-support
models/gat_operator.pt                      0.7 MB   GAT operator-support
models/cooc_graph.joblib                    0.4 MB   variable co-occurrence
models/support_predictor_v2_40k.joblib    ~213 MB    RF variable-support partner (fetch separately)
models/operator_predictor_full.joblib     ~358 MB    RF operator partner (fetch separately, ablation only)
```

The first three ship in this repo. The two RF partners are too large for Git
and must be downloaded (see `../scripts/fetch_weights.sh`). The deployed
ensemble combines RF and GAT: `score = RF(col) + sigmoid(GAT(col))`; keep
column iff `score >= 0.10`. A GAT-only fallback at the same threshold reaches
recall >= 0.995 on AI-Feynman (see `support_recall_deployed.json::ablation_feynman118`).

## Headline results (paper figures)

See `../README.md` for the headline table; numbers are computed by the
paper-reproducing scripts above from the artifacts in `../data/results/`.
