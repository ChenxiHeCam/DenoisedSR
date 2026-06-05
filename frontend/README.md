# Neural SR Front-End — Code

Graph-attention variable-support front-end for symbolic regression.
Companion code for the results in `../RESULTS_SUMMARY.md`.

## Layout

```
frontend/
├── models/          trained models (ship these)
│   ├── gat_best.pt                    GAT variable-support predictor (PRIMARY)
│   ├── support_predictor_v2_40k.joblib  RF variable-support (ensemble partner)
│   ├── gat_operator.pt                GAT operator predictor (ablation)
│   ├── operator_predictor_full.joblib   RF operator predictor (ablation baseline)
│   └── cooc_graph.joblib              variable co-occurrence graph (GAT edges)
├── train/           model training
├── eval/            evaluation + benchmarks
├── data_prep/       formula-pool construction
└── analysis/        diagnostics, ablations, case studies
```

## Pipeline

### 1. Data prep (`data_prep/`)
- `build_formula_pool.py`   — sample observation data for ~8k formulas (pre-sampled pool)
- `build_v3_pool.py`        — select 32k formulas from the v3 knowledge graph
- `presample_v3_pool.py`    — pre-sample v3 formulas (lazy → eager)

### 2. Training (`train/`)
- `train_support_predictor_v2.py`  — RF variable-support (statistical + unit features)
- `train_gat.py`                   — **GAT variable-support (the main model)**
- `train_gat_operator.py`          — GAT operator predictor (graph + behavioral features)
- `train_operator_predictor.py`    — RF operator predictor (behavioral features, baseline)
- `finetune_support_predictor.py`  — RF finetune on 10–30 distractor range
- `train_gan_gat.py`               — GAN-GAT experiment (abandoned: adversarial gave no gain)

### 3. Evaluation (`eval/`)
- `eval_support_predictor.py`  — precision/recall of variable support (real591)
- `eval_v5_suite.py`           — variable support on v5 physics65 (1085 formulas)
- `eval_feynman_suite.py`      — variable support on AI-Feynman (118)
- `eval_pysr_frontend.py`      — **3-way: Full PySR vs Ours-vars vs Ours-vars+ops**
- `eval_final_ensemble.py`     — RF+GAT ensemble voting strategies
- `eval_speed_vars.py`         — time-to-solution A (all cols) vs B (selected)
- `eval_numeric_equiv.py`      — numeric-equivalence judging (vs sympy)
- `eval_gat_operator.py`       — operator-predictor recall/precision
- `measure_frontend_overhead.py` — front-end inference time (40 ms)

### 4. Analysis (`analysis/`)
- `inspect_highr2.py`          — show spurious high-R² formulas PySR finds
- `compare_op_on_spurious.py`  — does operator restriction rescue spurious fits?
- `show_nonexact.py`           — list all high-R²-non-exact cases for human judging
- `analyze_recall_failures.py` — which variables get missed + prediction variance

## Key models to ship

For inference you need just three files:
```
models/gat_best.pt
models/support_predictor_v2_40k.joblib
models/cooc_graph.joblib
```
Select variables with: `score = RF(col) + sigmoid(GAT(col)); keep if score >= 0.10`.

## Headline results (see ../RESULTS_SUMMARY.md)

- Variable recall = **1.000** on real591 / v5 / AI-Feynman (perfect-recall 100%).
- PySR exact recovery **+13pp (q=100) → +28pp (q=500)**, 84% column reduction.
- Time-to-solution **3–6× faster** on typical formulas; front-end overhead ~40 ms.
- Operator prediction: net-neutral overall; useful only as an adaptive safeguard.
