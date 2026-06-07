# Raw / superseded result artifacts

Files in this directory are **not** referenced by `paper/main.tex` or by any
paper-reproducing eval script. They are kept here for transparency about the
intermediate / superseded runs that preceded the final results.

| File | Role |
|---|---|
| `feynman_exact_recovery_30_20260528.json` | Early single-seed exact-recovery snapshot; superseded by `../pysr_frontend_3way.json` and `../multiseed_distractor_aggregate.json`. |
| `pysr_easy4_q100_d10_t20.json` | Early 4-formula PySR smoke run; superseded by the 30-formula 3-way. |
| `pysr_frontend_comparison.json` | Older comparison table; superseded by `../pysr_frontend_3way.json`. |
| `pysr_manifest_local_s8_q100_noise0_d10_t20_ablation_20260506.json` | Manifest of an older ablation grid; superseded by `../sweep/`. |
| `real591_fullvars_30_20260528.json` | Older real591 full-vars run; superseded by `../pysr_frontend_3way.json`. |
| `real591_pysr_formulas_30_20260528.json` | Older real591 case-study formulas; superseded by `../pysr_frontend_3way_formulas.json`. |
| `verify_rerun_20260528.json` | Internal verification rerun log. |
| `numeric_equiv_rows.json` | Numeric-equivalence judging output (legacy). |

The figure-reproducing artifacts the paper actually depends on live in the
parent `data/results/` directory.
