"""
Finetune support predictor on harder examples: 10-30 confounders only.
Loads the v2 base model, generates new training data in the 10-30 distractor
range, then fits additional trees (warm_start) to the existing forest.
"""
import sys, json, warnings, re, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from joblib import load as joblib_load, dump as joblib_dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import infer_symbols, infer_ops, OP_ORDER
from train_support_predictor_v2 import (
    load_formula_pool, pick_confounders, resample,
    column_features, task_features,
    NODES_FILE, SEED, Q,
    N_CONFOUNDER_MIN, N_CONFOUNDER_MAX,
)

BASE_MODEL  = Path('models/support_predictor_v2.joblib')
OUT_MODEL   = Path('models/support_predictor_v2_ft.joblib')
MAX_FT      = 3000   # finetune formulas
FT_SEED     = SEED + 999

# Finetune distractor range: harder, focused on 10-30
FT_CONF_MIN = 10
FT_CONF_MAX = 30
FT_RAND_MIN = 3
FT_RAND_MAX = 10

def build_ft_data(pool, seed):
    rng = np.random.default_rng(seed)
    sx, sy, ox, oy = [], [], [], []

    for idx, formula in enumerate(pool):
        vals      = formula['vals']
        target    = formula['target']
        truth     = formula['truth']
        true_vars = formula['true_vars']
        q         = len(vals[target])

        # Harder range: 10-30 confounders
        n_conf = int(rng.integers(FT_CONF_MIN, FT_CONF_MAX + 1))
        n_rand = int(rng.integers(FT_RAND_MIN, FT_RAND_MAX + 1))

        aug = {target: vals[target]}
        for v in true_vars:
            aug[v] = resample(vals[v], q, rng)

        aug.update(pick_confounders(pool, idx, true_vars | {target}, n_conf, rng, q))

        for i in range(n_rand):
            lo_r = float(rng.uniform(-5, 0))
            hi_r = float(rng.uniform(0, 5))
            aug[f'__r{i}'] = resample(np.array([lo_r, hi_r]), q, rng)

        all_cols = sorted(c for c in aug if c != target)
        y_vals   = aug[target]
        symbols  = infer_symbols(truth)

        for col in all_cols:
            sx.append(column_features(aug[col], y_vals, var_name=col))
            sy.append(int(col in symbols and col in true_vars))

        ox.append(task_features(aug, target, all_cols))
        ops = infer_ops(truth)
        oy.append([int(op in ops) for op in OP_ORDER])

        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(pool)}  examples={len(sx):,}  pos={np.mean(sy):.2%}", flush=True)

    print(f"FT data: {len(pool):,} formulas  {len(sx):,} examples  pos={np.mean(sy):.2%}")
    return sx, sy, ox, oy


if __name__ == '__main__':
    if not BASE_MODEL.exists():
        print(f"Base model not found: {BASE_MODEL}")
        print("Run train_support_predictor_v2.py first.")
        sys.exit(1)

    print(f"Loading base model from {BASE_MODEL}...")
    ckpt = joblib_load(BASE_MODEL)
    support_clf = ckpt['support_clf']
    op_clf      = ckpt['op_clf']
    print(f"  Base trained on {ckpt.get('n_train','?')} formulas, "
          f"{len(support_clf.estimators_)} trees")

    t0 = time.time()
    # Reuse existing pool cache (avoids sympy hang on pool rebuild)
    POOL_CACHE = Path('models/formula_pool.joblib')
    if POOL_CACHE.exists():
        from joblib import load as _jl
        pool = _jl(POOL_CACHE)
        # shuffle and take a different slice for finetune variety
        rng_sh = np.random.default_rng(FT_SEED)
        idx = rng_sh.permutation(len(pool))[:MAX_FT]
        pool = [pool[i] for i in idx]
        print(f"Loaded {len(pool)} formulas from cache (shuffled subset)")
    else:
        print(f"\nBuilding finetune pool ({MAX_FT} formulas)...")
        pool = load_formula_pool(NODES_FILE, MAX_FT, Q, FT_SEED)

    print("Building finetune training data...")
    sx, sy, ox, oy = build_ft_data(pool, FT_SEED)
    print(f"Data: {time.time()-t0:.1f}s\n")

    # Warm-start: add 200 more trees on hard examples
    print("Finetuning support classifier (+200 trees, warm_start)...")
    t1 = time.time()
    n_orig = len(support_clf.estimators_)
    support_clf.set_params(
        n_estimators=n_orig + 200,
        warm_start=True,
    )
    support_clf.fit(np.asarray(sx, dtype=np.float32), np.asarray(sy))
    print(f"  {n_orig} -> {len(support_clf.estimators_)} trees  ({time.time()-t1:.1f}s)")

    print("Finetuning operator classifier (+100 trees)...")
    t2 = time.time()
    for est in op_clf.estimators_:
        n_orig_op = len(est.estimators_)
        est.set_params(n_estimators=n_orig_op + 100, warm_start=True)
        est.fit(np.asarray(ox), np.asarray(oy)[:, op_clf.estimators_.index(est)])
    print(f"  done ({time.time()-t2:.1f}s)")

    ckpt.update({
        'n_ft_formulas': len(pool),
        'ft_conf_range': [FT_CONF_MIN, FT_CONF_MAX],
        'ft_rand_range': [FT_RAND_MIN, FT_RAND_MAX],
        'version': 'v2_ft_10to30',
        'support_clf': support_clf,
        'op_clf': op_clf,
    })
    joblib_dump(ckpt, OUT_MODEL)
    print(f"\nSaved -> {OUT_MODEL}  ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")
    print(f"Total finetune time: {time.time()-t0:.1f}s")
