"""
Retrain the support predictor using ~17k formulas from foundation_nodes dataset.
Saves the trained classifiers to models/support_predictor_large.joblib
"""
import sys, json, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import argparse, numpy as np
from pathlib import Path
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import (
    column_features, task_features, infer_symbols, infer_ops, OP_ORDER,
    add_distractors as _add_distractors_orig
)

NODES_FILE  = Path('D:/Physics Fundation model/data/_FOUNDATION_TRAINING_DATASET_20260522/nodes_full.jsonl')
OUT_MODEL   = Path('models/support_predictor_large.joblib')
OUT_MODEL.parent.mkdir(exist_ok=True)

Q           = 80
DISTRACTORS = 10
MAX_ROWS    = 20000
SEED        = 42

def add_distractors(vals, target, n, rng):
    out = dict(vals)
    m = len(next(iter(vals.values())))
    for i in range(n):
        out[f'__d{i}'] = rng.uniform(0.1, 3.0, m)
    return out

def build_training(nodes_path, max_rows, q, distractors, seed):
    rng = np.random.default_rng(seed)
    support_x, support_y = [], []
    op_x, op_y = [], []
    used = skipped = 0

    with open(nodes_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            truth = r.get('canonical_template', '')
            if not truth or len(truth) < 4: continue

            vals, err = sample_truth_points(truth, q, rng)
            if vals is None or len(vals) < 2:
                skipped += 1
                continue

            # pick target = first var (LHS of equation)
            import re
            lhs_match = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*[-\s]', truth)
            keys = list(vals.keys())
            target = lhs_match.group(1) if lhs_match and lhs_match.group(1) in vals else keys[0]

            vals_aug = add_distractors(vals, target, distractors, rng)
            columns  = sorted([c for c in vals_aug if c != target])
            symbols  = infer_symbols(truth)
            y_vals   = vals_aug[target]

            for col in columns:
                support_x.append(column_features(vals_aug[col], y_vals))
                support_y.append(int(col in symbols and col != target))

            op_x.append(task_features(vals_aug, target, columns))
            ops = infer_ops(truth)
            op_y.append([int(op in ops) for op in OP_ORDER])

            used += 1
            if used % 1000 == 0:
                print(f'  processed {used:,} formulas  (skipped {skipped})', flush=True)
            if used >= max_rows:
                break

    print(f'\nTotal: {used:,} formulas, {len(support_x):,} column examples, {skipped} skipped')
    return support_x, support_y, op_x, op_y, used

print(f'Building training data from {NODES_FILE.name} (max={MAX_ROWS:,})...')
t0 = time.time()
sx, sy, ox, oy, n_used = build_training(NODES_FILE, MAX_ROWS, Q, DISTRACTORS, SEED)
print(f'Data build: {time.time()-t0:.1f}s')

print('\nTraining support classifier (RF n=300)...')
t1 = time.time()
support_clf = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=3,
    class_weight='balanced_subsample', random_state=SEED, n_jobs=-1)
support_clf.fit(np.asarray(sx), np.asarray(sy))
print(f'  done in {time.time()-t1:.1f}s')

print('Training operator classifier (RF n=200)...')
t2 = time.time()
op_clf = MultiOutputClassifier(RandomForestClassifier(
    n_estimators=200, min_samples_leaf=3,
    class_weight='balanced_subsample', random_state=SEED+1, n_jobs=-1))
op_clf.fit(np.asarray(ox), np.asarray(oy))
print(f'  done in {time.time()-t2:.1f}s')

dump({'support_clf': support_clf, 'op_clf': op_clf,
      'n_train': n_used, 'q': Q, 'distractors': DISTRACTORS}, OUT_MODEL)
print(f'\nSaved -> {OUT_MODEL}  ({OUT_MODEL.stat().st_size/1e6:.1f} MB)')
print(f'Total time: {time.time()-t0:.1f}s')
