"""
Train operator predictor v2 with BEHAVIORAL features.

The key insight: global statistics (col count, std) cannot reveal which
operators a formula uses. We need features that expose operator signatures
in the relationship between each input column and the target:
  - periodicity (FFT)         -> sin / cos / tan
  - log-linearity             -> exp / log
  - power-law slope (log-log) -> sqrt / square / cube / ^
  - sign changes, curvature   -> general nonlinearity

Full operator vocabulary:
  binary: + - * / ^
  unary:  sqrt sin cos tan exp log sinh cosh tanh abs square cube
"""
import sys, re, time, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'D:/Physics Fundation model/src')
sys.path.insert(0, 'D:/Physics Fundation model/scripts')

import numpy as np
from pathlib import Path
from joblib import load as jload, dump as jdump
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import train_support_predictor_v2 as v2

try:
    import psutil; psutil.Process().nice(10)
except: pass

OP_FULL = ['+','-','*','/','^',
           'sqrt','sin','cos','tan','exp','log',
           'sinh','cosh','tanh','abs','square','cube']

Q      = 80
SEED   = 42
N_JOBS = max(1, int(os.cpu_count() * 0.6))
OUT    = Path('models/operator_predictor_full.joblib')


def infer_ops_full(expr_str):
    s = expr_str.replace('**', '^')
    found = set()
    for op in ['+','-','*','/','^']:
        if op in s: found.add(op)
    patterns = {
        'sqrt': r'\bsqrt\b', 'sin': r'\bsin\b', 'cos': r'\bcos\b',
        'tan': r'\btan\b', 'exp': r'\bexp\b', 'log': r'\b(log|ln)\b',
        'sinh': r'\bsinh\b', 'cosh': r'\bcosh\b', 'tanh': r'\btanh\b',
        'abs': r'\b(abs|Abs)\b',
    }
    for op, pat in patterns.items():
        if re.search(pat, s): found.add(op)
    if re.search(r'\^\s*2\b', s) or re.search(r'\*\*\s*2\b', s): found.add('square')
    if re.search(r'\^\s*3\b', s) or re.search(r'\*\*\s*3\b', s): found.add('cube')
    return found


def add_noise(vals, target, n, rng):
    out = dict(vals); q = len(vals[target])
    for i in range(n):
        lo, hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        out[f'__d{i}'] = v2.resample(np.array([lo,hi]), q, rng)
    return out


def behavioral_features(vals, target, cols):
    """
    Extract operator-revealing behavioral features.
    Aggregates per-column signals (over REAL columns, ignoring obvious noise)
    into a fixed-length task-level vector.
    """
    y = np.asarray(vals[target], dtype=np.float64)
    eps = 1e-9
    n = len(y)

    per_col = []
    for c in cols:
        x = np.asarray(vals[c], dtype=np.float64)
        if np.std(x) < eps: continue
        order = np.argsort(x)
        xs, ys = x[order], y[order]

        feats = []
        # 1. linear corr
        lc = np.corrcoef(x, y)[0,1]; feats.append(0. if not np.isfinite(lc) else lc)
        # 2. log-linearity: corr(log|x|, y) -> exp/log signature
        lx = np.log(np.abs(x)+eps)
        llc = np.corrcoef(lx, y)[0,1]; feats.append(0. if not np.isfinite(llc) else llc)
        # 3. exp signature: corr(x, log|y|)
        ly = np.log(np.abs(y)+eps)
        elc = np.corrcoef(x, ly)[0,1]; feats.append(0. if not np.isfinite(elc) else elc)
        # 4. power-law slope: log-log linear fit slope
        mask = (x>0)&(y>0)
        if mask.sum() > 10:
            sl = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0]
            feats.append(np.clip(sl, -5, 5))
        else:
            feats.append(0.)
        # 5. periodicity: FFT peak power ratio (sin/cos signature)
        #    sort by x, look at residual y after removing trend, FFT
        if n >= 16:
            yd = ys - np.polyval(np.polyfit(np.arange(n), ys, 1), np.arange(n))
            sp = np.abs(np.fft.rfft(yd - yd.mean()))
            if len(sp) > 2 and sp[1:].sum() > eps:
                peak_ratio = sp[1:].max() / (sp[1:].sum()+eps)
            else:
                peak_ratio = 0.
            feats.append(float(peak_ratio))
        else:
            feats.append(0.)
        # 6. monotonicity (rank corr) -> distinguishes oscillation vs monotone
        rc = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0,1]
        feats.append(0. if not np.isfinite(rc) else rc)
        # 7. curvature: 2nd-diff magnitude of sorted y
        if n > 3:
            d2 = np.diff(ys, 2)
            feats.append(float(np.clip(np.std(d2)/(np.std(ys)+eps), 0, 10)))
        else:
            feats.append(0.)
        # 8. sign changes in dy/dx (oscillation count) -> trig
        dy = np.diff(ys)
        sign_changes = np.sum(np.diff(np.sign(dy)) != 0) / max(n,1)
        feats.append(float(sign_changes))

        per_col.append(feats)

    per_col = np.array(per_col) if per_col else np.zeros((1, 8))

    # Aggregate per-column features: max/mean/std across columns
    agg = np.concatenate([
        per_col.max(axis=0), per_col.mean(axis=0), per_col.std(axis=0),
        np.abs(per_col).max(axis=0),
    ])
    # Global features
    glob = np.array([
        len(cols),
        float(np.std(y)),
        float(np.mean(np.abs(y))),
        float((y < 0).mean()),     # negative fraction in target
        float(np.log10(np.max(np.abs(y))/(np.min(np.abs(y))+eps)+1)),  # dynamic range
    ])
    return np.concatenate([agg, glob]).astype(np.float32)


# ── Build training data ───────────────────────────────────────────────────────
def main():
  pool = jload('models/formula_pool.joblib')
  pool = [f for f in pool if f['vals'] is not None]
  print(f"Pool: {len(pool)} formulas, {len(OP_FULL)} ops, BEHAVIORAL features\n")

  X, Y = [], []
  op_counts = {op: 0 for op in OP_FULL}
  rng = np.random.default_rng(SEED)
  t0 = time.time()

  for idx, formula in enumerate(pool):
    vals = formula['vals']; target = formula['target']; truth = formula['truth']
    n_noise = int(rng.integers(2, 15))
    aug = add_noise(vals, target, n_noise, rng)
    cols = [c for c in aug if c != target]

    feat = behavioral_features(aug, target, cols)
    ops  = infer_ops_full(truth)
    for op in ops:
        if op in op_counts: op_counts[op] += 1
    X.append(feat); Y.append([int(op in ops) for op in OP_FULL])

    if (idx+1) % 1000 == 0:
        print(f"  {idx+1}/{len(pool)}  feat_dim={len(feat)}", flush=True)

  X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0., posinf=100., neginf=-100.)
  Y = np.array(Y)
  print(f"\nData: {X.shape[0]} examples, {X.shape[1]} features, {Y.shape[1]} op labels")
  print("Operator frequencies:")
  for op in OP_FULL:
      print(f"  {op:8s}: {100*op_counts[op]/len(pool):.1f}%")

  print("\nTraining multi-label RF...")
  t1 = time.time()
  op_clf = MultiOutputClassifier(RandomForestClassifier(
      n_estimators=300, min_samples_leaf=2,
      class_weight='balanced_subsample', random_state=SEED, n_jobs=N_JOBS))
  op_clf.fit(X, Y)
  print(f"  {time.time()-t1:.1f}s")

  jdump({'op_clf': op_clf, 'op_vocab': OP_FULL,
         'feat_fn': 'behavioral', 'n_train': len(pool),
         'version': 'operator_behavioral_v2'}, OUT)
  print(f"\nSaved -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
  print(f"Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
