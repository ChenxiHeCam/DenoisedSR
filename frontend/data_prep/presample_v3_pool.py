"""
Pre-sample v3 formulas in batches with per-formula timeout.
Saves incrementally every 2000 formulas.
"""
import sys, json, re, time
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from joblib import load, dump

RAW   = Path('models/formula_pool_v3_32k.joblib')
OUT   = Path('models/formula_pool_v3_sampled.joblib')
Q     = 80
SEED  = 42

raw   = load(RAW)
print(f"Pre-sampling {len(raw)} v3 formulas...")
rng   = np.random.default_rng(SEED)
done  = []
t0    = time.time()

from evaluate_stage8g_open_generation import sample_truth_points

for i, formula in enumerate(raw):
    truth = formula['truth']
    try:
        vals, _ = sample_truth_points(truth, Q, rng)
    except Exception:
        continue
    if vals is None or len(vals) < 2: continue
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth)
    target = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
    true_vars = set(vals.keys()) - {target}
    if not true_vars: continue
    done.append({'truth': truth, 'vals': vals, 'target': target, 'true_vars': true_vars})

    if len(done) % 2000 == 0:
        rate = len(done) / (time.time() - t0 + 1)
        print(f"  {len(done):,} sampled  {rate:.1f}/s", flush=True)
        dump(done, OUT)  # incremental save

dump(done, OUT)
print(f"\nDone: {len(done):,}/{len(raw)} sampled in {time.time()-t0:.0f}s")
print(f"Saved -> {OUT}")
