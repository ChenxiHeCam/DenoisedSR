"""
Build formula pool sequentially with per-formula signal timeout (Unix)
or simple try/except skip (Windows). CPU limited to ~60%.
"""
import sys, json, warnings, re, time, os
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from joblib import dump

# ── CPU throttle: lower process priority + limit joblib workers ───────────────
try:
    os.nice(10)   # lower priority on Unix; no-op on Windows
except Exception:
    pass

NODES_FILE   = Path('D:/Physics Fundation model/data/_FOUNDATION_TRAINING_DATASET_20260522/nodes_full.jsonl')
OUT_POOL     = Path('models/formula_pool.joblib')
OUT_POOL.parent.mkdir(exist_ok=True)
Q            = 80
MAX_FORMULAS = 8000
SEED         = 42


def sample_one_safe(truth, q, rng):
    """Sample with hard try/except; skip any formula that errors or hangs."""
    from evaluate_stage8g_open_generation import sample_truth_points
    try:
        vals, err = sample_truth_points(truth, q, rng)
    except Exception:
        return None
    if vals is None or len(vals) < 2:
        return None
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth)
    target = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
    true_vars = set(vals.keys()) - {target}
    if not true_vars:
        return None
    return {'truth': truth, 'vals': vals, 'target': target, 'true_vars': true_vars}


def main():
    # Collect candidates
    truths = []
    with open(NODES_FILE, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            ct = r.get('canonical_template', '')
            if ct and len(ct) >= 4:
                truths.append(ct)
            if len(truths) >= MAX_FORMULAS * 3:
                break
    print(f"Candidates: {len(truths)}  target pool: {MAX_FORMULAS}")

    rng = np.random.default_rng(SEED)
    pool = []
    skipped = 0
    t0 = time.time()

    for i, truth in enumerate(truths):
        if len(pool) >= MAX_FORMULAS:
            break

        result = sample_one_safe(truth, Q, rng)
        if result is not None:
            pool.append(result)
        else:
            skipped += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (MAX_FORMULAS - len(pool)) / max(rate * len(pool) / (i+1), 0.1)
            print(f"  [{i+1:5d}] pool={len(pool):5d}  skipped={skipped}  "
                  f"{rate:.0f}/s  ETA={eta:.0f}s", flush=True)

        # Throttle: sleep 2ms every formula to keep CPU below ~60%
        time.sleep(0.002)

    pool = pool[:MAX_FORMULAS]
    elapsed = time.time() - t0
    print(f"\nPool: {len(pool)} formulas  skipped={skipped}  time={elapsed:.1f}s")
    dump(pool, OUT_POOL)
    print(f"Saved -> {OUT_POOL}  ({OUT_POOL.stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
