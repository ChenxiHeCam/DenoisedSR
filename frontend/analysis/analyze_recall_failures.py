"""
1. Analyze which 22 formulas have imperfect recall (missed variables)
2. Run predictor twice per formula with different sampling seeds,
   measure how much the prediction changes (variance analysis)
"""
import sys, json, warnings, re
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
sys.path.insert(0, 'D:/Physics Fundation model/src')
sys.path.insert(0, 'D:/Physics Fundation model/scripts')

import numpy as np
from pathlib import Path
from joblib import load as joblib_load
from collections import Counter

from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import positive_probability
import train_support_predictor_v2 as v2

THRESHOLD   = 0.25
DIST_MIN    = 4
DIST_MAX    = 40
Q           = 100
SEED        = 42
N_RUNS      = 5   # how many times to re-run per formula to check variance

# Load model
ckpt = joblib_load('models/support_predictor_v2_ft.joblib')
support_clf = ckpt['support_clf']

def add_dist(vals, target, n, rng):
    out = dict(vals)
    m = len(next(iter(vals.values())))
    for i in range(n):
        out[f'__d{i}'] = rng.uniform(0.1, 3.0, m)
    return out

def extract_syms(truth):
    import sympy as sp
    ts = re.sub(r'\s*=\s*0\s*$', '', truth.strip())
    if '=' in ts: ts = ts.split('=', 1)[1].strip()
    try:
        return sorted(str(s) for s in sp.sympify(ts, evaluate=False).free_symbols)
    except: return []

def get_target(truth, syms):
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth.strip())
    return m.group(1) if m and m.group(1) in syms else (syms[0] if syms else None)

def predict_once(vals_aug, target, all_cols, seed_offset=0):
    y = vals_aug[target]
    x = np.asarray([v2.column_features(vals_aug[c], y, var_name=c)
                    for c in all_cols], dtype=np.float64)
    probs = positive_probability(support_clf, x)
    ranked = sorted(zip(all_cols, probs), key=lambda t: -float(t[1]))
    selected = [c for c, p in ranked if float(p) >= THRESHOLD]
    if len(selected) < 3:
        selected = [c for c, _ in ranked[:3]]
    return set(selected[:12])

# Collect formulas
seen, tasks = set(), []
base = Path('D:/Physics Fundation model/artifacts/stage9_opensidr_expert_route_expansion_manifest_20260513/route_outputs/real591')
for route_dir in sorted(base.iterdir()):
    p = route_dir / 'records.jsonl'
    if not p.exists(): continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            lid = row.get('original_law_id') or row.get('law_id','')
            if lid in seen: continue
            truth = row.get('truth_surface','')
            if not truth: continue
            syms = extract_syms(truth)
            if len(syms) < 2 or len(syms) > 6: continue
            rng_c = np.random.default_rng(0)
            vals, _ = sample_truth_points(truth, 5, rng_c)
            if vals is None or len(vals) < 2: continue
            missing = [s for s in syms if s not in vals]
            if missing: continue
            seen.add(lid)
            tasks.append({'law_id': lid, 'truth': truth, 'symbols': syms})
            if len(tasks) >= 100: break
    if len(tasks) >= 100: break

print(f"Analyzing {len(tasks)} formulas\n")

# Per-formula analysis
perfect = []
imperfect = []
variance_data = []

for task in tasks:
    syms  = list(task['symbols'])
    truth = task['truth']
    rng_i = np.random.default_rng(SEED + tasks.index(task))

    vals, _ = sample_truth_points(truth, Q, rng_i)
    if vals is None: continue
    target = get_target(truth, syms)
    if not target or target not in vals: continue

    true_vars = set(s for s in syms if s != target and s in vals)

    # Run N_RUNS times with different distractor seeds
    all_predictions = []
    for run in range(N_RUNS):
        rng_run  = np.random.default_rng(SEED + tasks.index(task) + run * 1000)
        n_dist   = int(rng_run.integers(DIST_MIN, DIST_MAX + 1))
        vals_aug = add_dist(vals, target, n_dist, rng_run)
        all_cols = [c for c in vals_aug if c != target]
        pred = predict_once(vals_aug, target, all_cols)
        pred_real = pred - {f'__d{i}' for i in range(DISTRACTORS)}
        all_predictions.append(pred_real)

    # Recall per run
    recalls = [len(p & true_vars) / len(true_vars) for p in all_predictions]
    perfect_runs = sum(r >= 1.0 for r in recalls)

    # Variance: how often does the set change across runs?
    base_pred = all_predictions[0]
    changes = sum(1 for p in all_predictions[1:] if p != base_pred)
    always_in   = set.intersection(*all_predictions)  # always selected
    ever_in     = set.union(*all_predictions)           # selected at least once
    unstable    = ever_in - always_in                   # flip-flops

    row = {
        'law_id':      task['law_id'],
        'truth':       truth[:60],
        'true_vars':   sorted(true_vars),
        'n_vars':      len(true_vars),
        'mean_recall': float(np.mean(recalls)),
        'perfect_runs': perfect_runs,
        'changes':     changes,
        'always_selected': sorted(always_in),
        'unstable_vars':   sorted(unstable),
        'missed_vars':     sorted(true_vars - ever_in),  # NEVER selected
    }

    if perfect_runs == N_RUNS:
        perfect.append(row)
    else:
        imperfect.append(row)
    variance_data.append(row)

# ── Print results ─────────────────────────────────────────────────────────────

print(f"Perfect recall (all {N_RUNS} runs): {len(perfect)}/{len(variance_data)}")
print(f"Imperfect (at least 1 run missed):  {len(imperfect)}/{len(variance_data)}\n")

print("=" * 80)
print("IMPERFECT RECALL CASES — what's being missed:")
print("=" * 80)
for r in sorted(imperfect, key=lambda x: x['mean_recall']):
    print(f"\n  {r['law_id']}")
    print(f"  truth:   {r['truth']}")
    print(f"  vars:    {r['true_vars']}")
    print(f"  recall:  {r['mean_recall']:.2f} ({r['perfect_runs']}/{N_RUNS} perfect runs)")
    if r['missed_vars']:
        print(f"  NEVER selected:    {r['missed_vars']}  ← always dropped")
    if r['unstable_vars']:
        print(f"  unstable (flip):   {r['unstable_vars']}  ← sometimes in/out")
    if r['changes'] > 0:
        print(f"  prediction changed {r['changes']}/{N_RUNS-1} runs")

# ── Variance summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VARIANCE SUMMARY (across all formulas):")
print("=" * 80)
total_changes = sum(r['changes'] for r in variance_data)
any_change    = sum(1 for r in variance_data if r['changes'] > 0)
print(f"  Formulas where prediction changed across {N_RUNS} runs: {any_change}/{len(variance_data)} ({100*any_change/len(variance_data):.0f}%)")
print(f"  Total prediction changes: {total_changes} / {len(variance_data)*(N_RUNS-1)} runs")
print(f"  → {'HIGH variance — resampling changes results significantly' if any_change > len(variance_data)*0.3 else 'LOW variance — predictor is fairly stable across resampling'}")

# What variable types are most commonly missed?
missed_counts = Counter()
for r in imperfect:
    for v in r['missed_vars'] + r['unstable_vars']:
        missed_counts[v] += 1

if missed_counts:
    print(f"\nMost frequently missed/unstable variables:")
    for var, cnt in missed_counts.most_common(15):
        tag = v2.SYMBOL_TAGS.get(var, ('unknown', 0))
        print(f"  {var:20s}  missed {cnt}x  semantic={tag[0]}")
