import sys, json, warnings, re
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import argparse, numpy as np, pysr
from pathlib import Path
from run_pysr_pmlb_feynman_learned_prior import build_frontend_training, predict_prior, read_jsonl
from evaluate_stage8g_open_generation import sample_truth_points

TRAIN_RECORDS = Path('data/stage3/stage3_residual_proxy_s64_records.jsonl')
Q=100; HELDOUT=200; DISTRACTORS=10; TIMEOUT=10; SEED=42

args = argparse.Namespace(q=Q, heldout=HELDOUT, distractors=DISTRACTORS,
    timeout_seconds=TIMEOUT, support_threshold=0.55, min_prior_columns=2,
    max_prior_columns=8, op_threshold=0.28, seed=SEED, device='cpu')

rng0 = np.random.default_rng(SEED)
train_rows = read_jsonl(TRAIN_RECORDS, max_rows=512)
support_clf, op_clf, _ = build_frontend_training(train_rows, args, rng0)
print("Predictor ready.")

# Collect real591 formulas (2-4 vars)
seen, records = set(), []
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
            syms = row.get('symbols') or []
            if not truth or not syms or len(syms) < 2 or len(syms) > 4: continue
            rng_chk = np.random.default_rng(0)
            vals, err = sample_truth_points(truth, 10, rng_chk)
            if vals is None or len(vals) < 2: continue
            seen.add(lid)
            records.append({'law_id': lid, 'truth_surface': truth, 'symbols': syms})
            if len(records) >= 50: break
    if len(records) >= 50: break

tasks = records[:30]
print(f"Running {len(tasks)} formulas, timeout={TIMEOUT}s each\n")

def add_dist(vals, target, n, rng):
    out = dict(vals)
    m = len(next(iter(vals.values())))
    for i in range(n):
        out[f'__d{i}'] = rng.uniform(0.1, 3.0, m)
    return out

results = []
for i, task in enumerate(tasks):
    syms  = list(task['symbols'])
    truth = task['truth_surface']
    rng_i = np.random.default_rng(SEED + i)

    tr_vals, err = sample_truth_points(truth, Q, rng_i)
    te_vals, _   = sample_truth_points(truth, HELDOUT, rng_i)
    if tr_vals is None:
        print(f"[{i+1:02d}] SKIP {task['law_id']} ({err})\n")
        continue

    lhs_match = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*[-\s]', truth)
    target = lhs_match.group(1) if lhs_match and lhs_match.group(1) in tr_vals else list(tr_vals.keys())[0]

    tr_aug = add_dist(tr_vals, target, DISTRACTORS, rng_i)
    te_aug = add_dist(te_vals, target, DISTRACTORS, rng_i)
    all_cols = [c for c in tr_aug if c != target]
    y_tr = tr_aug[target]
    y_te = te_aug[target]

    pred_cols, _, _ = predict_prior(support_clf, op_clf, tr_aug, target, all_cols, args)
    pred_cols = [c for c in pred_cols if not c.startswith('__d')]
    if not pred_cols:
        pred_cols = syms

    row = {'law_id': task['law_id'], 'truth': truth, 'symbols': syms,
           'target': target, 'learned_cols': pred_cols}

    print(f"[{i+1:02d}] {task['law_id']}")
    print(f"  TRUTH:   {truth}")
    print(f"  vars={syms}  predicted={pred_cols}")

    for mode, cols in [
        ('full',    [c for c in syms + [f'__d{j}' for j in range(DISTRACTORS)] if c in tr_aug and c != target]),
        ('learned', [c for c in pred_cols if c in tr_aug and c != target])
    ]:
        if not cols:
            cols = [c for c in syms if c in tr_aug and c != target]
        X_tr = np.column_stack([tr_aug[c] for c in cols])
        X_te = np.column_stack([te_aug[c] for c in cols])
        model = pysr.PySRRegressor(
            niterations=60, timeout_in_seconds=TIMEOUT,
            binary_operators=['+', '-', '*', '/'],
            unary_operators=['sqrt', 'sin', 'cos', 'exp', 'log'],
            verbosity=0, random_state=SEED + i)
        try:
            model.fit(X_tr, y_tr)
            best = str(model.sympy())
            yp   = model.predict(X_te)
            ss_res = np.sum((y_te - yp)**2)
            ss_tot = np.sum((y_te - y_te.mean())**2)
            r2 = float(1 - ss_res / (ss_tot + 1e-12))
        except Exception as e:
            best = f'ERROR:{e}'; r2 = -1.0
        row[f'{mode}_expr'] = best
        row[f'{mode}_r2']   = round(r2, 4)
        tag = 'full   ' if mode == 'full' else 'learned'
        print(f"  [{tag}] R2={r2:.4f}  => {best}")

    print()
    results.append(row)

print("=" * 80)
print("SUMMARY for manual inspection:")
print("=" * 80)
for i, r in enumerate(results):
    print(f"\n[{i+1:02d}] {r['law_id']}")
    print(f"  TRUTH:   {r['truth']}")
    print(f"  full:    {r.get('full_expr','?')}   R2={r.get('full_r2','?')}")
    print(f"  learned: {r.get('learned_expr','?')}   R2={r.get('learned_r2','?')}")

out_path = Path('data/results/real591_pysr_formulas_30_20260528.json')
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nSaved -> {out_path}")
