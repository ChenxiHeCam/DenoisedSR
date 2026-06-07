import sys, json, warnings, re
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import argparse, numpy as np, pysr, sympy as sp
from pathlib import Path
from run_pysr_pmlb_feynman_learned_prior import build_frontend_training, predict_prior, read_jsonl
from evaluate_stage8g_open_generation import sample_truth_points

TRAIN_RECORDS = Path('data/stage3/stage3_residual_proxy_s64_records.jsonl')
Q=100; HELDOUT=200; DIST_MIN=4; DIST_MAX=30; TIMEOUT=12; SEED=42

# distractors varies per formula at inference time too
args = argparse.Namespace(q=Q, heldout=HELDOUT, distractors=15,  # default midpoint
    timeout_seconds=TIMEOUT, support_threshold=0.25, min_prior_columns=4,
    max_prior_columns=10, op_threshold=0.28, seed=SEED, device='cpu')

rng0 = np.random.default_rng(SEED)
train_rows = read_jsonl(TRAIN_RECORDS, max_rows=512)
support_clf, op_clf, _ = build_frontend_training(train_rows, args, rng0)
print("Predictor ready.\n")


def extract_all_symbols(truth_surface):
    """Parse truth_surface and return all free sympy symbols."""
    # Strip the "X - (...) = 0" wrapper to get the expression
    ts = truth_surface.strip()
    # Remove "= 0" at end
    ts = re.sub(r'\s*=\s*0\s*$', '', ts)
    # If "LHS = RHS" form, take RHS
    if '=' in ts:
        parts = ts.split('=', 1)
        ts = parts[1].strip()
    try:
        expr = sp.sympify(ts, evaluate=False)
        syms = [str(s) for s in expr.free_symbols]
        return sorted(syms)
    except Exception:
        return []


def get_target_from_truth(truth_surface, free_syms):
    """Extract LHS variable name as target."""
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth_surface.strip())
    if m and m.group(1) in free_syms:
        return m.group(1)
    return free_syms[0] if free_syms else None


# Collect real591 formulas — now filter by actual parsed symbol count
seen, records = set(), []
base = Path('D:/Physics Fundation model/artifacts/stage9_opensidr_expert_route_expansion_manifest_20260513/route_outputs/real591')
for route_dir in sorted(base.iterdir()):
    p = route_dir / 'records.jsonl'
    if not p.exists(): continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            lid = row.get('original_law_id') or row.get('law_id', '')
            if lid in seen: continue
            truth = row.get('truth_surface', '')
            if not truth: continue

            # Extract actual free symbols from the formula
            free_syms = extract_all_symbols(truth)
            if len(free_syms) < 3 or len(free_syms) > 6:
                continue  # want 3-6 vars so PySR has a real challenge

            # Verify sample_truth_points works and returns all needed vars
            rng_chk = np.random.default_rng(0)
            vals, err = sample_truth_points(truth, 10, rng_chk)
            if vals is None or len(vals) < 3:
                continue
            # Check all free_syms are in vals
            missing = [s for s in free_syms if s not in vals]
            if missing:
                continue

            seen.add(lid)
            records.append({
                'law_id': lid,
                'truth_surface': truth,
                'symbols': free_syms,   # full symbol set from parsing
            })
            if len(records) >= 80:
                break
    if len(records) >= 80:
        break

# pick 30 spread across 3,4,5,6-var formulas
by_n = {}
for r in records:
    by_n.setdefault(len(r['symbols']), []).append(r)

tasks = []
for n in sorted(by_n):
    tasks.extend(by_n[n][:8])
    if len(tasks) >= 30:
        break
tasks = tasks[:30]
print(f"Running {len(tasks)} formulas (all variables complete, timeout={TIMEOUT}s)\n")


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

    target = get_target_from_truth(truth, syms)
    if target not in tr_vals:
        target = list(tr_vals.keys())[0]

    n_dist = int(rng_i.integers(DIST_MIN, DIST_MAX + 1))
    tr_aug = add_dist(tr_vals, target, n_dist, rng_i)
    te_aug = add_dist(te_vals, target, n_dist, rng_i)
    all_cols = [c for c in tr_aug if c != target]
    y_tr = tr_aug[target]
    y_te = te_aug[target]

    # Support predictor on full augmented columns
    pred_cols, _, _ = predict_prior(support_clf, op_clf, tr_aug, target, all_cols, args)
    pred_cols = [c for c in pred_cols if not c.startswith('__d')]
    if not pred_cols:
        pred_cols = syms

    row = {'law_id': task['law_id'], 'truth': truth, 'symbols': syms,
           'target': target, 'learned_cols': pred_cols}

    print(f"[{i+1:02d}] {task['law_id']}")
    print(f"  TRUTH:   {truth}")
    print(f"  all_vars={syms}  predicted={pred_cols}")

    for mode, cols in [
        ('full',    [c for c in syms + [f'__d{j}' for j in range(DISTRACTORS)]
                     if c in tr_aug and c != target]),
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
            ss_res = np.sum((y_te - yp) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-12))
        except Exception as e:
            best = f'ERROR:{e}'; r2 = -1.0
        row[f'{mode}_expr'] = best
        row[f'{mode}_r2']   = round(r2, 4)
        row[f'{mode}_ncols'] = len(cols)
        tag = 'full   ' if mode == 'full' else 'learned'
        print(f"  [{tag}] R2={r2:.4f}  ncols={len(cols)}  => {best}")

    print()
    results.append(row)


print("=" * 90)
print("FINAL SUMMARY — judge each formula manually:")
print("=" * 90)
for i, r in enumerate(results):
    print(f"\n[{i+1:02d}] {r['law_id']}")
    print(f"  TRUTH:   {r['truth']}")
    print(f"  full  ({r.get('full_ncols','?')} cols):    R2={r.get('full_r2','?')}  =>  {r.get('full_expr','?')}")
    print(f"  learned ({r.get('learned_ncols','?')} cols): R2={r.get('learned_r2','?')}  =>  {r.get('learned_expr','?')}")

out_path = Path('data/results/real591_fullvars_30_20260528.json')
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nSaved -> {out_path}")
