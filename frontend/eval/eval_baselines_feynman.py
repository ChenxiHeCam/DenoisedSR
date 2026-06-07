"""
Baseline feature-selector comparison on AI-Feynman variable-support recall.

Question: can a 5-line sklearn baseline (Lasso / MI / RF-on-(X,y)) match
DenoisedSR's recall ~ 1.000 on physics laws? If yes, the "learned vs random"
framing collapses. If no (which we expect on nonlinear couplings), the
learned prior contributes irreducible information.

Setup: 118 AI-Feynman formulas, q=100 obs, N_DIST=20 random distractors,
seeded matching eval_feynman_suite.py. Each baseline produces a per-column
score; two operating points:
  (a) oracle top-k where k = |true vars| (gives baselines the budget for free)
  (b) generic threshold (mean + 0.5*std of column scores)
We report recall, precision, and "perfect-recall fraction" (tasks where every
true variable is selected).
"""
import sys, json, os, warnings, time
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None

import numpy as np, sympy as sp
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import train_support_predictor_v2 as v2

SEED = int(os.environ.get("SEED","42"))
Q    = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
OUT_PATH = os.environ.get("OUT_PATH","data/results/baselines_feynman_recall.json")

tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks_raw = [t for t in tasks_raw if len(t["features"])>=2]
print(f"Feynman tasks loaded: {len(tasks_raw)}, q={Q}, n_dist={N_DIST}, seed={SEED}\n")

def sample(task, n, rng):
    formula = task["formula"]; features = task["features"]; ranges = task.get("ranges",{})
    if "=" not in formula: return None, None
    lhs, rhs = formula.split("=",1); target = lhs.strip()
    try:
        syms = {f: sp.Symbol(f) for f in features}
        fn = sp.lambdify([syms[f] for f in features], sp.sympify(rhs.strip(), locals=syms), "numpy")
    except: return None, None
    cols = {}
    for f in features:
        rg = ranges.get(f,[1.0,5.0]); lo,hi = float(rg[0]), float(rg[1])
        if lo>=hi: hi = lo+1
        cols[f] = rng.uniform(lo,hi,n)
    try:
        y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except: return None, None
    if not np.all(np.isfinite(y)): return None, None
    cols[target] = y
    return cols, target

def pearson_scores(X, y):
    s = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        v = X[:,j]
        if np.std(v)<1e-12: s[j]=0; continue
        s[j] = abs(np.corrcoef(v,y)[0,1])
    return np.nan_to_num(s)

def spearman_scores(X, y):
    s = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        try: s[j] = abs(spearmanr(X[:,j], y).statistic)
        except: s[j] = 0
    return np.nan_to_num(s)

def mi_scores(X, y):
    try: return mutual_info_regression(X, y, random_state=0)
    except: return np.zeros(X.shape[1])

def lasso_scores(X, y):
    try:
        Xs = (X - X.mean(0)) / (X.std(0)+1e-12)
        ys = (y - y.mean()) / (y.std()+1e-12)
        m = LassoCV(cv=5, random_state=0, max_iter=2000, n_alphas=20).fit(Xs, ys)
        return np.abs(m.coef_)
    except: return np.zeros(X.shape[1])

def rf_scores(X, y):
    try:
        m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1).fit(X, y)
        return m.feature_importances_
    except: return np.zeros(X.shape[1])

BASELINES = {
    "pearson_abs": pearson_scores,
    "spearman_abs": spearman_scores,
    "mutual_info": mi_scores,
    "lasso_cv":   lasso_scores,
    "rf_importance": rf_scores,
}

def select_topk(scores, k):
    if k>=len(scores): return set(range(len(scores)))
    return set(np.argsort(-scores)[:k].tolist())

def select_threshold(scores):
    # generic: keep cols with score >= mean + 0.5*std (no oracle info)
    thr = scores.mean() + 0.5*scores.std()
    return {i for i,s in enumerate(scores) if s>=thr}

agg = {b: {"topk":[], "thr":[]} for b in BASELINES}
per_task = []
t0 = time.time()
for idx, task in enumerate(tasks_raw):
    features = task["features"]
    rng = np.random.default_rng(SEED + idx)
    aug, target = sample(task, Q, rng)
    if aug is None: continue
    true_set = set(features)
    # add distractors (deterministic per task index)
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
    cols = [c for c in aug if c!=target]
    X = np.column_stack([aug[c] for c in cols]).astype(float)
    y = np.asarray(aug[target], dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)): continue
    k_true = len(features)
    task_rec = {"law_id": task["law_id"], "nvars": k_true, "n_cols": len(cols)}
    for name, fn in BASELINES.items():
        sc = fn(X, y)
        # operating point (a): oracle top-k
        sel_a = select_topk(sc, k_true)
        true_idx = {cols.index(f) for f in features if f in cols}
        tp_a = len(sel_a & true_idx); fp_a = len(sel_a - true_idx); fn_a = len(true_idx - sel_a)
        rec_a = tp_a/(tp_a+fn_a) if (tp_a+fn_a) else 0
        prec_a = tp_a/(tp_a+fp_a) if (tp_a+fp_a) else 0
        agg[name]["topk"].append((rec_a, prec_a))
        # operating point (b): generic threshold
        sel_b = select_threshold(sc)
        tp_b = len(sel_b & true_idx); fp_b = len(sel_b - true_idx); fn_b = len(true_idx - sel_b)
        rec_b = tp_b/(tp_b+fn_b) if (tp_b+fn_b) else 0
        prec_b = tp_b/(tp_b+fp_b) if (tp_b+fp_b) else 0
        agg[name]["thr"].append((rec_b, prec_b))
        task_rec[name] = {"topk_rec": rec_a, "topk_prec": prec_a,
                          "thr_rec": rec_b, "thr_prec": prec_b}
    per_task.append(task_rec)
    if (idx+1) % 20 == 0:
        print(f"  [{idx+1}/{len(tasks_raw)}] elapsed {time.time()-t0:.0f}s")

print(f"\nn_used = {len(per_task)} / {len(tasks_raw)}\n")
print(f"{'Baseline':18s} | {'oracle top-k':30s} | {'generic threshold':30s}")
print(f"{'':18s} | {'recall  prec   perfect%':30s} | {'recall  prec   perfect%':30s}")
print("-"*86)
summary = {}
for name in BASELINES:
    for mode in ["topk","thr"]:
        arr = np.array(agg[name][mode])
        recs = arr[:,0]; precs = arr[:,1]
        perfect = float(np.mean(recs >= 0.999))
        summary[f"{name}_{mode}"] = {"recall_mean": float(recs.mean()),
                                     "precision_mean": float(precs.mean()),
                                     "perfect_recall_frac": perfect,
                                     "n_tasks": int(len(recs))}
    tk = summary[f"{name}_topk"]; th = summary[f"{name}_thr"]
    print(f"{name:18s} | {tk['recall_mean']:.3f}  {tk['precision_mean']:.3f}  "
          f"{100*tk['perfect_recall_frac']:5.1f}%        | "
          f"{th['recall_mean']:.3f}  {th['precision_mean']:.3f}  "
          f"{100*th['perfect_recall_frac']:5.1f}%")

print("\nReference (DenoisedSR ensemble @0.10): recall 1.000, perfect 100%")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({"config":{"seed":SEED,"q":Q,"n_dist":N_DIST,"n_tasks":len(per_task)},
                                       "summary": summary, "per_task": per_task}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
