"""
Evaluate the full operator predictor on Feynman formulas.
Measure per-operator recall/precision and how much it shrinks the op search space.
"""
import sys, json, re, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np
from pathlib import Path
from joblib import load as jl
import sympy as sp
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability
from train_operator_predictor import OP_FULL, infer_ops_full, add_noise, behavioral_features

SEED=42; Q=100; N_DIST=15
OP_THRESH = float(os.environ.get("OP_THRESH","0.15"))

ckpt = jl("models/operator_predictor_full.joblib")
op_clf = ckpt['op_clf']

tasks = [json.loads(l) for l in
         open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks = [t for t in tasks if 2 <= len(t["features"]) <= 5]
print(f"Operator predictor eval: {len(tasks)} Feynman formulas, threshold={OP_THRESH}\n")

def sample_feynman(task, n, seed):
    rng = np.random.default_rng(seed)
    formula=task["formula"]; features=task["features"]; ranges=task.get("ranges",{})
    if "=" not in formula: return None,None,None
    lhs,rhs=formula.split("=",1); target=lhs.strip()
    try:
        syms={f:sp.Symbol(f) for f in features}
        fn=sp.lambdify([syms[f] for f in features], sp.sympify(rhs.strip(),locals=syms),"numpy")
    except: return None,None,None
    cols={}
    for f in features:
        rg=ranges.get(f,[1.0,5.0]); lo,hi=float(rg[0]),float(rg[1])
        if lo>=hi: hi=lo+1.0
        cols[f]=rng.uniform(lo,hi,n)
    try: y=np.asarray(fn(*[cols[f] for f in features]),dtype=float)
    except: return None,None,None
    if not np.all(np.isfinite(y)): return None,None,None
    cols[target]=y
    return cols, target, rhs

# Always-keep safe binary ops (never penalize for these)
SAFE = {'+','-','*','/'}

precs, recs, kept_sizes = [], [], []
miss_examples = []
for i, task in enumerate(tasks):
    aug, target, rhs = sample_feynman(task, Q, SEED*1000+i)
    if aug is None: continue
    aug = add_noise(aug, target, N_DIST, np.random.default_rng(SEED*7+i))
    cols = [c for c in aug if c != target]

    feat = np.array([behavioral_features(aug, target, cols)], dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0., posinf=100., neginf=-100.)
    probs = positive_probability(op_clf, feat)[0]
    pred = {op for op,p in zip(OP_FULL, probs) if p >= OP_THRESH}
    pred |= SAFE  # always include safe binary

    true_ops = infer_ops_full(rhs) | SAFE  # ground truth ops (+safe)
    # only evaluate on non-safe ops (the interesting ones)
    eval_true = true_ops - SAFE
    eval_pred = pred - SAFE

    if eval_true:
        tp = len(eval_pred & eval_true)
        fn = len(eval_true - eval_pred)
        rec = tp / (tp + fn)
        recs.append(rec)
        if fn > 0:
            miss_examples.append((task["law_id"], sorted(eval_true-eval_pred), task["formula"][:50]))
    if eval_pred:
        precs.append(len(eval_pred & eval_true) / len(eval_pred))
    kept_sizes.append(len(pred))

print(f"=== Operator prediction quality (non-trivial ops only) ===")
print(f"  Recall (don't miss needed ops):  {np.mean(recs):.3f}")
print(f"  Precision (don't add junk ops):  {np.mean(precs):.3f}")
print(f"  Mean ops kept: {np.mean(kept_sizes):.1f} / {len(OP_FULL)} "
      f"({100*(1-np.mean(kept_sizes)/len(OP_FULL)):.0f}% reduction)")
print(f"\n  Full library: {len(OP_FULL)} ops -> predicted: {np.mean(kept_sizes):.1f} ops")

if miss_examples:
    print(f"\n  MISSED operators ({len(miss_examples)} formulas):")
    for lid, missed, f in miss_examples[:15]:
        print(f"    {lid:22s} missed {missed}  |  {f}")
