"""Operator-prior sweep: threshold sensitivity + named/anonymized comparison.

Operator filter is a separate model (GAT-operator) — predicts which of 17
operators are in the formula. We have not previously characterized it as
carefully as the variable predictor.
"""
import sys, json, os, warnings
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train"))
import numpy as np, sympy as sp, torch
from joblib import load as jl
from pathlib import Path
import train_support_predictor_v2 as v2
from train_gat_operator import GATOperator, build_graph_data, infer_ops_full, add_noise, OP_FULL

SEED=42; Q=100
SAFE = {'+','-','*','/'}
TAUS = [0.03, 0.10, 0.20, 0.30, 0.50, 0.70]

ckpt = torch.load("models/gat_operator.pt", map_location="cpu")
model = GATOperator(); model.load_state_dict(ckpt["model"]); model.eval()

tasks = [json.loads(l) for l in
         open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks = [t for t in tasks if 2 <= len(t["features"]) <= 5]
print(f"Feynman tasks: {len(tasks)} (2-5 vars subset)\n")

def sample_feynman(task, n, seed):
    rng=np.random.default_rng(seed)
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
    return cols,target,rhs

def anonymize(aug, target, features, rhs):
    name_map = {f: f"x{i}" for i, f in enumerate(features)}
    new_aug = {name_map.get(k, k): v for k, v in aug.items()}
    # rewrite rhs: replace each physical name with x_i
    new_rhs = rhs
    # sort by length desc to avoid sub-string clashes
    for orig, anon in sorted(name_map.items(), key=lambda x: -len(x[0])):
        new_rhs = new_rhs.replace(orig, anon)
    return new_aug, target, list(name_map.values()), new_rhs

def predict_probs(aug, target, rhs):
    try:
        g = build_graph_data(aug, target, rhs)
    except: return None
    with torch.no_grad():
        bidx = torch.zeros(g.x.shape[0], dtype=torch.long)
        logits = model(g.x, g.edge_index, bidx)
        return torch.sigmoid(logits)[0].numpy()

# Sweep tau under both name regimes
print(f"{'tau':>5} | {'NAMED rec/prec/filt%':^28s} | {'ANON  rec/prec/filt%':^28s} | named-anon delta")
print("-"*100)
results = {}
for tau in TAUS:
    nm_r, nm_p, nm_k = [], [], []
    an_r, an_p, an_k = [], [], []
    for i,task in enumerate(tasks):
        aug, target, rhs = sample_feynman(task, Q, SEED*1000+i)
        if aug is None: continue
        aug = add_noise(aug, target, 15, np.random.default_rng(SEED*7+i))
        # NAMED
        probs = predict_probs(aug, target, rhs)
        if probs is None: continue
        true_ops = infer_ops_full(rhs) | SAFE
        pred = {op for op,p in zip(OP_FULL, probs) if p>=tau} | SAFE
        et = true_ops - SAFE; ep = pred - SAFE
        if et:
            tp = len(ep & et); fn = len(et - ep)
            nm_r.append(tp/(tp+fn))
            if ep: nm_p.append(len(ep & et)/len(ep))
        nm_k.append(len(pred))
        # ANONYMIZED — same data, renamed
        aug_a, t_a, feats_a, rhs_a = anonymize(aug, target, task["features"], rhs)
        probs_a = predict_probs(aug_a, t_a, rhs_a)
        if probs_a is None: continue
        true_ops_a = infer_ops_full(rhs_a) | SAFE
        pred_a = {op for op,p in zip(OP_FULL, probs_a) if p>=tau} | SAFE
        et_a = true_ops_a - SAFE; ep_a = pred_a - SAFE
        if et_a:
            tp_a = len(ep_a & et_a); fn_a = len(et_a - ep_a)
            an_r.append(tp_a/(tp_a+fn_a))
            if ep_a: an_p.append(len(ep_a & et_a)/len(ep_a))
        an_k.append(len(pred_a))
    nr,np_,nk = float(np.mean(nm_r)), float(np.mean(nm_p)), float(np.mean(nm_k))
    ar,ap,ak = float(np.mean(an_r)), float(np.mean(an_p)), float(np.mean(an_k))
    # filter rate over 13 non-safe ops in OP_FULL = 17, minus 4 safe = 13
    nm_filt = 100*(1 - (nk-4)/13)
    an_filt = 100*(1 - (ak-4)/13)
    print(f"{tau:>5.2f} | {nr:.3f}/{np_:.3f}/{nm_filt:>5.1f}%  ({nk:.1f}/17 kept) | "
          f"{ar:.3f}/{ap:.3f}/{an_filt:>5.1f}%  ({ak:.1f}/17 kept) | "
          f"rec {nr-ar:+.3f}  filt {nm_filt-an_filt:+5.2f} pp")
    results[f"tau_{tau}"] = {"tau":tau,
        "named":{"recall":nr,"precision":np_,"kept":nk,"filter":nm_filt/100},
        "anon": {"recall":ar,"precision":ap,"kept":ak,"filter":an_filt/100}}

Path("data/results/operator_sweep.json").write_text(json.dumps(results, indent=2))
print("\nSaved -> data/results/operator_sweep.json")
