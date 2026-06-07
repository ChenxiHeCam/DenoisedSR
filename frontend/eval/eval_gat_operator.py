"""Eval GAT operator predictor on Feynman — does it generalize?"""
import sys, json, re, warnings, os
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None

import numpy as np, torch
from pathlib import Path
from joblib import load as jl
import sympy as sp
import train_support_predictor_v2 as v2
from train_gat_operator import (GATOperator, build_graph_data, infer_ops_full,
                                add_noise, OP_FULL, behav_node_feat, COOC)

SEED=42; Q=100; N_DIST=15
OP_THRESH = float(os.environ.get("OP_THRESH","0.30"))
SAFE = {'+','-','*','/'}

ckpt = torch.load("models/gat_operator.pt", map_location="cpu")
model = GATOperator(); model.load_state_dict(ckpt["model"]); model.eval()
print(f"GAT operator epoch {ckpt['epoch']}, threshold={OP_THRESH}\n")

tasks = [json.loads(l) for l in
         open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks = [t for t in tasks if 2 <= len(t["features"]) <= 5]

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

precs, recs, kept = [], [], []
miss=[]
for i,task in enumerate(tasks):
    aug,target,rhs=sample_feynman(task,Q,SEED*1000+i)
    if aug is None: continue
    aug=add_noise(aug,target,N_DIST,np.random.default_rng(SEED*7+i))
    try:
        g=build_graph_data(aug,target,rhs)
    except: continue
    with torch.no_grad():
        batch_idx=torch.zeros(g.x.shape[0],dtype=torch.long)
        logits=model(g.x,g.edge_index,batch_idx)
        probs=torch.sigmoid(logits)[0].numpy()
    pred={op for op,p in zip(OP_FULL,probs) if p>=OP_THRESH} | SAFE
    true_ops=infer_ops_full(rhs) | SAFE
    et=true_ops-SAFE; ep=pred-SAFE
    if et:
        tp=len(ep&et); fn=len(et-ep)
        recs.append(tp/(tp+fn))
        if fn>0: miss.append((task["law_id"],sorted(et-ep),task["formula"][:48]))
    if ep: precs.append(len(ep&et)/len(ep))
    kept.append(len(pred))

print("=== GAT Operator prediction on Feynman ===")
print(f"  Recall:    {np.mean(recs):.3f}")
print(f"  Precision: {np.mean(precs):.3f}")
print(f"  Mean ops kept: {np.mean(kept):.1f}/{len(OP_FULL)} ({100*(1-np.mean(kept)/len(OP_FULL)):.0f}% reduction)")
print(f"\n[RF behavioral baseline: recall=0.705 precision=0.423]")
if miss:
    print(f"\n  Missed ({len(miss)} formulas):")
    for lid,m,f in miss[:12]:
        print(f"    {lid:22s} missed {m}  |  {f}")
