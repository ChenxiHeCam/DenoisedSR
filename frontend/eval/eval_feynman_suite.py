"""
Eval GAT support predictor on AI-Feynman benchmark (119 formulas).
True external test: variable names (m_0, v, c, q, ...) differ from our
physics-graph training distribution. Samples observations from formula ranges.
"""
import sys, json, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import sympy as sp
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4; SEED=42; Q=100

class GATDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(FEAT,HID)
        self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*HEADS)
        self.gat2=GATConv(HID*HEADS,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm2=nn.LayerNorm(HID*HEADS)
        self.gat3=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1)
        self.norm3=nn.LayerNorm(HID)
        self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.15),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x)); h=F.gelu(self.norm1(self.gat1(h,ei))); h=F.gelu(self.norm2(self.gat2(h,ei))); h=self.norm3(self.gat3(h,ei)); return self.head(h).squeeze(-1)

COOC = jl("models/cooc_graph.joblib")
rf_clf = jl("models/support_predictor_v2_40k.joblib")['support_clf']
gat_ckpt = torch.load("models/gat_best.pt", map_location="cpu")
gat = GATDisc(); gat.load_state_dict(gat_ckpt["model"]); gat.eval()
print(f"GAT epoch {gat_ckpt['epoch']}")

# Load Feynman tasks
tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
print(f"Feynman tasks: {len(tasks_raw)}\n")

def sample_feynman(task, n, rng):
    """Sample observations from a Feynman formula given its ranges."""
    formula = task["formula"]          # e.g. "m = m_0/sqrt(1-v**2/c**2)"
    features = task["features"]
    ranges   = task.get("ranges", {})
    if "=" not in formula: return None, None
    lhs, rhs = formula.split("=", 1)
    target = lhs.strip()
    try:
        syms = {f: sp.Symbol(f) for f in features}
        expr = sp.sympify(rhs.strip(), locals=syms)
        fn   = sp.lambdify([syms[f] for f in features], expr, "numpy")
    except Exception:
        return None, None
    # sample each feature from its range
    cols = {}
    for f in features:
        rg = ranges.get(f, [1.0, 5.0])
        lo, hi = float(rg[0]), float(rg[1])
        if lo >= hi: hi = lo + 1.0
        cols[f] = rng.uniform(lo, hi, n)
    try:
        y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except Exception:
        return None, None
    if not np.all(np.isfinite(y)): return None, None
    cols[target] = y
    return cols, target

def get_scores(aug, target, cols):
    feats = np.array([v2.column_features(np.asarray(aug[c]), np.asarray(aug[target]), var_name=c)
                      for c in cols], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0., posinf=10., neginf=-10.)
    rf_s = positive_probability(rf_clf, feats)
    edges = set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges:
        for i in range(len(cols)):
            for j in range(i+1,len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst = zip(*edges)
    ei = torch.tensor([list(src),list(dst)],dtype=torch.long)
    ei,_ = add_self_loops(ei,num_nodes=len(cols))
    with torch.no_grad():
        gat_s = torch.sigmoid(gat(torch.tensor(feats),ei)).numpy()
    return rf_s, gat_s

def run_eval(name, selector, dist_lo=4, dist_hi=30):
    precs, recs, perfect, n_used = [], [], 0, 0
    for task in tasks_raw:
        features = task["features"]
        if len(features) < 2: continue
        rng = np.random.default_rng(SEED + tasks_raw.index(task))
        aug, target = sample_feynman(task, Q, rng)
        if aug is None: continue
        true_vars = set(features)  # all features are true vars
        n_dist = int(rng.integers(dist_lo, dist_hi + 1))
        q = len(aug[target])
        for i in range(n_dist):
            lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
            aug[f"__d{i}"] = v2.resample(np.array([lo,hi]), q, rng)
        cols = [c for c in aug if c != target]
        try:
            rf_s, gat_s = get_scores(aug, target, cols)
        except Exception:
            continue
        sel = selector(cols, rf_s, gat_s)
        sel = {c for c in sel if not c.startswith("__d")}
        if not sel: sel = {cols[int(np.argmax(gat_s))]}
        tp=len(sel&true_vars); fp=len(sel-true_vars); fn=len(true_vars-sel)
        if tp+fp>0: precs.append(tp/(tp+fp))
        if tp+fn>0: recs.append(tp/(tp+fn))
        if tp+fn>0 and tp/(tp+fn)>=0.999: perfect+=1
        n_used += 1
    n = len(precs)
    f1 = 2*np.mean(precs)*np.mean(recs)/(np.mean(precs)+np.mean(recs)+1e-9)
    print(f"  {name:35s}  prec={np.mean(precs):.3f}  rec={np.mean(recs):.3f}  "
          f"F1={f1:.3f}  perfect={perfect}/{n} ({100*perfect/n:.0f}%)")

strategies = [
    ("GAT only (t=0.08)",        lambda c,r,g: {x for x,s in zip(c,g) if s>=0.08}),
    ("RF only (t=0.10)",         lambda c,r,g: {x for x,s in zip(c,r) if s>=0.10}),
    ("Combined rf+gat >= 0.10",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs+gs>=0.10}),
    ("Union RF>=0.10 | GAT>=0.08", lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs>=0.10 or gs>=0.08}),
]

for lo, hi in [(1,2), (10,30), (4,30)]:
    print(f"\n=== AI-Feynman (119 formulas) — noise {lo}-{hi} distractors ===\n")
    for name, sel in strategies:
        run_eval(f"{name}  [{lo}-{hi}]", sel, dist_lo=lo, dist_hi=hi)
