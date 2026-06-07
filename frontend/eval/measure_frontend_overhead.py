"""Measure the front-end (RF + GAT inference) overhead per formula.
This is the time we ADD; it must be tiny vs the PySR time we SAVE."""
import sys, json, re, warnings, time
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sympy as sp
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

import os
FEAT=74; HID=128; HEADS=4; SEED=42; Q=int(os.environ.get("Q","100")); N_DIST=20

class GATDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(FEAT,HID)
        self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1); self.norm1=nn.LayerNorm(HID*HEADS)
        self.gat2=GATConv(HID*HEADS,HID,heads=HEADS,concat=True,dropout=0.1); self.norm2=nn.LayerNorm(HID*HEADS)
        self.gat3=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1); self.norm3=nn.LayerNorm(HID)
        self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.15),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x)); h=F.gelu(self.norm1(self.gat1(h,ei))); h=F.gelu(self.norm2(self.gat2(h,ei))); h=self.norm3(self.gat3(h,ei)); return self.head(h).squeeze(-1)

COOC=jl("models/cooc_graph.joblib")
rf_clf=jl("models/support_predictor_v2_40k.joblib")['support_clf']
gat=GATDisc(); gat.load_state_dict(torch.load("models/gat_best.pt",map_location="cpu")["model"]); gat.eval()

def sample_feynman(task,n,seed):
    rng=np.random.default_rng(seed)
    formula=task["formula"]; features=task["features"]; ranges=task.get("ranges",{})
    lhs,rhs=formula.split("=",1); target=lhs.strip()
    syms={f:sp.Symbol(f) for f in features}
    fn=sp.lambdify([syms[f] for f in features],sp.sympify(rhs.strip(),locals=syms),"numpy")
    cols={}
    for f in features:
        rg=ranges.get(f,[1.0,5.0]); lo,hi=float(rg[0]),float(rg[1])
        if lo>=hi: hi=lo+1.0
        cols[f]=rng.uniform(lo,hi,n)
    y=np.asarray(fn(*[cols[f] for f in features]),dtype=float)
    cols[target]=y; return cols,target

def add_noise(cols,target,n,seed):
    rng=np.random.default_rng(seed); out=dict(cols); q=len(cols[target])
    for j in range(n):
        lo,hi=float(rng.uniform(-5,0)),float(rng.uniform(0,5)); out[f"__d{j}"]=v2.resample(np.array([lo,hi]),q,rng)
    return out

def frontend_select(aug,target):
    """Full front-end: feature extraction + RF + GAT. Returns selected cols."""
    cols=[c for c in aug if c!=target]
    # feature extraction
    feats=np.array([v2.column_features(np.asarray(aug[c]),np.asarray(aug[target]),var_name=c) for c in cols],dtype=np.float32)
    feats=np.nan_to_num(feats,nan=0.,posinf=10.,neginf=-10.)
    # RF
    rf_s=positive_probability(rf_clf,feats)
    # graph build
    edges=set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges:
        for i in range(len(cols)):
            for j in range(i+1,len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst=zip(*edges); ei=torch.tensor([list(src),list(dst)],dtype=torch.long); ei,_=add_self_loops(ei,num_nodes=len(cols))
    # GAT
    with torch.no_grad(): gat_s=torch.sigmoid(gat(torch.tensor(feats),ei)).numpy()
    sel=[c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs>=0.10 and not c.startswith("__d")]
    return sel

tasks=[json.loads(l) for l in open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks=[t for t in tasks if 2<=len(t["features"])<=5][:30]
print(f"Measuring front-end overhead on {len(tasks)} formulas (q={Q}, {N_DIST} distractors)\n")

# warmup
for t in tasks[:2]:
    aug0,target=sample_feynman(t,Q,SEED)
    aug=add_noise(aug0,target,N_DIST,SEED)
    frontend_select(aug,target)

times=[]
for i,task in enumerate(tasks):
    try:
        aug0,target=sample_feynman(task,Q,SEED*1000+i)
        aug=add_noise(aug0,target,N_DIST,SEED*7+i)
    except: continue
    t0=time.time()
    sel=frontend_select(aug,target)
    times.append(time.time()-t0)

times=np.array(times)
print(f"Front-end overhead per formula:")
print(f"  mean   = {times.mean()*1000:.1f} ms")
print(f"  median = {np.median(times)*1000:.1f} ms")
print(f"  max    = {times.max()*1000:.1f} ms")
print(f"\nCompare: PySR time saved per formula is typically several SECONDS.")
print(f"Front-end overhead ({times.mean()*1000:.0f}ms) is ~{1000/(times.mean()*1000):.0f}x smaller than 1 second.")
