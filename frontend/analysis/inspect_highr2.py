"""Find high-R2-but-not-exact cases, re-run PySR, show spurious formula vs truth."""
import sys, json, re, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pysr, sympy as sp
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4; SEED=42; Q=100; HELDOUT=200; N_DIST=20; TIMEOUT=10; OP_THRESH=0.03

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
gat = GATDisc(); gat.load_state_dict(torch.load("models/gat_best.pt",map_location="cpu")["model"]); gat.eval()

def sample_feynman(task, n, seed):
    rng=np.random.default_rng(seed)
    formula=task["formula"]; features=task["features"]; ranges=task.get("ranges",{})
    if "=" not in formula: return None,None
    lhs,rhs=formula.split("=",1); target=lhs.strip()
    try:
        syms={f:sp.Symbol(f) for f in features}
        fn=sp.lambdify([syms[f] for f in features], sp.sympify(rhs.strip(),locals=syms),"numpy")
    except: return None,None
    cols={}
    for f in features:
        rg=ranges.get(f,[1.0,5.0]); lo,hi=float(rg[0]),float(rg[1])
        if lo>=hi: hi=lo+1.0
        cols[f]=rng.uniform(lo,hi,n)
    try: y=np.asarray(fn(*[cols[f] for f in features]),dtype=float)
    except: return None,None
    if not np.all(np.isfinite(y)): return None,None
    cols[target]=y
    return cols,target

def add_noise(cols,target,n,seed):
    rng=np.random.default_rng(seed); out=dict(cols); q=len(cols[target])
    for j in range(n):
        lo,hi=float(rng.uniform(-5,0)),float(rng.uniform(0,5))
        out[f"__d{j}"]=v2.resample(np.array([lo,hi]),q,rng)
    return out

def select_vars(aug,target):
    cols=[c for c in aug if c!=target]
    feats=np.array([v2.column_features(np.asarray(aug[c]),np.asarray(aug[target]),var_name=c) for c in cols],dtype=np.float32)
    feats=np.nan_to_num(feats,nan=0.,posinf=10.,neginf=-10.)
    rf_s=positive_probability(rf_clf,feats)
    edges=set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges:
        for i in range(len(cols)):
            for j in range(i+1,len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst=zip(*edges); ei=torch.tensor([list(src),list(dst)],dtype=torch.long); ei,_=add_self_loops(ei,num_nodes=len(cols))
    with torch.no_grad(): gat_s=torch.sigmoid(gat(torch.tensor(feats),ei)).numpy()
    sel=[c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs>=0.10 and not c.startswith("__d")]
    return sel if sel else [cols[int(np.argmax(rf_s))]]

saved = json.loads(Path("data/results/pysr_frontend_3way.json").read_text())
cands=[(r["law_id"],r["var_r2"]) for r in saved if 0.90<=r["var_r2"]<0.9999]
print("High-R2 non-exact cases:", [f'{l}={v:.3f}' for l,v in cands], "\n")

all_tasks=[json.loads(l) for l in open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
all_tasks=[t for t in all_tasks if 2<=len(t["features"])<=5]
tlookup={t["law_id"]:(idx,t) for idx,t in enumerate(all_tasks)}

for lid,_ in cands[:6]:
    if lid not in tlookup: continue
    i,task=tlookup[lid]
    aug0,target=sample_feynman(task,Q,SEED*1000+i)
    te0,_=sample_feynman(task,HELDOUT,SEED*1000+i+500000)
    if aug0 is None: continue
    aug=add_noise(aug0,target,N_DIST,SEED*7+i); te=add_noise(te0,target,N_DIST,SEED*7+i+500000)
    sel=select_vars(aug,target)
    Xtr=np.column_stack([aug[c] for c in sel]); Xte=np.column_stack([te[c] for c in sel])
    m=pysr.PySRRegressor(niterations=40,timeout_in_seconds=TIMEOUT,
        binary_operators=['+','-','*','/','^'],unary_operators=['sqrt','sin','cos','exp','log'],
        verbosity=0,random_state=SEED+i,deterministic=True,parallelism='serial')
    m.fit(Xtr,aug[target]); yp=m.predict(Xte)
    ss=np.sum((te[target]-yp)**2); tot=np.sum((te[target]-te[target].mean())**2); r2=1-ss/(tot+1e-12)
    print(f"=== {lid} ===")
    print(f"  TRUE:       {target} = {task['formula'].split('=',1)[1].strip()}")
    print(f"  vars(x0..): {sel}")
    print(f"  PySR FOUND: {m.sympy()}")
    print(f"  R2={r2:.5f}\n")
