"""
Speed test: does VARIABLE filtering speed up PySR search?
A (all ~24 cols) vs B (selected ~4 cols), both all-ops, early-stop on exact.
Variable filtering shrinks the TERMINAL set -> exponential search-space reduction.
"""
import sys, json, re, warnings, os, time
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pysr, sympy as sp
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4; SEED=42; Q=int(os.environ.get("Q","100")); HELDOUT=300; N_DIST=20
TIMEOUT=int(os.environ.get("TIMEOUT","60")); NITERS=int(os.environ.get("NITERS","500"))
N_TEST=int(os.environ.get("N_TEST","15"))
ALL_UN=['sqrt','sin','cos','exp','log']  # moderate unary set for both

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
    if "=" not in formula: return None,None
    lhs,rhs=formula.split("=",1); target=lhs.strip()
    try:
        syms={f:sp.Symbol(f) for f in features}
        fn=sp.lambdify([syms[f] for f in features],sp.sympify(rhs.strip(),locals=syms),"numpy")
    except: return None,None
    cols={}
    for f in features:
        rg=ranges.get(f,[1.0,5.0]); lo,hi=float(rg[0]),float(rg[1])
        if lo>=hi: hi=lo+1.0
        cols[f]=rng.uniform(lo,hi,n)
    try: y=np.asarray(fn(*[cols[f] for f in features]),dtype=float)
    except: return None,None
    if not np.all(np.isfinite(y)): return None,None
    cols[target]=y; return cols,target

def add_noise(cols,target,n,seed):
    rng=np.random.default_rng(seed); out=dict(cols); q=len(cols[target])
    for j in range(n):
        lo,hi=float(rng.uniform(-5,0)),float(rng.uniform(0,5)); out[f"__d{j}"]=v2.resample(np.array([lo,hi]),q,rng)
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

def time_to_exact(Xtr,ytr,Xte,yte,seed):
    m=pysr.PySRRegressor(niterations=NITERS,timeout_in_seconds=TIMEOUT,
        binary_operators=['+','-','*','/','^'],unary_operators=ALL_UN,
        verbosity=0,random_state=seed,deterministic=True,parallelism='serial',
        early_stop_condition="f(loss, complexity) = loss < 1e-10")
    t0=time.time()
    try:
        m.fit(Xtr,ytr); el=time.time()-t0; yp=m.predict(Xte)
        ss=np.sum((yte-yp)**2); tot=np.sum((yte-yte.mean())**2); r2=1-ss/(tot+1e-12)
        return (r2>=0.9999), el
    except: return False, time.time()-t0

rows=json.loads(Path("data/results/numeric_equiv_rows.json").read_text())
solvable=[r["law_id"] for r in rows if r["r2"]>=0.9999][:N_TEST]
print(f"VARIABLE-filtering speed test: A (all cols) vs B (selected), {len(solvable)} formulas, budget={TIMEOUT}s\n")

all_tasks=[json.loads(l) for l in open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
all_tasks=[t for t in all_tasks if 2<=len(t["features"])<=5]
tlookup={t["law_id"]:(idx,t) for idx,t in enumerate(all_tasks)}

print(f"{'Formula':24s} {'A_cols':>6s} {'B_cols':>6s} {'A_exact':>7s} {'A_time':>7s} {'B_exact':>7s} {'B_time':>7s}")
print("-"*82)
a_times=[]; b_times=[]; a_solved=0; b_solved=0; both=[]
for lid in solvable:
    if lid not in tlookup: continue
    i,task=tlookup[lid]
    aug0,target=sample_feynman(task,Q,SEED*1000+i)
    te0,_=sample_feynman(task,HELDOUT,SEED*1000+i+500000)
    if aug0 is None: continue
    aug=add_noise(aug0,target,N_DIST,SEED*7+i); te=add_noise(te0,target,N_DIST,SEED*7+i+500000)
    full=[c for c in aug if c!=target]
    sel=select_vars(aug,target)
    XAtr=np.column_stack([aug[c] for c in full]); XAte=np.column_stack([te[c] for c in full])
    XBtr=np.column_stack([aug[c] for c in sel]);  XBte=np.column_stack([te[c] for c in sel])
    aE,at=time_to_exact(XAtr,aug[target],XAte,te[target],SEED+i)
    bE,bt=time_to_exact(XBtr,aug[target],XBte,te[target],SEED+i)
    if aE: a_solved+=1; a_times.append(at)
    if bE: b_solved+=1; b_times.append(bt)
    if aE and bE: both.append((at,bt))
    print(f"{lid[:24]:24s} {len(full):6d} {len(sel):6d} {'Y' if aE else '-':>7s} {at:6.1f}s {'Y' if bE else '-':>7s} {bt:6.1f}s")

print("-"*82)
print(f"\nA (all cols):       solved {a_solved}/{len(solvable)}, mean time {np.mean(a_times) if a_times else 0:.1f}s")
print(f"B (selected cols):  solved {b_solved}/{len(solvable)}, mean time {np.mean(b_times) if b_times else 0:.1f}s")
if both:
    at_arr=np.array([x[0] for x in both]); bt_arr=np.array([x[1] for x in both])
    speedups=at_arr/np.maximum(bt_arr,0.05)
    print(f"\nOn {len(both)} formulas BOTH solved:")
    print(f"  A mean time = {at_arr.mean():.1f}s   B mean time = {bt_arr.mean():.1f}s")
    print(f"  Mean speedup (A/B)   = {at_arr.mean()/bt_arr.mean():.2f}x")
    print(f"  Median per-formula   = {np.median(speedups):.2f}x")
    print(f"  Total A time={at_arr.sum():.0f}s  B time={bt_arr.sum():.0f}s")
import json as _j
_j.dump({"a_times":a_times,"b_times":b_times,"both":both,
         "a_solved":a_solved,"b_solved":b_solved},
        open("data/results/speed_vars_result.json","w"))
print("\nSaved -> data/results/speed_vars_result.json")
