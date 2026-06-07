"""
Re-evaluate AI-Feynman with NUMERIC equivalence (not sympy).
For each formula, run PySR (Ours-vars condition), then judge exact by:
  - numeric: PySR(x) vs truth(x) on fresh random points, allowing constant
             ratio or constant offset (catches sympy false-negatives)
Outputs a table for human inspection of borderline cases.
"""
import sys, json, re, warnings, os
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

FEAT=74; HID=128; HEADS=4; SEED=42; Q=100; HELDOUT=300; N_DIST=20; TIMEOUT=10
MAX_VARS=int(os.environ.get("MAX_VARS","5"))

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

def numeric_equiv(yp, yt):
    """Are PySR predictions equivalent to truth, allowing constant ratio/offset?"""
    m = np.isfinite(yp) & np.isfinite(yt)
    if m.sum() < 10: return False, "too few valid"
    yp, yt = yp[m], yt[m]
    # exact
    rel = np.abs(yp-yt)/(np.abs(yt)+1e-9)
    if np.median(rel) < 1e-4: return True, "exact"
    # constant ratio
    ratio = yp/(yt+1e-12)
    if np.std(ratio)/(np.abs(np.mean(ratio))+1e-9) < 1e-3: return True, f"ratio={np.mean(ratio):.4g}"
    # constant offset
    off = yp-yt
    if np.std(off)/(np.abs(np.mean(yt))+1e-9) < 1e-3: return True, f"offset={np.mean(off):.4g}"
    return False, "not equiv"

tasks=[json.loads(l) for l in open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks=[t for t in tasks if 2<=len(t["features"])<=MAX_VARS]
print(f"Numeric-equivalence eval on {len(tasks)} Feynman formulas\n")

sympy_exact=0; numeric_exact=0; rescued=[]
rows=[]
for i,task in enumerate(tasks):
    aug0,target=sample_feynman(task,Q,SEED*1000+i)
    te0,_=sample_feynman(task,HELDOUT,SEED*1000+i+500000)
    if aug0 is None: continue
    aug=add_noise(aug0,target,N_DIST,SEED*7+i); te=add_noise(te0,target,N_DIST,SEED*7+i+500000)
    sel=select_vars(aug,target)
    Xtr=np.column_stack([aug[c] for c in sel]); Xte=np.column_stack([te[c] for c in sel])
    m=pysr.PySRRegressor(niterations=40,timeout_in_seconds=TIMEOUT,
        binary_operators=['+','-','*','/','^'],unary_operators=['sqrt','sin','cos','exp','log'],
        verbosity=0,random_state=SEED+i,deterministic=True,parallelism='serial')
    try:
        m.fit(Xtr,aug[target]); yp=m.predict(Xte); expr=str(m.sympy())
    except: continue
    yt=te[target]
    ss=np.sum((yt-yp)**2); tot=np.sum((yt-yt.mean())**2); r2=1-ss/(tot+1e-12)
    is_r2_exact = r2>=0.9999
    is_num, reason = numeric_equiv(yp, yt)
    if is_r2_exact: sympy_exact+=1   # proxy
    if is_num: numeric_exact+=1
    # rescued = numeric says equiv but R2 missed it
    if is_num and not is_r2_exact:
        rescued.append((task["law_id"], r2, reason, expr, task["formula"]))
    rows.append({"law_id":task["law_id"],"r2":r2,"numeric":is_num,"reason":reason,
                 "sel":sel,"expr":expr,"truth":task["formula"]})

print(f"=== EXACT RECOVERY: R2>=0.9999 vs NUMERIC equivalence ===")
print(f"  R2-based:        {sympy_exact}/{len(rows)} ({100*sympy_exact/len(rows):.0f}%)")
print(f"  Numeric-equiv:   {numeric_exact}/{len(rows)} ({100*numeric_exact/len(rows):.0f}%)")
print(f"  RESCUED (numeric equiv but R2 missed): {len(rescued)}")
print()
print("="*100)
print("RESCUED CASES — numeric says EQUIVALENT but R2<0.9999 (sympy/R2 false-negatives):")
print("="*100)
for lid,r2,reason,expr,truth in rescued:
    print(f"\n  {lid}  (R2={r2:.4f}, {reason})")
    print(f"    TRUTH: {truth}")
    print(f"    FOUND: {expr}")

Path("data/results/numeric_equiv_rows.json").write_text(json.dumps(rows,indent=2))
print(f"\nSaved -> data/results/numeric_equiv_rows.json")
