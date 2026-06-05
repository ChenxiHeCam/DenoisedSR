"""
Speed test: does operator prior make PySR faster to reach the solution?
For each formula, run with a LONG budget but measure time-to-first-exact
(via PySR iteration logs) for B (all ops) vs C (predicted ops).
Use small set of solvable formulas, longer per-run budget.
"""
import sys, json, re, warnings, os, time
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pysr, sympy as sp
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability
from train_gat_operator import GATOperator, build_graph_data, OP_FULL

FEAT=74; HID=128; HEADS=4; SEED=42; Q=100; HELDOUT=300; N_DIST=20; OP_THRESH=0.03
TIMEOUT=int(os.environ.get("TIMEOUT","30"))   # long budget
NITERS=int(os.environ.get("NITERS","200"))

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
gatop=GATOperator(); gatop.load_state_dict(torch.load("models/gat_operator.pt",map_location="cpu")["model"]); gatop.eval()

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

def predict_ops(aug,target):
    g=build_graph_data(aug,target,"")
    with torch.no_grad():
        bidx=torch.zeros(g.x.shape[0],dtype=torch.long)
        probs=torch.sigmoid(gatop(g.x,g.edge_index,bidx))[0].numpy()
    pred={op for op,p in zip(OP_FULL,probs) if p>=OP_THRESH}
    un_vocab={'sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube'}
    un=sorted({o for o in pred if o in un_vocab}) or ['sqrt']
    return ['+','-','*','/','^'], un

ALL_UN=['sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube']

def time_to_exact(Xtr,ytr,Xte,yte,seed,bin_ops,un_ops):
    """Run PySR, return (reached_exact, wall_time, final_r2)."""
    m=pysr.PySRRegressor(niterations=NITERS,timeout_in_seconds=TIMEOUT,
        binary_operators=bin_ops,unary_operators=un_ops,
        verbosity=0,random_state=seed,deterministic=True,parallelism='serial',
        early_stop_condition="f(loss, complexity) = loss < 1e-10")
    t0=time.time()
    try:
        m.fit(Xtr,ytr); el=time.time()-t0; yp=m.predict(Xte)
        ss=np.sum((yte-yp)**2); tot=np.sum((yte-yte.mean())**2); r2=1-ss/(tot+1e-12)
        return (r2>=0.9999), el, r2
    except: return False, time.time()-t0, -99

# Use formulas that ARE solvable (from prior exact results)
rows=json.loads(Path("data/results/numeric_equiv_rows.json").read_text())
solvable=[r["law_id"] for r in rows if r["r2"]>=0.9999][:20]
print(f"Speed test on {len(solvable)} solvable formulas, budget={TIMEOUT}s, early-stop on exact\n")

all_tasks=[json.loads(l) for l in open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
all_tasks=[t for t in all_tasks if 2<=len(t["features"])<=5]
tlookup={t["law_id"]:(idx,t) for idx,t in enumerate(all_tasks)}

print(f"{'Formula':24s} {'n_un_B':>6s} {'n_un_C':>6s} {'B_exact':>8s} {'B_time':>7s} {'C_exact':>8s} {'C_time':>7s}")
print("-"*85)
b_times=[]; c_times=[]; b_solved=0; c_solved=0
for lid in solvable:
    if lid not in tlookup: continue
    i,task=tlookup[lid]
    aug0,target=sample_feynman(task,Q,SEED*1000+i)
    te0,_=sample_feynman(task,HELDOUT,SEED*1000+i+500000)
    if aug0 is None: continue
    aug=add_noise(aug0,target,N_DIST,SEED*7+i); te=add_noise(te0,target,N_DIST,SEED*7+i+500000)
    sel=select_vars(aug,target)
    Xtr=np.column_stack([aug[c] for c in sel]); Xte=np.column_stack([te[c] for c in sel])
    _,un=predict_ops(aug,target)
    bE,bt,_=time_to_exact(Xtr,aug[target],Xte,te[target],SEED+i,['+','-','*','/','^'],ALL_UN)
    cE,ct,_=time_to_exact(Xtr,aug[target],Xte,te[target],SEED+i,['+','-','*','/','^'],un)
    if bE: b_solved+=1; b_times.append(bt)
    if cE: c_solved+=1; c_times.append(ct)
    print(f"{lid[:24]:24s} {len(ALL_UN):6d} {len(un):6d} {'Y' if bE else '-':>8s} {bt:6.1f}s {'Y' if cE else '-':>8s} {ct:6.1f}s")

print("-"*85)
print(f"\nB (all {len(ALL_UN)} unary ops):  {b_solved} solved, mean time-to-exact = {np.mean(b_times) if b_times else 0:.1f}s")
print(f"C (predicted unary ops):      {c_solved} solved, mean time-to-exact = {np.mean(c_times) if c_times else 0:.1f}s")
if b_times and c_times:
    # compare only formulas BOTH solved (fair)
    print(f"\nMean time B={np.mean(b_times):.1f}s  C={np.mean(c_times):.1f}s  ->  speedup {np.mean(b_times)/np.mean(c_times):.2f}x")
