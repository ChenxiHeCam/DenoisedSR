"""
KEY EXPERIMENT: PySR with vs without our neural front-end.

Front-end provides TWO priors:
  1. Variable support  — which columns are real variables (GAT+RF)
  2. Operator support  — which operators to search (RF op-classifier)

Conditions:
  A) Full PySR              — all columns, all operators
  B) Ours-vars + PySR       — selected columns, all operators
  C) Ours-vars+ops + PySR   — selected columns, restricted operators

Fixes vs prior version:
  - Decoupled sampling: true-variable values use a FIXED seed independent of N_DIST
  - Deterministic PySR (deterministic=True, parallelism='serial') for reproducibility
"""
import sys, json, re, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import sympy as sp, pysr
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability, OP_ORDER

FEAT=74; HID=128; HEADS=4; SEED=42
Q=int(os.environ.get("Q","100")); HELDOUT=200; N_DIST=int(os.environ.get("N_DIST","20"))
TIMEOUT=int(os.environ.get("TIMEOUT","5")); NITERS=int(os.environ.get("NITERS","40"))
N_TASKS=int(os.environ.get("N_TASKS","999"))
MAX_VARS=int(os.environ.get("MAX_VARS","9"))
# Large operator library — like a realistic open SR setting.
# Bigger search space => full PySR struggles more => front-end value shows.
ALL_BIN  = ['+','-','*','/','^']
ALL_UNARY= ['sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube']

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
rf_ckpt = jl("models/support_predictor_v2_40k.joblib")
rf_clf  = rf_ckpt['support_clf']
gat_ckpt = torch.load("models/gat_best.pt", map_location="cpu")
gat = GATDisc(); gat.load_state_dict(gat_ckpt["model"]); gat.eval()

# GAT operator predictor
from train_gat_operator import GATOperator, build_graph_data, OP_FULL
gatop_ckpt = torch.load("models/gat_operator.pt", map_location="cpu")
gatop = GATOperator(); gatop.load_state_dict(gatop_ckpt["model"]); gatop.eval()
OP_THRESH = float(os.environ.get("OP_THRESH","0.03"))

tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks_raw = [t for t in tasks_raw if 2 <= len(t["features"]) <= MAX_VARS][:N_TASKS]
print(f"PySR front-end experiment: {len(tasks_raw)} Feynman formulas")
print(f"q={Q}, {N_DIST} distractors, PySR timeout={TIMEOUT}s, deterministic\n")

def sample_feynman(task, n, seed):
    """Sample with a FIXED seed independent of distractor count."""
    rng = np.random.default_rng(seed)
    formula = task["formula"]; features = task["features"]; ranges = task.get("ranges", {})
    if "=" not in formula: return None, None
    lhs, rhs = formula.split("=", 1); target = lhs.strip()
    try:
        syms = {f: sp.Symbol(f) for f in features}
        fn = sp.lambdify([syms[f] for f in features], sp.sympify(rhs.strip(), locals=syms), "numpy")
    except: return None, None
    cols = {}
    for f in features:
        rg = ranges.get(f, [1.0, 5.0]); lo, hi = float(rg[0]), float(rg[1])
        if lo >= hi: hi = lo + 1.0
        cols[f] = rng.uniform(lo, hi, n)
    try:
        y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except: return None, None
    if not np.all(np.isfinite(y)): return None, None
    cols[target] = y
    return cols, target

def add_noise(cols, target, n, seed):
    """Add n noise columns with a separate seed (doesn't perturb true vars)."""
    rng = np.random.default_rng(seed)
    out = dict(cols); q = len(cols[target])
    for j in range(n):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        out[f"__d{j}"] = v2.resample(np.array([lo,hi]), q, rng)
    return out

def predict_support_and_ops(aug, target):
    cols = [c for c in aug if c != target]
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
    sel = [c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs>=0.10 and not c.startswith("__d")]
    if not sel: sel = [cols[int(np.argmax(rf_s))]]

    # Operator prediction — GAT operator model
    g = build_graph_data(aug, target, "")   # truth empty; we only need x/edge for inference
    with torch.no_grad():
        bidx = torch.zeros(g.x.shape[0], dtype=torch.long)
        op_probs = torch.sigmoid(gatop(g.x, g.edge_index, bidx))[0].numpy()
    pred_ops = {op for op, p in zip(OP_FULL, op_probs) if p >= OP_THRESH}
    pred_ops.update({'+','-','*','/'})  # always keep all safe binary ops
    return sel, pred_ops

def r2(yt, yp):
    ss=np.sum((yt-yp)**2); tot=np.sum((yt-np.mean(yt))**2); return float(1-ss/(tot+1e-12))

def run_pysr(X_tr, y_tr, X_te, y_te, seed, bin_ops=None, un_ops=None):
    """Run PySR; report R2, time, and how many iterations to reach exact on TRAIN."""
    bin_ops = bin_ops or ALL_BIN
    un_ops  = un_ops  if un_ops is not None else ALL_UNARY
    m = pysr.PySRRegressor(niterations=NITERS, timeout_in_seconds=TIMEOUT,
        binary_operators=bin_ops, unary_operators=un_ops,
        verbosity=0, random_state=seed, deterministic=True, parallelism='serial')
    t0 = time.time()
    try:
        m.fit(X_tr, y_tr)
        elapsed = time.time()-t0
        sc = r2(y_te, m.predict(X_te)); expr=str(m.sympy())
        # search-space proxy: n_ops^complexity grows the cost
        n_ops = len(bin_ops) + len(un_ops)
    except Exception:
        elapsed=time.time()-t0; sc=-99; expr="ERR"; n_ops=len(bin_ops)+len(un_ops)
    return sc, expr, elapsed, n_ops

def map_ops(pred_ops):
    """Map GAT operator predictions (17-op vocab) to PySR binary/unary lists.
    Always keep all binary ops; restrict unary ops to predicted subset of
    the full library. 'square'/'cube' map to PySR's square/cube unary."""
    bin_out = ['+','-','*','/','^']
    unary_vocab = {'sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube'}
    un_pred = sorted({o for o in pred_ops if o in unary_vocab})
    if not un_pred: un_pred = ['sqrt']   # safe default
    return bin_out, un_pred

results = []
print(f"{'#':3s} {'Formula':26s} {'nv':3s} {'sel':3s} {'pred_ops':22s} {'fR2':>6s} {'vR2':>6s} {'voR2':>6s} {'fEX vEX voEX'}")
print("-"*115)

for i, task in enumerate(tasks_raw):
    features = task["features"]
    # FIXED seed for true-variable sampling (independent of N_DIST)
    aug0, target = sample_feynman(task, Q, SEED*1000 + i)
    te0,  _      = sample_feynman(task, HELDOUT, SEED*1000 + i + 500000)
    if aug0 is None or te0 is None: continue

    # Add noise with separate seed
    aug = add_noise(aug0, target, N_DIST, SEED*7 + i)
    te  = add_noise(te0,  target, N_DIST, SEED*7 + i + 500000)

    full_cols = [c for c in aug if c != target]
    sel_cols, pred_ops = predict_support_and_ops(aug, target)
    b_ops, u_ops = map_ops(pred_ops)

    y_tr = aug[target]; y_te = te[target]
    n_full_vars = len(full_cols); n_sel_vars = len(sel_cols)
    n_full_ops  = len(ALL_BIN)+len(ALL_UNARY); n_pred_ops = len(b_ops)+len(u_ops)
    # A) Full PySR
    Xf_tr=np.column_stack([aug[c] for c in full_cols]); Xf_te=np.column_stack([te[c] for c in full_cols])
    full_r2,_,full_t,_ = run_pysr(Xf_tr,y_tr,Xf_te,y_te,SEED+i)
    # B) Ours-vars only
    Xo_tr=np.column_stack([aug[c] for c in sel_cols]); Xo_te=np.column_stack([te[c] for c in sel_cols])
    var_r2,_,var_t,_ = run_pysr(Xo_tr,y_tr,Xo_te,y_te,SEED+i)
    # C) Ours-vars + ops
    varop_r2,_,varop_t,_ = run_pysr(Xo_tr,y_tr,Xo_te,y_te,SEED+i,bin_ops=b_ops,un_ops=u_ops)

    # Search-space proxy: (n_vars + n_ops) ^ tree_depth ; report leaf-vocabulary size
    full_vocab = n_full_vars + n_full_ops
    var_vocab  = n_sel_vars  + n_full_ops
    varop_vocab= n_sel_vars  + n_pred_ops

    fE=full_r2>=0.9999; vE=var_r2>=0.9999; voE=varop_r2>=0.9999
    results.append({"law_id":task["law_id"],"nvars":len(features),"full_cols":len(full_cols),
                    "sel_cols":len(sel_cols),"pred_ops":sorted(pred_ops),"n_pred_ops":n_pred_ops,
                    "full_r2":full_r2,"var_r2":var_r2,"varop_r2":varop_r2,
                    "full_exact":fE,"var_exact":vE,"varop_exact":voE,
                    "full_t":full_t,"var_t":var_t,"varop_t":varop_t,
                    "full_vocab":full_vocab,"var_vocab":var_vocab,"varop_vocab":varop_vocab,
                    "recall":len(set(sel_cols)&set(features))/len(features)})
    print(f"{i+1:3d} {task['law_id'][:26]:26s} {len(features):3d} {len(sel_cols):3d} "
          f"{','.join(sorted(pred_ops))[:22]:22s} {full_r2:6.3f} {var_r2:6.3f} {varop_r2:6.3f}  "
          f"{'Y' if fE else '-'}   {'Y' if vE else '-'}   {'Y' if voE else '-'}")

print("-"*115)
n=len(results)
fe=sum(r["full_exact"] for r in results); ve=sum(r["var_exact"] for r in results); voe=sum(r["varop_exact"] for r in results)
print(f"\n{'='*60}")
print(f"EXACT RECOVERY  (n={n}, {N_DIST} distractors, {TIMEOUT}s):")
print(f"  A) Full PySR            ({np.mean([r['full_cols'] for r in results]):.0f} cols, all ops):  {fe}/{n} ({100*fe/n:.0f}%)")
print(f"  B) Ours-vars + PySR     ({np.mean([r['sel_cols'] for r in results]):.1f} cols, all ops):  {ve}/{n} ({100*ve/n:.0f}%)")
print(f"  C) Ours-vars+ops + PySR ({np.mean([r['sel_cols'] for r in results]):.1f} cols, pred ops): {voe}/{n} ({100*voe/n:.0f}%)")
print(f"\nMEAN R2:   full={np.mean([r['full_r2'] for r in results if r['full_r2']>-50]):.3f}  "
      f"vars={np.mean([r['var_r2'] for r in results if r['var_r2']>-50]):.3f}  "
      f"vars+ops={np.mean([r['varop_r2'] for r in results if r['varop_r2']>-50]):.3f}")
print(f"MEAN TIME: full={np.mean([r['full_t'] for r in results]):.2f}s  "
      f"vars={np.mean([r['var_t'] for r in results]):.2f}s  "
      f"vars+ops={np.mean([r['varop_t'] for r in results]):.2f}s")
fv=np.mean([r['full_vocab'] for r in results]); vv=np.mean([r['var_vocab'] for r in results]); vov=np.mean([r['varop_vocab'] for r in results])
print(f"\nSEARCH VOCAB (vars+ops, smaller=faster per iteration):")
print(f"  A) Full:        {fv:.1f}")
print(f"  B) Ours-vars:   {vv:.1f}  ({100*(1-vv/fv):.0f}% smaller)")
print(f"  C) Ours-vars+ops: {vov:.1f}  ({100*(1-vov/fv):.0f}% smaller)")
print(f"  mean pred_ops kept: {np.mean([r['n_pred_ops'] for r in results]):.1f} / {len(ALL_BIN)+len(ALL_UNARY)}")
print(f"COL REDUCTION: {np.mean([r['full_cols'] for r in results]):.0f} -> {np.mean([r['sel_cols'] for r in results]):.1f} "
      f"({100*(1-np.mean([r['sel_cols'] for r in results])/np.mean([r['full_cols'] for r in results])):.0f}%)")
print(f"VAR RECALL: {np.mean([r['recall'] for r in results]):.3f}")
# Op-recall: did pred_ops contain all truly-needed ops? (proxy: did C match B exact)
op_safe = sum(1 for r in results if r['var_exact']==r['varop_exact'])
print(f"OP-PRIOR SAFE (C matches B exact): {op_safe}/{len(results)}")

Path("data/results/pysr_frontend_3way.json").write_text(json.dumps(results, indent=2))
print("\nSaved -> data/results/pysr_frontend_3way.json")
