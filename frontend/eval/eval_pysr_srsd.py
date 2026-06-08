"""
End-to-end PySR comparison on external SRSD-Feynman dummy suite.
DenoisedSR + PySR vs. full PySR vs. + operator prior.
Same protocol as eval_pysr_frontend.py but reads SRSD txt files
(8000 rows per task, real physical-unit scales). q=100 subsample.

Note on dynamic range: SRSD intentionally uses physical units
(e.g. Planck constant ~ 1e-34, electron mass ~ 1e-30), so y values
can span ~1e-34 .. 1e+26 across tasks. PySR is not scale-invariant;
many tasks will likely give R^2 << 1 for ALL conditions. We report
honestly: the comparison is (DenoisedSR+PySR) vs (full PySR) on the
SAME hard data, not whether either reaches exact recovery.
"""
import sys, json, re, time, warnings, os
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train"))

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import pysr
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability, OP_ORDER

FEAT=74; HID=128; HEADS=4
SEED   = int(os.environ.get("SEED","42"))
Q      = int(os.environ.get("Q","100"))
HELDOUT= 200
TIMEOUT= int(os.environ.get("TIMEOUT","10"))
NITERS = int(os.environ.get("NITERS","40"))
N_TASKS= int(os.environ.get("N_TASKS","120"))
OUT_PATH = os.environ.get("OUT_PATH", "data/results/pysr_srsd_3way.json")
ALL_BIN  = ['+','-','*','/','^']
ALL_UNARY= ['sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube']
OP_THRESH = float(os.environ.get("OP_THRESH","0.03"))

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
from train_gat_operator import GATOperator, build_graph_data, OP_FULL
gatop_ckpt = torch.load("models/gat_operator.pt", map_location="cpu")
gatop = GATOperator(); gatop.load_state_dict(gatop_ckpt["model"]); gatop.eval()

manifest = json.load(open("data/benchmarks/srsd/manifest.json"))
keys = [k for k in manifest if manifest[k]["n_true"] >= 1][:N_TASKS]
print(f"SRSD-PySR 3-way: {len(keys)} formulas (q={Q}, t={TIMEOUT}s, seed={SEED}), tau=0.10\n")

def predict(aug, target):
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
            for j in range(i+1, len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst = zip(*edges)
    ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
    ei,_ = add_self_loops(ei, num_nodes=len(cols))
    with torch.no_grad():
        gat_s = torch.sigmoid(gat(torch.tensor(feats), ei)).numpy()
    sel = [c for c,rs,gs in zip(cols, rf_s, gat_s) if rs+gs >= 0.10]
    if not sel: sel = [cols[int(np.argmax(rf_s))]]
    g = build_graph_data(aug, target, "")
    with torch.no_grad():
        bidx = torch.zeros(g.x.shape[0], dtype=torch.long)
        op_probs = torch.sigmoid(gatop(g.x, g.edge_index, bidx))[0].numpy()
    pred_ops = {op for op,p in zip(OP_FULL, op_probs) if p >= OP_THRESH}
    pred_ops.update({'+','-','*','/'})
    return sel, pred_ops

def r2(yt, yp):
    yt = np.asarray(yt, dtype=float); yp = np.asarray(yp, dtype=float)
    if not np.all(np.isfinite(yp)): return float('-inf')
    ss = np.sum((yt-yp)**2); tot = np.sum((yt-np.mean(yt))**2)
    return float(1 - ss/(tot+1e-12))

def run_pysr(X_tr, y_tr, X_te, y_te, seed, bin_ops=None, un_ops=None):
    bin_ops = bin_ops or ALL_BIN
    un_ops  = un_ops if un_ops is not None else ALL_UNARY
    m = pysr.PySRRegressor(niterations=NITERS, timeout_in_seconds=TIMEOUT,
        binary_operators=bin_ops, unary_operators=un_ops,
        verbosity=0, random_state=seed, deterministic=True, parallelism='serial')
    t0 = time.time()
    try:
        m.fit(X_tr, y_tr)
        elapsed = time.time()-t0
        sc = r2(y_te, m.predict(X_te)); expr=str(m.sympy())
    except Exception as e:
        elapsed = time.time()-t0; sc = float('-inf'); expr = f"ERR: {e!s:.80}"
    return sc, expr, elapsed

def map_ops(pred_ops):
    bin_out = ['+','-','*','/','^']
    unary_vocab = {'sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube'}
    un_pred = sorted({o for o in pred_ops if o in unary_vocab})
    if not un_pred: un_pred = ['sqrt']
    return bin_out, un_pred

results = []
print(f"{'#':>3} {'task':30s} {'nt':>2} {'nc':>2} {'sel':>3}  {'fR2':>8} {'vR2':>8} {'voR2':>8}  exact?")
print("-"*100)
for i, key in enumerate(keys):
    info = manifest[key]
    true_set = set(info["true_vars"])
    rng = np.random.default_rng(SEED + i)
    data = np.loadtxt(info["file"])
    if data.ndim < 2 or data.shape[0] < Q + HELDOUT: continue
    # split into train + heldout
    perm = rng.permutation(data.shape[0])
    tr_idx, te_idx = perm[:Q], perm[Q:Q+HELDOUT]
    Xtr_full, ytr = data[tr_idx, :-1], data[tr_idx, -1]
    Xte_full, yte = data[te_idx, :-1], data[te_idx, -1]
    if not np.all(np.isfinite(ytr)) or not np.all(np.isfinite(yte)):
        continue
    n_cols = Xtr_full.shape[1]
    col_names = [f"x{j}" for j in range(n_cols)]
    aug = {col_names[j]: Xtr_full[:,j] for j in range(n_cols)}; aug["__y"] = ytr
    sel, pred_ops = predict(aug, "__y")
    sel_idx = [col_names.index(s) for s in sel if s in col_names]
    if not sel_idx: sel_idx = list(range(n_cols))
    b_ops, u_ops = map_ops(pred_ops)
    # Full
    fr2, fexp, ft = run_pysr(Xtr_full, ytr, Xte_full, yte, SEED+i)
    # Var prior
    Xtr_sel = Xtr_full[:, sel_idx]; Xte_sel = Xte_full[:, sel_idx]
    vr2, vexp, vt = run_pysr(Xtr_sel, ytr, Xte_sel, yte, SEED+i)
    # Var+op
    vor2, voexp, vot = run_pysr(Xtr_sel, ytr, Xte_sel, yte, SEED+i, bin_ops=b_ops, un_ops=u_ops)
    fE = fr2 >= 0.9999; vE = vr2 >= 0.9999; voE = vor2 >= 0.9999
    rec = len(set(sel) & true_set) / max(1, len(true_set))
    results.append({"key": key, "split": info["split"], "n_true": len(true_set),
                    "n_dummy": info["n_dummy"], "n_cols": n_cols, "sel_cols": len(sel_idx),
                    "full_r2": fr2, "var_r2": vr2, "varop_r2": vor2,
                    "full_exact": fE, "var_exact": vE, "varop_exact": voE,
                    "full_t": ft, "var_t": vt, "varop_t": vot,
                    "true_vars": sorted(true_set), "sel_vars": sorted(sel),
                    "full_expr": fexp, "var_expr": vexp, "varop_expr": voexp,
                    "recall_intrinsic": rec})
    sym = ("F" if fE else "-") + ("V" if vE else "-") + ("O" if voE else "-")
    print(f"{i+1:>3} {key[:30]:30s} {len(true_set):>2} {n_cols:>2} {len(sel_idx):>3}  "
          f"{fr2:>8.3f} {vr2:>8.3f} {vor2:>8.3f}   {sym}")

n = len(results)
if n:
    fe = sum(r['full_exact'] for r in results)
    ve = sum(r['var_exact'] for r in results)
    voe = sum(r['varop_exact'] for r in results)
    def mr2(k):
        vals = [max(r[k], -1.0) if np.isfinite(r[k]) else -1.0 for r in results]
        return sum(vals)/len(vals)
    print("="*80)
    print(f"Exact recovery: full {fe}/{n} ({100*fe/n:.1f}%), "
          f"DenoisedSR-vars {ve}/{n} ({100*ve/n:.1f}%), +ops {voe}/{n} ({100*voe/n:.1f}%)")
    print(f"Mean R^2 (floored at -1): full={mr2('full_r2'):.3f}, var={mr2('var_r2'):.3f}, +ops={mr2('varop_r2'):.3f}")
    print(f"Mean recall (intrinsic): {np.mean([r['recall_intrinsic'] for r in results]):.3f}")
    print(f"Col reduction: {np.mean([r['n_cols'] for r in results]):.1f} -> {np.mean([r['sel_cols'] for r in results]):.1f}")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {OUT_PATH}")
