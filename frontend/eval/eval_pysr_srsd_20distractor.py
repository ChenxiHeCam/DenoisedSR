"""SRSD-Feynman PySR 3-way under our standard 20-random-distractor protocol.

Setup: load each SRSD task, keep ONLY the true-variable columns (drop SRSD's
~2 physically plausible dummies), then pad with 20 random distractors sampled
from uniform[-5,5] (same protocol as the AI-Feynman headline). Run PySR full
vs +DenoisedSR-vars vs +var+op. Tests the boundary hypothesis: does the prior
help more when distractor density is the same as Feynman?
"""
import sys, json, os, re, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import pysr
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4
SEED   = int(os.environ.get("SEED","42"))
Q      = int(os.environ.get("Q","100"))
HELDOUT= 200
TIMEOUT= int(os.environ.get("TIMEOUT","10"))
NITERS = int(os.environ.get("NITERS","40"))
N_TASKS= int(os.environ.get("N_TASKS","120"))
N_DIST = int(os.environ.get("N_DIST","20"))
OUT_PATH = os.environ.get("OUT_PATH", "data/results/pysr_srsd_20dist_3way.json")
ALL_BIN  = ['+','-','*','/','^']
ALL_UNARY= ['sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube']
OP_THRESH = float(os.environ.get("OP_THRESH","0.03"))

class GATDisc(nn.Module):
    def __init__(self):
        super().__init__(); self.proj=nn.Linear(FEAT,HID)
        self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1); self.norm1=nn.LayerNorm(HID*HEADS)
        self.gat2=GATConv(HID*HEADS,HID,heads=HEADS,concat=True,dropout=0.1); self.norm2=nn.LayerNorm(HID*HEADS)
        self.gat3=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1); self.norm3=nn.LayerNorm(HID)
        self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.15),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x)); h=F.gelu(self.norm1(self.gat1(h,ei))); h=F.gelu(self.norm2(self.gat2(h,ei))); h=self.norm3(self.gat3(h,ei)); return self.head(h).squeeze(-1)

COOC = jl("models/cooc_graph.joblib")
rf_clf = jl("models/support_predictor_v2_40k.joblib")['support_clf']
gat = GATDisc(); gat.load_state_dict(torch.load("models/gat_best.pt", map_location="cpu")["model"]); gat.eval()
from train_gat_operator import GATOperator, build_graph_data, OP_FULL
gatop = GATOperator(); gatop.load_state_dict(torch.load("models/gat_operator.pt", map_location="cpu")["model"]); gatop.eval()

manifest = json.load(open("data/benchmarks/srsd/manifest.json"))
keys = [k for k in manifest if manifest[k]["n_true"] >= 1][:N_TASKS]
print(f"SRSD-PySR 3-way (20-random-distractor protocol): {len(keys)} tasks, q={Q}, t={TIMEOUT}s, seed={SEED}\n")

def predict(aug, target):
    cols = [c for c in aug if c != target]
    feats = np.array([v2.column_features(np.asarray(aug[c]), np.asarray(aug[target]), var_name=c) for c in cols], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0., posinf=10., neginf=-10.)
    rf_s = positive_probability(rf_clf, feats)
    edges = set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges: gat_s = np.zeros(len(cols))
    else:
        src,dst = zip(*edges); ei = torch.tensor([list(src),list(dst)],dtype=torch.long)
        ei,_ = add_self_loops(ei, num_nodes=len(cols))
        with torch.no_grad(): gat_s = torch.sigmoid(gat(torch.tensor(feats), ei)).numpy()
    sel = [c for c,r,g in zip(cols,rf_s,gat_s) if r+g >= 0.10]
    if not sel: sel = [cols[int(np.argmax(rf_s))]]
    g = build_graph_data(aug, target, "")
    with torch.no_grad():
        bidx = torch.zeros(g.x.shape[0], dtype=torch.long)
        op_probs = torch.sigmoid(gatop(g.x, g.edge_index, bidx))[0].numpy()
    pred_ops = {op for op,p in zip(OP_FULL, op_probs) if p >= OP_THRESH}; pred_ops.update({'+','-','*','/'})
    return sel, pred_ops

def r2(yt, yp):
    yt = np.asarray(yt,dtype=float); yp = np.asarray(yp,dtype=float)
    if not np.all(np.isfinite(yp)): return float('-inf')
    ss = np.sum((yt-yp)**2); tot = np.sum((yt-np.mean(yt))**2)
    return float(1 - ss/(tot+1e-12))

def run_pysr(X_tr, y_tr, X_te, y_te, seed, bin_ops=None, un_ops=None):
    bin_ops = bin_ops or ALL_BIN; un_ops = un_ops if un_ops is not None else ALL_UNARY
    m = pysr.PySRRegressor(niterations=NITERS, timeout_in_seconds=TIMEOUT,
        binary_operators=bin_ops, unary_operators=un_ops,
        verbosity=0, random_state=seed, deterministic=True, parallelism='serial')
    t0 = time.time()
    try:
        m.fit(X_tr, y_tr); el = time.time()-t0
        sc = r2(y_te, m.predict(X_te)); expr=str(m.sympy())
    except Exception as e:
        el = time.time()-t0; sc = float('-inf'); expr = f"ERR: {e!s:.80}"
    return sc, expr, el

def map_ops(pred_ops):
    unary_vocab = {'sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube'}
    un_pred = sorted({o for o in pred_ops if o in unary_vocab}) or ['sqrt']
    return ['+','-','*','/','^'], un_pred

results = []
print(f"{'#':>3} {'task':30s} {'nt':>3} {'nc':>3} {'sel':>3}  {'fR2':>8} {'vR2':>8} {'voR2':>8} flag")
print("-"*100)
for i, key in enumerate(keys):
    info = manifest[key]
    true_set = set(info["true_vars"])
    rng = np.random.default_rng(SEED + i)
    data = np.loadtxt(info["file"])
    if data.ndim < 2 or data.shape[0] < Q + HELDOUT: continue
    perm = rng.permutation(data.shape[0])
    tr_idx, te_idx = perm[:Q], perm[Q:Q+HELDOUT]
    # SRSD true columns: indices for each x_i listed in true_vars
    true_idx = [int(re.match(r"x(\d+)", v).group(1)) for v in info["true_vars"]]
    Xtr_true = data[tr_idx][:, true_idx]; ytr = data[tr_idx, -1]
    Xte_true = data[te_idx][:, true_idx]; yte = data[te_idx, -1]
    if not np.all(np.isfinite(ytr)) or not np.all(np.isfinite(yte)): continue
    # Pad with N_DIST random distractors
    true_cols = [f"x{j}" for j in true_idx]
    aug = {true_cols[j]: Xtr_true[:,j] for j in range(len(true_cols))}; aug["__y"] = ytr
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
    sel, pred_ops = predict(aug, "__y")
    all_col_names = [c for c in aug if c != "__y"]
    X_full_tr = np.column_stack([aug[c] for c in all_col_names])
    # build heldout matching the same distractor distribution (use independent draws)
    rng_te = np.random.default_rng(SEED + i + 99999)
    te_aug = {true_cols[j]: Xte_true[:,j] for j in range(len(true_cols))}
    for j in range(N_DIST):
        lo,hi = float(rng_te.uniform(-5,0)), float(rng_te.uniform(0,5))
        te_aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), HELDOUT, rng_te)
    X_full_te = np.column_stack([te_aug[c] for c in all_col_names])
    # Full
    fr2, fexp, ft = run_pysr(X_full_tr, ytr, X_full_te, yte, SEED+i)
    sel_idx = [all_col_names.index(s) for s in sel if s in all_col_names] or list(range(min(4, len(all_col_names))))
    X_sel_tr = X_full_tr[:, sel_idx]; X_sel_te = X_full_te[:, sel_idx]
    vr2, vexp, vt = run_pysr(X_sel_tr, ytr, X_sel_te, yte, SEED+i)
    b_ops, u_ops = map_ops(pred_ops)
    vor2, voexp, vot = run_pysr(X_sel_tr, ytr, X_sel_te, yte, SEED+i, bin_ops=b_ops, un_ops=u_ops)
    fE=fr2>=0.9999; vE=vr2>=0.9999; voE=vor2>=0.9999
    rec = len(set(sel) & true_set) / max(1, len(true_set))
    results.append({"key": key, "n_true": len(true_set), "n_cols": len(all_col_names), "sel_cols": len(sel_idx),
                    "full_r2":fr2, "var_r2":vr2, "varop_r2":vor2,
                    "full_exact":fE, "var_exact":vE, "varop_exact":voE,
                    "full_t":ft, "var_t":vt, "varop_t":vot,
                    "true_vars":sorted(true_set), "sel_vars":sorted(sel),
                    "full_expr":fexp, "var_expr":vexp, "varop_expr":voexp, "recall":rec})
    sym = ("F" if fE else "-")+("V" if vE else "-")+("O" if voE else "-")
    print(f"{i+1:>3} {key[:30]:30s} {len(true_set):>3} {len(all_col_names):>3} {len(sel_idx):>3}  "
          f"{fr2:>8.3f} {vr2:>8.3f} {vor2:>8.3f}  {sym}")

n = len(results)
fe = sum(r['full_exact'] for r in results); ve = sum(r['var_exact'] for r in results); voe = sum(r['varop_exact'] for r in results)
print("="*80)
print(f"SRSD-PySR (20-distractor protocol), n={n}:")
print(f"  Full       : {fe}/{n} ({100*fe/n:.0f}%) exact")
print(f"  +DenoisedSR vars : {ve}/{n} ({100*ve/n:.0f}%) exact")
print(f"  +var+op    : {voe}/{n} ({100*voe/n:.0f}%) exact")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"Saved -> {OUT_PATH}")
