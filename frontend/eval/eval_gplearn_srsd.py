"""gplearn backend on SRSD-Feynman (native or +N_DIST random distractors)."""
import sys, json, os, re, time, warnings
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
from gplearn.genetic import SymbolicRegressor
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

SEED = int(os.environ.get("SEED","42"))
Q = int(os.environ.get("Q","100"))
HELDOUT = 200
N_DIST = int(os.environ.get("N_DIST","0"))
N_TASKS = int(os.environ.get("N_TASKS","120"))
GP_GENS = int(os.environ.get("GP_GENS","20"))
POP = int(os.environ.get("POP","500"))
OUT_PATH = os.environ.get("OUT_PATH", f"data/results/gplearn_srsd_d{N_DIST}_3way.json")
GP_FN = ('add','sub','mul','div','sqrt','log','sin','cos','tan','abs')

FEAT=74; HID=128; HEADS=4
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
manifest = json.load(open("data/benchmarks/srsd/manifest.json"))
keys = [k for k in manifest if manifest[k]["n_true"] >= 1][:N_TASKS]
print(f"gplearn on SRSD ({'native' if N_DIST==0 else f'{N_DIST}-distractor'}): {len(keys)} tasks, q={Q}, gens={GP_GENS}, pop={POP}, seed={SEED}\n")

def predict_vars(aug, target):
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
    return [c for c,r,g in zip(cols,rf_s,gat_s) if r+g >= 0.10]

def r2(yt, yp):
    yt = np.asarray(yt,dtype=float); yp = np.asarray(yp,dtype=float)
    if not np.all(np.isfinite(yp)): return float('-inf')
    ss = np.sum((yt-yp)**2); tot = np.sum((yt-np.mean(yt))**2)
    return float(1 - ss/(tot+1e-12))

def run_gp(X_tr, y_tr, X_te, y_te, seed):
    m = SymbolicRegressor(population_size=POP, generations=GP_GENS, function_set=GP_FN,
                          random_state=seed, verbose=0, n_jobs=1,
                          parsimony_coefficient=0.001, max_samples=0.9, stopping_criteria=1e-8)
    t0 = time.time()
    try:
        m.fit(X_tr, y_tr); el = time.time()-t0
        sc = r2(y_te, m.predict(X_te))
        return sc, str(m._program), el
    except Exception as e:
        return float('-inf'), f"ERR: {e!s:.80}", time.time()-t0

results = []
print(f"{'#':>3} {'task':30s} {'nt':>3} {'nc':>3} {'sel':>3}  {'fR2':>8} {'vR2':>8}  flag")
print("-"*90)
for i, key in enumerate(keys):
    info = manifest[key]
    true_set = set(info["true_vars"])
    rng = np.random.default_rng(SEED + i)
    data = np.loadtxt(info["file"])
    if data.ndim < 2 or data.shape[0] < Q + HELDOUT: continue
    perm = rng.permutation(data.shape[0])
    tr_idx, te_idx = perm[:Q], perm[Q:Q+HELDOUT]
    if N_DIST == 0:
        Xtr = data[tr_idx, :-1]; ytr = data[tr_idx, -1]
        Xte = data[te_idx, :-1]; yte = data[te_idx, -1]
        col_names = [f"x{j}" for j in range(Xtr.shape[1])]
    else:
        true_idx = [int(re.match(r"x(\d+)", v).group(1)) for v in info["true_vars"]]
        Xtr_true = data[tr_idx][:, true_idx]; ytr = data[tr_idx, -1]
        Xte_true = data[te_idx][:, true_idx]; yte = data[te_idx, -1]
        true_cols = [f"x{j}" for j in true_idx]
        cols_tr = {c: Xtr_true[:,j] for j,c in enumerate(true_cols)}
        cols_te = {c: Xte_true[:,j] for j,c in enumerate(true_cols)}
        rng_te = np.random.default_rng(SEED + i + 99999)
        for j in range(N_DIST):
            lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
            cols_tr[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
            lo2,hi2 = float(rng_te.uniform(-5,0)), float(rng_te.uniform(0,5))
            cols_te[f"__d{j}"] = v2.resample(np.array([lo2,hi2]), HELDOUT, rng_te)
        col_names = list(cols_tr.keys())
        Xtr = np.column_stack([cols_tr[c] for c in col_names])
        Xte = np.column_stack([cols_te[c] for c in col_names])
    if not np.all(np.isfinite(ytr)) or not np.all(np.isfinite(yte)): continue
    aug = {col_names[j]: Xtr[:,j] for j in range(len(col_names))}; aug["__y"] = ytr
    sel = predict_vars(aug, "__y")
    sel_idx = [col_names.index(s) for s in sel if s in col_names] or list(range(min(4, len(col_names))))
    fr2, fexp, ft = run_gp(Xtr, ytr, Xte, yte, SEED+i)
    vr2, vexp, vt = run_gp(Xtr[:, sel_idx], ytr, Xte[:, sel_idx], yte, SEED+i)
    fE = fr2 >= 0.9999; vE = vr2 >= 0.9999
    rec = len(set(sel) & true_set) / max(1, len(true_set))
    results.append({"key": key, "n_true": len(true_set), "n_cols": Xtr.shape[1], "sel_cols": len(sel_idx),
                    "full_r2": fr2, "var_r2": vr2,
                    "full_exact": fE, "var_exact": vE,
                    "full_t": ft, "var_t": vt,
                    "true_vars": sorted(true_set), "sel_vars": sorted(sel),
                    "full_expr": fexp, "var_expr": vexp, "recall": rec})
    sym = ("F" if fE else "-")+("V" if vE else "-")
    print(f"{i+1:>3} {key[:30]:30s} {len(true_set):>3} {Xtr.shape[1]:>3} {len(sel_idx):>3}  {fr2:>8.3f} {vr2:>8.3f}  {sym}")

n = len(results)
if n:
    fe = sum(r['full_exact'] for r in results); ve = sum(r['var_exact'] for r in results)
    print(f"\ngplearn on SRSD ({'native' if N_DIST==0 else f'+{N_DIST}dist'}), n={n}: "
          f"full {fe}/{n} ({100*fe/n:.0f}%), DenoisedSR {ve}/{n} ({100*ve/n:.0f}%)")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"Saved -> {OUT_PATH}")
