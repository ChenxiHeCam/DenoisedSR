"""PSE 50-dim benchmark from Ruan et al. 2026 NCS paper.
20 synthetic problems, 12 true vars among 50 columns (38 distractors).
Runs full PSE vs DenoisedSR + PSE 3-way."""
import sys, json, os, re, time, warnings
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

PSE_REPO = "D:/Physics Fundation model/third_party/PSE"
DR_MASK = f"{PSE_REPO}/dr_mask"
DATA_DIR = f"{PSE_REPO}/data/many_input/synthetic_50d_datasets_with_gt"
SEED = int(os.environ.get("SEED","42"))
Q = int(os.environ.get("Q","100"))
HELDOUT = 200
TIMEOUT = int(os.environ.get("TIMEOUT","30"))
BACKEND = os.environ.get("BACKEND","pse").lower()
OUT_PATH = os.environ.get("OUT_PATH", f"data/results/{BACKEND}_pse50dim_3way.json")

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

# Load 20 PSE 50-dim tasks
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
txt_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
print(f"{BACKEND} on PSE-50dim ({len(csv_files)} tasks, q={Q}, t={TIMEOUT}s, seed={SEED})\n")

def parse_true_vars(txt_content):
    """Extract x_N indices used in formula. PSE uses 1-indexed x_N notation."""
    return sorted(set(int(m) for m in re.findall(r"x_(\d+)", txt_content)))

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
    yt=np.asarray(yt,dtype=float); yp=np.asarray(yp,dtype=float)
    if not np.all(np.isfinite(yp)): return float('-inf')
    ss=np.sum((yt-yp)**2); tot=np.sum((yt-np.mean(yt))**2)
    return float(1 - ss/(tot+1e-12))

# Backend dispatch
if BACKEND == "pse":
    from psrn import PSRN_Regressor
    OPS = ['Add','Mul','Sub','Div','Identity']  # matches PSE 50dim default
    PSE_STAGE = {"default":{"operators":OPS,"time_limit":TIMEOUT,"n_psrn_inputs":4,"n_sample_variables":4},
                 "stages":[{"time_limit":TIMEOUT,"n_psrn_inputs":4}]}
    PSE_TG = {"base":{"has_const":False,"tokens":OPS}}
    def run_backend(X, y, X_te, y_te, seed):
        var_names = [f"x{i}" for i in range(X.shape[1])]
        t0=time.time()
        try:
            torch.cuda.empty_cache()
            reg = PSRN_Regressor(variables=var_names, n_inputs=min(4,X.shape[1]),
                n_symbol_layers=3, use_const=False, use_extra_const=False,
                dr_mask_dir=DR_MASK, stage_config=PSE_STAGE, token_generator_config=PSE_TG,
                device='cuda')
            reg.fit(X, y.reshape(-1,1), n_down_sample=20, real_time_display=False, threshold=1e-10)
            el=time.time()-t0
            pf = reg.get_pf(sort_by="mse")
            expr = pf[0][0]
            yp = reg.predict(X_te).flatten()
            return r2(y_te, yp), expr, el
        except Exception as e:
            return float('-inf'), f"ERR:{type(e).__name__}", time.time()-t0
elif BACKEND == "pysr":
    import pysr
    def run_backend(X, y, X_te, y_te, seed):
        m = pysr.PySRRegressor(niterations=40, timeout_in_seconds=TIMEOUT,
            binary_operators=['+','-','*','/','^'], unary_operators=['sqrt','sin','cos','exp','log','abs','square'],
            verbosity=0, random_state=seed, deterministic=True, parallelism='serial')
        t0=time.time()
        try:
            m.fit(X, y); el=time.time()-t0
            return r2(y_te, m.predict(X_te)), str(m.sympy()), el
        except Exception as e:
            return float('-inf'), f"ERR:{e!s:.80}", time.time()-t0
elif BACKEND == "gplearn":
    from gplearn.genetic import SymbolicRegressor
    def run_backend(X, y, X_te, y_te, seed):
        m = SymbolicRegressor(population_size=500, generations=20, function_set=('add','sub','mul','div','sqrt','log','sin','cos','abs'),
                              random_state=seed, verbose=0, n_jobs=1, parsimony_coefficient=0.001, max_samples=0.9, stopping_criteria=1e-8)
        t0=time.time()
        try:
            m.fit(X, y); el=time.time()-t0
            return r2(y_te, m.predict(X_te)), str(m._program), el
        except Exception as e:
            return float('-inf'), f"ERR:{e!s:.80}", time.time()-t0
else: raise ValueError(f"BACKEND {BACKEND}")

results = []
print(f"{'#':>3} {'task':16s} {'true_n':>6} {'sel':>3}  {'fR2':>8} {'vR2':>8}  flag")
print("-"*80)
for i, (csv_name, txt_name) in enumerate(zip(csv_files, txt_files)):
    assert csv_name.replace(".csv","") == txt_name.replace(".txt","")
    tid = csv_name.replace(".csv","")
    formula = open(os.path.join(DATA_DIR, txt_name)).read().strip()
    true_idx = parse_true_vars(formula)  # 1-indexed
    rng = np.random.default_rng(SEED + i)
    data = np.loadtxt(os.path.join(DATA_DIR, csv_name), delimiter=",")
    n_total, n_cols_plus_y = data.shape
    n_cols = n_cols_plus_y - 1  # last col is y
    if n_total < Q + HELDOUT: continue
    perm = rng.permutation(n_total)
    tr_idx, te_idx = perm[:Q], perm[Q:Q+HELDOUT]
    Xtr = data[tr_idx, :-1]; ytr = data[tr_idx, -1]
    Xte = data[te_idx, :-1]; yte = data[te_idx, -1]
    if not np.all(np.isfinite(ytr)) or not np.all(np.isfinite(yte)): continue
    col_names = [f"x{j}" for j in range(n_cols)]  # 0-indexed for our model
    true_var_set = set(f"x{i_-1}" for i_ in true_idx)  # convert 1-indexed -> 0-indexed
    aug = {col_names[j]: Xtr[:,j] for j in range(n_cols)}; aug["__y"] = ytr
    sel = predict_vars(aug, "__y")
    sel_idx = [col_names.index(s) for s in sel if s in col_names] or list(range(min(4, n_cols)))
    fr2, fexp, ft = run_backend(Xtr, ytr, Xte, yte, SEED+i)
    vr2, vexp, vt = run_backend(Xtr[:,sel_idx], ytr, Xte[:,sel_idx], yte, SEED+i)
    fE = fr2 >= 0.9999; vE = vr2 >= 0.9999
    rec = len(set(sel) & true_var_set) / max(1, len(true_var_set))
    results.append({"task":tid, "n_true":len(true_var_set), "n_cols":n_cols, "sel_cols":len(sel_idx),
                    "full_r2":fr2, "var_r2":vr2, "full_exact":fE, "var_exact":vE,
                    "full_t":ft, "var_t":vt, "true_vars":sorted(true_var_set), "sel_vars":sorted(sel),
                    "full_expr":fexp, "var_expr":vexp, "recall":rec, "formula":formula})
    sym = ("F" if fE else "-")+("V" if vE else "-")
    print(f"{i+1:>3} {tid:16s} {len(true_var_set):>6} {len(sel_idx):>3}  {fr2:>8.3f} {vr2:>8.3f}  {sym}")

n = len(results)
if n:
    fe = sum(r['full_exact'] for r in results); ve = sum(r['var_exact'] for r in results)
    avg_rec = np.mean([r['recall'] for r in results])
    avg_sel = np.mean([r['sel_cols'] for r in results])
    print(f"\n{BACKEND} on PSE-50dim, n={n}:")
    print(f"  Full PSE (50 cols)  : {fe}/{n} ({100*fe/n:.0f}%) exact")
    print(f"  + DenoisedSR (~{avg_sel:.0f} cols): {ve}/{n} ({100*ve/n:.0f}%) exact")
    print(f"  Variable recall    : {avg_rec:.3f}")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"Saved -> {OUT_PATH}")
