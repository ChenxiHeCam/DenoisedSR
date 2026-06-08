"""Third backend: PSE (Ruan et al. 2026 Nature Computational Science cover).

GPU-accelerated parallel symbolic enumeration. Test whether the DenoisedSR
variable prior also speeds up + improves recovery for this very different
backend (PySR=evolutionary, gplearn=classical GP, PSE=parallel enumeration).
"""
import sys, json, os, time, warnings
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train"))

import numpy as np, sympy as sp, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability
from psrn import PSRN_Regressor

PSE_REPO = "D:/Physics Fundation model/third_party/PSE"
DR_MASK = f"{PSE_REPO}/dr_mask"
SEED = int(os.environ.get("SEED","42"))
Q = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
N_TASKS = int(os.environ.get("N_TASKS","30"))
TIMEOUT_S = int(os.environ.get("TIMEOUT","30"))  # per condition per task
OUT_PATH = os.environ.get("OUT_PATH", "data/results/pse_backend_3way.json")

OPS = ['Add', 'Mul', 'Sub', 'Div', 'Identity', 'Sin', 'Cos', 'Exp', 'Log']
STAGE_DICT = {
    "default": {"operators": OPS, "time_limit": TIMEOUT_S, "n_psrn_inputs": 4, "n_sample_variables": 4},
    "stages": [{"time_limit": TIMEOUT_S, "n_psrn_inputs": 4}],
}
TG_DICT = {"base": {"has_const": False, "tokens": OPS}}

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

def predict_vars(aug, target):
    cols = [c for c in aug if c != target]
    feats = np.array([v2.column_features(np.asarray(aug[c]), np.asarray(aug[target]), var_name=c) for c in cols], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0., posinf=10., neginf=-10.)
    rf_s = positive_probability(rf_clf, feats)
    edges = set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges:
        gat_s = np.zeros(len(cols))
    else:
        src,dst = zip(*edges)
        ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
        ei,_ = add_self_loops(ei, num_nodes=len(cols))
        with torch.no_grad():
            gat_s = torch.sigmoid(gat(torch.tensor(feats), ei)).numpy()
    return [c for c,r,g in zip(cols, rf_s, gat_s) if r+g >= 0.10]

tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks_raw = [t for t in tasks_raw if 2 <= len(t["features"]) <= 9][:N_TASKS]
print(f"PSE 3rd backend: {N_TASKS} tasks, q={Q}, n_dist={N_DIST}, timeout={TIMEOUT_S}s\n")

def sample(task, n, rng):
    formula = task["formula"]; features = task["features"]; ranges = task.get("ranges",{})
    lhs, rhs = formula.split("=",1); target = lhs.strip()
    syms = {f: sp.Symbol(f) for f in features}
    fn = sp.lambdify([syms[f] for f in features], sp.sympify(rhs.strip(), locals=syms), "numpy")
    cols = {}
    for f in features:
        rg = ranges.get(f,[1.0,5.0]); cols[f] = rng.uniform(float(rg[0]), float(rg[1]), n)
    y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    if not np.all(np.isfinite(y)): return None, None
    cols[target] = y
    return cols, target

def run_pse(X, y, label=""):
    var_names = [f"x{i}" for i in range(X.shape[1])]
    t0 = time.time()
    try:
        torch.cuda.empty_cache()
        reg = PSRN_Regressor(
            variables=var_names, n_inputs=min(4, X.shape[1]),
            n_symbol_layers=3, use_const=False, use_extra_const=True,
            dr_mask_dir=DR_MASK, stage_config=STAGE_DICT, token_generator_config=TG_DICT,
            device='cuda',
        )
        reg.fit(X, y.reshape(-1,1), n_down_sample=20, real_time_display=False, threshold=1e-10)
        elapsed = time.time()-t0
        pf = reg.get_pf(sort_by="mse")
        expr = pf[0][0]; mse = pf[0][2]
        # compute R^2
        yp = reg.predict(X).flatten()
        ss = np.sum((y-yp)**2); tot = np.sum((y-np.mean(y))**2)
        r2 = float(1 - ss/(tot+1e-12)) if np.all(np.isfinite(yp)) else float('-inf')
        return r2, expr, elapsed
    except Exception as e:
        return float('-inf'), f"ERR: {type(e).__name__}", time.time()-t0

results = []
print(f"{'#':>3} {'law':28s} {'nv':>2} {'sel':>3}  {'fR2':>7} {'vR2':>7}  {'fT':>5} {'vT':>5}  exact?")
print("-"*90)
for i, task in enumerate(tasks_raw):
    features = task["features"]
    rng = np.random.default_rng(SEED + i)
    cols, target = sample(task, Q, rng)
    if cols is None: continue
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        cols[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
    sel = predict_vars(cols, target)
    all_col_names = [c for c in cols if c != target]
    y_arr = np.asarray(cols[target], dtype=float)
    # FULL
    X_full = np.column_stack([cols[c] for c in all_col_names]).astype(float)
    fr2, fexp, ft = run_pse(X_full, y_arr, "FULL")
    # DenoisedSR vars
    sel_idx = [all_col_names.index(s) for s in sel if s in all_col_names] or list(range(min(4, len(all_col_names))))
    X_red = X_full[:, sel_idx]
    vr2, vexp, vt = run_pse(X_red, y_arr, "VAR")
    fE = fr2 >= 0.9999; vE = vr2 >= 0.9999
    rec = len(set(sel) & set(features))/max(1, len(features))
    results.append({"law_id": task['law_id'], "nvars": len(features),
                    "full_cols": len(all_col_names), "sel_cols": len(sel_idx),
                    "full_r2": fr2, "var_r2": vr2,
                    "full_exact": fE, "var_exact": vE,
                    "full_t": ft, "var_t": vt,
                    "true_vars": features, "sel_vars": sorted(sel),
                    "full_expr": fexp, "var_expr": vexp,
                    "recall": rec})
    print(f"{i+1:>3} {task['law_id'][:28]:28s} {len(features):>2} {len(sel_idx):>3}  "
          f"{fr2:>7.3f} {vr2:>7.3f}  {ft:>5.1f} {vt:>5.1f}  "
          f"{'F' if fE else '-'} {'V' if vE else '-'}")

n = len(results)
fe = sum(r['full_exact'] for r in results)
ve = sum(r['var_exact'] for r in results)
fr2m = float(np.mean([max(r['full_r2'], -1) if np.isfinite(r['full_r2']) else -1 for r in results]))
vr2m = float(np.mean([max(r['var_r2'], -1) if np.isfinite(r['var_r2']) else -1 for r in results]))
ftm = float(np.mean([r['full_t'] for r in results]))
vtm = float(np.mean([r['var_t'] for r in results]))
print("="*80)
print(f"PSE backend, n={n} tasks:")
print(f"  Full:       exact {fe}/{n} ({100*fe/n:.0f}%)  R2 {fr2m:.3f}  mean time {ftm:.1f}s")
print(f"  DenoisedSR: exact {ve}/{n} ({100*ve/n:.0f}%)  R2 {vr2m:.3f}  mean time {vtm:.1f}s")
print(f"  Speedup:    {ftm/max(vtm,0.01):.2f}x")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {OUT_PATH}")
