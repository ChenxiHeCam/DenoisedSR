"""PSE smoke test: full vs DenoisedSR-pruned on 1 Feynman task."""
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train"))

import numpy as np, sympy as sp, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

PSE_REPO = "D:/Physics Fundation model/third_party/PSE"
DR_MASK = f"{PSE_REPO}/dr_mask"
STAGE_CFG = f"{PSE_REPO}/model/stages_config/benchmark.yaml"
TG_CFG = f"{PSE_REPO}/token_generator_config.yaml"

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
    feats = np.array([v2.column_features(np.asarray(aug[c]), np.asarray(aug[target]), var_name=c)
                      for c in cols], dtype=np.float32)
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
    return [c for c,r,g in zip(cols,rf_s,gat_s) if r+g >= 0.10]

# Sample 1 Feynman task: I_12_5 (qvB/p type, simple)
tasks = [json.loads(l) for l in open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
task = [t for t in tasks if t['law_id'] == 'pmlb_feynman_I_12_5'][0]
print(f"Task: {task['law_id']} -- {task['formula']}")

# Sample 100 obs + 20 distractors
rng = np.random.default_rng(42)
features = task["features"]
formula = task["formula"]; lhs, rhs = formula.split("=", 1); target = lhs.strip()
syms = {f: sp.Symbol(f) for f in features}
fn = sp.lambdify([syms[f] for f in features], sp.sympify(rhs.strip(), locals=syms), "numpy")
ranges = task.get("ranges", {})
cols = {}
for f in features:
    lo,hi = ranges.get(f,[1.0,5.0]); cols[f] = rng.uniform(float(lo), float(hi), 100)
y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
cols[target] = y
for j in range(20):
    lo,hi = rng.uniform(-5,0), rng.uniform(0,5); cols[f"__d{j}"] = v2.resample(np.array([lo,hi]), 100, rng)

sel = predict_vars(cols, target)
print(f"DenoisedSR selected: {sel} (true: {features})")

# Try PSE
from psrn import PSRN_Regressor
OPS = ['Add', 'Mul', 'Sub', 'Div', 'Identity', 'Sin', 'Cos', 'Exp', 'Log']  # matches 3_4 mask
STAGE_DICT = {
    "default": {"operators": OPS, "time_limit": 60, "n_psrn_inputs": 4, "n_sample_variables": 4},
    "stages": [{"time_limit": 60, "n_psrn_inputs": 4}],
}
TG_DICT = {"base": {"has_const": False, "tokens": OPS}}
print("\n=== Running PSE on FULL X (all 22 columns) ===")
all_col_names = [c for c in cols if c != target]
X_full = np.column_stack([cols[c] for c in all_col_names])
y_arr = y.reshape(-1, 1)
print(f"X shape: {X_full.shape}, X cols: {all_col_names[:5]}...")

# PSE uses simple x0,x1,... internally; we set variables=['x0','x1',...]
var_names_full = [f"x{i}" for i in range(X_full.shape[1])]
t0 = time.time()
try:
    reg = PSRN_Regressor(
        variables=var_names_full, n_inputs=min(X_full.shape[1], 5),
        n_symbol_layers=3, use_const=False, use_extra_const=True,
        dr_mask_dir=DR_MASK, stage_config=STAGE_DICT, token_generator_config=TG_DICT,
        device='cuda',
    )
    reg.fit(X_full, y_arr, n_down_sample=20, real_time_display=False)
    pf = reg.get_pf(sort_by="mse")
    print(f"FULL  -> best expr: {pf[0][0]}  MSE: {pf[0][2]:.3e}  time: {time.time()-t0:.1f}s")
except Exception as e:
    print(f"FULL  -> ERR {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

print("\n=== Running PSE on REDUCED X (DenoisedSR-selected only) ===")
sel_arr = [all_col_names.index(c) for c in sel if c in all_col_names]
X_red = X_full[:, sel_arr] if sel_arr else X_full
var_names_red = [f"x{i}" for i in range(X_red.shape[1])]
t0 = time.time()
try:
    reg2 = PSRN_Regressor(
        variables=var_names_red, n_inputs=X_red.shape[1],
        n_symbol_layers=3, use_const=False, use_extra_const=True,
        dr_mask_dir=DR_MASK, stage_config=STAGE_DICT, token_generator_config=TG_DICT,
        device='cuda',
    )
    reg2.fit(X_red, y_arr, n_down_sample=20, real_time_display=False)
    pf = reg2.get_pf(sort_by="mse")
    print(f"REDUCED -> best expr: {pf[0][0]}  MSE: {pf[0][2]:.3e}  time: {time.time()-t0:.1f}s")
except Exception as e:
    print(f"REDUCED -> ERR {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
