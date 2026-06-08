"""
Variable-support recall on additional external SR benchmarks:

  Strogatz: 15 ODE-derived bivariate dynamics formulas
            (Bacterial growth, Bar Magnets, Glider, Lotka-Volterra,
             Predator-Prey, Shear Flow, Van der Pol, Lorenz).
            Source: SRBench / Strogatz "Nonlinear Dynamics and Chaos"
            textbook problem sets (Cranmer et al. 2020 / La Cava et al. 2021).

  Nguyen:   8 classical polynomial / trigonometric / logarithmic SR
            benchmarks. Source: Uy et al. 2011 (most-used baseline set
            in the SR literature).

Same protocol as AI-Feynman: q=100, 20 random distractor columns,
deployed RF+GAT ensemble at tau=0.10. Each suite is reported separately.
DenoisedSR was NEVER trained on any of these.
"""
import sys, json, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, sympy as sp, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

SEED = int(os.environ.get("SEED","42"))
Q    = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
OUT_PATH = os.environ.get("OUT_PATH","data/results/external_benchmarks_recall.json")
FEAT=74; HID=128; HEADS=4

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

# ---- Benchmarks: (suite, id, formula_str, vars, ranges) ----
# Strogatz: take the RHS of each ODE as y = f(state); samples are over
# physically reasonable state-space ranges.
STROGATZ = [
    ("strogatz_bacres1",   "20 - x0 - x0*x1 / (1 + 0.5*x0**2)", ["x0","x1"], {"x0":[0.5,5],"x1":[0.5,5]}),
    ("strogatz_bacres2",   "10 - x0*x1 / (1 + 0.5*x0**2)",      ["x0","x1"], {"x0":[0.5,5],"x1":[0.5,5]}),
    ("strogatz_barmag1",   "0.5*sin(x0 - x1) - sin(x1)",        ["x0","x1"], {"x0":[-3,3],"x1":[-3,3]}),
    ("strogatz_barmag2",   "0.5*sin(x1 - x0) - sin(x0)",        ["x0","x1"], {"x0":[-3,3],"x1":[-3,3]}),
    ("strogatz_glider1",   "-0.05*x0**2 - sin(x1)",             ["x0","x1"], {"x0":[0.1,5],"x1":[-3,3]}),
    ("strogatz_glider2",   "x0 - cos(x1)/x0",                   ["x0","x1"], {"x0":[0.5,5],"x1":[-3,3]}),
    ("strogatz_lv1",       "3*x0 - 2*x0*x1 - x0**2",            ["x0","x1"], {"x0":[0.1,3],"x1":[0.1,3]}),
    ("strogatz_lv2",       "2*x1 - x0*x1 - x1**2",              ["x0","x1"], {"x0":[0.1,3],"x1":[0.1,3]}),
    ("strogatz_predprey1", "x0*(4 - x0 - x1/(1+x0))",           ["x0","x1"], {"x0":[0.1,5],"x1":[0.1,5]}),
    ("strogatz_predprey2", "x1*(x0/(1+x0) - 0.075*x1)",         ["x0","x1"], {"x0":[0.1,5],"x1":[0.1,5]}),
    ("strogatz_shearflow1","cos(x0)/tan(x1)",                   ["x0","x1"], {"x0":[-3,3],"x1":[0.2,3]}),
    ("strogatz_shearflow2","(cos(x1)**2 + 0.1*sin(x1)**2)*sin(x0)", ["x0","x1"], {"x0":[-3,3],"x1":[-3,3]}),
    ("strogatz_vdp1",      "10*(x1 - (1/3)*(x0**3 - x0))",      ["x0","x1"], {"x0":[-3,3],"x1":[-3,3]}),
    ("strogatz_vdp2",      "-0.1*x0",                            ["x0"],      {"x0":[-3,3]}),
    ("strogatz_lorenz1",   "10*(x1 - x0)",                       ["x0","x1"], {"x0":[-10,10],"x1":[-10,10]}),
]
# Nguyen: classical SR benchmarks (Uy et al. 2011). Filter to >=1 true var
# (single-variable formulas are valid recall tests against distractors).
NGUYEN = [
    ("nguyen_1",  "x0**3 + x0**2 + x0",                              ["x0"],      {"x0":[-1,1]}),
    ("nguyen_2",  "x0**4 + x0**3 + x0**2 + x0",                      ["x0"],      {"x0":[-1,1]}),
    ("nguyen_3",  "x0**5 + x0**4 + x0**3 + x0**2 + x0",              ["x0"],      {"x0":[-1,1]}),
    ("nguyen_4",  "x0**6 + x0**5 + x0**4 + x0**3 + x0**2 + x0",      ["x0"],      {"x0":[-1,1]}),
    ("nguyen_5",  "sin(x0**2)*cos(x0) - 1",                          ["x0"],      {"x0":[-1,1]}),
    ("nguyen_6",  "sin(x0) + sin(x0 + x0**2)",                       ["x0"],      {"x0":[-1,1]}),
    ("nguyen_7",  "log(x0+1) + log(x0**2+1)",                        ["x0"],      {"x0":[0.1,2]}),
    ("nguyen_8",  "sqrt(x0)",                                         ["x0"],      {"x0":[0.1,4]}),
    ("nguyen_9",  "sin(x0) + sin(x1**2)",                            ["x0","x1"], {"x0":[-1,1],"x1":[-1,1]}),
    ("nguyen_10", "2*sin(x0)*cos(x1)",                               ["x0","x1"], {"x0":[-1,1],"x1":[-1,1]}),
    ("nguyen_11", "x0**x1",                                          ["x0","x1"], {"x0":[0.1,2],"x1":[0.1,2]}),
    ("nguyen_12", "x0**4 - x0**3 + (x1**2)/2 - x1",                  ["x0","x1"], {"x0":[-1,1],"x1":[-1,1]}),
]

SUITES = {"Strogatz": STROGATZ, "Nguyen": NGUYEN}

def sample_task(formula_str, features, ranges, q, rng):
    try:
        syms = {f: sp.Symbol(f) for f in features}
        fn = sp.lambdify([syms[f] for f in features], sp.sympify(formula_str, locals=syms), "numpy")
    except Exception as e:
        return None, None
    cols = {}
    for f in features:
        rg = ranges.get(f, [-1, 1]); lo, hi = float(rg[0]), float(rg[1])
        if lo >= hi: hi = lo + 1
        cols[f] = rng.uniform(lo, hi, q)
    try:
        y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except Exception:
        return None, None
    if not np.all(np.isfinite(y)): return None, None
    cols["__y"] = y
    return cols, "__y"

def denoisedsr_select(aug, target):
    # Honest precision: do NOT filter __d columns by name. The model has to
    # decide on its own. Returns the model's raw selection at tau=0.10.
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
    return {c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs >= 0.10}

summary = {}
per_task = []
for suite_name, tasks in SUITES.items():
    recs, precs, perfect = [], [], 0
    n_used = 0
    for i, (task_id, formula, features, ranges) in enumerate(tasks):
        rng = np.random.default_rng(SEED + i + hash(task_id) % 10000)
        aug, target = sample_task(formula, features, ranges, Q, rng)
        if aug is None:
            print(f"  SKIP {task_id}: sampling failed")
            continue
        for j in range(N_DIST):
            lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
            aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
        true_set = set(features)
        try:
            sel = denoisedsr_select(aug, target)
        except Exception as e:
            print(f"  ERR {task_id}: {e}")
            continue
        tp = len(sel & true_set); fp = len(sel - true_set); fn = len(true_set - sel)
        rec = tp/(tp+fn) if (tp+fn) else 0
        prec = tp/(tp+fp) if (tp+fp) else (1.0 if tp else 0)
        recs.append(rec); precs.append(prec)
        if rec >= 0.999: perfect += 1
        n_used += 1
        per_task.append({"suite": suite_name, "id": task_id, "formula": formula,
                         "true_vars": features, "selected": sorted(sel),
                         "recall": rec, "precision": prec, "perfect": rec >= 0.999})
    summary[suite_name] = {
        "n": n_used,
        "recall_mean": float(np.mean(recs)),
        "precision_mean": float(np.mean(precs)),
        "perfect_recall": int(perfect),
        "perfect_recall_frac": float(perfect/n_used) if n_used else 0,
    }
    s = summary[suite_name]
    print(f"{suite_name:>10s} n={s['n']:>2}: recall={s['recall_mean']:.3f}  precision={s['precision_mean']:.3f}  "
          f"perfect {s['perfect_recall']}/{s['n']} ({100*s['perfect_recall_frac']:.1f}%)")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({
    "config":{"seed":SEED,"q":Q,"n_dist":N_DIST,"threshold":0.10},
    "summary": summary,
    "per_task": per_task
}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
