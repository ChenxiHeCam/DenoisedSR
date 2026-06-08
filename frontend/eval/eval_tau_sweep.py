"""
Threshold sensitivity sweep on AI-Feynman 118.
Show how precision/recall trade off as the deployed RF+GAT ensemble
threshold tau is varied. Reviewer answer: tau=0.10 is not cherry-picked.
"""
import sys, json, os, warnings, time
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

SEED = int(os.environ.get("SEED", "42"))
Q    = int(os.environ.get("Q", "100"))
N_DIST = int(os.environ.get("N_DIST", "20"))
OUT_PATH = os.environ.get("OUT_PATH", "data/results/tau_sweep.json")
FEAT=74; HID=128; HEADS=4
TAUS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

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

tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks_raw = [t for t in tasks_raw if len(t["features"]) >= 2]
print(f"Feynman tasks: {len(tasks_raw)}; q={Q}, n_dist={N_DIST}, seed={SEED}\n")

def sample(task, n, rng):
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
        if lo >= hi: hi = lo + 1
        cols[f] = rng.uniform(lo, hi, n)
    try:
        y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except: return None, None
    if not np.all(np.isfinite(y)): return None, None
    cols[target] = y
    return cols, target

def scores(aug, target):
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
    return cols, rf_s, gat_s

per_task_scores = []
t0 = time.time()
for idx, task in enumerate(tasks_raw):
    features = task["features"]
    rng = np.random.default_rng(SEED + idx)
    aug, target = sample(task, Q, rng)
    if aug is None: continue
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
    cols, rf_s, gat_s = scores(aug, target)
    per_task_scores.append({"law_id": task["law_id"], "cols": cols, "rf": rf_s.tolist(),
                            "gat": gat_s.tolist(), "true_vars": list(features)})
print(f"Score computation done ({time.time()-t0:.0f}s).\n")

# Sweep tau
print(f"{'tau':>6} | {'recall':>7} {'prec':>7} {'F1':>7} {'perfect%':>8} {'mean_kept':>9}")
print("-"*55)
sweep_summary = {}
for tau in TAUS:
    recs, precs, kept = [], [], []
    perfect = 0
    for t in per_task_scores:
        sel = {c for c, r, g in zip(t['cols'], t['rf'], t['gat']) if r+g >= tau}
        true_set = set(t['true_vars'])
        tp = len(sel & true_set); fp = len(sel - true_set); fn = len(true_set - sel)
        rec = tp/(tp+fn) if (tp+fn) else 0
        prec = tp/(tp+fp) if (tp+fp) else (1.0 if tp else 0)
        recs.append(rec); precs.append(prec); kept.append(len(sel))
        if rec >= 0.999: perfect += 1
    rmean, pmean = float(np.mean(recs)), float(np.mean(precs))
    f1 = 2*rmean*pmean/(rmean+pmean+1e-9)
    sweep_summary[f"tau_{tau}"] = {
        "tau": tau,
        "recall": rmean, "precision": pmean, "f1": f1,
        "perfect_recall_frac": perfect/len(per_task_scores),
        "mean_kept_cols": float(np.mean(kept)),
        "n_tasks": len(per_task_scores)
    }
    print(f"{tau:>6.2f} | {rmean:>7.3f} {pmean:>7.3f} {f1:>7.3f} {100*perfect/len(per_task_scores):>7.1f}%  {np.mean(kept):>9.2f}")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({"config":{"seed":SEED,"q":Q,"n_dist":N_DIST,"taus":TAUS},
                                       "summary": sweep_summary}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
