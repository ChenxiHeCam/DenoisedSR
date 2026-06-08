"""
Controlled ablation: WITH physical names vs WITHOUT (anonymized).

Same observation data (same X, y, same distractors), only variable
NAMES are changed. Measures the marginal contribution of the
physical-name prior (which drives the GAT co-occurrence-graph edges
in the deployed ensemble).

Note on "units": in our system units come in through the name (we
look up variable name in the co-occurrence graph; column magnitudes
themselves are scale-normalized in the per-column features). So
"with names" and "with units" are coupled here.
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

SEED   = int(os.environ.get("SEED","42"))
Q      = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
OUT_PATH = os.environ.get("OUT_PATH", "data/results/name_ablation.json")
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
gat = GATDisc(); gat.load_state_dict(torch.load("models/gat_best.pt", map_location="cpu")["model"]); gat.eval()

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
        rg = ranges.get(f,[1.0,5.0]); lo,hi = float(rg[0]), float(rg[1])
        if lo>=hi: hi = lo+1
        cols[f] = rng.uniform(lo,hi,n)
    try:
        y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except: return None, None
    if not np.all(np.isfinite(y)): return None, None
    cols[target] = y
    return cols, target

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
        gat_s = np.zeros(len(cols))
    else:
        src,dst = zip(*edges)
        ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
        ei,_ = add_self_loops(ei, num_nodes=len(cols))
        with torch.no_grad():
            gat_s = torch.sigmoid(gat(torch.tensor(feats), ei)).numpy()
    sel = {c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs >= 0.10}
    return sel, rf_s, gat_s, cols

def anonymize(aug, target, features):
    """Replace physical names with x0, x1, ... keeping target as __y."""
    new_aug = {}
    # map: physical_name -> x_i
    name_map = {f: f"x{i}" for i, f in enumerate(features)}
    new_features = [name_map[f] for f in features]
    for f in features:
        new_aug[name_map[f]] = aug[f]
    new_aug["__y"] = aug[target]
    # carry over distractors as-is
    for k, v in aug.items():
        if k.startswith("__d"):
            new_aug[k] = v
    return new_aug, "__y", new_features

results = {"named": [], "anonymized": []}
gat_fired = {"named": 0, "anonymized": 0}
for idx, task in enumerate(tasks_raw):
    features = task["features"]
    rng = np.random.default_rng(SEED + idx)
    aug, target = sample(task, Q, rng)
    if aug is None: continue
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)

    # --- Named (physical) ---
    sel_n, rf_n, gat_n, cols_n = predict(aug, target)
    true_set_n = set(features)
    tp_n = len(sel_n & true_set_n); fp_n = len(sel_n - true_set_n); fn_n = len(true_set_n - sel_n)
    rec_n = tp_n/(tp_n+fn_n) if (tp_n+fn_n) else 0
    prec_n = tp_n/(tp_n+fp_n) if (tp_n+fp_n) else (1.0 if tp_n else 0)
    dist_kept_n = sum(1 for c in sel_n if c.startswith("__d"))
    fr_n = (N_DIST - dist_kept_n)/N_DIST
    if np.max(gat_n) > 0:  # GAT actually fired
        gat_fired["named"] += 1
    results["named"].append({"law": task["law_id"], "recall": rec_n, "precision": prec_n,
                              "filter_rate": fr_n, "kept": len(sel_n)})

    # --- Anonymized ---
    aug_a, target_a, features_a = anonymize(aug, target, features)
    sel_a, rf_a, gat_a, cols_a = predict(aug_a, target_a)
    true_set_a = set(features_a)
    tp_a = len(sel_a & true_set_a); fp_a = len(sel_a - true_set_a); fn_a = len(true_set_a - sel_a)
    rec_a = tp_a/(tp_a+fn_a) if (tp_a+fn_a) else 0
    prec_a = tp_a/(tp_a+fp_a) if (tp_a+fp_a) else (1.0 if tp_a else 0)
    dist_kept_a = sum(1 for c in sel_a if c.startswith("__d"))
    fr_a = (N_DIST - dist_kept_a)/N_DIST
    if np.max(gat_a) > 0:
        gat_fired["anonymized"] += 1
    results["anonymized"].append({"law": task["law_id"], "recall": rec_a, "precision": prec_a,
                                    "filter_rate": fr_a, "kept": len(sel_a)})

n = len(results["named"])
def stats(rows):
    arr = np.array([[r["recall"], r["precision"], r["filter_rate"], r["kept"]] for r in rows])
    return arr.mean(axis=0), (arr[:,0] >= 0.999).mean()

(mn_r, mn_p, mn_fr, mn_k), perf_n = stats(results["named"])
(ma_r, ma_p, ma_fr, ma_k), perf_a = stats(results["anonymized"])

print(f"\n{'Setting':>14s} | {'recall':>7} {'precision':>9} {'filter%':>7} {'perfect%':>8} {'kept':>5} {'GAT-fired':>10}")
print("-"*80)
print(f"{'NAMED (physical)':>14s} | {mn_r:>7.3f} {mn_p:>9.3f} {100*mn_fr:>6.1f}% {100*perf_n:>7.1f}% {mn_k:>5.2f}   {gat_fired['named']:>3}/{n}")
print(f"{'ANONYMIZED':>14s} | {ma_r:>7.3f} {ma_p:>9.3f} {100*ma_fr:>6.1f}% {100*perf_a:>7.1f}% {ma_k:>5.2f}   {gat_fired['anonymized']:>3}/{n}")
print(f"\nDelta from physical-name prior:")
print(f"  filter_rate: {100*(mn_fr - ma_fr):+5.2f} pp")
print(f"  precision:   {(mn_p - ma_p):+5.3f}")
print(f"  perfect %:   {100*(perf_n - perf_a):+5.2f} pp")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({
    "config":{"seed":SEED,"q":Q,"n_dist":N_DIST,"n_tasks":n},
    "summary": {
        "named":      {"recall":float(mn_r),"precision":float(mn_p),"filter_rate":float(mn_fr),
                       "perfect_frac":float(perf_n),"mean_kept":float(mn_k),"gat_fired":gat_fired["named"]},
        "anonymized": {"recall":float(ma_r),"precision":float(ma_p),"filter_rate":float(ma_fr),
                       "perfect_frac":float(perf_a),"mean_kept":float(ma_k),"gat_fired":gat_fired["anonymized"]},
        "delta": {"filter_rate_pp": float(100*(mn_fr-ma_fr)),
                  "precision":      float(mn_p-ma_p),
                  "perfect_pp":     float(100*(perf_n-perf_a))}
    },
    "per_task": results
}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
