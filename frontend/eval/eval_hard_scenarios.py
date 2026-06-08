"""
Hard-scenario sweep: when does the GAT physical-name prior actually help?

Three conditions per scenario:
  RF-only          : keep iff RF(col) >= tau
  RF+GAT named     : deployed ensemble, physical variable names
  RF+GAT anonymized: ensemble, but columns renamed to x0, x1, ...

Sweep:
  small q in {20, 50, 100}
  noise eta in {0.0, 0.10, 0.30, 0.50}
  many distractors n_dist in {20, 50, 100}
  bumped tau in {0.10, 0.50, 1.00} (higher tau needs GAT to clear the bar)

Run on AI-Feynman 118.
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

SEED = int(os.environ.get("SEED","42"))
OUT_PATH = os.environ.get("OUT_PATH", "data/results/hard_scenarios.json")
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
print(f"Feynman tasks: {len(tasks_raw)}\n")

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

def score(aug, target, q_used):
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
    return cols, rf_s, gat_s

def anonymize_cols(cols_dict, features):
    name_map = {f: f"x{i}" for i, f in enumerate(features)}
    new = {}
    for k, v in cols_dict.items():
        if k in name_map:
            new[name_map[k]] = v
        else:
            new[k] = v
    new_features = [name_map[f] for f in features]
    return new, new_features

def evaluate(task_iter, label, q, n_dist, eta, tau):
    """Run all 3 conditions on each task. Return aggregated metrics."""
    rf_recs, rf_frs = [], []
    nm_recs, nm_frs = [], []
    an_recs, an_frs = [], []
    for idx, task in enumerate(task_iter):
        features = task["features"]
        rng = np.random.default_rng(SEED + idx)
        aug0, target = sample(task, q, rng)
        if aug0 is None: continue
        # add distractors
        for j in range(n_dist):
            lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
            aug0[f"__d{j}"] = v2.resample(np.array([lo,hi]), q, rng)
        # add noise to target
        if eta > 0:
            y = aug0[target].copy()
            y_std = float(np.std(y)) if np.std(y)>0 else 1.0
            aug0[target] = y + eta * y_std * rng.standard_normal(q)
        # NAMED ensemble
        try:
            cols_n, rf_n, gat_n = score(aug0, target, q)
        except Exception:
            continue
        true_set_n = set(features)
        # RF-only on the named call (same RF scores)
        sel_rf = {c for c,r in zip(cols_n, rf_n) if r >= tau}
        # RF+GAT named
        sel_nm = {c for c,r,g in zip(cols_n, rf_n, gat_n) if r+g >= tau}
        # Anonymized
        aug_a, features_a = anonymize_cols(aug0, features)
        try:
            cols_a, rf_a, gat_a = score(aug_a, target, q)
        except Exception:
            continue
        true_set_a = set(features_a)
        sel_an = {c for c,r,g in zip(cols_a, rf_a, gat_a) if r+g >= tau}
        # metrics per condition
        for sel, true_set, recs, frs in [(sel_rf, true_set_n, rf_recs, rf_frs),
                                          (sel_nm, true_set_n, nm_recs, nm_frs),
                                          (sel_an, true_set_a, an_recs, an_frs)]:
            tp = len(sel & true_set); fn = len(true_set - sel)
            rec = tp/(tp+fn) if (tp+fn) else 0
            dist_kept = sum(1 for c in sel if c.startswith("__d"))
            fr = (n_dist - dist_kept)/n_dist
            recs.append(rec); frs.append(fr)
    n = len(rf_recs)
    return {
        "label": label, "n_tasks": n, "q": q, "n_dist": n_dist, "eta": eta, "tau": tau,
        "RF":   {"recall": float(np.mean(rf_recs)),  "filter_rate": float(np.mean(rf_frs)),
                 "perfect": float(np.mean(np.array(rf_recs) >= 0.999))},
        "NM":   {"recall": float(np.mean(nm_recs)),  "filter_rate": float(np.mean(nm_frs)),
                 "perfect": float(np.mean(np.array(nm_recs) >= 0.999))},
        "AN":   {"recall": float(np.mean(an_recs)),  "filter_rate": float(np.mean(an_frs)),
                 "perfect": float(np.mean(np.array(an_recs) >= 0.999))},
    }

scenarios = [
    # (label, q, n_dist, eta, tau)
    ("baseline_q100",       100, 20, 0.0,  0.10),
    ("small_q50",            50, 20, 0.0,  0.10),
    ("very_small_q20",       20, 20, 0.0,  0.10),
    ("noise_eta0.30",       100, 20, 0.30, 0.10),
    ("noise_eta0.50",       100, 20, 0.50, 0.10),
    ("noise_eta1.00",       100, 20, 1.00, 0.10),
    ("many_distract_50",    100, 50, 0.0,  0.10),
    ("many_distract_100",   100,100, 0.0,  0.10),
    ("bump_tau0.50",        100, 20, 0.0,  0.50),
    ("bump_tau1.00",        100, 20, 0.0,  1.00),
    ("hard_q50_eta0.30",     50, 20, 0.30, 0.10),
    ("hard_q20_eta0.50",     20, 20, 0.50, 0.10),
]

print(f"{'scenario':>22s} | q   d   eta  tau  | {'RF rec/filt':^14s} | {'NM (named) rec/filt':^20s} | {'AN (anon) rec/filt':^20s} | GAT helps?")
print("-"*140)
all_results = []
for s in scenarios:
    r = evaluate(tasks_raw, *s)
    delta = 100*(r['NM']['filter_rate'] - r['AN']['filter_rate'])
    print(f"{r['label']:>22s} | {r['q']:>3} {r['n_dist']:>3} {r['eta']:>4.2f} {r['tau']:>4.2f} | "
          f"{r['RF']['recall']:.3f}/{100*r['RF']['filter_rate']:>4.1f}% | "
          f"{r['NM']['recall']:.3f}/{100*r['NM']['filter_rate']:>4.1f}%  | "
          f"{r['AN']['recall']:.3f}/{100*r['AN']['filter_rate']:>4.1f}%  | {delta:+5.1f} pp")
    all_results.append(r)

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({"scenarios": all_results}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
