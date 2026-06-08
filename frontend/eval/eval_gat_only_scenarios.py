"""
GAT-alone in the same hard scenarios.

Conditions per scenario:
  GAT-only named       : keep iff sigmoid(GAT(col)) >= tau, with COOC edges
  GAT-only anonymized  : same, but columns renamed x0,x1,...
"""
import sys, json, os, warnings
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

SEED = 42
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
gat = GATDisc(); gat.load_state_dict(torch.load("models/gat_best.pt", map_location="cpu")["model"]); gat.eval()

tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks_raw = [t for t in tasks_raw if len(t["features"]) >= 2]

def sample(task, n, rng):
    formula = task["formula"]; features = task["features"]; ranges = task.get("ranges", {})
    if "=" not in formula: return None, None
    lhs, rhs = formula.split("=",1); target = lhs.strip()
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

def gat_only_score(aug, target):
    cols = [c for c in aug if c != target]
    feats = np.array([v2.column_features(np.asarray(aug[c]), np.asarray(aug[target]), var_name=c)
                      for c in cols], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0., posinf=10., neginf=-10.)
    edges = set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges:
        return cols, np.zeros(len(cols)), False
    src,dst = zip(*edges)
    ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
    ei,_ = add_self_loops(ei, num_nodes=len(cols))
    with torch.no_grad():
        gat_s = torch.sigmoid(gat(torch.tensor(feats), ei)).numpy()
    return cols, gat_s, True

def anonymize(cols_dict, features):
    name_map = {f: f"x{i}" for i, f in enumerate(features)}
    new = {}
    for k, v in cols_dict.items():
        new[name_map.get(k, k)] = v
    return new, [name_map[f] for f in features]

def run_one(label, q, n_dist, eta, tau):
    nm_recs, nm_frs, nm_gat_fired = [], [], 0
    an_recs, an_frs, an_gat_fired = [], [], 0
    for idx, task in enumerate(tasks_raw):
        features = task["features"]
        rng = np.random.default_rng(SEED + idx)
        aug, target = sample(task, q, rng)
        if aug is None: continue
        for j in range(n_dist):
            lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
            aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), q, rng)
        if eta > 0:
            y = aug[target].copy()
            y_std = float(np.std(y)) if np.std(y)>0 else 1.0
            aug[target] = y + eta*y_std*rng.standard_normal(q)
        # NAMED GAT-only
        try:
            cols_n, gat_n, fired_n = gat_only_score(aug, target)
        except Exception: continue
        true_set_n = set(features)
        sel_n = {c for c,g in zip(cols_n, gat_n) if g >= tau}
        tp = len(sel_n & true_set_n); fn = len(true_set_n - sel_n)
        rec = tp/(tp+fn) if (tp+fn) else 0
        dk = sum(1 for c in sel_n if c.startswith("__d"))
        nm_recs.append(rec); nm_frs.append((n_dist-dk)/n_dist)
        if fired_n: nm_gat_fired += 1
        # ANONYMIZED GAT-only
        aug_a, features_a = anonymize(aug, features)
        try:
            cols_a, gat_a, fired_a = gat_only_score(aug_a, target)
        except Exception: continue
        true_set_a = set(features_a)
        sel_a = {c for c,g in zip(cols_a, gat_a) if g >= tau}
        tp = len(sel_a & true_set_a); fn = len(true_set_a - sel_a)
        rec = tp/(tp+fn) if (tp+fn) else 0
        dk = sum(1 for c in sel_a if c.startswith("__d"))
        an_recs.append(rec); an_frs.append((n_dist-dk)/n_dist)
        if fired_a: an_gat_fired += 1
    n = len(nm_recs)
    return {
        "label":label,"q":q,"n_dist":n_dist,"eta":eta,"tau":tau,"n_tasks":n,
        "NM_GAT":{"recall":float(np.mean(nm_recs)),"filter":float(np.mean(nm_frs)),
                  "perfect":float(np.mean(np.array(nm_recs)>=0.999)),"fired":nm_gat_fired},
        "AN_GAT":{"recall":float(np.mean(an_recs)),"filter":float(np.mean(an_frs)),
                  "perfect":float(np.mean(np.array(an_recs)>=0.999)),"fired":an_gat_fired},
    }

scenarios = [
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

print(f"{'scenario':>22s} | tau   | {'GAT-only NAMED rec/filt fire':^32s} | {'GAT-only ANON rec/filt fire':^32s} | named-anon delta")
print("-"*135)
out = []
for s in scenarios:
    r = run_one(*s)
    nm = r['NM_GAT']; an = r['AN_GAT']
    delta = 100*(nm['filter']-an['filter'])
    print(f"{r['label']:>22s} | {r['tau']:>4.2f} | {nm['recall']:.3f}/{100*nm['filter']:>5.1f}%  {nm['fired']:>3}/{r['n_tasks']} | "
          f"{an['recall']:.3f}/{100*an['filter']:>5.1f}%  {an['fired']:>3}/{r['n_tasks']} | {delta:+5.1f} pp")
    out.append(r)

Path("data/results/gat_only_scenarios.json").write_text(json.dumps({"scenarios": out}, indent=2))
print("\nSaved -> data/results/gat_only_scenarios.json")
