"""
Noise robustness of variable-support recall.
DenoisedSR vs. best classical baseline (Lasso CV) on AI-Feynman, q=100,
N_DIST=20, y perturbed with Gaussian noise eta * std(y_clean) for
eta in {0, 0.01, 0.05, 0.10}.
"""
import sys, json, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, sympy as sp, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from sklearn.linear_model import LassoCV
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

SEED = int(os.environ.get("SEED","42"))
Q    = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
OUT_PATH = os.environ.get("OUT_PATH","data/results/noise_sweep.json")
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

tasks_raw = [json.loads(l) for l in
             open("data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl",encoding="utf-8").read().splitlines() if l.strip()]
tasks_raw = [t for t in tasks_raw if len(t["features"])>=2]
print(f"Tasks: {len(tasks_raw)}; q={Q}, n_dist={N_DIST}\n")

def sample(task, n, rng):
    formula = task["formula"]; features = task["features"]; ranges = task.get("ranges",{})
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

def denoisedsr_select(aug, target):
    cols = [c for c in aug if c!=target]
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
            for j in range(i+1,len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst = zip(*edges)
    ei = torch.tensor([list(src),list(dst)],dtype=torch.long)
    ei,_ = add_self_loops(ei,num_nodes=len(cols))
    with torch.no_grad():
        gat_s = torch.sigmoid(gat(torch.tensor(feats),ei)).numpy()
    return {c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs>=0.10 and not c.startswith("__d")}, cols

def _topk(scores, k):
    if k>=len(scores): return set(range(len(scores)))
    return set(np.argsort(-scores)[:k].tolist())

def lasso_select_topk(X, y, k):
    try:
        Xs = (X - X.mean(0)) / (X.std(0)+1e-12)
        ys = (y - y.mean()) / (y.std()+1e-12)
        m = LassoCV(cv=5, random_state=0, max_iter=2000, n_alphas=20).fit(Xs, ys)
        sc = np.abs(m.coef_)
    except:
        sc = np.zeros(X.shape[1])
    return _topk(sc, k)

def pearson_select_topk(X, y, k):
    from numpy import corrcoef
    s = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if np.std(X[:,j])<1e-12: continue
        s[j] = abs(corrcoef(X[:,j], y)[0,1])
    return _topk(np.nan_to_num(s), k)

def spearman_select_topk(X, y, k):
    from scipy.stats import spearmanr
    s = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        try: s[j] = abs(spearmanr(X[:,j], y).statistic)
        except: pass
    return _topk(np.nan_to_num(s), k)

def mi_select_topk(X, y, k):
    from sklearn.feature_selection import mutual_info_regression
    try: s = mutual_info_regression(X, y, random_state=0)
    except: s = np.zeros(X.shape[1])
    return _topk(s, k)

def rf_select_topk(X, y, k):
    from sklearn.ensemble import RandomForestRegressor
    try:
        m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1).fit(X, y)
        s = m.feature_importances_
    except: s = np.zeros(X.shape[1])
    return _topk(s, k)

BASELINE_SELECTORS = {
    "pearson":  pearson_select_topk,
    "spearman": spearman_select_topk,
    "mi":       mi_select_topk,
    "rf":       rf_select_topk,
    "lasso":    lasso_select_topk,
}

ETAS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]
results = {f"eta_{e}": {"denoisedsr": [], **{b: [] for b in BASELINE_SELECTORS}} for e in ETAS}
t0 = time.time()
for idx, task in enumerate(tasks_raw):
    features = task["features"]
    rng_base = np.random.default_rng(SEED + idx)
    aug0, target = sample(task, Q, rng_base)
    if aug0 is None: continue
    for j in range(N_DIST):
        lo,hi = float(rng_base.uniform(-5,0)), float(rng_base.uniform(0,5))
        aug0[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng_base)
    y_clean = np.array(aug0[target], dtype=float).copy()
    y_std = float(y_clean.std()) if y_clean.std()>0 else 1.0
    cols_all = [c for c in aug0 if c!=target]
    Xm = np.column_stack([aug0[c] for c in cols_all]).astype(float)
    true_set = set(features)
    for eta in ETAS:
        aug = {k:v for k,v in aug0.items()}
        noise_rng = np.random.default_rng(SEED*1000 + idx + int(eta*10000))
        aug[target] = y_clean + eta * y_std * noise_rng.standard_normal(Q)
        y_noisy = aug[target]
        # DenoisedSR
        try:
            sel, _cols = denoisedsr_select(aug, target)
            tp=len(sel&true_set); fn=len(true_set-sel)
            rec_d = tp/(tp+fn) if (tp+fn) else 0
        except Exception:
            rec_d = float('nan')
        results[f"eta_{eta}"]["denoisedsr"].append(rec_d)
        # All 5 baselines (each at oracle top-k)
        k = len(features)
        for bname, bfn in BASELINE_SELECTORS.items():
            try:
                sel_idx = bfn(Xm, y_noisy, k)
                sel_names = {cols_all[i] for i in sel_idx}
                tp=len(sel_names&true_set); fn=len(true_set-sel_names)
                rec_b = tp/(tp+fn) if (tp+fn) else 0
            except Exception:
                rec_b = float('nan')
            results[f"eta_{eta}"][bname].append(rec_b)
    if (idx+1) % 30 == 0:
        print(f"  [{idx+1}/{len(tasks_raw)}] elapsed {time.time()-t0:.0f}s")

def _stats(arr):
    a = np.array([x for x in arr if not np.isnan(x)])
    return {"mean": float(a.mean()), "perfect": float((a>=0.999).mean()), "n": int(len(a))}

methods = ["denoisedsr"] + list(BASELINE_SELECTORS.keys())
print(f"\n{'eta':>5} | " + " | ".join(f"{m[:10]:>10}" for m in methods))
print("-" * (8 + 13*len(methods)))
summary = {}
for eta in ETAS:
    row = {m: _stats(results[f"eta_{eta}"][m]) for m in methods}
    summary[f"eta_{eta}"] = row
    print(f"{eta:>5.3f} | " + " | ".join(f"{row[m]['mean']:>10.3f}" for m in methods))
print(f"{'pft%':>5} | " + " | ".join(f"{100*row[m]['perfect']:>9.0f}%" for m in methods))

# back-compat keys for existing fig_noise (reads denoisedsr_recall, lasso_recall, *_perfect)
for eta in ETAS:
    summary[f"eta_{eta}"]["denoisedsr_recall"] = summary[f"eta_{eta}"]["denoisedsr"]["mean"]
    summary[f"eta_{eta}"]["denoisedsr_perfect"] = summary[f"eta_{eta}"]["denoisedsr"]["perfect"]
    summary[f"eta_{eta}"]["lasso_recall"] = summary[f"eta_{eta}"]["lasso"]["mean"]
    summary[f"eta_{eta}"]["lasso_perfect"] = summary[f"eta_{eta}"]["lasso"]["perfect"]

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({"config":{"seed":SEED,"q":Q,"n_dist":N_DIST,"etas":ETAS},
                                       "summary": summary,
                                       "per_task": results}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
