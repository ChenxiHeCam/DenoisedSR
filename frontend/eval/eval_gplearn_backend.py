"""
Second backend: gplearn (classical GP). Test the "solver-agnostic" claim.
Same 30 Feynman headline subset, same n_dist=20, same q=100. We don't expect
gplearn to match PySR's recovery rate — it's the comparison condition (with
vs without DenoisedSR prior on the SAME backend) that matters.
"""
import sys, json, os, time, warnings
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None

import numpy as np, sympy as sp, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from gplearn.genetic import SymbolicRegressor
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

SEED = int(os.environ.get("SEED","42"))
Q    = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
N_TASKS = int(os.environ.get("N_TASKS","30"))
GP_GENS = int(os.environ.get("GP_GENS","20"))      # keep small for fairness; PySR also short
POP    = int(os.environ.get("POP","500"))
OUT_PATH = os.environ.get("OUT_PATH","data/results/gplearn_backend_3way.json")
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
tasks_raw = [t for t in tasks_raw if 2 <= len(t["features"]) <= 9][:N_TASKS]
print(f"gplearn backend test: {len(tasks_raw)} formulas, q={Q}, n_dist={N_DIST}, gens={GP_GENS}, pop={POP}\n")

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
    sel = [c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs>=0.10 and not c.startswith("__d")]
    if not sel: sel = [cols[int(np.argmax(rf_s))]]
    return sel

def r2(yt, yp):
    ss=np.sum((yt-yp)**2); tot=np.sum((yt-np.mean(yt))**2); return float(1-ss/(tot+1e-12))

GP_FN = ('add','sub','mul','div','sqrt','log','sin','cos','tan','abs')

def run_gp(X_tr, y_tr, X_te, y_te, seed):
    m = SymbolicRegressor(population_size=POP, generations=GP_GENS, function_set=GP_FN,
                          random_state=seed, verbose=0, n_jobs=1,
                          parsimony_coefficient=0.001, max_samples=0.9,
                          stopping_criteria=1e-8)
    t0 = time.time()
    try:
        m.fit(X_tr, y_tr); el = time.time()-t0
        sc = r2(y_te, m.predict(X_te))
        return sc, str(m._program), el
    except Exception as e:
        return -99, f"ERR: {e}", time.time()-t0

results = []
t0g = time.time()
for i, task in enumerate(tasks_raw):
    features = task["features"]
    rng = np.random.default_rng(SEED + i)
    aug, target = sample(task, Q, rng); te, _ = sample(task, 200, np.random.default_rng(SEED + i + 99999))
    if aug is None or te is None: continue
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng); te[f"__d{j}"] = v2.resample(np.array([lo,hi]), 200, rng)
    full_cols = [c for c in aug if c!=target]
    sel = denoisedsr_select(aug, target)
    y_tr = aug[target]; y_te = te[target]
    Xf_tr = np.column_stack([aug[c] for c in full_cols]); Xf_te = np.column_stack([te[c] for c in full_cols])
    Xo_tr = np.column_stack([aug[c] for c in sel]);       Xo_te = np.column_stack([te[c] for c in sel])
    fr2, fexp, ft = run_gp(Xf_tr, y_tr, Xf_te, y_te, SEED+i)
    vr2, vexp, vt = run_gp(Xo_tr, y_tr, Xo_te, y_te, SEED+i)
    fe = fr2>=0.9999; ve = vr2>=0.9999
    results.append({"law_id":task["law_id"],"nvars":len(features),
                    "full_cols":len(full_cols),"sel_cols":len(sel),
                    "full_r2":fr2,"var_r2":vr2,
                    "full_exact":fe,"var_exact":ve,
                    "full_t":ft,"var_t":vt,
                    "full_expr":fexp,"var_expr":vexp,
                    "true_vars":sorted(features),"sel_vars":sorted(sel),
                    "recall": len(set(sel)&set(features))/len(features)})
    print(f"{i+1:3d} {task['law_id'][:24]:24s} fR2={fr2:7.3f} vR2={vr2:7.3f}  "
          f"{'F' if fe else '-'}  {'V' if ve else '-'}")

n=len(results)
fe=sum(r['full_exact'] for r in results); ve=sum(r['var_exact'] for r in results)
print(f"\n=== gplearn ({GP_GENS} gens, pop={POP}, {N_DIST} distractors), n={n}, total {time.time()-t0g:.0f}s ===")
print(f"Full       : exact {fe}/{n} ({100*fe/n:.0f}%) mean R2 = {np.mean([r['full_r2'] for r in results if r['full_r2']>-50]):.3f}")
print(f"DenoisedSR : exact {ve}/{n} ({100*ve/n:.0f}%) mean R2 = {np.mean([r['var_r2'] for r in results if r['var_r2']>-50]):.3f}")
print(f"COL: {np.mean([r['full_cols'] for r in results]):.0f} -> {np.mean([r['sel_cols'] for r in results]):.1f}")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"Saved -> {OUT_PATH}")
