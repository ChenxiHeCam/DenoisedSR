"""Unified 3-way backend test on Strogatz, Nguyen, or full Feynman-118.
Reads env: BACKEND in {pysr, gplearn, pse}, BENCHMARK in {strogatz, nguyen, feynman118}."""
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

BACKEND = os.environ.get("BACKEND", "pysr").lower()
BENCH   = os.environ.get("BENCHMARK", "strogatz").lower()
SEED = int(os.environ.get("SEED","42"))
Q = int(os.environ.get("Q","100"))
N_DIST = int(os.environ.get("N_DIST","20"))
TIMEOUT = int(os.environ.get("TIMEOUT","10"))
HELDOUT = 200
OUT_PATH = os.environ.get("OUT_PATH", f"data/results/{BACKEND}_{BENCH}_3way.json")

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

# Task pools
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

if BENCH == "strogatz": TASKS = [("strogatz", *t) for t in STROGATZ]
elif BENCH == "nguyen": TASKS = [("nguyen",   *t) for t in NGUYEN]
elif BENCH == "feynman118":
    fpath = "data/benchmarks/pmlb_feynman_open_formula_tasks_20260506.jsonl"
    raw = [json.loads(l) for l in open(fpath,encoding="utf-8").read().splitlines() if l.strip()]
    raw = [t for t in raw if 2 <= len(t["features"]) <= 9]
    TASKS = []
    for t in raw:
        f = t["formula"]; lhs, rhs = f.split("=",1); target = lhs.strip()
        TASKS.append(("feynman118", t["law_id"], rhs.strip(), t["features"], t.get("ranges",{})))
else:
    raise ValueError(f"unknown BENCHMARK {BENCH}")
print(f"{BACKEND.upper()} on {BENCH}: {len(TASKS)} tasks, q={Q}, n_dist={N_DIST}, seed={SEED}\n")

def sample(formula, features, ranges, n, rng):
    try:
        syms = {f: sp.Symbol(f) for f in features}
        fn = sp.lambdify([syms[f] for f in features], sp.sympify(formula, locals=syms), "numpy")
    except: return None
    cols = {}
    for f in features:
        rg = ranges.get(f,[1.0,5.0]); lo,hi = float(rg[0]), float(rg[1])
        if lo>=hi: hi = lo+1
        cols[f] = rng.uniform(lo,hi,n)
    try: y = np.asarray(fn(*[cols[f] for f in features]), dtype=float)
    except: return None
    if not np.all(np.isfinite(y)): return None
    cols["__y"] = y
    return cols

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

# Backends
if BACKEND == "pysr":
    import pysr
    ALL_BIN  = ['+','-','*','/','^']
    ALL_UNARY= ['sqrt','sin','cos','tan','exp','log','sinh','cosh','tanh','abs','square','cube']
    def run_backend(X, y, X_te, y_te, seed):
        m = pysr.PySRRegressor(niterations=40, timeout_in_seconds=TIMEOUT,
            binary_operators=ALL_BIN, unary_operators=ALL_UNARY,
            verbosity=0, random_state=seed, deterministic=True, parallelism='serial')
        t0=time.time()
        try:
            m.fit(X, y); el=time.time()-t0
            return r2(y_te, m.predict(X_te)), str(m.sympy()), el
        except Exception as e:
            return float('-inf'), f"ERR:{e!s:.80}", time.time()-t0
elif BACKEND == "gplearn":
    from gplearn.genetic import SymbolicRegressor
    GP_FN = ('add','sub','mul','div','sqrt','log','sin','cos','tan','abs')
    def run_backend(X, y, X_te, y_te, seed):
        m = SymbolicRegressor(population_size=500, generations=20, function_set=GP_FN,
                              random_state=seed, verbose=0, n_jobs=1,
                              parsimony_coefficient=0.001, max_samples=0.9, stopping_criteria=1e-8)
        t0=time.time()
        try:
            m.fit(X, y); el=time.time()-t0
            return r2(y_te, m.predict(X_te)), str(m._program), el
        except Exception as e:
            return float('-inf'), f"ERR:{e!s:.80}", time.time()-t0
elif BACKEND == "pse":
    from psrn import PSRN_Regressor
    PSE_REPO = "D:/Physics Fundation model/third_party/PSE"
    DR_MASK = f"{PSE_REPO}/dr_mask"
    OPS = ['Add','Mul','Sub','Div','Identity','Sin','Cos','Exp','Log']
    PSE_STAGE = {"default":{"operators":OPS,"time_limit":30,"n_psrn_inputs":4,"n_sample_variables":4},
                 "stages":[{"time_limit":30,"n_psrn_inputs":4}]}
    PSE_TG = {"base":{"has_const":False,"tokens":OPS}}
    def run_backend(X, y, X_te, y_te, seed):
        var_names = [f"x{i}" for i in range(X.shape[1])]
        t0=time.time()
        try:
            torch.cuda.empty_cache()
            reg = PSRN_Regressor(variables=var_names, n_inputs=min(4,X.shape[1]),
                n_symbol_layers=3, use_const=False, use_extra_const=True,
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
else: raise ValueError(f"unknown BACKEND {BACKEND}")

results = []
print(f"{'#':>3} {'task':30s} {'nv':>3} {'sel':>3}  {'fR2':>8} {'vR2':>8}  flag")
print("-"*90)
for i, (suite, tid, formula, features, ranges) in enumerate(TASKS):
    rng = np.random.default_rng(SEED + i)
    aug = sample(formula, features, ranges, Q, rng)
    if aug is None: continue
    te  = sample(formula, features, ranges, HELDOUT, np.random.default_rng(SEED + i + 99999))
    if te is None: continue
    for j in range(N_DIST):
        lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
        aug[f"__d{j}"] = v2.resample(np.array([lo,hi]), Q, rng)
        rng_te = np.random.default_rng(SEED + i + 99999 + j)
        lo2,hi2 = float(rng_te.uniform(-5,0)), float(rng_te.uniform(0,5))
        te[f"__d{j}"] = v2.resample(np.array([lo2,hi2]), HELDOUT, rng_te)
    sel = predict_vars(aug, "__y")
    all_cols = [c for c in aug if c != "__y"]
    Xtr = np.column_stack([aug[c] for c in all_cols])
    Xte = np.column_stack([te[c]  for c in all_cols])
    ytr = aug["__y"]; yte = te["__y"]
    sel_idx = [all_cols.index(s) for s in sel if s in all_cols] or list(range(min(4,len(all_cols))))
    fr2, fexp, ft = run_backend(Xtr, ytr, Xte, yte, SEED+i)
    vr2, vexp, vt = run_backend(Xtr[:,sel_idx], ytr, Xte[:,sel_idx], yte, SEED+i)
    fE = fr2>=0.9999; vE = vr2>=0.9999
    rec = len(set(sel) & set(features)) / max(1, len(features))
    results.append({"suite":suite,"id":tid,"nvars":len(features),"sel_cols":len(sel_idx),
                    "full_r2":fr2,"var_r2":vr2,"full_exact":fE,"var_exact":vE,
                    "full_t":ft,"var_t":vt,"full_expr":fexp,"var_expr":vexp,
                    "true_vars":sorted(features),"sel_vars":sorted(sel),"recall":rec})
    sym = ("F" if fE else "-")+("V" if vE else "-")
    print(f"{i+1:>3} {tid[:30]:30s} {len(features):>3} {len(sel_idx):>3}  {fr2:>8.3f} {vr2:>8.3f}  {sym}")

n = len(results)
if n:
    fe = sum(r['full_exact'] for r in results); ve = sum(r['var_exact'] for r in results)
    print(f"\n{BACKEND} on {BENCH}, n={n}: full {fe}/{n} ({100*fe/n:.0f}%), DenoisedSR {ve}/{n} ({100*ve/n:.0f}%)")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps(results, indent=2))
print(f"Saved -> {OUT_PATH}")
