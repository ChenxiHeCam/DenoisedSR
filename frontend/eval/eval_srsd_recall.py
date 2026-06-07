"""
DenoisedSR variable-support recall on SRSD-Feynman (Matsubara) dummy variants.
External, physical-unit benchmark we never trained on. 120 formulas across
easy/medium/hard. Each formula's file has 8000 rows; we subsample q=100.
True variables come from parsing the SRSD sympy_eq_str; dummies are listed
in supp_info. We never use the formula at test time.

Compares DenoisedSR (RF+GAT ensemble @ tau=0.10) to Lasso CV (oracle top-k).
"""
import sys, json, os, warnings, time
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from sklearn.linear_model import LassoCV
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

SEED = int(os.environ.get("SEED","42"))
Q    = int(os.environ.get("Q","100"))
OUT_PATH = os.environ.get("OUT_PATH","data/results/srsd_dummy_recall.json")
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

manifest = json.load(open("data/benchmarks/srsd/manifest.json"))
keys = list(manifest.keys())
print(f"SRSD-Feynman dummy: {len(keys)} formulas (q={Q}, seed={SEED})\n")

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
    return {c for c,rs,gs in zip(cols,rf_s,gat_s) if rs+gs>=0.10}

def lasso_select_topk(X, y, k):
    try:
        Xs = (X - X.mean(0)) / (X.std(0)+1e-12)
        ys = (y - y.mean()) / (y.std()+1e-12)
        m = LassoCV(cv=5, random_state=0, max_iter=2000, n_alphas=20).fit(Xs, ys)
        sc = np.abs(m.coef_)
    except:
        sc = np.zeros(X.shape[1])
    if k>=len(sc): return set(range(len(sc)))
    return set(np.argsort(-sc)[:k].tolist())

per_task = []
agg_d, agg_l = [], []
agg_by_split = {"easy":[], "medium":[], "hard":[]}
t0 = time.time()

for idx, key in enumerate(keys):
    info = manifest[key]
    split = info["split"]; true_set = set(info["true_vars"])
    if not true_set:
        continue
    rng = np.random.default_rng(SEED + idx)
    data = np.loadtxt(info["file"])
    if data.ndim<2 or data.shape[0]<Q: continue
    sel_rows = rng.choice(data.shape[0], Q, replace=False)
    sub = data[sel_rows]
    X = sub[:, :-1]; y = sub[:, -1]
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)): continue
    n_cols = X.shape[1]
    col_names = [f"x{i}" for i in range(n_cols)]
    aug = {col_names[i]: X[:,i] for i in range(n_cols)}; aug["__y"] = y
    # DenoisedSR
    try:
        sel = denoisedsr_select(aug, "__y")
    except Exception as e:
        sel = set()
    tp_d = len(sel & true_set); fn_d = len(true_set - sel); fp_d = len(sel - true_set - {"__y"})
    rec_d = tp_d/(tp_d+fn_d) if (tp_d+fn_d) else 0
    prec_d = tp_d/(tp_d+fp_d) if (tp_d+fp_d) else (1.0 if tp_d else 0)
    # Lasso oracle k
    sel_idx = lasso_select_topk(X, y, len(true_set))
    sel_l = {col_names[i] for i in sel_idx}
    tp_l = len(sel_l & true_set); fn_l = len(true_set - sel_l); fp_l = len(sel_l - true_set)
    rec_l = tp_l/(tp_l+fn_l) if (tp_l+fn_l) else 0
    prec_l = tp_l/(tp_l+fp_l) if (tp_l+fp_l) else (1.0 if tp_l else 0)
    per_task.append({"key":key, "split":split, "n_true":len(true_set),
                     "n_dummy":info["n_dummy"], "n_cols":n_cols,
                     "denoisedsr": {"rec":rec_d,"prec":prec_d,"sel":sorted(sel)},
                     "lasso_oracle":{"rec":rec_l,"prec":prec_l,"sel":sorted(sel_l)}})
    agg_d.append((rec_d,prec_d)); agg_l.append((rec_l,prec_l))
    agg_by_split[split].append((rec_d, rec_l))
    if (idx+1)%30==0: print(f"  [{idx+1}/{len(keys)}] elapsed {time.time()-t0:.0f}s")

ad = np.array(agg_d); al = np.array(agg_l)
print(f"\nn_used = {len(per_task)} / {len(keys)}\n")
print(f"{'Method':25s} | recall  precision  perfect%")
print("-"*55)
print(f"{'DenoisedSR (ours @0.10)':25s} | {ad[:,0].mean():.3f}  {ad[:,1].mean():.3f}     "
      f"{100*(ad[:,0]>=0.999).mean():5.1f}%")
print(f"{'Lasso CV (oracle k)':25s} | {al[:,0].mean():.3f}  {al[:,1].mean():.3f}     "
      f"{100*(al[:,0]>=0.999).mean():5.1f}%")
print()
print(f"{'Per-split (recall)':25s} | {'DenoisedSR':12s} | {'Lasso oracle':12s}")
print("-"*55)
summary = {"overall":{"denoisedsr_rec":float(ad[:,0].mean()),
                       "denoisedsr_prec":float(ad[:,1].mean()),
                       "denoisedsr_perfect":float((ad[:,0]>=0.999).mean()),
                       "lasso_rec":float(al[:,0].mean()),
                       "lasso_prec":float(al[:,1].mean()),
                       "lasso_perfect":float((al[:,0]>=0.999).mean()),
                       "n":int(len(per_task))}}
for s in ["easy","medium","hard"]:
    arr = np.array(agg_by_split[s]) if agg_by_split[s] else np.zeros((0,2))
    if len(arr)==0: continue
    summary[s] = {"denoisedsr_rec":float(arr[:,0].mean()),
                  "denoisedsr_perfect":float((arr[:,0]>=0.999).mean()),
                  "lasso_rec":float(arr[:,1].mean()),
                  "lasso_perfect":float((arr[:,1]>=0.999).mean()),
                  "n":int(len(arr))}
    print(f"  {s:23s} | {arr[:,0].mean():.3f} (n={len(arr)})  | {arr[:,1].mean():.3f}")

Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_PATH).write_text(json.dumps({"config":{"seed":SEED,"q":Q,"benchmark":"SRSD-Feynman dummy"},
                                       "summary":summary,"per_task":per_task}, indent=2))
print(f"\nSaved -> {OUT_PATH}")
