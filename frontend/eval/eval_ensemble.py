"""
Ensemble RF + GAT — test three voting strategies:
  union:        select if RF >= t_rf  OR  GAT >= t_gat
  intersection: select if RF >= t_rf  AND GAT >= t_gat
  soft_avg:     select if (rf_score + gat_score)/2 >= t
"""
import sys, json, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
from evaluate_stage8g_open_generation import sample_truth_points
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4
SEED=42

class GATDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(FEAT,HID); self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*HEADS); self.gat2=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1)
        self.norm2=nn.LayerNorm(HID); self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x)); h=F.gelu(self.norm1(self.gat1(h,ei))); h=F.gelu(self.norm2(self.gat2(h,ei))); return self.head(h).squeeze(-1)

COOC = jl("models/cooc_graph.joblib")
gat_ckpt = torch.load("models/gan_gat_pretrained.pt", map_location="cpu")
gat_model = GATDisc(); gat_model.load_state_dict(gat_ckpt["D"]); gat_model.eval()

rf_ckpt = jl("models/support_predictor_v2_ft.joblib")
rf_clf  = rf_ckpt['support_clf']

# load formulas
seen, tasks = [], []
base = Path("D:/Physics Fundation model/artifacts/stage9_opensidr_expert_route_expansion_manifest_20260513/route_outputs/real591")
for rd in sorted(base.iterdir()):
    p = rd/"records.jsonl"
    if not p.exists(): continue
    with open(p, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lid = row.get("original_law_id") or row.get("law_id","")
            if lid in seen: continue
            truth = row.get("truth_surface","")
            if not truth: continue
            import sympy as sp
            ts = re.sub(r"\s*=\s*0\s*$","",truth.strip())
            if "=" in ts: ts = ts.split("=",1)[1].strip()
            try: syms = sorted(str(s) for s in sp.sympify(ts,evaluate=False).free_symbols)
            except: continue
            if len(syms)<2 or len(syms)>6: continue
            rng = np.random.default_rng(0)
            vals, _ = sample_truth_points(truth,5,rng)
            if vals is None or len(vals)<2: continue
            if any(s not in vals for s in syms): continue
            seen.append(lid); tasks.append({"law_id":lid,"truth":truth,"symbols":syms})
            if len(tasks)>=91: break
    if len(tasks)>=91: break

print(f"Evaluating on {len(tasks)} formulas\n")

def get_scores(vals_aug, target, cols, feats):
    # GAT scores
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
        gat_s = torch.sigmoid(gat_model(torch.tensor(feats),ei)).numpy()
    # RF scores
    rf_s = positive_probability(rf_clf, feats)
    return gat_s, rf_s

def eval_strategy(name, selector_fn):
    precs,recs,perfect = [],[],0
    for task in tasks:
        syms=list(task["symbols"]); truth=task["truth"]
        rng=np.random.default_rng(SEED+tasks.index(task))
        vals,_ = sample_truth_points(truth,100,rng)
        if vals is None: continue
        m=re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)",truth)
        target=m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
        true_vars=set(s for s in syms if s!=target and s in vals)
        n_dist=int(rng.integers(4,40))
        aug=dict(vals)
        for i in range(n_dist*2):
            if sum(1 for k in aug if k.startswith("__d"))>=n_dist: break
            col=v2.resample(np.array([float(rng.uniform(-3,0)),float(rng.uniform(0,3))]),len(vals[target]),rng)
            if not v2.is_functionally_related(col,vals[target]): aug[f"__d{i}"]=col
        cols=[c for c in aug if c!=target]
        feats=np.array([v2.column_features(aug[c],aug[target],var_name=c) for c in cols],dtype=np.float32)
        feats=np.nan_to_num(feats,nan=0.,posinf=10.,neginf=-10.)
        gat_s, rf_s = get_scores(aug, target, cols, feats)
        sel = selector_fn(cols, gat_s, rf_s)
        sel = {c for c in sel if not c.startswith("__d")}
        if len(sel)<2: sel=sel|{cols[int(np.argmax(rf_s))]}
        tp=len(sel&true_vars); fp=len(sel-true_vars); fn=len(true_vars-sel)
        if tp+fp>0: precs.append(tp/(tp+fp))
        if tp+fn>0: recs.append(tp/(tp+fn))
        if tp+fn>0 and tp/(tp+fn)>=0.999: perfect+=1
    n=len(precs)
    print(f"[{name}]  prec={np.mean(precs):.3f}  recall={np.mean(recs):.3f}  "
          f"F1={2*np.mean(precs)*np.mean(recs)/(np.mean(precs)+np.mean(recs)+1e-9):.3f}  "
          f"perfect={perfect}/{n} ({100*perfect/n:.0f}%)")

# Individual models first
eval_strategy("RF only        (t=0.25)",
    lambda cols,g,r: {c for c,s in zip(cols,r) if s>=0.25})
eval_strategy("GAT only       (t=0.10)",
    lambda cols,g,r: {c for c,s in zip(cols,g) if s>=0.10})

print()
# Ensemble strategies
eval_strategy("Union  RF≥0.25 | GAT≥0.10",
    lambda cols,g,r: {c for c,gs,rs in zip(cols,g,r) if rs>=0.25 or gs>=0.10})
eval_strategy("Union  RF≥0.20 | GAT≥0.10",
    lambda cols,g,r: {c for c,gs,rs in zip(cols,g,r) if rs>=0.20 or gs>=0.10})
eval_strategy("Intersect RF≥0.25 & GAT≥0.10",
    lambda cols,g,r: {c for c,gs,rs in zip(cols,g,r) if rs>=0.25 and gs>=0.10})
eval_strategy("Soft avg ≥0.15",
    lambda cols,g,r: {c for c,gs,rs in zip(cols,g,r) if (gs+rs)/2>=0.15})
eval_strategy("Soft avg ≥0.12",
    lambda cols,g,r: {c for c,gs,rs in zip(cols,g,r) if (gs+rs)/2>=0.12})
print()
print("[RF v2_ft baseline: prec=0.945  recall=0.964  perfect=86%]")
