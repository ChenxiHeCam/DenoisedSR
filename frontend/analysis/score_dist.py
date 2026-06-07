import sys, json, re, warnings
warnings.filterwarnings("ignore")
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));
sys.path.insert(0, os.path.join(_R, "src")); sys.path.insert(0, os.path.join(_R, "frontend", "train")); sys.path.insert(0, os.path.join(_R, "src", "physics_fm")) if os.path.isdir(os.path.join(_R, "src", "physics_fm")) else None
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
from evaluate_stage8g_open_generation import sample_truth_points
import train_support_predictor_v2 as v2

FEAT=74; HID=128; HEADS=4

class GATDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(FEAT,HID); self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*HEADS); self.gat2=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1)
        self.norm2=nn.LayerNorm(HID); self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x)); h=F.gelu(self.norm1(self.gat1(h,ei))); h=F.gelu(self.norm2(self.gat2(h,ei))); return self.head(h).squeeze(-1)

COOC=jl("models/cooc_graph.joblib")
ckpt=torch.load("models/gan_gat_pretrained.pt",map_location="cpu")
model=GATDisc(); model.load_state_dict(ckpt["D"]); model.eval()

# RF v2_ft
from joblib import load as jl2
rf_ckpt = jl2("models/support_predictor_v2_ft.joblib")
rf_clf  = rf_ckpt['support_clf']

# collect formulas
seen,tasks=[],[]
base=Path("D:/Physics Fundation model/artifacts/stage9_opensidr_expert_route_expansion_manifest_20260513/route_outputs/real591")
for rd in sorted(base.iterdir()):
    p=rd/"records.jsonl"
    if not p.exists(): continue
    with open(p,encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            lid=row.get("original_law_id") or row.get("law_id","")
            if lid in seen: continue
            truth=row.get("truth_surface","")
            if not truth: continue
            import sympy as sp
            ts=re.sub(r"\s*=\s*0\s*$","",truth.strip())
            if "=" in ts: ts=ts.split("=",1)[1].strip()
            try: syms=sorted(str(s) for s in sp.sympify(ts,evaluate=False).free_symbols)
            except: continue
            if len(syms)<2 or len(syms)>6: continue
            rng=np.random.default_rng(0)
            vals,_=sample_truth_points(truth,5,rng)
            if vals is None or len(vals)<2: continue
            if any(s not in vals for s in syms): continue
            seen.append(lid); tasks.append({"law_id":lid,"truth":truth,"symbols":syms})
            if len(tasks)>=60: break
    if len(tasks)>=60: break

print(f"Analyzing {len(tasks)} formulas\n")

gat_true_scores, gat_noise_scores = [], []
rf_true_scores,  rf_noise_scores  = [], []
SEED=42

for task in tasks:
    syms=list(task["symbols"]); truth=task["truth"]
    rng=np.random.default_rng(SEED+tasks.index(task))
    vals,_=sample_truth_points(truth,100,rng)
    if vals is None: continue
    m=re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)",truth)
    target=m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
    true_vars=set(s for s in syms if s!=target and s in vals)
    n_dist=15
    aug=dict(vals)
    for i in range(n_dist*2):
        if sum(1 for k in aug if k.startswith("__d"))>=n_dist: break
        col=v2.resample(np.array([float(rng.uniform(-3,0)),float(rng.uniform(0,3))]),len(vals[target]),rng)
        if not v2.is_functionally_related(col,vals[target]): aug[f"__d{i}"]=col
    cols=[c for c in aug if c!=target]
    feats=np.array([v2.column_features(aug[c],aug[target],var_name=c) for c in cols],dtype=np.float32)
    feats=np.nan_to_num(feats,nan=0.,posinf=10.,neginf=-10.)

    # GAT scores
    edges=set()
    for i,a in enumerate(cols):
        for j,b in enumerate(cols):
            if i!=j and COOC.get(a,{}).get(b,0)>=5: edges.add((i,j))
    if not edges:
        for i in range(len(cols)):
            for j in range(i+1,len(cols)): edges.add((i,j)); edges.add((j,i))
    src,dst=zip(*edges)
    ei=torch.tensor([list(src),list(dst)],dtype=torch.long)
    ei,_=add_self_loops(ei,num_nodes=len(cols))
    with torch.no_grad():
        gat_probs=torch.sigmoid(model(torch.tensor(feats),ei)).numpy()

    # RF scores
    from run_pysr_pmlb_feynman_learned_prior import positive_probability
    rf_probs=positive_probability(rf_clf, feats)

    for i,col in enumerate(cols):
        is_true = col in true_vars
        gat_s = float(gat_probs[i])
        rf_s  = float(rf_probs[i])
        if is_true:
            gat_true_scores.append(gat_s)
            rf_true_scores.append(rf_s)
        else:
            gat_noise_scores.append(gat_s)
            rf_noise_scores.append(rf_s)

print("=== TRUE VARIABLE SCORE DISTRIBUTION ===")
for thresh in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    gat_pct = 100*np.mean(np.array(gat_true_scores)>=thresh)
    rf_pct  = 100*np.mean(np.array(rf_true_scores) >=thresh)
    print(f"  score >= {thresh:.2f}:  GAT={gat_pct:.0f}%  RF={rf_pct:.0f}%")

print("\n=== NOISE VARIABLE SCORE DISTRIBUTION (false positive rate) ===")
for thresh in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
    gat_fp = 100*np.mean(np.array(gat_noise_scores)>=thresh)
    rf_fp  = 100*np.mean(np.array(rf_noise_scores) >=thresh)
    print(f"  score >= {thresh:.2f}:  GAT={gat_fp:.1f}%  RF={rf_fp:.1f}%")

print(f"\nTotal: {len(gat_true_scores)} true var scores, {len(gat_noise_scores)} noise scores")
print(f"GAT true mean={np.mean(gat_true_scores):.3f}  noise mean={np.mean(gat_noise_scores):.3f}")
print(f"RF  true mean={np.mean(rf_true_scores):.3f}   noise mean={np.mean(rf_noise_scores):.3f}")
