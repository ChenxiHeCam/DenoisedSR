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
        self.proj=nn.Linear(FEAT,HID)
        self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*HEADS)
        self.gat2=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1)
        self.norm2=nn.LayerNorm(HID)
        self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x))
        h=F.gelu(self.norm1(self.gat1(h,ei)))
        h=F.gelu(self.norm2(self.gat2(h,ei)))
        return self.head(h).squeeze(-1)

COOC=jl("models/cooc_graph.joblib")

for ckpt_name, label in [
    ("models/gan_gat_pretrained.pt", "GAT pretrained (5ep supervised only)"),
    ("models/gan_gat_final.pt",      "GAT final (5ep pretrain + 25ep GAN)"),
]:
    cp = Path(ckpt_name)
    if not cp.exists():
        print(f"Not found: {ckpt_name}")
        continue
    ckpt=torch.load(cp, map_location="cpu")
    model=GATDisc(); model.load_state_dict(ckpt["D"]); model.eval()

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
                if len(tasks)>=91: break
        if len(tasks)>=91: break

    SEED=42; THRESH=0.25; precs,recs,perfect=[],[],0
    for task in tasks:
        syms=list(task["symbols"]); truth=task["truth"]
        rng=np.random.default_rng(SEED+tasks.index(task))
        vals,_=sample_truth_points(truth,100,rng)
        if vals is None: continue
        m=re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)",truth)
        target=m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
        true_vars=set(s for s in syms if s!=target and s in vals)
        n_dist=int(rng.integers(4,40))
        aug=dict(vals)
        for i in range(n_dist*2):
            if sum(1 for k in aug if k.startswith("__d"))>=n_dist: break
            lo,hi=float(rng.uniform(-3,0)),float(rng.uniform(0,3))
            col=v2.resample(np.array([lo,hi]),len(vals[target]),rng)
            if not v2.is_functionally_related(col,vals[target]):
                aug[f"__d{i}"]=col
        cols=[c for c in aug if c!=target]
        feats=np.array([v2.column_features(aug[c],aug[target],var_name=c) for c in cols],dtype=np.float32)
        feats=np.nan_to_num(feats,nan=0.0,posinf=10.0,neginf=-10.0)
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
            probs=torch.sigmoid(model(torch.tensor(feats),ei)).numpy()
        ranked=sorted(zip(cols,probs),key=lambda t:-float(t[1]))
        sel=set(c for c,p in ranked if float(p)>=THRESH and not c.startswith("__d"))
        if len(sel)<2: sel=set(c for c,_ in ranked[:2] if not c.startswith("__d"))
        tp=len(sel&true_vars); fp=len(sel-true_vars); fn=len(true_vars-sel)
        if tp+fp>0: precs.append(tp/(tp+fp))
        if tp+fn>0: recs.append(tp/(tp+fn))
        if tp+fn>0 and tp/(tp+fn)>=0.999: perfect+=1

    n=len(precs)
    print(f"\n=== {label} ===")
    print(f"N={n}  precision={np.mean(precs):.3f}  recall={np.mean(recs):.3f}  perfect={perfect}/{n} ({100*perfect/n:.0f}%)")

print(f"\n[RF v2_ft baseline: prec=0.945  recall=0.964  perfect=78/91 (86%)]")
