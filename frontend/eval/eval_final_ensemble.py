"""
Final ensemble evaluation: RF + GAT combination strategies
Goal: maximize recall (don't miss variables) while reducing noise
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

FEAT=74; HID=128; HEADS=4; SEED=42; Q=100

class GATDisc(nn.Module):
    def __init__(self, heads=HEADS):
        super().__init__()
        self.proj=nn.Linear(FEAT,HID)
        self.gat1=GATConv(HID,HID,heads=heads,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*heads)
        self.gat2=GATConv(HID*heads,HID,heads=heads,concat=True,dropout=0.1)
        self.norm2=nn.LayerNorm(HID*heads)
        self.gat3=GATConv(HID*heads,HID,heads=1,concat=False,dropout=0.1)
        self.norm3=nn.LayerNorm(HID)
        self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.15),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x))
        h=F.gelu(self.norm1(self.gat1(h,ei)))
        h=F.gelu(self.norm2(self.gat2(h,ei)))
        h=self.norm3(self.gat3(h,ei))
        return self.head(h).squeeze(-1)

COOC = jl("models/cooc_graph.joblib")
rf_ckpt = jl("models/support_predictor_v2_40k.joblib")
rf_clf  = rf_ckpt['support_clf']

# Load best GAT
gat_path = Path("models/gat_best.pt")
gat_ckpt = torch.load(gat_path, map_location="cpu")
gat_model = GATDisc(); gat_model.load_state_dict(gat_ckpt["model"]); gat_model.eval()
print(f"Loaded GAT from epoch {gat_ckpt['epoch']}")

# Collect real591 formulas
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
            vals,_ = sample_truth_points(truth,5,rng)
            if vals is None or len(vals)<2: continue
            if any(s not in vals for s in syms): continue
            seen.append(lid); tasks.append({"law_id":lid,"truth":truth,"symbols":syms})
            if len(tasks)>=91: break
    if len(tasks)>=91: break

print(f"Eval: {len(tasks)} formulas, 4-40 random noise vars\n")

def get_scores(vals_aug, target, cols):
    feats = np.array([v2.column_features(vals_aug[c], vals_aug[target], var_name=c)
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
        gat_s = torch.sigmoid(gat_model(torch.tensor(feats),ei)).numpy()
    return rf_s, gat_s

def run_eval(name, selector):
    precs, recs, perfect = [], [], 0
    for task in tasks:
        syms=list(task["symbols"]); truth=task["truth"]
        rng=np.random.default_rng(SEED+tasks.index(task))
        vals,_ = sample_truth_points(truth,Q,rng)
        if vals is None: continue
        m=re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)",truth)
        target=m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
        true_vars=set(s for s in syms if s!=target and s in vals)
        n_dist=int(rng.integers(4,40))
        aug=dict(vals)
        for i in range(n_dist):
            lo,hi=float(rng.uniform(-5,0)),float(rng.uniform(0,5))
            aug[f"__d{i}"]=v2.resample(np.array([lo,hi]),len(vals[target]),rng)
        cols=[c for c in aug if c!=target]
        rf_s, gat_s = get_scores(aug, target, cols)
        sel = selector(cols, rf_s, gat_s)
        sel = {c for c in sel if not c.startswith("__d")}
        if len(sel)<1: sel={cols[int(np.argmax(rf_s))]}
        tp=len(sel&true_vars); fp=len(sel-true_vars); fn=len(true_vars-sel)
        if tp+fp>0: precs.append(tp/(tp+fp))
        if tp+fn>0: recs.append(tp/(tp+fn))
        if tp+fn>0 and tp/(tp+fn)>=0.999: perfect+=1
    n=len(precs)
    f1 = 2*np.mean(precs)*np.mean(recs)/(np.mean(precs)+np.mean(recs)+1e-9)
    print(f"  {name:40s}  prec={np.mean(precs):.3f}  rec={np.mean(recs):.3f}  "
          f"F1={f1:.3f}  perfect={perfect}/{n} ({100*perfect/n:.0f}%)")

print("Individual models:")
run_eval("RF only (t=0.10)",       lambda c,r,g: {x for x,s in zip(c,r) if s>=0.10})
run_eval("RF only (t=0.15)",       lambda c,r,g: {x for x,s in zip(c,r) if s>=0.15})
run_eval("GAT only (t=0.08)",      lambda c,r,g: {x for x,s in zip(c,g) if s>=0.08})
run_eval("GAT only (t=0.25)",      lambda c,r,g: {x for x,s in zip(c,g) if s>=0.25})

print("\nUnion strategies (maximize recall):")
run_eval("Union RF≥0.15 | GAT≥0.08",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs>=0.15 or gs>=0.08})
run_eval("Union RF≥0.10 | GAT≥0.08",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs>=0.10 or gs>=0.08})

print("\nCombined score (rf+gat, reduce noise + high recall):")
run_eval("Combined≥0.15 (rf+gat)",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs+gs>=0.15})
run_eval("Combined≥0.12 (rf+gat)",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs+gs>=0.12})
run_eval("Combined≥0.10 (rf+gat)",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs+gs>=0.10})
run_eval("Combined≥0.08 (rf+gat)",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs+gs>=0.08})

print("\nWeighted (2*rf + gat, RF trusted more):")
run_eval("2*RF+GAT ≥ 0.25",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if 2*rs+gs>=0.25})
run_eval("2*RF+GAT ≥ 0.20",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if 2*rs+gs>=0.20})
