"""
Eval ensemble on real_for_sure_v5_physics65 dataset.
Uses real observation values (train_values/val_values) from the dataset.
"""
import sys, json, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4; SEED=42

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
print(f"GAT epoch {gat_ckpt['epoch']}")

# Load v5 dataset
v5_path = Path("D:/Physics Fundation model/artifacts/stage8g_real_for_sure_20260513_v5_physics65/real_for_sure_v5_physics65_observations_p160v160_seed20260512.jsonl")
tasks = []
with open(v5_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        syms = r.get("symbols", [])
        tv   = r.get("train_values", {})
        vv   = r.get("val_values", {})
        if not syms or not tv or len(tv) < 2: continue
        tasks.append({"law_id": r["law_id"], "symbols": syms,
                      "train_vals": tv, "val_vals": vv})

print(f"v5 dataset: {len(tasks)} formulas\n")

def get_scores(aug, target, cols):
    feats = np.array([v2.column_features(np.array(aug[c]), np.array(aug[target]), var_name=c)
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
    return rf_s, gat_s

def run_eval(name, selector):
    precs, recs, perfect = [], [], 0
    for task in tasks:
        syms   = task["symbols"]
        tv     = {k: np.array(v) for k,v in task["train_vals"].items()}
        if len(tv) < 2: continue
        # target = first symbol that appears as key
        target = syms[0] if syms[0] in tv else list(tv.keys())[0]
        true_vars = set(s for s in syms if s != target and s in tv)
        if not true_vars: continue

        rng = np.random.default_rng(SEED + tasks.index(task))
        n_dist = int(rng.integers(4, 30))
        aug = dict(tv)
        q   = len(tv[target])
        for i in range(n_dist):
            lo,hi = float(rng.uniform(-5,0)), float(rng.uniform(0,5))
            aug[f"__d{i}"] = v2.resample(np.array([lo,hi]), q, rng)

        cols = [c for c in aug if c != target]
        rf_s, gat_s = get_scores(aug, target, cols)
        sel = selector(cols, rf_s, gat_s)
        sel = {c for c in sel if not c.startswith("__d")}
        if not sel: sel = {cols[int(np.argmax(rf_s))]}

        tp=len(sel&true_vars); fp=len(sel-true_vars); fn=len(true_vars-sel)
        if tp+fp>0: precs.append(tp/(tp+fp))
        if tp+fn>0: recs.append(tp/(tp+fn))
        if tp+fn>0 and tp/(tp+fn)>=0.999: perfect+=1

    n = len(precs)
    f1 = 2*np.mean(precs)*np.mean(recs)/(np.mean(precs)+np.mean(recs)+1e-9)
    print(f"  {name:40s}  prec={np.mean(precs):.3f}  rec={np.mean(recs):.3f}  "
          f"F1={f1:.3f}  perfect={perfect}/{n} ({100*perfect/n:.0f}%)")

print("=== v5 physics65 (real observations, unseen domain) ===\n")
run_eval("RF only (t=0.10)",         lambda c,r,g: {x for x,s in zip(c,r) if s>=0.10})
run_eval("GAT only (t=0.08)",        lambda c,r,g: {x for x,s in zip(c,g) if s>=0.08})
run_eval("Combined rf+gat >= 0.10",  lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs+gs>=0.10})
run_eval("Union RF≥0.10 | GAT≥0.08", lambda c,r,g: {x for x,rs,gs in zip(c,r,g) if rs>=0.10 or gs>=0.08})
run_eval("RF≥0.15 (stricter)",       lambda c,r,g: {x for x,s in zip(c,r) if s>=0.15})
