"""
Compare two noise distributions:
  A) Random uniform noise  (current eval setup)
  B) Physics confounders   (real variable names from other formulas, range-sampled)

For each: measure support predictor recall, then run PySR and count exact recovery.
"""
import sys, json, re, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"D:/Physics Fundation model/src"); sys.path.insert(0,"D:/Physics Fundation model/scripts")

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jl
import pysr
from evaluate_stage8g_open_generation import sample_truth_points
import train_support_predictor_v2 as v2
from run_pysr_pmlb_feynman_learned_prior import positive_probability

FEAT=74; HID=128; HEADS=4; SEED=42; Q=100; N_DIST=15; TIMEOUT=10

# ── Load models ───────────────────────────────────────────────────────────────
COOC = jl("models/cooc_graph.joblib")
pool = jl("models/formula_pool.joblib")  # for physics confounders

class GATDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(FEAT,HID); self.gat1=GATConv(HID,HID,heads=HEADS,concat=True,dropout=0.1)
        self.norm1=nn.LayerNorm(HID*HEADS); self.gat2=GATConv(HID*HEADS,HID,heads=1,concat=False,dropout=0.1)
        self.norm2=nn.LayerNorm(HID); self.head=nn.Sequential(nn.Linear(HID,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(self,x,ei):
        h=F.gelu(self.proj(x)); h=F.gelu(self.norm1(self.gat1(h,ei))); h=F.gelu(self.norm2(self.gat2(h,ei))); return self.head(h).squeeze(-1)

gat_ckpt = torch.load("models/gan_gat_pretrained.pt", map_location="cpu")
gat = GATDisc(); gat.load_state_dict(gat_ckpt["D"]); gat.eval()

rf_ckpt = jl("models/support_predictor_v2_ft.joblib")
rf_clf  = rf_ckpt['support_clf']

# ── Load 20 real591 formulas (2-4 vars, tractable for PySR) ──────────────────
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
            if len(syms)<2 or len(syms)>4: continue  # 2-4 vars only
            rng = np.random.default_rng(0)
            vals,_ = sample_truth_points(truth,5,rng)
            if vals is None or len(vals)<2: continue
            if any(s not in vals for s in syms): continue
            seen.append(lid); tasks.append({"law_id":lid,"truth":truth,"symbols":syms})
            if len(tasks)>=20: break
    if len(tasks)>=20: break

print(f"Testing on {len(tasks)} formulas, {N_DIST} distractors each, PySR timeout={TIMEOUT}s\n")

# ── Noise generators ──────────────────────────────────────────────────────────

def add_random_noise(vals, target, n, rng):
    """Pure random uniform noise columns."""
    aug = dict(vals); q = len(vals[target])
    added = 0
    for i in range(n*3):
        if added >= n: break
        col = v2.resample(np.array([float(rng.uniform(-5,0)), float(rng.uniform(0,5))]), q, rng)
        if not v2.is_functionally_related(col, vals[target]):
            aug[f"__d{added}"] = col; added += 1
    return aug

def add_physics_confounders(vals, target, true_vars, task_idx, n, rng):
    """Physics confounders from other formulas in pool, range-resampled."""
    aug = dict(vals); q = len(vals[target])
    conf = v2.pick_confounders(pool, task_idx % len(pool), true_vars | {target},
                               n, rng, q, y_vals=vals[target])
    aug.update(conf)
    # fill remaining with random if needed
    while sum(1 for k in aug if k.startswith("__d") or (k not in vals and k not in conf)) < n:
        col = v2.resample(np.array([float(rng.uniform(-3,0)), float(rng.uniform(0,3))]), q, rng)
        if not v2.is_functionally_related(col, vals[target]):
            aug[f"__dr{len(aug)}"] = col
        break
    return aug

# ── Predictor ─────────────────────────────────────────────────────────────────

def get_ensemble_cols(aug, target, vals, thresh_rf=0.20, thresh_gat=0.10):
    cols = [c for c in aug if c != target]
    feats = np.array([v2.column_features(aug[c], aug[target], var_name=c) for c in cols], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0., posinf=10., neginf=-10.)
    # RF scores
    rf_s = positive_probability(rf_clf, feats)
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
    ei,_ = add_self_loops(ei, num_nodes=len(cols))
    with torch.no_grad():
        gat_s = torch.sigmoid(gat(torch.tensor(feats),ei)).numpy()
    # Union
    selected = [c for c,rs,gs in zip(cols,rf_s,gat_s)
                if (rs>=thresh_rf or gs>=thresh_gat) and not c.startswith("__d") and not c.startswith("__dr")]
    return selected if selected else list(vals.keys()-{target})

def r2(yt, yp):
    ss = np.sum((yt-yp)**2); tot = np.sum((yt-yt.mean())**2)
    return float(1 - ss/(tot+1e-12))

# ── Run experiment ────────────────────────────────────────────────────────────

results = {"random": [], "physics": []}

for noise_type, noise_fn in [("random", None), ("physics", None)]:
    print(f"\n{'='*60}")
    print(f"NOISE TYPE: {noise_type.upper()}")
    print(f"{'='*60}")
    print(f"{'Formula':35s} {'true_vars':20s} {'pred_cols':20s} {'full_R2':>8s} {'pred_R2':>8s} {'full_EX':>7s} {'pred_EX':>7s}")
    print('-'*110)

    for i, task in enumerate(tasks):
        syms  = list(task["symbols"]); truth = task["truth"]
        rng_i = np.random.default_rng(SEED + i)
        vals, _ = sample_truth_points(truth, Q, rng_i)
        te_vals, _ = sample_truth_points(truth, 200, rng_i)
        if vals is None or te_vals is None: continue
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", truth)
        target = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
        true_vars = set(s for s in syms if s!=target and s in vals)

        # Add noise
        if noise_type == "random":
            aug    = add_random_noise(vals, target, N_DIST, rng_i)
            te_aug = add_random_noise(te_vals, target, N_DIST, rng_i)
        else:
            aug    = add_physics_confounders(vals, target, true_vars, i, N_DIST, rng_i)
            te_aug = add_physics_confounders(te_vals, target, true_vars, i, N_DIST, rng_i)

        all_cols  = [c for c in aug if c!=target]
        pred_cols = get_ensemble_cols(aug, target, vals, 0.20, 0.10)
        y_tr = aug[target]; y_te = te_aug[target]

        row = {"law_id": task["law_id"], "truth": truth,
               "true_vars": sorted(true_vars), "pred_cols": pred_cols,
               "recall": len(set(pred_cols)&true_vars)/len(true_vars) if true_vars else 0}

        for mode, cols in [("full", [c for c in all_cols if c in te_aug]),
                            ("pred", [c for c in pred_cols if c in te_aug])]:
            if not cols: cols = list(true_vars & set(te_aug.keys()))
            X_tr = np.column_stack([aug[c]    for c in cols if c in aug])
            X_te = np.column_stack([te_aug[c] for c in cols if c in te_aug])
            mdl = pysr.PySRRegressor(niterations=50, timeout_in_seconds=TIMEOUT,
                binary_operators=['+','-','*','/'], unary_operators=['sqrt','sin','cos','exp','log'],
                verbosity=0, random_state=SEED+i)
            try:
                mdl.fit(X_tr, y_tr)
                sc = r2(y_te, mdl.predict(X_te))
                expr = str(mdl.sympy())
            except: sc=-1.0; expr="ERROR"
            row[f"{mode}_r2"]   = round(sc,4)
            row[f"{mode}_expr"] = expr
            row[f"{mode}_exact"] = sc >= 0.9999

        results[noise_type].append(row)
        fex = "Y" if row["full_exact"] else " "
        pex = "Y" if row["pred_exact"] else " "
        missed = sorted(true_vars - set(pred_cols))
        print(f"{task['law_id']:35s} {str(sorted(true_vars)):20s} {str(pred_cols):20s} "
              f"{row['full_r2']:8.4f} {row['pred_r2']:8.4f} {fex:>7s} {pex:>7s}"
              + (f"  MISSED:{missed}" if missed else ""))

    n = len(results[noise_type])
    full_ex = sum(r["full_exact"] for r in results[noise_type])
    pred_ex = sum(r["pred_exact"] for r in results[noise_type])
    mean_rec = np.mean([r["recall"] for r in results[noise_type]])
    print(f"\n{'─'*110}")
    print(f"EXACT RECOVERY:  full={full_ex}/{n}  ensemble_pred={pred_ex}/{n}")
    print(f"mean variable recall: {mean_rec:.3f}")

print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
for nt in ["random","physics"]:
    r = results[nt]
    n = len(r)
    print(f"{nt:10s}:  full={sum(x['full_exact'] for x in r)}/{n}  "
          f"pred={sum(x['pred_exact'] for x in r)}/{n}  "
          f"recall={np.mean([x['recall'] for x in r]):.3f}")
