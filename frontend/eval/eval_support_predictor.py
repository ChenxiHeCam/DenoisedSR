"""
Evaluate support predictor quality: precision/recall on real591 formulas
given true variables + 20 noise variables.
No PySR involved — just measure can the predictor filter noise without hurting recall.
Compare old (64-record) model vs new large model.
"""
import sys, json, warnings, re
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import argparse, numpy as np
from pathlib import Path
from joblib import load as joblib_load

from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import (
    build_frontend_training, read_jsonl, positive_probability
)
import run_pysr_pmlb_feynman_learned_prior as _orig_mod
import train_support_predictor_v2 as _v2_mod

Q              = 100
DIST_MIN       = 4     # vary distractors per formula, not fixed
DIST_MAX       = 40
SEED           = 42
THRESHOLD      = 0.25

def add_dist(vals, target, n, rng):
    """Add n physics-like + random confounder columns, skip functionally related ones."""
    from train_support_predictor_v2 import resample, is_functionally_related
    out = dict(vals)
    y   = vals[target]
    m   = len(y)
    added = 0
    # physics-like: random uniform in plausible ranges, skip if related to target
    for i in range(n * 3):   # try 3x more to account for filtered ones
        if added >= n: break
        lo = float(rng.uniform(-5, 0))
        hi = float(rng.uniform(0, 5))
        col = resample(np.array([lo, hi]), m, rng)
        if is_functionally_related(col, y):  # skip derived-like columns
            continue
        out[f'__d{added}'] = col
        added += 1
    return out

def extract_all_symbols(truth):
    import sympy as sp
    ts = re.sub(r'\s*=\s*0\s*$', '', truth.strip())
    if '=' in ts:
        ts = ts.split('=', 1)[1].strip()
    try:
        return sorted(str(s) for s in sp.sympify(ts, evaluate=False).free_symbols)
    except:
        return []

def get_target(truth, syms):
    m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth.strip())
    return m.group(1) if m and m.group(1) in syms else (syms[0] if syms else None)

# ---- collect real591 formulas ----
seen, tasks = set(), []
base = Path('D:/Physics Fundation model/artifacts/stage9_opensidr_expert_route_expansion_manifest_20260513/route_outputs/real591')
for route_dir in sorted(base.iterdir()):
    p = route_dir / 'records.jsonl'
    if not p.exists(): continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            lid = row.get('original_law_id') or row.get('law_id','')
            if lid in seen: continue
            truth = row.get('truth_surface','')
            if not truth: continue
            syms = extract_all_symbols(truth)
            if len(syms) < 2 or len(syms) > 6: continue
            rng_c = np.random.default_rng(0)
            vals, _ = sample_truth_points(truth, 5, rng_c)
            if vals is None or len(vals) < 2: continue
            missing = [s for s in syms if s not in vals]
            if missing: continue
            seen.add(lid)
            tasks.append({'law_id': lid, 'truth': truth, 'symbols': syms})
            if len(tasks) >= 100: break
    if len(tasks) >= 100: break

print(f'Evaluating on {len(tasks)} real591 formulas, {DIST_MIN}-{DIST_MAX} noise vars (random)\n')

def predict_with_model(support_clf, op_clf, vals_aug, target, all_cols, threshold, col_feat_fn):
    """Run support prediction using the right column_features function for this model."""
    y = vals_aug[target]
    x_support = np.asarray([col_feat_fn(vals_aug[c], y, var_name=c) for c in all_cols], dtype=np.float64)
    support_prob = positive_probability(support_clf, x_support)
    ranked = sorted(zip(all_cols, support_prob), key=lambda t: (-float(t[1]), t[0]))
    selected = [c for c, p in ranked if float(p) >= threshold]
    if len(selected) < 3:
        selected = [c for c, _ in ranked[:3]]
    selected = selected[:12]
    return selected


def evaluate_model(support_clf, op_clf, tasks, threshold, label, col_feat_fn=None):
    if col_feat_fn is None:
        col_feat_fn = lambda col, y, var_name='': _orig_mod.column_features(col, y)
    precisions, recalls, f1s = [], [], []
    perfect_recall = 0

    for task in tasks:
        syms  = list(task['symbols'])
        truth = task['truth']
        rng_i = np.random.default_rng(SEED + tasks.index(task))

        vals, _ = sample_truth_points(truth, Q, rng_i)
        if vals is None: continue

        target = get_target(truth, syms)
        if not target or target not in vals: continue

        # Randomly vary number of distractors per formula
        n_dist   = int(rng_i.integers(DIST_MIN, DIST_MAX + 1))
        vals_aug = add_dist(vals, target, n_dist, rng_i)
        all_cols = [c for c in vals_aug if c != target]
        true_vars = set(s for s in syms if s != target and s in vals_aug)

        try:
            pred_cols = predict_with_model(support_clf, op_clf, vals_aug, target, all_cols, threshold, col_feat_fn)
        except:
            continue

        pred_real = set(c for c in pred_cols if not c.startswith('__d'))
        tp = len(pred_real & true_vars)
        fp = len(pred_real - true_vars)
        fn = len(true_vars - pred_real)

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        if rec >= 0.999: perfect_recall += 1

    n = len(precisions)
    print(f'[{label}]')
    print(f'  N={n}  mean_precision={np.mean(precisions):.3f}  mean_recall={np.mean(recalls):.3f}  mean_F1={np.mean(f1s):.3f}')
    print(f'  perfect_recall (1.0): {perfect_recall}/{n} ({100*perfect_recall/n:.0f}%)')
    print(f'  recall>=0.8:  {sum(r>=0.8 for r in recalls)}/{n}')
    print(f'  precision>=0.5: {sum(p>=0.5 for p in precisions)}/{n}')
    print()
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# ---- OLD model (64 records) ----
print('=== OLD MODEL (64 training records) ===')
args_old = argparse.Namespace(q=Q, heldout=200, distractors=20,
    timeout_seconds=10, support_threshold=THRESHOLD, min_prior_columns=3,
    max_prior_columns=12, op_threshold=0.28, seed=SEED, device='cpu')
rng0 = np.random.default_rng(SEED)
old_rows = read_jsonl(Path('data/stage3/stage3_residual_proxy_s64_records.jsonl'), max_rows=512)
old_support, old_op, _ = build_frontend_training(old_rows, args_old, rng0)
evaluate_model(old_support, old_op, tasks, THRESHOLD, 'OLD (64 records)')

# ---- GAN-GAT model ----
gat_path = Path('models/gan_gat_final.pt')
if gat_path.exists():
    print('\n=== GAN-GAT (8k formulas, adversarial training) ===')
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import add_self_loops
    from joblib import load as _jload2
    _COOC  = _jload2('models/cooc_graph.joblib')
    _FEAT  = 74; _HID = 128; _HEADS = 4; _COOC_T = 5

    class _GATDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj  = nn.Linear(_FEAT, _HID)
            self.gat1  = GATConv(_HID, _HID, heads=_HEADS, concat=True, dropout=0.1)
            self.norm1 = nn.LayerNorm(_HID * _HEADS)
            self.gat2  = GATConv(_HID * _HEADS, _HID, heads=1, concat=False, dropout=0.1)
            self.norm2 = nn.LayerNorm(_HID)
            self.head  = nn.Sequential(nn.Linear(_HID, 64), nn.GELU(),
                                       nn.Dropout(0.1), nn.Linear(64, 1))
        def forward(self, x, ei):
            h = F.gelu(self.proj(x))
            h = F.gelu(self.norm1(self.gat1(h, ei)))
            h = F.gelu(self.norm2(self.gat2(h, ei)))
            return self.head(h).squeeze(-1)

    def _build_graph_eval(col_names, n_nodes):
        edges = set()
        for i, a in enumerate(col_names):
            for j, b in enumerate(col_names):
                if i != j and _COOC.get(a, {}).get(b, 0) >= _COOC_T:
                    edges.add((i, j))
        if not edges:
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    edges.add((i,j)); edges.add((j,i))
        src, dst = zip(*edges)
        ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
        ei, _ = add_self_loops(ei, num_nodes=n_nodes)
        return ei
    ckpt_gat = torch.load(gat_path, map_location='cpu')
    D_model  = _GATDisc()
    D_model.load_state_dict(ckpt_gat['D'])
    D_model.eval()

    def predict_gat(vals_aug, target, all_cols, threshold):
        y = vals_aug[target]
        feats = np.array([_v2_mod.column_features(vals_aug[c], y, var_name=c)
                          for c in all_cols], dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
        ei    = _build_graph_eval(all_cols, len(all_cols))
        x     = torch.tensor(feats)
        with torch.no_grad():
            probs = torch.sigmoid(D_model(x, ei)).numpy()
        ranked   = sorted(zip(all_cols, probs), key=lambda t: -float(t[1]))
        selected = [c for c, p in ranked if float(p) >= threshold]
        if len(selected) < 3:
            selected = [c for c, _ in ranked[:3]]
        return selected[:12]

    precisions, recalls, f1s, perfect = [], [], [], 0
    for task in tasks:
        syms  = list(task['symbols'])
        truth = task['truth']
        rng_i = np.random.default_rng(SEED + tasks.index(task))
        vals, _ = sample_truth_points(truth, Q, rng_i)
        if vals is None: continue
        target = get_target(truth, syms)
        if not target or target not in vals: continue
        n_dist   = int(rng_i.integers(DIST_MIN, DIST_MAX + 1))
        vals_aug = add_dist(vals, target, n_dist, rng_i)
        all_cols = [c for c in vals_aug if c != target]
        true_vars = set(s for s in syms if s != target and s in vals_aug)
        try:
            pred = predict_gat(vals_aug, target, all_cols, THRESHOLD)
        except Exception:
            continue
        pred_real = set(c for c in pred if not c.startswith('__d'))
        tp = len(pred_real & true_vars)
        fp = len(pred_real - true_vars)
        fn = len(true_vars - pred_real)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2*prec*rec / (prec+rec+1e-9)
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        if rec >= 0.999: perfect += 1

    n = len(precisions)
    label_gat = 'GAN-GAT (8k, adversarial)'
    print(f'[{label_gat}]')
    print(f'  N={n}  mean_precision={np.mean(precisions):.3f}  mean_recall={np.mean(recalls):.3f}  mean_F1={np.mean(f1s):.3f}')
    print(f'  perfect_recall (1.0): {perfect}/{n} ({100*perfect/n:.0f}%)')
    print()

# ---- v2 model ----
for model_path, label in [
    (Path('models/support_predictor_v2.joblib'),    'v2 (8k formulas, physics confounders, 8-dist, unit feats)'),
    (Path('models/support_predictor_v2_ft.joblib'), 'v2_ft (+ finetune 10-30 confounders)'),
]:
    if model_path.exists():
        print(f'\n=== {label} ===')
        ckpt = joblib_load(model_path)
        evaluate_model(ckpt['support_clf'], ckpt['op_clf'], tasks, THRESHOLD, label,
                       col_feat_fn=_v2_mod.column_features)
    else:
        print(f'  Not found: {model_path}')
