"""
Pure supervised GAT support predictor.
- 40k formulas (8k pre-sampled + 32k v3 lazy-sampled)
- Weighted BCE for class imbalance (pos ~12%, weight ~7x)
- Dynamic distractors 4-40 per example
- Physics confounders filtered by functional dependence
- No GAN
"""
import sys, re, time, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'D:/Physics Fundation model/src')
sys.path.insert(0, 'D:/Physics Fundation model/scripts')

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jload, dump as jdump

from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import infer_symbols
import train_support_predictor_v2 as v2

try:
    import psutil; psutil.Process().nice(10)
except: pass

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_DIM   = 74
HIDDEN     = 128
HEADS      = 4
EPOCHS     = 40
BATCH_SIZE = 32
LR         = 3e-4
POS_WEIGHT = 7.0   # compensate ~12% positive rate
Q          = 80
SEED       = 42
COOC_THRESH = 5

OUT_DIR    = Path('models')
OUT_DIR.mkdir(exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────

class GATClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj  = nn.Linear(FEAT_DIM, HIDDEN)
        self.gat1  = GATConv(HIDDEN, HIDDEN, heads=HEADS, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(HIDDEN * HEADS)
        self.gat2  = GATConv(HIDDEN * HEADS, HIDDEN, heads=HEADS, concat=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(HIDDEN * HEADS)
        self.gat3  = GATConv(HIDDEN * HEADS, HIDDEN, heads=1, concat=False, dropout=0.1)
        self.norm3 = nn.LayerNorm(HIDDEN)
        self.head  = nn.Sequential(
            nn.Linear(HIDDEN, 64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        h = F.gelu(self.proj(x))
        h = F.gelu(self.norm1(self.gat1(h, edge_index)))
        h = F.gelu(self.norm2(self.gat2(h, edge_index)))
        h = self.norm3(self.gat3(h, edge_index))
        return self.head(h).squeeze(-1)

# ── Graph builder ─────────────────────────────────────────────────────────────

COOC = jload(OUT_DIR / 'cooc_graph.joblib')

def build_graph(col_names, n):
    edges = set()
    for i, a in enumerate(col_names):
        for j, b in enumerate(col_names):
            if i != j and COOC.get(a, {}).get(b, 0) >= COOC_THRESH:
                edges.add((i, j))
    if not edges:
        for i in range(n):
            for j in range(i+1, n):
                edges.add((i, j)); edges.add((j, i))
    src, dst = zip(*edges)
    ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
    ei, _ = add_self_loops(ei, num_nodes=n)
    return ei

# ── Example builder ───────────────────────────────────────────────────────────

def make_example(formula, pool, idx, rng, n_conf, n_rand):
    if formula['vals'] is None:
        try:
            vals, _ = sample_truth_points(formula['truth'], Q, rng)
        except Exception:
            return None
        if vals is None or len(vals) < 2: return None
        m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', formula['truth'])
        target    = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
        true_vars = set(vals.keys()) - {target}
        if not true_vars: return None
    else:
        vals, target, true_vars = formula['vals'], formula['target'], set(formula['true_vars'])

    q      = len(vals[target])
    y_vals = vals[target]
    aug    = {target: y_vals}
    for v in true_vars:
        aug[v] = v2.resample(vals[v], q, rng)

    # Simple random noise — original approach, fast and stable
    n_noise = n_conf + n_rand
    for i in range(n_noise):
        lo, hi = float(rng.uniform(-5, 0)), float(rng.uniform(0, 5))
        aug[f'__d{i}'] = v2.resample(np.array([lo, hi]), q, rng)

    all_cols = sorted(c for c in aug if c != target)
    symbols  = infer_symbols(formula['truth'])

    feats, labels = [], []
    for col in all_cols:
        feat = v2.column_features(aug[col], y_vals, var_name=col)
        feats.append(feat)
        # Label: 1 if name is in formula symbols, 0 otherwise
        labels.append(int(col in symbols and col in true_vars))

    feat_mat = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0., posinf=10., neginf=-10.)
    labels   = np.array(labels, dtype=np.float32)
    return feat_mat, labels, all_cols

# ── Load pool ─────────────────────────────────────────────────────────────────

pool_path = OUT_DIR / 'formula_pool_40k.joblib'
if not pool_path.exists():
    pool_path = OUT_DIR / 'formula_pool.joblib'
pool = [f for f in jload(pool_path) if f['vals'] is not None]  # pre-sampled only
print(f"Pool: {len(pool)} formulas (pre-sampled)  device={DEVICE}")

# ── Training ──────────────────────────────────────────────────────────────────

model   = GATClassifier().to(DEVICE)
opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
pos_w   = torch.tensor(POS_WEIGHT, device=DEVICE)
bce     = nn.BCEWithLogitsLoss(pos_weight=pos_w)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
print(f"POS_WEIGHT={POS_WEIGHT}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}\n")

rng     = np.random.default_rng(SEED)
indices = list(range(len(pool)))
best_f1 = 0.0

for epoch in range(1, EPOCHS + 1):
    rng.shuffle(indices)
    losses, tp_all, fp_all, fn_all = [], 0, 0, 0
    t0 = time.time()
    model.train()

    for batch_start in range(0, len(indices) - BATCH_SIZE, BATCH_SIZE):
        batch_idx = indices[batch_start: batch_start + BATCH_SIZE]
        n_conf = int(rng.integers(4, 20))
        n_rand = int(rng.integers(2, 8))

        graphs, label_list = [], []
        for ii in batch_idx:
            result = make_example(pool[ii], pool, ii, rng, n_conf, n_rand)
            if result is None: continue
            feat_mat, labels, col_names = result
            if feat_mat.shape[0] < 2: continue
            ei = build_graph(col_names, feat_mat.shape[0])
            graphs.append(Data(x=torch.tensor(feat_mat), edge_index=ei))
            label_list.append(torch.tensor(labels, dtype=torch.float32))

        if not graphs: continue
        batch      = Batch.from_data_list(graphs).to(DEVICE)
        labels_cat = torch.cat(label_list).to(DEVICE)

        opt.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss   = bce(logits, labels_cat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        # Track train metrics
        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.25).float()
            tp_all += ((pred == 1) & (labels_cat == 1)).sum().item()
            fp_all += ((pred == 1) & (labels_cat == 0)).sum().item()
            fn_all += ((pred == 0) & (labels_cat == 1)).sum().item()

    sched.step()
    prec = tp_all / (tp_all + fp_all + 1e-8)
    rec  = tp_all / (tp_all + fn_all + 1e-8)
    f1   = 2*prec*rec / (prec+rec+1e-8)
    print(f"Epoch {epoch:3d}/{EPOCHS}  loss={np.mean(losses):.4f}  "
          f"prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}  t={time.time()-t0:.1f}s", flush=True)

    if f1 > best_f1:
        best_f1 = f1
        torch.save({'model': model.state_dict(), 'epoch': epoch,
                    'feat_dim': FEAT_DIM, 'hidden': HIDDEN, 'heads': HEADS,
                    'version': 'gat_supervised_v1'},
                   OUT_DIR / 'gat_best.pt')

torch.save({'model': model.state_dict(), 'epoch': EPOCHS,
            'feat_dim': FEAT_DIM, 'hidden': HIDDEN, 'heads': HEADS,
            'version': 'gat_supervised_v1'},
           OUT_DIR / 'gat_final.pt')
print(f"\nSaved gat_final.pt  best_F1={best_f1:.3f}")
