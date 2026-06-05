"""
GAN-GAT Support Predictor (clean version)

Discriminator: Graph Attention Network
  - Nodes: candidate variable columns (true vars + confounders)
  - Edges: physics co-occurrence prior + observation correlation
  - Output: P(relevant) per node

Generator: MLP, two modes
  - mode=0: physics-like hard negatives (conditioned on other formula's true vars)
  - mode=1: random-noise-like easy negatives
  - G tries to fool D; D learns to detect real functional dependence
"""
import sys, json, re, time, os
sys.path.insert(0, 'src')
sys.path.insert(0, 'D:/Physics Fundation model/src')
sys.path.insert(0, 'D:/Physics Fundation model/scripts')

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from pathlib import Path
from joblib import load as jload, dump as jdump
from collections import defaultdict

from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import infer_symbols, infer_ops, OP_ORDER
import train_support_predictor_v2 as v2

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_DIM    = 74
NOISE_DIM   = 32
HIDDEN_DIM  = 128
GAT_HEADS   = 4
PRETRAIN_EPOCHS = 5   # supervised warm-up before GAN
EPOCHS          = 25  # adversarial epochs after warm-up
BATCH_SIZE  = 32
LR_D, LR_G = 2e-4, 1e-4
N_GAN_GEN   = 4
Q           = 80
SEED        = 42
COOC_THRESH = 5
CORR_THRESH = 0.3

OUT_DIR    = Path('models')
OUT_DIR.mkdir(exist_ok=True)
POOL_CACHE = OUT_DIR / 'formula_pool.joblib'
COOC_CACHE = OUT_DIR / 'cooc_graph.joblib'

# ── Load co-occurrence graph ──────────────────────────────────────────────────
if COOC_CACHE.exists():
    COOC = jload(COOC_CACHE)
    print(f"Loaded cooc: {len(COOC)} vars")
else:
    COOC = defaultdict(dict)
    print("No cooc cache, using empty graph")

# ── Models ────────────────────────────────────────────────────────────────────

class GATDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj  = nn.Linear(FEAT_DIM, HIDDEN_DIM)
        self.gat1  = GATConv(HIDDEN_DIM, HIDDEN_DIM, heads=GAT_HEADS, concat=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(HIDDEN_DIM * GAT_HEADS)
        self.gat2  = GATConv(HIDDEN_DIM * GAT_HEADS, HIDDEN_DIM, heads=1, concat=False, dropout=0.1)
        self.norm2 = nn.LayerNorm(HIDDEN_DIM)
        self.head  = nn.Sequential(nn.Linear(HIDDEN_DIM, 64), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(64, 1))

    def encode(self, x, edge_index):
        h = F.gelu(self.proj(x))
        h = F.gelu(self.norm1(self.gat1(h, edge_index)))
        h = F.gelu(self.norm2(self.gat2(h, edge_index)))
        return h

    def forward(self, x, edge_index):
        return self.head(self.encode(x, edge_index)).squeeze(-1)

    def forward_flat(self, x):
        """No graph edges — self-loops only. For generated samples."""
        N = x.shape[0]
        ei = torch.stack([torch.arange(N, device=x.device),
                          torch.arange(N, device=x.device)])
        return self.forward(x, ei)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_emb = nn.Embedding(2, 16)
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + 16 + FEAT_DIM, HIDDEN_DIM), nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(),
            nn.Linear(HIDDEN_DIM, FEAT_DIM),
        )

    def forward(self, noise, mode, ctx):
        return self.net(torch.cat([noise, self.mode_emb(mode), ctx], dim=-1))


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(col_names, feat_mat, n_nodes):
    edges = set()
    # Co-occurrence edges
    for i, a in enumerate(col_names):
        for j, b in enumerate(col_names):
            if i != j and COOC.get(a, {}).get(b, 0) >= COOC_THRESH:
                edges.add((i, j))
    # Correlation edges (from statistical features)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Use first feature (linear corr) as proxy
            c = float(feat_mat[i, 0] - feat_mat[j, 0])
            if abs(feat_mat[i, 0]) > CORR_THRESH or abs(feat_mat[j, 0]) > CORR_THRESH:
                edges.add((i, j)); edges.add((j, i))
    if not edges:
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                edges.add((i, j)); edges.add((j, i))
    src, dst = zip(*edges)
    ei = torch.tensor([list(src), list(dst)], dtype=torch.long)
    ei, _ = add_self_loops(ei, num_nodes=n_nodes)
    return ei


# ── Example builder ───────────────────────────────────────────────────────────

def make_example(formula, pool, idx, rng, n_conf, n_rand):
    # Handle lazy (vals=None) formulas
    if formula['vals'] is None:
        try:
            vals, _ = sample_truth_points(formula['truth'], Q, rng)
        except Exception:
            return None
        if vals is None or len(vals) < 2:
            return None
        m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', formula['truth'])
        target    = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
        true_vars = set(vals.keys()) - {target}
        if not true_vars:
            return None
    else:
        vals, target, true_vars = formula['vals'], formula['target'], set(formula['true_vars'])

    q      = len(vals[target])
    y_vals = vals[target]

    # Re-sample true vars with random distribution
    aug = {target: y_vals}
    for v in true_vars:
        aug[v] = v2.resample(vals[v], q, rng)

    # Physics confounders — skip functionally related ones
    conf_dict = v2.pick_confounders(pool, idx, true_vars | {target},
                                    n_conf, rng, q, y_vals=y_vals)
    aug.update(conf_dict)
    conf_set = set(conf_dict.keys())

    # Random noise
    for i in range(n_rand):
        lo, hi = float(rng.uniform(-5, 0)), float(rng.uniform(0, 5))
        aug[f'__r{i}'] = v2.resample(np.array([lo, hi]), q, rng)

    all_cols = sorted(c for c in aug if c != target)
    symbols  = infer_symbols(formula['truth'])

    feats, labels = [], []
    for col in all_cols:
        feat = v2.column_features(aug[col], y_vals, var_name=col)
        feats.append(feat)
        if col in conf_set or col.startswith('__r'):
            labels.append(0)
        else:
            nm = int(col in symbols and col in true_vars)
            fr = v2.is_functionally_related(aug[col], y_vals)
            labels.append(max(nm, fr))

    feat_mat = np.array(feats, dtype=np.float32)
    labels   = np.array(labels, dtype=np.float32)
    feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=10.0, neginf=-10.0)

    # Context for G = mean features of true vars from a DIFFERENT formula
    ctx = feat_mat[labels.astype(bool)].mean(axis=0) if labels.any() else feat_mat.mean(axis=0)

    return feat_mat, labels, all_cols, ctx.astype(np.float32)


# ── Load pool ─────────────────────────────────────────────────────────────────

pool = jload(POOL_CACHE)
# Filter to only pre-sampled formulas (vals != None) for stable training
pool = [f for f in pool if f['vals'] is not None]
print(f"Pool: {len(pool)} pre-sampled formulas  device={DEVICE}")

# ── Init models + optimisers ──────────────────────────────────────────────────

D     = GATDiscriminator().to(DEVICE)
G     = Generator().to(DEVICE)
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
bce   = nn.BCEWithLogitsLoss()
print(f"D params: {sum(p.numel() for p in D.parameters()):,}")
print(f"G params: {sum(p.numel() for p in G.parameters()):,}\n")

# ── Training ──────────────────────────────────────────────────────────────────

rng     = np.random.default_rng(SEED)
indices = list(range(len(pool)))

# ── Phase 1: Supervised warm-up (D only, no G) ───────────────────────────────
print(f"Phase 1: supervised warm-up ({PRETRAIN_EPOCHS} epochs)...")
for epoch in range(1, PRETRAIN_EPOCHS + 1):
    rng.shuffle(indices)
    losses = []
    t0 = time.time()
    for batch_start in range(0, len(indices) - BATCH_SIZE, BATCH_SIZE):
        batch_idx = indices[batch_start: batch_start + BATCH_SIZE]
        n_conf = int(rng.integers(4, 20)); n_rand = int(rng.integers(2, 8))
        graphs_list, label_list = [], []
        for ii in batch_idx:
            result = make_example(pool[ii], pool, ii, rng, n_conf, n_rand)
            if result is None: continue
            feat_mat, labels, col_names, _ = result
            if feat_mat.shape[0] < 2: continue
            ei = build_graph(col_names, feat_mat, feat_mat.shape[0])
            graphs_list.append(Data(x=torch.tensor(feat_mat), edge_index=ei))
            label_list.append(torch.tensor(labels, dtype=torch.float32))
        if not graphs_list: continue
        batch = Batch.from_data_list(graphs_list).to(DEVICE)
        labels_cat = torch.cat(label_list).to(DEVICE)
        D.train(); opt_D.zero_grad()
        logits = D(batch.x, batch.edge_index)
        loss   = bce(logits, labels_cat)
        loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt_D.step()
        losses.append(loss.item())
    print(f"  Pretrain {epoch}/{PRETRAIN_EPOCHS}  D_loss={np.mean(losses):.4f}  t={time.time()-t0:.1f}s", flush=True)

torch.save({'D': D.state_dict(), 'G': G.state_dict(), 'epoch': 0},
           OUT_DIR / 'gan_gat_pretrained.pt')
print(f"Pretrain done → gan_gat_pretrained.pt\n")

# ── Phase 2: GAN adversarial training ─────────────────────────────────────────
print(f"Phase 2: GAN adversarial training ({EPOCHS} epochs)...")
for epoch in range(1, EPOCHS + 1):
    rng.shuffle(indices)
    d_losses, g_losses = [], []
    t0 = time.time()

    for batch_start in range(0, len(indices) - BATCH_SIZE, BATCH_SIZE):
        batch_idx = indices[batch_start: batch_start + BATCH_SIZE]
        n_conf = int(rng.integers(4, 20))
        n_rand = int(rng.integers(2, 8))

        graphs_list, label_list, ctx_list = [], [], []
        for ii in batch_idx:
            result = make_example(pool[ii], pool, ii, rng, n_conf, n_rand)
            if result is None: continue
            feat_mat, labels, col_names, ctx = result
            if feat_mat.shape[0] < 2: continue
            ei = build_graph(col_names, feat_mat, feat_mat.shape[0])
            graphs_list.append(Data(x=torch.tensor(feat_mat), edge_index=ei))
            label_list.append(torch.tensor(labels, dtype=torch.float32))
            ctx_list.append(torch.tensor(ctx))

        if not graphs_list: continue
        batch      = Batch.from_data_list(graphs_list).to(DEVICE)
        labels_cat = torch.cat(label_list).to(DEVICE)
        ctx_cat    = torch.stack(ctx_list).to(DEVICE)       # (B, FEAT_DIM)
        B          = len(graphs_list)

        # ── Discriminator step ────────────────────────────────────────────────
        D.train(); G.eval(); opt_D.zero_grad()

        real_logits = D(batch.x, batch.edge_index)
        loss_d_real = bce(real_logits, labels_cat)

        with torch.no_grad():
            noise   = torch.randn(B * N_GAN_GEN, NOISE_DIM, device=DEVICE)
            mode    = torch.randint(0, 2, (B * N_GAN_GEN,), device=DEVICE)
            ctx_exp = ctx_cat.unsqueeze(1).expand(-1, N_GAN_GEN, -1).reshape(-1, FEAT_DIM)
            fake_x  = G(noise, mode, ctx_exp)

        fake_logits = D.forward_flat(fake_x)
        loss_d_fake = bce(fake_logits, torch.zeros(fake_x.shape[0], device=DEVICE))
        loss_d      = loss_d_real + 0.5 * loss_d_fake
        loss_d.backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0)
        opt_D.step()
        d_losses.append(loss_d.item())

        # ── Generator step ────────────────────────────────────────────────────
        D.eval(); G.train(); opt_G.zero_grad()

        noise   = torch.randn(B * N_GAN_GEN, NOISE_DIM, device=DEVICE)
        mode    = torch.randint(0, 2, (B * N_GAN_GEN,), device=DEVICE)
        fake_x  = G(noise, mode, ctx_exp.detach())
        # G wants D to output HIGH score (fool D into thinking it's relevant)
        fake_logits = D.forward_flat(fake_x)
        loss_g  = bce(fake_logits, torch.ones(fake_x.shape[0], device=DEVICE))
        loss_g.backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_G.step()
        g_losses.append(loss_g.item())

    print(f"Epoch {epoch:3d}/{EPOCHS}  "
          f"D={np.mean(d_losses):.4f}  G={np.mean(g_losses):.4f}  "
          f"t={time.time()-t0:.1f}s", flush=True)

    if epoch % 5 == 0:
        torch.save({'D': D.state_dict(), 'G': G.state_dict(), 'epoch': epoch},
                   OUT_DIR / f'gan_gat_ep{epoch:03d}.pt')
        # Quick inline eval on pool subset
        D.eval()
        precisions, recalls = [], []
        eval_rng = np.random.default_rng(SEED + epoch)
        for fi in eval_rng.choice(len(pool), size=min(100, len(pool)), replace=False):
            result = make_example(pool[fi], pool, fi, eval_rng, 10, 4)
            if result is None: continue
            feat_mat, gt_labels, col_names, _ = result
            if feat_mat.shape[0] < 2: continue
            ei = build_graph(col_names, feat_mat, feat_mat.shape[0]).to(DEVICE)
            x  = torch.tensor(feat_mat).to(DEVICE)
            with torch.no_grad():
                probs = torch.sigmoid(D(x, ei)).cpu().numpy()
            pred = probs >= 0.25
            tp = float(np.sum(pred & gt_labels.astype(bool)))
            fp = float(np.sum(pred & ~gt_labels.astype(bool)))
            fn = float(np.sum(~pred & gt_labels.astype(bool)))
            if tp + fp > 0: precisions.append(tp / (tp + fp))
            if tp + fn > 0: recalls.append(tp / (tp + fn))
        if precisions:
            print(f"  → inline eval: prec={np.mean(precisions):.3f}  recall={np.mean(recalls):.3f}  "
                  f"[RF v2_ft target: prec=0.945 recall=0.964]", flush=True)
        D.train()

torch.save({'D': D.state_dict(), 'G': G.state_dict(),
            'feat_dim': FEAT_DIM, 'noise_dim': NOISE_DIM,
            'hidden_dim': HIDDEN_DIM, 'gat_heads': GAT_HEADS,
            'n_pool': len(pool), 'version': 'gan_gat_v2_clean'},
           OUT_DIR / 'gan_gat_final.pt')
print(f"\nSaved -> {OUT_DIR}/gan_gat_final.pt")
