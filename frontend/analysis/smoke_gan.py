import sys, os
# Skip cooc build for smoke test
os.environ['SKIP_COOC'] = '1'
sys.path.insert(0, 'src')
import train_gan_gat as m
print('Import OK')
print(f'D params: {sum(p.numel() for p in m.D.parameters()):,}')
print(f'G params: {sum(p.numel() for p in m.G.parameters()):,}')
print(f'Pool size: {len(m.pool)}')
print(f'Device: {m.DEVICE}')
# Quick single batch test
import numpy as np, torch
rng = np.random.default_rng(0)
formula = m.pool[0]
feat_mat, labels, col_names, ctx = m.make_example(formula, m.pool, 0, rng, 6, 3)
print(f'Example: {len(col_names)} cols, {labels.sum():.0f} true vars, feat_dim={feat_mat.shape[1]}')
edge_index = m.build_graph(col_names, feat_mat, m.COOC)
print(f'Edges: {edge_index.shape[1]}')
x = torch.tensor(feat_mat, dtype=torch.float32).to(m.DEVICE)
logits = m.D(x, edge_index.to(m.DEVICE))
print(f'D output shape: {logits.shape}, range [{logits.min():.2f}, {logits.max():.2f}]')
print('Smoke test PASSED')
