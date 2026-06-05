"""
Select formulas from v3 graph — no sympy sampling, just filter and save expr strings.
Sampling happens on-demand during training.
"""
import sys, json, time
sys.path.insert(0, 'src')

from pathlib import Path
from joblib import dump

KG_V3        = Path('data/kg_v3/FINAL_nodes_v3.jsonl')
OUT          = Path('models/formula_pool_v3_32k.joblib')
OUT.parent.mkdir(exist_ok=True)
TARGET       = 32000
MAX_EXPR_LEN = 150

print(f"Selecting formulas from v3 (target={TARGET}, max_expr_len={MAX_EXPR_LEN})...")
t0 = time.time()

formulas = []
scanned = 0

with open(KG_V3, encoding='utf-8', errors='ignore') as f:
    for line in f:
        if len(formulas) >= TARGET:
            break
        if not line.strip():
            continue
        r = json.loads(line)
        scanned += 1

        if r.get('ds3_verdict') != 'ACCEPT':
            continue
        if r.get('ds3_physical') not in ('YES', True, 'true'):
            continue

        expr = str(r.get('expr', ''))
        if not expr or len(expr) < 4 or len(expr) > MAX_EXPR_LEN:
            continue

        formulas.append({'truth': expr, 'vals': None,
                         'target': None, 'true_vars': None})

        if len(formulas) % 5000 == 0:
            print(f"  {len(formulas):,}/{TARGET}  scanned={scanned:,}", flush=True)

print(f"\nSelected {len(formulas):,} formulas from {scanned:,} lines in {time.time()-t0:.1f}s")
dump(formulas, OUT)
print(f"Saved -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
