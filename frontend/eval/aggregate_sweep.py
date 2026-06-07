"""Aggregate the multi-seed and distractor-sweep JSONs into summary stats."""
import json, math, glob
from pathlib import Path
import numpy as np

ROOT = Path("data/results")
SWEEP = ROOT/"sweep"
HEADLINE = ROOT/"pysr_frontend_3way.json"  # seed=42, n_dist=20

def load_run(path):
    d = json.load(open(path))
    n = len(d)
    def cnt(k): return sum(1 for r in d if r.get(k))
    def mr2(k):
        v = [max(r[k], -1.0) if (r.get(k) is not None and math.isfinite(r[k])) else -1.0 for r in d]
        return sum(v)/len(v)
    return {
        "n": n,
        "full_exact_pct": 100*cnt('full_exact')/n,
        "var_exact_pct":  100*cnt('var_exact')/n,
        "varop_exact_pct":100*cnt('varop_exact')/n,
        "full_r2":  mr2('full_r2'),
        "var_r2":   mr2('var_r2'),
        "varop_r2": mr2('varop_r2'),
        "full_t":   float(np.mean([r['full_t'] for r in d])),
        "var_t":    float(np.mean([r['var_t']  for r in d])),
        "recall":   float(np.mean([r['recall'] for r in d])),
        "sel_cols": float(np.mean([r['sel_cols'] for r in d])),
        "full_cols":float(np.mean([r['full_cols'] for r in d])),
    }

# Multi-seed at d=20 (variance on headline)
seed_runs = {42: load_run(HEADLINE)}
for s in [43, 44]:
    p = SWEEP/f"3way_seed{s}_d20.json"
    if p.exists(): seed_runs[s] = load_run(p)
seeds_summary = {}
keys = ["full_exact_pct","var_exact_pct","varop_exact_pct",
        "full_r2","var_r2","varop_r2","full_t","var_t","recall","sel_cols"]
for k in keys:
    vals = np.array([seed_runs[s][k] for s in sorted(seed_runs)])
    seeds_summary[k] = {"mean": float(vals.mean()),
                        "std":  float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                        "vals": [float(x) for x in vals]}
print("=== seed variance at n_dist=20 (n_seeds = {}) ===".format(len(seed_runs)))
for k in keys:
    s = seeds_summary[k]
    print(f"  {k:18s}  mean={s['mean']:7.3f}  std={s['std']:6.3f}  vals={s['vals']}")

# Distractor sweep across all available seeds
def maybe(path):
    p = SWEEP/path
    return load_run(p) if p.exists() else None
dist_by_seed = {42: {20: load_run(HEADLINE)},
                43: {20: maybe("3way_seed43_d20.json")},
                44: {20: maybe("3way_seed44_d20.json")}}
for s in [42, 43, 44]:
    for d in [5, 10, 30]:
        r = maybe(f"3way_seed{s}_d{d}.json")
        if r is not None: dist_by_seed[s][d] = r

print(f"\n=== distractor sweep (multi-seed if available) ===")
print(f"{'n_dist':>7} {'seeds':>6} {'full% (mean+-std)':>22} {'vars% (mean+-std)':>22} {'recall':>7}")
dist_multi = {}
for d in [5, 10, 20, 30]:
    runs = [dist_by_seed[s][d] for s in [42,43,44] if d in dist_by_seed[s]]
    if not runs: continue
    fs = np.array([r['full_exact_pct'] for r in runs])
    vs = np.array([r['var_exact_pct']  for r in runs])
    fr = np.array([r['full_r2'] for r in runs]); vr = np.array([r['var_r2'] for r in runs])
    dist_multi[str(d)] = {
        "n_seeds": len(runs),
        "full_exact_pct_mean": float(fs.mean()), "full_exact_pct_std": float(fs.std(ddof=1)) if len(fs)>1 else 0,
        "var_exact_pct_mean":  float(vs.mean()), "var_exact_pct_std":  float(vs.std(ddof=1)) if len(vs)>1 else 0,
        "full_r2_mean": float(fr.mean()), "full_r2_std": float(fr.std(ddof=1)) if len(fr)>1 else 0,
        "var_r2_mean":  float(vr.mean()), "var_r2_std":  float(vr.std(ddof=1)) if len(vr)>1 else 0,
        "recall_mean": float(np.mean([r['recall'] for r in runs])),
    }
    print(f"{d:>7} {len(runs):>6}  {fs.mean():6.1f} +- {fs.std(ddof=1) if len(fs)>1 else 0:5.2f}     "
          f"  {vs.mean():6.1f} +- {vs.std(ddof=1) if len(vs)>1 else 0:5.2f}   "
          f"  {np.mean([r['recall'] for r in runs]):.3f}")

# Also: 80-Feynman scale-up if available
scale_up = None
n80_path = ROOT/"pysr_frontend_3way_n80.json"
if n80_path.exists():
    scale_up = load_run(n80_path)
    print(f"\n=== Feynman 80-task scale-up (seed=42, d=20) ===")
    print(f"  n={scale_up['n']}  full={scale_up['full_exact_pct']:.0f}%  "
          f"vars={scale_up['var_exact_pct']:.0f}%  +ops={scale_up['varop_exact_pct']:.0f}%  "
          f"full R2={scale_up['full_r2']:.3f}  var R2={scale_up['var_r2']:.3f}  recall={scale_up['recall']:.3f}")

# gplearn multi-seed if available
gp_runs = {42: ROOT/"gplearn_backend_3way.json",
           43: ROOT/"gplearn_backend_3way_seed43.json",
           44: ROOT/"gplearn_backend_3way_seed44.json"}
gp_summary = {}
for s, p in gp_runs.items():
    if not p.exists(): continue
    d = json.load(open(p)); n = len(d)
    fe = sum(1 for r in d if r.get('full_exact'))
    ve = sum(1 for r in d if r.get('var_exact'))
    fr = [max(r['full_r2'], -1.0) if math.isfinite(r['full_r2']) else -1.0 for r in d]
    vr = [max(r['var_r2'],  -1.0) if math.isfinite(r['var_r2'])  else -1.0 for r in d]
    gp_summary[str(s)] = {"n":n, "full_exact_pct":100*fe/n, "var_exact_pct":100*ve/n,
                           "full_r2":float(np.mean(fr)), "var_r2":float(np.mean(vr))}
if len(gp_summary) > 1:
    print(f"\n=== gplearn multi-seed ===")
    print(f"{'seed':>5} {'full%':>7} {'vars%':>7} {'fullR2':>8} {'varsR2':>8}")
    for s,r in gp_summary.items():
        print(f"{s:>5} {r['full_exact_pct']:>7.0f} {r['var_exact_pct']:>7.0f} {r['full_r2']:>8.3f} {r['var_r2']:>8.3f}")
    fes = [r['full_exact_pct'] for r in gp_summary.values()]
    ves = [r['var_exact_pct']  for r in gp_summary.values()]
    frs = [r['full_r2'] for r in gp_summary.values()]
    vrs = [r['var_r2'] for r in gp_summary.values()]
    gp_summary["__mean_std__"] = {
        "n_seeds": len(gp_summary),
        "full_exact_pct_mean": float(np.mean(fes)), "full_exact_pct_std": float(np.std(fes, ddof=1)) if len(fes)>1 else 0,
        "var_exact_pct_mean":  float(np.mean(ves)), "var_exact_pct_std":  float(np.std(ves, ddof=1)) if len(ves)>1 else 0,
        "full_r2_mean": float(np.mean(frs)), "full_r2_std": float(np.std(frs, ddof=1)) if len(frs)>1 else 0,
        "var_r2_mean":  float(np.mean(vrs)), "var_r2_std":  float(np.std(vrs, ddof=1)) if len(vrs)>1 else 0,
    }
    print(f"  MEAN  full={np.mean(fes):5.1f}+-{np.std(fes,ddof=1) if len(fes)>1 else 0:.1f}  "
          f"vars={np.mean(ves):5.1f}+-{np.std(ves,ddof=1) if len(ves)>1 else 0:.1f}")

# legacy distractor_sweep_seed42 for back-compat with existing figure
dist_summary = {str(d): dist_by_seed[42][d] for d in dist_by_seed[42]}
out = {"headline_d20_seeds": seeds_summary,
       "distractor_sweep_seed42": dist_summary,
       "distractor_sweep_multiseed": dist_multi,
       "scale_up_n80": scale_up,
       "gplearn_multiseed": gp_summary,
       "n_seeds": len(seed_runs)}
Path("data/results/multiseed_distractor_aggregate.json").write_text(json.dumps(out, indent=2))
print("\nSaved -> data/results/multiseed_distractor_aggregate.json")
