"""
Support predictor v2:
- Physics confounders (variables from other formulas, range-sampled independently)
- 8 sampling distributions (uniform, gaussian, log-uniform, bimodal,
  exponential, beta, power-law, student-t) applied randomly to both
  true vars AND confounders so the model can't cheat on distribution shape
- Unit/semantic-class features from symbol_semantic_tags
"""
import sys, json, warnings, re, time, os
warnings.filterwarnings('ignore')

# Limit CPU to ~60%: lower process priority
try:
    p = os.getpid()
    import psutil
    proc = psutil.Process(p)
    proc.nice(10)   # lower niceness = lower priority
except Exception:
    pass
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from evaluate_stage8g_open_generation import sample_truth_points
from run_pysr_pmlb_feynman_learned_prior import infer_symbols, infer_ops, OP_ORDER

NODES_FILE   = Path('D:/Physics Fundation model/data/_FOUNDATION_TRAINING_DATASET_20260522/nodes_full.jsonl')
TAGS_FILE    = Path('D:/Physics Fundation model/data/symbol_semantic_tags_20260522.jsonl')
OUT_MODEL    = Path('models/support_predictor_v2_40k.joblib')
OUT_MODEL.parent.mkdir(exist_ok=True)

Q                 = 80
N_CONFOUNDER_MIN  = 2   # physics confounders range
N_CONFOUNDER_MAX  = 32
N_RANDOM_MIN      = 2   # pure random noise range
N_RANDOM_MAX      = 8
MAX_FORMULAS      = 6000
SEED              = 42

# ── Semantic class vocabulary ─────────────────────────────────────────────────
SEMANTIC_CLASSES = [
    'acceleration','action','angle','angular_frequency','angular_momentum',
    'area','capacitance','charge','charge_density','chemical_potential',
    'concentration','conductance','cross_section','current','current_density',
    'decay_rate','density_of_states','diffusion_coeff','dimensionless_ratio',
    'elastic_modulus','electric_field','electric_potential','energy_density',
    'energy_generic','energy_kinetic','energy_potential','energy_thermal',
    'entropy','force','frequency','fundamental_constant','impedance',
    'inductance','intensity','length','luminosity','magnetic_field',
    'magnetic_flux','mass','mass_density_generic','momentum','number_density',
    'other','power','pressure','resistance','specific_heat','spin','strain',
    'stress_tensor','surface_tension','temperature','time','torque',
    'velocity','viscosity','volume','wavelength','wavenumber',
]
CLASS_IDX = {c: i for i, c in enumerate(SEMANTIC_CLASSES)}
N_CLASSES  = len(SEMANTIC_CLASSES)   # 59 after dedup

def load_symbol_tags(path):
    tags = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            sym = r.get('symbol','')
            cls = r.get('semantic_class','other')
            conf = float(r.get('confidence', 0.5))
            if sym:
                tags[sym] = (cls, conf)
    return tags

SYMBOL_TAGS = load_symbol_tags(TAGS_FILE)

def unit_feature_vec(var_name):
    """Return a fixed-length unit/semantic vector for a variable name."""
    vec = np.zeros(N_CLASSES + 1, dtype=np.float32)  # +1 for confidence
    cls, conf = SYMBOL_TAGS.get(var_name, ('other', 0.0))
    idx = CLASS_IDX.get(cls, CLASS_IDX.get('other', N_CLASSES - 1))
    vec[idx] = 1.0
    vec[-1]  = conf
    return vec

# ── Rich sampling ─────────────────────────────────────────────────────────────

def resample(src_array, q, rng):
    """
    Re-sample q values using the source array's range+stats but a randomly
    chosen distribution so the model never learns dist-shape == signal.
    8 strategies, chosen uniformly at random.
    """
    lo  = float(np.min(src_array))
    hi  = float(np.max(src_array))
    mid = (lo + hi) / 2.0
    std = float(np.std(src_array)) + 1e-8
    if lo >= hi:
        lo, hi = mid - 1.0, mid + 1.0

    dist = int(rng.integers(0, 8))

    if dist == 0:                          # uniform
        s = rng.uniform(lo, hi, q)

    elif dist == 1:                        # Gaussian
        s = np.clip(rng.normal(mid, std, q), lo, hi)

    elif dist == 2:                        # log-uniform (positive range only)
        if lo > 0:
            s = np.exp(rng.uniform(np.log(lo + 1e-12), np.log(hi + 1e-12), q))
        else:
            s = rng.uniform(lo, hi, q)

    elif dist == 3:                        # bimodal (two clusters)
        w  = (hi - lo) * 0.15
        s  = np.where(rng.random(q) < 0.5,
                      rng.normal(lo + w, w, q),
                      rng.normal(hi - w, w, q))
        s  = np.clip(s, lo, hi)

    elif dist == 4:                        # exponential (shifted)
        scale = max(std, (hi - lo) / 4)
        s  = np.clip(lo + rng.exponential(scale, q), lo, hi)

    elif dist == 5:                        # beta — varied shapes
        a, b = rng.uniform(0.5, 4.0), rng.uniform(0.5, 4.0)
        s  = lo + (hi - lo) * rng.beta(a, b, q)

    elif dist == 6:                        # power-law (log-space uniform)
        exp = rng.uniform(0.2, 3.0)
        t   = rng.uniform(0, 1, q) ** exp
        s   = lo + (hi - lo) * t

    else:                                  # Student-t (heavy tails)
        df  = rng.uniform(1.5, 5.0)
        s   = np.clip(mid + std * rng.standard_t(df, q), lo, hi)

    return s.astype(np.float64)

# ── Functional dependence label ──────────────────────────────────────────────

def is_functionally_related(col_vals, y_vals, threshold=0.55):
    """
    Return 1 if col_vals has a detectable functional relationship with y_vals
    under any of several simple transformations.
    This replaces name-matching: derived quantities (kT, 1/r, x²) that are
    causally related to the target will be labelled relevant (1), even if their
    name does not appear verbatim in the formula.
    True noise = statistically independent from target → label 0.
    """
    eps = 1e-8
    x = np.asarray(col_vals, dtype=np.float64)
    y = np.asarray(y_vals,   dtype=np.float64)

    if np.std(x) < eps or np.std(y) < eps:
        return 0

    transforms = [
        x,                                    # linear
        x ** 2,                               # quadratic
        np.sqrt(np.abs(x)),                   # sqrt
        np.log(np.abs(x) + eps),              # log
        1.0 / (np.abs(x) + eps),              # inverse
        x ** 3,                               # cubic
        np.exp(np.clip(x, -10, 10)),          # exp (clipped)
        np.abs(x),                            # abs
    ]
    for t in transforms:
        if np.std(t) < eps: continue
        try:
            c = float(np.corrcoef(t, y)[0, 1])
            if np.isfinite(c) and abs(c) >= threshold:
                return 1
        except Exception:
            pass
    return 0

# ── Column features (statistical + unit) ─────────────────────────────────────

def column_features(col_vals, y_vals, var_name=''):
    """Statistical features + unit/semantic one-hot."""
    x = np.asarray(col_vals, dtype=np.float64)
    y = np.asarray(y_vals,   dtype=np.float64)
    eps = 1e-8

    # ---- statistical features (14) ----
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > eps else 0.0
    corr = 0.0 if not np.isfinite(corr) else corr

    sx, sy_ = np.std(x) + eps, np.std(y) + eps
    log_x = np.log(np.abs(x) + eps)
    log_y = np.log(np.abs(y) + eps)
    log_corr = float(np.corrcoef(log_x, log_y)[0, 1]) if np.std(log_x) > eps else 0.0
    log_corr = 0.0 if not np.isfinite(log_corr) else log_corr

    inv_corr = float(np.corrcoef(1.0 / (np.abs(x) + eps), y)[0, 1])
    inv_corr = 0.0 if not np.isfinite(inv_corr) else inv_corr

    sq_corr  = float(np.corrcoef(x ** 2, y)[0, 1])
    sq_corr  = 0.0 if not np.isfinite(sq_corr) else sq_corr

    var_ratio = float(sx / sy_)
    skew_x = float(np.mean(((x - np.mean(x)) / sx) ** 3))
    kurt_x = float(np.mean(((x - np.mean(x)) / sx) ** 4)) - 3.0
    skew_x = np.clip(skew_x, -10, 10)
    kurt_x = np.clip(kurt_x, -10, 10)

    sign_agree = float(np.mean(np.sign(x - np.median(x)) == np.sign(y - np.median(y))))
    rank_corr  = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    rank_corr  = 0.0 if not np.isfinite(rank_corr) else rank_corr

    frac_pos   = float(np.mean(x > 0))
    has_neg    = float(np.any(x < 0))
    dyn_range  = float(np.log10(np.max(np.abs(x)) / (np.min(np.abs(x)) + eps) + 1))

    stat_feats = np.array([
        corr, log_corr, inv_corr, sq_corr,
        var_ratio, skew_x, kurt_x,
        sign_agree, rank_corr,
        frac_pos, has_neg, dyn_range,
        float(np.mean(x)), float(np.std(x) / (np.abs(np.mean(x)) + eps)),
    ], dtype=np.float32)
    stat_feats = np.nan_to_num(stat_feats, nan=0.0, posinf=10.0, neginf=-10.0)
    stat_feats = np.clip(stat_feats, -10.0, 10.0)

    # ---- unit/semantic features ----
    unit_feats = unit_feature_vec(var_name)

    return np.concatenate([stat_feats, unit_feats])

def task_features(vals, target, all_cols):
    """Global task features (unchanged from original)."""
    y = vals[target]
    n_cols = len(all_cols)
    col_stds = [np.std(vals[c]) for c in all_cols]
    feats = np.array([
        n_cols,
        float(np.std(y)),
        float(np.mean(col_stds)),
        float(np.max(col_stds)),
        float(np.min(col_stds)),
        float(len(y)),
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=100.0, neginf=-100.0).tolist()

# ── Formula pool ─────────────────────────────────────────────────────────────

def load_formula_pool(nodes_path, max_formulas, q, seed):
    rng = np.random.default_rng(seed)
    pool = []
    print(f"Building formula pool (max={max_formulas}, q={q})...")
    with open(nodes_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            truth = r.get('canonical_template', '')
            if not truth or len(truth) < 4: continue

            vals, err = sample_truth_points(truth, q, rng)
            if vals is None or len(vals) < 2: continue

            m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth)
            target = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
            true_vars = set(vals.keys()) - {target}
            if not true_vars: continue

            pool.append({
                'truth': truth, 'vals': vals,
                'target': target, 'true_vars': true_vars,
            })
            if len(pool) % 1000 == 0:
                print(f"  {len(pool):,} formulas...", flush=True)
            if len(pool) >= max_formulas:
                break
    print(f"Pool: {len(pool):,} formulas\n")
    return pool

# ── Confounders ───────────────────────────────────────────────────────────────

def pick_confounders(pool, formula_idx, true_vars, n, rng, q, y_vals=None, func_thresh=0.55):
    """
    Pick variables from other formulas, independently range-sampled.
    If y_vals is provided, skip any candidate that turns out to be
    functionally related to the current target — derived quantities
    simply don't appear as confounders at all.
    """
    result = {}
    tried  = 0
    n_pool = len(pool)
    while len(result) < n and tried < n_pool * 3:
        other_idx = int(rng.integers(0, n_pool))
        if other_idx == formula_idx:
            tried += 1; continue
        other = pool[other_idx]
        if other['vals'] is None: tried += 1; continue
        fresh = set(other['vals'].keys()) - true_vars - {other['target']}
        for v in sorted(fresh):
            if v in result or len(result) >= n: break
            candidate = resample(other['vals'][v], q, rng)
            # Skip if functionally related to current target (e.g. kT when T is true var)
            if y_vals is not None and is_functionally_related(candidate, y_vals, func_thresh):
                continue
            result[v] = candidate
        tried += 1
    return result

# ── Training data ─────────────────────────────────────────────────────────────

def build_training(pool, n_random, seed):
    rng = np.random.default_rng(seed)
    support_x, support_y = [], []
    op_x, op_y = [], []

    for idx, formula in enumerate(pool):
        truth = formula['truth']
        # Lazy sampling: v3 pool stores only expr strings
        if formula['vals'] is None:
            try:
                vals, _ = sample_truth_points(truth, q, rng)
            except Exception:
                continue
            if vals is None or len(vals) < 2:
                continue
            m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', truth)
            target = m.group(1) if m and m.group(1) in vals else list(vals.keys())[0]
            true_vars = set(vals.keys()) - {target}
            if not true_vars:
                continue
        else:
            vals      = formula['vals']
            target    = formula['target']
            true_vars = formula['true_vars']
        q         = len(vals[target])

        # Re-sample TRUE variables with a random distribution each time
        aug = {target: vals[target]}
        for v in true_vars:
            aug[v] = resample(vals[v], q, rng)

        # Randomly pick how many confounders/noise for this example (4-40 total)
        n_conf = int(rng.integers(N_CONFOUNDER_MIN, N_CONFOUNDER_MAX + 1))
        n_rand = int(rng.integers(N_RANDOM_MIN,     N_RANDOM_MAX + 1))

        # Simple random noise — original approach, fast and stable
        n_noise = n_conf + n_rand
        for i in range(n_noise):
            lo_r = float(rng.uniform(-5, 0))
            hi_r = float(rng.uniform(0, 5))
            aug[f'__d{i}'] = resample(np.array([lo_r, hi_r]), q, rng)

        all_cols = sorted(c for c in aug if c != target)
        y_vals   = aug[target]
        symbols  = infer_symbols(truth)

        for col in all_cols:
            feat = column_features(aug[col], y_vals, var_name=col)
            support_x.append(feat)
            # Original label: 1 if name in formula, 0 otherwise
            support_y.append(int(col in symbols and col in true_vars))

        op_x.append(task_features(aug, target, all_cols))
        ops = infer_ops(truth)
        op_y.append([int(op in ops) for op in OP_ORDER])

        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(pool)}  examples={len(support_x):,}  "
                  f"pos={np.mean(support_y):.2%}", flush=True)

    print(f"\nDone: {len(pool):,} formulas  {len(support_x):,} column examples")
    print(f"  positive rate: {np.mean(support_y):.2%}")
    return support_x, support_y, op_x, op_y

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
 t0 = time.time()
 POOL_CACHE = Path('models/formula_pool.joblib')  # 8k pre-sampled
 if POOL_CACHE.exists():
     from joblib import load as _jload
     pool = _jload(POOL_CACHE)
     print(f"Loaded pool: {len(pool)} formulas from {POOL_CACHE.name}")
 else:
     pool = load_formula_pool(NODES_FILE, MAX_FORMULAS, Q, SEED)

 print("Building training examples...")
 sx, sy, ox, oy = build_training(pool, 0, SEED)
 print(f"Data build: {time.time()-t0:.1f}s\n")

 feat_dim = len(sx[0])
 print(f"Feature dim: {feat_dim}  (14 statistical + {N_CLASSES+1} unit/semantic)")

 print("\nTraining support RF (n=200)...")
 t1 = time.time()
 N_JOBS = max(1, int(os.cpu_count() * 0.6))
 MAX_SUPPORT_EXAMPLES = 80000
 if len(sx) > MAX_SUPPORT_EXAMPLES:
     rng_sub = np.random.default_rng(SEED)
     idx = rng_sub.choice(len(sx), MAX_SUPPORT_EXAMPLES, replace=False)
     sx_fit = np.asarray([sx[i] for i in idx], dtype=np.float32)
     sy_fit = np.asarray([sy[i] for i in idx])
     print(f"Sub-sampled to {len(sx_fit):,} examples")
 else:
     sx_fit = np.asarray(sx, dtype=np.float32)
     sy_fit = np.asarray(sy)

 support_clf = RandomForestClassifier(
     n_estimators=200, min_samples_leaf=3,
     class_weight='balanced_subsample', random_state=SEED, n_jobs=N_JOBS)
 support_clf.fit(sx_fit, sy_fit)
 print(f"  {time.time()-t1:.1f}s")

 print("Training operator RF (n=200)...")
 t2 = time.time()
 op_clf = MultiOutputClassifier(RandomForestClassifier(
     n_estimators=200, min_samples_leaf=3,
     class_weight='balanced_subsample', random_state=SEED+1, n_jobs=N_JOBS))
 op_clf.fit(np.asarray(ox), np.asarray(oy))
 print(f"  {time.time()-t2:.1f}s")

 dump({
     'support_clf': support_clf, 'op_clf': op_clf,
     'n_train': len(pool), 'q': Q,
     'n_confounders_range': [N_CONFOUNDER_MIN, N_CONFOUNDER_MAX],
     'n_random_range': [N_RANDOM_MIN, N_RANDOM_MAX],
     'feat_dim': feat_dim, 'n_semantic_classes': N_CLASSES,
     'version': 'v2_physics_confounders_unit_features',
 }, OUT_MODEL)

 print(f"\nSaved -> {OUT_MODEL}  ({OUT_MODEL.stat().st_size/1e6:.1f} MB)")
 print(f"Total: {time.time()-t0:.1f}s")
