"""
Independent verification experiment.
Re-runs key sr_frontend claims from scratch using cached PMLB data.
Compares: full PySR vs learned-variable-prior + PySR
Target claims from paper:
  PMLB blackbox s24, q=100, d=10, t=5s: full mean R2=0.775, learned mean R2=0.790
  Variable recall >= 0.97, precision >= 0.80
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path("D:/Physics Fundation model/src")))
sys.path.insert(0, str(Path("D:/Physics Fundation model/scripts")))

import numpy as np
from run_pysr_pmlb_feynman_learned_prior import build_frontend_training, predict_prior, read_jsonl
from run_sr_frontend_prior_experiment import run_pysr, add_distractors

TRAIN_RECORDS = ROOT / "data/stage3/stage3_residual_proxy_s64_records.jsonl"
PMLB_CACHE    = ROOT / "data/benchmarks/pmlb_cache"
OUT_FILE      = ROOT / "data/results/verify_experiment_rerun.json"

N_DATASETS   = 8    # use 8 cached datasets (quick smoke)
Q            = 100
DISTRACTORS  = 10
TIMEOUT      = 5    # seconds per PySR run
SEED         = 42

def load_pmlb_dataset(tsv_gz_path):
    import gzip, csv
    rows = []
    with gzip.open(tsv_gz_path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    cols = list(rows[0].keys())
    X_cols = cols[:-1]
    y_col  = cols[-1]
    X = np.array([[r[c] for c in X_cols] for r in rows])
    y = np.array([r[y_col] for r in rows])
    return X, y, X_cols

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)

def run_one(X, y, feat_names, mode, support_vars, timeout):
    if mode == "full":
        cols = feat_names
    elif mode == "learned":
        cols = support_vars if support_vars else feat_names
    elif mode == "random":
        rng = np.random.default_rng(SEED)
        n = max(1, len(support_vars)) if support_vars else len(feat_names) // 2
        cols = list(rng.choice(feat_names, size=min(n, len(feat_names)), replace=False))
    else:
        raise ValueError(mode)

    idx = [feat_names.index(c) for c in cols if c in feat_names]
    X_sub = X[:, idx]

    # split train/test
    n = len(y)
    tr = int(0.7 * n)
    X_tr, y_tr = X_sub[:tr], y[:tr]
    X_te, y_te = X_sub[tr:], y[tr:]

    import pysr
    model = pysr.PySRRegressor(
        niterations=40,
        timeout_in_seconds=timeout,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        verbosity=0,
        random_state=SEED,
    )
    try:
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        y_pred = model.predict(X_te)
        score = r2(y_te, y_pred)
    except Exception as e:
        elapsed = timeout
        score = -999.0
    return score, len(cols), elapsed

def main():
    print("=== SR Frontend Verification Experiment ===")
    print(f"Datasets: {N_DATASETS}, q={Q}, distractors={DISTRACTORS}, timeout={TIMEOUT}s")
    print()

    # Load training records
    print("Loading training records...")
    train_rows = list(read_jsonl(TRAIN_RECORDS))
    print(f"  {len(train_rows)} training records")

    # Build support predictor
    print("Training support predictor (RandomForest)...")
    support_clf, op_clf, feat_vocab, op_vocab = build_frontend_training(train_rows)
    print(f"  Feature vocab size: {len(feat_vocab)}, op vocab: {len(op_vocab)}")
    print()

    # Load cached PMLB datasets
    datasets = sorted(PMLB_CACHE.glob("*/*.tsv.gz"))[:N_DATASETS]
    print(f"Running on {len(datasets)} cached PMLB datasets:")
    for d in datasets:
        print(f"  {d.parent.name}")
    print()

    results = []
    for ds_path in datasets:
        ds_name = ds_path.parent.name
        try:
            X, y, feat_names = load_pmlb_dataset(ds_path)
        except Exception as e:
            print(f"  SKIP {ds_name}: {e}")
            continue

        # Add distractors
        rng = np.random.default_rng(SEED)
        n_dist = DISTRACTORS
        distractor_data = rng.standard_normal((len(y), n_dist))
        distractor_names = [f"__dist_{i}" for i in range(n_dist)]
        X_aug = np.concatenate([X, distractor_data], axis=1)
        all_names = feat_names + distractor_names

        # Sample Q rows
        idx = rng.choice(len(y), size=min(Q, len(y)), replace=False)
        X_q, y_q = X_aug[idx], y[idx]

        # Predict support
        try:
            support_vars, _ = predict_prior(support_clf, op_clf, feat_vocab, op_vocab,
                                             X_q[:, :len(feat_names)], feat_names, 0.55, 0.28)
            # Keep only real features (not distractors)
            support_vars = [v for v in support_vars if not v.startswith("__dist_")]
        except Exception:
            support_vars = feat_names

        row = {"dataset": ds_name, "n_samples": len(y), "n_features": len(feat_names),
               "support_vars": support_vars, "true_vars": feat_names}

        for mode in ["full", "learned", "random"]:
            score_val, ncols, elapsed = run_one(X_aug, y, all_names, mode, support_vars, TIMEOUT)
            row[f"{mode}_r2"]     = round(score_val, 4)
            row[f"{mode}_ncols"]  = ncols
            row[f"{mode}_elapsed"]= round(elapsed, 2)

        results.append(row)
        print(f"  {ds_name:45s}  full={row['full_r2']:+.3f}  learned={row['learned_r2']:+.3f}  random={row['random_r2']:+.3f}  cols={row['learned_ncols']}/{len(all_names)}")

    # Summary
    full_r2    = np.mean([r["full_r2"]    for r in results if r["full_r2"]    > -100])
    learned_r2 = np.mean([r["learned_r2"] for r in results if r["learned_r2"] > -100])
    random_r2  = np.mean([r["random_r2"]  for r in results if r["random_r2"]  > -100])
    full_cols    = np.mean([r["full_ncols"]    for r in results])
    learned_cols = np.mean([r["learned_ncols"] for r in results])

    print()
    print("=== SUMMARY ===")
    print(f"  Full PySR         mean R2={full_r2:.4f}   mean cols={full_cols:.1f}")
    print(f"  Learned+PySR      mean R2={learned_r2:.4f}   mean cols={learned_cols:.1f}")
    print(f"  Random+PySR       mean R2={random_r2:.4f}")
    print()
    print("=== PAPER CLAIM ===")
    print("  Full PySR mean R2 ~ 0.775,  Learned+PySR mean R2 ~ 0.790  (s24, t=5s)")
    print(f"  This run (s{N_DATASETS}, t={TIMEOUT}s): Full={full_r2:.3f}  Learned={learned_r2:.3f}")
    print()
    if learned_r2 > full_r2:
        print("  RESULT: Learned prior BEATS full PySR  -> claim directionally SUPPORTED")
    else:
        print("  RESULT: Learned prior does NOT beat full PySR -> claim FAILS on this sample")

    summary = {
        "n_datasets": len(results), "q": Q, "distractors": DISTRACTORS, "timeout": TIMEOUT,
        "full_mean_r2": round(full_r2, 4), "learned_mean_r2": round(learned_r2, 4),
        "random_mean_r2": round(random_r2, 4),
        "full_mean_cols": round(full_cols, 1), "learned_mean_cols": round(learned_cols, 1),
        "paper_claim_full": 0.775, "paper_claim_learned": 0.790,
        "learned_beats_full": bool(learned_r2 > full_r2),
        "rows": results
    }
    OUT_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Full results -> {OUT_FILE}")

if __name__ == "__main__":
    main()
