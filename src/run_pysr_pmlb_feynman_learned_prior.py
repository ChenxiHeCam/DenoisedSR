from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_stage8g_decoder_fidelity import OPS, token_list
from evaluate_stage8g_open_generation import sample_truth_points
from physics_fm.data.simulators.generic_algebraic import _extract_symbols, _parse_canonical_form
from run_pysr_pmlb_feynman_baseline import make_values
from run_sr_frontend_prior_experiment import add_distractors, run_pysr, score


OP_ORDER = ["+", "-", "*", "/", "^", "sin", "cos", "exp", "log", "sqrt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a non-oracle data-visible support/operator prior on internal open records and evaluate PySR on PMLB Feynman."
    )
    parser.add_argument(
        "--train-records-jsonl",
        type=Path,
        default=ROOT / "artifacts/sr_frontend_prior/stage3_residual_proxy_32345_s64_records.jsonl",
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=ROOT / "artifacts/sr_frontend_prior/pmlb_feynman_open_formula_tasks_20260506.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/sr_frontend_prior/pysr_pmlb_feynman_learned_prior_s12_q100_d10_t20_20260506.json",
    )
    parser.add_argument("--max-train-records", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=12)
    parser.add_argument("--q", type=int, default=100)
    parser.add_argument("--heldout", type=int, default=200)
    parser.add_argument("--distractors", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--modes",
        action="append",
        choices=["full", "oracle_support", "learned_variables", "learned_ops", "learned_prior"],
        help="Modes to run. Defaults to all modes.",
    )
    parser.add_argument("--support-threshold", type=float, default=0.55)
    parser.add_argument("--max-prior-columns", type=int, default=8)
    parser.add_argument("--min-prior-columns", type=int, default=2)
    parser.add_argument("--op-threshold", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=20260506)
    return parser.parse_args()


def read_jsonl(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return 0.0
    a = a[finite]
    b = b[finite]
    if float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def safe_transform(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        if name == "raw":
            out = x
        elif name == "abs":
            out = np.abs(x)
        elif name == "square":
            out = x * x
        elif name == "cube":
            out = x * x * x
        elif name == "sqrt":
            out = np.sqrt(np.abs(x))
        elif name == "log":
            out = np.log1p(np.abs(x))
        elif name == "inv":
            out = 1.0 / np.where(np.abs(x) < 1.0e-6, np.nan, x)
        elif name == "sin":
            out = np.sin(x)
        elif name == "cos":
            out = np.cos(x)
        elif name == "exp":
            clipped = np.clip(x, -8.0, 8.0)
            out = np.exp(clipped)
        else:
            out = x
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


TRANSFORMS = ["raw", "abs", "square", "cube", "sqrt", "log", "inv", "sin", "cos", "exp"]


def column_features(x: np.ndarray, y: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    feats: list[float] = []
    for name in TRANSFORMS:
        tx = safe_transform(name, x)
        feats.append(abs(finite_corr(tx, y)))
        feats.append(abs(finite_corr(tx, np.abs(y))))
    for arr in (x, y):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feats.extend(
            [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.median(arr)),
            ]
        )
    feats.append(float(np.std(x) / (np.std(y) + 1.0e-8)))
    return [0.0 if not math.isfinite(v) else float(v) for v in feats]


def task_features(values: dict[str, np.ndarray], target: str, columns: list[str]) -> list[float]:
    y = np.asarray(values[target], dtype=np.float64)
    per_col = np.asarray([column_features(values[name], y) for name in columns], dtype=np.float64)
    if per_col.size == 0:
        return [0.0] * 20
    corr0 = per_col[:, 0]
    feats = [
        float(len(columns)),
        float(np.mean(y)),
        float(np.std(y)),
        float(np.min(y)),
        float(np.max(y)),
        float(np.max(corr0)),
        float(np.mean(corr0)),
        float(np.median(corr0)),
    ]
    for q in (0.25, 0.5, 0.75, 0.9):
        feats.append(float(np.quantile(corr0, q)))
    best_by_transform = per_col[:, : len(TRANSFORMS) * 2 : 2]
    feats.extend([float(v) for v in np.max(best_by_transform, axis=0)])
    return [0.0 if not math.isfinite(v) else float(v) for v in feats]


def make_xy(values: dict[str, np.ndarray], target: str, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values[target], dtype=np.float64).reshape(-1)
    x = np.asarray([[values[name][i] for name in columns] for i in range(len(y))], dtype=np.float64)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[finite], y[finite]


def infer_symbols(surface: str) -> set[str]:
    expr = _parse_canonical_form(surface)
    return set(_extract_symbols(expr)) if expr is not None else set()


def infer_ops(surface: str) -> set[str]:
    compact = surface.replace("**", "^")
    toks = set(token_list(compact))
    ops = {tok for tok in toks if tok in OPS or tok == "^"}
    for op in ["+", "-", "*", "/", "^"]:
        if op in compact:
            ops.add(op)
    for op in ["sin", "cos", "exp", "log", "sqrt"]:
        if re.search(rf"\b{op}\b", compact):
            ops.add(op)
    return ops


def target_from_surface(surface: str) -> str | None:
    if "=" not in surface:
        return None
    lhs = surface.split("=", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs):
        return lhs
    leading = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[-+]", lhs)
    if leading:
        return leading.group(1)
    return None


def build_frontend_training(rows: list[dict[str, Any]], args: argparse.Namespace, rng: np.random.Generator):
    support_x: list[list[float]] = []
    support_y: list[int] = []
    op_x: list[list[float]] = []
    op_y: list[list[int]] = []
    used = 0
    for row in rows:
        surface = str(row.get("truth_surface") or "")
        target = target_from_surface(surface)
        if target is None:
            continue
        values, err = sample_truth_points(surface, args.q, rng)
        if values is None or target not in values:
            continue
        values = add_distractors(values, args.distractors, rng)
        columns = sorted([name for name in values if name != target])
        symbols = infer_symbols(surface)
        y = values[target]
        for name in columns:
            support_x.append(column_features(values[name], y))
            support_y.append(int(name in symbols and name != target))
        op_x.append(task_features(values, target, columns))
        ops = infer_ops(surface)
        op_y.append([int(op in ops) for op in OP_ORDER])
        used += 1
    if not support_x or not op_x:
        raise RuntimeError("no usable frontend training rows")
    support_clf = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=1,
    )
    support_clf.fit(np.asarray(support_x), np.asarray(support_y))
    op_clf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=args.seed + 1,
            n_jobs=1,
        )
    )
    op_clf.fit(np.asarray(op_x), np.asarray(op_y))
    return support_clf, op_clf, {"usable_train_records": used, "support_examples": len(support_y), "op_examples": len(op_y)}


def positive_probability(clf: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(clf, MultiOutputClassifier):
        cols = []
        for estimator in clf.estimators_:
            probs = estimator.predict_proba(x)
            classes = list(getattr(estimator, "classes_", []))
            if 1 in classes:
                cols.append(probs[:, classes.index(1)])
            else:
                cols.append(np.zeros(x.shape[0], dtype=np.float64))
        return np.asarray(cols, dtype=np.float64).T
    probs = clf.predict_proba(x)
    classes = list(getattr(clf, "classes_", []))
    if 1 in classes:
        return probs[:, classes.index(1)]
    return np.zeros(x.shape[0], dtype=np.float64)


def predict_prior(
    support_clf: Any,
    op_clf: Any,
    values: dict[str, np.ndarray],
    target: str,
    all_columns: list[str],
    args: argparse.Namespace,
) -> tuple[list[str], set[str], dict[str, Any]]:
    y = values[target]
    x_support = np.asarray([column_features(values[name], y) for name in all_columns], dtype=np.float64)
    support_prob = positive_probability(support_clf, x_support)
    ranked = sorted(zip(all_columns, support_prob), key=lambda item: (-float(item[1]), item[0]))
    selected = [name for name, prob in ranked if float(prob) >= args.support_threshold]
    if len(selected) < args.min_prior_columns:
        selected = [name for name, _prob in ranked[: args.min_prior_columns]]
    selected = selected[: args.max_prior_columns]
    op_probs = positive_probability(op_clf, np.asarray([task_features(values, target, all_columns)], dtype=np.float64))[0]
    ops = {op for op, prob in zip(OP_ORDER, op_probs) if float(prob) >= args.op_threshold}
    ops.update({"+", "*"})
    if "-" in OP_ORDER and op_probs[OP_ORDER.index("-")] >= 0.15:
        ops.add("-")
    meta = {
        "support_probabilities": {name: float(prob) for name, prob in ranked},
        "op_probabilities": {op: float(prob) for op, prob in zip(OP_ORDER, op_probs)},
    }
    return selected, ops, meta


def run_task(
    task: dict[str, Any],
    support_clf: Any,
    op_clf: Any,
    args: argparse.Namespace,
    rng: np.random.Generator,
    idx: int,
) -> dict[str, Any]:
    train_values, train_error = make_values(task, args.q, rng)
    test_values, test_error = make_values(task, args.heldout, rng)
    base = {
        "idx": idx,
        "law_id": task["law_id"],
        "method": "pysr",
        "formula": task["formula"],
        "truth_surface": task["truth_surface"],
        "q": args.q,
        "distractors": args.distractors,
        "timeout_seconds": args.timeout_seconds,
    }
    if train_values is None or test_values is None:
        return {**base, "ok": False, "error": train_error or test_error}
    train_values = add_distractors(train_values, args.distractors, rng)
    test_values = add_distractors(test_values, args.distractors, rng)
    target = str(task["target"])
    oracle_columns = list(task["features"])
    full_columns = sorted([name for name in train_values if name != target])
    prior_columns, prior_ops, prior_meta = predict_prior(support_clf, op_clf, train_values, target, full_columns, args)
    mode_specs = {
        "full": (full_columns, None),
        "oracle_support": (oracle_columns, None),
        "learned_variables": (prior_columns, None),
        "learned_ops": (full_columns, prior_ops),
        "learned_prior": (prior_columns, prior_ops),
    }
    row = {**base, "ok": True, "learned_prior_meta": prior_meta}
    for mode in (args.modes or list(mode_specs)):
        columns, ops = mode_specs[mode]
        x_train, y_train = make_xy(train_values, target, columns)
        x_test, y_test = make_xy(test_values, target, columns)
        if len(x_train) == 0 or len(x_test) == 0:
            row[mode] = {"ok": False, "error": "empty_after_finite_filter"}
            continue
        try:
            started = time.time()
            result = run_pysr(
                x_train,
                y_train,
                x_test,
                y_test,
                columns,
                pred_ops=ops,
                timeout_seconds=args.timeout_seconds,
                seed=args.seed + idx,
            )
            row[mode] = {
                **result,
                "mode": mode,
                "ok": True,
                "elapsed_seconds": time.time() - started,
                "columns": columns,
                "column_count": len(columns),
                "full_column_count": len(full_columns),
                "oracle_columns": oracle_columns,
                "support_recall": len(set(columns) & set(oracle_columns)) / max(len(set(oracle_columns)), 1),
                "support_precision": len(set(columns) & set(oracle_columns)) / max(len(set(columns)), 1),
                "predicted_ops": sorted(ops) if ops is not None else None,
            }
        except Exception as exc:
            row[mode] = {"ok": False, "error": f"{type(exc).__name__}:{exc}"}
    return row


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    train_rows = read_jsonl(args.train_records_jsonl, args.max_train_records)
    support_clf, op_clf, train_summary = build_frontend_training(train_rows, args, rng)
    tasks = read_jsonl(args.tasks_jsonl, args.max_records)
    rows = []
    started = time.time()
    for idx, task in enumerate(tasks):
        row = run_task(task, support_clf, op_clf, args, rng, idx)
        rows.append(row)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary: dict[str, Any] = {
        "train_records_jsonl": str(args.train_records_jsonl),
        "tasks_jsonl": str(args.tasks_jsonl),
        "records": len(rows),
        "q": args.q,
        "heldout": args.heldout,
        "distractors": args.distractors,
        "timeout_seconds": args.timeout_seconds,
        "support_threshold": args.support_threshold,
        "op_threshold": args.op_threshold,
        "elapsed_seconds": time.time() - started,
        **train_summary,
    }
    for mode in (args.modes or ["full", "oracle_support", "learned_variables", "learned_ops", "learned_prior"]):
        ok = [row[mode] for row in rows if row.get("ok") and row.get(mode, {}).get("ok")]
        r2s = [float(item.get("score", {}).get("r2", float("nan"))) for item in ok]
        recalls = [float(item.get("support_recall", float("nan"))) for item in ok if mode not in {"full", "learned_ops"}]
        precisions = [float(item.get("support_precision", float("nan"))) for item in ok if mode not in {"full", "learned_ops"}]
        summary[f"{mode}_success_r2_0999"] = sum(1 for value in r2s if value >= 0.999)
        summary[f"{mode}_mean_r2"] = finite_mean(r2s)
        summary[f"{mode}_mean_columns"] = finite_mean([float(item.get("column_count", 0)) for item in ok])
        if recalls:
            summary[f"{mode}_support_recall"] = finite_mean(recalls)
            summary[f"{mode}_support_precision"] = finite_mean(precisions)
    payload = {"summary": summary, "rows": rows}
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
