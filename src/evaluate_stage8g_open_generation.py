from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import sympy
except ImportError:  # pragma: no cover
    sympy = None

from stage8g_generic_algebraic_loader import load_generic_algebraic


def normalize_formula(text: str) -> str:
    return "".join(text.lower().split())


_extract_symbols, _parse_canonical_form = load_generic_algebraic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open formula generation checker: strict parse plus residual on points sampled from the truth formula."
    )
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--catalog-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=2048)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--nmse-threshold", type=float, default=1.0e-3)
    parser.add_argument("--r2-threshold", type=float, default=0.999)
    parser.add_argument(
        "--composite-constraints",
        action="store_true",
        help="Evaluate multi-equation surfaces component-wise. Use for eliminate/recombination diagnostics, not legacy baselines.",
    )
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


def read_catalog_norms(path: Path | None) -> set[str]:
    if path is None:
        return set()
    out = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("canonical_formula") or row.get("formula_surface") or row.get("formula") or row.get("surface")
            if text:
                out.add(normalize_formula(str(text)))
    return out


def lhs_symbol(surface: str, symbols: set[str]) -> str | None:
    if "=" not in surface:
        return None
    lhs = surface.split("=", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lhs) and lhs in symbols:
        return lhs
    return None


def split_constraint_components(surface: str) -> tuple[list[str], str | None]:
    text = surface.strip()
    eliminate = None
    match = re.search(r"=>\s*eliminate\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        eliminate = match.group(1)
        text = text[: match.start()].strip()
    chunks = [chunk.strip() for chunk in re.split(r"\s*=\s*0\s*\+\s*", text) if chunk.strip()]
    if len(chunks) <= 1:
        return [text] if text else [], eliminate
    components = []
    for idx, chunk in enumerate(chunks):
        if idx < len(chunks) - 1:
            components.append(chunk + " = 0")
        elif re.search(r"=\s*0\s*$", chunk):
            components.append(chunk)
        else:
            components.append(chunk + " = 0")
    return components, eliminate


def parse_constraint_components(surface: str) -> tuple[list[tuple[str, Any, set[str]]], str | None]:
    components, eliminate = split_constraint_components(surface)
    parsed = []
    for component in components:
        expr = _parse_canonical_form(component)
        if expr is None:
            return [], eliminate
        parsed.append((component, expr, _extract_symbols(expr)))
    return parsed, eliminate


def choose_solve_symbol(surface: str, expr: Any, symbols: set[str]) -> str | None:
    lhs = lhs_symbol(surface, symbols)
    if lhs:
        return lhs
    if not symbols:
        return None
    try:
        degrees = {}
        for name in symbols:
            degree = sympy.degree(expr, sympy.Symbol(name))
            degrees[name] = degree if degree is not None else -1
        best = [name for name, degree in degrees.items() if degree == max(degrees.values())]
        return sorted(best)[-1]
    except Exception:
        return sorted(symbols)[-1]


def solve_symbol_candidates(surface: str, expr: Any, symbols: set[str]) -> list[str]:
    if not symbols:
        return []
    ordered = []
    first = choose_solve_symbol(surface, expr, symbols)
    if first:
        ordered.append(first)
    try:
        degree_order = sorted(
            symbols,
            key=lambda name: (
                sympy.degree(expr, sympy.Symbol(name)) if sympy.degree(expr, sympy.Symbol(name)) is not None else -1,
                name,
            ),
            reverse=True,
        )
    except Exception:
        degree_order = sorted(symbols)
    for name in degree_order:
        if name not in ordered:
            ordered.append(name)
    return ordered


def safe_lambdify(expr: Any, ordered_symbols: list[str]):
    sym_objects = [sympy.Symbol(name) for name in ordered_symbols]
    return sympy.lambdify(sym_objects, expr, modules=["numpy"])


def random_symbol_env(symbols: list[str], points: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    env = {name: rng.uniform(0.25, 3.0, size=points) for name in symbols}
    # Keep common relativistic terms real for sqrt(1 - v**2/c**2).
    if "c" in env and "v" in env:
        env["c"] = rng.uniform(2.0, 5.0, size=points)
        env["v"] = rng.uniform(0.05, 0.85, size=points) * env["c"]
    # Keep common waveguide / plasma square-root domains real.
    for cutoff, carrier in (("f_c", "f"), ("omega_p", "omega"), ("theta_c", "theta_i")):
        if cutoff in env and carrier in env:
            env[cutoff] = rng.uniform(0.15, 1.0, size=points)
            env[carrier] = rng.uniform(1.15, 4.0, size=points) * env[cutoff]
    if "lambda" in env and "lambda_g" in env:
        env["lambda"] = rng.uniform(0.25, 2.0, size=points)
    for name in ("gamma_SE", "Ap", "Bp", "p", "d"):
        if name in env:
            env[name] = rng.uniform(0.75, 3.0, size=points)
    return env


def try_linear_truth_points(
    expr: Any,
    surface: str,
    symbols: set[str],
    points: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Fast path for equations that are linear in at least one variable."""
    for solve_name in solve_symbol_candidates(surface, expr, symbols):
        solve_sym = sympy.Symbol(solve_name)
        try:
            coeff = sympy.diff(expr, solve_sym)
            if coeff == 0 or solve_sym in coeff.free_symbols:
                continue
            rest = expr.subs(solve_sym, 0)
            sol = -rest / coeff
            if solve_sym in sol.free_symbols:
                continue
        except Exception:
            continue
        other_symbols = sorted(symbols - {solve_name})
        for _ in range(16):
            env = random_symbol_env(other_symbols, points, rng)
            try:
                fn = safe_lambdify(sol, other_symbols)
                with np.errstate(all="ignore"):
                    raw_values = np.asarray(fn(*[env[name] for name in other_symbols]))
                if np.iscomplexobj(raw_values):
                    if not np.all(np.abs(np.imag(raw_values)) <= 1.0e-8):
                        continue
                    raw_values = np.real(raw_values)
                values = np.asarray(raw_values, dtype=np.float64)
                if values.ndim == 0:
                    values = np.full(points, float(values), dtype=np.float64)
                values = values.reshape(-1)[:points]
                if values.size != points or not np.isfinite(values).all():
                    continue
                out = dict(env)
                out[solve_name] = values
                return out, None
            except Exception:
                continue
    return None, "truth_not_linear"


def sample_truth_points(
    surface: str,
    points: int,
    rng: np.random.Generator,
    composite_constraints: bool = False,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    if sympy is None:
        return None, "sympy_missing"
    if composite_constraints:
        composite_points, composite_error = sample_composite_truth_points(surface, points, rng)
        if composite_points is not None or composite_error != "not_composite":
            return composite_points, composite_error
    truth_expr = _parse_canonical_form(surface)
    if truth_expr is None:
        return None, "truth_parse_failed"
    symbols = _extract_symbols(truth_expr)
    linear_points, linear_error = try_linear_truth_points(truth_expr, surface, symbols, points, rng)
    if linear_points is not None:
        return linear_points, None
    candidates = solve_symbol_candidates(surface, truth_expr, symbols)
    if not candidates:
        return None, "no_solve_symbol"
    saw_solution = False
    for solve_name in candidates:
        solve_sym = sympy.Symbol(solve_name)
        try:
            solutions = sympy.solve(truth_expr, solve_sym)
        except Exception:
            continue
        if not solutions:
            continue
        saw_solution = True
        other_symbols = sorted(symbols - {solve_name})
        for _ in range(16):
            env = random_symbol_env(other_symbols, points, rng)
            for sol in solutions:
                try:
                    fn = safe_lambdify(sol, other_symbols)
                    with np.errstate(all="ignore"):
                        raw_values = np.asarray(fn(*[env[name] for name in other_symbols]))
                    if np.iscomplexobj(raw_values):
                        if not np.all(np.abs(np.imag(raw_values)) <= 1.0e-8):
                            continue
                        raw_values = np.real(raw_values)
                    values = np.asarray(raw_values, dtype=np.float64)
                    if values.ndim == 0:
                        values = np.full(points, float(values), dtype=np.float64)
                    values = values.reshape(-1)[:points]
                    if values.size != points or not np.isfinite(values).all():
                        continue
                    out = dict(env)
                    out[solve_name] = values
                    return out, None
                except Exception:
                    continue
    return None, "truth_solution_nonfinite" if saw_solution else linear_error or "truth_no_solution"


def solve_component_for(
    expr: Any,
    solve_name: str,
    env: dict[str, np.ndarray],
    points: int,
) -> np.ndarray | None:
    solve_sym = sympy.Symbol(solve_name)
    other_symbols = sorted(str(symbol) for symbol in expr.free_symbols if str(symbol) != solve_name)
    try:
        coeff = sympy.diff(expr, solve_sym)
        if coeff != 0 and solve_sym not in coeff.free_symbols:
            rest = expr.subs(solve_sym, 0)
            solutions = [-rest / coeff]
        else:
            solutions = sympy.solve(expr, solve_sym)
    except Exception:
        return None
    finite_candidates: list[np.ndarray] = []
    for sol in solutions:
        if solve_sym in getattr(sol, "free_symbols", set()):
            continue
        missing = [name for name in other_symbols if name not in env]
        if missing:
            continue
        try:
            fn = safe_lambdify(sol, other_symbols)
            with np.errstate(all="ignore"):
                raw_values = np.asarray(fn(*[env[name] for name in other_symbols]))
            if np.iscomplexobj(raw_values):
                if not np.all(np.abs(np.imag(raw_values)) <= 1.0e-8):
                    continue
                raw_values = np.real(raw_values)
            values = np.asarray(raw_values, dtype=np.float64)
            if values.ndim == 0:
                values = np.full(points, float(values), dtype=np.float64)
            values = values.reshape(-1)[:points]
            if values.size == points and np.isfinite(values).all():
                finite_candidates.append(values)
        except Exception:
            continue
    if not finite_candidates:
        return None
    positive_names = {"V", "T", "T_temp", "R", "gamma", "c", "m", "n", "P", "E", "A", "d", "k", "omega", "omega0"}
    if solve_name in positive_names:
        for values in finite_candidates:
            if np.all(values > 1.0e-12):
                return values
    return finite_candidates[0]


def component_solve_order(parsed: list[tuple[str, Any, set[str]]], eliminate: str | None) -> list[tuple[str, Any, set[str]]]:
    if not eliminate:
        return parsed
    def elim_degree(item: tuple[str, Any, set[str]]) -> int:
        _, expr, symbols = item
        if eliminate not in symbols:
            return 99
        try:
            degree = sympy.degree(expr, sympy.Symbol(eliminate))
            return int(degree) if degree is not None else 98
        except Exception:
            return 98

    return sorted(parsed, key=lambda item: (0 if eliminate in item[2] else 1, elim_degree(item), len(item[2]), item[0]))


def choose_component_solve_symbol(
    component: str,
    expr: Any,
    symbols: set[str],
    eliminate: str | None,
    locked_symbols: set[str],
) -> str | None:
    candidates = [name for name in symbols if name not in locked_symbols]
    if eliminate and eliminate in symbols and eliminate not in locked_symbols:
        return eliminate
    lhs = lhs_symbol(component, symbols)
    if lhs and lhs in candidates:
        return lhs
    if not candidates:
        return None
    try:
        ranked = sorted(
            candidates,
            key=lambda name: (
                1 if sympy.degree(expr, sympy.Symbol(name)) == 1 else 0,
                -(sympy.degree(expr, sympy.Symbol(name)) if sympy.degree(expr, sympy.Symbol(name)) is not None else 99),
                name,
            ),
            reverse=True,
        )
        return ranked[0] if ranked else None
    except Exception:
        return sorted(candidates)[-1]


def sample_composite_truth_points(surface: str, points: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray] | None, str | None]:
    parsed, eliminate = parse_constraint_components(surface)
    if len(parsed) <= 1:
        return None, "not_composite"
    all_symbols = sorted(set().union(*(symbols for _, _, symbols in parsed)))
    if not all_symbols:
        return None, "no_solve_symbol"
    last_error = "composite_solution_nonfinite"
    for _ in range(16):
        env = random_symbol_env(all_symbols, points, rng)
        locked_symbols: set[str] = set()
        solved_any = False
        failed = False
        for component, expr, symbols in component_solve_order(parsed, eliminate):
            solve_name = choose_component_solve_symbol(component, expr, symbols, eliminate, locked_symbols)
            if not solve_name:
                return None, "composite_no_solve_symbol"
            values = solve_component_for(expr, solve_name, env, points)
            if values is None:
                failed = True
                last_error = "composite_solution_nonfinite"
                break
            env[solve_name] = values
            if solve_name == "V" and "b" in env and "b" not in locked_symbols:
                env["b"] = rng.uniform(0.05, 0.5, size=points) * np.maximum(values, 1.0e-6)
            locked_symbols.update(symbols)
            solved_any = True
        if solved_any and not failed:
            return env, None
    return None, last_error


def residual_metrics(
    surface: str,
    values: dict[str, np.ndarray],
    composite_constraints: bool = False,
) -> dict[str, float | str]:
    if sympy is None:
        return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "sympy_missing"}
    if composite_constraints:
        component_metrics = residual_metrics_components(surface, values)
        if component_metrics is not None:
            return component_metrics
    expr = _parse_canonical_form(surface)
    if expr is None:
        return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "decoded_parse_failed"}
    symbols = sorted(_extract_symbols(expr))
    missing = [name for name in symbols if name not in values]
    if missing:
        return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "symbol_mismatch:" + ",".join(missing[:8])}
    try:
        fn = safe_lambdify(expr, symbols)
        with np.errstate(all="ignore"):
            residual = np.asarray(fn(*[values[name] for name in symbols]), dtype=np.float64)
    except Exception:
        return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "decoded_eval_failed"}
    if residual.ndim == 0:
        residual = np.full(next(iter(values.values())).shape, float(residual), dtype=np.float64)
    residual = residual.reshape(-1)
    if residual.size == 0 or not np.isfinite(residual).all():
        return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "decoded_nonfinite"}
    mse = float(np.mean(residual**2))
    scale_terms = [np.asarray(arr, dtype=np.float64).reshape(-1) for arr in values.values()]
    scale = float(np.mean(np.concatenate([arr * arr for arr in scale_terms]))) if scale_terms else 1.0
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    nmse = mse / scale
    r2 = 1.0 - nmse
    return {"mse": mse, "nmse": nmse, "r2": r2, "error": ""}


def residual_metrics_components(surface: str, values: dict[str, np.ndarray]) -> dict[str, float | str] | None:
    parsed, _ = parse_constraint_components(surface)
    if len(parsed) <= 1:
        return None
    residuals = []
    for _, expr, symbols_set in parsed:
        symbols = sorted(symbols_set)
        missing = [name for name in symbols if name not in values]
        if missing:
            return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "symbol_mismatch:" + ",".join(missing[:8])}
        try:
            fn = safe_lambdify(expr, symbols)
            with np.errstate(all="ignore"):
                residual = np.asarray(fn(*[values[name] for name in symbols]), dtype=np.float64)
        except Exception:
            return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "decoded_eval_failed"}
        if residual.ndim == 0:
            residual = np.full(next(iter(values.values())).shape, float(residual), dtype=np.float64)
        residual = residual.reshape(-1)
        if residual.size == 0 or not np.isfinite(residual).all():
            return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "decoded_nonfinite"}
        residuals.append(residual)
    joined = np.concatenate(residuals) if residuals else np.asarray([], dtype=np.float64)
    if joined.size == 0:
        return {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": "decoded_nonfinite"}
    mse = float(np.mean(joined**2))
    scale_terms = [np.asarray(arr, dtype=np.float64).reshape(-1) for arr in values.values()]
    scale = float(np.mean(np.concatenate([arr * arr for arr in scale_terms]))) if scale_terms else 1.0
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    nmse = mse / scale
    r2 = 1.0 - nmse
    return {"mse": mse, "nmse": nmse, "r2": r2, "error": ""}


def main() -> None:
    from audit_stage8g_decoder_fidelity import decoded_formula_valid

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    records = read_jsonl(args.records_jsonl, args.max_records)
    catalog_norms = read_catalog_norms(args.catalog_jsonl)
    checked = []
    totals = {
        "strict_valid": 0,
        "finite_residual": 0,
        "pass_nmse": 0,
        "pass_r2": 0,
        "outside_catalog": 0,
    }
    errors: dict[str, int] = {}

    for rec in records:
        truth = str(rec.get("truth_surface") or rec.get("truth") or "")
        decoded = str(rec.get("strict_decoded_surface") or rec.get("decoded") or "")
        valid, validity = decoded_formula_valid(decoded)
        points, point_error = sample_truth_points(truth, args.points, rng, args.composite_constraints)
        metrics = (
            residual_metrics(decoded, points, args.composite_constraints)
            if points is not None and valid
            else {"mse": float("inf"), "nmse": float("inf"), "r2": float("-inf"), "error": point_error or "strict_invalid"}
        )
        error = str(metrics.get("error") or "")
        if error:
            errors[error] = errors.get(error, 0) + 1
        outside = normalize_formula(decoded) not in catalog_norms if catalog_norms else None
        finite = math.isfinite(float(metrics["nmse"]))
        pass_nmse = float(metrics["nmse"]) <= args.nmse_threshold
        pass_r2 = float(metrics["r2"]) >= args.r2_threshold
        totals["strict_valid"] += int(valid)
        totals["finite_residual"] += int(finite)
        totals["pass_nmse"] += int(pass_nmse)
        totals["pass_r2"] += int(pass_r2)
        if outside is not None:
            totals["outside_catalog"] += int(outside)
        checked.append(
            {
                "law_id": rec.get("law_id"),
                "truth_surface": truth,
                "decoded": decoded,
                "strict_valid": valid,
                **validity,
                "outside_catalog": outside,
                "residual_mse": metrics["mse"],
                "residual_nmse": metrics["nmse"],
                "residual_r2": metrics["r2"],
                "checker_error": error,
                "pass_nmse": pass_nmse,
                "pass_r2": pass_r2,
            }
        )

    total = max(len(checked), 1)
    summary = {
        "records_jsonl": str(args.records_jsonl),
        "items": len(checked),
        "points": args.points,
        "nmse_threshold": args.nmse_threshold,
        "r2_threshold": args.r2_threshold,
        "composite_constraints": args.composite_constraints,
        "strict_valid_rate": totals["strict_valid"] / total,
        "finite_residual_fraction": totals["finite_residual"] / total,
        "checker_pass@nmse": totals["pass_nmse"] / total,
        "checker_pass@r2": totals["pass_r2"] / total,
        "outside_catalog_fraction": totals["outside_catalog"] / total if catalog_norms else None,
        "errors": dict(sorted(errors.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in checked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
