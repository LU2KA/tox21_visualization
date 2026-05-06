"""Benchmark helpers."""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from bpr.constants import ALL_METRICS, DEFAULT_RANDOM_STATE, SPLIT_NAMES
from bpr.training.approx_jaccard_knn import ApproximateJaccardKNeighborsClassifier
from bpr.training.per_class_split_counts import per_class_benchmark_count_rows

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"^`algorithm` parameter is deprecated in 0\.12 and will be removed in 0\.14\..*",
)


def _kneighbors_factory_for_fingerprint(
    model_name: str,
    fingerprint_method: str,
    random_state: int,
    base_factory: Callable[..., Any],
) -> Callable[..., Any]:
    """Binary fingerprints use approximate Jaccard k-NN; estate/erg keep the notebook factory."""
    if model_name == "KNeighbors" and fingerprint_method not in ("estate", "erg"):

        def _f(**kw: Any) -> Any:
            return ApproximateJaccardKNeighborsClassifier(random_state=random_state, **kw)

        return _f
    return base_factory


def get_X_y(df: pd.DataFrame, target_col: str = "target") -> tuple[np.ndarray, np.ndarray]:
    """Extract X and y from a dataset DataFrame."""
    if target_col not in df.columns:
        raise ValueError(f"Missing column '{target_col}'")
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col]).values
    return X, y


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute requested metrics (all by default)."""
    to_compute = metrics if metrics is not None else ALL_METRICS
    out: dict[str, float] = {}
    if "accuracy" in to_compute:
        out["accuracy"] = accuracy_score(y_true, y_pred)
    if "f1_macro" in to_compute:
        out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if "f1_weighted" in to_compute:
        out["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    has_proba = y_proba is not None and len(np.unique(y_true)) == 2
    if "roc_auc" in to_compute:
        out["roc_auc"] = (
            roc_auc_score(y_true, y_proba[:, 1]) if has_proba else float("nan")
        )
    if "pr_auc" in to_compute:
        out["pr_auc"] = (
            average_precision_score(y_true, y_proba[:, 1]) if has_proba else float("nan")
        )
    return out


def fit_predict_eval(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    scale: bool = True,
    metrics_to_compute: list[str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Fit on train; return val metrics and optionally test metrics."""
    _ = random_state
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        if X_test is not None:
            X_test = scaler.transform(X_test)
    model = clone(model)
    model.fit(X_train, y_train)
    has_proba = hasattr(model, "predict_proba") and callable(
        getattr(model, "predict_proba")
    )
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val) if has_proba else None
    val_metrics = evaluate(y_val, y_val_pred, y_val_proba, metrics=metrics_to_compute)
    if X_test is None or y_test is None:
        return val_metrics, None
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test) if has_proba else None
    test_metrics = evaluate(
        y_test, y_test_pred, y_test_proba, metrics=metrics_to_compute
    )
    return val_metrics, test_metrics


def selection_score(
    val_metrics: dict[str, float],
    optimize_metrics: list[str],
) -> float:
    """Primary metric score for basic selection (defaults to last in optimize_metrics)."""
    if not optimize_metrics:
        return val_metrics.get("roc_auc", float("-inf"))
    primary_metric = optimize_metrics[-1]
    score = val_metrics.get(primary_metric, float("-inf"))
    return score if pd.notna(score) else float("-inf")


def _run_one_params(
    params: dict[str, Any],
    factory: Callable[..., Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale: bool,
    metrics_to_compute: list[str] | None,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """Run one parameter combination; return (params, val_metrics) or (params, None) on failure."""
    try:
        model = factory(**params)
        start_time = time.time()
        val_m, _ = fit_predict_eval(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test=None,
            y_test=None,
            scale=scale,
            metrics_to_compute=metrics_to_compute,
            random_state=random_state,
        )
        val_m["training_time"] = time.time() - start_time
        return (params, val_m)
    except Exception:  # pylint: disable=broad-exception-caught
        return (params, None)


def grid_search_best(
    factory: Callable[..., Any],
    param_grid: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimize_metrics: list[str],
    scale: bool = True,
    return_history: bool = False,
    metrics_to_compute: list[str] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict | None, dict | None, Any]:
    """Try all param combinations on train/val sequentially. Return best params, best val metrics; optionally full history."""
    to_compute = metrics_to_compute if metrics_to_compute is not None else optimize_metrics
    param_list = list(ParameterGrid(param_grid))
    results = [
        _run_one_params(
            p,
            factory,
            X_train,
            y_train,
            X_val,
            y_val,
            scale,
            to_compute,
            random_state,
        )
        for p in param_list
    ]
    best_score = -np.inf
    best_params = None
    best_val_m = None
    history: list[tuple[dict, dict]] = []
    for params, val_m in results:
        if val_m is None:
            continue
        score = selection_score(val_m, optimize_metrics)
        if return_history:
            history.append((dict(params), val_m))
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_params = params
            best_val_m = val_m
    if best_params is None:
        return (None, None, history) if return_history else (None, None)
    return (best_params, best_val_m, history) if return_history else (best_params, best_val_m)


def grid_size(param_grid: dict[str, list[Any]]) -> int:
    """Number of parameter combinations in the grid."""
    if not param_grid:
        return 1
    return int(np.prod([len(v) for v in param_grid.values()]))


def get_benchmark_combos(
    datasets: dict[str, pd.DataFrame],
    targets: list[str],
    methods: list[str],
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> list[tuple[str, str, tuple[str, str, str]]]:
    """Build list of (target, method, (train_k, val_k, test_k)) that have all splits in datasets."""
    combos: list[tuple[str, str, tuple[str, str, str]]] = []
    for t in targets:
        for m in methods:
            train_k = f"{t}_{split_names[0]}_{m}"
            val_k = f"{t}_{split_names[1]}_{m}"
            test_k = f"{t}_{split_names[2]}_{m}"
            if train_k in datasets and val_k in datasets and test_k in datasets:
                combos.append((t, m, (train_k, val_k, test_k)))
    return combos


def count_benchmark_runs(
    n_combos: int,
    model_names: list[str],
    resampling_variants: list[str],
    imbalance_model_names: set[str],
) -> int:
    """Actual number of runs when imbalance models only run with resampling='none'."""
    n_models = len(model_names)
    n_resampling = len(resampling_variants)
    n_imbalance = sum(1 for name in model_names if name in imbalance_model_names)
    return n_combos * (
        n_models * n_resampling - (n_resampling - 1) * n_imbalance
    )


def _build_benchmark_tasks(
    combos: list[tuple[str, str, tuple[str, str, str]]],
    resampling_variants: list[str],
    model_configs: list[tuple[str, Callable[..., Any], Any]],
    imbalance_models: set[str],
) -> list[tuple[Any, ...]]:
    """Build a flat task list in the same order as the triple loop."""
    tasks: list[tuple[Any, ...]] = []
    for target, method, (train_k, val_k, test_k) in combos:
        for resampling in resampling_variants:
            for model_name, factory, param_grid in model_configs:
                if resampling != "none" and model_name in imbalance_models:
                    continue
                # If param_grid is a callable, evaluate it with the resampling strategy
                if callable(param_grid):
                    grid_to_use = param_grid(resampling)
                else:
                    grid_to_use = param_grid
                tasks.append(
                    (train_k, val_k, test_k, target, method, resampling, model_name, factory, grid_to_use)
                )
    return tasks


def benchmark_completeness_report(
    combos: list[tuple[str, str, tuple[str, str, str]]],
    resampling_variants: list[str],
    model_configs: list[tuple[str, Callable[..., Any], Any]],
    imbalance_models: set[str],
    optimize_metrics: list[str],
    results_df: pd.DataFrame,
    missing_sample_limit: int = 20,
) -> dict[str, Any]:
    """Compare expected benchmark keys to rows in ``results_df``."""
    cols = ["target", "fingerprint", "resampling", "model", "optimized_for"]
    for c in cols:
        if c not in results_df.columns:
            raise ValueError(f"results_df missing column '{c}'")

    tasks = _build_benchmark_tasks(
        combos, resampling_variants, model_configs, imbalance_models
    )
    expected: set[tuple[str, str, str, str, str]] = set()
    for t in tasks:
        _, _, _, target, method, resampling, model_name, _, _ = t
        for opt in optimize_metrics:
            expected.add(
                (str(target), str(method), str(resampling), str(model_name), str(opt))
            )

    def _row_key(row: pd.Series) -> tuple[str, str, str, str, str] | None:
        if pd.isna(row.get("optimized_for")):
            return None
        return (
            str(row["target"]),
            str(row["fingerprint"]),
            str(row["resampling"]),
            str(row["model"]),
            str(row["optimized_for"]),
        )

    actual: set[tuple[str, str, str, str, str]] = set()
    n_skip = 0
    for _, row in results_df.iterrows():
        k = _row_key(row)
        if k is None:
            n_skip += 1
            continue
        actual.add(k)

    missing = expected - actual
    unexpected = actual - expected
    n_dup_rows = int(len(results_df) - results_df.drop_duplicates(subset=cols).shape[0])

    report: dict[str, Any] = {
        "ok": len(missing) == 0 and n_skip == 0 and n_dup_rows == 0,
        "n_expected": len(expected),
        "n_actual_unique": len(actual),
        "n_rows": len(results_df),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "n_rows_with_na_optimized_for": n_skip,
        "n_duplicate_key_rows": n_dup_rows,
        "missing_keys": sorted(missing)[:missing_sample_limit],
        "unexpected_keys": sorted(unexpected)[:missing_sample_limit],
    }
    return report


def _binary_val_roc_pr_lists(
    model: Any,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    has_proba: bool,
) -> tuple[list[float], list[float], list[float], list[float]] | None:
    if not has_proba or len(np.unique(y_val)) != 2:
        return None
    y_val_proba = model.predict_proba(X_val_scaled)
    fpr, tpr, _ = roc_curve(y_val, y_val_proba[:, 1])
    prec, rec, _ = precision_recall_curve(y_val, y_val_proba[:, 1])
    return (fpr.tolist(), tpr.tolist(), prec.tolist(), rec.tolist())


def _compute_final_metrics_and_roc(
    factory: Callable[..., Any],
    best_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    optimize_metrics: list[str],
    scale: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
    return_per_class_counts: bool = False,
    target: str = "",
    fingerprint: str = "",
    model_name: str = "",
    resampling: str = "none",
    opt_metric: str = "",
    best_params_str: str = "",
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    tuple[list[float], list[float], list[float], list[float]] | None,
    list[dict[str, Any]],
    float | None,
]:
    """
    Fit model with best_params on resampled training data.
    Return (train_metrics, test_metrics, curve tuple for val set, per_class_rows, refit_time) or
    (None, None, None, [], None) if not available. The curve tuple is
    (fpr, tpr, precision, recall) lists for binary validation data with probabilities, else None.
    per_class_rows is empty when return_per_class_counts=False.
    """
    _ = random_state
    try:
        model = factory(**best_params)
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        else:
            X_train_scaled, X_val_scaled, X_test_scaled = X_train, X_val, X_test
        model = clone(model)

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        refit_time = time.time() - start_time

        has_proba = hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))

        y_train_pred = model.predict(X_train_scaled)
        y_train_proba = model.predict_proba(X_train_scaled) if has_proba else None
        train_metrics = evaluate(y_train, y_train_pred, y_train_proba, metrics=optimize_metrics)

        test_metrics = None
        y_test_pred = None
        if X_test_scaled is not None and y_test is not None:
            y_test_pred = model.predict(X_test_scaled)
            y_test_proba = model.predict_proba(X_test_scaled) if has_proba else None
            test_metrics = evaluate(y_test, y_test_pred, y_test_proba, metrics=optimize_metrics)

        roc_data = _binary_val_roc_pr_lists(model, X_val_scaled, y_val, has_proba)

        per_class_rows: list[dict[str, Any]] = []
        if return_per_class_counts:
            if has_proba and len(np.unique(y_val)) == 2:
                y_val_p = model.predict_proba(X_val_scaled)
                y_val_pred = np.argmax(y_val_p, axis=1)
            else:
                y_val_pred = model.predict(X_val_scaled)
            preds = {"train": (y_train, y_train_pred), "val": (y_val, y_val_pred)}
            if y_test_pred is not None and y_test is not None:
                preds["test"] = (y_test, y_test_pred)
            per_class_rows = per_class_benchmark_count_rows(
                preds,
                target=target,
                model_name=model_name,
                fingerprint=fingerprint,
                opt_metric=opt_metric,
                resampling=resampling,
                best_params_str=best_params_str,
            )

        return train_metrics, test_metrics, roc_data, per_class_rows, refit_time
    except Exception:  # pylint: disable=broad-exception-caught
        return None, None, None, [], None
