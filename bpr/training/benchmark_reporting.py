"""Post-processing helpers for benchmark result tables."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from bpr.constants import DEFAULT_RANDOM_STATE
from bpr.training.benchmark import _kneighbors_factory_for_fingerprint, get_X_y
from bpr.training.per_class_split_counts import per_class_reporting_count_rows
from bpr.training.resampling import apply_resampling


def format_metrics_long(best_df: pd.DataFrame) -> pd.DataFrame:
    """Format benchmark results into long form metrics dataframe."""
    rows = []
    for _, r in best_df.iterrows():
        opt_for = r.get("optimized_for", float("nan"))
        for col in best_df.columns:
            if col.startswith("val_"):
                metric_type = "val"
                metric_name = col[len("val_") :]
                n_samples = r["val_n"] if "val_n" in r.index else np.nan
            elif col.startswith("final_test_"):
                metric_type = "test"
                metric_name = col[len("final_test_") :]
                n_samples = r["final_test_n"] if "final_test_n" in r.index else np.nan
            elif col.startswith("final_train_"):
                metric_type = "train"
                metric_name = col[len("final_train_") :]
                n_samples = r["final_train_n"] if "final_train_n" in r.index else np.nan
            else:
                continue
            val = r[col]
            if pd.isna(val):
                continue
            if metric_name == "accuracy" and not pd.isna(n_samples):
                n_correct = float(val) * float(n_samples)
                n_incorrect = float(n_samples) - n_correct
            else:
                n_correct = np.nan
                n_incorrect = np.nan

            row_dict = {
                "target": r["target"],
                "model": r["model"],
                "finger_print_method": r["fingerprint"],
                "optimized_for": opt_for,
                "metric": metric_name,
                "metric_type": metric_type,
                "metric_value": float(val),
                "n_samples": float(n_samples) if not pd.isna(n_samples) else np.nan,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "parameters": r["best_params"],
            }
            if "resampling" in r.index:
                row_dict["resampling"] = r["resampling"]
            rows.append(row_dict)
    return pd.DataFrame(rows)


def _factory_and_params_from_row(
    r: pd.Series,
    model_configs: list[tuple[str, Callable, dict[str, list[Any]]]],
) -> tuple[Callable[..., Any], dict[str, Any]] | None:
    try:
        params_str = r["best_params"]
        params = ast.literal_eval(params_str) if isinstance(params_str, str) else params_str
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    model_name = r["model"]
    fingerprint = r["fingerprint"]
    base_factory = next((c[1] for c in model_configs if c[0] == model_name), None)
    if base_factory is None:
        return None
    factory = _kneighbors_factory_for_fingerprint(
        model_name, fingerprint, DEFAULT_RANDOM_STATE, base_factory
    )
    return factory, params


def _split_x_for_predict(
    s: str,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    scale: bool,
    scaler: MinMaxScaler | None,
    x_train_s: np.ndarray,
    model_name: str,
    fingerprint: str,
) -> np.ndarray:
    if scale:
        x_s = scaler.transform(data[s][0]) if s != "train" else x_train_s
    else:
        x_s = data[s][0]
    if model_name == "KNeighbors" and fingerprint not in ("estate", "erg"):
        x_s = x_s.astype(bool, copy=False)
    return x_s


def _preds_for_class_count(
    r: pd.Series,
    datasets: dict[str, pd.DataFrame],
    factory: Callable[..., Any],
    params: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    target = r["target"]
    fingerprint = r["fingerprint"]
    model_name = r["model"]
    keys = {
        "train": f"{target}_train_{fingerprint}",
        "val": f"{target}_val_{fingerprint}",
        "test": f"{target}_test_{fingerprint}",
    }
    splits_to_use = ["train", "val"]
    if keys["test"] in datasets:
        splits_to_use.append("test")
    if any(keys[s] not in datasets for s in ("train", "val")):
        return None
    data = {s: get_X_y(datasets[keys[s]]) for s in splits_to_use}
    scale = fingerprint in ("estate", "erg")
    if scale:
        scaler = MinMaxScaler()
        x_train_s = scaler.fit_transform(data["train"][0])
    else:
        scaler = None
        x_train_s = data["train"][0]
    resampling = r.get("resampling", "none")
    x_train_use, y_train_use = apply_resampling(x_train_s, data["train"][1], resampling)
    if model_name == "KNeighbors" and fingerprint not in ("estate", "erg"):
        x_train_use = x_train_use.astype(bool, copy=False)
    model = factory(**params)
    model.fit(x_train_use, y_train_use)
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in splits_to_use:
        x_s = _split_x_for_predict(s, data, scale, scaler, x_train_s, model_name, fingerprint)
        preds[s] = (data[s][1], model.predict(x_s))
    return preds


def _run_one_class_count(
    r: pd.Series,
    datasets: dict[str, pd.DataFrame],
    model_configs: list[tuple[str, Callable, dict[str, list[Any]]]],
) -> list[dict[str, Any]]:
    parsed = _factory_and_params_from_row(r, model_configs)
    if parsed is None:
        return []
    factory, params = parsed
    preds = _preds_for_class_count(r, datasets, factory, params)
    if preds is None:
        return []
    target = r["target"]
    model_name = r["model"]
    fingerprint = r["fingerprint"]
    opt_for = r.get("optimized_for", float("nan"))
    return per_class_reporting_count_rows(
        preds,
        r,
        target=target,
        model_name=model_name,
        fingerprint=fingerprint,
        opt_for=opt_for,
    )


def compute_per_class_counts(
    results_df: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    model_configs: list[tuple[str, Callable, dict[str, list[Any]]]],
    max_workers: int = 8,
) -> pd.DataFrame:
    """Compute per-class counts evaluating best models on datasets (parallel)."""
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one_class_count, r, datasets, model_configs): r
            for _, r in results_df.iterrows()
        }
        for future in as_completed(futures):
            res = future.result()
            rows.extend(res)
    return pd.DataFrame(rows)
