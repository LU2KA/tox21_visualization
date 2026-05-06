"""Orchestration for one benchmark task and full ``run_benchmark`` loops."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm

from bpr.constants import DEFAULT_RANDOM_STATE
from bpr.training.benchmark import (
    _build_benchmark_tasks,
    _compute_final_metrics_and_roc,
    _kneighbors_factory_for_fingerprint,
    count_benchmark_runs,
    get_X_y,
    grid_search_best,
    selection_score,
)
from bpr.training.resampling import apply_resampling


class _PreparedBenchmarkTask(NamedTuple):
    target: str
    method: str
    resampling: str
    model_name: str
    factory: Callable[..., Any]
    param_grid: Any
    run_date: str
    X_train_use: np.ndarray
    y_train_use: np.ndarray
    X_val_s: np.ndarray
    y_val: np.ndarray
    X_test_s: np.ndarray | None
    y_test: np.ndarray | None


def _prepare_benchmark_task_arrays(
    task: tuple[Any, ...],
    datasets: dict[str, pd.DataFrame],
    target_col: str,
    random_state: int,
    random_under_sampling_strategy: float,
) -> _PreparedBenchmarkTask:
    (
        train_k,
        val_k,
        test_k,
        target,
        method,
        resampling,
        model_name,
        factory,
        param_grid,
    ) = task
    run_date = datetime.now(timezone.utc).isoformat()
    x_train, y_train = get_X_y(datasets[train_k], target_col)
    x_val, y_val = get_X_y(datasets[val_k], target_col)
    x_test, y_test = (
        get_X_y(datasets[test_k], target_col) if test_k in datasets else (None, None)
    )
    scale = method in ("estate", "erg")
    if scale:
        scaler = MinMaxScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_val_s = scaler.transform(x_val)
        x_test_s = scaler.transform(x_test) if x_test is not None else None
    else:
        x_train_s, x_val_s, x_test_s = x_train, x_val, x_test
    x_train_use, y_train_use = apply_resampling(
        x_train_s,
        y_train,
        resampling,
        random_state=random_state,
        random_under_sampling_strategy=random_under_sampling_strategy,
    )
    factory_wrapped = _kneighbors_factory_for_fingerprint(
        model_name, method, random_state, factory
    )
    if model_name == "KNeighbors" and method not in ("estate", "erg"):
        x_train_use = x_train_use.astype(bool, copy=False)
        x_val_s = x_val_s.astype(bool, copy=False)
        if x_test_s is not None:
            x_test_s = x_test_s.astype(bool, copy=False)
    return _PreparedBenchmarkTask(
        target,
        method,
        resampling,
        model_name,
        factory_wrapped,
        param_grid,
        run_date,
        x_train_use,
        y_train_use,
        x_val_s,
        y_val,
        x_test_s,
        y_test,
    )


def _grid_rows_from_search_history(
    history: list[tuple[dict, dict]],
    target: str,
    method: str,
    model_name: str,
    resampling: str,
    optimize_metrics: list[str],
) -> list[dict[str, Any]]:
    grid_rows_chunk: list[dict[str, Any]] = []
    for params_dict, val_m in history:
        sel = selection_score(val_m, optimize_metrics)
        row: dict[str, Any] = {
            "target": target,
            "fingerprint": method,
            "model": model_name,
            "resampling": resampling,
            "params": str(params_dict),
            "val_selection": sel,
            "training_time": val_m.get("training_time", float("nan")),
        }
        for m in optimize_metrics:
            row["val_" + m] = val_m[m]
        for pk, pv in params_dict.items():
            row[f"param_{pk}"] = pv
        grid_rows_chunk.append(row)
    return grid_rows_chunk


def _best_params_for_opt_metric(
    history: list[tuple[dict, dict]],
    opt_metric: str,
) -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    best_score = -np.inf
    best_params = None
    best_val_m = None
    for params_dict, val_m in history:
        score = val_m.get(opt_metric, float("-inf"))
        if pd.isna(score):
            score = float("-inf")
        if score > best_score:
            best_score = score
            best_params = params_dict
            best_val_m = val_m
    return best_params, best_val_m


def _empty_result_rows_for_metrics(
    target: str,
    method: str,
    model_name: str,
    resampling: str,
    optimize_metrics: list[str],
    val_n: int,
    run_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opt_metric in optimize_metrics:
        row: dict[str, Any] = {
            "target": target,
            "fingerprint": method,
            "model": model_name,
            "resampling": resampling,
            "optimized_for": opt_metric,
            "best_params": "",
            "val_selection": float("nan"),
            "val_n": val_n,
            "run_date": run_date,
        }
        for m in optimize_metrics:
            row["val_" + m] = float("nan")
        rows.append(row)
    return rows


def _result_row_when_no_best(
    target: str,
    method: str,
    model_name: str,
    resampling: str,
    opt_metric: str,
    val_n: int,
    run_date: str,
    optimize_metrics: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target": target,
        "fingerprint": method,
        "model": model_name,
        "resampling": resampling,
        "optimized_for": opt_metric,
        "best_params": "",
        "val_selection": float("nan"),
        "val_n": val_n,
        "training_time": float("nan"),
        "refit_time": float("nan"),
        "run_date": run_date,
    }
    for m in optimize_metrics:
        row["val_" + m] = float("nan")
    return row


def _finalize_result_row_metrics(
    result_row: dict[str, Any],
    train_metrics: dict[str, float] | None,
    test_metrics: dict[str, float] | None,
    y_train_use: np.ndarray,
    y_test: np.ndarray | None,
    optimize_metrics: list[str],
    refit_time: float | None,
) -> None:
    result_row["refit_time"] = refit_time if refit_time is not None else float("nan")
    if train_metrics:
        for m in optimize_metrics:
            result_row["final_train_" + m] = train_metrics.get(m, float("nan"))
        result_row["final_train_n"] = len(y_train_use)
    else:
        for m in optimize_metrics:
            result_row["final_train_" + m] = float("nan")
        result_row["final_train_n"] = float("nan")
    if test_metrics:
        for m in optimize_metrics:
            result_row["final_test_" + m] = test_metrics.get(m, float("nan"))
        result_row["final_test_n"] = len(y_test) if y_test is not None else float("nan")
    else:
        for m in optimize_metrics:
            result_row["final_test_" + m] = float("nan")
        result_row["final_test_n"] = float("nan")


def _run_one_benchmark_task(
    task: tuple[Any, ...],
    datasets: dict[str, pd.DataFrame],
    target_col: str,
    optimize_metrics: list[str],
    random_state: int,
    random_under_sampling_strategy: float,
    return_roc_curves: bool,
    return_per_class_counts: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one (combo, resampling, model) benchmark. Returns (result_rows, grid_rows_chunk, roc_entries)."""
    p = _prepare_benchmark_task_arrays(
        task, datasets, target_col, random_state, random_under_sampling_strategy
    )
    _, _, history = grid_search_best(
        p.factory,
        p.param_grid,
        p.X_train_use,
        p.y_train_use,
        p.X_val_s,
        p.y_val,
        optimize_metrics=optimize_metrics,
        scale=False,
        return_history=True,
        metrics_to_compute=optimize_metrics,
        random_state=random_state,
    )
    result_rows: list[dict[str, Any]] = []
    roc_entries: list[dict[str, Any]] = []
    per_class_entries: list[dict[str, Any]] = []
    val_n = len(p.y_val)
    if not history:
        return (
            _empty_result_rows_for_metrics(
                p.target,
                p.method,
                p.model_name,
                p.resampling,
                optimize_metrics,
                val_n,
                p.run_date,
            ),
            [],
            roc_entries,
            per_class_entries,
        )
    grid_rows_chunk = _grid_rows_from_search_history(
        history,
        p.target,
        p.method,
        p.model_name,
        p.resampling,
        optimize_metrics,
    )
    for opt_metric in optimize_metrics:
        best_params, best_val_m = _best_params_for_opt_metric(history, opt_metric)
        if best_params is None or best_val_m is None:
            result_rows.append(
                _result_row_when_no_best(
                    p.target,
                    p.method,
                    p.model_name,
                    p.resampling,
                    opt_metric,
                    val_n,
                    p.run_date,
                    optimize_metrics,
                )
            )
            continue
        result_row: dict[str, Any] = {
            "target": p.target,
            "fingerprint": p.method,
            "model": p.model_name,
            "resampling": p.resampling,
            "optimized_for": opt_metric,
            "best_params": str(best_params),
            "val_selection": selection_score(best_val_m, optimize_metrics),
            "val_n": val_n,
            "training_time": best_val_m.get("training_time", float("nan")),
            "run_date": p.run_date,
        }
        for m in optimize_metrics:
            result_row["val_" + m] = best_val_m[m]
        train_metrics, test_metrics, roc_data, per_class_rows, refit_time = _compute_final_metrics_and_roc(
            p.factory,
            best_params,
            p.X_train_use,
            p.y_train_use,
            p.X_val_s,
            p.y_val,
            p.X_test_s,
            p.y_test,
            optimize_metrics,
            scale=False,
            random_state=random_state,
            return_per_class_counts=return_per_class_counts,
            target=p.target,
            fingerprint=p.method,
            model_name=p.model_name,
            resampling=p.resampling,
            opt_metric=opt_metric,
            best_params_str=str(best_params),
        )
        _finalize_result_row_metrics(
            result_row,
            train_metrics,
            test_metrics,
            p.y_train_use,
            p.y_test,
            optimize_metrics,
            refit_time,
        )
        result_rows.append(result_row)
        if return_roc_curves and roc_data is not None:
            fpr, tpr, pr_prec, pr_rec = roc_data
            pos_rate = float(np.mean(p.y_val))
            roc_entries.append({
                "target": p.target,
                "fingerprint": p.method,
                "model": p.model_name,
                "resampling": p.resampling,
                "optimized_for": opt_metric,
                "fpr": fpr,
                "tpr": tpr,
                "precision": pr_prec,
                "recall": pr_rec,
                "positive_rate": pos_rate,
            })
        if per_class_rows:
            per_class_entries.extend(per_class_rows)
    return result_rows, grid_rows_chunk, roc_entries, per_class_entries


def _merge_benchmark_chunk(
    results: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    roc_curves: list[dict[str, Any]],
    per_class_counts: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    grid_rows_chunk: list[dict[str, Any]],
    roc_entries: list[dict[str, Any]],
    per_class_entries: list[dict[str, Any]],
    pbar: Any,
) -> None:
    results.extend(result_rows)
    grid_rows.extend(grid_rows_chunk)
    if roc_entries:
        roc_curves.extend(roc_entries)
    if per_class_entries:
        per_class_counts.extend(per_class_entries)
    if pbar is None:
        return
    if result_rows:
        r0 = result_rows[0]
        pbar.set_postfix_str(
            f"{r0['target'][:12]} × {r0['fingerprint']} | {r0['model']} ({r0['resampling']})"
        )
    pbar.update(1)


def _delayed_benchmark_task(
    task: tuple[Any, ...],
    datasets: dict[str, pd.DataFrame],
    target_col: str,
    optimize_metrics: list[str],
    random_state: int,
    random_under_sampling_strategy: float,
    return_roc_curves: bool,
    return_per_class_counts: bool,
) -> Any:
    return delayed(_run_one_benchmark_task)(
        task,
        datasets,
        target_col,
        optimize_metrics,
        random_state,
        random_under_sampling_strategy,
        return_roc_curves,
        return_per_class_counts,
    )


def _run_parallel_benchmark_tasks(
    tasks: list[tuple[Any, ...]],
    parallel_kw: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
    target_col: str,
    optimize_metrics: list[str],
    random_state: int,
    random_under_sampling_strategy: float,
    return_roc_curves: bool,
    return_per_class_counts: bool,
    results: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    roc_curves: list[dict[str, Any]],
    per_class_counts: list[dict[str, Any]],
    pbar: Any,
) -> None:
    try:
        parallel = Parallel(**parallel_kw, return_as="generator")
        job_iter = (
            _delayed_benchmark_task(
                task,
                datasets,
                target_col,
                optimize_metrics,
                random_state,
                random_under_sampling_strategy,
                return_roc_curves,
                return_per_class_counts,
            )
            for task in tasks
        )
        for chunk in parallel(job_iter):
            res_r, grid_c, roc_e, pc_e = chunk
            _merge_benchmark_chunk(
                results,
                grid_rows,
                roc_curves,
                per_class_counts,
                res_r,
                grid_c,
                roc_e,
                pc_e,
                pbar,
            )
    except TypeError:
        jobs = [
            _delayed_benchmark_task(
                task,
                datasets,
                target_col,
                optimize_metrics,
                random_state,
                random_under_sampling_strategy,
                return_roc_curves,
                return_per_class_counts,
            )
            for task in tasks
        ]
        for chunk in Parallel(**parallel_kw)(jobs):
            res_r, grid_c, roc_e, pc_e = chunk
            _merge_benchmark_chunk(
                results,
                grid_rows,
                roc_curves,
                per_class_counts,
                res_r,
                grid_c,
                roc_e,
                pc_e,
                pbar,
            )


def run_benchmark(
    combos: list[tuple[str, str, tuple[str, str, str]]],
    datasets: dict[str, pd.DataFrame],
    model_configs: list[tuple[str, Callable, Any]],
    resampling_variants: list[str],
    imbalance_models: set[str],
    optimize_metrics: list[str],
    target_col: str = "target",
    random_state: int = DEFAULT_RANDOM_STATE,
    use_tqdm: bool = True,
    tqdm_desc: str = "Benchmark",
    random_under_sampling_strategy: float = 0.5,
    return_roc_curves: bool = True,
    return_per_class_counts: bool = False,
    use_thread_pool: bool = False,
    max_workers: int | None = None,
    parallel_backend: str = "loky",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Run full benchmark: for each (target, method), each resampling, each model (skipping
    imbalance models when resampling != 'none'), run grid search and collect results.
    Returns (results_list, grid_rows, roc_curves, per_class_counts). When return_roc_curves=True,
    each entry includes ROC (fpr, tpr) and precision-recall (precision, recall) plus positive_rate on validation. When return_per_class_counts=True,
    computes per-class prediction counts during the benchmark (no retraining). per_class_counts
    is [] when the flag is False. When use_thread_pool=True, runs tasks in parallel:
    parallel_backend ``loky`` (process pool, multi-core) or ``threading`` (GIL-limited for CPU-bound sklearn).
    """
    model_names = [c[0] for c in model_configs]
    total_runs = count_benchmark_runs(
        len(combos), model_names, resampling_variants, imbalance_models
    )
    results: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    roc_curves: list[dict[str, Any]] = []
    per_class_counts: list[dict[str, Any]] = []

    pbar = tqdm(total=total_runs, desc=tqdm_desc, unit="run") if use_tqdm else None

    try:
        tasks = _build_benchmark_tasks(
            combos, resampling_variants, model_configs, imbalance_models
        )
        n_tasks = len(tasks)
        if pbar is not None:
            pbar.total = n_tasks
        if use_thread_pool:
            n_workers = (
                max_workers
                if max_workers is not None
                else min(32, (os.cpu_count() or 1) + 1)
            )
            if parallel_backend not in ("loky", "threading"):
                raise ValueError("parallel_backend must be 'loky' or 'threading'")
            parallel_kw: dict[str, Any] = {
                "n_jobs": n_workers,
                "backend": parallel_backend,
            }
            if parallel_backend == "loky":
                # setting to prevent process kiling
                parallel_kw["pre_dispatch"] = "2*n_jobs"
                parallel_kw["timeout"] = 9999
            _run_parallel_benchmark_tasks(
                tasks,
                parallel_kw,
                datasets,
                target_col,
                optimize_metrics,
                random_state,
                random_under_sampling_strategy,
                return_roc_curves,
                return_per_class_counts,
                results,
                grid_rows,
                roc_curves,
                per_class_counts,
                pbar,
            )
        else:
            for task in tasks:
                res_r, grid_c, roc_e, pc_e = _run_one_benchmark_task(
                    task,
                    datasets,
                    target_col,
                    optimize_metrics,
                    random_state,
                    random_under_sampling_strategy,
                    return_roc_curves,
                    return_per_class_counts,
                )
                _merge_benchmark_chunk(
                    results,
                    grid_rows,
                    roc_curves,
                    per_class_counts,
                    res_r,
                    grid_c,
                    roc_e,
                    pc_e,
                    pbar,
                )

    finally:
        if pbar is not None:
            pbar.close()

    return results, grid_rows, roc_curves, per_class_counts
