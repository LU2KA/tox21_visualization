"""Shared helpers for per-split, per-class prediction counts (binary labels)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np


def is_binary_classification_y(y: np.ndarray) -> bool:
    """Return True if ``y`` is a binary or single-class {0}/{1} label vector."""
    unique = np.unique(y)
    return bool(
        np.array_equal(unique, [0, 1])
        or np.array_equal(unique, [0])
        or np.array_equal(unique, [1])
        or np.array_equal(unique, [1, 0])
    )


def iter_per_class_correct_wrong(
    preds: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> Iterator[tuple[str, int, int, int, int]]:
    """Yield (split_name, class_id, n_true, n_correct, n_wrong) for binary splits."""
    for split, (y_true, y_pred) in preds.items():
        if not is_binary_classification_y(y_true):
            continue
        for cls in (0, 1):
            mask = y_true == cls
            n_true = int(mask.sum())
            if n_true == 0:
                continue
            correct = int(((y_true == cls) & (y_pred == cls)).sum())
            wrong = n_true - correct
            yield split, cls, n_true, correct, wrong


def per_class_row_core_fields(
    split: str, cls: int, n_true: int, n_correct: int, n_wrong: int
) -> dict[str, Any]:
    """Common dict fields for per-class count rows (split / class / counts)."""
    return {
        "split": split,
        "class": cls,
        "n_true": n_true,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
    }


def per_class_benchmark_count_rows(
    preds: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    target: str,
    model_name: str,
    fingerprint: str,
    opt_metric: str,
    resampling: str,
    best_params_str: str,
) -> list[dict[str, Any]]:
    """Per-class count rows for benchmark `_compute_final_metrics_and_roc` output."""
    rows: list[dict[str, Any]] = []
    for split, cls, n_true, correct, wrong in iter_per_class_correct_wrong(preds):
        rows.append({
            "target": target,
            "model": model_name,
            "finger_print_method": fingerprint,
            "optimized_for": opt_metric,
            "resampling": resampling,
            "parameters": best_params_str,
            **per_class_row_core_fields(split, cls, n_true, correct, wrong),
        })
    return rows


def per_class_reporting_count_rows(
    preds: Mapping[str, tuple[np.ndarray, np.ndarray]],
    r: Any,
    *,
    target: str,
    model_name: str,
    fingerprint: str,
    opt_for: Any,
) -> list[dict[str, Any]]:
    """Per-class count rows for post-hoc reporting (`compute_per_class_counts`)."""
    rows: list[dict[str, Any]] = []
    for split, cls, n_true, correct, wrong in iter_per_class_correct_wrong(preds):
        row: dict[str, Any] = {
            "target": target,
            "model": model_name,
            "finger_print_method": fingerprint,
            "optimized_for": opt_for,
            "parameters": r["best_params"],
            **per_class_row_core_fields(split, cls, n_true, correct, wrong),
        }
        if "resampling" in r.index:
            row["resampling"] = r["resampling"]
        rows.append(row)
    return rows
