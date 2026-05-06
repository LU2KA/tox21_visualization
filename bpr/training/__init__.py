"""Training and benchmark utilities for model × fingerprint × target runs."""

from bpr.training.resampling import apply_resampling
from bpr.training.benchmark import (
    ALL_METRICS,
    benchmark_completeness_report,
    count_benchmark_runs,
    evaluate,
    fit_predict_eval,
    get_benchmark_combos,
    get_X_y,
    grid_search_best,
    grid_size,
    selection_score,
)
from bpr.training.benchmark_task import run_benchmark
from bpr.training.benchmark_reporting import (
    compute_per_class_counts,
    format_metrics_long,
)

__all__ = [
    "ALL_METRICS",
    "apply_resampling",
    "benchmark_completeness_report",
    "count_benchmark_runs",
    "evaluate",
    "fit_predict_eval",
    "get_benchmark_combos",
    "get_X_y",
    "grid_search_best",
    "grid_size",
    "run_benchmark",
    "selection_score",
    "format_metrics_long",
    "compute_per_class_counts",
]
