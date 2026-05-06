"""
Dataset building, splitting, and loading (fingerprint + target → train/val/test).
"""

from bpr.datasets.build import (
    _split_indices,
    create_dataset,
    drop_duplicate_problems,
    generate_datasets,
    generate_datasets_dedup_all,
    load_datasets,
    parse_dataset_key,
    targets_and_methods_from_keys,
)

__all__ = [
    "_split_indices",
    "create_dataset",
    "drop_duplicate_problems",
    "generate_datasets",
    "generate_datasets_dedup_all",
    "load_datasets",
    "parse_dataset_key",
    "targets_and_methods_from_keys",
]
