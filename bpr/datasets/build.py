"""
Dataset generation and loading utilities.

Folder layout when saving (fingerprint / target):
"""

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from bpr.constants import DATASET_EXT, SPLIT_NAMES
from bpr.features.fingerprints import compute_fingerprints


def _save_dataframe(
    df: pd.DataFrame, path: Path, fmt: str, verbose: bool = False
) -> None:
    """Save DataFrame to CSV. fmt must be 'csv'; path should have .csv extension."""
    if fmt.lower() != "csv":
        raise ValueError(f"Only CSV is supported; got save_format={fmt!r}")
    df.to_csv(path, index=False)
    if verbose:
        print(f"  Saved to {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")


def _print_dataset_stats(combined_df: pd.DataFrame, fp_df: pd.DataFrame, split_series: pd.Series | None) -> None:
    """Print dataset statistics."""
    if split_series is not None:
        n_train = (combined_df["split"] == "train").sum()
        n_val = (combined_df["split"] == "val").sum()
        n_test = (combined_df["split"] == "test").sum()
        print(f"  Split: train={n_train}, val={n_val}, test={n_test}")
    print(
        f"  Dataset: {len(combined_df)} samples, "
        f"{len(fp_df.columns)} features, "
        f"target: {combined_df['target'].value_counts().to_dict()}"
    )

def _save_dataset_splits(
    combined_df: pd.DataFrame,
    save_splits: tuple[str, str, str],
    save_format: str,
    verbose: bool
) -> None:
    """Save train, val, and test splits to disk."""
    train_path, val_path, test_path = save_splits
    for path, part in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
        subset = combined_df[combined_df["split"] == part].drop(columns=["split"])
        if not subset.empty:
            _save_dataframe(subset, Path(path), save_format, verbose=verbose)

def create_dataset(
    df: pd.DataFrame,
    target_column: str,
    fingerprint_method: str = "morgan",
    fingerprint_params: dict[str, Any] | None = None,
    drop_na_target: bool = True,
    save_path: str | Path | None = None,
    save_format: str = "csv",
    split_series: pd.Series | None = None,
    save_splits: tuple[str, str, str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one dataset: fingerprints for a single target and fingerprint method.

    Filters to rows with valid smiles and non-null target (if drop_na_target),
    computes fingerprints, then returns a DataFrame with fingerprint columns
    plus a 'target' column. Optionally adds a 'split' column and saves to CSV.

    Args:
        df: DataFrame with 'smiles' and the target column.
        target_column: Name of the target column (e.g. 'NR-AR').
        fingerprint_method: Method name (see fingerprints.SUPPORTED_METHODS).
        fingerprint_params: Extra kwargs for the fingerprint function.
        drop_na_target: If True, drop rows where target is NaN.
        save_path: If set, save the full dataset to this path (CSV).
        save_format: Must be 'csv'.
        split_series: Optional Series mapping df index to 'train'|'val'|'test'.
        save_splits: Optional (train_path, val_path, test_path); requires split_series.
        verbose: If True, print sample counts and save paths.

    Returns:
        DataFrame with fingerprint columns, 'target', and optionally 'split'.
    """
    if "smiles" not in df.columns:
        raise ValueError("DataFrame must contain a 'smiles' column")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    if save_splits is not None and split_series is None:
        raise ValueError("save_splits requires split_series")

    params = fingerprint_params or {}

    if drop_na_target:
        df_clean = df[df[target_column].notna()].copy()
    else:
        df_clean = df.copy()

    df_clean = df_clean[df_clean["smiles"].notna()].copy()
    df_clean = df_clean[df_clean["smiles"].astype(str).str.strip() != ""].copy()

    if df_clean.empty:
        raise ValueError("No valid SMILES strings found")

    fp_df = compute_fingerprints(
        df_clean["smiles"], method=fingerprint_method, verbose=verbose, **params
    )

    df_clean = df_clean.loc[fp_df.index]
    combined_df = pd.concat([
        fp_df,
        df_clean[[target_column]].rename(columns={target_column: "target"}),
    ], axis=1)
    combined_df = combined_df.dropna(subset=fp_df.columns.tolist())

    if split_series is not None:
        combined_df["split"] = split_series.reindex(combined_df.index).values
        if save_splits is not None:
            combined_df = combined_df[combined_df["split"].notna()].copy()

    if verbose:
        _print_dataset_stats(combined_df, fp_df, split_series)

    if save_path is not None:
        _save_dataframe(combined_df, Path(save_path), save_format, verbose=verbose)

    if save_splits is not None and split_series is not None:
        _save_dataset_splits(combined_df, save_splits, save_format, verbose)

    return combined_df


def _split_indices(
    indices: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split  into train/val/test using sklearn.train_test_split."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_frac, random_state=random_state, shuffle=True
    )
    val_frac_of_remaining = val_frac / (1.0 - test_frac) if test_frac < 1.0 else 0.0
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_frac_of_remaining,
        random_state=random_state,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def _generate_scaffolds(smiles_series: pd.Series) -> pd.Series:
    """Generate Murcko scaffolds for a Series of SMILES strings."""
    def get_scaffold(s):
        if pd.isna(s) or not str(s).strip():
            return None
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    return smiles_series.apply(get_scaffold)


def _split_indices_scaffold(
    df: pd.DataFrame,
    target_col: str,
    smiles_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train/val/test using StratifiedGroupKFold on Murcko scaffolds.
    Returns integer positions (iloc indices) relative to the input df.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    scaffolds = _generate_scaffolds(df[smiles_col])
    # Fill missing scaffolds with unique values so they don't group together
    missing_mask = scaffolds.isna()
    if missing_mask.any():
        scaffolds.loc[missing_mask] = [f"missing_{i}" for i in range(missing_mask.sum())]

    y = df[target_col].values
    groups = scaffolds.values
    indices = np.arange(len(df))

    # Split into (train+val) and test
    if test_frac > 0.0:
        n_splits_test = max(2, int(np.round(1.0 / test_frac)))
        sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_state)

        train_val_idx, test_idx = next(sgkf_test.split(indices, y, groups))
    else:
        train_val_idx = indices
        test_idx = np.array([], dtype=int)

    # Split (train+val) into train and val
    if val_frac > 0.0:
        val_frac_relative = val_frac / (train_frac + val_frac)
        n_splits_val = max(2, int(np.round(1.0 / val_frac_relative)))

        y_train_val = y[train_val_idx]
        groups_train_val = groups[train_val_idx]

        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)
        train_idx_relative, val_idx_relative = next(sgkf_val.split(train_val_idx, y_train_val, groups_train_val))

        train_idx = train_val_idx[train_idx_relative]
        val_idx = train_val_idx[val_idx_relative]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    return train_idx, val_idx, test_idx


def drop_duplicate_problems(
    df: pd.DataFrame,
    targets: list[str],
    smiles_col: str = "smiles",
    keep_one_per_canonical: bool = True,
) -> pd.DataFrame:
    """Drop invalid SMILES, remove conflicts, and optionally deduplicate by canonical SMILES."""
    if smiles_col not in df.columns:
        raise ValueError(f"DataFrame must contain column '{smiles_col}'")
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Targets not in df: {missing}")

    mols = (
        df[smiles_col]
        .dropna()
        .astype(str)
        .apply(lambda s: Chem.MolFromSmiles(s.strip()) if s.strip() else None)
    )
    valid_mask = mols.notna()
    if not valid_mask.any():
        return df.copy()
    canonical = mols[valid_mask].apply(Chem.MolToSmiles)
    canonical_aligned = canonical.reindex(df.index)

    df_work = df.loc[canonical_aligned.dropna().index].copy()
    df_work["_canon"] = canonical_aligned.dropna()

    inconsistent = [
        can
        for can, grp in df_work.groupby("_canon")
        if any(
            len(vals := grp[t].dropna()) >= 2 and vals.nunique() > 1
            for t in targets
        )
    ]

    if inconsistent:
        df_work = df_work[~df_work["_canon"].isin(inconsistent)]

    if keep_one_per_canonical:
        df_work = df_work.drop_duplicates(subset=["_canon"], keep="first")

    df_work[smiles_col] = df_work["_canon"].values
    df_work = df_work.drop(columns=["_canon"])
    return df_work.reset_index(drop=True)


def canonicalize_smiles_column(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Replace SMILES with canonical SMILES (no deduplication)."""
    if smiles_col not in df.columns:
        return df.copy()

    def canon(s):
        if pd.isna(s) or not isinstance(s, str) or not s.strip():
            return s
        m = Chem.MolFromSmiles(s.strip())
        return Chem.MolToSmiles(m) if m is not None else s

    out = df.copy()
    out[smiles_col] = out[smiles_col].apply(canon)
    return out


def _save_original_dedup_splits(
    df: pd.DataFrame,
    combined: pd.DataFrame,
    split_series: pd.Series,
    target: str,
    smiles_col: str,
    output_dir: Path,
    ext: str,
    save_format: str,
    verbose: bool
) -> None:
    """Save the original dataset splits (before fingerprinting) to disk."""
    orig_cols = [c for c in [smiles_col, "DSSTox_CID", "Formula", "FW", target] if c in df.columns]
    df_orig = combined[orig_cols].copy()
    df_orig = df_orig.rename(columns={target: "target"})
    df_orig["split"] = split_series
    original_target_dir = output_dir / "original" / target
    original_target_dir.mkdir(parents=True, exist_ok=True)
    for part in SPLIT_NAMES:
        subset = df_orig[df_orig["split"] == part].drop(columns=["split"])
        if not subset.empty:
            _save_dataframe(
                subset, original_target_dir / f"{part}{ext}", save_format, verbose=verbose
            )
    if verbose:
        print(f"  Saved original split: {output_dir / 'original' / target}/train|val|test{ext}")

def _process_single_target_dedup(
    df: pd.DataFrame,
    target: str,
    smiles_col: str,
    test_df: pd.DataFrame | None,
    split_fracs: tuple[float, float, float],
    random_state: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter, deduplicate, and split data for a single target."""
    train_frac, val_frac, test_frac = split_fracs
    valid_mask = (
        df[target].notna()
        & df[smiles_col].notna()
        & (df[smiles_col].astype(str).str.strip() != "")
    )
    valid_df = df.loc[valid_mask].copy()

    if valid_df.empty:
        raise ValueError(f"No valid rows for target '{target}'")

    clean_df = drop_duplicate_problems(
        valid_df, [target], smiles_col=smiles_col, keep_one_per_canonical=True
    )

    if test_df is not None:
        train_f, val_f, _ = split_fracs
        val_relative = val_f / (train_f + val_f)

        # Use StratifiedGroupKFold on scaffolds for train/val split
        scaffolds = _generate_scaffolds(clean_df[smiles_col])
        missing_mask = scaffolds.isna()
        if missing_mask.any():
            scaffolds.loc[missing_mask] = [f"missing_{i}" for i in range(missing_mask.sum())]

        y = clean_df[target].values
        groups = scaffolds.values
        indices = np.arange(len(clean_df))

        n_splits_val = max(2, int(np.round(1.0 / val_relative)))
        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)

        train_idx_rel, val_idx_rel = next(sgkf_val.split(indices, y, groups))

        train_split_df = clean_df.iloc[train_idx_rel].copy()
        val_split_df = clean_df.iloc[val_idx_rel].copy()

        valid_test_mask = (
            test_df[target].notna()
            & test_df[smiles_col].notna()
            & (test_df[smiles_col].astype(str).str.strip() != "")
        ) if target in test_df.columns else pd.Series(False, index=test_df.index)
        test_split_df = test_df.loc[valid_test_mask].copy()
    else:
        train_idx_rel, val_idx_rel, test_idx_rel = _split_indices_scaffold(
            clean_df, target, smiles_col, train_frac, val_frac, test_frac, random_state
        )
        train_split_df = clean_df.iloc[train_idx_rel].copy()
        val_split_df = clean_df.iloc[val_idx_rel].copy()
        test_split_df = clean_df.iloc[test_idx_rel].copy()

    combined = pd.concat([train_split_df, val_split_df, test_split_df], ignore_index=True)
    n_train = len(train_split_df)
    n_val = len(val_split_df)
    split_series = pd.Series(index=combined.index, dtype=object)
    split_series.iloc[:n_train] = "train"
    split_series.iloc[n_train : n_train + n_val] = "val"
    split_series.iloc[n_train + n_val :] = "test"

    return combined, split_series

def generate_datasets_dedup_all(
    df: pd.DataFrame,
    targets: list[str],
    fingerprints: list[dict[str, Any]],
    output_dir: str | Path,
    split_fracs: tuple[float, float, float],
    random_state: int | None = None,
    save_format: str = "csv",
    save_original_dataset: bool = True,
    smiles_col: str = "smiles",
    verbose: bool = False,
    test_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate datasets with train/val/test split after deduplicating the entire dataset.

    Drops duplicate molecules and invalid/conflicting labels for each target on the
    *entire* dataset first, and then splits the clean dataset into train/val/test.
    This prevents data leakage between splits. The output uses canonical SMILES
    in the smiles column.

    Args:
        df: Full DataFrame (must have smiles + targets).
        targets: Target column names.
        fingerprints: List of dicts with "method" + optional params.
        output_dir: Root directory for saved files (e.g. "data/datasets_dedup").
        split_fracs: (train_frac, val_frac, test_frac).
        random_state: Seed for splits.
        save_format: 'csv' (only CSV is supported).
        save_original_dataset: If True, save original/full.csv and original/{target}/train|val|test.
        smiles_col: SMILES column name for drop_duplicate_problems.

    Returns:
        Dict mapping "{target}_{method}" -> DataFrame (with 'split' column).
    """
    for fp in fingerprints:
        if "method" not in fp:
            raise ValueError(f"Each fingerprint config must contain 'method', got: {fp}")

    output_dir = Path(output_dir)
    train_frac, val_frac, test_frac = split_fracs
    ext = DATASET_EXT

    if verbose:
        print(f"Generating fully deduplicated datasets ({len(targets)} targets × {len(fingerprints)} methods)...")
        print(f"Split: train={train_frac}, val={val_frac}, test={test_frac} (random_state={random_state})")
        print(f"Saving to: {output_dir}/{{method}}/{{target}}/train|val|test{ext}")

    if save_original_dataset:
        original_dir = output_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        full_path = original_dir / f"full{ext}"
        _save_dataframe(df, full_path, save_format, verbose=verbose)
        if verbose:
            print(f"Saved original full dataset: {full_path}")

    datasets: dict[str, pd.DataFrame] = {}

    for target in targets:
        combined, split_series = _process_single_target_dedup(
            df, target, smiles_col, test_df, split_fracs, random_state
        )

        if save_original_dataset:
            _save_original_dedup_splits(
                df, combined, split_series, target, smiles_col, output_dir, ext, save_format, verbose
            )

        for fp_config in fingerprints:
            method = fp_config["method"]
            params = {k: v for k, v in fp_config.items() if k != "method"}
            key = f"{target}_{method}"
            if verbose:
                print(f"Processing: {key}  params={params}")

            method_dir = output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            target_dir = method_dir / target
            target_dir.mkdir(parents=True, exist_ok=True)
            save_splits = (
                target_dir / f"train{ext}",
                target_dir / f"val{ext}",
                target_dir / f"test{ext}",
            )

            dataset = create_dataset(
                df=combined,
                target_column=target,
                fingerprint_method=method,
                fingerprint_params=params,
                save_path=None,
                save_format=save_format,
                split_series=split_series,
                save_splits=save_splits,
                verbose=verbose,
            )
            datasets[key] = dataset

    if verbose:
        print(f"Generated {len(datasets)} datasets (fully deduplicated). Saved to: {output_dir}")
    return datasets


def split_raw_10k_dataset(
    df: pd.DataFrame,
    targets: list[str],
    split_fracs: tuple[float, float, float],
    random_state: int | None,
    smiles_col: str = "smiles",
    test_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a raw (non-deduplicated) TOX21 10k-style DataFrame into train/val/test,
    using the same split fractions and random state configuration as the training
    pipeline. Returns three DataFrames built from df (train, val, test).

    The split is defined per-target using the shared helper `_split_indices` and
    the same validity mask as in `generate_datasets`. When `test_df` is provided,
    the test split is drawn from that DataFrame, mirroring how challenge
    score/test data are handled elsewhere.

    Args:
        df: Full raw DataFrame (must contain smiles + target columns).
        targets: Target column names.
        split_fracs: (train_frac, val_frac, test_frac).
        random_state: Seed for reproducible splits.
        smiles_col: Name of the SMILES column.
        test_df: Optional external DataFrame providing test/score rows.

    Returns:
        (df_train_raw, df_val_raw, df_test_raw)
    """
    _ = targets
    train_frac, val_frac, test_frac = split_fracs
    ref_target = "NR-AR"
    if ref_target not in df.columns:
        raise ValueError(f"Reference target '{ref_target}' not found in raw dataset columns")

    valid_mask = (
        df[ref_target].notna()
        & df[smiles_col].notna()
        & (df[smiles_col].astype(str).str.strip() != "")
    )
    valid_idx = np.asarray(df.index[valid_mask])
    if len(valid_idx) == 0:
        raise ValueError(f"No valid rows for reference target '{ref_target}'")

    if test_df is not None:
        train_f, val_f, _ = split_fracs
        val_relative = val_f / (train_f + val_f)

        # Use StratifiedGroupKFold on scaffolds for train/val split
        valid_df = df.loc[valid_idx].copy()
        scaffolds = _generate_scaffolds(valid_df[smiles_col])
        missing_mask = scaffolds.isna()
        if missing_mask.any():
            scaffolds.loc[missing_mask] = [f"missing_{i}" for i in range(missing_mask.sum())]

        y = valid_df[ref_target].values
        groups = scaffolds.values
        indices = np.arange(len(valid_df))

        n_splits_val = max(2, int(np.round(1.0 / val_relative)))
        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)

        train_idx_rel, val_idx_rel = next(sgkf_val.split(indices, y, groups))

        # Map relative indices back to original DataFrame indices
        train_idx = valid_idx[train_idx_rel]
        val_idx = valid_idx[val_idx_rel]

        if smiles_col in test_df.columns:
            valid_test_mask = (
                test_df[smiles_col].notna()
                & (test_df[smiles_col].astype(str).str.strip() != "")
            )
            test_split_df = test_df.loc[valid_test_mask].copy()
        else:
            test_split_df = test_df.copy()
    else:
        valid_df = df.loc[valid_idx].copy()
        train_idx_rel, val_idx_rel, test_idx_rel = _split_indices_scaffold(
            valid_df, ref_target, smiles_col, train_frac, val_frac, test_frac, random_state
        )
        train_idx = valid_idx[train_idx_rel]
        val_idx = valid_idx[val_idx_rel]
        test_idx = valid_idx[test_idx_rel]
        test_split_df = df.loc[test_idx].copy()

    df_train_raw = df.loc[train_idx].copy()
    df_val_raw = df.loc[val_idx].copy()

    return df_train_raw.reset_index(drop=True), df_val_raw.reset_index(drop=True), test_split_df.reset_index(drop=True)


def _scaffold_train_val_idx(
    valid_df: pd.DataFrame,
    valid_idx: np.ndarray,
    target: str,
    val_relative: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    scaffolds = _generate_scaffolds(valid_df["smiles"])
    missing_mask = scaffolds.isna()
    if missing_mask.any():
        scaffolds.loc[missing_mask] = [f"missing_{i}" for i in range(missing_mask.sum())]
    y = valid_df[target].values
    groups = scaffolds.values
    indices = np.arange(len(valid_df))
    n_splits_val = max(2, int(np.round(1.0 / val_relative)))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)
    train_idx_rel, val_idx_rel = next(sgkf_val.split(indices, y, groups))
    return valid_idx[train_idx_rel], valid_idx[val_idx_rel]


def _concat_labeled_splits(
    train_df_split: pd.DataFrame,
    val_df_split: pd.DataFrame,
    test_split_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    combined = pd.concat([train_df_split, val_df_split, test_split_df], ignore_index=True)
    n_train = len(train_df_split)
    n_val = len(val_df_split)
    split_series = pd.Series(index=combined.index, dtype=object)
    split_series.iloc[:n_train] = "train"
    split_series.iloc[n_train : n_train + n_val] = "val"
    split_series.iloc[n_train + n_val :] = "test"
    return combined, split_series


def _combined_frame_and_split_for_target(
    df: pd.DataFrame,
    target: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int | None,
    test_df: pd.DataFrame | None,
    split_fracs: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.Series | None]:
    valid_mask = (
        df[target].notna()
        & df["smiles"].notna()
        & (df["smiles"].astype(str).str.strip() != "")
    )
    valid_idx = np.asarray(df.index[valid_mask])
    if len(valid_idx) == 0:
        raise ValueError(f"No valid rows for target '{target}'")
    valid_df = df.loc[valid_idx].copy()
    if test_df is not None:
        train_f, val_f, _ = split_fracs
        val_relative = val_f / (train_f + val_f)
        train_idx, val_idx = _scaffold_train_val_idx(
            valid_df, valid_idx, target, val_relative, random_state
        )
        valid_test_mask = (
            test_df[target].notna()
            & test_df["smiles"].notna()
            & (test_df["smiles"].astype(str).str.strip() != "")
        ) if target in test_df.columns else pd.Series(False, index=test_df.index)
        test_split_df = test_df.loc[valid_test_mask].copy()
    else:
        train_idx_rel, val_idx_rel, test_idx_rel = _split_indices_scaffold(
            valid_df, target, "smiles", train_frac, val_frac, test_frac, random_state
        )
        train_idx = valid_idx[train_idx_rel]
        val_idx = valid_idx[val_idx_rel]
        test_idx = valid_idx[test_idx_rel]
        test_split_df = df.loc[test_idx].copy()

    train_df_split = df.loc[train_idx].copy()
    val_df_split = df.loc[val_idx].copy()
    return _concat_labeled_splits(train_df_split, val_df_split, test_split_df)


def _save_original_splits_for_target(
    target_df: pd.DataFrame,
    split_series: pd.Series,
    target: str,
    output_dir: Path,
    ext: str,
    save_format: str,
    verbose: bool,
) -> None:
    orig_cols = [c for c in ["smiles", "DSSTox_CID", "Formula", "FW", target] if c in target_df.columns]
    df_orig = target_df[orig_cols].copy()
    df_orig = df_orig.rename(columns={target: "target"})
    df_orig["split"] = split_series.values
    df_orig = df_orig[df_orig["split"].notna()]
    original_dir = output_dir / "original" / target
    original_dir.mkdir(parents=True, exist_ok=True)
    for part in SPLIT_NAMES:
        subset = df_orig[df_orig["split"] == part].drop(columns=["split"])
        if not subset.empty:
            _save_dataframe(subset, original_dir / f"{part}{ext}", save_format, verbose=verbose)
    if verbose:
        print(f"  Saved original split: {original_dir}/train|val|test{ext}")


def _append_datasets_for_target_fingerprints(
    datasets: dict[str, pd.DataFrame],
    target: str,
    target_df: pd.DataFrame,
    split_series: pd.Series | None,
    fingerprints: list[dict[str, Any]],
    use_splits: bool,
    save_datasets: bool,
    output_dir: Path | None,
    ext: str,
    save_format: str,
    verbose: bool,
) -> None:
    for fp_config in fingerprints:
        method = fp_config["method"]
        params = {k: v for k, v in fp_config.items() if k != "method"}
        key = f"{target}_{method}"
        if verbose:
            print(f"Processing: {key}  params={params}")
        save_path = None
        save_splits = None
        if save_datasets and output_dir:
            method_dir = output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            if use_splits:
                target_dir = method_dir / target
                target_dir.mkdir(parents=True, exist_ok=True)
                save_splits = (
                    target_dir / f"train{ext}",
                    target_dir / f"val{ext}",
                    target_dir / f"test{ext}",
                )
            else:
                save_path = method_dir / f"{target}{ext}"
        datasets[key] = create_dataset(
            df=target_df,
            target_column=target,
            fingerprint_method=method,
            fingerprint_params=params,
            save_path=save_path,
            save_format=save_format,
            split_series=split_series,
            save_splits=save_splits,
            verbose=verbose,
        )


def _setup_generation_and_print_stats(
    df: pd.DataFrame,
    targets: list[str],
    fingerprints: list[dict[str, Any]],
    output_dir: Path | None,
    use_splits: bool,
    split_fracs: tuple[float, float, float] | None,
    random_state: int | None,
    save_datasets: bool,
    save_original_dataset: bool,
    save_format: str,
    verbose: bool,
    ext: str
) -> None:
    """Print stats and save the original full dataset if requested."""
    if verbose:
        method_names = [fp["method"] for fp in fingerprints]
        total = len(targets) * len(fingerprints)
        print(f"Generating {total} dataset combinations...")
        print(f"Targets: {targets}")
        print(f"Fingerprint methods: {method_names}")
        if use_splits and split_fracs is not None:
            train_frac, val_frac, test_frac = split_fracs
            print(f"Split: train={train_frac}, val={val_frac}, test={test_frac} (random_state={random_state})")
        if save_datasets:
            if use_splits:
                print(f"Saving to: {output_dir}/{{method}}/{{target}}/train|val|test{ext}")
                if save_original_dataset:
                    print(f"Original: {output_dir}/original/full{ext} + original/{{target}}/train|val|test{ext}")
            else:
                print(f"Saving to: {output_dir}/{{method}}/{{target}}{ext}")

    if save_datasets and output_dir and save_original_dataset:
        original_dir = output_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        full_path = original_dir / f"full{ext}"
        _save_dataframe(df, full_path, save_format, verbose=verbose)
        if verbose:
            print(f"Saved original full dataset: {full_path}")

def generate_datasets(
    df: pd.DataFrame,
    targets: list[str],
    fingerprints: list[dict[str, Any]],
    output_dir: str | Path | None = None,
    save_format: str = "csv",
    save_datasets: bool = False,
    split_fracs: tuple[float, float, float] | None = None,
    random_state: int | None = None,
    save_original_dataset: bool = False,
    verbose: bool = False,
    test_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate datasets for every target x fingerprint combination.

    When split_fracs is set, the same train/val/test split is used for all
    fingerprint methods (per target). Saves under method/target/ as
    train.{ext}, val.{ext}, test.{ext}. If save_original_dataset is True,
    creates data/datasets/original/ with full.csv (raw data) and, when using
    splits, original/<target>/train|val|test.{ext}.

    Args:
        df: DataFrame with 'smiles' and target columns
        targets: Target column names
        fingerprints: List of dicts, each with "method" + optional params
        output_dir: Root directory for saved files (e.g. "data/datasets")
        save_format: 'csv' (only CSV is supported)
        save_datasets: Whether to write each dataset to disk
        split_fracs: Optional (train_frac, val_frac, test_frac), e.g. (0.7, 0.15, 0.15)
        random_state: Seed for reproducible splits (used when split_fracs is set)
        save_original_dataset: If True, create original/ folder with full raw dataset and splits.

    Returns:
        Dict mapping "{target}_{method}" -> DataFrame (with 'split' column if split_fracs set)
    """
    for fp in fingerprints:
        if "method" not in fp:
            raise ValueError(f"Each fingerprint config must contain 'method', got: {fp}")

    if save_datasets and not output_dir:
        raise ValueError("output_dir is required when save_datasets=True")

    if output_dir:
        output_dir = Path(output_dir)

    use_splits = split_fracs is not None
    datasets: dict[str, pd.DataFrame] = {}
    ext = DATASET_EXT

    _setup_generation_and_print_stats(
        df, targets, fingerprints, output_dir, use_splits, split_fracs,
        random_state, save_datasets, save_original_dataset, save_format, verbose, ext
    )

    for target in targets:
        split_series = None
        if use_splits:
            assert split_fracs is not None
            train_frac, val_frac, test_frac = split_fracs
            target_df, split_series = _combined_frame_and_split_for_target(
                df,
                target,
                train_frac,
                val_frac,
                test_frac,
                random_state,
                test_df,
                split_fracs,
            )
        else:
            target_df = df

        _append_datasets_for_target_fingerprints(
            datasets,
            target,
            target_df,
            split_series,
            fingerprints,
            use_splits,
            save_datasets,
            output_dir,
            ext,
            save_format,
            verbose,
        )

        if save_datasets and output_dir and use_splits and save_original_dataset:
            assert split_series is not None
            _save_original_splits_for_target(
                target_df,
                split_series,
                target,
                output_dir,
                ext,
                save_format,
                verbose,
            )

    if verbose:
        msg = f"Generated {len(datasets)} datasets."
        if save_datasets:
            msg += f" Saved to: {output_dir}"
        print(msg)
    return datasets


def parse_dataset_key(key: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse a dataset key into (target, split, method).

    Key format is "{target}_{split}_{method}" (e.g. "NR-AR_train_morgan").
    Method may contain underscores (e.g. "morgan_2048"), so we split on
    _train_ / _val_ / _test_ only. Returns (None, None, None) if key does not match.
    """
    for split in SPLIT_NAMES:
        sep = f"_{split}_"
        if sep in key:
            parts = key.split(sep, 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                return parts[0], split, parts[1]
    return None, None, None


def targets_and_methods_from_keys(
    keys: Iterable[str],
    skip_methods: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    From dataset keys (e.g. from load_datasets), return sorted unique targets and methods.

    If skip_methods is set, keys whose method is in skip_methods are excluded
    (e.g. skip_methods={"original"} to drop raw-feature datasets).
    """
    skip = skip_methods or set()
    targets: set[str] = set()
    methods_set: set[str] = set()
    for k in keys:
        target, _, method = parse_dataset_key(k)
        if method is not None and method not in skip:
            if target:
                targets.add(target)
            methods_set.add(method)
    return sorted(targets), sorted(methods_set)


def load_datasets(
    data_dir: str | Path,
    methods: list[str] | None = None,
    targets: list[str] | None = None,
    splits: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load previously saved datasets from the fingerprint/target folder tree.

    Layout:
    - method/target/train.csv, val.csv, test.csv (target as subdirectory)
    Loaded keys are "{target}_{split}_{method}" (e.g. "NR-AR_train_morgan").

    Args:
        data_dir: Root directory (e.g. "data/datasets")
        methods: Restrict to these fingerprint method folders (None = all)
        targets: Restrict to these target names (without _train/_val/_test)
        splits: Restrict to these split names (None = all); e.g. ["train", "val"]

    Returns:
        Dict mapping "{target}_{method}" or "{target}_{split}_{method}" -> DataFrame
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    datasets: dict[str, pd.DataFrame] = {}

    method_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (methods is None or d.name in methods)
    )

    def _read(path: Path):
        return pd.read_csv(path) if path.suffix == DATASET_EXT else None
    for method_dir in method_dirs:
        method = method_dir.name
        for item in sorted(method_dir.iterdir()):
            if item.is_dir():
                base_target = item.name
                if targets is not None and base_target not in targets:
                    continue
                for split in SPLIT_NAMES:
                    if splits is not None and split not in splits:
                        continue
                    path = item / f"{split}{DATASET_EXT}"
                    if not path.is_file():
                        continue
                    df = _read(path)
                    if df is not None:
                        datasets[f"{base_target}_{split}_{method}"] = df

    if verbose:
        print(f"Loaded {len(datasets)} datasets from {data_dir}")
    return datasets
