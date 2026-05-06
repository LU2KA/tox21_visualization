"""TOX21 EDA Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Draw

from bpr.constants import DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_FRACS
from bpr.datasets import _split_indices

REFERENCE_TARGET = "NR-AR"
SPLIT_FRACS = DEFAULT_SPLIT_FRACS
RANDOM_STATE = DEFAULT_RANDOM_STATE
TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]

DATA_DIR = Path("data/datasets")
FULL_CSV = DATA_DIR / "original" / "full.csv"
FALLBACK_CSV = Path("data/tox21/tox21_10k_data_all.csv")
FALLBACK_CSV_LEGACY = Path("tox21_10k_data_all.csv")

ANALYSIS_DIR = Path("data/analysis")
ANALYSIS_TRAIN_RAW = ANALYSIS_DIR / "tox21_10k_train_raw.csv"
ANALYSIS_TRAIN_RAW_THESIS = ANALYSIS_DIR / "tox21_10k_train_raw_thesis_split.csv"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "white"
CMAP_BASE = sns.color_palette("vlag", as_cmap=True)


def sample_base_palette(n: int, low: float = 0.15, high: float = 0.85):
    vals = np.linspace(low, high, n)
    return [CMAP_BASE(v) for v in vals]


PALETTE_2 = sample_base_palette(2)
PALETTE_3 = sample_base_palette(3)
PALETTE_4 = sample_base_palette(4)
CMAP_SEQ = CMAP_BASE
CMAP_DIV = CMAP_BASE
CMAP_QUAL = PALETTE_4


@st.cache_data
def load_train_df(csv_path: str) -> pd.DataFrame | None:
    """Load a full CSV and return the cached train split."""
    path = Path(csv_path)
    if not path.is_file():
        return None
    df_full = pd.read_csv(path)
    valid_mask = (
        df_full[REFERENCE_TARGET].notna()
        & df_full["smiles"].notna()
        & (df_full["smiles"].astype(str).str.strip() != "")
    )
    valid_idx = np.asarray(df_full.index[valid_mask])
    train_idx, _val_idx, _test_idx = _split_indices(valid_idx, *SPLIT_FRACS, RANDOM_STATE)
    return df_full.loc[train_idx].reset_index(drop=True)


@st.cache_data
def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Add structural descriptor columns."""
    from rdkit.Chem import Descriptors, rdMolDescriptors

    def count_halogens(smiles):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        return np.nan if mol is None else sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))

    def n_aromatic_rings(smiles):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        return np.nan if mol is None else rdMolDescriptors.CalcNumAromaticRings(mol)

    def has_substruct(smiles, pat):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        return 1.0 if (mol and pat and mol.HasSubstructMatch(pat)) else 0.0

    def safe_desc(smiles, func):
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        return func(mol) if mol is not None else np.nan

    df = df.copy()
    df["n_halogen"] = df["smiles"].apply(count_halogens)
    df["n_aromatic_rings"] = df["smiles"].apply(n_aromatic_rings)
    df["has_phenol"] = df["smiles"].apply(lambda s: has_substruct(s, Chem.MolFromSmarts("[OH]c")))
    df["has_aromatic_amine"] = df["smiles"].apply(lambda s: has_substruct(s, Chem.MolFromSmarts("[NX3H2][c]")))
    df["has_cooh"] = df["smiles"].apply(lambda s: has_substruct(s, Chem.MolFromSmarts("[CX3](=O)[OX2H1]")))
    df["MolWt"] = df["smiles"].apply(lambda s: safe_desc(s, Descriptors.MolWt))
    df["NumHDonors"] = df["smiles"].apply(lambda s: safe_desc(s, Descriptors.NumHDonors))
    df["NumHAcceptors"] = df["smiles"].apply(lambda s: safe_desc(s, Descriptors.NumHAcceptors))
    df["n_rotatable"] = df["smiles"].apply(lambda s: safe_desc(s, Descriptors.NumRotatableBonds))
    df["RingCount"] = df["smiles"].apply(lambda s: safe_desc(s, rdMolDescriptors.CalcNumRings))
    return df


def _load_app_data() -> pd.DataFrame | None:
    if ANALYSIS_TRAIN_RAW_THESIS.is_file():
        return pd.read_csv(ANALYSIS_TRAIN_RAW_THESIS)
    elif ANALYSIS_TRAIN_RAW.is_file():
        return pd.read_csv(ANALYSIS_TRAIN_RAW)
    elif FULL_CSV.is_file():
        return load_train_df(str(FULL_CSV))
    elif FALLBACK_CSV.is_file():
        return load_train_df(str(FALLBACK_CSV))
    elif FALLBACK_CSV_LEGACY.is_file():
        return load_train_df(str(FALLBACK_CSV_LEGACY))
    else:
        st.warning(
            f"Neither `{ANALYSIS_TRAIN_RAW_THESIS}` nor `{ANALYSIS_TRAIN_RAW}` nor `{FULL_CSV}` nor `{FALLBACK_CSV}` found. "
            "Upload a TOX21-style CSV (columns: smiles, NR-AR, ...) to continue."
        )
        uploaded = st.file_uploader("Upload full TOX21 CSV", type=["csv"])
        if uploaded is not None:
            df_full = pd.read_csv(uploaded)
            if REFERENCE_TARGET not in df_full.columns or "smiles" not in df_full.columns:
                st.error("CSV must contain 'smiles' and reference target columns.")
                st.stop()
            valid_mask = (
                df_full[REFERENCE_TARGET].notna()
                & df_full["smiles"].notna()
                & (df_full["smiles"].astype(str).str.strip() != "")
            )
            valid_idx = np.asarray(df_full.index[valid_mask])
            train_idx, _, _ = _split_indices(valid_idx, *SPLIT_FRACS, RANDOM_STATE)
            return df_full.loc[train_idx].reset_index(drop=True)
        return None

def run_eda():
    """Render the EDA page content."""
    df = _load_app_data()
    if df is None or df.empty:
        st.stop()

    target_choice = "All"

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset & Chemistry",
        "Resampling",
        "Structure-Assay",
        "Molecule Viewer",
    ])

    with tab1:
        _render_dataset_chemistry_tab(df, target_choice)

    with tab2:
        _render_resampling_tab()

    with tab3:
        _render_structure_assay_tab(df, target_choice)

    with tab4:
        _render_molecule_viewer(df)


def main():
    st.set_page_config(page_title="TOX21 EDA", layout="wide")
    st.title("TOX21 Dataset — Exploratory Data Analysis")
    run_eda()


def _render_dataset_overview(df: pd.DataFrame):
    st.caption("Train split only (scaffold split, 70/15/15, random_state=42).")
    st.subheader("1. Dataset Overview")
    st.write("TOX21: chemical compounds tested across 12 toxicity assays (binary labels).")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Targets", len(TARGETS))
    st.dataframe(df.head(), use_container_width=True)

def _render_duplicates_analysis(df: pd.DataFrame):
    st.subheader("1b. Duplicates")
    smiles_dup_mask = df["smiles"].duplicated(keep=False)
    n_dup_smiles_rows = smiles_dup_mask.sum()
    n_unique_smiles = df["smiles"].nunique()
    full_dup_mask = df.duplicated(keep=False)
    n_full_dup = full_dup_mask.sum()
    canon_group_dist = pd.Series(dtype=float)
    full_group_dist = pd.Series(dtype=float)

    canonical_aligned = None
    mols = df["smiles"].dropna().astype(str).apply(lambda s: Chem.MolFromSmiles(s.strip()) if s.strip() else None)
    valid_mask = mols.reindex(df.index).notna()
    n_valid = valid_mask.sum()
    if n_valid > 0:
        canonical = mols[mols.notna()].apply(lambda m: Chem.MolToSmiles(m))
        canonical_aligned = canonical.reindex(df.index)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows with duplicate SMILES (exact)", n_dup_smiles_rows)
    with col2:
        st.metric("Unique SMILES", n_unique_smiles)
    with col3:
        st.metric("Fully duplicate rows", n_full_dup)

    exact_only = smiles_dup_mask & ~full_dup_mask
    full_only = full_dup_mask & ~smiles_dup_mask
    both_exact_full = smiles_dup_mask & full_dup_mask
    if canonical_aligned is not None:
        canonical_dup_mask = canonical_aligned.duplicated(keep=False) & canonical_aligned.notna()
        canon_only = canonical_dup_mask & ~smiles_dup_mask & ~full_dup_mask
    else:
        canonical_dup_mask = pd.Series(False, index=df.index)
        canon_only = pd.Series(False, index=df.index)
    overlap_lines = [
        f"**Overlap:** {both_exact_full.sum()} rows are both exact SMILES and fully identical; "
        f"{exact_only.sum()} rows have exact SMILES duplicate but not fully identical; "
        f"{full_only.sum()} rows are fully identical but SMILES string is unique in the table."
    ]
    if canonical_aligned is not None and canon_only.any():
        overlap_lines.append(f" {canon_only.sum()} rows share a canonical structure with another row but have a unique SMILES string.")
    st.caption(" ".join(overlap_lines))

    if n_dup_smiles_rows > 0:
        st.write("**SMILES appearing more than once**")
        vc = df["smiles"].value_counts()
        dup_vc = vc[vc > 1]
        dup_smiles = dup_vc.reset_index()
        dup_smiles.columns = ["smiles", "count"]
        st.dataframe(dup_smiles, use_container_width=True)

    valid_mask = canonical_aligned.notna() if canonical_aligned is not None else pd.Series(False, index=df.index)
    n_valid = valid_mask.sum()
    if n_valid > 0 and canonical_aligned is not None:
        canonical = canonical_aligned.dropna()
        n_unique_canonical = canonical.nunique()
        st.write(f"**Canonical SMILES:** {n_unique_canonical} unique molecules among {n_valid} rows with valid SMILES.")
        if n_unique_canonical < n_valid:
            st.caption(f"Canonical duplicates: {n_valid - n_unique_canonical} rows share a molecule with another (different string, same structure).")
        st.write("**Canonical duplicates (details)**")
        canon_vc = canonical.value_counts()
        canon_dup = canon_vc[canon_vc > 1]
        canon_group_dist = canon_dup.value_counts().sort_index() if len(canon_dup) > 0 else pd.Series(dtype=float)
        df_canon = df.loc[canonical.dropna().index].copy()
        df_canon["_canon"] = canonical_aligned.loc[df_canon.index]
        distinct_per_canon = df_canon.groupby("_canon")["smiles"].nunique()
        multi_string = distinct_per_canon[distinct_per_canon > 1].sort_values(ascending=False)
        if len(multi_string) > 0:
            st.write("**Same structure, different SMILES strings:**")
            rows_list = []
            for can, _ in multi_string.head(20).items():
                subset = df_canon[df_canon["_canon"] == can]
                n_r = len(subset)
                n_s = subset["smiles"].nunique()
                example = subset["smiles"].iloc[0]
                rows_list.append({"canonical_smiles": can, "n_rows": n_r, "n_distinct_input_smiles": n_s, "example_input_smiles": example})
            st.dataframe(pd.DataFrame(rows_list), use_container_width=True)
            if len(multi_string) > 20:
                st.caption(f"Showing first 20 of {len(multi_string)} canonical SMILES with multiple distinct input strings.")
    else:
        st.write("**Canonical SMILES:** No valid molecules to compare.")

    if n_full_dup > 0:
        st.write("**Fully duplicate rows**")
        full_dup_df = df[full_dup_mask]
        try:
            groups = full_dup_df.groupby(by=list(full_dup_df.columns), dropna=False).size()
            full_group_dist = groups.value_counts().sort_index()
        except Exception:
            pass
        dup_df = df[full_dup_mask].head(50)
        st.dataframe(dup_df, use_container_width=True)

    has_canon = len(canon_group_dist) > 0
    has_full = len(full_group_dist) > 0
    if has_canon or has_full:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if has_canon:
            canon_group_dist.plot(
                kind="bar",
                ax=axes[0],
                color=PALETTE_2[0],
                edgecolor="black",
                linewidth=0.5,
            )
            axes[0].bar_label(axes[0].containers[0])
            axes[0].set_xlabel("Canonical group size (number of rows)")
            axes[0].set_ylabel("Number of canonical SMILES")
            axes[0].set_title("Canonical Structure: Group Size Distribution")
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        else:
            axes[0].axis("off")
            axes[0].set_title("Canonical Structure: Group Size Distribution")

        if has_full:
            full_group_dist.plot(
                kind="bar",
                ax=axes[1],
                color=PALETTE_3[2],
                edgecolor="black",
                linewidth=0.5,
            )
            axes[1].bar_label(axes[1].containers[0])
            axes[1].set_xlabel("Group size (identical rows)")
            axes[1].set_ylabel("Number of groups")
            axes[1].set_title("Fully Identical Rows: Group Size Distribution")
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        else:
            axes[1].axis("off")
            axes[1].set_title("Fully Identical Rows: Group Size Distribution")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if n_dup_smiles_rows > 0 and all(t in df.columns for t in TARGETS):
        st.write("**Assay agreement among duplicate SMILES**")
        vc = df["smiles"].value_counts()
        dup_smiles_list = vc[vc > 1].index.tolist()
        agreement_rows = []
        for target in TARGETS:
            agreed = 0
            conflicted = 0
            for smi in dup_smiles_list:
                target_values = df.loc[df["smiles"] == smi, target].dropna()
                if len(target_values) < 2:
                    continue
                if target_values.nunique() == 1:
                    agreed += 1
                else:
                    conflicted += 1
            total = agreed + conflicted
            pct = (100.0 * agreed / total) if total else 0
            agreement_rows.append({"target": target, "groups_agreed": agreed, "groups_conflicted": conflicted, "agreement_pct": round(pct, 1)})
        agree_df = pd.DataFrame(agreement_rows)
        st.dataframe(agree_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(agree_df["target"], agree_df["agreement_pct"], color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Agreement (%)")
        ax.set_ylabel("Target")
        ax.set_title("Assay Agreement within Duplicate-SMILES Groups")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("1c. Duplicates vs conflicting labels")
    st.caption("By canonical SMILES: duplicate rows = same structure repeated; conflicting labels = same structure with different labels.")
    if canonical_aligned is not None and canonical_aligned.notna().any() and all(t in df.columns for t in TARGETS):
        df_work = df.loc[canonical_aligned.dropna().index].copy()
        df_work["_canon"] = canonical_aligned.dropna()
        n_dup_rows = df_work.groupby("_canon").transform("size").gt(1).sum()
        inconsistent_groups = []
        for can, grp in df_work.groupby("_canon"):
            conflict = False
            for t in TARGETS:
                vals = grp[t].dropna()
                if len(vals) >= 2 and vals.nunique() > 1:
                    conflict = True
                    break
            if conflict:
                inconsistent_groups.append(can)
        n_problem_molecules = len(inconsistent_groups)
        problem_rows = df_work[df_work["_canon"].isin(inconsistent_groups)].shape[0] if inconsistent_groups else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duplicate rows (by structure)", n_dup_rows)
        with col2:
            st.metric("Conflicting label groups (same structure, different values)", n_problem_molecules)
        with col3:
            st.metric("Rows in conflicting groups", problem_rows)
    else:
        st.info("Canonical SMILES and target columns are required. Enable RDKit and use a dataset with target columns.")

def _render_missing_values(df: pd.DataFrame):
    st.subheader("2. Missing Values")
    key_cols = ["smiles", "DSSTox_CID", "Formula", "FW"]
    key_cols = [c for c in key_cols if c in df.columns]
    if key_cols:
        missing_key = df[key_cols].isna().sum()
        if missing_key.any():
            st.write("Missing in key columns:")
            st.dataframe(missing_key.to_frame("missing").T, use_container_width=True)
        else:
            st.caption("Key columns (smiles, DSSTox_CID, Formula, FW) have no missing values.")
    def _is_invalid_smiles(smiles_value):
        if pd.isna(smiles_value) or not isinstance(smiles_value, str) or not smiles_value.strip():
            return True
        return Chem.MolFromSmiles(smiles_value.strip()) is None
    invalid_mask = df["smiles"].apply(_is_invalid_smiles)
    n_invalid = invalid_mask.sum()
    st.write(f"**Invalid or empty SMILES:** {n_invalid} rows.")
    if n_invalid > 0:
        with st.expander("First invalid SMILES"):
            invalid_smiles = df.loc[invalid_mask, "smiles"].head(20).tolist()
            st.write(", ".join(repr(s) for s in invalid_smiles))

    missing = df[TARGETS].isna().sum().sort_values(ascending=False)
    present = df[TARGETS].notna().sum().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(missing.index, missing.values, color=PALETTE_2[1])
    axes[0].set_xlabel("Missing labels")
    axes[0].set_title("Missing values per target")
    axes[1].barh(present.index, present.values, color=PALETTE_2[0])
    axes[1].set_xlabel("Non-null labels")
    axes[1].set_title("Available labels per target")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    labels_per_compound = df[TARGETS].notna().sum(axis=1)
    vc = labels_per_compound.value_counts().sort_index()
    fig2, ax = plt.subplots(figsize=(10, 5))
    vc.plot(kind="bar", ax=ax, color=PALETTE_2[0])
    ax.set_xlabel("Number of Labeled Assays")
    ax.set_ylabel("Number of Compounds")
    ax.set_title("Number of Tested Assays per Compound")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    all_compounds = df[TARGETS].copy()
    all_compounds = all_compounds.loc[all_compounds.isna().sum(axis=1).sort_values(ascending=False).index]
    missing_mat = all_compounds.isna().astype(int).T
    missing_cmap = sns.color_palette([PALETTE_2[0], PALETTE_2[1]], as_cmap=True)
    fig3, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(missing_mat, cbar=False, vmin=0, vmax=1, cmap=missing_cmap,
                yticklabels=True, xticklabels=False, ax=ax)
    cbar = fig3.colorbar(ax.collections[0], ax=ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Present", "Missing"])
    ax.set_title("Missing-Value Pattern (compounds sorted by missingness)")
    ax.set_xlabel("Compounds")
    ax.set_ylabel("Target")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

def _render_target_distributions(df: pd.DataFrame, target_choice: str):
    st.subheader("3. Target Distributions (Class Imbalance)")
    targets_show = TARGETS if target_choice == "All" else [target_choice]
    counts = pd.DataFrame({
        "Inactive": [(df[t] == 0).sum() for t in targets_show],
        "Active": [(df[t] == 1).sum() for t in targets_show],
    }, index=targets_show)
    counts["Labeled"] = counts["Inactive"] + counts["Active"]
    pct = counts[["Inactive", "Active"]].div(counts["Labeled"], axis=0).replace(0, np.nan) * 100
    fig, ax = plt.subplots(figsize=(max(8, len(targets_show) * 1.2), 6))
    pct.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE_2[0], PALETTE_2[1]])
    ax.set_ylabel("Share of labeled compounds (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Class Imbalance: Inactive vs Active Share")
    for i, t in enumerate(targets_show):
        ax.text(i, 102, f"n={int(counts.loc[t, 'Labeled'])}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    positive_ratio = pd.Series(
        {t: (df[t] == 1).sum() / df[t].notna().sum() * 100 for t in targets_show if df[t].notna().sum() > 0}
    ).sort_values()
    if not positive_ratio.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        positive_ratio.plot(kind="barh", ax=ax, color=PALETTE_2[1])
        ax.set_xlabel("% positive (active)")
        ax.set_title("Class Imbalance: % Active per Target")
        ax.axvline(x=10, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def _render_target_correlations(df: pd.DataFrame):
    st.subheader("4. Target Correlations")
    corr = df[TARGETS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=CMAP_DIV,
                vmin=-1, vmax=1, center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Pairwise Target Correlations")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    top_pairs = (
        corr.where(~mask)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Target A", "level_1": "Target B", 0: "Correlation"})
        .query("`Target A` != `Target B`")
        .sort_values("Correlation", ascending=False)
    )
    st.write("Top 5 most correlated pairs:")
    st.dataframe(top_pairs.head(5), use_container_width=True)
    st.write("Top 5 least correlated pairs:")
    st.dataframe(top_pairs.tail(5), use_container_width=True)

def _render_molecular_properties(df: pd.DataFrame, target_choice: str):
    st.subheader("5. Molecular Properties")
    smiles_col = "smiles"
    mols = df[smiles_col].dropna().apply(Chem.MolFromSmiles)
    valid_mask = mols.notna()
    mols_valid = mols[valid_mask]
    if mols_valid.empty:
        st.warning("No valid molecules from SMILES.")
    else:
        props = pd.DataFrame(index=mols_valid.index)
        props["MolWt"] = mols_valid.apply(Descriptors.MolWt)
        props["HeavyAtomCount"] = mols_valid.apply(Descriptors.HeavyAtomCount)
        props["SMILES_len"] = df.loc[mols_valid.index, smiles_col].str.len()
        props["NumRings"] = mols_valid.apply(rdMolDescriptors.CalcNumRings)
        props["RotatableBonds"] = mols_valid.apply(Descriptors.NumRotatableBonds)
        props["LogP"] = mols_valid.apply(Descriptors.MolLogP)
        props["TPSA"] = mols_valid.apply(Descriptors.TPSA)
        split_target = target_choice if target_choice != "All" else "NR-AhR"
        props_plot = props.copy()
        props_plot["activity"] = df.loc[props_plot.index, split_target].map({0.0: "Inactive", 1.0: "Active"})
        props_plot = props_plot[props_plot["activity"].notna()]
        if not props_plot.empty:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, (col, xlabel) in zip(axes, [
                ("MolWt", "Molecular Weight (Da)"),
                ("SMILES_len", "SMILES string length"),
                ("HeavyAtomCount", "Heavy Atom Count"),
            ]):
                sns.histplot(data=props_plot, x=col, hue="activity",
                             hue_order=["Inactive", "Active"],
                             palette={"Inactive": PALETTE_2[0], "Active": PALETTE_2[1]},
                             bins=28, stat="density", common_norm=False, multiple="layer",
                             element="bars", alpha=0.55, ax=ax, legend=(ax is axes[0]))
                sns.kdeplot(data=props_plot, x=col, hue="activity",
                            hue_order=["Inactive", "Active"],
                            palette={"Inactive": PALETTE_2[0], "Active": PALETTE_2[1]},
                            common_norm=False, fill=True, alpha=0.2, linewidth=2, ax=ax, legend=False)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Density")
                ax.set_title(f"{col} (split by {split_target})")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        st.dataframe(props.describe(), use_container_width=True)

def _render_chemical_diversity(df: pd.DataFrame):
    st.subheader("6. Chemical Diversity")
    smiles_col = "smiles"
    mols = df[smiles_col].dropna().apply(Chem.MolFromSmiles)
    valid_mask = mols.notna()
    mols_valid = mols[valid_mask]
    if not mols_valid.empty:
        props = pd.DataFrame(index=mols_valid.index)
        props["MolWt"] = mols_valid.apply(Descriptors.MolWt)
        props["HeavyAtomCount"] = mols_valid.apply(Descriptors.HeavyAtomCount)
        props["NumRings"] = mols_valid.apply(rdMolDescriptors.CalcNumRings)
        props["RotatableBonds"] = mols_valid.apply(Descriptors.NumRotatableBonds)
        props["LogP"] = mols_valid.apply(Descriptors.MolLogP)
        props["TPSA"] = mols_valid.apply(Descriptors.TPSA)
        desc_cols = ["NumRings", "RotatableBonds", "LogP", "TPSA"]
        desc_cols = [c for c in desc_cols if c in props.columns]
        if desc_cols:
            ncols = min(3, len(desc_cols))
            nrows = (len(desc_cols) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes = np.atleast_2d(axes)
            for idx, col in enumerate(desc_cols):
                r, c = idx // ncols, idx % ncols
                ax = axes[r, c]
                vals = props[col].dropna()
                ax.hist(vals, bins=40, color=PALETTE_2[0], edgecolor="black", linewidth=0.4, alpha=0.85)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {col}")
                ax.axvline(vals.median(), color=PALETTE_2[1], linestyle="--", alpha=0.9, label=f"Median: {vals.median():.1f}")
                ax.legend(fontsize=9)
            for j in range(len(desc_cols), nrows * ncols):
                axes.flat[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No valid molecules for diversity plots.")

def _render_dataset_chemistry_tab(df: pd.DataFrame, target_choice: str):
    _render_dataset_overview(df)
    _render_duplicates_analysis(df)
    _render_missing_values(df)
    _render_target_distributions(df, target_choice)
    _render_target_correlations(df)
    _render_molecular_properties(df, target_choice)
    _render_chemical_diversity(df)

def _render_molecule_viewer(df: pd.DataFrame):
    """Render a single-molecule RDKit viewer for selected SMILES."""
    st.caption("Pick a SMILES from the train set or paste any SMILES to visualize with RDKit.")
    from rdkit.Chem import rdFingerprintGenerator
    unique_smiles = df["smiles"].dropna().astype(str).unique()
    if len(unique_smiles) > 500:
        np.random.seed(42)
        unique_smiles = np.random.choice(unique_smiles, size=500, replace=False)
    options = ["— Other (paste below) —"] + sorted(unique_smiles.tolist(), key=lambda s: (len(s), s))
    chosen_from_list = st.selectbox(
        "Choose from dataset",
        options,
        index=0,
        key="single_mol_select",
    )
    paste_smiles = st.text_input("Or paste any SMILES", placeholder="e.g. CCO", key="single_mol_paste")
    if paste_smiles and paste_smiles.strip():
        smiles_to_show = paste_smiles.strip()
    elif chosen_from_list and chosen_from_list != "— Other (paste below) —":
        smiles_to_show = chosen_from_list
    else:
        smiles_to_show = None
    if smiles_to_show:
        mol = Chem.MolFromSmiles(smiles_to_show)
        if mol is None:
            st.error("Invalid SMILES.")
        else:
            st.subheader("2D structure")
            from rdkit.Chem.Draw import rdMolDraw2D
            draw_opts = rdMolDraw2D.MolDrawOptions()
            rdMolDraw2D.SetMonochromeMode(draw_opts, (0, 0, 0), (1, 1, 1))  # black on white
            img_2d = Draw.MolToImage(mol, size=(400, 400), options=draw_opts)
            _, col_img, _ = st.columns([1, 2, 1])
            with col_img:
                st.image(img_2d)

            st.subheader("Descriptors")
            desc_row = {
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "RingCount": rdMolDescriptors.CalcNumRings(mol),
                "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            }
            desc_df = pd.DataFrame(list(desc_row.items()), columns=["Descriptor", "Value"])
            st.dataframe(desc_df, use_container_width=True, hide_index=True)

            heteroatoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O", "S", "F", "Cl", "Br", "I")]
            if heteroatoms:
                st.caption("Structure with heteroatoms (N, O, S, halogens) highlighted")
                try:
                    img_het = Draw.MolsToGridImage(
                        [mol],
                        molsPerRow=1,
                        subImgSize=(400, 400),
                        highlightAtomLists=[heteroatoms],
                    )
                    st.image(img_het)
                except Exception:
                    pass
            n_morgan_bits = 8
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            tpls_morgan = []
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fp = fpg.GetFingerprint(mol, additionalOutput=ao)
            bit_info = ao.GetBitInfoMap()
            on_bits = list(fp.GetOnBits())[:n_morgan_bits]
            for bit in on_bits:
                tpls_morgan.append((mol, bit, bit_info))
            if tpls_morgan:
                st.caption("Morgan (ECFP) bits (first 8)")
                legends_morgan = [f"Bit {b}" for (_, b, _) in tpls_morgan]
                img_morgan = Draw.DrawMorganBits(tpls_morgan, molsPerRow=4, legends=legends_morgan, subImgSize=(220, 220))
                st.image(img_morgan)
            n_rdk_bits = 8
            fpg_rdk = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=2048)
            rdk_tpls = []
            ao_rdk = rdFingerprintGenerator.AdditionalOutput()
            ao_rdk.AllocateBitPaths()
            fp_rdk = fpg_rdk.GetFingerprint(mol, additionalOutput=ao_rdk)
            bit_paths = ao_rdk.GetBitPaths()
            on_bits_rdk = list(fp_rdk.GetOnBits())[:n_rdk_bits]
            for bit in on_bits_rdk:
                if bit in bit_paths:
                    paths = bit_paths[bit]
                    if paths and len(paths) > 0:
                        # Keep tuple shape required by DrawRDKitBits.
                        rdk_tpls.append((mol, bit, bit_paths))
            if rdk_tpls:
                st.caption("RDKit path bits (first 8)")
                legends_rdk = [f"Bit {b}" for (_, b, _) in rdk_tpls]
                try:
                    img_rdk = Draw.DrawRDKitBits(rdk_tpls, molsPerRow=4, legends=legends_rdk, subImgSize=(220, 220))
                    st.image(img_rdk)
                except (IndexError, TypeError) as e:
                    st.warning(f"Could not draw RDKit path bits for this molecule: {e}")
    else:
        st.caption("Select a SMILES from the dataset or paste one above to visualize.")


def _render_assay_context_and_signals():
    st.subheader("1. Assay Context")
    st.caption("Matches the thesis assay overview table (Huang et al., 2016).")
    st.markdown("""
| Assay | Brief description |
|-------|-------------------|
| **NR-AR** | Androgen receptor signalling activity. |
| **NR-AR-LBD** | Androgen receptor ligand-binding domain activation. |
| **NR-AhR** | Aryl hydrocarbon receptor activation by xenobiotic ligands. |
| **NR-Aromatase** | Inhibition of aromatase enzyme activity involved in estrogen synthesis. |
| **NR-ER** | Estrogen receptor signalling activity. |
| **NR-ER-LBD** | Estrogen receptor ligand-binding domain activation. |
| **NR-PPAR-gamma** | Peroxisome proliferator-activated receptor gamma activation. |
| **SR-ARE** | Activation of antioxidant response element-mediated oxidative stress pathways. |
| **SR-ATAD5** | Induction of ATAD5 reporter indicating DNA damage response. |
| **SR-HSE** | Activation of heat shock response elements under proteotoxic or heat stress. |
| **SR-MMP** | Disruption of mitochondrial membrane potential. |
| **SR-p53** | Activation of p53-dependent DNA damage and stress response signalling. |
""")
    st.caption("NR = nuclear receptor; SR = stress response.")

def _render_descriptor_distributions(df: pd.DataFrame, target_choice: str, descriptors: list):
    st.subheader("3. Descriptor Distributions by Assay")
    desc_sel = st.selectbox("Descriptor", [d[0] for d in descriptors], key="part3_desc")
    col_name = desc_sel
    title = next(d[1] for d in descriptors if d[0] == col_name)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for idx, assay in enumerate(TARGETS):
        ax = axes[idx]
        sub = df[df[assay].notna()].copy()
        sub["label"] = sub[assay].map({0.0: "inactive", 1.0: "active"})
        if col_name not in sub.columns:
            ax.set_visible(False)
            continue
        sub = sub[sub[col_name].notna()]
        if sub.empty:
            ax.set_visible(False)
            continue
        summary = sub.groupby("label")[col_name].mean()
        colors = [PALETTE_2[1] if i == "active" else PALETTE_2[0] for i in summary.index]
        summary.plot(kind="bar", ax=ax, legend=False, color=colors)
        ax.set_title(assay, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel("")
    for j in range(len(TARGETS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{title} vs activity (train) — one plot per assay", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def _render_two_descriptor_map(df: pd.DataFrame, target_choice: str, descriptors: list):
    st.subheader("4. Two-Descriptor Map")
    desc_options = [(d[0], d[1]) for d in descriptors]
    desc1_col = st.selectbox("Descriptor 1 (x-axis)", desc_options, format_func=lambda x: x[1], key="two_desc_1")
    desc2_col = st.selectbox("Descriptor 2 (y-axis)", desc_options, format_func=lambda x: x[1], key="two_desc_2", index=1)
    x_descriptor, y_descriptor = desc1_col[0], desc2_col[0]
    title1, title2 = desc1_col[1], desc2_col[1]
    targets_show = TARGETS if target_choice == "All" else [target_choice]
    panels_show = targets_show if target_choice != "All" else TARGETS
    n_show = len(panels_show)
    nr = (n_show + 3) // 4
    nc = min(4, n_show)
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 4 * nr))
    axes = np.atleast_2d(axes)
    for idx, assay in enumerate(panels_show):
        r, c = idx // nc, idx % nc
        ax = axes[r, c]
        sub = df[df[assay].notna()].copy()
        sub["active"] = (sub[assay] == 1.0).astype(float)
        sub = sub[[x_descriptor, y_descriptor, "active"]].dropna()
        if sub.empty or sub["active"].nunique() == 0:
            ax.set_visible(False)
            continue
        x = (sub[x_descriptor] - sub[x_descriptor].min()) / ((sub[x_descriptor].max() - sub[x_descriptor].min()) + 1e-9)
        y = (sub[y_descriptor] - sub[y_descriptor].min()) / ((sub[y_descriptor].max() - sub[y_descriptor].min()) + 1e-9)
        inact = sub["active"] == 0.0
        act = sub["active"] == 1.0
        ax.scatter(x[inact], y[inact], s=8, alpha=0.12, color=PALETTE_2[0], label="inactive")
        ax.scatter(x[act], y[act], s=14, alpha=0.85, color=PALETTE_2[1], edgecolor="black", linewidth=0.2, label="active")
        ax.set_xlabel(f"{title1} (norm)")
        ax.set_ylabel(f"{title2} (norm)")
        ax.set_title(f"{assay} (n={len(sub)})", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8)
    for j in range(n_show, axes.size):
        axes.flat[j].set_visible(False)
    fig.suptitle(f"{title1} vs {title2} — normalized scatter (train)", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def _render_top2_descriptor_maps(df: pd.DataFrame, target_choice: str, all_desc_cols: list):
    importance = []
    for assay in TARGETS:
        sub = df[df[assay].notna()].copy()
        sub["active"] = (sub[assay] == 1.0).astype(float)
        for col in all_desc_cols:
            if col not in sub.columns:
                continue
            subc = sub[[col, "active"]].dropna()
            if subc.empty or subc["active"].nunique() < 2:
                continue
            m0 = subc.loc[subc["active"] == 0, col].mean()
            m1 = subc.loc[subc["active"] == 1, col].mean()
            importance.append({"assay": assay, "descriptor": col, "importance": abs(m1 - m0)})
    imp_df = pd.DataFrame(importance)
    top2_per_assay = {}
    for assay in TARGETS:
        sub = imp_df[imp_df["assay"] == assay].nlargest(2, "importance")
        top2_per_assay[assay] = list(sub["descriptor"].values)

    st.subheader("5. Top-2 Descriptor Maps per Assay")
    targets_show = TARGETS if target_choice == "All" else [target_choice]
    panels_show = targets_show if target_choice != "All" else TARGETS
    n_show = len(panels_show)
    nr = (n_show + 3) // 4
    nc = min(4, n_show)
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 4 * nr))
    axes = np.atleast_2d(axes)
    for idx, assay in enumerate(panels_show):
        r, c = idx // nc, idx % nc
        ax = axes[r, c]
        cols = top2_per_assay.get(assay, [])
        if len(cols) < 2:
            ax.set_visible(False)
            continue
        c1, c2 = cols[0], cols[1]
        sub = df[df[assay].notna()].copy()
        sub["active"] = (sub[assay] == 1.0).astype(float)
        sub = sub[[c1, c2, "active"]].dropna()
        if sub.empty or sub["active"].nunique() == 0:
            ax.set_visible(False)
            continue
        x = (sub[c1] - sub[c1].min()) / ((sub[c1].max() - sub[c1].min()) + 1e-9)
        y = (sub[c2] - sub[c2].min()) / ((sub[c2].max() - sub[c2].min()) + 1e-9)
        inact = sub["active"] == 0.0
        act = sub["active"] == 1.0
        ax.scatter(x[inact], y[inact], s=8, alpha=0.12, color=PALETTE_2[0])
        ax.scatter(x[act], y[act], s=14, alpha=0.85, color=PALETTE_2[1], edgecolor="black", linewidth=0.2)
        ax.set_xlabel(f"{c1} (norm)", fontsize=8)
        ax.set_ylabel(f"{c2} (norm)", fontsize=8)
        ax.set_title(f"{assay}\n({c1} vs {c2})", fontsize=9)
    for j in range(n_show, axes.size):
        axes.flat[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def _render_effect_size_summary(df: pd.DataFrame, target_choice: str):
    st.subheader("6. Compact Effect-Size Summary")
    def diff_metric(df_sub: pd.DataFrame, descriptor_col: str) -> float:
        if descriptor_col not in df_sub.columns:
            return np.nan
        active_mean = df_sub.loc[df_sub["label"] == "active", descriptor_col].mean()
        inactive_mean = df_sub.loc[df_sub["label"] == "inactive", descriptor_col].mean()
        return active_mean - inactive_mean

    rows_eff = []
    targets_show = TARGETS if target_choice == "All" else [target_choice]
    for assay in (targets_show if target_choice != "All" else TARGETS):
        assay_df = df[df[assay].notna()].copy()
        assay_df["label"] = assay_df[assay].map({0.0: "inactive", 1.0: "active"})
        if "n_halogen" in assay_df.columns:
            assay_df = assay_df[assay_df["n_halogen"].notna()]
            if not assay_df.empty:
                rows_eff.append({"assay": assay, "descriptor": "n_halogen", "diff (active−inactive)": diff_metric(assay_df, "n_halogen")})
    if "NR-ER" in df.columns:
        nr_er_df = df[df["NR-ER"].notna()].copy()
        nr_er_df["label"] = nr_er_df["NR-ER"].map({0.0: "inactive", 1.0: "active"})
        if "has_phenol" in nr_er_df.columns and not nr_er_df.empty:
            rows_eff.append({"assay": "NR-ER", "descriptor": "% phenol", "diff (active−inactive)": diff_metric(nr_er_df, "has_phenol") * 100})
    if "NR-PPAR-gamma" in df.columns:
        nr_ppar_gamma_df = df[df["NR-PPAR-gamma"].notna()].copy()
        nr_ppar_gamma_df["label"] = nr_ppar_gamma_df["NR-PPAR-gamma"].map({0.0: "inactive", 1.0: "active"})
        if "has_cooh" in nr_ppar_gamma_df.columns and not nr_ppar_gamma_df.empty:
            rows_eff.append({"assay": "NR-PPAR-gamma", "descriptor": "% COOH", "diff (active−inactive)": diff_metric(nr_ppar_gamma_df, "has_cooh") * 100})

    if rows_eff:
        summary_struct = pd.DataFrame(rows_eff)
        diff_col = "diff (active−inactive)"
        n_bars = len(summary_struct)
     
        fig_height = max(5.0, n_bars * 0.45)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y_pos = np.arange(n_bars)
        diffs = summary_struct[diff_col].values
        colors = [PALETTE_2[1] if v > 0 else PALETTE_2[0] for v in diffs]
        ax.barh(y_pos, diffs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['assay']} — {r['descriptor']}" for _, r in summary_struct.iterrows()], fontsize=9)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Difference (active − inactive)")
        ax.set_title("Structure-Assay Effect Size (Active vs Inactive)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(summary_struct, use_container_width=True)
    else:
        st.info("No effect-size data for the selected target(s). Ensure descriptors were computed (RDKit) and data has active/inactive labels.")

def _render_structure_assay_tab(df: pd.DataFrame, target_choice: str):
    _render_assay_context_and_signals()
    df = compute_descriptors(df)
    DESCRIPTORS = [
        ("n_halogen", "Halogen count", "Mean"),
        ("n_aromatic_rings", "Aromatic ring count", "Mean"),
        ("has_phenol", "Phenol (0/1)", "Mean"),
        ("has_aromatic_amine", "Aromatic amine (0/1)", "Mean"),
        ("has_cooh", "Carboxylic acid (0/1)", "Mean"),
        ("MolWt", "Molecular weight", "Mean"),
        ("NumHDonors", "H-bond donors", "Mean"),
        ("NumHAcceptors", "H-bond acceptors", "Mean"),
        ("n_rotatable", "Rotatable bonds", "Mean"),
        ("RingCount", "Ring count", "Mean"),
    ]
    ALL_DESC_COLS = [c for c, _, _ in DESCRIPTORS]
    _render_descriptor_distributions(df, target_choice, DESCRIPTORS)
    _render_two_descriptor_map(df, target_choice, DESCRIPTORS)
    _render_top2_descriptor_maps(df, target_choice, ALL_DESC_COLS)
    _render_effect_size_summary(df, target_choice)

def _render_resampling_tab():
    st.subheader("1. Synthetic Resampling Demo (thesis parity)")
    st.caption("Nominal 20x20 synthetic grid with jittered plotting, matching the thesis setup conceptually.")

    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
        from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
        from imblearn.combine import SMOTEENN
        from imblearn.pipeline import Pipeline
    except ImportError:
        st.warning("`imbalanced-learn` is not available. Install dependencies to enable this tab.")
        return

    rng = np.random.default_rng(42)
    k = 20
    jitter = 0.12
    rs = 42
    knn = 3

    def make_nominal_imbalanced_dataset(
        dataset_rng: np.random.Generator, n_majority: int = 130, n_minority: int = 18
    ) -> tuple[np.ndarray, np.ndarray]:
        majority = np.clip(np.round(dataset_rng.normal(loc=[8.0, 8.0], scale=3.5, size=(n_majority, 2))), 0, k - 1).astype(np.int64)
        minority = np.clip(np.round(dataset_rng.normal(loc=[12.0, 12.0], scale=3.0, size=(n_minority, 2))), 0, k - 1).astype(np.int64)
        x_data = np.vstack([majority, minority])
        y_data = np.array([0] * n_majority + [1] * n_minority, dtype=np.int64)
        return x_data, y_data

    def jitter_coords(c0: np.ndarray, c1: np.ndarray, plot_rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        j0 = c0.astype(float) + plot_rng.uniform(-jitter, jitter, size=len(c0))
        j1 = c1.astype(float) + plot_rng.uniform(-jitter, jitter, size=len(c1))
        return j0, j1

    def scatter_nominal(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray, title: str, plot_rng: np.random.Generator) -> None:
        for cls, label, color in ((0, "Majority", PALETTE_2[0]), (1, "Minority", PALETTE_2[1])):
            mask = y_data == cls
            if not np.any(mask):
                continue
            jx, jy = jitter_coords(x_data[mask, 0], x_data[mask, 1], plot_rng)
            ax.scatter(jx, jy, color=color, s=58, alpha=1.0, edgecolors="0.12", linewidths=0.9, label=label, zorder=3)
        ax.set_title(title)
        ax.set_xlim(-0.6, k - 0.4)
        ax.set_ylim(-0.6, k - 0.4)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.tick_params(axis="both", which="major", labelbottom=False, labelleft=False)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_aspect("equal", adjustable="box")

    def smoten_target_count(y_data: np.ndarray) -> int:
        n_maj = int(np.sum(y_data == 0))
        n_min = int(np.sum(y_data == 1))
        return max(n_min + 1, int(round(n_maj * 1.5)))

    def resample_ros(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return RandomOverSampler(random_state=rs).fit_resample(x_data, y_data)

    def resample_rus(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return RandomUnderSampler(sampling_strategy=0.5, random_state=rs).fit_resample(x_data, y_data)

    def resample_smote(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return SMOTE(random_state=rs, k_neighbors=knn).fit_resample(x_data.astype(np.float64), y_data)

    def resample_smoten(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return SMOTEN(random_state=rs, k_neighbors=knn, sampling_strategy={1: smoten_target_count(y_data)}).fit_resample(x_data, y_data)

    def resample_smote_enn(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return SMOTEENN(random_state=rs, smote=SMOTE(random_state=rs, k_neighbors=knn)).fit_resample(x_data.astype(np.float64), y_data)

    def resample_smoten_enn(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = smoten_target_count(y_data)
        pipe = Pipeline([
            ("smoten", SMOTEN(random_state=rs, k_neighbors=knn, sampling_strategy={1: target})),
            ("enn", EditedNearestNeighbours()),
        ])
        return pipe.fit_resample(x_data, y_data)

    methods = {
        "ROS": resample_ros,
        "RUS": resample_rus,
        "SMOTE": resample_smote,
        "SMOTE-N": resample_smoten,
        "SMOTE-ENN": resample_smote_enn,
        "SMOTEN-ENN": resample_smoten_enn,
    }

    x_data, y_data = make_nominal_imbalanced_dataset(rng)
    selected_method = st.selectbox("Resampling method", list(methods.keys()), index=0)
    show_all = st.checkbox("Show all methods", value=False)

    if show_all:
        for name, fn in methods.items():
            x_res, y_res = fn(x_data, y_data)
            pair_rng = np.random.default_rng(1)
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
            scatter_nominal(axes[0], x_data, y_data, "No resampling", pair_rng)
            scatter_nominal(axes[1], x_res, y_res, name, pair_rng)
            st.pyplot(fig)
            plt.close(fig)
    else:
        x_res, y_res = methods[selected_method](x_data, y_data)
        pair_rng = np.random.default_rng(1)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
        scatter_nominal(axes[0], x_data, y_data, "No resampling", pair_rng)
        scatter_nominal(axes[1], x_res, y_res, selected_method, pair_rng)
        st.pyplot(fig)
        plt.close(fig)



if __name__ == "__main__":
    main()
