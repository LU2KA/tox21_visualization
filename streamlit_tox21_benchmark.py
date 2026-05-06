from pathlib import Path
from io import BytesIO
from zipfile import ZipFile

import altair as alt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "white"

_vlag_base = sns.color_palette("vlag", as_cmap=True)
CMAP_HEATMAP_SEQ = mcolors.ListedColormap(_vlag_base(np.linspace(0.0, 1.0, 256)))

RESULTS_DIR = Path("results")

CHART_WIDTH = 900
CHART_HEIGHT = 420

Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
    10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
    17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


@st.cache_data
def load_csv(name: str, uploaded_csv_bytes: dict[str, bytes] | None = None) -> pd.DataFrame | None:
    if uploaded_csv_bytes is not None and name in uploaded_csv_bytes:
        return pd.read_csv(BytesIO(uploaded_csv_bytes[name]))
    path = RESULTS_DIR / name
    if not path.is_file():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_roc_curves(uploaded_json_bytes: dict[str, bytes] | None = None) -> list | None:
    """Load ROC curve data."""
    import json

    name = "roc_curves.json"
    if uploaded_json_bytes is not None and name in uploaded_json_bytes:
        return json.loads(uploaded_json_bytes[name].decode("utf-8"))
    path = RESULTS_DIR / name
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_uploaded_results(files: list) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """
    Return (csv_bytes, json_bytes) extracted from uploaded files.

    Accepts either:
    - One ZIP containing results files
    - Multiple individual files (CSVs / JSON)
    """
    csv_bytes: dict[str, bytes] = {}
    json_bytes: dict[str, bytes] = {}
    if not files:
        return csv_bytes, json_bytes

    # If any ZIP is present, prefer ZIP contents (first ZIP wins).
    zip_file = next((f for f in files if getattr(f, "name", "").lower().endswith(".zip")), None)
    if zip_file is not None:
        zdata = zip_file.getvalue()
        with ZipFile(BytesIO(zdata)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if not name:
                    continue
                if name.lower().endswith(".csv"):
                    csv_bytes[name] = zf.read(info)
                elif name.lower().endswith(".json"):
                    json_bytes[name] = zf.read(info)
        return csv_bytes, json_bytes

    for f in files:
        name = getattr(f, "name", "")
        if not name:
            continue
        if name.lower().endswith(".csv"):
            csv_bytes[name] = f.getvalue()
        elif name.lower().endswith(".json"):
            json_bytes[name] = f.getvalue()
    return csv_bytes, json_bytes


def _format_metric_name(col: str) -> str:
    """Format metric column name for display."""
    if col.startswith("val_"):
        return col.replace("val_", "val ")
    elif col.startswith("final_test_"):
        return col.replace("final_test_", "test ")
    elif col.startswith("final_train_"):
        return col.replace("final_train_", "train ")
    return col


def _get_metric_column_from_name(display_name: str, val_cols: list[str]) -> str:
    """Map display name back to the DataFrame column."""
    for col in val_cols:
        if _format_metric_name(col) == display_name:
            return col
    return display_name


def _get_metric_columns(df: pd.DataFrame, include_test: bool = False) -> list[str]:
    """Return metric columns (val/train and optionally test)."""
    cols = []
    for c in df.columns:
        if c in ("val_selection", "val_n", "final_test_n", "final_train_n"):
            continue
        if c.startswith("val_") or c.startswith("final_train_"):
            cols.append(c)
        elif include_test and c.startswith("final_test_"):
            cols.append(c)
    return cols


def _options(series: pd.Series) -> list:
    """Unique values excluding the header-row artifact."""
    col = series.name
    return [v for v in series.dropna().astype(str).unique().tolist() if v != str(col)]


def _group_imbalanced_models(df: pd.DataFrame) -> pd.DataFrame:
    """Group imbalance-aware models under one resampling label."""
    if df is None or "resampling" not in df.columns or "model" not in df.columns:
        return df
    return df


def _resampling_display_label(value: str) -> str:
    """Map resampling value to a display label."""
    labels = {
        "none": "Classical",
        "imbalanced_models": "Imbalanced Models",
        "smoteenn": "SMOTEENN",
        "smotetomek": "SMOTETomek",
        "smote": "SMOTE only",
        "random_over": "Random over",
        "random_under": "Random under",
    }
    return labels.get(value, value)


def _metric_domain(series: pd.Series, padding: float = 0.05, include_zero: bool = False) -> list[float]:
    """Metric axis domain with padding (clamped to [0, 1])."""
    if series.empty or series.isna().all():
        return [0.0, 1.0]
    v_min, v_max = float(series.min()), float(series.max())
    pad = (v_max - v_min) * padding if v_max > v_min else padding
    lo = max(0.0, v_min - pad)
    hi = min(1.0, v_max + pad)
    if include_zero:
        lo = 0.0
    return [lo, hi] if lo < hi else [0.0, 1.0]



def nemenyi_cd(num_blocks: int, num_treatments: int, alpha005: dict[int, float] = None) -> float | None:
    if alpha005 is None:
        alpha005 = Q_ALPHA_005
    k = num_treatments
    if k not in alpha005:
        return None
    n = num_blocks
    q = alpha005[k]
    return float(q * np.sqrt((k * (k + 1)) / (6 * n)))

def compute_mean_ranks(df: pd.DataFrame, group_col: str, target_col: str, metric_col: str) -> tuple[pd.Series, float | None]:
    """
    Compute average ranks for items in `group_col` across `target_col` based on `metric_col`.
    Returns (mean_ranks_series, nemenyi_cd).
    """
    df_unique = df[[target_col, group_col, metric_col]].drop_duplicates([target_col, group_col], keep="first")
    pivot = df_unique.pivot(index=target_col, columns=group_col, values=metric_col)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[1] < 2 or pivot.empty:
        return pd.Series(dtype=float), None
    
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean().sort_values()
    n, k = pivot.shape
    cd = nemenyi_cd(n, k)
    return mean_ranks, cd

def _plot_nemenyi_cd_graph(mean_ranks: pd.Series, cd: float | None, title: str, ylabel: str):
    """Plot Nemenyi CD graph using matplotlib."""
    order = mean_ranks.sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(order) * 0.4)), constrained_layout=True)
    
    POINT_BLUE, LINE_BLUE = "#3250a8", "#6b8cce"
    GRID_GREY = "#b8c4d9"
    
    for i, (label, rank) in enumerate(order.items()):
        ax.errorbar(
            x=rank, y=i, xerr=(cd / 2.0) if cd else 0.0, 
            fmt="o", color=POINT_BLUE, ecolor=LINE_BLUE, 
            elinewidth=2, capsize=5, markersize=8, linewidth=2
        )
        
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order.index, fontsize=12)
    ax.set_xlabel("Average rank (lower is better)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.45, color=GRID_GREY)
    
    if cd:
        ax.text(
            0.98, 0.98, f"CD = {cd:.2f}", 
            transform=ax.transAxes, ha="right", va="top", 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c9d4e8", alpha=0.95)
        )
    return fig

def run_benchmark() -> None:

    """Runnable from combined app. Do not call st.set_page_config here."""
    benchmark_results = load_csv("benchmark_results.csv")
    best_per_combo = load_csv("best_per_target_fingerprint.csv")
    metrics_long = load_csv("metrics_long.csv")
    per_class_counts = load_csv("per_class_counts.csv")
    grid_history = load_csv("grid_history.csv")
    mean_by_model = load_csv("mean_test_metrics_by_model.csv")

    benchmark_results = _group_imbalanced_models(benchmark_results)
    best_per_combo = _group_imbalanced_models(best_per_combo)
    metrics_long = _group_imbalanced_models(metrics_long)
    per_class_counts = _group_imbalanced_models(per_class_counts)
    grid_history = _group_imbalanced_models(grid_history)
    mean_by_model = _group_imbalanced_models(mean_by_model)

    roc_curves_raw = load_roc_curves()
    if roc_curves_raw is not None:
        imb_models = {"BalancedRandomForest", "EasyEnsemble", "RUSBoost", "BalancedBagging"}
        for rc in roc_curves_raw:
            if rc.get("resampling") == "none" and rc.get("model") in imb_models:
                rc["resampling"] = "imbalanced_models"
    roc_curves: list | None = roc_curves_raw

    opt_choice: str | None = None
    has_optimized_for = (
        benchmark_results is not None
        and "optimized_for" in benchmark_results.columns
    )
    if has_optimized_for:
        opt_values = list(benchmark_results["optimized_for"].dropna().unique())
        if opt_values:
            if "roc_auc" in opt_values:
                default_index = opt_values.index("roc_auc")
            elif "f1_macro" in opt_values:
                default_index = opt_values.index("f1_macro")
            else:
                default_index = 0
            opt_choice = st.selectbox(
                "Results Optimized For",
                options=opt_values,
                index=default_index,
                help="Select which metric the hyperparameter tuning was optimized for."
            )
            benchmark_results = benchmark_results[benchmark_results["optimized_for"] == opt_choice].copy()
            if best_per_combo is not None and "optimized_for" in best_per_combo.columns:
                best_per_combo = best_per_combo[best_per_combo["optimized_for"] == opt_choice].copy()
            if metrics_long is not None and "optimized_for" in metrics_long.columns:
                metrics_long = metrics_long[metrics_long["optimized_for"] == opt_choice].copy()
            if per_class_counts is not None and "optimized_for" in per_class_counts.columns:
                per_class_counts = per_class_counts[per_class_counts["optimized_for"] == opt_choice].copy()
            if roc_curves is not None:
                roc_curves_raw = [r for r in roc_curves_raw if r.get("optimized_for") == opt_choice]
                roc_curves = roc_curves_raw

    has_resampling = (
        benchmark_results is not None
        and "resampling" in benchmark_results.columns
    )
    current_dataset_label = "—"
    resampling_filter = None
    if has_resampling:
        resampling_values = set(benchmark_results["resampling"].dropna().unique())
        options = []
        if "none" in resampling_values:
            options.append("Classical")
        if "smoteenn" in resampling_values:
            options.append("SMOTEENN")
        if "smotetomek" in resampling_values:
            options.append("SMOTETomek")
        if "smote" in resampling_values:
            options.append("SMOTE only")
        if "random_over" in resampling_values:
            options.append("Random over")
        if "random_under" in resampling_values:
            options.append("Random under")
        options.append("All")
        for v in sorted(resampling_values - {"none", "smoteenn", "smotetomek", "smote", "random_over", "random_under"}):
            options.append(v)
        dataset_choice = st.selectbox(
            "Resampling/Dataset",
            options=options,
            index=options.index("All") if "All" in options else 0,
            help="Classical = no resampling; SMOTEENN; SMOTETomek; SMOTE only; Random over/under; All = all runs.",
        )
        current_dataset_label = dataset_choice
        if dataset_choice == "Classical":
            resampling_filter = "none"
        elif dataset_choice == "SMOTEENN":
            resampling_filter = "smoteenn"
        elif dataset_choice == "SMOTETomek":
            resampling_filter = "smotetomek"
        elif dataset_choice == "SMOTE only":
            resampling_filter = "smote"
        elif dataset_choice == "Random over":
            resampling_filter = "random_over"
        elif dataset_choice == "Random under":
            resampling_filter = "random_under"
        elif dataset_choice == "All":
            resampling_filter = None
        else:
            resampling_filter = dataset_choice
        benchmark_results_all = (
            benchmark_results.copy() if benchmark_results is not None else None
        )
        if resampling_filter is not None:
            benchmark_results = benchmark_results[benchmark_results["resampling"] == resampling_filter].copy()
            if best_per_combo is not None and "resampling" in best_per_combo.columns:
                best_per_combo = best_per_combo[best_per_combo["resampling"] == resampling_filter].copy()
            if grid_history is not None and "resampling" in grid_history.columns:
                grid_history = grid_history[grid_history["resampling"] == resampling_filter].copy()
            if per_class_counts is not None and "resampling" in per_class_counts.columns:
                per_class_counts = per_class_counts[per_class_counts["resampling"] == resampling_filter].copy()
            if roc_curves is not None and resampling_filter is not None:
                roc_curves = [r for r in roc_curves if r.get("resampling") == resampling_filter]
    else:
        benchmark_results_all = None

    tab_summary_overview, tab_grid, tab_per_class, tab_best, tab_assay_boxplot, tab_resampling_cmp, tab_fp_tgt, tab_generalization, tab_final_test = st.tabs([
        "Summary and overview", "Grid search", "Per-class", "Best model", "Assay score", "Best resampling", "Best fingerprints", "Generalization", "Final Test Results"
    ])

    with tab_summary_overview:
        _render_summary_overview(benchmark_results, best_per_combo)
    with tab_grid:
        _render_grid(grid_history, optimized_for=opt_choice, roc_curves=roc_curves)
    with tab_best:
        _render_best(benchmark_results, best_per_combo, optimized_for=opt_choice)
    with tab_fp_tgt:
        _render_fp_tgt(benchmark_results, best_per_combo, roc_curves_raw, optimized_for=opt_choice)
    with tab_assay_boxplot:
        _render_assay_boxplot(benchmark_results, optimized_for=opt_choice)
    with tab_resampling_cmp:
        _render_resampling_cmp(benchmark_results_all, optimized_for=opt_choice)
    with tab_per_class:
        _render_per_class(
            per_class_counts, best_per_combo, benchmark_results, has_resampling,
            resampling_filter, current_dataset_label,
        )
    with tab_generalization:
        _render_generalization(benchmark_results, benchmark_results_all, has_resampling, optimized_for=opt_choice)
    with tab_final_test:
        _render_final_test(best_per_combo)


def main() -> None:
    st.set_page_config(page_title="Tox21 fingerprint ML comparison", layout="wide")
    st.title("Tox21 fingerprint ML comparison")
    run_benchmark()


def _render_summary_overview(
    benchmark_results: pd.DataFrame | None,
    best_per_combo: pd.DataFrame | None,
) -> None:
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found in `results/`.")
    else:
        df = benchmark_results.copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Targets", df["target"].nunique())
        with col2:
            st.metric("Fingerprints", df["fingerprint"].nunique())
        with col3:
            st.metric("Models", df["model"].nunique())

        st.dataframe(df, use_container_width=True)


def _render_models(
    benchmark_results: pd.DataFrame | None,
    has_resampling: bool,
) -> None:
    st.subheader("Average Ranks by Model")
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found.")
        return

    include_test = st.checkbox("Include test metrics", value=True, key="models_include_test")
    val_cols = _get_metric_columns(benchmark_results, include_test=include_test)
    metric_names = [_format_metric_name(c) for c in val_cols]
    if not val_cols:
        st.info("No metric columns found.")
        return

    metric_plot_display = st.selectbox(
        "Metric to rank by (higher = better)", metric_names,
        index=metric_names.index("val roc_auc") if "val roc_auc" in metric_names else 0,
    )
    val_col = _get_metric_column_from_name(metric_plot_display, val_cols)

    df_plot = benchmark_results[["model", "fingerprint", "target", val_col]].dropna(subset=[val_col])
    if df_plot.empty:
        st.info(f"No data for metric '{metric_plot_display}'.")
        return

    df_agg = df_plot.groupby(["target", "model"])[val_col].mean().reset_index()
    mean_ranks, cd = compute_mean_ranks(df_agg, group_col="model", target_col="target", metric_col=val_col)
    
    if mean_ranks.empty:
        st.info("Not enough data to compute ranks.")
        return
        
    fig = _plot_nemenyi_cd_graph(
        mean_ranks, cd, 
        title=f"Model average ranks with Nemenyi CD (by {metric_plot_display})", 
        ylabel="Classifier"
    )
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.write("Mean metrics by model:")
    summary = benchmark_results.groupby("model")[val_cols].mean().round(4)
    summary.columns = metric_names
    
    summary["Average Rank"] = mean_ranks
    cols = ["Average Rank"] + [c for c in summary.columns if c != "Average Rank"]
    summary = summary[cols].sort_values("Average Rank")
    
    st.dataframe(summary, use_container_width=True)


def _prepare_grid_data(df: pd.DataFrame, metric: str, param_cols: list[str], group_col: str) -> pd.DataFrame:
    df = df.copy()
    df["line_id"] = df["target"] + " | " + df["model"] + " | " + df["fingerprint"]
    if "resampling" in df.columns:
        df["line_id"] = df["line_id"] + " | " + df["resampling"].astype(str)
        
    df["param_desc"] = df[param_cols].apply(
        lambda r: ", ".join(f"{c.replace('param_','') }={r[c]}" for c in param_cols if pd.notna(r[c])),
        axis=1,
    )
    
    plot_dfs = []
    for _, group_df in df.groupby("line_id", sort=False):
        if metric not in group_df.columns:
            continue
        group_df = group_df.dropna(subset=[metric]).sort_values(metric).reset_index(drop=True)
        if group_df.empty:
            continue
        group_df = group_df.copy()
        group_df["grid_index"] = np.arange(1, len(group_df) + 1)
        plot_dfs.append(group_df)
        
    if not plot_dfs:
        return pd.DataFrame()
        
    agg_df = pd.concat(plot_dfs, ignore_index=True)
    agg_df["color_group"] = agg_df[group_col]
    return agg_df

def _render_grid_roc_curves(df_mt: pd.DataFrame, roc_curves: list, targets_choice: list[str], models_choice: list[str]) -> None:
    st.divider()
    st.markdown("### ROC curve")
    
    fp_opts = sorted(_options(df_mt["fingerprint"]))
    fp_sel = st.selectbox("Fingerprint (ROC curve)", fp_opts or ["(no fingerprints)"], key="grid_roc_fp")
    tgt_sel = st.selectbox("Target (ROC curve)", targets_choice, index=0, key="grid_roc_target")
    mdl_sel = st.selectbox("Model (ROC curve)", models_choice, index=0, key="grid_roc_model")

    resampling_sel = None
    if "resampling" in df_mt.columns:
        resampling_opts = df_mt["resampling"].dropna().astype(str).unique().tolist()
        if resampling_opts:
            resampling_sel = resampling_opts[0]

    matches = [
        rc_dict for rc_dict in roc_curves
        if rc_dict.get("target") == tgt_sel
        and rc_dict.get("model") == mdl_sel
        and rc_dict.get("fingerprint") == fp_sel
        and (resampling_sel is None or rc_dict.get("resampling") == resampling_sel)
    ]

    if not matches:
        st.info("No ROC curve found for this selection.")
        return

    MAIN_BLUE = "#4a6fa5"
    GRID_GREY = "#b8c4d9"
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for i, rc_dict in enumerate(matches):
        fpr = rc_dict.get("fpr") or []
        tpr = rc_dict.get("tpr") or []
        if not fpr or not tpr:
            continue
        ax.plot(fpr, tpr, color=MAIN_BLUE, lw=1.4, alpha=0.85 if i == 0 else 0.45)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", lw=1.0, alpha=0.6)
    ax.set_title(f"ROC curve ({tgt_sel} | {mdl_sel} | {fp_sel})", pad=10)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.45, color=GRID_GREY)
    st.pyplot(fig)
    plt.close(fig)


def _render_grid(grid_history: pd.DataFrame | None, optimized_for: str | None, roc_curves: list | None = None) -> None:
    st.subheader("Grid search history (validation only)")
    if grid_history is None or grid_history.empty:
        st.warning("grid_history.csv not found or empty.")
    else:
        df = grid_history.copy()

        if df.empty:
            st.info("No grid history rows match the current filters.")
        else:
            st.write("Current filtered grid history (head):")
            st.dataframe(df, use_container_width=True)

            target_opts = sorted(_options(df["target"]))
            targets_choice = st.multiselect(
                "Targets for line plots", target_opts, default=target_opts[:1]
            )

            if not targets_choice:
                st.info("Select at least one target to show a plot.")
            else:
                df_t = df[df["target"].isin(targets_choice)].copy()
                if df_t.empty:
                    st.info("No rows for these targets.")
                else:
                    model_opts = sorted(_options(df_t["model"]))
                    default_models = ["XGBoost"] if "XGBoost" in model_opts else (model_opts[:1] if model_opts else [])
                    models_choice = st.multiselect(
                        "Models to show",
                        model_opts,
                        default=default_models,
                    )
                    if not models_choice:
                        st.info("Select at least one model to show a plot.")
                    else:
                        df_mt = df_t[df_t["model"].isin(models_choice)].copy()
                        if df_mt.empty:
                            st.info("No rows for this selection (targets + models).")
                        else:
                            metric_plot = None
                            metric_plot_display = None
                            if optimized_for:
                                preferred = f"val_{optimized_for}"
                                if preferred in df_mt.columns:
                                    metric_plot = preferred
                                    metric_plot_display = _format_metric_name(preferred)
                            if metric_plot is None and "val_selection" in df_mt.columns:
                                metric_plot = "val_selection"
                                metric_plot_display = _format_metric_name("val_selection")
                            if metric_plot is None:
                                metric_cols = [c for c in df_mt.columns if c.startswith("val_")]
                                if metric_cols:
                                    metric_plot = metric_cols[0]
                                    metric_plot_display = _format_metric_name(metric_plot)
                            if metric_plot is None or metric_plot_display is None:
                                st.info("No validation metrics found in grid history.")
                                return

                            st.caption(f"Plotting metric: {metric_plot_display}")

                            param_cols = []
                            for c in df_mt.columns:
                                if not c.startswith("param_"):
                                    continue
                                series = df_mt[c]
                                if not series.notna().any():
                                    continue
                                if not pd.api.types.is_numeric_dtype(series):
                                    continue
                                if series.nunique(dropna=True) <= 1:
                                    continue
                                param_cols.append(c)
                                
                            if not param_cols:
                                st.info("No param_* columns to plot against.")
                            else:
                                color_by = st.radio("Color lines by", ["Model", "Target"], horizontal=True)
                                group_col = "model" if color_by == "Model" else "target"

                                agg = _prepare_grid_data(df_mt, metric_plot, param_cols, group_col)
                                if agg.empty:
                                    st.info("No data available for plotting.")
                                    return
                                    
                                groups = sorted(agg["color_group"].unique().tolist())
                                single_group = len(groups) == 1
                                if single_group:
                                    color_col = "fingerprint"
                                    color_title = "Fingerprint"
                                    color_values = sorted(agg["fingerprint"].unique().tolist())
                                else:
                                    color_col = "color_group"
                                    color_title = color_by
                                    color_values = groups

                                y_min = float(agg[metric_plot].min())
                                y_max = float(agg[metric_plot].max())
                                pad = 0.02 * (y_max - y_min) if y_max > y_min else 0.01
                                domain = [max(0.0, y_min - pad), min(1.0, y_max + pad)]

                                chart = (
                                    alt.Chart(agg)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X("grid_index:Q", title="Grid point (1, 2, 3, …)"),
                                        y=alt.Y(f"{metric_plot}:Q", title=metric_plot_display, scale=alt.Scale(domain=domain)),
                                        color=alt.Color(
                                            f"{color_col}:N", title=color_title,
                                            scale=alt.Scale(domain=color_values, scheme="category10"),
                                            legend=alt.Legend(orient="right", labelLimit=0, values=color_values),
                                        ),
                                        detail="line_id:N",
                                        tooltip=["line_id", "grid_index", "param_desc", metric_plot] + (["resampling"] if "resampling" in agg.columns else []),
                                    )
                                    .properties(
                                        title=f"{metric_plot_display} by grid point (hover for param values)",
                                        width=CHART_WIDTH, height=CHART_HEIGHT,
                                    )
                                    .configure_legend(orient="right", labelLimit=0, symbolLimit=0)
                                )
                                st.altair_chart(chart, use_container_width=True)

                                if "val_roc_auc" in df_mt.columns and metric_plot != "val_roc_auc":
                                    metric_plot_roc = "val_roc_auc"
                                    metric_plot_roc_display = _format_metric_name(metric_plot_roc)
                                    agg_roc = _prepare_grid_data(df_mt, metric_plot_roc, param_cols, group_col)

                                    if not agg_roc.empty:
                                        y_min = float(agg_roc[metric_plot_roc].min())
                                        y_max = float(agg_roc[metric_plot_roc].max())
                                        pad = 0.02 * (y_max - y_min) if y_max > y_min else 0.01
                                        domain = [max(0.0, y_min - pad), min(1.0, y_max + pad)]

                                        chart_roc = (
                                            alt.Chart(agg_roc)
                                            .mark_line(point=True)
                                            .encode(
                                                x=alt.X("grid_index:Q", title="Grid point (1, 2, 3, …)"),
                                                y=alt.Y(f"{metric_plot_roc}:Q", title=metric_plot_roc_display, scale=alt.Scale(domain=domain)),
                                                color=alt.Color(
                                                    f"{color_col}:N", title=color_title,
                                                    scale=alt.Scale(domain=color_values, scheme="category10"),
                                                    legend=alt.Legend(orient="right", labelLimit=0, values=color_values),
                                                ),
                                                detail="line_id:N",
                                                tooltip=["line_id", "grid_index", "param_desc", metric_plot_roc] + (["resampling"] if "resampling" in agg_roc.columns else []),
                                            )
                                            .properties(
                                                title=f"{metric_plot_roc_display} by grid point (hover for param values)",
                                                width=CHART_WIDTH, height=CHART_HEIGHT,
                                            )
                                            .configure_legend(orient="right", labelLimit=0, symbolLimit=0)
                                        )
                                        st.altair_chart(chart_roc, use_container_width=True)

                                if roc_curves:
                                    _render_grid_roc_curves(df_mt, roc_curves, targets_choice, models_choice)

                                st.markdown("### Hyperparameter sensitivity (selected targets & models)")
                                if len(param_cols) == 1:
                                    param_x = param_cols[0]
                                else:
                                    param_x = st.selectbox("Parameter on X-axis", param_cols, key="hp_param_x")

                                df_hp = df_mt.dropna(subset=[param_x, metric_plot]).copy()
                                if df_hp.empty:
                                    st.info("No rows with this parameter and metric.")
                                else:
                                    hp_chart = (
                                        alt.Chart(df_hp)
                                        .mark_circle(size=60, opacity=0.7)
                                        .encode(
                                            x=alt.X(f"{param_x}:Q", title=param_x.replace("param_", "")),
                                            y=alt.Y(f"{metric_plot}:Q", title=metric_plot_display),
                                            color=alt.Color("model:N", title="Model"),
                                            tooltip=["target", "model", "fingerprint", param_x, metric_plot],
                                        )
                                        .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
                                    )
                                    st.altair_chart(hp_chart, use_container_width=True)



def _render_best_wins_chart(best_by_metric: pd.DataFrame) -> None:
    wins = best_by_metric["model"].value_counts().reset_index()
    wins.columns = ["model", "n_wins"]
    wins = wins.sort_values("n_wins", ascending=False)

    st.markdown("### Number of wins per model")
    chart_wins = (
        alt.Chart(wins)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort="-y", title="Model"),
            y=alt.Y("n_wins:Q", title="Number of (target × fingerprint) wins"),
            color=alt.Color("model:N", legend=None),
            tooltip=["model", "n_wins"],
        )
        .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
    )
    st.altair_chart(chart_wins, use_container_width=True)

def _render_best_rank_distribution(benchmark_results: pd.DataFrame, score_col: str) -> None:
    st.markdown("### Model rank distribution (rank 1/2/3 per target × fingerprint)")
    df_rank = (
        benchmark_results[["target", "fingerprint", "model", score_col]]
        .dropna(subset=[score_col])
    )
    if df_rank.empty:
        st.info("No data to compute ranks.")
        return

    df_rank = df_rank.copy()
    df_rank["rank"] = (
        df_rank.groupby(["target", "fingerprint"])[score_col]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    rank_counts = (
        df_rank[df_rank["rank"] <= 3]
        .groupby(["model", "rank"])
        .size()
        .reset_index(name="n")
    )
    rank_counts["rank_str"] = "Rank " + rank_counts["rank"].astype(str)
    
    if rank_counts.empty:
        st.info("No rank counts for top 3.")
        return

    sort_models = (
        rank_counts[rank_counts["rank"] == 1]
        .sort_values("n", ascending=False)["model"]
        .tolist()
    )
    rank_order = ["Rank 1", "Rank 2", "Rank 3"]
    chart_rank = (
        alt.Chart(rank_counts)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=sort_models, title="Model", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("n:Q", title="Count of (target × fingerprint) combos"),
            color=alt.Color(
                "rank_str:N",
                title="Rank",
                sort=rank_order,
                scale=alt.Scale(domain=rank_order),
            ),
            order=alt.Order("rank:Q", sort="ascending"),
            tooltip=["model", "rank_str", "n"],
        )
        .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
    )
    st.altair_chart(chart_rank, use_container_width=True)

def _render_best(
    benchmark_results: pd.DataFrame | None,
    best_per_combo: pd.DataFrame | None,
    optimized_for: str | None,
) -> None:
    st.subheader("Best model (average ranks)")
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found.")
        return

    score_col = None
    if optimized_for:
        preferred = f"final_test_{optimized_for}"
        if preferred in benchmark_results.columns:
            score_col = preferred
    if score_col is None:
        test_cols = [c for c in benchmark_results.columns if c.startswith("final_test_") and c != "final_test_n"]
        score_col = test_cols[0] if test_cols else None
    if score_col is None:
        st.info("No final test metric columns found.")
        return

    metric_choice_display = _format_metric_name(score_col)
    st.caption(f"Using final test metric: {metric_choice_display}")

    idx_best = benchmark_results.groupby(["target", "fingerprint"])[score_col].idxmax()
    best_by_metric = benchmark_results.loc[idx_best].sort_values(
        ["target", "fingerprint"]
    ).reset_index(drop=True)
    best_by_metric = best_by_metric.rename(columns={score_col: "best_value"})

    st.markdown("**Best model per (target, fingerprint)**")
    st.dataframe(best_by_metric, use_container_width=True)


    _render_best_wins_chart(best_by_metric)
    _render_best_rank_distribution(benchmark_results, score_col)

    st.divider()
    st.markdown("### Classifier ranks (Nemenyi CD) + mean-rank heatmap")

    df_rank_base = benchmark_results.dropna(subset=[score_col]).copy()
    if df_rank_base.empty:
        st.info("No rows with valid metric values.")
        return

    has_resampling = "resampling" in df_rank_base.columns
    resampling_opts = (
        sorted(_options(df_rank_base["resampling"])) if has_resampling else ["none"]
    )
    fp_opts = sorted(_options(df_rank_base["fingerprint"]))

    col1, col2 = st.columns(2)
    with col1:
        resampling_choice = st.selectbox(
            "Resampling",
            resampling_opts,
            index=0,
            key="best_model_rank_resampling",
        )
    with col2:
        fp_choice = st.selectbox(
            "Fingerprint",
            fp_opts or ["(no fingerprints)"],
            index=0,
            key="best_model_rank_fingerprint",
        )

    if has_resampling:
        df_rank_base = df_rank_base[df_rank_base["resampling"] == resampling_choice].copy()


    if df_rank_base.empty:
        st.info("No rows match this resampling selection.")
        return

    df_cd = df_rank_base[df_rank_base["fingerprint"] == fp_choice].copy()
    if df_cd.empty:
        st.info("No rows for this fingerprint and resampling.")
    else:
        df_cd = df_cd[["target", "model", score_col]].drop_duplicates(["target", "model"], keep="first")
        mean_ranks, cd = compute_mean_ranks(
            df_cd,
            group_col="model",
            target_col="target",
            metric_col=score_col,
        )
        if mean_ranks.empty:
            st.info("Not enough data to compute classifier ranks for this selection.")
        else:
            pivot = (
                df_cd.drop_duplicates(["target", "model"], keep="first")
                .pivot(index="target", columns="model", values=score_col)
                .dropna(axis=0, how="any")
            )
            if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
                _chi2, p_value = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
                st.caption(f"Friedman p = {p_value:.3g}")

            fig = _plot_nemenyi_cd_graph(
                mean_ranks,
                cd,
                title=f"Classifier average ranks with Nemenyi CD ({fp_choice} × {resampling_choice})",
                ylabel="Classifier",
            )
            st.pyplot(fig)
            plt.close(fig)

    targets = sorted(df_rank_base["target"].dropna().unique().tolist())
    fp_cols = sorted(df_rank_base["fingerprint"].dropna().unique().tolist())
    model_cols = sorted(df_rank_base["model"].dropna().unique().tolist())

    cols: dict[str, pd.Series] = {}
    for fp in fp_cols:
        per_assay: list[pd.Series] = []
        for t in targets:
            block = df_rank_base[(df_rank_base["target"] == t) & (df_rank_base["fingerprint"] == fp)]
            series = (
                block.drop_duplicates("model", keep="first")
                .set_index("model")[score_col]
                .reindex(model_cols)
                .dropna()
            )
            if len(series) >= 2:
                per_assay.append(series.rank(ascending=False, method="average"))
        if per_assay:
            cols[fp] = pd.concat(per_assay, axis=1).mean(axis=1).reindex(model_cols)
        else:
            cols[fp] = pd.Series(np.nan, index=model_cols)

    mat_models = pd.DataFrame(cols).reindex(index=model_cols)  # models x fingerprints
    mat_hm = mat_models.T  # fingerprints x models
    if mat_hm.empty or not np.any(np.isfinite(mat_hm.to_numpy(dtype=float))):
        st.info("Not enough data to build the mean-rank heatmap.")
        return

    fig, ax = plt.subplots(
        figsize=(max(11, len(model_cols) * 0.9), max(6, len(fp_cols) * 0.4)),
        constrained_layout=True,
    )
    sns.heatmap(
        mat_hm,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=CMAP_HEATMAP_SEQ,
        linewidths=0.5,
        cbar_kws={"label": "Mean rank (lower is better)"},
    )
    ax.set_title(
        f"Classifier mean rank across assays ({resampling_choice}; columns: classifier; rows: fingerprint)",
        fontsize=14,
        pad=10,
    )
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("Fingerprint", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

def _render_fp_tgt(
    benchmark_results: pd.DataFrame | None,
    best_per_combo: pd.DataFrame | None,
    roc_curves_raw: dict | None,
    optimized_for: str | None,
) -> None:
    st.subheader("Best fingerprints (final test)")
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found.")
        return

    score_col = None
    if optimized_for:
        preferred = f"final_test_{optimized_for}"
        if preferred in benchmark_results.columns:
            score_col = preferred
    if score_col is None:
        test_cols = [c for c in benchmark_results.columns if c.startswith("final_test_") and c != "final_test_n"]
        score_col = test_cols[0] if test_cols else None
    if score_col is None:
        st.info("No final test metric columns found.")
        return

    metric_fp_display = _format_metric_name(score_col)
    st.caption(f"Using final test metric: {metric_fp_display}")

    model_opts = sorted(benchmark_results["model"].dropna().unique())
    resampling_opts = sorted(benchmark_results["resampling"].dropna().unique()) if "resampling" in benchmark_results.columns else ["none"]
    
    col1, col2 = st.columns(2)
    with col1:
        default_model_idx = model_opts.index("XGBoost") if "XGBoost" in model_opts else 0
        model_choice = st.selectbox("Model", model_opts, index=default_model_idx)
    with col2:
        resampling_choice = st.selectbox("Resampling", resampling_opts, index=0)
        
    df_fingerprint = benchmark_results[
        (benchmark_results["model"] == model_choice) &
        (benchmark_results.get("resampling", "none") == resampling_choice)
    ].dropna(subset=[score_col])
    
    if df_fingerprint.empty:
        st.info("No data for this model and resampling combination.")
        return

    mean_ranks, cd = compute_mean_ranks(df_fingerprint, group_col="fingerprint", target_col="target", metric_col=score_col)
    
    if mean_ranks.empty:
        st.info("Not enough data to compute ranks.")
    else:
        df_unique = df_fingerprint[["target", "fingerprint", score_col]].drop_duplicates(["target", "fingerprint"], keep="first")
        pivot = df_unique.pivot(index="target", columns="fingerprint", values=score_col).dropna(axis=0, how="any")
        if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
            _chi2, p_value = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
            st.caption(f"Friedman p = {p_value:.3g}")

        fig = _plot_nemenyi_cd_graph(
            mean_ranks, cd, 
            title=f"Fingerprint average ranks with Nemenyi CD ({model_choice} × {resampling_choice})", 
            ylabel="Molecular fingerprint encoding"
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Fingerprint Mean Rank Across Models (Heatmap)")
    
    df_heatmap = benchmark_results.dropna(subset=[score_col])
    if df_heatmap.empty:
        return
        
    heatmap_resampling = st.selectbox("Resampling for heatmap", resampling_opts, index=0, key="heatmap_resampling")
    df_heatmap = df_heatmap[df_heatmap.get("resampling", "none") == heatmap_resampling]
    
    if df_heatmap.empty:
        st.info("No data for this resampling method.")
        return
        
    models = sorted(df_heatmap["model"].unique())
    pivot_rows = []
    
    for m in models:
        subset_df = df_heatmap[df_heatmap["model"] == m]
        mean_ranks_series, _ = compute_mean_ranks(subset_df, group_col="fingerprint", target_col="target", metric_col=score_col)
        if not mean_ranks_series.empty:
            mean_ranks_series.name = m
            pivot_rows.append(mean_ranks_series)
            
    if not pivot_rows:
        st.info("Not enough data to compute heatmap.")
        return
        
    rank_df = pd.DataFrame(pivot_rows)
    
    fig, ax = plt.subplots(
        figsize=(max(10, len(rank_df.columns) * 0.8), max(6, len(rank_df) * 0.45)),
        constrained_layout=True,
    )
    sns.heatmap(
        rank_df, ax=ax, annot=True, fmt=".2f", cmap=CMAP_HEATMAP_SEQ,
        linewidths=0.5, cbar_kws={"label": "Mean rank (lower is better)"}
    )
    ax.set_title(f"Classifier mean rank across assays ({heatmap_resampling})", fontsize=14, pad=10)
    ax.set_xlabel("Fingerprint", fontsize=12)
    ax.set_ylabel("Classifier", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

def _render_best_per_tgt(benchmark_results: pd.DataFrame | None) -> None:
    st.subheader("Best model × fingerprint per target")
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found.")
    else:
        metric_cols = _get_metric_columns(benchmark_results)
        metric_names = [_format_metric_name(c) for c in metric_cols]
        if not metric_names:
            st.info("No metric columns found.")
        else:
            target_opts = sorted(_options(benchmark_results["target"]))
            target_choice = st.selectbox(
                "Target",
                ["All"] + (target_opts or []),
                index=0,
                key="best_per_tgt_target",
            )
            metric_display = st.selectbox(
                "Metric (higher = better)",
                metric_names,
                index=metric_names.index("val roc_auc") if "val roc_auc" in metric_names else 0,
                key="best_per_tgt_metric",
            )
            metric_col = _get_metric_column_from_name(metric_display, metric_cols)
            df_clean = benchmark_results.dropna(subset=[metric_col])
            if target_choice != "All":
                df_clean = df_clean[df_clean["target"] == target_choice]
            cols = ["target", "model", "fingerprint", metric_col, "best_params"]
            if "resampling" in benchmark_results.columns:
                cols.append("resampling")
            best_per_target = df_clean.nlargest(20, metric_col)[cols].copy()
            best_per_target = best_per_target.rename(columns={metric_col: "best_value"})
            if best_per_target.empty:
                st.info("No rows with valid metric values.")
            else:
                st.dataframe(best_per_target, use_container_width=True)
                st.markdown("### Best value by target (top 20 runs)")
                best_per_target = best_per_target.copy()
                best_per_target["rank"] = range(1, len(best_per_target) + 1)
                best_per_target["run_label"] = best_per_target["rank"].astype(str) + ". " + best_per_target["target"] + " | " + best_per_target["model"]
                best_per_target = best_per_target.sort_values("best_value", ascending=False).reset_index(drop=True)
                sort_order = best_per_target["run_label"].tolist()
                tooltip_cols = ["rank", "target", "model", "fingerprint", "best_value", "best_params"]
                if "resampling" in best_per_target.columns:
                    tooltip_cols.append("resampling")
                metric_domain = _metric_domain(best_per_target["best_value"], include_zero=True)
                chart_runs = (
                    alt.Chart(best_per_target)
                    .mark_bar(size=14)
                    .encode(
                        y=alt.Y("run_label:N", sort=sort_order, title="Run (target | model)", axis=alt.Axis(labelLimit=120)),
                        x=alt.X("best_value:Q", title=f"Best {metric_display}", scale=alt.Scale(domain=metric_domain)),
                        color=alt.Color("model:N", title="Model"),
                        tooltip=tooltip_cols,
                    )
                    .properties(width=CHART_WIDTH, height=min(420, 28 * len(best_per_target)))
                )
                st.altair_chart(chart_runs, use_container_width=True)


def _render_assay_boxplot(benchmark_results: pd.DataFrame | None, optimized_for: str | None) -> None:
    st.subheader("Distribution of test metric per assay")
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found.")
        return

    score_col = None
    if optimized_for:
        preferred = f"final_test_{optimized_for}"
        if preferred in benchmark_results.columns:
            score_col = preferred
    if score_col is None:
        if "final_test_roc_auc" in benchmark_results.columns:
            score_col = "final_test_roc_auc"
        else:
            test_cols = [c for c in benchmark_results.columns if c.startswith("final_test_") and c != "final_test_n"]
            score_col = test_cols[0] if test_cols else None
    if score_col is None:
        st.info("No final test metric columns found.")
        return

    df_source = benchmark_results.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        model_opts = sorted(_options(df_source["model"])) if "model" in df_source.columns else []
        model_sel = st.multiselect(
            "Models",
            options=model_opts,
            default=model_opts,
            key="assay_score_models",
        )
    with col2:
        fp_opts = sorted(_options(df_source["fingerprint"])) if "fingerprint" in df_source.columns else []
        fp_sel = st.multiselect(
            "Fingerprints",
            options=fp_opts,
            default=fp_opts,
            key="assay_score_fps",
        )
    with col3:
        if "resampling" in df_source.columns:
            rs_opts = sorted(_options(df_source["resampling"]))
            rs_sel = st.multiselect(
                "Resampling",
                options=rs_opts,
                default=rs_opts,
                key="assay_score_resampling",
            )
        else:
            rs_sel = None

    if model_sel and "model" in df_source.columns:
        df_source = df_source[df_source["model"].isin(model_sel)].copy()
    if fp_sel and "fingerprint" in df_source.columns:
        df_source = df_source[df_source["fingerprint"].isin(fp_sel)].copy()
    if rs_sel is not None and rs_sel and "resampling" in df_source.columns:
        df_source = df_source[df_source["resampling"].isin(rs_sel)].copy()

    df = df_source[["target", score_col]].dropna(subset=[score_col]).copy()
    if df.empty:
        st.info("No rows with valid values for this test metric.")
        return

    MAIN_BLUE = "#4a6fa5"
    VLAG_BLUE = _vlag_base(0.15)
    GRID_GREY = "#b8c4d9"

    order_targets = (
        df.groupby("target")[score_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.boxplot(
        data=df,
        x="target",
        y=score_col,
        order=order_targets,
        ax=ax,
        color=MAIN_BLUE,
        medianprops={"color": VLAG_BLUE, "linewidth": 1.8},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.3},
    )
    metric_label = _format_metric_name(score_col)
    ax.set_title(f"Distribution of {metric_label} per assay", pad=10)
    ax.set_xlabel("Assay Target")
    ax.set_ylabel(metric_label)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, color=GRID_GREY)
    st.pyplot(fig)
    plt.close(fig)


def _render_resampling_cmp(benchmark_results_all: pd.DataFrame | None, optimized_for: str | None) -> None:
    st.subheader("Best resampling (final test)")
    if benchmark_results_all is None or "resampling" not in benchmark_results_all.columns:
        st.warning("benchmark_results.csv not found or missing 'resampling' column.")
        return

    score_col = None
    if optimized_for:
        preferred = f"final_test_{optimized_for}"
        if preferred in benchmark_results_all.columns:
            score_col = preferred
    if score_col is None:
        test_cols = [c for c in benchmark_results_all.columns if c.startswith("final_test_") and c != "final_test_n"]
        score_col = test_cols[0] if test_cols else None
    if score_col is None:
        st.info("No final test metric columns found.")
        return

    metric_display = _format_metric_name(score_col)
    st.caption(f"Using final test metric: {metric_display}")

    df = benchmark_results_all.dropna(subset=[score_col]).copy()
    if df.empty:
        st.info("No rows with valid final test metric values.")
        return

    imb_models = {"BalancedRandomForest", "EasyEnsemble", "RUSBoost", "BalancedBagging"}
    model_opts = sorted([m for m in _options(df["model"]) if m not in imb_models])
    fp_opts = sorted(_options(df["fingerprint"]))
    resampling_opts = sorted(_options(df["resampling"]))

    col1, col2 = st.columns(2)
    with col1:
        model_options = model_opts or ["(no models)"]
        default_model_idx = model_options.index("XGBoost") if "XGBoost" in model_options else 0
        model_choice = st.selectbox("Model", model_options, index=default_model_idx, key="best_resampling_model")
    with col2:
        fp_choice = st.selectbox("Fingerprint (for CD plot)", fp_opts or ["(no fingerprints)"], key="best_resampling_fp")

    df_model = df[df["model"] == model_choice].copy()
    if df_model.empty:
        st.info("No rows for selected model.")
        return

    resampling_order = ["none", "smoteenn", "smotetomek", "smote", "random_over", "random_under", "imbalanced_models"]
    resampling_order = [r for r in resampling_order if r in set(df_model["resampling"].dropna().unique())]
    for r in sorted(set(df_model["resampling"].dropna().unique()) - set(resampling_order)):
        resampling_order.append(r)

    st.divider()
    st.markdown("### Resampling ranks (Nemenyi CD)")
    st.caption("Ranks are computed per target across resampling methods (lower = better).")

    df_fingerprint = df_model[df_model["fingerprint"] == fp_choice].copy()
    if df_fingerprint.empty:
        st.info("No rows for this fingerprint.")
        return

    df_unique = df_fingerprint[["target", "resampling", score_col]].drop_duplicates(["target", "resampling"], keep="first")
    pivot = df_unique.pivot(index="target", columns="resampling", values=score_col)
    pivot = pivot.reindex(columns=resampling_order)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty or pivot.shape[1] < 2:
        st.info("Not enough data to compute resampling ranks for this selection.")
        return

    chi2, p_value = stats.friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean().sort_values()
    cd = nemenyi_cd(int(pivot.shape[0]), int(pivot.shape[1]))
    st.caption(f"Friedman p = {p_value:.3g}")

    fig = _plot_nemenyi_cd_graph(
        mean_ranks,
        cd,
        title=f"Resampling average ranks with Nemenyi CD ({model_choice} × {fp_choice})",
        ylabel="Resampling",
    )
    st.pyplot(fig)
    plt.close(fig)

    fp_cols = sorted(df_model["fingerprint"].dropna().unique().tolist())
    targets = sorted(df_model["target"].dropna().unique().tolist())

    series_by_r: dict[str, pd.Series] = {}
    for r in resampling_order:
        ranks_per_fp: dict[str, list[float]] = {fp: [] for fp in fp_cols}
        for t in targets:
            block = df_model[(df_model["target"] == t) & (df_model["resampling"] == r)]
            series = (
                block.drop_duplicates("fingerprint", keep="first")
                .set_index("fingerprint")[score_col]
                .dropna()
            )
            if len(series) < 2:
                continue
            rank_series = series.rank(ascending=False, method="average")
            for fp in fp_cols:
                if fp in rank_series.index:
                    ranks_per_fp[fp].append(float(rank_series[fp]))
        series_by_r[r] = pd.Series(
            {fp: float(np.mean(ranks_per_fp[fp])) if ranks_per_fp[fp] else np.nan for fp in fp_cols}
        )

    heatmap_df = pd.DataFrame(series_by_r).reindex(columns=resampling_order)  # fingerprint x resampling
    if heatmap_df.empty or not np.any(np.isfinite(heatmap_df.to_numpy(dtype=float))):
        st.info("Not enough data to build heatmap.")
    else:
        fig, ax = plt.subplots(
            figsize=(max(10, len(resampling_order) * 1.1), max(6, len(fp_cols) * 0.4)),
            constrained_layout=True,
        )
        sns.heatmap(
            heatmap_df,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=CMAP_HEATMAP_SEQ,
            linewidths=0.5,
            cbar_kws={"label": "Mean rank (lower is better)"},
        )
        ax.set_title(
            f"Fingerprint mean rank across assays ({model_choice}; lower = better)",
            fontsize=14,
            pad=10,
        )
        ax.set_xlabel("Resampling", fontsize=12)
        ax.set_ylabel("Fingerprint", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Bump plot: fingerprint mean rank vs resampling")
    if "none" in resampling_order:
        resampling_choices = [r for r in resampling_order if r != "none"]
        default_choices = [r for r in ["smote", "smoten", "smoteenn", "smoten_enn"] if r in resampling_choices][:2]
        bump_choices = st.multiselect(
            "Resampling methods to compare against baseline (baseline is always included)",
            options=resampling_choices,
            default=default_choices,
            key="best_resampling_bump_choices",
        )
        selected_resampling_methods = ["none"] + bump_choices
    else:
        selected_resampling_methods = st.multiselect(
            "Resampling methods to show (baseline not available)",
            options=resampling_order,
            default=resampling_order[: min(3, len(resampling_order))],
            key="best_resampling_bump_cols_no_none",
        )

    if heatmap_df is not None and selected_resampling_methods and len(selected_resampling_methods) >= 3:
        subset_df = heatmap_df[selected_resampling_methods].dropna(how="any")
        if subset_df.empty:
            st.info("No complete rows for the selected resampling methods.")
        else:
            subset_df = subset_df.sort_values(by=selected_resampling_methods[0], ascending=True, kind="mergesort")
            plot_data = (-subset_df).T.reset_index()
            plot_data = plot_data.rename(columns={plot_data.columns[0]: "x"})
            fingerprint_columns = [c for c in plot_data.columns if c != "x"]

            try:
                from bumplot import bumplot
            except Exception:
                st.warning("`bumplot` is not available; cannot render thesis-style bump plot.")
                y = subset_df.to_numpy(dtype=float)
                fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * len(selected_resampling_methods)), 4.8), constrained_layout=True)
                for i, fp in enumerate(subset_df.index):
                    ax.plot(np.arange(len(selected_resampling_methods)), y[i], marker="o", lw=1.2, ms=5.0, alpha=0.9)
                ax.invert_yaxis()
                ax.set_xticks(np.arange(len(selected_resampling_methods), dtype=float))
                ax.set_xticklabels([str(c) for c in selected_resampling_methods], fontsize=10)
                st.pyplot(fig)
                plt.close(fig)
            else:
                fp_to_ci = {fp: i for i, fp in enumerate(fingerprint_columns)}
                cmap = plt.get_cmap("tab20")
                colors = [cmap((fp_to_ci.get(k, 0) % 20) / 19.0) for k in fingerprint_columns]

                fig, ax = plt.subplots(figsize=(8.0, 4.25))
                ax, art = bumplot(
                    x="x",
                    y_columns=fingerprint_columns,
                    data=plot_data,
                    ax=ax,
                    curve_force=0.5,
                    invert_y_axis=True,
                    colors=colors,
                    plot_kwargs={"lw": 1.0},
                    scatter_kwargs={"s": 36, "ec": "0.25", "lw": 0.4, "zorder": 3},
                )
                ax.set_title(f"{model_choice}: mean rank vs resampling", fontsize=13, pad=10)
                ax.set_xlabel("Resampling", fontsize=11, labelpad=5)
                ax.set_ylabel("Rank (1 = best)", fontsize=11, labelpad=5)
                ax.grid(True, axis="y", alpha=0.3, linestyle="--")
                ax.grid(True, axis="x", alpha=0.2, linestyle=":")
                ax.tick_params(axis="x", labelsize=11)
                ax.tick_params(axis="y", labelsize=11)
                ax.set_xticks(np.arange(len(selected_resampling_methods), dtype=float))
                ax.set_xticklabels([str(c) for c in selected_resampling_methods], ha="center", fontsize=11)

                ann_fs = 8
                for k in fingerprint_columns:
                    o = art[k][1].get_offsets()
                    if len(o) < 1:
                        continue
                    x0, y0 = o[0]
                    ax.annotate(
                        k,
                        (x0, y0),
                        textcoords="offset points",
                        xytext=(-4, 0),
                        ha="right",
                        va="center",
                        fontsize=ann_fs,
                        color="0.2",
                    )

                xa, xb = ax.get_xlim()
                ax.set_xlim(xa - 0.32 * (xb - xa + 0.01), xb)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    elif selected_resampling_methods and len(selected_resampling_methods) < 3:
        st.info("Select at least 2 resampling methods in addition to baseline to draw a bump plot.")

def _render_per_class(
    per_class_counts: pd.DataFrame | None,
    best_per_combo: pd.DataFrame | None,
    benchmark_results: pd.DataFrame | None,
    has_resampling: bool,
    resampling_filter: str | None,
    current_dataset_label: str,
) -> None:
    st.subheader("Per-class (0/1) behavior on train/val")
    if per_class_counts is None:
        st.warning("per_class_counts.csv not found.")
    else:
        df = per_class_counts.copy()

        if df.empty:
            st.info("No per-class rows match the current filters.")
        else:
                split_opts = sorted(s for s in _options(df["split"]) if s != "test")
                split_choice = st.selectbox("Split", split_opts or ["train"], index=0)
                df_s = df[df["split"] == split_choice].copy()
                df_s["acc"] = df_s["n_correct"] / df_s["n_true"]
                df_s["class_str"] = df_s["class"].astype(str)

                has_best = best_per_combo is not None and not best_per_combo.empty
                scope_opts = ["All models"]
                if has_best:
                    scope_opts.append("Best models only")
                scope = st.radio("Scope", scope_opts, horizontal=True)
                if scope == "Best models only" and has_best:
                    best_df_scope = best_per_combo
                    best_keys = best_df_scope[["target", "fingerprint", "model"]].drop_duplicates()
                    if "resampling" in best_df_scope.columns and "resampling" in df_s.columns:
                        best_keys = best_df_scope[["target", "fingerprint", "model", "resampling"]].drop_duplicates()
                    merge_on_left = ["target", "finger_print_method", "model"]
                    merge_on_right = ["target", "fingerprint", "model"]
                    if "resampling" in best_keys.columns and "resampling" in df_s.columns:
                        merge_on_left.append("resampling")
                        merge_on_right.append("resampling")
                    df_s = df_s.merge(best_keys, left_on=merge_on_left, right_on=merge_on_right, how="inner").drop(columns=["fingerprint"], errors="ignore")
                    if df_s.empty:
                        st.info("No per-class rows for best models with current filters.")

                if not df_s.empty:
                    st.write("Per-class counts (all rows):")
                    st.dataframe(df_s, use_container_width=True)

                    st.divider()
                    st.markdown("### Final test metric trade-offs (per assay)")
                    if benchmark_results is None:
                        st.info("No benchmark results available for trade-off plots.")
                    else:
                        df_bt = benchmark_results.copy()
                        metric_x = "final_test_roc_auc" if "final_test_roc_auc" in df_bt.columns else None
                        y_candidates = [
                            c
                            for c in [
                                "final_test_pr_auc",
                                "final_test_accuracy",
                                "final_test_f1_macro",
                                "final_test_f1_weighted",
                            ]
                            if c in df_bt.columns
                        ]
                        if metric_x is None or not y_candidates:
                            st.info("Final test metrics not found for trade-off plots.")
                        else:
                            target_opts = sorted(_options(df_bt["target"])) if "target" in df_bt.columns else []
                            target_choice = st.selectbox(
                                "Target",
                                target_opts or ["(no targets)"],
                                key="perclass_tradeoff_target",
                            )
                            metric_y_disp = st.selectbox(
                                "Y metric",
                                [_format_metric_name(c) for c in y_candidates],
                                index=0,
                                key="perclass_tradeoff_y",
                            )
                            metric_y = _get_metric_column_from_name(metric_y_disp, y_candidates)

                            df_t = df_bt[df_bt["target"] == target_choice].dropna(subset=[metric_x, metric_y]).copy()
                            if df_t.empty:
                                st.info("No rows for this target/metric selection.")
                            else:
                                MAIN_BLUE = "#4a6fa5"
                                GRID_GREY = "#b8c4d9"
                                fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                                sns.scatterplot(
                                    data=df_t,
                                    x=metric_x,
                                    y=metric_y,
                                    ax=ax,
                                    color=MAIN_BLUE,
                                    edgecolor=MAIN_BLUE,
                                    linewidth=0.35,
                                    alpha=0.65,
                                )
                                ax.set_title(
                                    f"Trade-off between {_format_metric_name(metric_x)} and {_format_metric_name(metric_y)} ({target_choice})",
                                    pad=10,
                                )
                                ax.set_xlabel(_format_metric_name(metric_x))
                                ax.set_ylabel(_format_metric_name(metric_y))
                                ax.grid(True, linestyle="--", alpha=0.45, color=GRID_GREY)
                                st.pyplot(fig)
                                plt.close(fig)

                    rows_cm = []
                    for (tgt, model_name, fp_method), g in df_s.groupby(
                        ["target", "model", "finger_print_method"]
                    ):
                        rec = {
                            "target": tgt,
                            "model": model_name,
                            "finger_print_method": fp_method,
                            "best_params": g["parameters"].iloc[0] if "parameters" in g.columns else None,
                        }
                        tn = tp = fn = fp_ = np.nan
                        g0 = g[g["class"] == 0]
                        if not g0.empty:
                            tn = int(g0["n_correct"].iloc[0])
                            fp_ = int(g0["n_wrong"].iloc[0])
                        g1 = g[g["class"] == 1]
                        if not g1.empty:
                            tp = int(g1["n_correct"].iloc[0])
                            fn = int(g1["n_wrong"].iloc[0])
                        rec.update({"TN": tn, "TP": tp, "FN": fn, "FP": fp_})
                        rows_cm.append(rec)
                    cm_df = pd.DataFrame(rows_cm)
                    st.write("Confusion-style table (per target/model/fingerprint):")
                    st.dataframe(cm_df, use_container_width=True)

                    if scope == "Best models only":
                        st.markdown("### Best models: per-class comparison")
                        rows_comp = []
                        for (tgt, fp_method, model_name), g in df_s.groupby(
                            ["target", "finger_print_method", "model"]
                        ):
                            total_true = int(g["n_true"].sum())
                            total_correct = int(g["n_correct"].sum())
                            acc_overall = total_correct / total_true if total_true > 0 else np.nan
                            g0 = g[g["class"] == 0]
                            g1 = g[g["class"] == 1]
                            acc_0 = (g0["n_correct"].iloc[0] / g0["n_true"].iloc[0]) if not g0.empty and g0["n_true"].iloc[0] > 0 else np.nan
                            acc_1 = (g1["n_correct"].iloc[0] / g1["n_true"].iloc[0]) if not g1.empty and g1["n_true"].iloc[0] > 0 else np.nan
                            rows_comp.append({
                                "target": tgt,
                                "fingerprint": fp_method,
                                "model": model_name,
                                "best_params": g["parameters"].iloc[0] if "parameters" in g.columns else None,
                                "acc_class_0": round(acc_0, 4) if pd.notna(acc_0) else None,
                                "acc_class_1": round(acc_1, 4) if pd.notna(acc_1) else None,
                                "acc_overall": round(acc_overall, 4) if pd.notna(acc_overall) else None,
                            })
                        comp_df = pd.DataFrame(rows_comp)
                        st.dataframe(comp_df, use_container_width=True)
                        comp_plot = comp_df.dropna(subset=["acc_class_0", "acc_class_1"])
                        if not comp_plot.empty:
                            domain_acc0 = _metric_domain(comp_plot["acc_class_0"])
                            domain_acc1 = _metric_domain(comp_plot["acc_class_1"])
                            scatter_best = (
                                alt.Chart(comp_plot)
                                .mark_circle(size=70, opacity=0.7)
                                .encode(
                                    x=alt.X("acc_class_0:Q", title="Class 0 accuracy", scale=alt.Scale(domain=domain_acc0)),
                                    y=alt.Y("acc_class_1:Q", title="Class 1 accuracy", scale=alt.Scale(domain=domain_acc1)),
                                    color=alt.Color("model:N", title="Model"),
                                    tooltip=["target", "fingerprint", "model", "best_params", "acc_class_0", "acc_class_1", "acc_overall"],
                                )
                                .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
                            )
                            st.altair_chart(scatter_best, use_container_width=True)

                    st.markdown("### Distribution of Per-Class Accuracy")
                    box_title = f"{split_choice} Accuracy Distribution" + (f" ({current_dataset_label})" if current_dataset_label != "—" else "")
                    domain_box_pc = _metric_domain(df_s["acc"])
                    box_pc = (
                        alt.Chart(df_s)
                        .mark_boxplot(extent="min-max")
                        .encode(
                            x=alt.X("model:N", title="Model", axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("acc:Q", title=box_title, scale=alt.Scale(domain=domain_box_pc)),
                            color=alt.Color("class_str:N", title="Class"),
                            xOffset="class_str:N"
                        )
                        .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
                    )
                    st.altair_chart(box_pc, use_container_width=True)

                    st.markdown("### Best Run Per-Class Accuracy by Model")
                    rows_run = []
                    for (tgt, fp_method, model_name), g in df_s.groupby(
                        ["target", "finger_print_method", "model"]
                    ):
                        total_true = int(g["n_true"].sum())
                        total_correct = int(g["n_correct"].sum())
                        acc_overall = total_correct / total_true if total_true > 0 else np.nan
                        g0 = g[g["class"] == 0]
                        g1 = g[g["class"] == 1]
                        acc_0 = (g0["n_correct"].iloc[0] / g0["n_true"].iloc[0]) if not g0.empty and g0["n_true"].iloc[0] > 0 else np.nan
                        acc_1 = (g1["n_correct"].iloc[0] / g1["n_true"].iloc[0]) if not g1.empty and g1["n_true"].iloc[0] > 0 else np.nan
                        rows_run.append({
                            "target": tgt,
                            "fingerprint": fp_method,
                            "model": model_name,
                            "acc_class_0": acc_0,
                            "acc_class_1": acc_1,
                            "acc_overall": acc_overall,
                        })
                    runs_df = pd.DataFrame(rows_run).dropna(subset=["acc_overall"])
                    if not runs_df.empty:
                        idx_best = runs_df.groupby("model")["acc_overall"].idxmax()
                        best_runs = runs_df.loc[idx_best].copy()
                        best_long_0 = best_runs[["model", "acc_class_0", "target", "fingerprint"]].copy()
                        best_long_0 = best_long_0.rename(columns={"acc_class_0": "acc"})
                        best_long_0["class_str"] = "0"
                        best_long_1 = best_runs[["model", "acc_class_1", "target", "fingerprint"]].copy()
                        best_long_1 = best_long_1.rename(columns={"acc_class_1": "acc"})
                        best_long_1["class_str"] = "1"
                        best_long = pd.concat([
                            best_long_0[["model", "class_str", "acc", "target", "fingerprint"]],
                            best_long_1[["model", "class_str", "acc", "target", "fingerprint"]],
                        ], ignore_index=True).dropna(subset=["acc"])
                        best_title = f"Best run {split_choice} accuracy (by overall acc)" + (f" ({current_dataset_label})" if current_dataset_label != "—" else "")
                        domain_best_pc = _metric_domain(best_long["acc"], include_zero=True)
                        chart_best_pc = (
                            alt.Chart(best_long)
                            .mark_bar()
                            .encode(
                                x=alt.X("model:N", title="Model", axis=alt.Axis(labelAngle=-45)),
                                y=alt.Y("acc:Q", title=best_title, scale=alt.Scale(domain=domain_best_pc)),
                                color=alt.Color("class_str:N", title="Class (0=Majority, 1=Minority)"),
                                xOffset="class_str:N",
                                tooltip=["model", "class_str", "acc", "target", "fingerprint"]
                            )
                            .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
                        )
                        st.altair_chart(chart_best_pc, use_container_width=True)
                    else:
                        st.caption("No run-level data to compute best run per model.")

                    st.markdown("### Class imbalance vs overall accuracy")
                    rows_imb = []
                    for (tgt, model_name, fp_method, split), g in df.groupby(
                        ["target", "model", "finger_print_method", "split"]
                    ):
                        total_true = int(g["n_true"].sum())
                        if total_true == 0:
                            continue
                        total_correct = int(g["n_correct"].sum())
                        acc_overall = total_correct / total_true
                        g1 = g[g["class"] == 1]
                        pos_true = int(g1["n_true"].iloc[0]) if not g1.empty else 0
                        pos_ratio = pos_true / total_true if total_true > 0 else np.nan
                        rows_imb.append(
                            {
                                "target": tgt,
                                "model": model_name,
                                "fingerprint": fp_method,
                                "split": split,
                                "acc_overall": acc_overall,
                                "pos_ratio": pos_ratio,
                            }
                        )
                    if rows_imb:
                        imb_df = pd.DataFrame(rows_imb)
                        imb_s = imb_df[imb_df["split"] == split_choice].dropna(subset=["pos_ratio"])
                        if imb_s.empty:
                            st.info("No imbalance data for this split.")
                        else:
                            domain_imb_x = _metric_domain(imb_s["pos_ratio"])
                            domain_imb_y = _metric_domain(imb_s["acc_overall"])
                            imb_chart = (
                                alt.Chart(imb_s)
                                .mark_circle(size=70, opacity=0.7)
                                .encode(
                                    x=alt.X(
                                        "pos_ratio:Q",
                                        title="Positive class ratio (class=1)",
                                        scale=alt.Scale(domain=domain_imb_x),
                                    ),
                                    y=alt.Y(
                                        "acc_overall:Q",
                                        title=f"Overall {split_choice} accuracy",
                                        scale=alt.Scale(domain=domain_imb_y),
                                    ),
                                    color=alt.Color("model:N", title="Model"),
                                    shape=alt.Shape("fingerprint:N", title="Fingerprint"),
                                    tooltip=[
                                        "target",
                                        "model",
                                        "fingerprint",
                                        "split",
                                        "pos_ratio",
                                        "acc_overall",
                                    ],
                                )
                                .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
                            )
                            st.altair_chart(imb_chart, use_container_width=True)


def _render_generalization(
    benchmark_results: pd.DataFrame | None,
    benchmark_results_all: pd.DataFrame | None = None,
    has_resampling: bool = False,
    optimized_for: str | None = None,
) -> None:
    st.subheader("Generalization (Average Ranks on Final Test)")
    if benchmark_results is None:
        st.warning("benchmark_results.csv not found.")
        return

    df_gen = benchmark_results_all if (has_resampling and benchmark_results_all is not None) else benchmark_results

    score_col = None
    if optimized_for:
        preferred = f"final_test_{optimized_for}"
        if preferred in df_gen.columns:
            score_col = preferred
    if score_col is None:
        test_cols = [c for c in df_gen.columns if c.startswith("final_test_") and c != "final_test_n"]
        score_col = test_cols[0] if test_cols else None
    if score_col is None:
        st.info("No final test metrics found.")
        return

    st.caption(f"Using final test metric: {_format_metric_name(score_col)}")

    df_gen = df_gen.dropna(subset=[score_col]).copy()
    if df_gen.empty:
        st.info("No rows with valid final test metric values.")
        return

    resampling_vals = sorted(_options(df_gen["resampling"])) if "resampling" in df_gen.columns else ["none"]
    model_vals = sorted(_options(df_gen["model"]))

    col1, col2 = st.columns(2)
    with col1:
        resampling_sel = st.multiselect(
            "Resampling methods",
            options=resampling_vals,
            default=resampling_vals,
            key="gen_rank_resampling",
        )
    with col2:
        model_sel = st.multiselect(
            "Models",
            options=model_vals,
            default=model_vals,
            key="gen_rank_models",
        )

    if "resampling" in df_gen.columns:
        df_gen = df_gen[df_gen["resampling"].isin(resampling_sel)].copy()
    df_gen = df_gen[df_gen["model"].isin(model_sel)].copy()
    if df_gen.empty:
        st.info("No rows after applying filters.")
        return

    grp_cols = ["target", "model"]
    if "resampling" in df_gen.columns:
        grp_cols.append("resampling")

    df_rank = df_gen[grp_cols + ["fingerprint", score_col]].drop_duplicates(grp_cols + ["fingerprint"], keep="first")
    df_rank = df_rank.dropna(subset=[score_col]).copy()
    if df_rank.empty:
        st.info("Not enough data to compute ranks.")
        return

    df_rank["rank"] = df_rank.groupby(grp_cols)[score_col].rank(ascending=False, method="average")

    st.markdown("### Mean rank summary (lower is better)")
    if "resampling" in df_rank.columns:
        mean_rank_tbl = (
            df_rank.groupby(["resampling", "fingerprint"])["rank"]
            .mean()
            .reset_index()
            .sort_values(["resampling", "rank"], ascending=[True, True])
        )
    else:
        mean_rank_tbl = (
            df_rank.groupby(["fingerprint"])["rank"]
            .mean()
            .reset_index()
            .sort_values("rank", ascending=True)
        )
    st.dataframe(mean_rank_tbl, use_container_width=True)

    st.markdown("### Rank distribution box plot")
    MAIN_BLUE = "#4a6fa5"
    VLAG_BLUE = _vlag_base(0.15)
    GRID_GREY = "#b8c4d9"

    avg_over_resampling = True
    if "resampling" in df_rank.columns and len(resampling_sel) > 1:
        avg_over_resampling = st.radio(
            "Generalization view",
            ["Average over selected resampling (single box per fingerprint)", "Show one resampling (single box per fingerprint)"],
            index=0,
            horizontal=True,
            key="gen_rank_view",
        ) == "Average over selected resampling (single box per fingerprint)"

    if "resampling" in df_rank.columns and not avg_over_resampling:
        rs_one = st.selectbox(
            "Resampling to show",
            options=resampling_sel,
            index=0,
            key="gen_rank_one_resampling",
        )
        df_rank_view = df_rank[df_rank["resampling"] == rs_one].copy()
    else:
        df_rank_view = df_rank.copy()

    grp_avg = ["target", "fingerprint"]
    if "resampling" in df_rank_view.columns and not avg_over_resampling:
        grp_avg.insert(1, "resampling")
    df_avg = df_rank_view.groupby(grp_avg, as_index=False)["rank"].mean().rename(columns={"rank": "avg_rank"})

    order_fp = (
        df_avg.groupby("fingerprint")["avg_rank"]
        .median()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(order_fp)), 6), constrained_layout=True)
    sns.boxplot(
        data=df_avg,
        x="fingerprint",
        y="avg_rank",
        order=order_fp,
        ax=ax,
        color=MAIN_BLUE,
        medianprops={"color": VLAG_BLUE, "linewidth": 1.8},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.3},
    )
    if "resampling" in df_rank.columns and not avg_over_resampling:
        ax.set_title(f"Fingerprint average-rank distribution across targets ({rs_one})", pad=10)
    else:
        ax.set_title("Fingerprint average-rank distribution across targets", pad=10)

    ax.set_xlabel("Fingerprint")
    ax.set_ylabel("Average rank (1 = best)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, color=GRID_GREY)
    st.pyplot(fig)
    plt.close(fig)


def _render_final_test(best_per_combo: pd.DataFrame | None) -> None:
    st.subheader("Final Test Results")
    if best_per_combo is None or best_per_combo.empty:
        st.warning("best_per_target_fingerprint.csv not found or empty.")
        return
    test_cols = [
        c for c in best_per_combo.columns if c.startswith("final_test_") and c != "final_test_n"
    ]
    if not test_cols:
        st.info("No final test metric columns found.")
        return

    metric_names = [_format_metric_name(c) for c in test_cols]
    metric_choice_display = st.selectbox(
        "Metric to view (higher = better)",
        metric_names,
        index=metric_names.index("test roc_auc") if "test roc_auc" in metric_names else 0,
        key="final_test_metric",
    )
    test_col = _get_metric_column_from_name(metric_choice_display, test_cols)

    val_opt_col: str | None = None
    if "optimized_for" in best_per_combo.columns:
        opt_values = best_per_combo["optimized_for"].dropna().unique()
        if len(opt_values) > 0:
            opt_metric = opt_values[0]
            candidate = f"val_{opt_metric}"
            if candidate in best_per_combo.columns:
                val_opt_col = candidate

    if val_opt_col is not None:
        df_best = best_per_combo.dropna(subset=[val_opt_col]).copy()
        if df_best.empty:
            st.info(
                f"No rows with validation metric '{val_opt_col}' to select best models for final test."
            )
            return

        idx_best = df_best.groupby("target")[val_opt_col].idxmax()
        best_overall = (
            df_best.loc[idx_best]
            .sort_values(val_opt_col, ascending=False)
            .reset_index(drop=True)
        )

        best_overall = best_overall.dropna(subset=[test_col])
        if best_overall.empty:
            st.info(
                "Best validation models have no recorded values for the selected test metric."
            )
            return

        df_test_for_plot = best_overall
    else:
        df_test = best_per_combo.dropna(subset=[test_col]).copy()
        if df_test.empty:
            st.info("No valid rows for this test metric.")
            return
        idx_best = df_test.groupby("target")[test_col].idxmax()
        best_overall = (
            df_test.loc[idx_best]
            .sort_values(test_col, ascending=False)
            .reset_index(drop=True)
        )
        df_test_for_plot = df_test

    st.markdown(f"### Best models by {metric_choice_display}")

    disp_cols = ["target", "fingerprint", "model"]
    if "resampling" in best_overall.columns:
        disp_cols.append("resampling")
    disp_cols.extend([test_col, "best_params"])
    
    st.dataframe(best_overall[disp_cols], use_container_width=True)

    if test_col == "final_test_roc_auc":
        st.divider()
        st.markdown("### Thesis Benchmark vs DeepTox (Test ROC-AUC)")
        st.caption(
            "Test ROC-AUC on the held-out test set, comparing the thesis benchmark with DeepTox."
        )
        deeptox_rows = [
            ("NR-AhR", 0.870, 0.928),
            ("NR-AR", 0.664, 0.807),
            ("NR-AR-LBD", 0.758, 0.879),
            ("NR-Aromatase", 0.775, 0.834),
            ("NR-ER", 0.666, 0.810),
            ("NR-ER-LBD", 0.735, 0.814),
            ("NR-PPAR-gamma", 0.694, 0.861),
            ("SR-ARE", 0.746, 0.840),
            ("SR-ATAD5", 0.756, 0.793),
            ("SR-HSE", 0.809, 0.865),
            ("SR-MMP", 0.897, 0.942),
            ("SR-p53", 0.772, 0.862),
        ]
        df_deeptox = pd.DataFrame(
            deeptox_rows, columns=["Endpoint", "Thesis Benchmark", "DeepTox"]
        )

        app_scores = (
            best_overall[["target", test_col]]
            .dropna(subset=[test_col])
            .drop_duplicates(["target"], keep="first")
            .set_index("target")[test_col]
        )
        app_nr = app_scores[app_scores.index.astype(str).str.startswith("NR-")]
        app_sr = app_scores[app_scores.index.astype(str).str.startswith("SR-")]
        app_avg_map: dict[str, float] = {k: float(v) for k, v in app_scores.items()}
        df_deeptox["This app"] = df_deeptox["Endpoint"].map(app_avg_map)

        st.dataframe(
            df_deeptox.style.format(
                {"Thesis Benchmark": "{:.3f}", "DeepTox": "{:.3f}", "This app": "{:.3f}"}
            ),
            use_container_width=True,
        )

        st.markdown("#### Averages")
        df_avg = pd.DataFrame(
            [
                ("NR Average", 0.737, 0.826, float(app_nr.mean()) if len(app_nr) else np.nan),
                ("SR Average", 0.796, 0.858, float(app_sr.mean()) if len(app_sr) else np.nan),
                ("Overall Average", 0.762, 0.846, float(app_scores.mean()) if len(app_scores) else np.nan),
            ],
            columns=["Endpoint", "Thesis Benchmark", "DeepTox", "This app"],
        )
        st.dataframe(
            df_avg.style.format(
                {"Thesis Benchmark": "{:.3f}", "DeepTox": "{:.3f}", "This app": "{:.3f}"}
            ),
            use_container_width=True,
        )

if __name__ == "__main__":
    main()

