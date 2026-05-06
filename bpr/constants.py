"""Shared constants for dataset layout, splits, and defaults."""

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_FRACS = (0.7, 0.15, 0.15)
DEFAULT_RANDOM_STATE = 42
DATASET_EXT = ".csv"

SKIP_METHODS = {"original"}
ALL_METRICS = ["accuracy", "f1_weighted", "f1_macro", "roc_auc", "pr_auc"]
OPTIMIZE_METRICS = ALL_METRICS
RESAMPLING_VARIANTS = [
    "none",
    "random_over",
    "random_under",
    "smote",
    "smoteenn",
    "smoten",
    "smoten_enn",
]
IMBALANCE_MODELS = {"BalancedRandomForest", "EasyEnsemble", "RUSBoost"}

BENCHMARK_MODEL_ORDER = [
    "XGBoost",
    "LogisticRegression",
    "BernoulliNB",
    "RandomForest",
    "BalancedRandomForest",
    "EasyEnsemble",
    "RUSBoost",
    "KNeighbors",
    "DecisionTree",
]

TOX21_ASSAY_ORDER = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def order_assays(targets):
    """Sort assay names by `TOX21_ASSAY_ORDER`, then append any others alphabetically."""
    want = set(targets)
    out = [t for t in TOX21_ASSAY_ORDER if t in want]
    out.extend(sorted(want.difference(out)))
    return out


def order_models(names):
    """Sort model names by `BENCHMARK_MODEL_ORDER`, then append any others alphabetically."""
    want = {str(n) for n in names}
    out = [m for m in BENCHMARK_MODEL_ORDER if m in want]
    out.extend(sorted(want.difference(out)))
    return out
