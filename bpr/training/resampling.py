"""Train-set resampling strategies for the benchmark pipeline."""

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

from bpr.constants import DEFAULT_RANDOM_STATE


def apply_resampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resampling: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    random_under_sampling_strategy: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply resampling to (X_train, y_train). resampling='none' returns unchanged."""
    if resampling == "none":
        return X_train, y_train
    if resampling == "smoteenn":
        resampler = SMOTEENN(
            random_state=random_state,
            enn=EditedNearestNeighbours(n_neighbors=3),
        )
    elif resampling == "smote":
        resampler = SMOTE(random_state=random_state)
    elif resampling == "smoten":
        resampler = SMOTEN(random_state=random_state)
    elif resampling == "smoten_enn":
        resampler = Pipeline(
            steps=[
                ("smoten", SMOTEN(random_state=random_state)),
                ("enn", EditedNearestNeighbours(n_neighbors=3)),
            ]
        )
    elif resampling == "random_over":
        resampler = RandomOverSampler(random_state=random_state)
    elif resampling == "random_under":
        resampler = RandomUnderSampler(
            sampling_strategy=random_under_sampling_strategy,
            random_state=random_state,
        )
    else:
        return X_train, y_train
    return resampler.fit_resample(X_train, y_train)
