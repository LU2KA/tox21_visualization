"""Approximate k-NN with Jaccard distance (pynndescent) for binary fingerprints."""

from __future__ import annotations

import warnings

import numpy as np
from pynndescent import NNDescent
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _as_bool01(X: np.ndarray) -> np.ndarray:
    if X.dtype in (bool, np.bool_):
        return X.astype(np.float32, copy=False)
    return (X > 0).astype(np.float32, copy=False)


def _nndescent_tuning(
    n_train: int,
    n_dim: int,
    n_neighbors: int,
    n_trees: int | None,
    max_c: int | None,
    n_iter: int | None,
) -> dict:
    n_neighbors = int(n_neighbors)
    n_dim = int(n_dim)
    n_train = int(n_train)
    build_k = min(
        max(8, 3 * n_neighbors + 1),
        max(2, n_train - 1),
    )
    t_val = n_trees
    n_trees = int(
        t_val
        if t_val is not None
        else min(64, max(4, int(4 * np.log2(n_train + 1))))
    )
    max_c = int(
        max_c
        if max_c is not None
        else min(256, max(40, 2 * n_dim, 4 * n_neighbors + 4))
    )
    n_iters = int(n_iter if n_iter is not None else (10 if n_dim > 2000 else 5))
    return {
        "n_neighbors": build_k,
        "n_trees": n_trees,
        "max_candidates": max_c,
        "n_iters": n_iters,
    }


class ApproximateJaccardKNeighborsClassifier(  # pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
    BaseEstimator, ClassifierMixin
):
    """Binary k-NN with Jaccard distance via pynndescent; supports `n_neighbors` and `weights` like sklearn."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        random_state: int | None = None,
        n_trees: int | None = None,
        max_candidates: int | None = None,
        n_iter: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_candidates = max_candidates
        self.n_iter = n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> ApproximateJaccardKNeighborsClassifier:
        """Build the Jaccard NNDescent index on binarized training features."""
        t = type_of_target(y)
        if t not in ("binary", "multiclass"):
            raise ValueError(f"expected binary or multiclass target, got {t!r}")
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=None,
            ensure_2d=True,
        )
        if y.shape[0] < 2:
            raise ValueError("n_samples < 2")
        self._X_f_ = _as_bool01(X)
        self._y_ = y.astype(int, copy=False)
        self.classes_ = np.unique(self._y_)
        n_train, n_dim = self._X_f_.shape
        k_query = int(min(self.n_neighbors, n_train - 1))
        k_query = max(1, k_query)
        self._k_query_ = k_query
        tun = _nndescent_tuning(
            n_train,
            n_dim,
            k_query,
            n_trees=self.n_trees,
            max_c=self.max_candidates,
            n_iter=self.n_iter,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*n_neighbors.*")
            self._index_ = NNDescent(
                self._X_f_,
                metric="jaccard",
                n_neighbors=tun["n_neighbors"],
                max_candidates=tun["max_candidates"],
                n_iters=tun["n_iters"],
                n_trees=tun["n_trees"],
                random_state=self.random_state,
            )
        return self

    def _neighbor_vote(
        self,
        neigh_idx: np.ndarray,
        dist: np.ndarray,
    ) -> np.ndarray:
        w = self.weights
        y = self._y_
        n_q, k = neigh_idx.shape
        if w == "uniform":
            wv = np.ones((n_q, k), dtype=np.float64)
        elif w == "distance":
            wv = 1.0 / (np.asarray(dist, dtype=np.float64) + 1e-10)
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")
        classes = self.classes_
        n_classes = len(classes)
        out = np.zeros((n_q, n_classes), dtype=np.float64)
        for ci, c in enumerate(classes):
            mask = y[neigh_idx] == c
            out[:, ci] = (mask * wv).sum(axis=1) / np.maximum(wv.sum(axis=1), 1e-10)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for ``X``."""
        proba = self.predict_proba(X)
        j = np.argmax(proba, axis=1)
        return self.classes_[j]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from neighbor votes."""
        check_is_fitted(self, ("_index_", "_y_"))
        Xv = _as_bool01(
            check_array(
                X,
                accept_sparse=False,
                dtype=None,
            )
        )
        if Xv.shape[1] != self._X_f_.shape[1]:
            raise ValueError("feature count mismatch with training data")
        k = self._k_query_
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*n_neighbors.*")
            neigh_idx, dist = self._index_.query(Xv, k=k)
        proba = self._neighbor_vote(neigh_idx, dist)
        s = proba.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        proba = proba / s
        return proba
