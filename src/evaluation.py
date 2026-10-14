"""Leakage-safe evaluation: session-grouped folds, nested hyper-parameter tuning, metrics.

Why grouped CV: consecutive 60-s windows from one recording are nearly identical (same
person, same day, same sensor placement). With ordinary random K-fold, windows from the
same session land in both training and test folds, so the model is tested on data it has
effectively already seen and scores are inflated. Here every session is entirely in either
the training or the test part of a fold.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from .config import STRAIN_ORDER

LABEL_TO_INT = {c: i for i, c in enumerate(STRAIN_ORDER)}


# --------------------------------------------------------------------------- transformer
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Unsupervised redundancy filter, fitted on TRAINING data only (inside each fold).
    Walks features in their given order and drops any feature whose |Spearman rho| with an
    already-kept feature exceeds `threshold`. Label-free, so it cannot leak the target."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.rank().corr().abs().to_numpy()
        keep = []
        for j in range(X.shape[1]):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_idx_ = np.array(keep)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features) if input_features is not None else np.arange(self.n_features_in_)
        return names[self.keep_idx_]


# --------------------------------------------------------------------------- folds
def session_folds(df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Assign every SESSION to a fold, round-robin within each class (sorted by activity,
    then date), so each fold tests on unseen sessions of every class where possible.
    Returns a fold id per window (row of df)."""
    sess = (df.groupby("session_id")
            .agg(label=("strain_label", "first"), activity=("activity", "first"),
                 start=("window_start_ms", "min")).reset_index()
            .sort_values(["label", "activity", "start"]))
    sess["fold"] = sess.groupby("label").cumcount() % n_folds
    return df["session_id"].map(sess.set_index("session_id")["fold"]).to_numpy()


def fold_table(df: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame({"session_id": df.session_id, "label": df.strain_label, "fold": folds})
    return t.drop_duplicates("session_id").sort_values(["fold", "label", "session_id"])


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, proba=None, labels=STRAIN_ORDER) -> dict:
    """Macro averages treat every strain level as equally important regardless of how many
    windows it has (important because Moderate is the minority class)."""
    present = [c for c in labels if c in set(y_true)]
    m = dict(accuracy=accuracy_score(y_true, y_pred),
             balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
             precision_macro=precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
             recall_macro=recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
             f1_macro=f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, c in enumerate(labels):
        if c not in present:
            continue
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        m[f"sensitivity_{c}"] = tp / (tp + fn) if tp + fn else np.nan        # = recall of class c
        m[f"specificity_{c}"] = tn / (tn + fp) if tn + fp else np.nan
        m[f"precision_{c}"] = tp / (tp + fp) if tp + fp else np.nan
        m[f"f1_{c}"] = f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)
    if proba is not None:
        y_bin = np.array([[t == c for c in labels] for t in y_true], dtype=int)
        aucs = [roc_auc_score(y_bin[:, i], proba[:, i]) for i, c in enumerate(labels) if c in present
                and 0 < y_bin[:, i].sum() < len(y_bin)]
        # macro one-vs-rest AUC; only defined for classes present in y_true
        m["roc_auc_ovr_macro"] = float(np.mean(aucs)) if aucs else np.nan
    return m


# --------------------------------------------------------------------------- nested CV
def inner_cv_feasible(y, inner_folds) -> bool:
    """Tuning needs >= 2 inner folds and every class present in every inner TRAINING part."""
    folds = np.unique(inner_folds)
    if len(folds) < 2:
        return False
    classes = set(y)
    return all(set(y[inner_folds != f]) == classes for f in folds)


def tune(estimator, grid: dict, X, y, inner_folds, default_params: dict | None = None) -> tuple:
    """Grid search with session-grouped inner folds; selection metric = macro-F1.
    If session-grouped tuning is impossible (too few sessions per class), the model is fitted
    with fixed default hyper-parameters instead, and this is recorded in the returned params."""
    if not grid:
        return clone(estimator).fit(X, y), {}
    if not inner_cv_feasible(y, inner_folds):
        est = clone(estimator).set_params(**(default_params or {}))
        return est.fit(X, y), {"FIXED_DEFAULTS(inner CV infeasible)": default_params}
    gs = GridSearchCV(estimator, grid, scoring="f1_macro", cv=PredefinedSplit(inner_folds),
                      n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def nested_cv(spec, df: pd.DataFrame, features: list[str], n_outer: int = 3, n_inner: int = 2):
    """Outer loop: session-grouped folds -> unbiased performance estimate.
    Inner loop: session-grouped folds on the outer-training part only -> hyper-parameters.
    `spec` is a ModelSpec (see ensemble_model.py) with .fit(X, y, df_train, n_inner).
    Returns (pooled out-of-fold metrics, per-fold metrics, out-of-fold predictions)."""
    X = df[features].to_numpy(float)
    y = df["strain_label"].to_numpy()
    outer = session_folds(df, n_outer)
    oof_pred = np.empty(len(df), dtype=object)
    oof_proba = np.zeros((len(df), len(STRAIN_ORDER)))
    per_fold, chosen = [], []
    for k in range(n_outer):
        tr, te = outer != k, outer == k
        model, params = spec.fit(X[tr], y[tr], df[tr].reset_index(drop=True), n_inner)
        proba = align_proba(model, X[te])
        oof_proba[te] = proba
        oof_pred[te] = np.array(STRAIN_ORDER)[proba.argmax(1)]
        fm = compute_metrics(y[te], oof_pred[te], proba)
        fm["fold"] = k
        per_fold.append(fm)
        chosen.append(params)
    pooled = compute_metrics(y, oof_pred, oof_proba)
    pooled["f1_macro_fold_mean"] = np.mean([f["f1_macro"] for f in per_fold])
    pooled["f1_macro_fold_std"] = np.std([f["f1_macro"] for f in per_fold], ddof=1)
    preds = pd.DataFrame({"session_id": df.session_id, "y_true": y, "y_pred": oof_pred,
                          **{f"p_{c}": oof_proba[:, i] for i, c in enumerate(STRAIN_ORDER)}})
    return pooled, pd.DataFrame(per_fold), preds, chosen


def align_proba(model, X) -> np.ndarray:
    """predict_proba with columns ordered as STRAIN_ORDER (robust to missing classes)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(STRAIN_ORDER)))
    for j, c in enumerate(model.classes_):
        out[:, STRAIN_ORDER.index(c)] = p[:, j]
    return out
