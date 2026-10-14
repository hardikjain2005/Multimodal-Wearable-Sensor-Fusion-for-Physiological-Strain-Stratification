"""Model definitions: individual classifiers and the proposed hybrid ensembles.

Every model is a pipeline whose preprocessing (correlation filter, scaling, optional SMOTE)
is fitted inside the training data of each fold only.

Class imbalance (development: Low 1044, Moderate 218, High 874 windows) is handled by
`imbalance`:
    "balanced" : class weights inversely proportional to class frequency (default)
    "none"     : no correction
    "smote"    : SMOTE oversampling of the TRAINING fold only (imblearn pipeline)
"""
from dataclasses import dataclass, field

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_SEED
from .evaluation import CorrelationFilter, session_folds, tune


def _cw(imbalance):
    return "balanced" if imbalance == "balanced" else None


def _pipe(clf, imbalance, scale=False):
    steps = [("corr", CorrelationFilter(0.95))]
    if scale:
        steps.append(("scale", StandardScaler()))
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
        return ImbPipeline(steps + [("clf", clf)])
    return Pipeline(steps + [("clf", clf)])


def logistic_regression(imbalance="balanced"):
    return (_pipe(LogisticRegression(class_weight=_cw(imbalance), max_iter=5000), imbalance, scale=True),
            {"clf__C": [0.01, 0.1, 1, 10]})


def decision_tree(imbalance="balanced"):
    return (_pipe(DecisionTreeClassifier(class_weight=_cw(imbalance), random_state=RANDOM_SEED), imbalance),
            {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [10, 50]})


def random_forest(imbalance="balanced"):
    cw = "balanced_subsample" if imbalance == "balanced" else None
    return (_pipe(RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=1,
                                         random_state=RANDOM_SEED), imbalance),
            {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [5, 20],
             "clf__max_features": ["sqrt", 0.5]})


def gradient_boosting(imbalance="balanced"):
    """Histogram-based gradient boosting (sklearn HistGradientBoostingClassifier)."""
    return (_pipe(HistGradientBoostingClassifier(max_iter=300, early_stopping=False, class_weight=_cw(imbalance),
                                                 random_state=RANDOM_SEED), imbalance),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [7, 15],
             "clf__min_samples_leaf": [20, 50]})


# Fixed defaults used only when session-grouped inner tuning is impossible (very few sessions)
DEFAULTS = {
    "logistic_regression": {"clf__C": 1},
    "decision_tree": {"clf__max_depth": 5, "clf__min_samples_leaf": 10},
    "random_forest": {"clf__max_depth": None, "clf__min_samples_leaf": 5, "clf__max_features": "sqrt"},
    "gradient_boosting": {"clf__learning_rate": 0.1, "clf__max_leaf_nodes": 15, "clf__min_samples_leaf": 20},
}

INDIVIDUAL = {"Logistic Regression": logistic_regression, "Decision Tree": decision_tree,
              "Random Forest": random_forest, "Gradient Boosting": gradient_boosting}


@dataclass
class ModelSpec:
    """A model + its hyper-parameter grid; `fit` tunes on session-grouped inner folds."""
    name: str
    kind: str = "single"              # single | soft_voting | stacking
    builder: object = None
    imbalance: str = "balanced"
    best_params_: dict = field(default_factory=dict)

    def fit(self, X, y, df_train, n_inner):
        inner = session_folds(df_train, n_inner)
        if self.kind == "single":
            est, grid = self.builder(self.imbalance)
            model, params = tune(est, grid, X, y, inner, DEFAULTS[self.builder.__name__])
            return model, params
        # Ensembles: tune RF and GB separately on the same inner folds, then combine.
        rf_est, rf_grid = random_forest(self.imbalance)
        gb_est, gb_grid = gradient_boosting(self.imbalance)
        rf, rf_p = tune(rf_est, rf_grid, X, y, inner, DEFAULTS["random_forest"])
        gb, gb_p = tune(gb_est, gb_grid, X, y, inner, DEFAULTS["gradient_boosting"])
        params = {**{f"rf__{k}": v for k, v in rf_p.items()}, **{f"gb__{k}": v for k, v in gb_p.items()}}
        if self.kind == "soft_voting":
            # average of the two models' class probabilities (equal weights)
            model = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")
        else:
            # meta-learner (logistic regression) learns how to weight the base models' probabilities;
            # its training data are out-of-fold base predictions from SESSION-GROUPED inner folds
            model = StackingClassifier([("rf", rf), ("gb", gb)],
                                       final_estimator=LogisticRegression(class_weight="balanced", max_iter=5000),
                                       cv=list(PredefinedSplit(inner).split()) if len(set(inner)) > 1 else 2,
                                       stack_method="predict_proba")
        model.fit(X, y)
        return model, params


def all_specs(imbalance="balanced") -> list[ModelSpec]:
    specs = [ModelSpec(n, "single", b, imbalance) for n, b in INDIVIDUAL.items()]
    specs += [ModelSpec("Ensemble RF+GB (soft voting)", "soft_voting", None, imbalance),
              ModelSpec("Ensemble RF+GB (stacking)", "stacking", None, imbalance)]
    return specs
