"""HR-only baseline.

Question answered: how well can activity-derived strain be predicted from heart-rate
information alone? Logistic regression is used because it is simple, interpretable
(one weight per feature per class) and a fair, commonly used reference model. The same HR
features are also given to every other model (see ensemble_model.py) so that improvements
can be attributed to features vs. algorithm separately.
"""
from .ensemble_model import ModelSpec, logistic_regression
from .feature_engineering import HR_KEYS

BASELINE_FEATURES = HR_KEYS


def baseline_spec(imbalance: str = "balanced") -> ModelSpec:
    return ModelSpec("Logistic Regression", "single", logistic_regression, imbalance)
