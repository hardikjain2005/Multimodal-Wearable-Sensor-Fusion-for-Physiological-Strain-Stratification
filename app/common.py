"""Shared helpers for the Streamlit dashboard: cached models/data, prediction, explanation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import feature_engineering as fe
from src.config import METRICS_DIR, FIGURES_DIR, STRAIN_ORDER, RANDOM_SEED
from src.evaluation import align_proba
from src.inference import load_models

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "assets"
CORE = fe.HR_KEYS + fe.HRV_MODEL_KEYS + fe.ECG_KEYS
CLASS_COLORS = {"Low": "#2a78d6", "Moderate": "#1baf7a", "High": "#eb6834"}
CLASS_DOMAIN, CLASS_RANGE = list(CLASS_COLORS), list(CLASS_COLORS.values())
REPO_URL = "https://github.com/hardikjain2005/Multimodal-Wearable-Sensor-Fusion-for-Physiological-Strain-Stratification"
MODEL_NAMES = {"proposed": "Proposed: RF + GB soft-voting ensemble (HR + HRV + ECG)",
               "baseline": "Baseline: logistic regression (heart rate only)"}

# Plain-language names for features shown in the UI
LABELS = {
    "hr_mean": "Mean heart rate (bpm)", "hr_std": "Heart-rate variability within the minute, SD (bpm)",
    "hr_slope": "Heart-rate trend (bpm per minute)", "hrv_sdnn": "SDNN (ms)", "hrv_rmssd": "RMSSD (ms)",
    "hrv_pnn50": "pNN50 (%)", "hrv_cv_rr": "Coefficient of variation of RR",
    "ecg_kurtosis": "ECG kurtosis (spikiness)", "ecg_hf_noise_ratio": "ECG high-frequency noise ratio",
    "ecg_lf_wander_ratio": "ECG baseline-wander ratio", "ecg_ramp_cv": "R-peak amplitude variability (CV)",
    "ecg_beat_artifact_frac": "Fraction of artifact beats",
    "hr_median": "Median HR", "hr_min": "Min HR", "hr_max": "Max HR", "hr_range": "HR range",
    "hr_p10": "HR 10th percentile", "hr_p90": "HR 90th percentile",
}


@st.cache_resource(show_spinner="Loading models…")
def models():
    return load_models()


@st.cache_data
def reference() -> dict:
    return json.loads((ASSETS / "feature_reference.json").read_text())


@st.cache_data
def dev_sample() -> pd.DataFrame:
    return pd.read_csv(ASSETS / "dev_sample.csv")


@st.cache_data
def metrics(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / name)


def figure(name: str) -> str:
    return str(FIGURES_DIR / name)


def predict_proba(X: pd.DataFrame) -> dict:
    """Class probabilities (columns ordered Low, Moderate, High) from both locked models."""
    out = {}
    for key, bundle in models().items():
        out[key] = align_proba(bundle["model"], X[bundle["features"]].to_numpy(float))
    return out


@st.cache_resource(show_spinner=False)
def _explainer():
    import shap
    bg = pd.read_csv(ASSETS / "shap_background.csv")[CORE]
    model = models()["proposed"]["model"]
    return shap.PermutationExplainer(lambda X: model.predict_proba(np.asarray(X, float)), bg, seed=RANDOM_SEED)


def explain(x: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions of each input to every class probability for ONE input row."""
    sv = _explainer()(x[CORE], max_evals=2 * len(CORE) + 1, silent=True)
    classes = list(models()["proposed"]["model"].classes_)
    df = pd.DataFrame(sv.values[0], index=CORE, columns=classes)
    return df[STRAIN_ORDER]


def disclaimer():
    st.caption("⚠️ Research and education only, **not a medical device**. Predictions are *activity-derived* "
               "strain categories learned from 11 recording sessions of essentially one person. "
               "They are not a clinical assessment.")


def sidebar_footer():
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Project:** [GitHub repository]({REPO_URL})")
        st.caption("Healthcare Analytics Mini Project · Hardik Jain · Data: AIDLAB wearable datasets")
