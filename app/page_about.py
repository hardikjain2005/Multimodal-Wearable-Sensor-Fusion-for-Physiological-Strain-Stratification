"""Page 4 — what the project is, how it was done, and its limits."""
import streamlit as st

from app.common import REPO_URL, disclaimer, figure, reference

ref = reference()
st.title("ℹ️ About this project")
st.markdown(f"""
**Multimodal Wearable Sensor Fusion for Physiological Strain Stratification: A Hybrid HRV-Based Risk
Monitoring Approach.** Healthcare Analytics Mini Project · Hardik Jain · Course instructor: Dr. Prerna.

**Question.** Does combining heart rate (HR) with heart-rate variability (HRV) and ECG signal-quality/movement
features, using a hybrid Random Forest + Gradient Boosting ensemble, classify physiological strain better than
heart rate alone?

**Data.** [AIDLAB wearable datasets](https://aidlab.com/datasets). There are 18 chest-strap recordings of sleep,
yoga, hiking and squash, with 250 Hz ECG, RR intervals, HR, respiration and skin temperature. Activity is used as
a proxy label:

- sleep → **Low**
- yoga → **Moderate**
- hike and squash → **High**

**How the models were built**

1. Every signal keeps its own sampling rate. All signals are cut on the same 60-second windows; nothing is
   resampled.
2. Heartbeats are detected in the raw ECG, giving HRV features (SDNN, RMSSD, pNN50, CV-RR).
3. ECG quality and movement features are added: kurtosis, high-frequency noise, baseline wander, R-peak
   amplitude variability and artifact-beat fraction.
4. Data are split by **recording session**, never by minute, and the split was fixed before training. This
   gives:
   - **{ref['n_dev_sessions']} development sessions** ({ref['n_dev_windows']:,} minutes), for training and tuning;
   - **two held-out test sets**: later sessions of the same person, and a different person.
5. Cross-validation is nested and grouped by session. Six models are compared with an HR-only baseline, and an
   ablation measures what each feature group adds.

**Main result.** Macro-F1 rose from 0.663 (HR only) to 0.802 (proposed) in cross-validation, and from 0.628 to
0.901 on later, unseen sessions. Most of the gain comes from the added ECG and HRV information rather than from
the ensemble itself.
""")
st.image(figure("01_sensor_availability.png"), width=520,
         caption="Sensors differ between sessions, so the main model uses only signals present in every recording.")
st.markdown("""
**Limitations**

1. Activity is a *proxy* for strain. There are no clinically validated labels, and this is not a diagnosis.
2. Development data come from essentially **one person**. The different-person test covers only squash vs
   sleep.
3. Movement corrupts HRV during sport. In this data, HRV did **not** fall with strain.
4. ECG quality features partly reflect movement and recording conditions, not only physiology.
5. With only 11 development sessions, the uncertainty is large: scores vary by about ±0.1 between folds.

**Ways to use this dashboard**

- **Predict strain:** set one minute of physiological values and see the prediction, the baseline's answer
  and the explanation.
- **Analyse a recording:** upload an AIDLAB export (or load an official example) and get per-minute
  predictions.
- **Model results:** explore all evaluation results interactively.
""")
st.markdown(f"Full code, report and reproducible pipeline: **[GitHub repository]({REPO_URL})**")
disclaimer()
