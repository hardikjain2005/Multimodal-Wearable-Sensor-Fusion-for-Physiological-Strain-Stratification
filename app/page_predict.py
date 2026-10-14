"""Page 1 — interactive prediction from user-entered physiological values."""
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.common import (CLASS_COLORS, CLASS_DOMAIN, CLASS_RANGE, CORE, LABELS, MODEL_NAMES, STRAIN_ORDER,
                        dev_sample, disclaimer, explain, predict_proba, reference)

ref = reference()
PRESET_NAMES = {"Custom": "✏️ My own values", "Low": "😴 Typical sleep minute (Low)",
                "Moderate": "🧘 Typical yoga minute (Moderate)",
                "High": "🥾 Typical hike / squash minute (High)"}

# Inputs the locked models actually use (6 further HR statistics are removed by the model's
# redundancy filter and are derived from mean and SD below).
GROUPS = {
    "❤️ Heart rate": [("hr_mean", 0.5), ("hr_std", 0.1), ("hr_slope", 0.1)],
    "〰️ Heart-rate variability": [("hrv_rmssd", 0.5), ("hrv_sdnn", 0.5), ("hrv_pnn50", 0.5),
                                                  ("hrv_cv_rr", 0.001)],
    "📈 ECG quality & movement": [("ecg_kurtosis", 0.1), ("ecg_hf_noise_ratio", 0.001),
                                        ("ecg_lf_wander_ratio", 0.005), ("ecg_ramp_cv", 0.005),
                                        ("ecg_beat_artifact_frac", 0.005)],
}
HELP = {
    "hr_mean": "Average beats per minute over the 60-second window.",
    "hr_std": "How much the heart rate moved up and down within the minute.",
    "hr_slope": "Is HR rising (positive) or falling (negative) during the minute?",
    "hrv_rmssd": "Beat-to-beat variability (short-term, parasympathetic). Usually higher at rest — but movement "
                 "artifacts can inflate it during sport.",
    "hrv_sdnn": "Overall variability of the time between heartbeats.",
    "hrv_pnn50": "% of successive beats differing by more than 50 ms.",
    "hrv_cv_rr": "SD of RR intervals divided by their mean.",
    "ecg_kurtosis": "Clean, still ECG is 'spiky' (high). Movement and noise lower it.",
    "ecg_hf_noise_ratio": "Muscle/movement noise (30–100 Hz, mains excluded) relative to heartbeat energy.",
    "ecg_lf_wander_ratio": "Slow baseline drift from body movement and breathing, relative to heartbeat energy.",
    "ecg_ramp_cv": "How much the height of the R-peaks varies (posture, breathing, movement).",
    "ecg_beat_artifact_frac": "Share of detected beats rejected as implausible (motion artifacts).",
}


def snap(x: float, step: float, how=round) -> float:
    """Put a value exactly on the slider's step grid (otherwise the slider snaps it itself and
    reports that as a user change)."""
    return round(how(x / step) * step, 6)


def apply_preset():
    if st.session_state.preset == "Custom":
        return  # keep whatever the user has set
    preset = ref["presets"][st.session_state.preset]
    for feats in GROUPS.values():
        for f, step in feats:
            st.session_state[f] = snap(float(preset[f]), step)


st.title("🎛️ Predict physiological strain")
st.markdown("Enter **one minute** of chest-strap measurements using the sliders. The model predicts the "
            "activity-derived strain level, and the result below updates as soon as you change a value.")

if "preset" not in st.session_state:
    st.session_state.preset = "Moderate"
    apply_preset()

def mark_custom():
    """Any slider change means the inputs are no longer a typical example minute."""
    st.session_state.preset = "Custom"


st.radio("Load example values (optional)", list(PRESET_NAMES), format_func=PRESET_NAMES.get, key="preset",
         horizontal=True, on_change=apply_preset)
st.caption("An example fills all 12 sliders with the typical values of a real minute from the training data. "
           "Move any slider to enter your own values; the selection then switches to *My own values*.")

values = {}
cols = st.columns(3, gap="large")
for col, (group, feats) in zip(cols, GROUPS.items()):
    with col:
        st.subheader(group, divider="gray")
        for f, step in feats:
            # slider spans the 1st-99th training percentile, padded by 30 % on each side
            r = ref["range"][f]
            pad = 0.3 * (r["p99"] - r["p01"])
            lo, hi = r["p01"] - pad, r["p99"] + pad
            if r["min"] >= 0:
                lo = max(0.0, lo)
            lo, hi = snap(lo, step, math.floor), snap(hi, step, math.ceil)
            fmt = "%.3f" if step < 0.01 else ("%.2f" if step < 0.5 else "%.1f")
            values[f] = st.slider(LABELS[f], float(lo), float(hi), step=float(step), key=f, format=fmt,
                                  on_change=mark_custom,
                                  help=HELP[f])

# Derived HR statistics (not used by the models after their redundancy filter; kept for completeness)
m, sd = values["hr_mean"], values["hr_std"]
values.update(hr_median=m, hr_min=m - 2 * sd, hr_max=m + 2 * sd, hr_range=4 * sd,
              hr_p10=m - 1.28 * sd, hr_p90=m + 1.28 * sd)
x = pd.DataFrame([values])[CORE]

proba = predict_proba(x)
pred = STRAIN_ORDER[int(proba["proposed"][0].argmax())]
base_pred = STRAIN_ORDER[int(proba["baseline"][0].argmax())]
conf = proba["proposed"][0].max()

st.markdown("---")
left, right = st.columns([1, 1.4], gap="large")
with left:
    st.markdown(f"#### Predicted strain level")
    st.markdown(
        f"<div style='padding:18px 22px;border-radius:12px;background:{CLASS_COLORS[pred]}1A;"
        f"border:2px solid {CLASS_COLORS[pred]}'><div style='font-size:clamp(1.5rem, 3.2vw, 2.6rem);font-weight:700;white-space:nowrap;"
        f"color:{CLASS_COLORS[pred]};line-height:1.1'>{pred.upper()}</div>"
        f"<div style='font-size:1rem;margin-top:4px'>probability {conf:.0%} · proposed multimodal model</div></div>",
        unsafe_allow_html=True)
    st.markdown("")
    if base_pred == pred:
        st.info(f"The heart-rate-only baseline agrees: **{base_pred}**.", icon="🤝")
    else:
        st.warning(f"The heart-rate-only baseline would say **{base_pred}**. The other signals change the answer.",
                   icon="🔀")
    outside = [LABELS[f] for f in CORE if not (ref["range"][f]["p01"] <= values[f] <= ref["range"][f]["p99"])
               and f in dict(sum(GROUPS.values(), []))]
    if outside:
        st.error("Outside the range seen in training (1st–99th percentile): " + "; ".join(outside) +
                 ". Treat this prediction with extra caution.", icon="🧭")

with right:
    st.markdown("#### Class probabilities")
    for key in ["proposed", "baseline"]:
        pdf = pd.DataFrame({"level": STRAIN_ORDER, "probability": proba[key][0]})
        base = alt.Chart(pdf).encode(
            y=alt.Y("level:N", sort=STRAIN_ORDER, title=None, axis=alt.Axis(labelFontSize=13, labelOverlap=False)),
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None))
        bars = base.mark_bar(cornerRadiusEnd=4, height=20).encode(
            color=alt.Color("level:N", scale=alt.Scale(domain=CLASS_DOMAIN, range=CLASS_RANGE), legend=None),
            tooltip=["level", alt.Tooltip("probability:Q", format=".1%")])
        text = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("probability:Q", format=".0%"))
        st.caption(MODEL_NAMES[key])
        st.altair_chart((bars + text).properties(height=110), use_container_width=True)

st.markdown("---")
e1, e2 = st.columns([1.2, 1], gap="large")
with e1:
    st.markdown(f"#### Why **{pred}**? Contribution of each input")
    with st.spinner("Explaining this prediction (SHAP)…"):
        contrib = explain(x)[pred].reindex([f for g in GROUPS.values() for f, _ in g])
    cdf = pd.DataFrame({"input": [LABELS[f] for f in contrib.index], "contribution": contrib.values})
    cdf["direction"] = np.where(cdf.contribution >= 0, f"pushes towards {pred}", f"pushes away from {pred}")
    bars = (alt.Chart(cdf).mark_bar(cornerRadiusEnd=3)
            .encode(x=alt.X("contribution:Q", title=f"Change in P({pred})", axis=alt.Axis(format="+.0%")),
                    y=alt.Y("input:N", sort=alt.EncodingSortField("contribution", op="max", order="descending"),
                            title=None, axis=alt.Axis(labelLimit=320, labelFontSize=12, labelOverlap=False)),
                    color=alt.Color("direction:N",
                                    scale=alt.Scale(domain=[f"pushes towards {pred}", f"pushes away from {pred}"],
                                                    range=[CLASS_COLORS[pred], "#9b9a94"]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["input", alt.Tooltip("contribution:Q", format="+.1%")])
            .properties(height=12 * 30))
    st.altair_chart(bars, use_container_width=True)
    st.caption("SHAP values relative to an average development window. They describe what the model relies on, "
               "not what physiologically causes strain.")
with e2:
    st.markdown("#### Where your input sits")
    xf = st.selectbox("Horizontal axis", [f for g in GROUPS.values() for f, _ in g], index=0,
                      format_func=LABELS.get)
    yf = st.selectbox("Vertical axis", [f for g in GROUPS.values() for f, _ in g], index=8,
                      format_func=LABELS.get)
    ds = dev_sample()
    pts = (alt.Chart(ds).mark_circle(size=18, opacity=0.45)
           .encode(x=alt.X(f"{xf}:Q", title=LABELS[xf], scale=alt.Scale(zero=False)),
                   y=alt.Y(f"{yf}:Q", title=LABELS[yf], scale=alt.Scale(zero=False)),
                   color=alt.Color("strain_label:N", scale=alt.Scale(domain=CLASS_DOMAIN, range=CLASS_RANGE),
                                   legend=alt.Legend(orient="bottom", title="Training minutes"))))
    you = (alt.Chart(pd.DataFrame([{xf: values[xf], yf: values[yf]}]))
           .mark_point(shape="diamond", size=260, filled=True, color="#0b0b0b", stroke="white", strokeWidth=1.5)
           .encode(x=f"{xf}:Q", y=f"{yf}:Q", tooltip=[alt.Tooltip(f"{xf}:Q", format=".3g"),
                                                       alt.Tooltip(f"{yf}:Q", format=".3g")]))
    st.altair_chart((pts + you).properties(height=330), use_container_width=True)
    st.caption("◆ = your input · dots = 600 one-minute windows from the development recordings.")

disclaimer()
