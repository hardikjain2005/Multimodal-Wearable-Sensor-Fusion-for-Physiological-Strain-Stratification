"""Page 2 — analyse a whole recording: per-minute strain predictions."""
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from app.common import CLASS_COLORS, CLASS_DOMAIN, CLASS_RANGE, MODEL_NAMES, STRAIN_ORDER, disclaimer, models
from src.inference import fetch_example, load_recording, predict_recording, safe_unzip

BASE = "https://aidlab-production-datasets.s3.eu-central-1.amazonaws.com/"
EXAMPLES = {
    "Yoga, 62 min (development set → in-sample)": ("datasets_2026-04-30_yoga_62min.zip", "Moderate"),
    "Yoga, 73 min (held-out later-session test)": ("datasets_2026-05-05_yoga_73min.zip", "Moderate"),
    "Hike, 82 min (held-out later-session test)": ("datasets_2026-06-07_hike_1h22m.zip", "High"),
    "Squash, 60 min, second person (held-out cross-person test)": ("datasets_2026-04-27_squash_60min.zip", "High"),
    "Squash, 54 min (development set → in-sample)": ("datasets_2026-04-27_squash_54min.zip", "High"),
    "Sleep, 4 h 40 min (development set → in-sample, 45 MB)": ("datasets_2026-04-21_sleep_4h40m.zip", "Low"),
}


@st.cache_data(show_spinner=False, max_entries=4)
def example_bytes(file: str) -> bytes:
    return fetch_example(BASE + file)


@st.cache_data(show_spinner=False, max_entries=6)
def analyse_zip(data: bytes, name: str, max_minutes: int) -> tuple[pd.DataFrame, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        folder = safe_unzip(data, Path(tmp))
        session, notes = load_recording(folder, name, max_minutes)
        return predict_recording(session, models()), notes


def analyse_files(files, max_minutes: int) -> tuple[pd.DataFrame, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            (Path(tmp) / Path(f.name).name).write_bytes(f.getvalue())
        session, notes = load_recording(Path(tmp), "uploaded files", max_minutes)
        return predict_recording(session, models()), notes


st.title("📼 Analyse a recording")
st.markdown("Run the locked models over a whole chest-strap recording. It is cut into 60-second windows, "
            "and every usable minute gets a strain prediction.")

src = st.radio("Input", ["Example from the AIDLAB catalogue", "Upload my own recording"], horizontal=True)
max_minutes = st.slider("Analyse at most the first … minutes", 5, 420, 120, step=5,
                        help="Long recordings take longer to process; 2 hours is ~10 seconds.")

result, notes, true_label, name = None, [], None, None
if src.startswith("Example"):
    choice = st.selectbox("Example session", list(EXAMPLES))
    file, true_label = EXAMPLES[choice]
    st.caption(f"Downloaded on demand from the official AIDLAB source: `{BASE}{file}`. "
               "Sessions marked *in-sample* were used to train the model, so their predictions are optimistic.")
    if st.button("▶ Download and analyse", type="primary"):
        try:
            with st.spinner("Downloading from AIDLAB… (10–45 MB, cached afterwards)"):
                data = example_bytes(file)
            with st.spinner("Detecting heartbeats, extracting features and predicting…"):
                result, notes = analyse_zip(data, choice, max_minutes)
            name = choice
        except Exception as e:  # network or format problems are shown to the user
            st.error(f"Could not analyse this example: {e}")
else:
    st.markdown("Upload an **AIDLAB session export**: either the original **.zip**, or the CSV files "
                "(`ecg.csv` required; `pulse.csv` and `dataset.json` recommended). CSVs need the columns "
                "`timestamp` (Unix ms) and `value`; the ECG must be sampled at 250 Hz.")
    up = st.file_uploader("Recording", type=["zip", "csv", "json"], accept_multiple_files=True)
    if up and st.button("▶ Analyse upload", type="primary"):
        try:
            with st.spinner("Detecting heartbeats, extracting features and predicting…"):
                zips = [f for f in up if f.name.lower().endswith(".zip")]
                if zips:
                    result, notes = analyse_zip(zips[0].getvalue(), zips[0].name, max_minutes)
                    name = zips[0].name
                else:
                    result, notes = analyse_files(up, max_minutes)
                    name = "uploaded files"
        except Exception as e:
            st.error(f"Could not analyse the upload: {e}")

if result is not None:
    st.session_state["recording_result"] = (result, notes, true_label, name)
elif "recording_result" in st.session_state:
    result, notes, true_label, name = st.session_state["recording_result"]

if result is not None:
    for n in notes:
        st.info(n, icon="ℹ️")
    u = result[result.usable]
    st.markdown(f"### Results — {name}")
    if u.empty:
        st.error("No minute passed quality control (ECG coverage ≥ 90 % and ≥ 20 valid beats).")
    else:
        model_key = st.segmented_control("Model", ["proposed", "baseline"], default="proposed",
                                         format_func=lambda k: MODEL_NAMES[k].split(" (")[0]) or "proposed"
        share = u[f"{model_key}_pred"].value_counts(normalize=True).reindex(STRAIN_ORDER).fillna(0)
        c = st.columns(5)
        c[0].metric("Minutes analysed", len(result))
        c[1].metric("Usable minutes", len(u), help="Windows passing the same quality control used in training")
        for col, lvl in zip(c[2:], STRAIN_ORDER):
            col.metric(f"{lvl} strain", f"{share[lvl]:.0%}")
        if true_label:
            acc = (u[f"{model_key}_pred"] == true_label).mean()
            st.success(f"Activity label of this session: **{true_label}**. "
                       f"{acc:.0%} of usable minutes were predicted as {true_label}.", icon="🎯")

        long = u.melt(id_vars=["minute"], value_vars=[f"{model_key}_p_{c}" for c in STRAIN_ORDER],
                      var_name="level", value_name="probability")
        long["level"] = long.level.str.replace(f"{model_key}_p_", "")
        area = (alt.Chart(long).mark_area(opacity=0.9)
                .encode(x=alt.X("minute:Q", title="Minute of recording"),
                        y=alt.Y("probability:Q", stack="normalize", axis=alt.Axis(format="%"), title="Probability"),
                        color=alt.Color("level:N", scale=alt.Scale(domain=CLASS_DOMAIN, range=CLASS_RANGE),
                                        sort=STRAIN_ORDER, legend=alt.Legend(orient="top", title=None)),
                        order=alt.Order("level_order:Q"),
                        tooltip=["minute", "level", alt.Tooltip("probability:Q", format=".0%")])
                .transform_calculate(level_order="indexof(['Low','Moderate','High'], datum.level)")
                .properties(height=220, title="Strain probability per minute"))
        hr = (alt.Chart(result).mark_line(color="#e34948", strokeWidth=1.5)
              .encode(x=alt.X("minute:Q", title="Minute of recording"),
                      y=alt.Y("hr_mean:Q", title="Mean HR (bpm)", scale=alt.Scale(zero=False)),
                      tooltip=["minute", alt.Tooltip("hr_mean:Q", format=".0f")])
              .properties(height=150, title="Heart rate"))
        st.altair_chart(alt.vconcat(area, hr).resolve_scale(x="shared"), use_container_width=True)

        table = result[["minute", "time", "usable", "hr_mean", "hrv_rmssd", "ecg_hf_noise_ratio",
                        "proposed_pred", "baseline_pred"] + [f"proposed_p_{c}" for c in STRAIN_ORDER]]
        with st.expander("Per-minute table"):
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("⬇ Download per-minute predictions (CSV)", table.to_csv(index=False).encode(),
                           file_name="strain_predictions.csv", mime="text/csv")

disclaimer()
