"""Streamlit dashboard entry point.

    streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Wearable Strain Dashboard", page_icon="🫀", layout="wide")

from app.common import sidebar_footer  # noqa: E402  (after set_page_config)

pages = [
    st.Page("app/page_predict.py", title="Predict strain", icon="🎛️", default=True),
    st.Page("app/page_recording.py", title="Analyse a recording", icon="📼"),
    st.Page("app/page_results.py", title="Model results", icon="📊"),
    st.Page("app/page_about.py", title="About & limitations", icon="ℹ️"),
]
with st.sidebar:
    st.markdown("### 🫀 Wearable Strain Dashboard")
    st.caption("Multimodal ECG/HRV fusion for activity-derived physiological strain")
nav = st.navigation(pages)
sidebar_footer()
nav.run()
