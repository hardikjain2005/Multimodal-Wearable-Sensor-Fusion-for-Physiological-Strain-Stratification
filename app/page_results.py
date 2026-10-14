"""Page 3 — interactive view of the project's evaluation results (read from results/metrics)."""
import altair as alt
import pandas as pd
import streamlit as st

from app.common import CLASS_COLORS, CLASS_DOMAIN, CLASS_RANGE, STRAIN_ORDER, disclaimer, figure, metrics

st.title("📊 Model results")
st.markdown("Every number on this page is read from the files that `main.py` produced. Evaluation is grouped "
            "by recording session, and the held-out tests were run once with locked models.")

test = metrics("test_results.csv")
cv = metrics("table_cv_comparison.csv")
PROPOSED, CORE_FS = "Ensemble RF+GB (soft voting)", "HR + HRV + ECG (multimodal core)"


def pick(df, model, fs, **kw):
    q = df[(df.model == model) & (df.feature_set == fs)]
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0]


st.subheader("Headline: proposed model vs heart-rate-only baseline")
k = st.columns(4)
for col, (label, df, extra) in zip(k, [
        ("Dev. cross-validation macro-F1", cv, {"experiment": "main"}),
        ("Later-sessions test macro-F1", test, {"test_set": "test_temporal"}),
        ("Different-person test macro-F1", test, {"test_set": "test_cross_person"}),
        ("Moderate recall (later sessions)", test, {"test_set": "test_temporal"})]):
    metric = "sensitivity_Moderate" if "Moderate" in label else "f1_macro"
    p = pick(df, PROPOSED, CORE_FS, **extra)[metric]
    b = pick(df, "Logistic Regression", "HR only", **extra)[metric]
    col.metric(label, f"{p:.3f}", f"{p - b:+.3f} vs baseline ({b:.3f})")

st.markdown("---")
st.subheader("Compare models and feature sets")
METRICS = {"f1_macro": "Macro-F1", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
           "roc_auc_ovr_macro": "Macro ROC-AUC", "recall_macro": "Macro recall", "precision_macro": "Macro precision",
           "sensitivity_Low": "Recall: Low", "sensitivity_Moderate": "Recall: Moderate",
           "sensitivity_High": "Recall: High"}
c1, c2 = st.columns([1, 2])
with c1:
    source = st.radio("Evaluation", ["Development cross-validation", "Held-out: later sessions",
                                     "Held-out: different person"])
    metric = st.selectbox("Metric", list(METRICS), format_func=METRICS.get)
    if source != "Development cross-validation" and metric == "sensitivity_Moderate" and "different" in source:
        st.caption("The different-person test has no Moderate sessions.")
if source == "Development cross-validation":
    d = cv[cv.experiment == "main"]
else:
    d = test[test.test_set == ("test_temporal" if "later" in source else "test_cross_person")]
d = d.assign(fs=d.feature_set.str.replace(" (multimodal core)", "", regex=False))
with c2:
    chart = (alt.Chart(d).mark_bar(cornerRadiusEnd=3)
             .encode(x=alt.X(f"{metric}:Q", title=METRICS[metric], scale=alt.Scale(domain=[0, 1])),
                     y=alt.Y("model:N", title=None, sort=list(dict.fromkeys(d.model)), axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                     color=alt.Color("fs:N", title="Features", sort=["HR only", "HR + HRV", "HR + HRV + ECG"],
                                     scale=alt.Scale(domain=["HR only", "HR + HRV", "HR + HRV + ECG"], range=["#9b9a94", "#86b6ef", "#1c5cab"]),
                                     legend=alt.Legend(orient="bottom")),
                     yOffset=alt.YOffset("fs:N", sort=["HR only", "HR + HRV", "HR + HRV + ECG"]),
                     tooltip=["model", "fs", alt.Tooltip(f"{metric}:Q", format=".3f")])
             .properties(height=380))
    st.altair_chart(chart, use_container_width=True)
st.caption("Pattern: giving heart rate to stronger algorithms barely helps; adding HRV and ECG information does.")

st.markdown("---")
a1, a2 = st.columns(2, gap="large")
with a1:
    st.subheader("Ablation: what each feature group adds")
    ab = metrics("ablation_ladder.csv")
    ab["kind"] = ["baseline"] + ["add"] * 3 + ["remove"] * 3
    ch = (alt.Chart(ab).mark_bar(cornerRadiusEnd=3)
          .encode(x=alt.X("f1_macro:Q", title="Dev. CV macro-F1", scale=alt.Scale(domain=[0, 1])),
                  y=alt.Y("feature_set:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
                  color=alt.Color("kind:N", scale=alt.Scale(domain=["baseline", "add", "remove"],
                                                            range=["#52514e", "#2a78d6", "#eb6834"]),
                                  legend=alt.Legend(orient="bottom", title=None)),
                  tooltip=["feature_set", alt.Tooltip("f1_macro:Q", format=".3f"),
                           alt.Tooltip("f1_macro_fold_std:Q", format=".3f", title="SD across folds")])
          .properties(height=alt.Step(34)))
    st.altair_chart(ch, use_container_width=True)
    st.caption("Removing ECG features costs the most; removing HR costs nothing (HRV/ECG features carry it).")
with a2:
    st.subheader("Explainability (SHAP)")
    g = metrics("shap_group_share.csv").melt(id_vars="group", var_name="target", value_name="share")
    ch2 = (alt.Chart(g).mark_bar()
           .encode(x=alt.X("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of mean |SHAP|"),
                   y=alt.Y("target:N", sort=["Low", "Moderate", "High", "overall"], title=None),
                   color=alt.Color("group:N", scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a"]),
                                   legend=alt.Legend(orient="bottom", title="Feature group")),
                   tooltip=["target", "group", alt.Tooltip("share:Q", format=".0%")])
           .properties(height=260))
    st.altair_chart(ch2, use_container_width=True)
    st.caption("Mean HR drives High; ECG noise features drive Moderate. Importance ≠ physiological cause.")

st.markdown("---")
st.subheader("Held-out test sessions")
sess = metrics("test_per_session_predictions.csv")
which = st.radio("Model", ["Proposed (multimodal ensemble)", "Baseline (HR-only logistic regression)"],
                 horizontal=True)
s = sess[(sess.model == PROPOSED) & (sess.feature_set == CORE_FS)] if which.startswith("Proposed") else \
    sess[(sess.model == "Logistic Regression") & (sess.feature_set == "HR only")]
long = s.melt(id_vars=["session_id", "true_label", "test_set"], value_vars=[f"pred_{c}_pct" for c in STRAIN_ORDER],
              var_name="pred", value_name="pct")
long["pred"] = long.pred.str.replace("pred_", "").str.replace("_pct", "")
long["session"] = long.session_id + "  (true: " + long.true_label + ")"
ch3 = (alt.Chart(long).mark_bar()
       .encode(x=alt.X("pct:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of minutes"),
               y=alt.Y("session:N", title=None, axis=alt.Axis(labelLimit=400, labelOverlap=False)),
               color=alt.Color("pred:N", scale=alt.Scale(domain=CLASS_DOMAIN, range=CLASS_RANGE), sort=STRAIN_ORDER,
                               legend=alt.Legend(orient="bottom", title="Predicted")),
               order=alt.Order("pred_order:Q"),
               tooltip=["session_id", "test_set", "pred", alt.Tooltip("pct:Q", format=".1f", title="%")])
       .transform_calculate(pred_order="indexof(['Low','Moderate','High'], datum.pred)")
       .properties(height=alt.Step(34)))
st.altair_chart(ch3, use_container_width=True)

with st.expander("More figures (confusion matrices, ROC, SHAP beeswarm, session profiles)"):
    for f, cap in [("cm_test_test_temporal.png", "Confusion matrices — later-sessions test"),
                   ("roc_test_test_temporal.png", "ROC curves — later-sessions test"),
                   ("shap_summary.png", "SHAP beeswarm per class"),
                   ("16_session_profiles.png", "Session-level physiological profiles"),
                   ("15_why_hr_alone_fails.png", "Why heart rate alone fails")]:
        st.image(figure(f), caption=cap, use_container_width=True)
with st.expander("Full tables"):
    st.markdown("**Development cross-validation (all experiments)**")
    st.dataframe(cv, hide_index=True, use_container_width=True)
    st.markdown("**Held-out tests (all models)**")
    st.dataframe(test.round(3), hide_index=True, use_container_width=True)

disclaimer()
