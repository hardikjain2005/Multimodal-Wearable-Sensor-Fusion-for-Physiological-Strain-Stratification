# Presentation Outline (15 slides) — with speaker notes

**Title:** Multimodal Wearable Sensor Fusion for Physiological Strain Stratification:
A Hybrid HRV-Based Risk Monitoring Approach.
All numbers come from `results/metrics/`. Suggested figures are in `results/figures/`.

---

## Slide 1 — Title
- Project title and subtitle; name, roll number, course, instructor (Dr. Prerna), date
- One-line disclaimer: *activity-derived strain categories, not a medical diagnosis*

**Speaker notes:** "Good morning. My project asks whether a chest-strap wearable can tell how hard the
body is working better if we combine several signals instead of using heart rate alone. I want to be
clear from the start that this is a monitoring and analytics project, not a diagnostic tool."

## Slide 2 — Healthcare Problem
- Strain = the body's overall load during an activity (training, work, rehabilitation)
- Most wearables summarise strain mainly from heart rate (HR)
- HR lags, is affected by temperature, posture and emotion, and overlaps between activities

**Speaker notes:** "Heart rate is useful but blunt. Yoga and a gentle hike can give the same heart rate,
even though the body is doing very different things. If we only look at HR, we confuse them."

## Slide 3 — Motivation
- Chest straps record ECG at 250 Hz: heartbeat timing (HRV), signal quality and movement traces
- HRV reflects nervous-system balance; ECG noise reflects muscle and movement
- Many published wearable studies report inflated accuracy because of data leakage between training and test

**Speaker notes:** "The ECG contains much more than heart rate. The second motivation is honesty in
evaluation: if windows from the same recording appear in both training and testing, the model looks far
better than it really is. I designed the project to avoid that."

## Slide 4 — Dataset
- AIDLAB wearable datasets (professor-approved): 18 sessions of sleep, yoga, hike and squash
- 250 Hz ECG, RR intervals, HR, respiration, skin temperature, accelerometer
- Labels: sleep → Low, yoga → Moderate, hike/squash → High (activity used as a proxy for strain)
- Figure: `01_sensor_availability.png`

**Speaker notes:** "The four originally approved sessions gave only one recording per class, so I used
all 18 sessions of these activities from the catalog. Inspection showed that the sensors present differ
by activity. For example, every hike lacks respiration. If I had filled in the missing values, the model
could simply learn 'no breathing signal means hike'. So the main model only uses signals that exist in
every session. I also checked AIDLAB-HAR v3: it has only motion data and exercise labels, so it could
not test a physiological model."

## Slide 5 — Objectives
1. Leakage-safe preprocessing and time alignment
2. HR, HRV and ECG-quality feature engineering, validated against the device
3. HR-only baseline and multiple comparison models
4. Ablation to prove what each component adds
5. One-time testing on held-out sessions and on a different person
6. Explainability (SHAP) and session-level statistics

**Speaker notes:** "These six objectives map directly onto the professor's required workflow."

## Slide 6 — Methodology
- No resampling: each signal keeps its own rate; all are cut on the same 60-s windows (3,192 windows)
- Splits at session level, fixed before training: development (11 sessions), temporal test (3),
  cross-person test (4)
- Nested, session-grouped cross-validation; primary metric macro-F1
- Analysis plan written before training (`docs/analysis_plan.md`)

**Speaker notes:** "The key idea is grouping by session. Neighbouring minutes of the same recording are
almost identical, so they must never be split between training and testing. Hyperparameters were tuned
in an inner loop that also respects sessions."

## Slide 7 — Feature Engineering
- **HR (9):** mean, median, min, max, SD, range, percentiles, slope
- **HRV (4):** SDNN, RMSSD, pNN50, CV-RR, from our own R-peak detector (matches the device, r ≥ 0.94 at rest)
- **ECG quality/motion (5):** kurtosis, high-frequency noise, baseline wander, R-amplitude variability,
  beat-artifact fraction
- LF/HF not used: 60-s windows are too short and exercise signals are non-stationary

**Speaker notes:** "I wrote my own heartbeat detector because one person's squash files had no RR file.
I removed mean RR from the HRV group because it is just 60000 divided by HR, a duplicate of the baseline."

## Slide 8 — Baseline Model
- Logistic regression on 9 HR features (interpretable, class-weighted)
- Development CV macro-F1 **0.663**; temporal test **0.628**
- Calls 44 % of yoga minutes Low and 46 % of hike minutes Moderate

**Speaker notes:** "The baseline recognises sleep well but fails in the middle: yoga and hiking overlap
in heart rate."

## Slide 9 — Proposed Novel Approach
- Multi-domain fusion (HR + HRV + ECG quality/motion) → Random Forest + Gradient Boosting **soft-voting** ensemble
- Ensemble type chosen by development CV *before* testing (soft voting 0.802 > stacking 0.769)
- Novelty = combination of fusion, HRV engineering, ensemble, leakage-safe validation and independent testing,
  each verified by ablation

**Speaker notes:** "I don't claim that nobody has ever used an ensemble. In the literature I reviewed,
I did not find these components combined in one experimental framework on this kind of data."

## Slide 10 — Model Comparison
- Table: HR only vs HR+HRV vs HR+HRV+ECG across 6 models (`table_cv_comparison.csv`)
- HR features: every model 0.648–0.671; multimodal: 0.759–0.802 (Decision Tree 0.618 is the exception)
- Figure: `09_cv_model_comparison_f1.png`

**Speaker notes:** "Giving HR to fancier models does not help. Giving models more information does. That
tells us the gain comes from the features, not the algorithm."

## Slide 11 — Results
- Development CV: 0.663 → **0.802** macro-F1; ROC-AUC 0.874 → 0.960
- Temporal test: 0.628 → **0.901**; Moderate recall 0.42 → 0.82; High recall 0.45 → 0.84
- Cross-person (Low vs High): 0.943 → **0.975**
- Ablation: −ECG → 0.714, −HRV → 0.767, −HR → 0.807
- Figures: `10_ablation_ladder.png`, `cm_test_test_temporal.png`

**Speaker notes:** "On sessions recorded later, which the model never saw, the macro-F1 rose from 0.63
to 0.90. Honestly, the ensemble itself added little over a single gradient boosting model (0.800 vs
0.802). The ECG features made the difference."

## Slide 12 — Explainable AI
- SHAP on the whole ensemble: ECG features 47 %, HR 42 %, HRV 12 % of mean |SHAP|
- Mean HR drives High; ECG high-frequency noise drives Moderate
- Figure: `shap_summary.png` (or `shap_group_share.png` for a simpler view)

**Speaker notes:** "SHAP tells us which inputs the model leans on. It does not prove a physiological
cause. The yoga signal is probably muscle activity and posture showing up as ECG noise."

## Slide 13 — Healthcare Insights
- HR rises with strain (66 → 96 → 116 bpm, session medians)
- HRV did **not** fall with strain here: motion artifacts inflate RMSSD during squash
- Breathing is faster and less regular with activity (17.5 → 22.5 br/min)
- Temperature mainly reflects bedding and environment
- Figures: `15_why_hr_alone_fails.png`, `17_test_session_predictions.png`

**Speaker notes:** "One insight surprised me. Textbooks say HRV drops during exercise, but in this data
it didn't, because chest movement corrupts beat timing. That is an important warning for anyone using
wearable HRV during sport."

## Slide 14 — Limitations
- Activity is a proxy; no clinically validated strain labels; not a diagnosis
- Essentially one participant (plus one other person, squash only)
- Sensors differ between sessions, so respiration, temperature and motion stay outside the main model
- Motion artifacts; inter-person variability; small number of sessions (fold SD ≈ 0.1)
- AIDLAB-HAR v3 incompatible

**Speaker notes:** "These limitations mean the results are promising but preliminary."

## Slide 15 — Conclusion & Future Scope
- Fusing ECG-derived domains clearly beats HR alone for activity-derived strain, on unseen sessions
- Gain comes from features; ensemble adds robustness, not a large boost
- Future: multi-person data with all sensors, validated strain labels (RPE, lactate), motion-robust HRV,
  personalised and real-time edge models, deep/transformer models when data allows

**Speaker notes:** "To conclude: more signals from the same chest strap tell a much clearer story than
heart rate alone, as long as we evaluate honestly. Thank you, I'm happy to take questions."
