// Usage: npm install docx@9.5.1 && node report/build_report.js
// Builds report/Wearable_Strain_Stratification_Report.docx from the project's result files.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
  TableOfContents,
} = require("docx");

const P = path.resolve(__dirname, "..");
const FIG = path.join(P, "results/figures");
const OUT = path.join(P, "report/Wearable_Strain_Stratification_Report.docx");

const SIZES = {
  "01_sensor_availability.png": [1782, 1199], "15_why_hr_alone_fails.png": [2177, 740],
  "16_session_profiles.png": [3619, 675], "09_cv_model_comparison_f1.png": [1693, 889],
  "10_ablation_ladder.png": [2129, 751], "cm_test_test_temporal.png": [2069, 833],
  "17_test_session_predictions.png": [2709, 1072], "shap_summary.png": [3307, 1186],
  "shap_group_share.png": [1299, 772], "13_cv_confusion.png": [2069, 833],
  "02_session_windows.png": [2416, 1183],
};

const FONT = "Arial";
const W = 9026; // A4 content width in DXA with 1" margins
let figNo = 0, tabNo = 0;

// ---- inline markup: **bold**, *italic*
function runs(text, base = {}) {
  base = { font: FONT, ...base };
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const p = (t, o = {}) => new Paragraph({ children: runs(t), spacing: { after: 90, line: 250 },
  alignment: AlignmentType.JUSTIFIED, ...o });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t),
  spacing: { after: 40, line: 252 } });
let numInstance = 1;
const num = (t) => new Paragraph({ numbering: { reference: "numbers", level: 0, instance: numInstance }, children: runs(t),
  spacing: { after: 40, line: 252 } });

function figure(file, caption, widthIn = 6.2) {
  const [w, h] = SIZES[file];
  const wpx = Math.round(widthIn * 96), hpx = Math.round(wpx * h / w);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(FIG, file)),
        transformation: { width: wpx, height: hpx },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: runs(`**Figure ${figNo}.** ${caption}`, { size: 18, color: "52514E" }) }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, caption, opts = {}) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (cells, isHead, shade) => new TableRow({ tableHeader: isHead, children: cells.map((c, i) =>
    new TableCell({ borders, width: { size: widths[i], type: WidthType.DXA },
      shading: isHead ? { fill: "DCE8F7", type: ShadingType.CLEAR, color: "auto" }
        : shade ? { fill: "FFF2E5", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(String(c), { size: 17, bold: isHead || undefined }) })] })) });
  return [
    new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
      children: runs(`**Table ${tabNo}.** ${caption}`, { size: 18, color: "52514E" }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [mk(header, true), ...rows.map((r, i) => mk(r, false, (opts.highlight || []).includes(i)))] }),
    new Paragraph({ spacing: { after: 100 }, children: [] }),
  ];
}

// =========================================================================== CONTENT
const title = [
  new Paragraph({ spacing: { before: 2200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ font: FONT, text: "Multimodal Wearable Sensor Fusion for Physiological Strain Stratification",
      bold: true, size: 40, color: "1C5CAB" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 600 },
    children: [new TextRun({ font: FONT, text: "A Hybrid HRV-Based Risk Monitoring Approach", size: 30, color: "52514E" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ font: FONT, text: "Healthcare Analytics Mini Project — Project Report", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 900 }, children: [new TextRun({ font: FONT, text: "Submitted by", size: 24, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ font: FONT, text: "Hardik Jain (PRN: 23070521056)", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ font: FONT, text: "Nikhil Halambi (PRN: 23070521099)", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300 }, children: [new TextRun({ font: FONT, text: "Course Instructor: Dr. Prerna", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300 }, children: [new TextRun({ font: FONT, text: "October 2026", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1400 },
    children: [new TextRun({ font: FONT, text: "This project estimates activity-derived physiological strain categories from wearable biosignals. It is not a clinical diagnostic system.",
      italics: true, size: 18, color: "52514E" })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstract = [
  h1("Abstract"),
  p("**Problem.** Wearable chest straps record heart rate (HR), the electrocardiogram (ECG) and other signals continuously, but most consumer strain or effort scores rely mainly on HR, which responds slowly and overlaps between activities of different intensity. **Data.** We used 18 recordings from the professor-approved AIDLAB wearable dataset catalog (sleep, yoga, hiking, squash; 250 Hz ECG, beat-to-beat RR intervals, HR, respiration, skin temperature). Activity type served as a proxy label for strain (sleep → Low, yoga → Moderate, hike/squash → High) because no clinical strain labels exist. **Method.** Signals were aligned on a common clock without resampling and cut into 3,192 non-overlapping 60-second windows. We detected heartbeats in the raw ECG, computed heart-rate-variability (HRV) and ECG signal-quality/motion features, and compared an HR-only logistic-regression baseline with five other classifiers and two Random Forest + Gradient Boosting ensembles. All evaluation was grouped by recording session to prevent leakage, with nested hyper-parameter tuning and two held-out test sets fixed before training: later recordings of the same person and recordings of a different or unknown person. **Novel approach.** The framework combines multi-domain feature fusion, HRV-focused feature engineering, a hybrid ensemble and independent-session testing, and quantifies each component through ablation. **Results.** Macro-F1 rose from 0.663 (HR-only baseline) to 0.802 (proposed soft-voting ensemble) in session-grouped cross-validation, from 0.628 to 0.901 on the temporal held-out test, and from 0.943 to 0.975 on the cross-person test (Low vs High only). Moderate-strain recall on the temporal test improved from 0.42 to 0.82. Ablation and SHAP showed that the gain came mainly from ECG-derived signal-quality/motion features and HRV, not from the ensemble itself: a single Gradient Boosting model reached 0.800. **Conclusion.** Fusing several ECG-derived feature domains clearly outperformed HR alone for separating activity-derived strain levels. The evidence comes mostly from one participant, and the labels are activity proxies, so the findings are preliminary and not clinical."),
  p("**Keywords:** wearable sensors, heart rate variability, ECG, sensor fusion, ensemble learning, physiological strain, explainable AI, leakage-safe validation."),
  new Paragraph({ children: [new PageBreak()] }),
];

const body = [
  h1("1. Introduction"),
  p("Wearable health devices such as chest straps, smart watches and patches now record physiological signals continuously outside the hospital. A chest strap like the AIDLAB device records the **electrocardiogram (ECG)**, the electrical activity of the heart, at 250 samples per second, together with breathing, skin temperature and body movement. From the ECG we can measure **heart rate (HR)**, the number of beats per minute, and **heart-rate variability (HRV)**, the small beat-to-beat changes in the time between heartbeats (the RR interval). HRV reflects the balance between the 'rest-and-digest' (parasympathetic) and 'fight-or-flight' (sympathetic) branches of the nervous system: at rest it is usually larger, and during physical or mental stress it usually shrinks [1, 2]."),
  p("**Physiological strain** is the overall load the body experiences in response to an activity. Monitoring it matters for training load management, occupational safety (for example heat or physical workload), rehabilitation and early detection of excessive effort. Heart rate is the most common strain indicator, but it has limitations. It lags behind changes in effort, it is affected by temperature, posture and emotion, and different activities can produce similar heart rates. Using several sensors together (**multimodal sensing**) and letting **machine learning (ML)** learn how to combine them can address these limits. ML models learn patterns from labelled examples; ensemble models combine several learners to make more robust predictions [3–5]."),
  p("This project tests whether such a multimodal, HRV-focused ensemble approach classifies activity-derived strain better than HR alone, using a careful evaluation design that avoids the over-optimistic results common in wearable-sensor studies."),

  h1("2. Problem Statement"),
  p("To develop and rigorously evaluate a machine-learning framework that classifies **activity-derived physiological strain levels** (Low, Moderate, High) from multimodal wearable biosignals, and to determine whether fusing ECG-derived feature domains (HR, HRV, ECG signal-quality/motion) with a hybrid ensemble classifier improves classification compared with a heart-rate-only baseline, when tested on recording sessions never seen during training."),

  h1("3. Objectives"),
  num("Inspect the AIDLAB datasets and build a leakage-safe preprocessing pipeline: loading, de-duplication, quality control and time alignment of streams with different sampling rates."),
  num("Engineer window-level features for HR, HRV (from our own ECG R-peak detection), ECG signal quality/motion, respiration and skin temperature, and validate them against device-provided values."),
  num("Build an HR-only baseline and compare it with Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, soft-voting and stacking ensembles using session-grouped, nested cross-validation."),
  num("Demonstrate the contribution of each component through ablation experiments: HR → HR+HRV → multimodal → ensemble, and leave-one-group-out."),
  num("Evaluate the locked models once on two independent held-out test sets: later sessions of the same person, and a different or unknown person."),
  num("Explain the proposed model with SHAP and support findings with session-level statistical tests."),
  num("Derive careful healthcare insights and document limitations of proxy labels and dataset size."),

  h1("4. Literature Review / Related Work"),
  p("**HRV standards and physiology.** The ESC/NASPE Task Force [1] defined time-domain HRV measures (SDNN, RMSSD, pNN50) and frequency-domain measures (LF, HF, LF/HF). It recommended at least ~2 minutes of data for LF power and 5-minute stationary recordings for short-term frequency analysis. Shaffer and Ginsberg [2] summarised norms and noted that ultra-short (<5 min) time-domain metrics such as RMSSD can be valid, whereas LF/HF is not. Exercise reviews [6, 7] report that HRV falls as exercise intensity rises, while warning that motion artifacts and non-stationarity complicate measurement during activity."),
  p("**ECG processing and signal quality.** Pan and Tompkins [8] introduced the classic real-time QRS (heartbeat) detector that our R-peak detector simplifies. Clifford et al. [9] showed that ECG signal-quality indices such as kurtosis and spectral power ratios can flag noise and motion. We reuse such indices as features that partly encode movement. Charlton et al. [10] reviewed breathing-rate estimation from ECG and PPG and found that such estimates degrade during movement."),
  p("**Multimodal wearable datasets and ML.** WESAD [11] combined chest- and wrist-worn ECG, respiration, temperature, accelerometer and electrodermal activity to classify stress with Random Forest and other classifiers. PPG-DaLiA [12] used daily-life activities to estimate HR with deep learning. Czekaj et al. [13] used the AIDLAB chest strap for real-time activity recognition from motion data. This is the source of the AIDLAB-HAR dataset examined in this project."),
  p("**Ensembles, validation and explainability.** Random Forests [3], gradient boosting [4] and stacked generalization [5] are standard strong learners for tabular physiological features. Saeb et al. [14] demonstrated that record-wise (random) cross-validation grossly overestimates accuracy when samples from the same subject or recording appear in both training and test data. Varma and Simon [15] showed that tuning hyper-parameters on the same folds used for evaluation is optimistically biased, and recommended nested CV. SHAP [16] provides model-agnostic feature attributions."),
  p("**Research gap.** Existing studies commonly investigate individual wearable signals, HRV-based physiological monitoring, multimodal sensing, or ML-based physiological classification. Based on the literature reviewed for this project, we did not identify a study that combines multimodal feature fusion, HRV-focused feature engineering, hybrid ensemble classification, session-grouped leakage-safe validation and independent held-out-session testing within the same experimental framework on the AIDLAB data. This project addresses that combination."),

  h1("5. Dataset Description"),
  p("**Source.** AIDLAB wearable datasets (https://aidlab.com/datasets), the professor-approved resource. Each session is a ZIP archive containing a `dataset.json` file (user metadata, units, start/end time) and one CSV per signal with a `timestamp` column (Unix epoch in milliseconds, UTC) and value column(s). Two metadata schemas exist: ISO `from`/`to`, and `from_timestamp_ms`/`to_timestamp_ms`. The loader handles both."),
  p("**Sessions.** The four originally approved sessions (one each of sleep, yoga, squash and hike) gave only one recording per class, which makes session-level validation impossible. With the author's confirmation that the approval covers the AIDLAB catalog, all 18 sleep, yoga, squash and hike sessions were used. Inspection revealed two participants. All sleep, yoga and hike sessions and two squash sessions come from a female participant (30–40 y, 170–180 cm). Three squash sessions come from a male participant (30–40 y, 190–200 cm), recorded at the same time as two of the female squash sessions, which suggests opponents in the same matches. One sleep session has no participant metadata."),
  p("**Measured sampling rates.** ECG 250 Hz (median sample spacing 4 ms in every session); respiration impedance 50 Hz; accelerometer ~50–59 Hz (in g, although metadata says m/s²); HR and RR are event-based (1–2 reports/s); skin temperature 0.1 Hz. No cell was missing (NaN). Missing data appears as **time gaps** instead, for example 379 s of ECG contact loss in one yoga session. Duplicate timestamps were rare (≤ 40 in any stream) and were removed."),
  ...figure("01_sensor_availability.png", "Sensor availability per session. Only ECG and pulse exist in every recording. Respiration is absent in every hike, and the accelerometer is present in only five sessions, so modality availability is confounded with the activity label.", 4.0),
  ...table(["Split (fixed before training)", "Sessions", "Low", "Moderate", "High", "Purpose"], [
    ["Development", "11 (female: 3 sleep, 3 yoga, 3 hike, 2 squash)", "1044", "218", "874", "Training, tuning, CV"],
    ["Test — temporal", "3 (last sleep, yoga, hike of the same person)", "349", "62", "74", "Generalisation over time"],
    ["Test — cross-person", "4 (3 male squash + 1 unknown-subject sleep)", "390", "–", "181", "Generalisation to other people"],
  ], [1700, 2650, 700, 900, 700, 2376], "Data splits and number of 60-s windows after quality control (3,192 windows in total; 40 windows, 1.2 %, removed by quality control)."),
  p("**Independent dataset (AIDLAB-HAR v3).** Inspection of all 180 EDF files showed exactly seven channels: 3-axis acceleration plus orientation quaternions at 50 Hz. There was **no ECG, HR, HRV, respiration or temperature**. The labels are 16 exercise types (squats, burpees, plank, …) rather than strain levels, and most recordings last 12–120 s. Because it shares no physiological signal with our model, using it as a test set would be invalid. Held-out sessions from the AIDLAB catalog were used as the independent tests instead."),

  h1("6. Data Preprocessing"),
  p("**Time alignment without resampling.** The streams have very different rates, so we did not resample them to one frequency. Up-sampling the 0.1 Hz temperature would invent data, and down-sampling the ECG would destroy the beat timing that HRV relies on. Every stream instead keeps its native rate, and all streams share one clock. A window is the interval [t₀ + 60k, t₀ + 60(k+1)) seconds applied to each stream's timestamps, and features are computed per stream inside it."),
  p("**Windowing.** Sessions were cut into **non-overlapping 60-s windows**. Sixty seconds contains roughly 60–160 heartbeats and 6–40 breaths, enough for stable time-domain HRV and breathing features. Non-overlapping windows avoid sharing samples between neighbours. The more important protection against leakage is that every split is by **session**, never by window."),
  p("**Quality control (label-blind, identical for all sessions).** A window was kept only if ECG coverage was ≥ 90 % and all core features were computable (≥ 10 HR samples and ≥ 20 valid beats). Forty windows (1.2 %) failed."),
  p("**Labels.** Labels are activity-derived proxies: sleep → Low, yoga → Moderate, squash and hike → High. Hike HR (median 96–107 bpm) lies closer to yoga (86–97) than to squash (143–145), so 'Hike = High' is debatable. It was kept as the author's chosen design and is discussed as a limitation. Labels were deliberately **not** defined by HR thresholds, because that would make the HR-only baseline circular."),
  p("**Missing modalities, imbalance and scaling.** Missing modalities were **never imputed**. Respiration, for example, is absent in every hike, so filling it in would let a model learn 'no respiration signal = hike = High'. The development set is imbalanced (Moderate = 10 % of windows), which was handled with class weights; SMOTE was tested as an alternative (Section 10). Standardisation (logistic regression only) and a correlation filter were fitted inside each training fold."),

  h1("7. Proposed Methodology"),
  p("The workflow is: inspection → alignment and windowing → feature extraction → quality control → session-level split → baseline → comparison models and ensembles → ablation → held-out testing → SHAP and statistics → insights. The analysis plan, including the model-selection rule, was written down before any model was trained (docs/analysis_plan.md)."),
  h2("7.1 Feature engineering"),
  bullet("**HR (baseline, 9 features)** from the device pulse stream: mean, median, minimum, maximum, standard deviation, range, 10th/90th percentiles and slope (bpm/min)."),
  bullet("**HRV (4 model features).** R-peaks were detected in the raw ECG with a simplified Pan-Tompkins detector [8]: 5–20 Hz band-pass, derivative, squaring, 150 ms integration, adaptive threshold and 250 ms refractory period. RR intervals were cleaned at session level: values outside 300–2000 ms, or more than 20 % from the local 5-beat median, were marked as artifacts. From the cleaned beats we computed SDNN, RMSSD, pNN50 and CV-RR. Mean and median RR were computed but excluded from modelling, because they equal 60000/HR (Spearman ρ = −0.997) and would give the 'HR+HRV' model a second copy of heart rate. **LF/HF was not computed**: 60-s windows are shorter than the 2–5 min the Task Force requires [1], and exercise signals are non-stationary."),
  bullet("**ECG signal quality / motion (5 features):** kurtosis of the filtered ECG, high-frequency noise ratio (power 30–45 + 55–100 Hz relative to 1–30 Hz; the 50 Hz mains band is excluded so indoor/outdoor electrical environment cannot act as a hidden label), baseline-wander ratio (0–1 Hz / 1–30 Hz), R-peak amplitude variability and beat-artifact fraction [9]. These capture muscle activity, movement and breathing-related baseline shifts."),
  bullet("**Add-on modalities** (only where recorded): respiration rate by peak counting, breath-interval variability, amplitude and spectral purity; skin-temperature mean, min, max, SD, change and slope. Accelerometer features were computed but used descriptively only, since they exist in just five sessions."),
  p("**Validation of own signal processing.** ECG-derived HR differed from the device HR by a median 0.4–0.9 bpm per window in every activity. RMSSD from our R-peaks agreed with RMSSD from the device RR stream at r = 0.96 (sleep) and 0.94 (yoga), but only 0.69 (hike) and 0.53 (squash): motion degrades beat timing during vigorous activity. A spectral breathing rate and an ECG-derived respiration estimate agreed poorly with the device breathing rate during activity (r ≤ 0.24) and were discarded. Peak counting agreed best (r = 0.72 in yoga, 0.46 in squash)."),
  h2("7.2 Models and validation"),
  p("**Baseline:** HR-only logistic regression (standardised, class-weighted). It is simple, interpretable and a fair reference. The same HR features were also given to every other model, so feature effects and algorithm effects can be separated. **Comparison models:** Decision Tree, Random Forest (300 trees), Gradient Boosting (histogram-based) and two hybrid ensembles combining Random Forest and Gradient Boosting: **soft voting** (averaging the two models' class probabilities) and **stacking** (a logistic-regression meta-learner trained on session-grouped out-of-fold base predictions)."),
  p("**Why grouped CV?** Neighbouring 60-s windows from the same recording are nearly identical. With ordinary random K-fold, windows of one session fall into both training and test folds, so the model is tested on data it has effectively seen and accuracy is inflated [14]. We therefore assigned whole **sessions** to three outer folds, each holding out one sleep, one yoga and one or two High sessions. Hyper-parameters were tuned inside each outer training part with a second, session-grouped 2-fold CV (**nested CV** [15]); the selection metric was macro-F1. The proposed model was defined in advance as the ensemble with the higher development-CV macro-F1."),

  h1("8. Novelty / Innovation of the Proposed Approach"),
  p("**What was the baseline?** A logistic regression on 9 heart-rate statistics, representing the HR-centred view most consumer strain scores rely on."),
  p("**What was changed?** (1) **Multi-domain feature fusion:** HR, HRV and ECG signal-quality/motion features are combined, with respiration and temperature tested where available. (2) **HRV-focused engineering:** our own validated R-peak detection, session-level artifact cleaning, exclusion of HR-duplicating RR means, and no unreliable LF/HF. (3) **A hybrid RF + GB ensemble.** (4) **Independent testing** on held-out later sessions and on a different person. (5) **Ablation** that measures each component instead of assuming it helps."),
  p("**Why is it better than a single standard algorithm?** The results show honestly where the improvement comes from. The large gain comes from the **features**: on HR features alone all six algorithms score 0.648–0.671 CV macro-F1, whereas on multimodal features they reach 0.759–0.802 (except the single decision tree). The ensemble's own contribution is small in CV (soft voting 0.802 vs 0.800 for a single Gradient Boosting model) but slightly larger on the temporal test (0.901 vs 0.891). Its main practical value is robustness: it was never the worst model on any feature set or test set. The innovation is therefore the **framework**, meaning leakage-safe, ablation-verified fusion of ECG-derived domains, rather than the ensemble algorithm alone."),

  h1("9. Implementation"),
  p("The project is written in Python 3.13.9 with pandas 3.0.6, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, imbalanced-learn 0.14.2, SHAP 0.53.0, matplotlib 3.11.2, seaborn 0.13.2 and pyEDFlib 0.1.42. The code is modular (`src/`: data_loader, preprocessing, feature_engineering, build_dataset, splits, statistics, baseline_model, ensemble_model, evaluation, visualization). One command, `python main.py --stage all`, reproduces every number and figure in about 7 minutes. The random seed is 42; two independent runs produced byte-identical result files. Raw data is git-ignored and downloaded from the official source by a script."),
  ...table(["Model", "Parameters searched (inner session-grouped CV)", "Selected on full development set (core features)"], [
    ["Logistic Regression", "C ∈ {0.01, 0.1, 1, 10}", "C = 0.1"],
    ["Decision Tree", "max_depth ∈ {3,5,8}; min_samples_leaf ∈ {10,50}", "depth 8, leaf 10"],
    ["Random Forest (300 trees)", "max_depth ∈ {None,8}; min_samples_leaf ∈ {5,20}; max_features ∈ {sqrt,0.5}", "None, 5, sqrt"],
    ["Gradient Boosting (300 iterations)", "learning_rate ∈ {0.05,0.1}; max_leaf_nodes ∈ {7,15}; min_samples_leaf ∈ {20,50}", "0.05, 7, 50"],
    ["Ensembles", "RF and GB tuned separately, then combined", "as RF and GB above"],
  ], [2100, 4226, 2700], "Hyper-parameter search spaces (kept deliberately small to avoid over-fitting 11 sessions)."),

  h1("10. Experimental Results"),
  p("Table 3 gives pooled out-of-fold development-CV results. Macro averages weight Low, Moderate and High equally, so the small Moderate class is not ignored."),
  ...table(["Model", "Feature set", "Acc.", "Prec.", "Recall", "F1", "F1 fold SD", "ROC-AUC"], [
    ["Logistic Regression (baseline)", "HR only", "0.742", "0.693", "0.699", "0.663", "0.062", "0.874"],
    ["Random Forest", "HR only", "0.775", "0.680", "0.687", "0.671", "0.069", "0.889"],
    ["Ensemble soft voting", "HR only", "0.776", "0.670", "0.676", "0.664", "0.076", "0.890"],
    ["Logistic Regression", "HR + HRV", "0.760", "0.720", "0.750", "0.692", "0.013", "0.928"],
    ["Gradient Boosting", "HR + HRV", "0.807", "0.714", "0.735", "0.710", "0.065", "0.905"],
    ["Ensemble soft voting", "HR + HRV", "0.806", "0.718", "0.743", "0.714", "0.068", "0.922"],
    ["Logistic Regression", "HR + HRV + ECG", "0.832", "0.760", "0.825", "0.766", "0.053", "0.936"],
    ["Decision Tree", "HR + HRV + ECG", "0.677", "0.691", "0.677", "0.618", "0.111", "0.867"],
    ["Random Forest", "HR + HRV + ECG", "0.826", "0.749", "0.802", "0.759", "0.112", "0.956"],
    ["Gradient Boosting", "HR + HRV + ECG", "0.848", "0.789", "0.822", "0.800", "0.109", "0.951"],
    ["**Ensemble soft voting (proposed)**", "HR + HRV + ECG", "0.850", "0.791", "0.824", "**0.802**", "0.109", "**0.960**"],
    ["Ensemble stacking", "HR + HRV + ECG", "0.830", "0.756", "0.814", "0.769", "0.091", "0.919"],
  ], [2500, 1500, 700, 700, 800, 700, 1000, 1126], "Development cross-validation (3 session-grouped outer folds, nested tuning; macro-averaged metrics). Full table for all 18 model × feature-set combinations: results/metrics/table_cv_comparison.csv.", { highlight: [10] }),
  p("**Ablation (novelty experiment).** Figure 2 shows the additive ladder and leave-one-group-out ablation of the proposed ensemble. HR → HR+HRV added +0.050 macro-F1; adding ECG features added +0.088. Removing ECG features from the full model dropped it to 0.714, and removing HRV dropped it to 0.767. Removing HR did *not* reduce performance (0.807): the HRV and ECG features already carry the heart-rate information, since RR-based features and ECG morphology change with HR."),
  ...figure("10_ablation_ladder.png", "Ablation of the proposed ensemble (development CV). Error bars = standard deviation across three session-grouped folds; with only three folds, differences below ~0.05 should not be over-interpreted.", 5.4),
  p("**Class imbalance.** With class weights the proposed model reached 0.802 macro-F1, without correction 0.799, and with SMOTE (applied to training folds only) 0.787. SMOTE raised Moderate recall (0.743 → 0.780) but lowered overall F1, so class weights were kept as pre-specified. **Respiration add-on** (sleep, yoga and squash sessions only; 2-fold grouped CV; fixed default hyper-parameters because inner grouped tuning was impossible with so few sessions): core features 0.819 → core + respiration 0.826, a negligible gain given the fold SD of about 0.12."),

  h1("11. Performance Evaluation"),
  p("**Metrics in simple words.** *Accuracy* is the share of windows classified correctly, which can look good even if a small class is ignored. *Precision* for a class asks: when the model says High, how often is it right? *Recall* (= *sensitivity*) asks: of all truly High windows, how many did it catch? *Specificity* asks: of all windows that are not High, how many were correctly not called High? *F1* balances precision and recall. *ROC-AUC* measures how well the predicted probabilities rank a class above the others across all thresholds (0.5 = chance, 1 = perfect). In monitoring, a missed high-strain period (low recall) is usually worse than a false alarm, so we emphasise recall and macro-F1."),
  p("**Held-out tests.** The locked models were tuned on all development sessions and evaluated exactly once on each test set (Table 4). Nothing was changed after seeing these results."),
  ...table(["Test set", "Model (features)", "Acc.", "Macro-F1", "AUC", "Sens. Low", "Sens. Mod.", "Sens. High", "Spec. High"], [
    ["Temporal (3 classes)", "Baseline LR (HR)", "0.812", "0.628", "0.921", "0.960", "0.419", "0.446", "0.978"],
    ["Temporal (3 classes)", "Gradient Boosting (core)", "0.946", "0.891", "0.986", "0.997", "0.806", "0.824", "0.993"],
    ["Temporal (3 classes)", "**Proposed soft voting (core)**", "0.951", "**0.901**", "0.991", "0.997", "**0.823**", "**0.838**", "0.995"],
    ["Temporal (3 classes)", "Stacking (core)", "0.953", "0.907", "0.990", "0.997", "0.839", "0.838", "0.998"],
    ["Cross-person (Low/High)", "Baseline LR (HR)", "0.926", "0.943", "0.999", "0.969", "–", "0.834", "1.000"],
    ["Cross-person (Low/High)", "**Proposed soft voting (core)**", "0.975", "**0.975**", "1.000", "1.000", "–", "**0.923**", "1.000"],
  ], [1550, 2150, 650, 850, 650, 750, 800, 800, 826], "Held-out test results (window level). Cross-person macro metrics cover Low and High only; predicting 'Moderate' for these windows counts as an error.", { highlight: [2, 5] }),
  ...figure("cm_test_test_temporal.png", "Confusion matrices on the temporal held-out test (row-normalised; counts in brackets). HR alone calls 44 % of yoga windows Low and 46 % of hike windows Moderate; the multimodal ensemble fixes most of these errors.", 5.0),
  p("The stacking ensemble scored marginally higher on the temporal test (0.907) than the pre-selected soft-voting model (0.901). Following the analysis plan, we did not switch models after seeing test data; the difference is about one window in accuracy terms (0.953 vs 0.951 of 485 windows)."),

  h1("12. Data Visualization and Healthcare Insights"),
  h2("12.1 Visual and statistical analysis"),
  ...figure("15_why_hr_alone_fails.png", "Why HR alone fails. Left: the Moderate (yoga) HR distribution overlaps both Low and High. Right: at the same heart rate, yoga windows show a higher ECG high-frequency noise ratio than hike or squash windows.", 5.6),
  ...figure("16_session_profiles.png", "Session-level physiological profile (one dot = one development recording). HR rises consistently with strain level, whereas RMSSD does not fall.", 6.6),
  p("**Session-level statistics.** Windows from the same session are not independent, so treating thousands of windows as separate observations would make almost any difference 'significant' (pseudo-replication). We therefore compared session medians with the Kruskal-Wallis test. Its null hypothesis (H₀) is that the distribution is the same for all strain levels; the alternative (H₁) is that at least one level differs. It was chosen because it does not assume normality, which matters with 3–5 sessions per group. The tests were run on development sessions only."),
  ...table(["Feature", "Median Low", "Median Mod.", "Median High", "H", "p", "ε²"], [
    ["Mean HR (bpm)", "65.7", "95.6", "116.0", "8.73", "0.013", "0.84"],
    ["RMSSD (ms)", "19.3", "13.8", "28.5", "3.08", "0.215", "0.13"],
    ["SDNN (ms)", "33.2", "35.6", "28.9", "3.93", "0.140", "0.24"],
    ["CV of RR", "0.036", "0.052", "0.058", "6.01", "0.050", "0.50"],
    ["ECG kurtosis", "11.0", "7.4", "6.3", "6.98", "0.031", "0.62"],
    ["ECG HF-noise ratio", "0.012", "0.020", "0.013", "6.01", "0.050", "0.50"],
    ["ECG baseline-wander ratio", "0.027", "0.313", "0.223", "6.30", "0.043", "0.54"],
    ["Respiration rate (br/min)*", "17.5", "21.2", "22.5", "6.25", "0.044", "0.85"],
    ["Breath-interval CV*", "0.21", "0.39", "0.42", "6.25", "0.044", "0.85"],
  ], [2600, 1000, 1000, 1000, 800, 800, 1826], "Kruskal-Wallis test on session medians (development set, n = 11 sessions; *n = 8 sessions with respiration, High = squash only). ε² = effect size (0 = none, 1 = perfect separation)."),
  p("Pairwise Mann-Whitney post-hoc tests with Holm correction were **not significant** for any pair (smallest adjusted p = 0.104), because with 3–5 sessions per group the smallest attainable p-values are large. Statistical significance is also not clinical significance: these tests describe one or two people, not a population."),
  h2("12.2 Explainable AI (SHAP)"),
  p("SHAP [16] assigns each feature a contribution (positive or negative) to the predicted probability of each class, for every window. We used a model-agnostic permutation explainer on the whole soft-voting ensemble, so all contributions are on the same probability scale. It was run on 450 development windows (150 per class)."),
  ...figure("shap_summary.png", "SHAP beeswarm plots of the proposed model. Each dot is one window; colour = feature value (red high, blue low); position = effect on the class probability. Six HR statistics removed by the correlation filter (redundant with mean HR) have zero contribution.", 6.6),
  p("**Which features mattered?** Mean HR was the top feature overall and dominated the High-vs-Low decision (high HR raises P(High) by up to ~0.45). For **Moderate**, the most influential feature was the ECG high-frequency noise ratio (high values raise P(Moderate)), followed by HR, R-amplitude variability, kurtosis and baseline wander. By feature group (Figure 7), ECG features accounted for 47 % of mean |SHAP| overall (63 % for Moderate), HR for 42 % (53 % for High) and HRV for 12 %. HRV therefore contributes, mostly via pNN50 and RMSSD for the High class, but it is not the dominant signal. Respiration, temperature and motion were not part of the main model. **SHAP shows what the model uses, not what causes strain**: a feature's importance does not prove a physiological or clinical mechanism."),
  ...figure("shap_group_share.png", "Share of mean |SHAP| per feature group for each class and overall.", 3.7),

  h2("12.3 Insights supported by the data"),
  bullet("**HR increases consistently with strain level** (session medians 66 → 96 → 116 bpm) and is the single strongest feature, but yoga overlaps both sleep and hiking (Figure 4)."),
  bullet("**HRV did not decrease with strain here.** RMSSD was highest in the High class. Female squash sessions show RMSSD of 32–34 ms at ~145 bpm, while the male squash sessions show 7–9 ms. Combined with the poor agreement with device RR during squash (r = 0.53), this points to motion artifacts inflating beat-to-beat variability during vigorous activity. The textbook HRV decrease [6, 7] could not be confirmed from these data."),
  bullet("**Breathing became faster and less regular with activity** (17.5 → 21.2 → 22.5 br/min; breath-interval CV 0.21 → 0.39 → 0.42), consistent with physiology. However, impedance respiration is dominated by motion artifacts during squash, and adding respiration did not measurably improve classification."),
  bullet("**ECG signal-quality / motion features add the most information beyond HR.** Clean, spiky ECG (high kurtosis) indicates rest. More baseline wander and high-frequency (muscle) noise indicate movement. These features partly measure *movement and recording conditions* rather than cardiac physiology."),
  bullet("**Skin temperature** was 35.9–36.5 °C during sleep and 32.3–34.4 °C otherwise. It improved Moderate recall in a secondary test (Section 13), but it likely reflects bedding and environment, not strain."),
  bullet("**Multimodal fusion generalised** to later sessions and to the second person (Figure 8), but the cross-person test covers only sleep versus squash, the easiest pair."),
  ...figure("17_test_session_predictions.png", "Held-out test sessions: share of windows assigned to each strain level by the baseline (left) and the proposed model (right).", 5.8),

  h1("13. Comparison with Existing / Baseline Approach"),
  p("Against the HR-only baseline, the proposed approach improved development-CV macro-F1 by +0.139 (0.663 → 0.802), temporal-test macro-F1 by +0.273 (0.628 → 0.901) and cross-person macro-F1 by +0.032 (0.943 → 0.975). The clearest healthcare-relevant change is **recall of Moderate strain** on the temporal test (0.42 → 0.82) and of **High strain** (0.45 → 0.84). The baseline mostly detected rest. Giving the same HR features to stronger algorithms (RF, GB, ensembles) did **not** close the gap (temporal macro-F1 0.622–0.635), which confirms that the improvement comes from information beyond heart rate rather than from model complexity."),
  p("**Temperature add-on (pre-declared secondary analysis).** Only one development yoga session records temperature, so this experiment was evaluated on the temporal test instead. The proposed model was trained on the seven development sessions with temperature, using the locked hyper-parameters. Adding temperature raised macro-F1 from 0.840 to 0.924 and Moderate recall from 0.57 to 0.87. Because Moderate training data came from one session, and temperature reflects environment as much as physiology, this result is suggestive only."),

  h1("14. Discussion"),
  p("**Why the model performed well.** Activity-derived strain levels differ not only in heart rate but also in how the body moves and how the ECG looks. During sleep the ECG is clean and stable; during yoga, muscle tension and posture changes add high-frequency noise at moderate HR; during hiking and squash, HR is high and baseline wander increases. Combining these domains resolves the yoga/hike HR overlap that defeats the baseline."),
  p("**Did the ensemble help?** Only slightly. The ensemble was the best or near-best model in every evaluation, but a single Gradient Boosting model was within 0.01 macro-F1. Stacking was worse in CV (0.769), probably because its meta-learner had very few independent sessions to learn from. Honest reporting of this is part of the contribution."),
  p("**Generalisation.** Performance on the temporal test (0.901) was as good as in CV, so the model did not over-fit the development sessions of this person. On the cross-person test the High-class recall rose from 0.83 to 0.92 despite the male participant's very different HRV, but this test contains only the easiest contrast (sleep vs squash)."),
  p("**Sources of error and limitations.**"),
  ...(() => { numInstance = 2; return []; })(),
  num("**Proxy labels:** activity is not a measured strain level. A calm hike stretch or an intense yoga flow is labelled by activity, not by effort, and there are no clinically validated strain labels."),
  num("**Not medical diagnosis:** the model must not be used to assess patient risk."),
  num("**Small and narrow sample:** development and the temporal test come from one female participant; only one other person (squash only) and one unknown-subject sleep were available. Fold-to-fold SD of about 0.1 shows high uncertainty."),
  num("**Sensor availability differs by session,** so respiration, temperature and accelerometer data could not enter the main model; 'multimodal' here mostly means several ECG-derived domains."),
  num("**Sensor noise:** motion artifacts corrupt RR timing and impedance respiration during vigorous activity, which affects HRV interpretation."),
  num("**Inter-person variability:** HR and HRV levels depend on age, fitness and sex; a single-person model may not transfer."),
  num("**Confounding by context:** ECG noise, baseline wander and temperature also reflect posture, bedding, environment and electrode contact, not only physiological strain."),
  num("**Independent dataset incompatibility:** AIDLAB-HAR v3 contains no physiological signals and could not be used."),
  num("**'Hike = High'** is debatable: hike HR is closer to yoga than to squash."),

  h1("15. Conclusion"),
  p("Using 18 AIDLAB recordings and a leakage-safe, session-grouped evaluation, we showed that fusing heart rate with HRV and ECG signal-quality/motion features substantially improves classification of activity-derived physiological strain over heart rate alone. Macro-F1 rose from 0.663 to 0.802 in cross-validation, from 0.628 to 0.901 on later held-out sessions and from 0.943 to 0.975 on a different person. The improvement came mostly from the added feature domains rather than the ensemble, and ECG-derived features were the largest contributors. HRV did not decrease with strain in this dataset, a finding we attribute to motion artifacts. The results support multimodal wearable fusion for strain *monitoring*, but with proxy labels and essentially one participant they are preliminary and not clinical."),

  h1("16. Future Scope"),
  bullet("Larger, multi-person datasets with every sensor recorded in every session, to test true multimodal fusion (respiration, temperature, accelerometer) and person-independent generalisation."),
  bullet("Clinically or physiologically validated strain labels (e.g. ratings of perceived exertion, lactate, VO₂ or core temperature) instead of activity proxies."),
  bullet("Motion-robust HRV (accelerometer-guided artifact rejection) and personalised models calibrated to each user's resting baseline."),
  bullet("Real-time and edge/IoT deployment on the wearable or phone, continuous monitoring with early-warning thresholds, and integration with healthcare or occupational-safety applications."),
  bullet("Deep learning and transformer-based time-series models on raw signals, once enough sessions exist to train them without over-fitting."),
  p("None of these were implemented in this project."),
];

const refs = [
  h1("References"),
  ...[
    "[1] Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. Circulation, 93(5), 1043–1065.",
    "[2] Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in Public Health, 5, 258.",
    "[3] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "[4] Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.",
    "[5] Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.",
    "[6] Michael, S., Graham, K. S., & Davis, G. M. (2017). Cardiac autonomic responses during exercise and post-exercise recovery using heart rate variability and systolic time intervals — a review. Frontiers in Physiology, 8, 301.",
    "[7] Buchheit, M. (2014). Monitoring training status with HR measures: do all roads lead to Rome? Frontiers in Physiology, 5, 73.",
    "[8] Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, BME-32(3), 230–236.",
    "[9] Clifford, G. D., Behar, J., Li, Q., & Rezek, I. (2012). Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms. Physiological Measurement, 33(9), 1419–1433.",
    "[10] Charlton, P. H., Birrenkott, D. A., Bonnici, T., Pimentel, M. A. F., Johnson, A. E. W., Alastruey, J., Tarassenko, L., Watkinson, P. J., Beale, R., & Clifton, D. A. (2018). Breathing rate estimation from the electrocardiogram and photoplethysmogram: a review. IEEE Reviews in Biomedical Engineering, 11, 2–20.",
    "[11] Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI), 400–408.",
    "[12] Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: large-scale heart rate estimation with convolutional neural networks. Sensors, 19(14), 3079.",
    "[13] Czekaj, Ł., et al. (2024). Real-time sensor-based human activity recognition for eFitness and eHealth platforms. Sensors, 24(12), 3891. https://doi.org/10.3390/s24123891",
    "[14] Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. GigaScience, 6(5), 1–9.",
    "[15] Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics, 7, 91.",
    "[16] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30 (NeurIPS 2017).",
    "[17] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.",
    "[18] Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(260), 583–621.",
    "[19] Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.",
    "[20] Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "[21] Aidlab. Aidlab datasets (sleep, yoga, squash and hike sessions; AIDLAB-HAR v3). https://aidlab.com/datasets (accessed 14 October 2026).",
  ].map((t) => new Paragraph({ children: [new TextRun({ text: t, size: 19 })], spacing: { after: 70 },
    indent: { left: 440, hanging: 440 } })),
];

const doc = new Document({
  creator: "Hardik Jain, Nikhil Halambi",
  title: "Multimodal Wearable Sensor Fusion for Physiological Strain Stratification",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1C5CAB" },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "0B0B0B" },
        paragraph: { spacing: { before: 160, after: 70 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "52514E" })] })] }) },
    children: [...title, ...abstract, ...body, new Paragraph({ children: [new PageBreak()] }), ...refs],
  }],
});

fs.mkdirSync(path.dirname(OUT), { recursive: true });
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
