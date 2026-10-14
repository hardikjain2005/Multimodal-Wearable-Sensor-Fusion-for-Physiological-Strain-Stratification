"""Pipeline entry point.

    python main.py --stage data      # build windows, QC, data-understanding figures, validation
    python main.py --stage stats     # feature quality + session-level statistical tests
    python main.py --stage models    # baseline, comparison models, ensemble, ablations (dev CV)
    python main.py --stage test      # final locked models on the two held-out test sets
    python main.py --stage explain   # feature importance + SHAP
    python main.py --stage all
"""
import argparse

import numpy as np
import pandas as pd

from src import feature_engineering as fe
from src.config import METRICS_DIR, PROCESSED_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "windows_clean.csv")


# ----------------------------------------------------------------------------- data
def stage_data():
    from src import build_dataset, visualization as viz
    windows = build_dataset.build()
    clean = build_dataset.quality_control(windows)

    inv = pd.read_csv(METRICS_DIR / "session_inventory.csv")
    viz.plot_sensor_availability(inv, ["ecg", "pulse", "rr", "respiration", "temperature", "acceleration"])
    viz.plot_session_overview(clean)
    viz.plot_class_distribution(clean)

    # Validation of own signal processing against device-provided values
    rows = []
    for act, d in clean.groupby("activity"):
        r = dict(activity=act, windows=len(d))
        hr_ecg = 60000 / d.hrv_mean_rr
        r["hr_ecg_vs_device_median_abs_diff_bpm"] = np.median(np.abs(hr_ecg - d.hr_median))
        v = d.dropna(subset=["val_dev_rmssd"])
        if len(v) > 10:
            r["rmssd_ecg_vs_deviceRR_pearson_r"] = np.corrcoef(v.hrv_rmssd, v.val_dev_rmssd)[0, 1]
            r["rmssd_ecg_vs_deviceRR_median_abs_diff_ms"] = np.median(np.abs(v.hrv_rmssd - v.val_dev_rmssd))
            r["mean_rr_ecg_vs_deviceRR_pearson_r"] = np.corrcoef(v.hrv_mean_rr, v.val_dev_mean_rr)[0, 1]
        rows.append(r)
    pd.DataFrame(rows).round(3).to_csv(METRICS_DIR / "signal_validation.csv", index=False)

    # Dataset dimension summary
    summary = clean.groupby(["split", "strain_label"]).agg(
        sessions=("session_id", "nunique"), windows=("session_id", "size")).reset_index()
    summary.to_csv(METRICS_DIR / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.read_csv(METRICS_DIR / "signal_validation.csv").to_string(index=False))


# ----------------------------------------------------------------------------- stats
def stage_stats():
    from src import statistics as st, visualization as viz
    clean = load_clean()
    dev = clean[clean.split == "development"]
    core = fe.HR_KEYS + fe.HRV_KEYS + fe.ECG_KEYS
    addon = fe.TEMP_KEYS + fe.RESP_KEYS

    # Feature quality (development set): missingness, near-zero variance, outliers
    q = []
    for f in core + addon + fe.ACC_KEYS:
        x = dev[f]
        xs = x.dropna()
        mad = np.median(np.abs(xs - xs.median())) if len(xs) else np.nan
        q.append(dict(feature=f, group=f.split("_")[0], missing_pct=100 * x.isna().mean(),
                      top_value_share=xs.value_counts(normalize=True).iloc[0] if len(xs) else np.nan,
                      n_unique=xs.nunique(),
                      robust_outlier_pct=100 * np.mean(np.abs(xs - xs.median()) / (1.4826 * mad) > 5)
                      if mad and mad > 0 else np.nan))
    pd.DataFrame(q).round(3).to_csv(METRICS_DIR / "feature_quality_dev.csv", index=False)

    corr = viz.plot_correlation_heatmap(dev, core)
    high = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(core) for b in core[i + 1:]
            if abs(corr.loc[a, b]) > 0.95]
    pd.DataFrame(high, columns=["feature_a", "feature_b", "spearman_rho"]).to_csv(
        METRICS_DIR / "highly_correlated_pairs_dev.csv", index=False)

    kw, post = st.session_level_tests(dev, core + addon)
    kw = kw.merge(st.window_level_effect_sizes(dev, core + addon), on="feature")
    kw.round(4).to_csv(METRICS_DIR / "stats_kruskal_session_level.csv", index=False)
    post.round(4).to_csv(METRICS_DIR / "stats_posthoc_session_level.csv", index=False)

    viz.plot_feature_distributions(dev, ["hr_mean", "hr_std", "hr_range", "hr_slope"],
                                   "04_hr_distributions", "Heart rate features by strain level (development)", ncols=4)
    viz.plot_feature_distributions(dev, ["hrv_mean_rr", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_cv_rr"],
                                   "05_hrv_distributions", "HRV features by strain level (development)", ncols=5)
    viz.plot_feature_distributions(dev, fe.ECG_KEYS, "06_ecg_quality_distributions",
                                   "ECG quality / motion-artifact features by strain level (development)", ncols=5)
    viz.plot_feature_distributions(dev, ["temp_mean", "temp_slope", "resp_rate", "resp_bbi_cv", "resp_amp_std",
                                         "resp_spectral_purity"], "07_temp_resp_distributions",
                                   "Add-on modalities: skin temperature & respiration (sessions where recorded)", ncols=3)
    print(kw[["feature", "n_sessions", "H", "p_value", "epsilon_sq", "epsilon_sq_windows"]].round(4).to_string(index=False))
    print("\nHighly correlated pairs (|rho| > 0.95):", high)


# ----------------------------------------------------------------------------- models
FEATURE_SETS = {
    "HR only": fe.HR_KEYS,
    "HR + HRV": fe.HR_KEYS + fe.HRV_MODEL_KEYS,
    "HR + HRV + ECG (multimodal core)": fe.HR_KEYS + fe.HRV_MODEL_KEYS + fe.ECG_KEYS,
}
ABLATIONS = {
    "Core − HR (HRV + ECG)": fe.HRV_MODEL_KEYS + fe.ECG_KEYS,
    "Core − HRV (HR + ECG)": fe.HR_KEYS + fe.ECG_KEYS,
    "Core − ECG (HR + HRV)": fe.HR_KEYS + fe.HRV_MODEL_KEYS,
}
CORE = FEATURE_SETS["HR + HRV + ECG (multimodal core)"]


def _run(spec, df, features, feature_set, experiment, store, n_outer=3, n_inner=2):
    import time
    t = time.time()
    pooled, per_fold, preds, params = __import__("src.evaluation", fromlist=["nested_cv"]).nested_cv(
        spec, df, features, n_outer, n_inner)
    row = dict(experiment=experiment, model=spec.name, feature_set=feature_set,
               imbalance=spec.imbalance, n_features=len(features), n_windows=len(df),
               n_sessions=df.session_id.nunique(), **pooled)
    store["rows"].append(row)
    per_fold.insert(0, "feature_set", feature_set); per_fold.insert(0, "model", spec.name)
    per_fold.insert(0, "experiment", experiment)
    store["folds"].append(per_fold)
    store["params"].append(dict(experiment=experiment, model=spec.name, feature_set=feature_set,
                                fold_params=[str(p) for p in params]))
    preds.insert(0, "feature_set", feature_set); preds.insert(0, "model", spec.name)
    preds.insert(0, "experiment", experiment)
    store["preds"].append(preds)
    print(f"  [{experiment}] {spec.name:32s} | {feature_set:34s} | macro-F1 {pooled['f1_macro']:.3f} "
          f"(folds {pooled['f1_macro_fold_mean']:.3f}±{pooled['f1_macro_fold_std']:.3f}) "
          f"acc {pooled['accuracy']:.3f} AUC {pooled['roc_auc_ovr_macro']:.3f}  [{time.time() - t:.0f}s]")
    return pooled


def stage_models():
    import json
    from src.ensemble_model import ModelSpec, all_specs
    from src.evaluation import session_folds, fold_table
    clean = load_clean()
    dev = clean[clean.split == "development"].reset_index(drop=True)
    store = dict(rows=[], folds=[], params=[], preds=[])
    fold_table(dev, session_folds(dev, 3)).to_csv(METRICS_DIR / "cv_fold_assignment.csv", index=False)

    print("Experiments 1-4: feature sets x models")
    for fs_name, feats in FEATURE_SETS.items():
        for spec in all_specs():
            _run(spec, dev, feats, fs_name, "main", store)

    res = pd.DataFrame(store["rows"])
    core = res[(res.feature_set == "HR + HRV + ECG (multimodal core)") & res.model.str.startswith("Ensemble")]
    proposed_name = core.sort_values("f1_macro", ascending=False).model.iloc[0]
    proposed_kind = "soft_voting" if "soft" in proposed_name else "stacking"
    print(f"\nSelected proposed ensemble (higher dev-CV macro-F1 on core features): {proposed_name}")

    print("Experiment 5: leave-one-group-out ablation of the proposed model")
    for fs_name, feats in ABLATIONS.items():
        _run(ModelSpec(proposed_name, proposed_kind), dev, feats, fs_name, "ablation", store)

    print("Experiment 6: class-imbalance handling (proposed model, core features)")
    for imb in ["none", "smote"]:
        _run(ModelSpec(proposed_name, proposed_kind, imbalance=imb), dev, CORE,
             "HR + HRV + ECG (multimodal core)", "imbalance", store)

    print("Experiment 7: respiration add-on (dev sessions with respiration; 2-fold grouped CV)")
    resp_df = dev.dropna(subset=fe.RESP_KEYS).reset_index(drop=True)
    fold_table(resp_df, session_folds(resp_df, 2)).to_csv(METRICS_DIR / "cv_fold_assignment_resp.csv", index=False)
    from src.baseline_model import baseline_spec
    _run(baseline_spec(), resp_df, fe.HR_KEYS, "HR only", "resp_addon", store, n_outer=2, n_inner=2)
    _run(ModelSpec(proposed_name, proposed_kind), resp_df, CORE, "Core", "resp_addon", store, n_outer=2, n_inner=2)
    _run(ModelSpec(proposed_name, proposed_kind), resp_df, CORE + fe.RESP_KEYS, "Core + RESP", "resp_addon",
         store, n_outer=2, n_inner=2)

    res = pd.DataFrame(store["rows"])
    res.to_csv(METRICS_DIR / "cv_results.csv", index=False)
    pd.concat(store["folds"]).to_csv(METRICS_DIR / "cv_per_fold.csv", index=False)
    pd.concat(store["preds"]).to_csv(METRICS_DIR / "cv_oof_predictions.csv", index=False)
    (METRICS_DIR / "cv_best_params.json").write_text(json.dumps(store["params"], indent=1))
    (METRICS_DIR / "proposed_model.json").write_text(json.dumps({"name": proposed_name, "kind": proposed_kind}))


# ----------------------------------------------------------------------------- test
def _fit_final(spec, dev, features):
    """Tune on the full development set with 3 session-grouped folds and refit."""
    X, y = dev[features].to_numpy(float), dev.strain_label.to_numpy()
    model, params = spec.fit(X, y, dev, n_inner=3)
    return model, params


def stage_test():
    import json, joblib
    from sklearn.base import clone
    from sklearn.model_selection import PredefinedSplit
    from src.config import MODELS_DIR, STRAIN_ORDER
    from src.ensemble_model import ModelSpec, all_specs
    from src.baseline_model import baseline_spec
    from src.evaluation import compute_metrics, align_proba, session_folds
    from src import visualization as viz

    clean = load_clean()
    dev = clean[clean.split == "development"].reset_index(drop=True)
    tests = {s: clean[clean.split == s].reset_index(drop=True) for s in ["test_temporal", "test_cross_person"]}
    proposed = json.loads((METRICS_DIR / "proposed_model.json").read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Locked model list: every model on HR-only and on the multimodal core
    runs = [(spec, fs) for fs in ["HR only", "HR + HRV + ECG (multimodal core)"] for spec in all_specs()]
    rows, session_rows, final_params, preds_store = [], [], {}, {}
    for spec, fs in runs:
        feats = FEATURE_SETS[fs]
        model, params = _fit_final(spec, dev, feats)
        tag = f"{spec.name} | {fs}"
        final_params[tag] = str(params)
        joblib.dump(dict(model=model, features=feats, params=params),
                    MODELS_DIR / (tag.replace(" ", "_").replace("|", "").replace("+", "plus")
                                  .replace("(", "").replace(")", "") + ".joblib"))
        for tname, tdf in tests.items():
            proba = align_proba(model, tdf[feats].to_numpy(float))
            pred = np.array(STRAIN_ORDER)[proba.argmax(1)]
            m = compute_metrics(tdf.strain_label.to_numpy(), pred, proba)
            rows.append(dict(test_set=tname, model=spec.name, feature_set=fs, n_windows=len(tdf), **m))
            preds_store[(tname, spec.name, fs)] = (tdf.strain_label.to_numpy(), pred, proba)
            for sid, g in pd.DataFrame({"sid": tdf.session_id, "y": tdf.strain_label, "p": pred}).groupby("sid"):
                session_rows.append(dict(test_set=tname, model=spec.name, feature_set=fs, session_id=sid,
                                         true_label=g.y.iloc[0], n=len(g),
                                         **{f"pred_{c}_pct": 100 * (g.p == c).mean() for c in STRAIN_ORDER}))
        print(f"  {tag:70s} " + " ".join(
            f"{t}: F1={r['f1_macro']:.3f} acc={r['accuracy']:.3f}" for t, r in
            [(r_["test_set"], r_) for r_ in rows[-2:]]))
    pd.DataFrame(rows).to_csv(METRICS_DIR / "test_results.csv", index=False)
    pd.DataFrame(session_rows).round(1).to_csv(METRICS_DIR / "test_per_session_predictions.csv", index=False)
    (METRICS_DIR / "final_model_params.json").write_text(json.dumps(final_params, indent=1))

    # Confusion matrices + ROC for baseline vs proposed
    core_fs = "HR + HRV + ECG (multimodal core)"
    for tname in tests:
        pairs = {"Baseline: HR-only Logistic Regression": preds_store[(tname, "Logistic Regression", "HR only")],
                 f"Proposed: {proposed['name']}, multimodal core": preds_store[(tname, proposed["name"], core_fs)]}
        viz.plot_confusion_pair(pairs, f"cm_test_{tname}", f"Held-out test — {tname.replace('test_', '')}")
        viz.plot_roc_pair(pairs, f"roc_test_{tname}", f"ROC (one-vs-rest) — held-out {tname.replace('test_', '')}")

    # Pre-declared secondary analysis: temperature add-on on the temporal test set
    print("Experiment 8: temperature add-on (train: dev windows with temperature; test: test_temporal)")
    dev_t = dev.dropna(subset=fe.TEMP_KEYS).reset_index(drop=True)
    test_t = tests["test_temporal"].dropna(subset=fe.TEMP_KEYS).reset_index(drop=True)
    final_prop = joblib.load(next(MODELS_DIR.glob(f"{proposed['name'].replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}*multimodal_core*.joblib")))["model"]
    temp_rows = []
    for fs_name, feats in [("Core", CORE), ("Core + TEMP", CORE + fe.TEMP_KEYS)]:
        m = clone(final_prop)            # same hyper-parameters as the locked proposed model
        if proposed["kind"] == "stacking":
            m.set_params(cv=list(PredefinedSplit(session_folds(dev_t, 2)).split()))
        m.fit(dev_t[feats].to_numpy(float), dev_t.strain_label.to_numpy())
        proba = align_proba(m, test_t[feats].to_numpy(float))
        pred = np.array(STRAIN_ORDER)[proba.argmax(1)]
        r = compute_metrics(test_t.strain_label.to_numpy(), pred, proba)
        temp_rows.append(dict(feature_set=fs_name, train_windows=len(dev_t), train_sessions=dev_t.session_id.nunique(),
                              test_windows=len(test_t), **r))
        print(f"  {fs_name:12s} macro-F1 {r['f1_macro']:.3f} acc {r['accuracy']:.3f}")
    pd.DataFrame(temp_rows).to_csv(METRICS_DIR / "temp_addon_test_temporal.csv", index=False)


# ----------------------------------------------------------------------------- explain
def stage_explain():
    import json, joblib, shap
    from src.config import MODELS_DIR, STRAIN_ORDER, RANDOM_SEED
    from src import visualization as viz
    clean = load_clean()
    dev = clean[clean.split == "development"].reset_index(drop=True)
    proposed = json.loads((METRICS_DIR / "proposed_model.json").read_text())
    path = next(MODELS_DIR.glob(f"{proposed['name'].replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}*multimodal_core*.joblib"))
    bundle = joblib.load(path)
    model, feats = bundle["model"], bundle["features"]

    # Stratified sample of development windows (equal per class) for SHAP
    sample = dev.groupby("strain_label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 150), random_state=RANDOM_SEED))
    background = dev.sample(100, random_state=RANDOM_SEED)[feats]
    explainer = shap.PermutationExplainer(lambda X: model.predict_proba(np.asarray(X, float)),
                                          background, seed=RANDOM_SEED)
    sv = explainer(sample[feats], max_evals=2 * len(feats) + 1, silent=True)   # shape: windows x features x classes
    classes = list(model.classes_)
    np.save(METRICS_DIR / "shap_values.npy", sv.values)
    mean_abs = pd.DataFrame({c: np.abs(sv.values[:, :, i]).mean(0) for i, c in enumerate(classes)}, index=feats)
    mean_abs["overall"] = mean_abs.mean(1)
    mean_abs["group"] = [f.split("_")[0].upper() for f in feats]
    mean_abs.sort_values("overall", ascending=False).round(5).to_csv(METRICS_DIR / "shap_mean_abs.csv")
    grp = mean_abs.groupby("group")[classes + ["overall"]].sum()
    (grp / grp.sum()).round(4).to_csv(METRICS_DIR / "shap_group_share.csv")
    viz.plot_shap_summary(sv, sample, feats, classes, "shap_summary")
    viz.plot_shap_groups(grp, "shap_group_share")
    print(mean_abs.sort_values("overall", ascending=False).round(4).to_string())
    print((grp / grp.sum()).round(3).to_string())


# ----------------------------------------------------------------------------- report figures/tables
def stage_report():
    import json
    from src import visualization as viz
    from src.config import STRAIN_ORDER
    res = pd.read_csv(METRICS_DIR / "cv_results.csv")
    oof = pd.read_csv(METRICS_DIR / "cv_oof_predictions.csv")
    proposed = json.loads((METRICS_DIR / "proposed_model.json").read_text())["name"]
    core_fs = "HR + HRV + ECG (multimodal core)"
    main_res = res[res.experiment == "main"]

    viz.plot_model_comparison(main_res, "09_cv_model_comparison_f1",
                              "Development CV macro-F1: feature sets × models (session-grouped, nested)")
    viz.plot_model_comparison(main_res, "09b_cv_model_comparison_auc",
                              "Development CV macro ROC-AUC (one-vs-rest)", metric="roc_auc_ovr_macro")

    # Ablation ladder for the proposed ensemble
    ens = main_res[main_res.model == proposed][["feature_set", "f1_macro", "f1_macro_fold_std",
                                                "balanced_accuracy", "roc_auc_ovr_macro"]]
    abl = res[res.experiment == "ablation"][ens.columns]
    base = main_res[(main_res.model == "Logistic Regression") & (main_res.feature_set == "HR only")][ens.columns].copy()
    base["feature_set"] = "Baseline: HR only, Logistic Regression"
    ladder = pd.concat([base, ens.assign(feature_set="Ensemble: " + ens.feature_set),
                        abl.assign(feature_set="Ensemble: " + abl.feature_set)])
    ladder.to_csv(METRICS_DIR / "ablation_ladder.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(ladder))
    ax.barh(y, ladder.f1_macro, xerr=ladder.f1_macro_fold_std, color=["#52514e"] + ["#2a78d6"] * 3 + ["#eb6834"] * 3,
            height=0.6, error_kw=dict(ecolor="#52514e", lw=1, capsize=3))
    for yi, v in zip(y, ladder.f1_macro):
        ax.text(0.02, yi, f"{v:.3f}", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_yticks(y, ladder.feature_set); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Development CV macro-F1 (error bar = SD across 3 session-grouped folds)")
    ax.set_title("Ablation: what each feature group adds (blue = additive, orange = remove one group)")
    ax.grid(axis="y", visible=False)
    viz.save(fig, "10_ablation_ladder")

    # Precision / recall / F1 comparison and per-class recall
    pick = main_res[((main_res.model == "Logistic Regression") & (main_res.feature_set == "HR only")) |
                    ((main_res.model == proposed) & (main_res.feature_set == "HR only")) |
                    ((main_res.model == "Gradient Boosting") & (main_res.feature_set == core_fs)) |
                    ((main_res.model == proposed) & (main_res.feature_set == core_fs))].copy()
    pick["label"] = pick.model + " | " + pick.feature_set.str.replace(" (multimodal core)", "")
    viz.plot_metric_bars(pick, "label", ["precision_macro", "recall_macro", "f1_macro", "balanced_accuracy"],
                         "11_cv_precision_recall", "Development CV: precision, recall and F1 (macro)")
    viz.plot_metric_bars(pick, "label", [f"sensitivity_{c}" for c in STRAIN_ORDER] +
                         [f"specificity_{c}" for c in STRAIN_ORDER], "12_cv_sensitivity_specificity",
                         "Development CV: sensitivity and specificity per strain level")

    # Out-of-fold confusion matrices + ROC (baseline vs proposed)
    def get(model, fs):
        d = oof[(oof.experiment == "main") & (oof.model == model) & (oof.feature_set == fs)]
        return d.y_true.to_numpy(), d.y_pred.to_numpy(), d[[f"p_{c}" for c in STRAIN_ORDER]].to_numpy()
    pairs = {"Baseline: HR-only Logistic Regression": get("Logistic Regression", "HR only"),
             f"Proposed: {proposed}, multimodal core": get(proposed, core_fs)}
    viz.plot_confusion_pair(pairs, "13_cv_confusion", "Development CV (out-of-fold, pooled over 3 folds)")
    viz.plot_roc_pair(pairs, "14_cv_roc", "Development CV ROC curves (out-of-fold)")

    # Compact comparison table for the report
    cols = ["experiment", "model", "feature_set", "imbalance", "n_features", "accuracy", "precision_macro",
            "recall_macro", "f1_macro", "f1_macro_fold_std", "balanced_accuracy", "roc_auc_ovr_macro",
            "sensitivity_Low", "sensitivity_Moderate", "sensitivity_High"]
    res[cols].round(3).to_csv(METRICS_DIR / "table_cv_comparison.csv", index=False)
    print(res[cols].round(3).to_string(index=False))


# ----------------------------------------------------------------------------- insights
def stage_insights():
    import json
    from src import visualization as viz
    clean = load_clean()
    dev = clean[clean.split == "development"]
    viz.plot_hr_overlap(dev, "15_why_hr_alone_fails")
    viz.plot_session_profiles(dev, {"hr_mean": "Mean HR (bpm)", "hrv_rmssd": "RMSSD (ms)",
                                    "hrv_cv_rr": "CV of RR", "ecg_hf_noise_ratio": "ECG HF-noise ratio",
                                    "ecg_kurtosis": "ECG kurtosis", "temp_mean": "Skin temp (°C)",
                                    "resp_rate": "Resp. rate (br/min)"}, "16_session_profiles")
    proposed = json.loads((METRICS_DIR / "proposed_model.json").read_text())["name"]
    s = pd.read_csv(METRICS_DIR / "test_per_session_predictions.csv")
    base = s[(s.model == "Logistic Regression") & (s.feature_set == "HR only")].assign(
        label="Baseline: HR-only Logistic Regression")
    prop = s[(s.model == proposed) & (s.feature_set == "HR + HRV + ECG (multimodal core)")].assign(
        label=f"Proposed: {proposed} (multimodal core)")
    viz.plot_test_session_predictions(pd.concat([base, prop]), "17_test_session_predictions")


STAGES = {"data": stage_data, "stats": stage_stats, "models": stage_models, "report": stage_report,
          "test": stage_test, "explain": stage_explain, "insights": stage_insights}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    stage = ap.parse_args().stage
    for name, fn in STAGES.items():
        if stage in (name, "all"):
            print(f"\n===== stage: {name} =====")
            fn()
