# Pre-specified analysis plan (written before any model was trained — 2026-10-14)

## Data
- Unit: non-overlapping 60-s windows; label = activity-derived strain proxy
  (sleep → Low, yoga → Moderate, squash/hike → High). Not clinical labels.
- Splits (session-level, fixed in `src/splits.py`):
  - **development**: 11 female sessions (3 sleep, 3 yoga, 3 hike, 2 squash)
  - **test_temporal**: last female session per class (sleep 04-21 5h50m, yoga 05-05, hike 06-07)
  - **test_cross_person**: 3 male squash sessions (High) + sleep 10-03 with no subject metadata (Low)
  - AIDLAB-HAR v3: not used for evaluation (motion-only, exercise-type labels — incompatible)

## Feature groups
- HR (baseline, device pulse): 9 features
- HRV (from own ECG R-peak detection): SDNN, RMSSD, pNN50, CV-RR (mean/median RR excluded — duplicate of HR)
- ECG quality / motion-artifact: kurtosis, HF-noise ratio, LF-wander ratio, R-amplitude CV, beat-artifact fraction
- Add-ons (not in every session): temperature, respiration. Accelerometer: descriptive only.
- No imputation of missing modalities (missingness is label-dependent).

## Validation
- Outer CV: 3 session-grouped folds (each fold holds out ≥1 unseen session per class).
- Inner CV (tuning): 2 session-grouped folds within the outer training data.
- Primary metric: **macro-F1** (each strain level weighted equally). Also reported: accuracy,
  balanced accuracy, macro precision/recall, per-class sensitivity/specificity, macro OvR ROC-AUC.

## Experiments (development CV)
1. HR only — all models (Logistic Regression = baseline)
2. HR + HRV — all models
3. HR + HRV + ECG ("multimodal core") — all models
4. Ensembles RF+GB (soft voting, stacking) on each feature set
5. Leave-one-group-out ablation of the proposed model (−HR, −HRV, −ECG)
6. Imbalance handling for the proposed model: class weights vs none vs SMOTE (training folds only)
7. Respiration add-on: core vs core+RESP on dev sessions with respiration (sleep, yoga, squash), 2-fold grouped CV
8. Temperature add-on: dev has only ONE yoga session with temperature, so grouped CV is impossible for
   3 classes → evaluated as a secondary analysis on test_temporal (all 3 sessions have temperature),
   training on dev windows with temperature. Declared here in advance; not used for model selection.

## Model selection (locked before testing)
- Baseline: HR-only Logistic Regression.
- Proposed model: on the multimodal core feature set, the ensemble variant (soft voting vs stacking)
  with the higher development-CV macro-F1. Imbalance strategy: class weights unless experiment 6 shows
  a different strategy is better by > 0.02 macro-F1.
- Final models are re-tuned on the full development set (3-fold grouped CV) and saved.
- Test sets are evaluated **once**; no changes are made after seeing test results.
