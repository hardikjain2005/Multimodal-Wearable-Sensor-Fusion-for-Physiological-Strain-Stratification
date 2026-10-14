# Phase 1 — Dataset Inspection Findings

All numbers below come from executing `scripts/inspect_sessions.py`, `scripts/inspect_crossstream.py`
and `scripts/inspect_har.py` on the downloaded archives (2026-10-14). Full per-stream table:
`results/metrics/phase1_stream_inventory.csv`.

## 1. Files and formats

| Archive | Size | Format | Files |
|---|---|---|---|
| sleep_5h50m | 56.1 MB | CSV + `dataset.json` | ecg, respiration, rr, pulse, temperature, body-position |
| yoga_73min | 11.5 MB | CSV + `dataset.json` | ecg, respiration, respiration-rate, rr, pulse, hrv, temperature, body-position |
| squash_57min | 12.6 MB | CSV + `dataset.json` | acceleration, ecg, respiration, respiration-rate, rr, pulse, calories-burned |
| hike_5h20m | 50.8 MB | CSV + GPX + `dataset.json` | ecg, rr, pulse, temperature, steps, pace, distance, elevation, calories-burned, locations.gpx |
| AIDLAB-HAR v3 | 3.3 MB | EDF + CSV | 180 EDF recordings, 130 annotation CSVs, quality_intervals.csv (CC BY 4.0) |

Session CSV schema: `"timestamp","value"` (Unix epoch, **milliseconds**, UTC).
Exceptions: `acceleration.csv` → `timestamp,x,y,z`; `body-position.csv` → `timestamp,value,position`.
`dataset.json` holds user metadata, signal units, and session `from`/`to`.

## 2. Sensor availability (rows per stream) — modalities differ per session

| Stream | Sleep | Yoga | Squash | Hike | Rate (measured) |
|---|---:|---:|---:|---:|---|
| ECG | 5,224,328 | 1,000,930 | 855,018 | 4,793,106 | 250 Hz (dt = 4 ms) |
| RR intervals | 23,588 | 5,645 | 7,659 | 32,679 | event-based (per beat) |
| Pulse (HR) | 23,590 | 5,836 | 7,423 | 31,133 | event-based, 1–2 Hz reporting |
| Respiration (impedance, Ω) | 1,044,866 | 200,188 | 171,001 | **—** | 50 Hz |
| Respiration-rate | **—** | 274 | 227 | **—** | ~every 15 s |
| Skin temperature | 2,097 | 412 | **—** | 1,922 | ~0.1 Hz (every 10 s) |
| Acceleration (x,y,z) | **—** | **—** | 196,143 | **—** | ~58.8 Hz (dt = 17 ms) |
| Device HRV | **—** | 455 | **—** | **—** | ~every 10 s |
| Body position | 358 | 118 | — | — | 1/min (sleep) |
| Steps/pace/distance/GPS | — | — | — | yes | irregular |

**Only ECG, RR and pulse are present in all four sessions.**

## 3. Session boundaries, duration, subject

| Session | From (UTC) | To (UTC) | Duration | Median HR |
|---|---|---|---:|---:|
| Sleep | 2026-04-21 23:20 | 2026-04-22 05:10 | 350.0 min | 68 bpm |
| Yoga | 2026-05-05 16:40 | 2026-05-05 17:53 | 73.2 min | 86 bpm |
| Squash | 2026-05-25 11:01 | 2026-05-25 11:58 | 57.0 min | 145 bpm |
| Hike | 2026-06-06 07:20 | 2026-06-06 12:40 | 320.3 min | 102 bpm |

All four `dataset.json` files carry identical user metadata (female, 30–40, 170–180 cm).
→ Very likely **one person**, but this cannot be confirmed (no subject ID).
No activity/strain label column exists — the label is only the session (file) itself.

## 4. Data quality

- Missing cells: **0** in every stream. Missingness appears as **time gaps** instead.
- Yoga: 5 gaps > 5 s in ECG/respiration/RR (max 242 s; ≈379 s of ECG lost — likely contact loss).
- Duplicates: squash ECG 40 duplicate timestamps (10 exact duplicate rows); hike ECG 32 duplicate
  timestamps; squash respiration 1 duplicate row. Negligible but will be removed.
- RR artifacts: |successive change| > 20% in 0.34% (sleep), 0.90% (yoga), **11.1% (squash)**,
  **11.3% (hike)** of beats; hike pulse max 218 bpm and RR min 252 ms are implausible spikes.
  → HRV needs artifact filtering before computation, especially in motion sessions.
- Pulse ≈ 60000/RR (r = 0.83–0.89, median |Δ| 1.3–2.7 bpm) → pulse is RR-derived, not an
  independent sensor.
- Pulse reporting cadence differs per session (sleep ~1 Hz, hike/squash ~2 Hz) → sample counts
  must never be used as features.

## 5. Unit discrepancies

- `acceleration` declared "meters per second squared", but median |a| = 1.015 → values are in **g**.
- `respiration-rate` declared "milliseconds of breaths per minute"; values 7–43 → treated as breaths/min.

## 6. HRV
- No HRV file in sleep/squash/hike; yoga has a device HRV stream (~10 s, unknown algorithm).
- HRV **must be computed by us** from RR (all sessions) and/or ECG R-peaks (250 Hz, all sessions).
- Time-domain HRV (RMSSD, SDNN, pNN50, mean/median RR) is feasible on 60 s windows.
- LF/HF is **not reliable** on 30–60 s windows (LF band 0.04–0.15 Hz needs ≥ ~2–5 min of stationary
  data) and is ill-defined during non-stationary exercise. Recommended: exclude from the model.

## 7. AIDLAB-HAR v3 (intended independent test set)
- 180 EDF recordings, all with exactly 7 channels: acceleration x/y/z (g) + quaternion x/y/z/w, 50 Hz.
- **No ECG, HR, RR, HRV, respiration or temperature.**
- Labels: 16 exercise types (squats, burpees, plank, walk, downward dog, …), not strain levels.
- Recording length: mostly 12–120 s (walk ≈ 5.6 min); 172.8 min total.
- SUBxx codes are pseudonymous and not global participant IDs (per README).
→ **Incompatible** with an HR/HRV/multimodal model: it shares no physiological modality with the
  training sessions, and its only modality (accelerometer) exists in just one training session (squash).

## 8. Critical design implications

1. **One session per activity** → session-grouped CV (GroupKFold/StratifiedGroupKFold) is impossible
   for 3 classes: holding out the only sleep session removes the whole Low class from training.
2. **Modality availability is confounded with the label.** Temperature is missing only in squash,
   respiration only in hike, and acceleration exists only in squash. Imputation or a "missing" flag
   would let the model learn *which sensor exists* rather than physiology → label leakage.
3. Hike median HR (102) is closer to yoga (86) than to squash (145), so "Hike = High" is debatable.
4. Labels must NOT be derived from HR thresholds, since the baseline uses HR (circular evaluation).
5. The AIDLAB catalog has more sessions of the same activities (5 sleep, 5 squash, 4 yoga, 4 hike,
   plus walks/runs), which would make session-level CV and a genuine held-out-session test possible,
   if the professor's approval covers them.

---

# Phase 1b — Additional catalog sessions (author confirmed the approval covers the AIDLAB catalog)

14 further sessions of the same four activities were downloaded (18 in total). Full table:
`results/metrics/session_inventory.csv` (generated by `scripts/session_inventory.py`).

## New findings

1. **Two participants (plus one unknown).** `dataset.json` user metadata shows
   *female, 30–40, 170–180 cm* (all sleep/yoga/hike sessions + squash 04-27_54min and 05-25_57min) and
   *male, 30–40, 190–200 cm* (squash 04-27_60min, 05-25_61min, 06-01_60min).
   The female and male squash sessions overlap in time on the same days → most likely the two
   opponents of the same matches, **not duplicate exports** (different subjects, different sensors).
   `sleep_2026-10-03_6h30m` uses a newer metadata schema with **no user field** → subject unknown.
   The male participant appears **only in the High class** → subject is confounded with label if
   the sessions are pooled.
2. **Two `dataset.json` schemas**: ISO `from`/`to` vs `from_timestamp_ms`/`to_timestamp_ms`;
   file naming differs (`respirationRate.csv` vs `respiration-rate.csv`). The loader handles both.
3. **Licences**: `sleep_2026-10-03_6h30m` states MIT; HAR v3 is CC BY 4.0; the other session
   archives state no licence → raw data is not committed to git.
4. **Modality availability by session** (Y = present):

| Session | Subject | ECG | device RR | Resp. | Temp. | Accel. |
|---|---|---|---|---|---|---|
| sleep ×4 (03-23, 04-11, 04-21 4h40m, 04-21 5h50m) | F | Y | Y | Y | Y | – |
| sleep 10-03 6h30m | unknown | Y | Y | Y | – | Y |
| yoga 03-24, 04-09 | F | Y | Y | Y | – | – |
| yoga 04-30, 05-05 | F | Y | Y | Y | Y | – |
| hike ×3 (+03-11 pending) | F | Y | Y | – | Y | – |
| squash 04-27 54min, 05-25 57min | F | Y | Y | Y | – | Y |
| squash ×3 (male) | M | Y | – | – | Y | 2 of 3 |

   Only **ECG and pulse** exist in every session. Respiration is absent in every hike and every
   male squash session; accelerometer data is present in 5 sessions only.
5. **Per-session medians** (female): HR sleep 64–75, yoga 86–97, hike 96–107, squash 143–145 bpm.
   Yoga and hike HR overlap → Moderate vs High is the hard boundary for an HR-only model.
   Skin temperature: sleep 35.9–36.5 °C vs 32.3–34.4 °C for all other activities, which is plausibly
   driven by bedding/insulation (environment) rather than physiological strain.

## Signal-processing validation (Phase 2 preview, from executed code)

- Own R-peak detector (simplified Pan-Tompkins, 250 Hz ECG) vs device HR per 60 s window:
  median |Δ| 0.7 bpm (yoga) and 0.9 bpm (squash); 85–86 % of windows within 5 bpm.
- Impedance-respiration rate vs device respiration-rate stream: peak-counting r = 0.72
  (yoga, MAE 5.2 br/min) and r = 0.46 (squash, MAE 11.9 br/min); spectral and ECG-derived
  respiration (EDR) estimates failed during activity (r ≤ 0.24) → EDR is **not** used.
  During squash the impedance signal is dominated by motion artifacts (visual inspection).
