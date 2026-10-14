"""Time alignment, windowing and window-level dataset construction.

Alignment strategy (no resampling): every stream keeps its native sampling rate
(ECG 250 Hz, respiration 50 Hz, RR/pulse event-based, temperature ~0.1 Hz). All streams
share one clock (Unix ms), so a window is the half-open interval
[t0 + k*W, t0 + (k+1)*W) applied to each stream's timestamps. Features are computed per
stream inside the window with a method suited to that stream's rate. This avoids
up-sampling slow signals (inventing data) or down-sampling ECG (destroying the R-peak
timing that HRV depends on).
"""
import numpy as np
import pandas as pd

from . import feature_engineering as fe
from .config import WINDOW_SECONDS, ECG_FS


def window_bounds(session, window_s: int = WINDOW_SECONDS) -> np.ndarray:
    """Window start times (ms). Only complete, non-overlapping windows are kept."""
    w_ms = window_s * 1000
    n_windows = int((session.end_ms - session.start_ms) // w_ms)
    return session.start_ms + np.arange(n_windows, dtype=np.int64) * w_ms


def slice_indices(timestamps: np.ndarray, starts: np.ndarray, window_s: int = WINDOW_SECONDS):
    """For sorted `timestamps`, return (lo, hi) so timestamps[lo[k]:hi[k]] lie in window k."""
    lo = np.searchsorted(timestamps, starts, side="left")
    hi = np.searchsorted(timestamps, starts + window_s * 1000, side="left")
    return lo, hi


def extract_session_windows(session, window_s: int = WINDOW_SECONDS) -> pd.DataFrame:
    """Slice every available stream of one session into aligned windows and compute features.

    A modality the session does not record yields NaN columns. These are NEVER imputed,
    because sensor availability differs by activity (imputation would leak the label).
    """
    starts = window_bounds(session, window_s)
    st = session.streams
    rows = [dict(session_id=session.session_id, activity=session.activity,
                 strain_label=session.strain_label, subject=session.subject,
                 window_index=k, window_start_ms=int(s)) for k, s in enumerate(starts)]

    def per_window(stream_name, value_cols, func, needs_time=True):
        if stream_name not in st:
            return
        df = st[stream_name]
        t = df["timestamp"].to_numpy(np.int64)
        vals = df[value_cols].to_numpy()
        if len(value_cols) == 1:
            vals = vals[:, 0]
        lo, hi = slice_indices(t, starts, window_s)
        for k in range(len(starts)):
            args = (t[lo[k]:hi[k]], vals[lo[k]:hi[k]]) if needs_time else (vals[lo[k]:hi[k]],)
            rows[k].update(func(*args))

    # ---- ECG: coverage, quality features, and session-level beat detection -> HRV
    t_ecg = st["ecg"]["timestamp"].to_numpy(np.int64)
    v_ecg = st["ecg"]["value"].to_numpy(float)
    lo, hi = slice_indices(t_ecg, starts, window_s)
    for k in range(len(starts)):
        rows[k]["ecg_coverage"] = (hi[k] - lo[k]) / (window_s * ECG_FS)
        rows[k].update(fe.ecg_quality_features(v_ecg[lo[k]:hi[k]]))

    beats = fe.detect_session_beats(t_ecg, v_ecg)
    b_t = beats["timestamp"].to_numpy(np.int64)
    b_rr = beats["rr_ms"].to_numpy(float)
    b_amp = beats["r_amp"].to_numpy(float)
    b_valid = fe.clean_rr_series(b_rr)
    lo, hi = slice_indices(b_t, starts, window_s)
    for k in range(len(starts)):
        sl = slice(lo[k], hi[k])
        rows[k].update(fe.hrv_features(b_rr[sl], b_valid[sl]))
        n = hi[k] - lo[k]
        rows[k]["ecg_beat_artifact_frac"] = 1 - b_valid[sl].mean() if n else np.nan
        amps = b_amp[sl][b_valid[sl]]
        rows[k]["ecg_ramp_cv"] = amps.std(ddof=1) / amps.mean() if len(amps) >= 20 else np.nan

    # ---- device streams
    per_window("pulse", ["value"], fe.hr_features)
    per_window("respiration", ["value"], fe.resp_features, needs_time=False)
    per_window("temperature", ["value"], fe.temp_features)
    per_window("acceleration", ["x", "y", "z"], fe.acc_features)

    # ---- validation only: HRV from the device RR stream (to check our own beat detection)
    if "rr" in st:
        t = st["rr"]["timestamp"].to_numpy(np.int64)
        values = st["rr"]["value"].to_numpy(float)
        valid = fe.clean_rr_series(values)
        lo, hi = slice_indices(t, starts, window_s)
        for k in range(len(starts)):
            rows[k].update(fe.hrv_features(values[lo[k]:hi[k]], valid[lo[k]:hi[k]], prefix="val_dev_"))

    df = pd.DataFrame(rows)
    # Guarantee every feature column exists (NaN where the modality is not recorded)
    for keys in fe.FEATURE_GROUPS.values():
        for c in keys:
            if c not in df:
                df[c] = np.nan
    return df
