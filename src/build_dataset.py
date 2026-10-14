"""Build the window-level feature table for all sessions and apply quality control.

Output: data/processed/windows_all.csv     (every window, before quality filtering)
        data/processed/windows_clean.csv   (windows passing quality control, with split column)
        results/metrics/window_qc_summary.csv
"""
import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, METRICS_DIR, RAW_SESSIONS_DIR
from .data_loader import list_session_dirs, load_session
from .preprocessing import extract_session_windows
from .splits import assign_split
from . import feature_engineering as fe

# Streams needed for features (body-position, steps, GPS etc. are not used)
STREAMS = ["ecg", "rr", "pulse", "respiration", "temperature", "acceleration"]

# Core features = available in every session -> define the common window set
CORE_FEATURES = fe.HR_KEYS + fe.HRV_KEYS + fe.ECG_KEYS


def build(verbose: bool = True) -> pd.DataFrame:
    frames, sessions = [], []
    for d in list_session_dirs(RAW_SESSIONS_DIR):
        s = load_session(d, streams=STREAMS)
        w = extract_session_windows(s)
        frames.append(w)
        sessions.append(dict(session_id=s.session_id, strain_label=s.strain_label,
                             subject=s.subject, start_ms=s.start_ms))
        if verbose:
            print(f"  {s.session_id:38s} windows={len(w):4d}")
    windows = pd.concat(frames, ignore_index=True)
    sess = pd.DataFrame(sessions)
    sess["split"] = assign_split(sess)
    windows = windows.merge(sess[["session_id", "split"]], on="session_id")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_csv(PROCESSED_DIR / "windows_all.csv", index=False)
    sess.to_csv(PROCESSED_DIR / "sessions.csv", index=False)
    return windows


INVENTORY_STREAMS = ["ecg", "rr", "pulse", "respiration", "temperature", "acceleration",
                     "hrv", "respiration-rate", "body-position", "steps"]


def session_inventory() -> pd.DataFrame:
    """Per session: activity, proxy label, pseudo-subject, duration, and for each stream the
    number of rows and time coverage (share of 1-s bins of the session with >= 1 sample).
    Written to results/metrics/session_inventory.csv."""
    rows = []
    for d in list_session_dirs(RAW_SESSIONS_DIR):
        s = load_session(d, streams=INVENTORY_STREAMS)
        r = dict(session_id=s.session_id, activity=s.activity, strain_label=s.strain_label,
                 subject=s.subject, duration_min=round(s.duration_min, 1))
        n_sec = int((s.end_ms - s.start_ms) // 1000)
        for name in INVENTORY_STREAMS:
            if name in s.streams:
                t = s.streams[name]["timestamp"].to_numpy(np.int64)
                sec = np.unique((t - s.start_ms) // 1000)
                sec = sec[(sec >= 0) & (sec < n_sec)]
                r[f"{name}_rows"] = len(t)
                # coverage for continuous streams; slow streams (<1 Hz) are naturally sparse
                r[f"{name}_cov"] = round(len(sec) / n_sec, 3)
            else:
                r[f"{name}_rows"] = 0
        rows.append(r)
    inv = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(METRICS_DIR / "session_inventory.csv", index=False)
    return inv


def quality_control(windows: pd.DataFrame) -> pd.DataFrame:
    """Label-blind quality rules applied identically to every session:
    1. ECG coverage >= 90 % of expected samples (contact loss / recording gaps)
    2. all core features computable (>= 10 HR samples, >= 20 valid beats for HRV)
    Duplicate windows are impossible by construction (non-overlapping, unique start times),
    but are checked anyway.
    """
    qc = windows.copy()
    qc["fail_coverage"] = qc["ecg_coverage"] < 0.9
    qc["fail_core_missing"] = ~qc["fail_coverage"] & qc[CORE_FEATURES].isna().any(axis=1)
    keep = ~(qc.fail_coverage | qc.fail_core_missing)
    assert not windows.duplicated(subset=["session_id", "window_start_ms"]).any()
    summary = (qc.groupby(["split", "strain_label", "session_id"])
               .agg(windows=("window_index", "size"),
                    dropped_coverage=("fail_coverage", "sum"),
                    dropped_core_missing=("fail_core_missing", "sum"))
               .reset_index())
    summary["kept"] = summary.windows - summary.dropped_coverage - summary.dropped_core_missing
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(METRICS_DIR / "window_qc_summary.csv", index=False)
    clean = windows[keep].reset_index(drop=True)
    clean.to_csv(PROCESSED_DIR / "windows_clean.csv", index=False)
    return clean


if __name__ == "__main__":
    w = build()
    c = quality_control(w)
    print(pd.read_csv(METRICS_DIR / "window_qc_summary.csv").to_string(index=False))
    print("\nClean windows per split x label:\n", c.groupby(["split", "strain_label"]).size())
