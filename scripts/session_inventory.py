"""Phase 1b: session-level inventory across all downloaded AIDLAB sessions.

Outputs results/metrics/session_inventory.csv with, per session: activity, proxy label,
pseudo-subject, duration, and for each stream the number of rows and time coverage
(fraction of 1-second bins of the session containing at least one sample).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_SESSIONS_DIR, METRICS_DIR          # noqa: E402
from src.data_loader import list_session_dirs, load_session    # noqa: E402

STREAMS = ["ecg", "rr", "pulse", "respiration", "temperature", "acceleration",
           "hrv", "respiration-rate", "body-position", "steps"]
rows = []
for d in list_session_dirs(RAW_SESSIONS_DIR):
    s = load_session(d, streams=STREAMS)
    r = dict(session_id=s.session_id, activity=s.activity, strain_label=s.strain_label,
             subject=s.subject, duration_min=round(s.duration_min, 1))
    n_sec = int((s.end_ms - s.start_ms) // 1000)
    for name in STREAMS:
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
    print(f"{s.session_id:38s} {s.subject:28s} {r['duration_min']:6.1f} min  "
          + " ".join(f"{n}={'Y' if n in s.streams else '-'}" for n in STREAMS[:6]))

inv = pd.DataFrame(rows)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
inv.to_csv(METRICS_DIR / "session_inventory.csv", index=False)
print("\nECG coverage per session:\n", inv[["session_id", "ecg_cov"]].to_string(index=False))
print("\nSessions per (label, subject):\n", inv.groupby(["strain_label", "subject"]).size())
