"""Inference on a new recording (used by the Streamlit dashboard).

Re-uses exactly the training pipeline: same 60-s windowing, feature extraction and
label-blind quality control, then applies the saved (locked) models.
"""
import json
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import feature_engineering as fe
from .config import MODELS_DIR, STRAIN_ORDER
from .data_loader import STREAM_ALIASES, Session, load_stream, session_time_range, subject_key
from .evaluation import align_proba
from .preprocessing import extract_session_windows

CORE = fe.HR_KEYS + fe.HRV_MODEL_KEYS + fe.ECG_KEYS
STREAMS = ["ecg", "pulse", "rr", "respiration", "temperature", "acceleration"]
MODEL_FILES = {
    "baseline": "Logistic_Regression__HR_only.joblib",
    "proposed": "Ensemble_RFplusGB_soft_voting__HR_plus_HRV_plus_ECG_multimodal_core.joblib",
}
MAX_UNZIPPED_BYTES = 1_500_000_000


def load_models() -> dict:
    """{'baseline': {'model', 'features', 'params'}, 'proposed': {...}}"""
    return {k: joblib.load(MODELS_DIR / f) for k, f in MODEL_FILES.items()}


def safe_unzip(data: bytes, target: Path) -> Path:
    """Extract an uploaded ZIP after checking for path traversal and zip bombs.
    Returns the folder that contains ecg.csv."""
    zip_path = target / "upload.zip"
    zip_path.write_bytes(data)
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        if any(i.filename.startswith("/") or ".." in Path(i.filename).parts for i in infos):
            raise ValueError("The ZIP contains unsafe file paths.")
        if sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
            raise ValueError("The ZIP is too large once extracted.")
        z.extractall(target / "x")
    hits = sorted((target / "x").rglob("ecg.csv"))
    if not hits:
        raise ValueError("No ecg.csv found in the ZIP (expected an AIDLAB session export).")
    return hits[0].parent


def load_recording(folder: Path, name: str = "uploaded recording", max_minutes: int | None = None) -> tuple[Session, list[str]]:
    """Build a Session from a folder of AIDLAB CSVs (dataset.json optional).
    Returns the session and a list of notes about how it was interpreted."""
    notes = []
    streams = {}
    for p in sorted(folder.glob("*.csv")):
        key = STREAM_ALIASES.get(p.stem, p.stem)
        if key in STREAMS:
            streams[key] = load_stream(p)
    if "ecg" not in streams:
        raise ValueError("ecg.csv is required (raw ECG with columns timestamp,value).")
    for key, df in streams.items():
        if "timestamp" not in df.columns:
            raise ValueError(f"{key}.csv has no 'timestamp' column.")

    meta_path = folder / "dataset.json"
    user = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        start_ms, end_ms = session_time_range(meta)
        user = meta.get("user", {}) or {}
    else:
        t = streams["ecg"]["timestamp"]
        start_ms, end_ms = int(t.min()), int(t.max())
        notes.append("No dataset.json: the recording period was taken from the ECG timestamps.")

    if max_minutes:
        end_ms = min(end_ms, start_ms + max_minutes * 60_000)

    if "pulse" not in streams:
        beats = fe.detect_session_beats(streams["ecg"]["timestamp"].to_numpy(np.int64),
                                        streams["ecg"]["value"].to_numpy(float))
        hr = 60000 / beats["rr_ms"]
        ok = (hr >= 30) & (hr <= 220)
        streams["pulse"] = pd.DataFrame({"timestamp": beats["timestamp"][ok], "value": hr[ok].round()})
        notes.append("No pulse.csv: heart rate was derived from ECG beats (the device HR stream was used in training).")

    session = Session(session_id=name, activity="unknown", strain_label="Unknown", start_ms=start_ms,
                      end_ms=end_ms, user=user, subject=subject_key(user), path=folder, streams=streams)
    if session.duration_min < 1:
        raise ValueError("The recording is shorter than one 60-second window.")
    return session, notes


def predict_recording(session: Session, models: dict) -> pd.DataFrame:
    """One row per 60-s window: time, quality flag, key features, and both models' predictions."""
    w = extract_session_windows(session)
    w["minute"] = np.arange(len(w))
    w["time"] = pd.to_datetime(w.window_start_ms, unit="ms", utc=True)
    w["usable"] = (w.ecg_coverage >= 0.9) & w[CORE].notna().all(axis=1)
    for name, bundle in models.items():
        cols = [f"{name}_p_{c}" for c in STRAIN_ORDER]
        w[cols] = np.nan
        w[f"{name}_pred"] = None
        if w.usable.any():
            X = w.loc[w.usable, bundle["features"]].to_numpy(float)
            proba = align_proba(bundle["model"], X)
            w.loc[w.usable, cols] = proba
            w.loc[w.usable, f"{name}_pred"] = np.array(STRAIN_ORDER)[proba.argmax(1)]
    return w


def fetch_example(url: str, cache_dir: Path | None = None) -> bytes:
    """Download an AIDLAB example archive (curl first: system certificates, retries)."""
    import shutil
    import subprocess
    import urllib.request
    cache_dir = Path(cache_dir or tempfile.gettempdir())
    dest = cache_dir / Path(url).name
    if not dest.exists():
        part = dest.with_suffix(".part")
        if shutil.which("curl"):
            subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(part), url],
                           check=True, timeout=900)
        else:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
        part.rename(dest)
    return dest.read_bytes()


