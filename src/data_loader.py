"""Load AIDLAB session exports.

Each session directory contains `dataset.json` (metadata) and one CSV per stream with a
`timestamp` column in Unix epoch milliseconds (verified in Phase 1). Streams are kept at
their native rates; alignment happens later by slicing every stream on the same
time-window boundaries (see preprocessing.py).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import RAW_SESSIONS_DIR, ACTIVITY_TO_STRAIN


@dataclass
class Session:
    session_id: str
    activity: str
    strain_label: str
    start_ms: int
    end_ms: int
    user: dict
    subject: str
    path: Path
    streams: dict = field(default_factory=dict)  # stream name -> DataFrame

    @property
    def duration_min(self) -> float:
        return (self.end_ms - self.start_ms) / 60000


def list_session_dirs(root: Path = RAW_SESSIONS_DIR) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "dataset.json").exists())


# Some exports use camelCase file names for the same stream.
STREAM_ALIASES = {"respirationRate": "respiration-rate", "bodyPosition": "body-position"}


def session_time_range(meta: dict) -> tuple[int, int]:
    """Two dataset.json schemas exist: ISO `from`/`to`, or `from_timestamp_ms`/`to_timestamp_ms`."""
    if "from_timestamp_ms" in meta:
        return int(meta["from_timestamp_ms"]), int(meta["to_timestamp_ms"])
    to_ms = lambda iso: int(pd.Timestamp(iso).value // 1_000_000)
    return to_ms(meta["from"]), to_ms(meta["to"])


def subject_key(user: dict) -> str:
    """Pseudo-subject identifier from the only demographic metadata available.
    Sessions without metadata are marked 'unknown' (cannot be attributed)."""
    if not user:
        return "unknown"
    return f"{user.get('gender', '?')}|{user.get('age', '?')}|{user.get('height', '?')}".replace(" ", "")


def load_stream(csv_path: Path) -> pd.DataFrame:
    """Read one stream, sort by time and drop duplicate timestamps (keep first)."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    return df


def load_session(session_dir: Path, streams: list[str] | None = None) -> Session:
    """Load a session. `streams=None` loads every CSV in the directory."""
    meta = json.loads((session_dir / "dataset.json").read_text())
    activity = session_dir.name.split("_")[0]
    start_ms, end_ms = session_time_range(meta)
    user = meta.get("user", {})
    session = Session(
        session_id=session_dir.name,
        activity=activity,
        strain_label=ACTIVITY_TO_STRAIN[activity],
        start_ms=start_ms,
        end_ms=end_ms,
        user=user,
        subject=subject_key(user),
        path=session_dir,
    )
    for csv_path in sorted(session_dir.glob("*.csv")):
        name = STREAM_ALIASES.get(csv_path.stem, csv_path.stem)
        if streams is None or name in streams:
            session.streams[name] = load_stream(csv_path)
    return session
