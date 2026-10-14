"""Central configuration: paths, random seed, windowing and label mapping."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SESSIONS_DIR = PROJECT_ROOT / "data" / "raw" / "sessions"
RAW_HAR_DIR = PROJECT_ROOT / "data" / "raw" / "har_v3" / "AIDLAB-HAR-DATASET-v3"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

RANDOM_SEED = 42

# Windowing: fixed-length, non-overlapping windows (no overlap -> no shared samples
# between neighbouring windows).
WINDOW_SECONDS = 60

# Activity -> strain proxy label. These are ACTIVITY-DERIVED categories, not clinical labels.
# Hike = High was chosen by the project author (see docs/phase1_dataset_inspection.md, sec. 8).
ACTIVITY_TO_STRAIN = {
    "sleep": "Low",
    "yoga": "Moderate",
    "squash": "High",
    "hike": "High",
}
STRAIN_ORDER = ["Low", "Moderate", "High"]

# Nominal ECG sampling rate, verified in Phase 1 (median dt = 4 ms in every session).
ECG_FS = 250.0
# Respiration impedance sampling rate, verified in Phase 1 (median dt = 20 ms).
RESP_FS = 50.0
