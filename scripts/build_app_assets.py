"""Build the small, derived-feature assets the Streamlit dashboard needs (no raw signals).

    python scripts/build_app_assets.py      # after `python main.py --stage data`

Outputs (app/assets/):
    feature_reference.json : per-class median feature vectors (presets) + training ranges
    shap_background.csv    : 120 development windows (core features) used as SHAP background
    dev_sample.csv         : 600 development windows (core features + label) for context plots
All rows come from the DEVELOPMENT split only (held-out test sessions are not exposed).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import feature_engineering as fe          # noqa: E402
from src.config import PROCESSED_DIR, RANDOM_SEED, STRAIN_ORDER  # noqa: E402

CORE = fe.HR_KEYS + fe.HRV_MODEL_KEYS + fe.ECG_KEYS
OUT = ROOT / "app" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

dev = pd.read_csv(PROCESSED_DIR / "windows_clean.csv")
dev = dev[dev.split == "development"].reset_index(drop=True)

ref = {
    "core_features": CORE,
    "presets": {c: dev.loc[dev.strain_label == c, CORE].median().round(5).to_dict() for c in STRAIN_ORDER},
    "range": {f: {"p01": float(dev[f].quantile(0.01)), "p99": float(dev[f].quantile(0.99)),
                  "min": float(dev[f].min()), "max": float(dev[f].max())} for f in CORE},
    "n_dev_windows": int(len(dev)),
    "n_dev_sessions": int(dev.session_id.nunique()),
}
(OUT / "feature_reference.json").write_text(json.dumps(ref, indent=1))

bg = dev.groupby("strain_label").sample(n=40, random_state=RANDOM_SEED)
bg[CORE].round(6).to_csv(OUT / "shap_background.csv", index=False)

sample = dev.groupby("strain_label").sample(n=200, random_state=RANDOM_SEED)
sample[CORE + ["strain_label"]].round(6).to_csv(OUT / "dev_sample.csv", index=False)
print("wrote", sorted(p.name for p in OUT.iterdir()))
