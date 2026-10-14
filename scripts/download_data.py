"""Download and extract the AIDLAB datasets used in this project from the official source.

    python scripts/download_data.py

Raw data is not redistributed in this repository (most session archives state no licence).
Source catalogue: https://aidlab.com/datasets
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://aidlab-production-datasets.s3.eu-central-1.amazonaws.com/"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

# session folder name -> archive file name
SESSIONS = {
    "sleep_2026-03-23_6h45m": "datasets_2026-03-23_sleep_6h45m.zip",
    "sleep_2026-04-11_6h": "datasets_2026-04-11_sleep_6h.zip",
    "sleep_2026-04-21_4h40m": "datasets_2026-04-21_sleep_4h40m.zip",
    "sleep_2026-04-21_5h50m": "datasets_2026-04-21_sleep_5h50m.zip",
    "sleep_2026-10-03_6h30m": "datasets_2026-10-03_sleep_6h30m.zip",
    "yoga_2026-03-24_80min": "datasets_2026-03-24_yoga_80min.zip",
    "yoga_2026-04-09_80min": "datasets_2026-04-09_yoga_80min.zip",
    "yoga_2026-04-30_62min": "datasets_2026-04-30_yoga_62min.zip",
    "yoga_2026-05-05_73min": "datasets_2026-05-05_yoga_73min.zip",
    "squash_2026-04-27_54min": "datasets_2026-04-27_squash_54min.zip",
    "squash_2026-04-27_60min": "datasets_2026-04-27_squash_60min.zip",
    "squash_2026-05-25_57min": "datasets_2026-05-25_squash_57min.zip",
    "squash_2026-05-25_61min_four_games": "datasets_2026-05-25_squash_61min_four_games.zip",
    "squash_2026-06-01_60min_four_games": "datasets_2026-06-01_squash_60min_four_games.zip",
    "hike_2026-03-11_4h16m": "datasets_2026-03-11_hike_4h16m.zip",
    "hike_2026-06-04_3h25m": "datasets_2026-06-04_hike_3h25m.zip",
    "hike_2026-06-06_5h20m": "datasets_2026-06-06_hike_5h20m.zip",
    "hike_2026-06-07_1h22m": "datasets_2026-06-07_hike_1h22m.zip",
}
HAR = "AIDLAB-HAR-DATASET_v3.zip"


def fetch(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    with urllib.request.urlopen(url) as r, open(dest.with_suffix(".part"), "wb") as f:
        shutil.copyfileobj(r, f)
    dest.with_suffix(".part").rename(dest)


def safe_extract(zip_path: Path, target: Path):
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in {zip_path}: {name}")
        z.extractall(target)


if __name__ == "__main__":
    for folder, archive in SESSIONS.items():
        d = ROOT / "sessions" / folder
        fetch(BASE + archive, d / "session.zip")
        if not (d / "dataset.json").exists():
            safe_extract(d / "session.zip", d)
    fetch(BASE + HAR, ROOT / "har_v3" / "har_v3.zip")
    if not (ROOT / "har_v3" / "AIDLAB-HAR-DATASET-v3").exists():
        safe_extract(ROOT / "har_v3" / "har_v3.zip", ROOT / "har_v3")
    print("done:", len(SESSIONS), "sessions + AIDLAB-HAR v3 in", ROOT)
