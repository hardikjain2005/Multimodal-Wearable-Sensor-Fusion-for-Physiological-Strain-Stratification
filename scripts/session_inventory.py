"""Phase 1b: session-level inventory across all downloaded AIDLAB sessions.

Writes results/metrics/session_inventory.csv (also produced by `python main.py --stage data`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build_dataset import session_inventory  # noqa: E402

inv = session_inventory()
print(inv[["session_id", "subject", "duration_min", "ecg_rows", "respiration_rows",
           "temperature_rows", "acceleration_rows"]].to_string(index=False))
print("\nECG coverage per session:\n", inv[["session_id", "ecg_cov"]].to_string(index=False))
print("\nSessions per (label, subject):\n", inv.groupby(["strain_label", "subject"]).size())
