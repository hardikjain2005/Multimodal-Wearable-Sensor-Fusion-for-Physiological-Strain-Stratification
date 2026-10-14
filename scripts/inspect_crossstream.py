"""Phase 1: cross-stream checks per session.

- Does pulse equal 60000/rr at matching timestamps (i.e. is HR derived from RR)?
- RR plausibility (fraction outside 300-2000 ms, successive-diff > 20%) -> HRV quality
- Device hrv.csv cadence (is it windowed? what window length?)
- Acceleration magnitude -> is the unit really m/s^2 or g?
- ECG gaps (contact loss) and exact sampling rate
"""
import sys, os
import numpy as np
import pandas as pd

raw_dir = sys.argv[1]
for s in sys.argv[2:]:
    d = os.path.join(raw_dir, s)
    rd = lambda n: pd.read_csv(os.path.join(d, n + ".csv")) if os.path.exists(os.path.join(d, n + ".csv")) else None
    print(f"\n################ {s}")
    rr, pulse = rd("rr"), rd("pulse")
    if rr is not None:
        v = rr["value"]
        print(f"  RR: n={len(v)}, median={v.median():.0f}ms, p1={v.quantile(.01):.0f}, p99={v.quantile(.99):.0f}, "
              f"outside[300,2000]ms={((v < 300) | (v > 2000)).mean():.2%}, "
              f"|succ diff|>20%={(v.pct_change().abs() > 0.2).mean():.2%}")
        if pulse is not None:
            m = pulse.merge(rr, on="timestamp", suffixes=("_hr", "_rr"))
            print(f"  pulse/rr shared timestamps: {len(m)}/{len(pulse)} pulse rows; "
                  f"corr(pulse, 60000/rr)={np.corrcoef(m.value_hr, 60000 / m.value_rr)[0, 1]:.3f}; "
                  f"median |pulse-60000/rr|={np.median(np.abs(m.value_hr - 60000 / m.value_rr)):.1f} bpm")
            print(f"  pulse: median={pulse.value.median():.0f} bpm, min={pulse.value.min()}, max={pulse.value.max()}")
    hrv = rd("hrv")
    if hrv is not None:
        dt = np.diff(hrv.timestamp) / 1000
        print(f"  device hrv.csv: n={len(hrv)}, cadence median={np.median(dt):.1f}s, "
              f"value median={hrv.value.median():.1f}ms range=[{hrv.value.min():.1f},{hrv.value.max():.1f}]")
    acc = rd("acceleration")
    if acc is not None:
        mag = np.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)
        print(f"  acceleration |a|: median={mag.median():.3f}, p99={mag.quantile(.99):.3f}, max={mag.max():.3f} "
              f"(median ~1 => units are g, ~9.8 => m/s^2)")
    ecg = rd("ecg")
    if ecg is not None:
        dt = np.diff(ecg.timestamp)
        print(f"  ECG: dt value counts (ms) {pd.Series(dt).value_counts().head(4).to_dict()}, "
              f"gaps>1s={(dt > 1000).sum()}, total gap time={dt[dt > 1000].sum() / 1000:.0f}s, "
              f"exact rate={(len(ecg) - 1) / ((ecg.timestamp.iloc[-1] - ecg.timestamp.iloc[0]) / 1000):.2f}Hz")
    resp = rd("respiration")
    if resp is not None:
        dt = np.diff(resp.timestamp)
        print(f"  respiration (impedance): dt counts {pd.Series(dt).value_counts().head(3).to_dict()}, gaps>1s={(dt > 1000).sum()}")
    rrate = rd("respiration-rate")
    if rrate is not None:
        dt = np.diff(rrate.timestamp) / 1000
        print(f"  respiration-rate: n={len(rrate)}, cadence median={np.median(dt):.1f}s, "
              f"value median={rrate.value.median()} range=[{rrate.value.min()},{rrate.value.max()}] "
              f"(values look like breaths/min)")
    bp = rd("body-position")
    if bp is not None:
        print(f"  body-position: {bp.position.value_counts().to_dict()}")
    temp = rd("temperature")
    if temp is not None:
        print(f"  temperature: median={temp.value.median():.2f}C range=[{temp.value.min():.2f},{temp.value.max():.2f}] "
              f"cadence={np.median(np.diff(temp.timestamp)) / 1000:.1f}s")
