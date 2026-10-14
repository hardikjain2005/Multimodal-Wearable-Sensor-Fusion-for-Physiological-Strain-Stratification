"""Phase 1: inspect the AIDLAB activity-session exports (sleep / yoga / squash / hike).

For every CSV stream reports: rows, columns, time span, inferred sampling rate,
gaps, duplicate rows/timestamps, missing values and value ranges.
Writes a tidy summary table to results/metrics/phase1_stream_inventory.csv.
"""
import sys, os, json, glob
import numpy as np
import pandas as pd

raw_dir, out_csv = sys.argv[1], sys.argv[2]   # raw_dir = data/raw/sessions
sessions = sorted(d for d in os.listdir(raw_dir) if os.path.exists(os.path.join(raw_dir, d, "dataset.json")))
rows = []
for s in sessions:
    sdir = os.path.join(raw_dir, s)
    meta = json.load(open(os.path.join(sdir, "dataset.json")))
    t_from = pd.Timestamp(meta["from"]); t_to = pd.Timestamp(meta["to"])
    print(f"\n################ {s}  user={meta['user']}  {t_from} -> {t_to}  "
          f"({(t_to - t_from).total_seconds() / 60:.1f} min)")
    for p in sorted(glob.glob(os.path.join(sdir, "*.csv"))):
        name = os.path.basename(p)[:-4]
        df = pd.read_csv(p)
        ts = df["timestamp"]
        dt = np.diff(ts.values)
        val_cols = [c for c in df.columns if c != "timestamp"]
        num = df[val_cols].select_dtypes("number")
        med_dt = float(np.median(dt)) if len(dt) else np.nan
        r = dict(session=s, stream=name, columns="|".join(df.columns), rows=len(df),
                 start=pd.to_datetime(ts.min(), unit="ms", utc=True),
                 end=pd.to_datetime(ts.max(), unit="ms", utc=True),
                 span_min=(ts.max() - ts.min()) / 60000,
                 median_dt_ms=med_dt, approx_hz=1000 / med_dt if med_dt else np.nan,
                 p05_dt_ms=np.percentile(dt, 5) if len(dt) else np.nan,
                 p95_dt_ms=np.percentile(dt, 95) if len(dt) else np.nan,
                 max_gap_s=dt.max() / 1000 if len(dt) else np.nan,
                 n_gaps_gt_5s=int((dt > 5000).sum()),
                 non_monotonic=int((dt < 0).sum()),
                 dup_rows=int(df.duplicated().sum()),
                 dup_timestamps=int(ts.duplicated().sum()),
                 missing_cells=int(df.isna().sum().sum()),
                 value_min=float(num.min().min()) if not num.empty else np.nan,
                 value_median=float(num.stack().median()) if not num.empty else np.nan,
                 value_max=float(num.max().max()) if not num.empty else np.nan)
        rows.append(r)
        print(f"  {name:18s} rows={r['rows']:>9,} span={r['span_min']:7.1f}min "
              f"~{r['approx_hz']:8.2f}Hz dt[p5,med,p95]=[{r['p05_dt_ms']:.0f},{med_dt:.0f},{r['p95_dt_ms']:.0f}]ms "
              f"maxgap={r['max_gap_s']:.1f}s gaps>5s={r['n_gaps_gt_5s']} dupTS={r['dup_timestamps']} "
              f"dupRows={r['dup_rows']} NA={r['missing_cells']} "
              f"range=[{r['value_min']:.4g},{r['value_median']:.4g},{r['value_max']:.4g}]")
        # non-numeric columns (e.g. body-position label) — show categories
        for c in val_cols:
            if df[c].dtype == object:
                print(f"      categorical '{c}':", df[c].value_counts().to_dict())

inv = pd.DataFrame(rows)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
inv.to_csv(out_csv, index=False)
print("\nSensor availability matrix (rows per stream):")
print(inv.pivot_table(index="stream", columns="session", values="rows", aggfunc="sum")
      .reindex(columns=sessions).fillna(0).astype(int))
