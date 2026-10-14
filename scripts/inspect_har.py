"""Phase 1: inspect AIDLAB-HAR v3 EDF/CSV structure (channels, rates, durations, labels)."""
import sys, glob, os, collections
import pyedflib, pandas as pd

root = sys.argv[1]
edfs = sorted(glob.glob(os.path.join(root, "data", "*.edf")))
rows = []
for p in edfs:
    f = pyedflib.EdfReader(p)
    labels = f.getSignalLabels()
    rates = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
    dims = [f.getPhysicalDimension(i) for i in range(f.signals_in_file)]
    rows.append(dict(file=os.path.basename(p), n_ch=f.signals_in_file, labels="|".join(labels),
                     rates="|".join(map(str, sorted(set(rates)))), dims="|".join(sorted(set(dims))),
                     duration_s=f.getFileDuration(), n_samples=f.getNSamples()[0]))
    f.close()
df = pd.DataFrame(rows)
df["activity"] = df.file.str.extract(r"SUB\d+_([A-Z]+)_")
df["subj"] = df.file.str.extract(r"(SUB\d+)")
print("EDF files:", len(df))
print("Distinct channel sets:", df.labels.unique())
print("Distinct sampling rates:", df.rates.unique(), "| dims:", df.dims.unique())
print("\nDuration (s) per activity:\n", df.groupby("activity").duration_s.describe()[["count","mean","min","max"]].round(1))
print("\nSubjects per activity:\n", df.groupby("activity").subj.nunique())
print("Total EDF duration (min):", round(df.duration_s.sum()/60, 1))

csvs = sorted(glob.glob(os.path.join(root, "data", "*.csv")))
print("\nAnnotation CSVs:", len(csvs))
print(open(csvs[0]).read()[:600])
ev = collections.Counter()
for c in csvs: ev.update(pd.read_csv(c)["event"])
print("Event counts:", dict(ev))
edf_stems = {os.path.splitext(os.path.basename(x))[0] for x in edfs}
csv_stems = {os.path.splitext(os.path.basename(x))[0] for x in csvs}
print("EDF without CSV:", len(edf_stems - csv_stems), sorted(edf_stems - csv_stems)[:12])
q = pd.read_csv(os.path.join(root, "quality_intervals.csv"))
print("\nquality_intervals.csv:", q.shape, list(q.columns)); print(q.head())
