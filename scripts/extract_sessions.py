"""Safely extract every downloaded session archive (data/raw/sessions/<id>/session.zip).

Skips archives that are incomplete/corrupt or already extracted; refuses unsafe paths.
"""
import sys, zipfile
from pathlib import Path

root = Path(sys.argv[1])
for z in sorted(root.glob("*/session.zip")):
    if (z.parent / "dataset.json").exists():
        continue
    try:
        zf = zipfile.ZipFile(z)
        names = zf.namelist()
        if any(n.startswith("/") or ".." in n or "/" in n.strip("/") for n in names):
            print("UNSAFE/NESTED paths, skipped:", z.parent.name, names); continue
        if zf.testzip() is not None:
            print("CORRUPT:", z.parent.name); continue
        zf.extractall(z.parent)
        print("extracted", z.parent.name, sorted(names))
    except zipfile.BadZipFile:
        print("incomplete (still downloading?):", z.parent.name)
