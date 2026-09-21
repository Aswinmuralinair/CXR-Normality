"""
Unpack the Kaggle archive of ChestX-ray14 (224 px mirror) into data/.

    python extract_zip.py path\\to\\archive.zip

The CSVs and the official split lists go to data/, every PNG goes flat into
data/images/. Files that already exist are skipped, so the script can be
re-run after an interruption and it continues where it stopped.

Use this when 00_download.py cannot run (no Kaggle token, or the archive was
downloaded by hand in a browser).
"""
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

# The Kaggle CLI and some terminals choke on non-ASCII progress output on
# Windows; force UTF-8 with replacement so a stray character cannot abort a run.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "archive.zip"
DATA = Path(__file__).parent / "data"
IMG = DATA / "images"
IMG.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    raise SystemExit(f"archive not found: {SRC}\nusage: python extract_zip.py path\\to\\archive.zip")

z = zipfile.ZipFile(SRC)
names = z.namelist()
pngs = [n for n in names if n.lower().endswith(".png")]
others = [n for n in names if n.lower().endswith((".csv", ".txt"))]

# Label CSV and the official train/val and test lists first; they are small.
for n in others:
    with z.open(n) as f, open(DATA / Path(n).name, "wb") as o:
        shutil.copyfileobj(f, o)
    print("extracted", Path(n).name)

# Then the images, flattened into one folder regardless of the archive layout.
t0 = time.time()
for i, n in enumerate(pngs, 1):
    tgt = IMG / Path(n).name
    if not tgt.exists():
        with z.open(n) as f, open(tgt, "wb") as o:
            shutil.copyfileobj(f, o, 1 << 20)
    if i % 5000 == 0 or i == len(pngs):
        rate = i / max(time.time() - t0, 0.1)
        print(f"\r{i:,}/{len(pngs):,}  {rate:.0f} img/s  eta {(len(pngs) - i) / max(rate, 1):.0f}s", end="")
print(f"\ndone: {len(list(IMG.glob('*.png'))):,} images in {IMG}")
