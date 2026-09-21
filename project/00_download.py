"""
Step 0 - fetch ChestX-ray14 via the Kaggle CLI.

The NIH Box links have rotted (404 as of September 2026), so Kaggle is the
reliable route. Two mirrors:

  --resized   about 2.5 GB  khanfashee/nih-chest-x-ray-14-224x224-resized   (use this one)
  --full      about 45 GB   nih-chest-xrays/data

The pipeline resizes to 224 px anyway, so the resized mirror is the right
choice. One limitation: a third party did the resize with an unknown filter.
It does not bear on the calibration question.

One-time setup, about two minutes:
  1. kaggle.com, your avatar, Settings, API, "Create New API Token"
  2. save the downloaded kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json
  3. pip install kaggle

  python 00_download.py --resized
"""
import argparse, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from cxrcfg import DATA, IMAGES

RESIZED = "khanfashee/nih-chest-x-ray-14-224x224-resized"
FULL = "nih-chest-xrays/data"


def have_kaggle():
    """True if the Kaggle API token and CLI are both present."""
    token = Path(os.environ.get("USERPROFILE", Path.home())) / ".kaggle" / "kaggle.json"
    if not token.exists():
        print(f"[!] No API token at {token}")
        print("    kaggle.com -> Settings -> API -> Create New API Token, save it there.")
        return False
    if shutil.which("kaggle") is None:
        print("[!] kaggle CLI not on PATH.  pip install kaggle")
        return False
    return True


def download(slug, dest):
    """Download one Kaggle dataset into dest and unzip it in place."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading {slug} -> {dest}")
    r = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)])
    if r.returncode != 0:
        print("[!] kaggle download failed - check the token and the dataset slug")
        return False
    for z in dest.glob("*.zip"):
        print(f"unzipping {z.name} ({z.stat().st_size/1e9:.1f} GB) ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        z.unlink()
    return True


def tidy():
    """Flatten whatever folder layout the mirror used into data/images/*.png."""
    IMAGES.mkdir(exist_ok=True)
    moved = 0
    for p in DATA.rglob("*.png"):
        if p.parent == IMAGES:
            continue
        target = IMAGES / p.name
        if not target.exists():
            p.rename(target); moved += 1
    if moved:
        print(f"moved {moved:,} images into {IMAGES}")

    csv = DATA / "Data_Entry_2017.csv"
    if not csv.exists():
        for c in DATA.rglob("Data_Entry_2017*.csv"):
            shutil.copy(c, csv); break
    n = len(list(IMAGES.glob("*.png")))
    print(f"\nimages: {n:,}   (expect 112,120)")
    print(f"label CSV: {'OK' if csv.exists() else 'MISSING - grab Data_Entry_2017.csv manually'}")
    if n and csv.exists():
        print("\nready -> python 01_make_splits.py")


def main():
    """Parse --resized / --full / --tidy-only and run the matching step."""
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resized", action="store_true", help="about 2.5 GB, 224x224 (recommended)")
    g.add_argument("--full", action="store_true", help="~45 GB, original 1024x1024")
    ap.add_argument("--tidy-only", action="store_true")
    a = ap.parse_args()

    if a.tidy_only:
        return tidy()
    if not have_kaggle():
        return 1

    if a.full:
        # The full-resolution set uses the same filenames as the resized set.
        # Keep it in a separate tree or the two overwrite each other.
        dest = DATA.parent / "data_full"
        print(f"full-resolution set -> {dest}  (kept separate from the 224px set)")
        if download(FULL, dest):
            n = len(list(dest.rglob("*.png")))
            print(f"\nfull-res images: {n:,}  (expect 112,120)")
            print("This is backup/verification only. The pipeline uses data/ (224px).")
        return

    if download(RESIZED, DATA):
        tidy()


if __name__ == "__main__":
    sys.exit(main())
