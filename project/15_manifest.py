"""
Step 15 - write out/RUN_MANIFEST.json: what the reported numbers were produced
with, so that someone rerunning the pipeline can tell whether they are looking
at the same experiment.

Records the Python and package versions, the GPU and CUDA build, the model
weight identifiers and the op_norm thresholds step 9 inverted with, the random
seed, and a SHA-256 of every input and output file the results depend on. The
split file's hash is the important one: if it differs, nothing downstream is
comparable.
"""
import hashlib
import json
import platform
import sys
from datetime import date

from cxrcfg import OUT, MODELS, SEED

FILES = ["splits.csv", "valid_idx.npy", "op_threshs.json", "opnorm_fix.csv", "_rev3_numbers.txt"]
FILES += [f"labels_{k}.json" for k in MODELS] + [f"probs_{k}.npy" for k in MODELS] + [f"probsraw_{k}.npy" for k in MODELS]
FILES += sorted(p.name for p in OUT.glob("rev3_*.csv")) + ["rev3_label_meta.json"]
# the outside check (step 18), the clustering measurement (step 19) and the demo's fitted bundle (step 16)
FILES += sorted(p.name for p in OUT.glob("outside_*.csv")) + ["design_effect.csv", "demo_models.pkl"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


manifest = {
    "written": str(date.today()),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": {m: version(m) for m in ["torch", "torchvision", "torchxrayvision", "numpy", "pandas",
                                          "scipy", "sklearn", "matplotlib", "cv2"]},
    "seed": SEED,
    "model_weights": MODELS,
    "dataset": "NIH ChestX-ray14, Kaggle mirror khanfashee/nih-chest-x-ray-14-224x224-resized (224 px)",
    "splits": "patient-level 60/15/25, see 01_make_splits.py",
    "files_sha256": {f: sha256(OUT / f) for f in FILES if (OUT / f).exists()},
}
try:
    import torch
    manifest["cuda"] = {"available": torch.cuda.is_available(), "build": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
except Exception:
    manifest["cuda"] = None

(OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
print("wrote", OUT / "RUN_MANIFEST.json", f"({len(manifest['files_sha256'])} file hashes)")
