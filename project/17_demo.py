"""
Step 17 - the demo. Upload one chest X-ray and see what every normality score
in the study would decide about it. A research demonstration, not a clinical
tool: the thresholds were chosen on ChestX-ray14 and do not carry over to
another hospital's images.

  python 17_demo.py                  serve on http://127.0.0.1:8765 and open the browser
  python 17_demo.py --port 9000 --no-browser
  python 17_demo.py --cpu            stay on the CPU even when a GPU is available
  python 17_demo.py --check 200      run 200 test images through the demo and compare
                                     every number with the stored research outputs

Run 16_demo_build.py once first; it writes out/demo_models.pkl. Needs no new
packages: the page is plain HTML served by the standard library.

For each image the server runs all four checkpoints the way 03_extract.py and
09_fix_opnorm.py ran the dataset, scores the seven constructions with the
components 13_analysis.py fitted, and applies every operating threshold the
study uses. When the file name is a ChestX-ray14 image index, the page also
shows that image's split, its report labels and the stored research scores
next to the live ones.
"""
import argparse
import base64
import json
import math
import pickle
import re
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchxrayvision as xrv

import demo_core as dc
from cxrcfg import OUT, CACHE, IMAGES, SEED

CHECKPOINT_INFO = {
    "pc": {"name": "PadChest", "role": "Headline checkpoint"},
    "chex": {"name": "CheXpert", "role": "Primary checkpoint"},
    "mimic_ch": {"name": "MIMIC-CXR", "role": "Primary checkpoint"},
    "nih": {"name": "NIH ChestX-ray14", "role": "Trained on this dataset. Shown as a leakage contrast and left out of every count."},
}

CONSTRUCTION_INFO = {
    "A": {"formula": "1 − max p", "name": "Fréchet upper bound",
          "note": "The published construction (Schalekamp et al., 2024). It is the most optimistic joint probability of no finding that the seven marginals allow."},
    "B": {"formula": "∏ (1 − p)", "name": "Product",
          "note": "Correct only if the seven findings are conditionally independent given the image. The natural alternative to A."},
    "A_cal": {"formula": "1 − max q", "name": "Calibrated bound",
              "note": "A computed on isotonic-calibrated marginals q, fitted on the train split."},
    "B_cal": {"formula": "∏ (1 − q)", "name": "Calibrated product",
              "note": "B on calibrated marginals. Its calibration error drops to the level of the learned linear combiner."},
    "C": {"formula": "logistic(7 logits)", "name": "Learned linear",
          "note": "Logistic regression on the seven standardised logits, fitted on train."},
    "C++": {"formula": "MLP(7 logits)", "name": "Learned MLP",
            "note": "A small neural network on the same seven logits, fitted on train."},
    "D": {"formula": "logistic(1024 features)", "name": "Direct classifier",
          "note": "Logistic regression on the frozen image features. It never reads the per-finding probabilities."},
}

PROCEDURE_INFO = {
    "empirical": {"name": "Empirical target", "short": "Empirical",
                  "note": "The most permissive threshold whose NPV on the calibration split reaches the target. It is a point estimate with no guarantee, and on fresh data it misses the target about half the time."},
    "bonferroni": {"name": "Learn-Then-Test, Bonferroni", "short": "LTT Bonferroni",
                   "note": "200 candidate thresholds, each tested with an exact Clopper-Pearson bound at 0.05/200. This is the formal finite-sample guarantee, valid if calibration images are independent; one patient can contribute many, so its nominal confidence is optimistic. The study ran it on five constructions."},
    "fixed_sequence": {"name": "Learn-Then-Test, fixed sequence", "short": "LTT fixed-sequence",
                       "note": "Tests thresholds from the 0.90 train quantile toward 0.50 and stops at the first failure. A sensitivity analysis: its starting point was chosen after looking at these data. The study ran it on five constructions."},
}

NIH_NAME = re.compile(r"^\d{8}_\d{3}\.png$", re.IGNORECASE)
MAX_UPLOAD = 30 * 1024 * 1024


def finite(x):
    """JSON has no NaN or infinity; send null instead."""
    if isinstance(x, dict):
        return {str(k): finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Engine:
    """Loads the checkpoints and the fitted components once, then scores images."""

    def __init__(self, device):
        if not dc.BUNDLE.exists():
            raise SystemExit(f"{dc.BUNDLE} is missing. Run `python 16_demo_build.py` first (a few minutes).")
        with open(dc.BUNDLE, "rb") as f:
            self.bundle = pickle.load(f)
        b = self.bundle
        have = {"sklearn": sklearn.__version__, "numpy": np.__version__, "torch": torch.__version__,
                "torchxrayvision": xrv.__version__}
        for k, v in b["versions"].items():
            if have.get(k) != v:
                print(f"  [warn] {k} {have.get(k)} here, the bundle was built with {v}; numbers may drift")
        self.device = device
        self.lock = threading.Lock()
        self.models, self.th = {}, {}
        for key in dc.CHECKPOINTS:
            ck = b["checkpoints"][key]
            model = dc.load_checkpoint(key, device)
            if list(model.pathologies) != ck["pathologies"]:
                raise SystemExit(f"{key}: the library's label order differs from labels_{key}.json")
            th = dc.op_threshs(key)
            if not np.array_equal(th, np.asarray(ck["op_threshs"]), equal_nan=True):
                raise SystemExit(f"{key}: op_norm thresholds differ from the ones step 9 inverted with")
            self.models[key], self.th[key] = model, th
        self.lookup = Lookup()

    # ------------------------------------------------------------ static description
    def meta(self):
        b = self.bundle
        cks = []
        for key in dc.CHECKPOINTS:
            ck = b["checkpoints"][key]
            rules = {}
            for nm in b["main_cons"]:
                rules[nm] = {}
                for proc, by_t in ck["rules"][nm].items():
                    rules[nm][proc] = {}
                    for t, th in by_t.items():
                        cov, npv, n = ck["test"][nm][proc][t]
                        rules[nm][proc][str(t)] = {
                            "threshold": th, "coverage": cov, "npv": npv, "n": n,
                            "holds": bool(n > 0 and npv >= t)}
            cks.append({"key": key, **CHECKPOINT_INFO[key], "weights": ck["weights"],
                        "trained": ck["trained_nih14"], "test_images": ck["test_images"],
                        "test_normal_rate": ck["test_normal_rate"], "rules": rules,
                        "checks": ck["checks"]})
        return finite({
            "checkpoints": cks,
            "constructions": [{"key": nm, **CONSTRUCTION_INFO[nm], "risk_control": nm in b["rc_cons"]}
                              for nm in b["main_cons"]],
            "procedures": [{"key": p, **PROCEDURE_INFO[p],
                            "targets": [str(t) for t in (b["naive_targets"] if p == "empirical" else b["ctrl_targets"])]}
                           for p in dc.PROCEDURES],
            "common7": b["common7"], "nih14": b["nih14"],
            "samples": b["samples"], "device": str(self.device), "built": b["built"],
            "versions": b["versions"],
        })

    # ------------------------------------------------------------ one image
    def analyze(self, data, filename):
        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            raise ValueError("Use a PNG, JPEG, BMP, TIFF or WebP image. DICOM files need converting first.")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f"upload{suffix}"
            path.write_bytes(data)
            img = dc.load_one(path)
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("The file could not be read as an image.")

        b = self.bundle
        t0 = time.time()
        out = {}
        with self.lock:
            for key in dc.CHECKPOINTS:
                ck = b["checkpoints"][key]
                P16, Z16 = dc.forward(self.models[key], img, self.device)
                R = dc.corrected(P16, self.th[key])
                S, p, q = dc.score_image(ck["comp"], R, Z16)
                out[key] = self.describe(key, P16, R, S, p, q)
        ms = (time.time() - t0) * 1000

        ok, png = cv2.imencode(".png", img)
        res = {"file": Path(filename).name, "original_size": [int(raw.shape[1]), int(raw.shape[0])],
               "model_input": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
               "results": out, "inference_ms": round(ms), "device": str(self.device)}
        known = self.lookup.find(Path(filename).name)
        if known is not None:
            res["known"] = self.compare(known, out)
        return finite(res)

    def describe(self, key, P16, R, S, p, q):
        b = self.bundle
        ck = b["checkpoints"][key]
        paths = ck["pathologies"]
        c7 = b["common7"]
        findings = []
        for lab in b["nih14"]:
            j = paths.index(lab) if lab in paths else None
            trained = lab in ck["trained_nih14"]
            findings.append({
                "label": lab, "trained": trained, "in_score": lab in c7,
                "prob": float(R[j]) if (j is not None and trained) else None,
                "library_output": float(P16[j]) if (j is not None and trained) else None,
                "op_threshold": float(self.th[key][j]) if (j is not None and trained) else None,
                "calibrated": float(q[c7.index(lab)]) if lab in c7 else None,
            })
        return {
            "findings": findings,
            "scores": S,
            "drivers": {"A": c7[int(np.argmax(p))], "A_cal": c7[int(np.argmax(q))]},
            "verdicts": {nm: {proc: {str(t): v for t, v in by_t.items()}
                              for proc, by_t in dc.verdicts(S[nm], ck["rules"][nm]).items()}
                         for nm in b["main_cons"]},
        }

    def compare(self, row, live):
        """Stored research values for a known ChestX-ray14 image, next to the live ones."""
        b = self.bundle
        lk = self.lookup
        info = {"image_index": lk.sp["Image Index"].iloc[row], "patient_id": int(lk.sp["patient_id"].iloc[row]),
                "split": lk.sp["split"].iloc[row],
                "findings": [l for l in b["nih14"] if lk.sp[l].iloc[row] == 1],
                "per_checkpoint": {}}
        info["normal7"] = not any(lk.sp[l].iloc[row] == 1 for l in b["common7"])
        for key in dc.CHECKPOINTS:
            ck = b["checkpoints"][key]
            c = {}
            R = lk.probsraw(key)
            if R is not None:
                stored = R[row].astype(np.float32)
                diffs = [abs(f["prob"] - float(stored[ck["pathologies"].index(f["label"])]))
                         for f in live[key]["findings"] if f["prob"] is not None]
                c["max_prob_diff"] = max(diffs) if diffs else None
            S = lk.scores(key)
            if S is not None:
                st = {nm: float(S[nm][row]) for nm in b["main_cons"]}
                c["stored_scores"] = st
                c["max_score_diff"] = max(abs(st[nm] - live[key]["scores"][nm]) for nm in b["main_cons"])
                agree = total = 0
                for nm in b["main_cons"]:
                    sv = dc.verdicts(st[nm], ck["rules"][nm])
                    for proc, by_t in sv.items():
                        for t, v in by_t.items():
                            total += 1
                            agree += v == live[key]["verdicts"][nm][proc][str(t)]
                c["verdicts_agree"], c["verdict_cells"] = agree, total
            info["per_checkpoint"][key] = c
        return info


class Lookup:
    """Stored research outputs, read on demand, for images named by their
    ChestX-ray14 image index. Each piece is optional: the score caches, for
    example, are not in the repository."""

    def __init__(self):
        self.sp, self.row, self._R, self._S = None, {}, {}, {}
        splits = OUT / "splits.csv"
        if splits.exists() and (OUT / "valid_idx.npy").exists():
            idx = np.load(OUT / "valid_idx.npy")
            self.sp = pd.read_csv(splits).iloc[idx].reset_index(drop=True)
            self.row = {n.lower(): i for i, n in enumerate(self.sp["Image Index"])}

    def find(self, name):
        if self.sp is None or not NIH_NAME.match(name):
            return None
        return self.row.get(name.lower())

    def probsraw(self, key):
        if key not in self._R:
            f = OUT / f"probsraw_{key}.npy"
            self._R[key] = np.load(f, mmap_mode="r") if f.exists() else None
        return self._R[key]

    def scores(self, key):
        if key not in self._S:
            f = OUT / f"rev3_scores_{key}_common7.npz"
            self._S[key] = dict(np.load(f, allow_pickle=True)) if f.exists() else None
        return self._S[key]


# ---------------------------------------------------------------- web server
PAGE = dc.DEMO_DIR / "index.html"


def make_handler(engine):
    meta_json = json.dumps(engine.meta()).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "CXRNormalityDemo"

        def log_message(self, fmt, *args):
            if not self.path.startswith(("/samples/", "/api/meta")):
                sys.stderr.write("  %s %s\n" % (self.command, unquote(self.path)))

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code, obj):
            self.send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                return self.send(200, meta_json, "application/json")
            if url.path.startswith("/samples/"):
                name = Path(unquote(url.path)).name
                f = dc.SAMPLES_DIR / name
                if f.parent == dc.SAMPLES_DIR and f.exists():
                    return self.send(200, f.read_bytes(), "image/png")
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/analyze":
                return self.send_json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_UPLOAD:
                return self.send_json(400, {"error": "Send one image of at most 30 MB."})
            data = self.rfile.read(n)
            name = parse_qs(url.query).get("name", ["upload.png"])[0]
            try:
                self.send_json(200, engine.analyze(data, name))
            except ValueError as e:
                self.send_json(400, {"error": str(e)})
            except Exception as e:          # keep the page alive during a demo
                self.send_json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


# ---------------------------------------------------------------- self-check
def check(engine, n):
    """Run n test-split images through Engine.analyze, the same call the page
    makes, and compare every number it returns with the stored outputs."""
    lk = engine.lookup
    te = np.flatnonzero(lk.sp["split"].values == "test")
    rows = np.random.default_rng(SEED + 1).choice(te, n, replace=False)
    cache = np.load(CACHE, mmap_mode="r") if CACHE.exists() else None
    stats = {k: {"pix": 0, "lib": 0.0, "prob": 0.0, "score": 0.0, "agree": 0, "cells": 0, "flips": []}
             for k in dc.CHECKPOINTS}
    t0 = time.time()
    for i in rows:
        name = lk.sp["Image Index"].iloc[i]
        res = engine.analyze((IMAGES / name).read_bytes(), name)
        if cache is not None:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(res["model_input"].split(",")[1]), np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            pix_same = np.array_equal(img, cache[i])
        for key in dc.CHECKPOINTS:
            s, live = stats[key], res["results"][key]
            P = np.load(OUT / f"probs_{key}.npy", mmap_mode="r")[i].astype(np.float32)
            paths = engine.bundle["checkpoints"][key]["pathologies"]
            for f in live["findings"]:
                if f["prob"] is None:
                    continue
                j = paths.index(f["label"])
                s["lib"] = max(s["lib"], abs(f["library_output"] - float(P[j])))
            c = res["known"]["per_checkpoint"][key]
            s["prob"] = max(s["prob"], c["max_prob_diff"])
            s["score"] = max(s["score"], c["max_score_diff"])
            s["agree"] += c["verdicts_agree"]
            s["cells"] += c["verdict_cells"]
            if c["verdicts_agree"] < c["verdict_cells"]:
                s["flips"].append(name)
            s["pix"] += int(cache is not None and pix_same)
    dt = time.time() - t0
    print(f"\n{n} test images through the demo in {dt:.0f}s ({dt / n * 1000:.0f} ms per image, four checkpoints, {engine.device})")
    print(f"{'checkpoint':10s} {'input=cache':>11s} {'max|lib out|':>13s} {'max|prob|':>10s} {'max|score|':>11s} {'verdicts agree':>16s}")
    for key, s in stats.items():
        print(f"{key:10s} {s['pix']:>5d}/{n:<5d} {s['lib']:13.2e} {s['prob']:10.2e} {s['score']:11.2e} "
              f"{s['agree']:>8d}/{s['cells']:<7d}" + (f"  flips: {', '.join(s['flips'][:5])}" if s["flips"] else ""))


def main():
    dc.utf8_console()
    ap = argparse.ArgumentParser(description="CXR-Normality demo")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cpu", action="store_true", help="stay on the CPU even when a GPU is available")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="compare N test images with the stored outputs, then exit")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    print(f"loading four checkpoints on {device} ...")
    engine = Engine(device)
    if a.check:
        return check(engine, a.check)

    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(engine))
    url = f"http://{a.host}:{a.port}/"
    print(f"demo running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
