"""
Step 18 - an outside check. Runs the demo's analysis on chest X-rays from a
hospital none of the four checkpoints was trained on, and compares how often
each operating threshold clears normal and abnormal studies there with the
same rates on the ChestX-ray14 test split.

  python 18_outside_check.py --download   pick and fetch the 340 X-rays once
  python 18_outside_check.py              score them and write the tables

Data: the Indiana University chest X-ray collection, released through Open-i
by the US National Library of Medicine (Demner-Fushman et al., JAMIA 2016),
read from the copy on Hugging Face (ykumards/open-i). Each exam has a
radiologist's report coded with MeSH terms. The script draws, with the
project seed and the rarest finding first, 20 exams for each of the seven
findings of the primary event (codes marked "borderline" left out, mild ones
kept) and 200 exams coded only "normal" whose impression says normal or no
acute disease. An exam with two of the seven findings is filed under the
rarer one. Exams without a frontal image are skipped. The images carry a
non-commercial licence, so they stay in "test images/", which git ignores.
The server behind the copy limits how often it answers, so the download goes
one request at a time and can be stopped and restarted.

Writes
  test images/<label>_NN.jpg, labels.csv, results.csv   per image, local only
  out/outside_rules.csv     per checkpoint, construction, procedure and target:
                            the threshold, the share of normal studies it clears
                            here and on ChestX-ray14 test, and the abnormal ones
  out/outside_auc.csv       AUC for normal against any of the seven findings
  out/outside_by_finding.csv  studies with each finding that are cleared, and the
                            checkpoint's median probability for that finding

This is a sanity check, not external validation. 340 selected studies cannot
measure the NPV a threshold delivers at that hospital, because NPV depends on
the case mix, and these studies were chosen by label, not taken in sequence.
"""
import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import demo_core as dc
from cxrcfg import ROOT, OUT, SEED

FOLDER = ROOT / "test images"
HF = "https://huggingface.co/datasets/ykumards/open-i/resolve/main/data/"
SHARDS = ["train-00000-of-00005-c80629f6027d8eb1.parquet", "train-00001-of-00005-07e6c158a3f56170.parquet",
          "train-00002-of-00005-653f05c8b15c22c6.parquet", "train-00003-of-00005-1d66a487a8489b7f.parquet",
          "train-00004-of-00005-208b7d689f1c7b70.parquet"]
ROWS = ("https://datasets-server.huggingface.co/rows?dataset=ykumards/open-i&config=default"
        "&split=train&offset={}&length=1")
N_NORMAL, N_EACH = 200, 20
# rarest first, so an exam with two findings is filed under the rarer one
SEVEN = [("pneumothorax", "Pneumothorax"), ("consolidation", "Consolidation"), ("pneumonia", "Pneumonia"),
         ("edema", "Pulmonary Edema"), ("effusion", "Pleural Effusion"), ("atelectasis", "Pulmonary Atelectasis"),
         ("cardiomegaly", "Cardiomegaly")]
NIH_NAME = {"pneumothorax": "Pneumothorax", "consolidation": "Consolidation", "pneumonia": "Pneumonia",
            "edema": "Edema", "effusion": "Effusion", "atelectasis": "Atelectasis", "cardiomegaly": "Cardiomegaly"}


# ------------------------------------------------------------------ download
class HttpRange(io.RawIOBase):
    """A remote file read with HTTP range requests, so pyarrow fetches only
    the parquet columns it is asked for and never the images."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.url, self.size = r.geturl(), int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def read_reports():
    import pyarrow.parquet as pq
    cols = ["uid", "MeSH", "Problems", "image", "indication", "findings", "impression"]
    parts = [pq.ParquetFile(io.BufferedReader(HttpRange(HF + s), buffer_size=1 << 20)).read(columns=cols).to_pandas()
             for s in SHARDS]
    return pd.concat(parts, ignore_index=True)


def fetch_row(i):
    for attempt in range(40):
        try:
            with urllib.request.urlopen(ROWS.format(i), timeout=120) as r:
                row = json.load(r)["rows"][0]["row"]
            time.sleep(2)
            return row
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After") or 0) or min(60 * (attempt + 1), 600)
            print(f"  server answered {e.code}, waiting {wait} s", flush=True)
            time.sleep(wait)
        except OSError as e:
            print(f"  {type(e).__name__}, retrying", flush=True)
            time.sleep(15 * (attempt + 1))
    raise SystemExit(f"row {i}: no answer after 40 tries; run the step again later")


def download():
    df = read_reports()
    split = lambda s: [] if pd.isna(s) else [x.strip() for x in str(s).split(";") if x.strip()]
    mesh, probs = df["MeSH"].map(split), df["Problems"].map(split)
    entries = lambda i, term: [m for m in mesh[i] if m.split("/")[0] == term]
    rng = np.random.default_rng(SEED)
    pools = []
    for lab, term in SEVEN:
        pool = [i for i in df.index if term in probs[i] and not any("borderline" in m for m in entries(i, term))]
        pools.append((lab, term, list(rng.permutation(pool))))
    normal = df.index[(df["Problems"].fillna("").str.strip() == "normal") & (df["MeSH"].fillna("").str.strip() == "normal")
                      & df["impression"].fillna("").str.contains(r"(?i)normal|no acute|negative|clear", regex=True)]
    pools.append(("normal", None, list(rng.permutation(list(normal)))))

    FOLDER.mkdir(exist_ok=True)
    progress = FOLDER / ".progress.json"
    seen = json.loads(progress.read_text()) if progress.exists() else {}    # row -> saved file, or "" if no frontal
    used, records = set(), []
    for lab, term, pool in pools:
        want, got = (N_NORMAL if lab == "normal" else N_EACH), 0
        for i in pool:
            if got == want:
                break
            if i in used:
                continue
            used.add(i)
            prev = seen.get(str(i))
            if prev == "":
                continue
            if prev and (FOLDER / prev).exists():
                got, name = got + 1, prev
                size = (FOLDER / prev).stat().st_size
            else:
                row = fetch_row(int(i))
                if int(row["uid"]) != int(df.at[i, "uid"]):
                    raise SystemExit(f"row {i}: the copy's row order changed; delete {progress} and start again")
                if not row.get("img_frontal"):
                    seen[str(i)] = ""
                    continue
                data = base64.b64decode(row["img_frontal"])
                got += 1
                name = f"{lab}_{got:03d}.jpg" if lab == "normal" else f"{lab}_{got:02d}.jpg"
                (FOLDER / name).write_bytes(data)
                size, seen[str(i)] = len(data), name
                progress.write_text(json.dumps(seen))
            others = [x for x, t in SEVEN if t in probs[i] and x != lab]
            records.append({"file": name, "label": lab, "severity": "; ".join(entries(i, term)) if term else "",
                            "also_has": ", ".join(others), "uid": int(df.at[i, "uid"]),
                            "problems": df.at[i, "Problems"], "mesh": df.at[i, "MeSH"],
                            "indication": df.at[i, "indication"], "impression": df.at[i, "impression"],
                            "findings": df.at[i, "findings"], "bytes": size})
        print(f"  {lab:14s} {got} images", flush=True)
    pd.DataFrame(records).to_csv(FOLDER / "labels.csv", index=False)
    print(f"wrote {len(records)} images and labels.csv to {FOLDER}")


# ------------------------------------------------------------------ check
def check():
    lab = pd.read_csv(FOLDER / "labels.csv")
    engine = dc.pipeline_defs("17_demo.py").Engine(torch.device("cpu"))
    b = engine.bundle
    t0 = time.time()
    res = {}
    for k, f in enumerate(lab["file"]):
        res[f] = engine.analyze((FOLDER / f).read_bytes(), f)["results"]
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} of {len(lab)} scored", flush=True)
    print(f"  {len(lab)} images scored in {time.time() - t0:.0f} s")

    cons, normal_iu = b["main_cons"], (lab["label"] == "normal").values
    rows = []
    for _, L in lab.iterrows():
        r, row = res[L["file"]], {"file": L["file"], "label": L["label"]}
        for ck in dc.CHECKPOINTS:
            for c in cons:
                row[f"{ck}_{c}"] = round(r[ck]["scores"][c], 5)
            for c in cons:
                for proc, by_t in r[ck]["verdicts"].get(c, {}).items():
                    for t, v in by_t.items():
                        row[f"{ck}_{c}_{proc}_{t}"] = "auto-clear" if v else "refer"
            top = max((f for f in r[ck]["findings"] if f["in_score"]), key=lambda f: f["prob"])
            row[f"{ck}_top_finding"] = top["label"]
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(FOLDER / "results.csv", index=False)

    an = dc.analysis()
    idx = np.load(OUT / "valid_idx.npy")
    sp = pd.read_csv(OUT / "splits.csv").iloc[idx].reset_index(drop=True)
    test = (sp["split"] == "test").values
    nih_normal = (sp[an.COMMON7].sum(axis=1) == 0).values
    rules_out, auc_out, find_out = [], [], []
    for ck in dc.CHECKPOINTS:
        z = np.load(OUT / f"rev3_scores_{ck}_common7.npz")
        rules = b["checkpoints"][ck]["rules"]
        for c in cons:
            s_iu, s_nih, y_nih = R[f"{ck}_{c}"].values, z[c][test], nih_normal[test]
            auc_out.append({"checkpoint": ck, "construction": c,
                            "auc_outside": roc_auc_score(normal_iu, s_iu), "auc_nih_test": roc_auc_score(y_nih, s_nih)})
            for proc, by_t in rules[c].items():
                for t, th in by_t.items():
                    # a rule with no admissible threshold is stored as +inf, so it clears nothing
                    clear_iu, clear_nih = s_iu >= th, s_nih >= th
                    rules_out.append({
                        "checkpoint": ck, "construction": c, "procedure": proc, "target": t, "threshold": th,
                        "outside_normal_n": int(normal_iu.sum()), "outside_normal_cleared": clear_iu[normal_iu].mean(),
                        "outside_abnormal_n": int((~normal_iu).sum()),
                        "outside_abnormal_cleared_n": int(clear_iu[~normal_iu].sum()),
                        "nih_test_normal_cleared": clear_nih[y_nih].mean(),
                        "nih_test_abnormal_cleared": clear_nih[~y_nih].mean()})
                    if t != 0.95:
                        continue
                    for l, col in NIH_NAME.items():
                        has = test & (sp[col] == 1).values
                        nih_clear = z[c][has] >= th
                        here = (lab["label"] == l).values
                        probs = [next(x["prob"] for x in res[f][ck]["findings"] if x["label"] == col)
                                 for f in lab.loc[here, "file"]]
                        find_out.append({"checkpoint": ck, "construction": c, "procedure": proc, "target": t,
                                         "finding": l, "outside_n": int(here.sum()),
                                         "outside_cleared_n": int(clear_iu[here].sum()),
                                         "outside_median_prob": float(np.median(probs)),
                                         "nih_test_n": int(has.sum()), "nih_test_cleared": nih_clear.mean()})
    pd.DataFrame(rules_out).to_csv(OUT / "outside_rules.csv", index=False)
    pd.DataFrame(auc_out).to_csv(OUT / "outside_auc.csv", index=False)
    pd.DataFrame(find_out).to_csv(OUT / "outside_by_finding.csv", index=False)

    ru = pd.DataFrame(rules_out)
    pc = ru[(ru.checkpoint == "pc") & (ru.procedure == "empirical") & (ru.target == 0.95)].set_index("construction")
    print(f"\n{normal_iu.sum()} normal and {(~normal_iu).sum()} abnormal outside studies")
    print("PadChest, empirical target at NPV 0.95: normal cleared outside / on ChestX-ray14 test, abnormal cleared outside")
    for c in ["A", "B_cal", "D"]:
        x = pc.loc[c]
        print(f"  {c:6s} {x.outside_normal_cleared:6.1%} / {x.nih_test_normal_cleared:6.1%}   "
              f"{x.outside_abnormal_cleared_n} of {x.outside_abnormal_n}")
    print("wrote out/outside_rules.csv, out/outside_auc.csv, out/outside_by_finding.csv and", FOLDER / "results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Outside check on the Indiana University chest X-ray collection")
    ap.add_argument("--download", action="store_true", help="pick and fetch the 340 X-rays first")
    a = ap.parse_args()
    if a.download:
        download()
    check()
