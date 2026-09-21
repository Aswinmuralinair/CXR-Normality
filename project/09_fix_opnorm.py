"""
Step 9 - undo torchxrayvision's op_norm warp. Mandatory before any analysis.

torchxrayvision's DenseNet.forward applies op_norm() whenever pretrained
weights are loaded. op_norm maps each label's operating threshold to 0.5, so
model(x) returns a warped score rather than a probability. PadChest's
pneumothorax threshold is 0.00126, so a raw probability of 0.13% comes out as
0.5.

What the warp does to each construction:
  A = 1 - max(p)      the max runs over labels warped by different amounts,
                      so even the ranking is not preserved
  B = prod(1 - p)     a product of non-probabilities; the independence
                      reading no longer applies
  C = learned(logit)  inputs warped; a linear model cannot fully undo it
  D = features        untouched, it never reads p

op_norm is invertible, so the raw sigmoid outputs can be recovered exactly:

  out < 0.5 :  raw = out * 2 * t
  out >= 0.5:  raw = 1 - (1 - out) * 2 * (1 - t)

Writes out/probsraw_<model>.npy (float16), out/op_threshs.json (the exact
thresholds and library version used for the inversion) and out/opnorm_fix.csv, a before and
after comparison of the three derived constructions on the test split. The
comparison is a sanity check for this step, not a reported result; the
reported analysis is 13_analysis.py.
"""
import json
import numpy as np
import pandas as pd
import torchxrayvision as xrv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from cxrcfg import OUT, MODELS, PRIMARY_MODELS, LEAKY_MODEL, SEED


def ece(y, s, bins=15):
    """Expected calibration error, 15 equal-width bins."""
    e = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(s, e) - 1, 0, bins - 1)
    return sum((b == k).mean() * abs(y[b == k].mean() - s[b == k].mean())
               for k in range(bins) if (b == k).any())


def fcr(y, s, cov=0.05):
    """False-clearance rate in the top 5% of scores."""
    n = max(1, int(round(cov * len(s))))
    return float(1 - y[np.argsort(-s)[:n]].mean())


def invert_op_norm(P, th):
    """Invert op_norm column by column. A label with a NaN threshold was never
    trained; its column stays NaN so nothing downstream can use it by accident."""
    R = np.full_like(P, np.nan, dtype=np.float32)
    for j in range(P.shape[1]):
        t = th[j]
        if not np.isfinite(t):
            continue
        c = P[:, j].astype(np.float32)
        lo = c < 0.5
        R[lo, j] = c[lo] * 2.0 * t
        R[~lo, j] = 1.0 - (1.0 - c[~lo]) * 2.0 * (1.0 - t)
    return np.clip(R, 1e-7, 1 - 1e-7)


rows = []
saved_thresholds = {}     # written to out/op_threshs.json so the inversion is reproducible
for key in list(PRIMARY_MODELS) + [LEAKY_MODEL]:
    lf = OUT / f"labels_{key}.json"
    if not lf.exists():
        continue
    meta = json.loads(lf.read_text())
    paths, usable = meta["pathologies"], meta["usable_nih14"]
    th = np.asarray(xrv.models.model_urls[MODELS[key]]["op_threshs"], dtype=np.float64)

    saved_thresholds[key] = {"weights": MODELS[key], "torchxrayvision": xrv.__version__,
                             "op_threshs": dict(zip(paths, [None if not np.isfinite(t) else float(t) for t in th]))}
    P = np.load(OUT / f"probs_{key}.npy").astype(np.float32)
    R = invert_op_norm(P, th)
    np.save(OUT / f"probsraw_{key}.npy", R.astype(np.float16))

    idx = np.load(OUT / "valid_idx.npy")
    sp = pd.read_csv(OUT / "splits.csv").iloc[idx].reset_index(drop=True)
    cols = [paths.index(u) for u in usable]
    y = (sp[usable].sum(axis=1) == 0).astype(int).values
    tr = (sp["split"] == "train").values
    ca = (sp["split"] == "calib").values
    te = (sp["split"] == "test").values

    print("\n" + "=" * 92)
    print(f"{key}   {len(usable)} labels")
    print(f"  op_threshs for used labels: "
          f"{', '.join(f'{u}={th[paths.index(u)]:.5f}' for u in usable[:5])} ...")
    print("=" * 92)
    print(f"  {'construction':14s} {'':>7s} {'AUC':>8s} {'ECE':>8s} {'FCR@5%':>8s} "
          f"{'n>=0.95':>9s}")

    # Before and after, on the test split. C is fitted on train here, like
    # every learned construction in the main analysis.
    for tag, M in [("as run (op_norm)", P), ("corrected (raw p)", R)]:
        p = np.clip(M[:, cols].astype(np.float32), 1e-6, 1 - 1e-6)
        A = 1.0 - p.max(axis=1)
        B = np.exp(np.log1p(-p).sum(axis=1))
        lg = np.log(p / (1 - p))
        fC = LogisticRegression(max_iter=3000).fit(lg[tr], y[tr])
        C = fC.predict_proba(lg)[:, 1]
        for nm, s_all in [("A_max", A), ("B_product", B), ("C_learned", C)]:
            s, yt = s_all[te], y[te]
            r = {"model": key, "version": tag, "construction": nm,
                 "auc": roc_auc_score(yt, s), "ece": ece(yt, s), "fcr05": fcr(yt, s),
                 "n_ge_095": int((s >= 0.95).sum())}
            rows.append(r)
            lbl = f"{nm}" if tag.startswith("as run") else ""
            print(f"  {lbl:14s} {tag.split()[0][:7]:>7s} {r['auc']:8.4f} {r['ece']:8.4f} "
                  f"{r['fcr05']:8.4f} {r['n_ge_095']:9d}")
        print()

df = pd.DataFrame(rows)
df.to_csv(OUT / "opnorm_fix.csv", index=False)
(OUT / "op_threshs.json").write_text(json.dumps(saved_thresholds, indent=1))

print("=" * 92)
print("Change from as-run to corrected")
print("=" * 92)
for m in df.model.unique():
    for c in ["A_max", "B_product", "C_learned"]:
        a = df[(df.model == m) & (df.construction == c) & (df.version.str.startswith("as run"))]
        b = df[(df.model == m) & (df.construction == c) & (df.version.str.startswith("corrected"))]
        if len(a) and len(b):
            a, b = a.iloc[0], b.iloc[0]
            print(f"  {m:9s} {c:11s} ECE {a.ece:.4f} -> {b.ece:.4f}   "
                  f"n>=0.95 {a.n_ge_095:6d} -> {b.n_ge_095:6d}   "
                  f"AUC {a.auc:.4f} -> {b.auc:.4f}")
print(f"\nwrote {OUT/'opnorm_fix.csv'} and out/probsraw_*.npy")
