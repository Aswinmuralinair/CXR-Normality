"""
Step 19 - how much patient clustering matters for the risk-control bounds.

Both Learn-Then-Test procedures bound the abnormal rate among the calibration
images a threshold clears with an exact binomial (Clopper-Pearson) bound,
which treats those images as independent. One patient can contribute many
images, and images of one patient tend to share their label, so the abnormal
rate varies more than the binomial assumes. This step measures by how much.

For every threshold the study uses (computed with 13_analysis.py's own
functions, as 16_demo_build.py does), it takes the calibration images the
threshold clears and compares the variance of their abnormal rate under
patient clustering (the standard ratio-estimator variance with patients as
clusters) with the binomial variance. The ratio is the design effect: 1 means
the independence assumption costs nothing, 2 means the cleared set carries
the information of half as many independent images.

  python 19_design_effect.py    writes out/design_effect.csv; needs the step 13 score caches

It only measures and does not correct the bounds. 20_guarantee_variants.py
tries two patient-level versions.
"""
import numpy as np
import pandas as pd

import demo_core as dc
from cxrcfg import OUT

A13 = dc.analysis()


def design_effect(abnormal, patient):
    """Cluster variance of the abnormal rate over its binomial variance."""
    n = len(abnormal)
    p = abnormal.mean()
    if n < 30 or p in (0.0, 1.0):
        return np.nan
    g = pd.DataFrame({"y": abnormal, "g": patient}).groupby("g")["y"].agg(["sum", "count"])
    k = len(g)
    v_cluster = k / (k - 1) * ((g["sum"] - p * g["count"]) ** 2).sum() / n ** 2
    return v_cluster / (p * (1 - p) / n)


idx = np.load(OUT / "valid_idx.npy")
sp = pd.read_csv(OUT / "splits.csv").iloc[idx].reset_index(drop=True)
tr, ca = (sp["split"] == "train").values, (sp["split"] == "calib").values
normal = (sp[A13.COMMON7].sum(axis=1) == 0).values.astype(int)
pid = sp["patient_id"].values

rows = []
for key in dc.CHECKPOINTS:
    S = np.load(OUT / f"rev3_scores_{key}_common7.npz")
    for nm in A13.RC_CONS:
        s = S[nm]
        grid = np.quantile(s[tr], np.linspace(0.50, 0.9995, A13.GRID))
        seq = np.quantile(s[tr], np.linspace(A13.FST_START_Q, 0.50, A13.GRID))
        rules = {"empirical": {t: A13.naive_t(normal[ca], s[ca], t) for t in A13.CTRL_TARGETS},
                 "bonferroni": {t: A13.bonf_t(normal[ca], s[ca], t, grid) for t in A13.CTRL_TARGETS},
                 "fixed_sequence": {t: A13.fst_t(normal[ca], s[ca], t, seq) for t in A13.CTRL_TARGETS}}
        for proc, by_t in rules.items():
            for t, th in by_t.items():
                cleared = ca & (s >= th)
                n = int(cleared.sum())
                de = design_effect(1 - normal[cleared], pid[cleared]) if n else np.nan
                rows.append({"model": key, "construction": nm, "procedure": proc, "target": t,
                             "threshold": th, "n_cleared": n, "n_patients": int(len(np.unique(pid[cleared]))),
                             "abnormal_rate": float(1 - normal[cleared].mean()) if n else np.nan,
                             "design_effect": de, "effective_n": n / de if n and np.isfinite(de) else np.nan})

D = pd.DataFrame(rows)
D.to_csv(OUT / "design_effect.csv", index=False)
prim = D[D.model.isin(["pc", "chex", "mimic_ch"]) & (D.target == 0.95) & D.design_effect.notna()]
for proc in ["bonferroni", "fixed_sequence", "empirical"]:
    x = prim[prim.procedure == proc].design_effect
    print(f"{proc:15s} at NPV 0.95, primary backbones: design effect {x.min():.2f} to {x.max():.2f} over {len(x)} thresholds that clear anything")
print("wrote", OUT / "design_effect.csv")
