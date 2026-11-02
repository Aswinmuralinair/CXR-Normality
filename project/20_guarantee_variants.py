"""
Step 20 - variants of the Learn-Then-Test guarantee, from the cached scores.

Step 13 certifies thresholds with two fixed procedures. This step measures
what each design choice in them costs or buys, and checks the op_norm
inversion that every later step rests on. Nothing from step 13 is changed;
every rule below is recomputed with 13_analysis.py's own functions.

  inversion    the library's op_norm applied to our inverse (Eq. 2), over a
               dense grid of probabilities and over every stored output, for
               every trained finding of every model
  grid         Bonferroni over 200 candidates against 1,000, and the
               calibration NPV at the 0.50 train quantile, the most permissive
               candidate, to show whether the 50% cap ever binds
  sequence     fixed sequence started from five quantiles (0.95 to 0.70), each
               alone at alpha = 0.05 and all five together at 0.05 / 5
  patient      two patient-level versions of Bonferroni LTT: one calibration
               image per patient (a seeded draw), and n divided by the design
               effect of each candidate's cleared set
  multirisk    Bonferroni LTT that needs both NPV >= 0.95 and an upper bound
               on the share of pneumothorax studies cleared (5% or 2%)
  subgroups    test coverage and NPV of the accepted Bonferroni thresholds by
               sex, age band and view position
  calibrator   per-finding Platt scaling in place of isotonic regression for
               the calibrated product

  python 20_guarantee_variants.py    writes out/variants_*.csv; needs the step 13
                                     score caches and data/Data_Entry_2017.csv; CPU only
"""
import json

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import demo_core as dc
from cxrcfg import OUT, ROOT, MODELS, PRIMARY_MODELS, SEED

A13 = dc.analysis()
design_effect = dc.pipeline_defs("19_design_effect.py").design_effect
TARGET = 0.95
STARTS = [0.95, 0.90, 0.85, 0.80, 0.70]
PTX_LIMITS = [0.05, 0.02]
DRAWS = 20


def bonf_alpha(sc, ok, grid_vals, alpha):
    """Bonferroni over a grid at a given family level; `ok(mask, a)` decides
    whether the candidate's cleared set passes at level a."""
    a = alpha / len(grid_vals)
    best = np.inf
    for t in grid_vals:
        m = sc >= t
        if m.sum() and ok(m, a):
            best = min(best, float(t))
    return best


def fst_alpha(yc, sc, target, seq_vals, alpha):
    """13_analysis.fst_t with the level as a parameter."""
    best = np.inf
    for t in seq_vals:
        m = sc >= t
        n = int(m.sum())
        if n > 0 and A13.cp_lower(int(yc[m].sum()), n, alpha) >= target:
            best = float(t)
        else:
            break
    return best


def ptx_upper(x, m, a):
    """One-sided Clopper-Pearson upper bound on a cleared share x / m."""
    return 1.0 if x >= m else float(beta.ppf(1 - a, x + 1, m - x))


def row(model, cons, variant, t, yt, st, extra=None):
    cov, npv, n = A13.apply_t(yt, st, t)
    r = {"model": model, "construction": cons, "variant": variant, "threshold": t,
         "test_coverage": cov, "test_npv": npv, "test_n": n,
         "held": bool(n and npv >= TARGET)}
    r.update(extra or {})
    return r


idx = np.load(OUT / "valid_idx.npy")
sp = pd.read_csv(OUT / "splits.csv").iloc[idx].reset_index(drop=True)
tr, ca, te = [(sp["split"] == k).values for k in ("train", "calib", "test")]
normal = (sp[A13.COMMON7].sum(axis=1) == 0).values.astype(int)
ptx = sp["Pneumothorax"].values.astype(int)
pid = sp["patient_id"].values
yc, yt = normal[ca], normal[te]

# ------------------------------------------------------------- inversion
inv_rows = []
for key in dc.CHECKPOINTS:
    meta = json.loads((OUT / f"labels_{key}.json").read_text())
    th = np.asarray(xrv.models.model_urls[MODELS[key]]["op_threshs"], dtype=np.float64)
    P = np.load(OUT / f"probs_{key}.npy").astype(np.float64)
    R = np.load(OUT / f"probsraw_{key}.npy").astype(np.float64)
    p_grid = np.unique(np.r_[np.linspace(0, 1, 200001), np.logspace(-8, 0, 20001)])
    for j, lab in enumerate(meta["pathologies"]):
        t = th[j]
        if not np.isfinite(t):
            continue
        # forward with the library's own function (it works in float32), back with Eq. (2)
        o = xrv.models.op_norm(torch.tensor(p_grid[:, None], dtype=torch.float32),
                               torch.tensor([[t]], dtype=torch.float32)).numpy()[:, 0].astype(np.float64)
        back = np.where(o < 0.5, 2 * t * o, 1 - 2 * (1 - t) * (1 - o))
        stored = np.where(P[:, j] < 0.5, 2 * t * P[:, j], 1 - 2 * (1 - t) * (1 - P[:, j]))
        inv_rows.append({"model": key, "finding": lab, "op_thresh": t,
                         "grid_points": len(p_grid), "grid_max_abs_err": float(np.abs(back - p_grid).max()),
                         "stored_outputs": len(P),
                         "stored_max_abs_err_vs_probsraw": float(np.abs(np.clip(stored, 1e-7, 1 - 1e-7) - R[:, j]).max())})
INV = pd.DataFrame(inv_rows)
INV.to_csv(OUT / "variants_inversion.csv", index=False)
print(f"inversion: {len(INV)} trained findings over {INV.model.nunique()} models; "
      f"grid error at most {INV.grid_max_abs_err.max():.1e}, stored outputs within {INV.stored_max_abs_err_vs_probsraw.max():.1e} of probsraw")

# ------------------------------------------------------------- per configuration
meta_e = pd.read_csv(ROOT / "data" / "Data_Entry_2017.csv").set_index("Image Index")
age = pd.to_numeric(meta_e.loc[sp["Image Index"], "Patient Age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").values
age = np.where(age > 110, np.nan, age)
groups = {"sex": meta_e.loc[sp["Image Index"], "Patient Gender"].values,
          "age": np.select([age < 40, age < 60, age >= 60], ["under 40", "40 to 59", "60 and over"], "unknown"),
          "view": meta_e.loc[sp["Image Index"], "View Position"].values}

grid_rows, seq_rows, pat_rows, mr_rows, sub_rows = [], [], [], [], []
calib_pid = pid[ca]
rng = np.random.default_rng(SEED)
draws = []
for d in range(DRAWS):
    order = rng.permutation(len(calib_pid))
    _, first = np.unique(calib_pid[order], return_index=True)
    draws.append(np.sort(order[first]))

for key in PRIMARY_MODELS:
    S = np.load(OUT / f"rev3_scores_{key}_common7.npz")
    for nm in A13.RC_CONS:
        s = S[nm]
        sc, st = s[ca], s[te]
        g200 = np.quantile(s[tr], np.linspace(0.50, 0.9995, A13.GRID))
        g1000 = np.quantile(s[tr], np.linspace(0.50, 0.9995, 1000))
        base = A13.bonf_t(yc, sc, TARGET, g200)

        # grid: denser candidates, and whether the 50% cap can bind
        cap = sc >= g200[0]
        naive = A13.naive_t(yc, sc, TARGET)
        for variant, t in [("bonferroni_200", base), ("bonferroni_1000", A13.bonf_t(yc, sc, TARGET, g1000))]:
            grid_rows.append(row(key, nm, variant, t, yt, st, {
                "calib_npv_at_cap": float(yc[cap].mean()), "calib_coverage_at_cap": float(cap.mean()),
                "calib_coverage_empirical": float((sc >= naive).mean())}))

        # sequence: the start quantile, alone and five starts together
        seqs = {q: np.quantile(s[tr], np.linspace(q, 0.50, A13.GRID)) for q in STARTS}
        for q in STARTS:
            seq_rows.append(row(key, nm, f"start_{q:.2f}", fst_alpha(yc, sc, TARGET, seqs[q], 0.05), yt, st))
        multi = min(fst_alpha(yc, sc, TARGET, seqs[q], 0.05 / len(STARTS)) for q in STARTS)
        seq_rows.append(row(key, nm, "five_starts_alpha_split", multi, yt, st))

        # patient: one image per patient, and design-effect-adjusted n
        p_cal = calib_pid
        t_one = [A13.bonf_t(yc[k], sc[k], TARGET, g200) for k in draws]
        cov_one = [A13.apply_t(yt, st, t)[0] for t in t_one]
        pat_rows.append(row(key, nm, "image_level", base, yt, st))
        pat_rows.append(row(key, nm, "one_image_per_patient", t_one[0], yt, st, {
            "draws": DRAWS, "draws_accepting": int(np.isfinite(t_one).sum()),
            "draw_coverage_min": float(np.min(cov_one)), "draw_coverage_median": float(np.median(cov_one)),
            "draw_coverage_max": float(np.max(cov_one))}))

        def deff_ok(m, a):
            n = int(m.sum())
            de = design_effect(1 - yc[m], p_cal[m])
            n_eff = n / max(de, 1.0) if np.isfinite(de) else float(len(np.unique(p_cal[m])))
            k_eff = n_eff * yc[m].mean()
            lb = a ** (1.0 / n_eff) if k_eff >= n_eff else float(beta.ppf(a, k_eff, n_eff - k_eff + 1))
            return lb >= TARGET
        pat_rows.append(row(key, nm, "design_effect_adjusted", bonf_alpha(sc, deff_ok, g200, 0.05), yt, st))

        # multirisk: NPV and the share of pneumothorax studies cleared
        pc_, pt_ = ptx[ca], ptx[te]
        for limit in [None] + PTX_LIMITS:
            def both_ok(m, a, limit=limit):
                if A13.cp_lower(int(yc[m].sum()), int(m.sum()), a) < TARGET:
                    return False
                return limit is None or ptx_upper(int((m & (pc_ == 1)).sum()), int(pc_.sum()), a) <= limit
            t = bonf_alpha(sc, both_ok, g200, 0.05)
            mr_rows.append(row(key, nm, "npv_only" if limit is None else f"npv_and_ptx_{limit:.2f}", t, yt, st, {
                "test_ptx_share_cleared": float(((st >= t) & (pt_ == 1)).sum() / pt_.sum())}))

        # subgroups of the accepted Bonferroni thresholds
        if np.isfinite(base):
            for gname, gv in groups.items():
                gt = gv[te]
                for level in sorted(set(gt)):
                    m = gt == level
                    sub_rows.append(row(key, nm, f"{gname}={level}", base, yt[m], st[m], {"test_studies": int(m.sum())}))

for name, rows in [("grid", grid_rows), ("sequence", seq_rows), ("patient", pat_rows),
                   ("multirisk", mr_rows), ("subgroups", sub_rows)]:
    pd.DataFrame(rows).to_csv(OUT / f"variants_{name}.csv", index=False)

# ------------------------------------------------------------- calibrator
cal_rows = []
for key in PRIMARY_MODELS:
    paths, R, _, _ = A13.load(key)
    p = np.clip(R[:, [paths.index(l) for l in A13.COMMON7]], 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    q_iso, q_platt = np.zeros_like(p), np.zeros_like(p)
    for j, lab in enumerate(A13.COMMON7):
        yl = sp[lab].values
        q_iso[:, j] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr, j], yl[tr]).predict(p[:, j])
        q_platt[:, j] = LogisticRegression(max_iter=1000).fit(lg[tr, j:j + 1], yl[tr]).predict_proba(lg[:, j:j + 1])[:, 1]
    cached = np.load(OUT / f"rev3_scores_{key}_common7.npz")["B_cal"]
    for nm, q in [("B", p), ("B_cal isotonic", q_iso), ("B_cal Platt", q_platt)]:
        b = np.exp(np.log1p(-np.clip(q, 1e-6, 1 - 1e-6)).sum(1))
        cal_rows.append({"model": key, "score": nm, "test_auc": roc_auc_score(yt, b[te]),
                         "test_ece": A13.ece(yt, b[te]),
                         "max_abs_diff_vs_cached_B_cal": float(np.abs(b - cached).max()) if nm == "B_cal isotonic" else np.nan})
pd.DataFrame(cal_rows).to_csv(OUT / "variants_calibrator.csv", index=False)
print("wrote", ", ".join(f"variants_{n}.csv" for n in ["inversion", "grid", "sequence", "patient", "multirisk", "subgroups", "calibrator"]))
