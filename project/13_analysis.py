"""
Step 13 - the whole analysis. Reads probsraw_* (op_norm undone by step 9) and
feats_* (from step 3), and writes every result table.

The design choices, one per point:

  R1  A threshold chosen at the last empirical NPV crossing is a point estimate
      with SD about 0.007, so it misses its target about half the time by
      construction. On our split the calib half happened to be 1.2 pp easier
      than test. This step measures that with a diagnostic (re-splitting
      calib+test by patient 300 times, and swapping the halves). It is not a
      held-out finding.

  R2  Risk control runs Bonferroni over 200 candidates, the least powerful
      valid procedure, side by side with fixed-sequence
      testing (the Conformal Triage reference procedure). Both were fixed
      before this run: the candidate grid is built from train-score quantiles,
      and the fixed sequence starts at the 0.90 train quantile and walks
      toward higher coverage, stopping at the first failure. Provenance note:
      the 0.90 start was chosen after comparing two starts (0.97 and 0.90)
      on these same data, so it is not independent of the data.

  R3  Isotonic calibrators are fitted on train, so calib is used exactly once,
      for thresholds. Adds A_cal = 1 - max(q) and B_cal = prod(1 - q), the
      direct test of whether the product construction's defect is dependence
      or marginal miscalibration.

  R4  One definition of every learned construction. Hyper-parameters are
      selected on a patient-level 15% slice of train, then the model is refit
      on all of train. Calib is never used for model selection.

  R5  Every confidence interval is a patient-cluster bootstrap (resample
      patients, not images), including the 2x2 bottleneck decomposition.

  R6  Coverage in the bottleneck table is at a calib-chosen threshold, never
      one found by scanning the test split.

  R7  Every summary count is over the primary backbones (pc, chex, mimic_ch).
      NIH was trained on this dataset; it stays in the tables as a labelled
      row and is excluded from every count.

Writes out/rev3_*.csv and out/_rev3_numbers.txt
"""
import json, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from cxrcfg import OUT, PRIMARY_MODELS, LEAKY_MODEL, SEED

warnings.filterwarnings("ignore")
COMMON7 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
           "Effusion", "Pneumonia", "Pneumothorax"]
NIH14 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
         "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
         "Fibrosis", "Pleural_Thickening", "Hernia"]
ALL = list(PRIMARY_MODELS) + [LEAKY_MODEL]
NAIVE_TARGETS = [0.95, 0.98, 0.99]
CTRL_TARGETS = [0.95, 0.98]
DELTA = 0.05
GRID = 200
FST_START_Q = 0.90          # fixed-sequence start: the 0.90 train quantile, i.e. 10% coverage
BOOT = 300
RESPLITS = 300
RNG = np.random.default_rng(SEED)
LOG = []


def say(s=""):
    """Print a line and keep it for out/_rev3_numbers.txt (the full printed results)."""
    print(s); LOG.append(s)


# ----------------------------------------------------------------- metrics
def ece(y, s, bins=15):
    """Expected calibration error with equal-width bins (15 by default)."""
    e = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(s, e) - 1, 0, bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - s[b == k].mean())
                     for k in range(bins) if (b == k).any()))


def fcr(y, s, cov=0.05):
    """False-clearance rate among the top `cov` fraction of scores: the share of
    abnormal images inside the region the score is most confident is normal."""
    n = max(1, int(round(cov * len(s))))
    return float(1 - y[np.argsort(-s)[:n]].mean())


def naive_t(yc, sc, target):
    """Most permissive threshold t such that the empirical calib NPV of the set
    {score >= t} reaches the target. A point estimate; it carries no guarantee.

    Scores are float16, so many images share a value. The running NPV is
    therefore read only at the last image of each tie group, because a
    threshold applied as >= t always admits the whole group. Cutting inside a
    group would report a prefix NPV the applied set does not achieve."""
    o = np.argsort(-sc, kind="stable")
    s = sc[o]
    run = np.cumsum(yc[o]) / np.arange(1, len(yc) + 1)
    last_of_group = np.r_[s[1:] != s[:-1], True]
    ok = np.flatnonzero((run >= target) & last_of_group)
    return float(s[ok[-1]]) if len(ok) else np.inf


def apply_t(yt, st, t):
    """Apply a threshold to the locked test split. Returns (coverage, realised NPV, n)."""
    m = st >= t
    if m.sum() == 0:
        return 0.0, np.nan, 0
    return float(m.mean()), float(yt[m].mean()), int(m.sum())


def cp_lower(k, n, alpha):
    """Exact one-sided Clopper-Pearson lower bound."""
    if n == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    return float(beta.ppf(alpha, k, n - k + 1))


def bonf_t(yc, sc, target, grid_vals):
    """Learn-Then-Test with a Bonferroni correction over the whole fixed grid.
    Every candidate is tested at level DELTA / len(grid); the most permissive
    one that passes is used."""
    a = DELTA / len(grid_vals)
    best = np.inf
    for t in grid_vals:
        m = sc >= t
        n = int(m.sum())
        if n == 0:
            continue
        if cp_lower(int(yc[m].sum()), n, a) >= target:
            best = min(best, float(t))
    return best


def fst_t(yc, sc, target, seq_vals):
    """Fixed-sequence testing: candidates in a fixed order from strict to
    permissive, each tested at level DELTA, stop at the first failure. Valid
    for any fixed order; more powerful than Bonferroni when the risk is close
    to monotone in the threshold."""
    best = np.inf
    for t in seq_vals:
        m = sc >= t
        n = int(m.sum())
        if n > 0 and cp_lower(int(yc[m].sum()), n, DELTA) >= target:
            best = float(t)
        else:
            break
    return best


def cluster_resamples(pid, B):
    """B bootstrap index arrays for one split, resampling patients rather than
    images. One patient contributes up to 184 correlated images, so an
    image-level bootstrap would understate the uncertainty."""
    u, inv = np.unique(pid, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(u))]
    out = []
    for _ in range(B):
        pick = RNG.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[g] for g in pick]))
    return out


def ci(v):
    """2.5 / 97.5 percentiles of a bootstrap vector, ignoring non-finite draws."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ----------------------------------------------------------------- models
def select_logistic(Xtr, ytr, Xtu, ytu, Cs):
    """Pick the regularisation C with the best AUC on the train-tune slice."""
    best, bestauc = None, -1
    for C in Cs:
        m = LogisticRegression(max_iter=4000, C=C).fit(Xtr, ytr)
        a = roc_auc_score(ytu, m.predict_proba(Xtu)[:, 1])
        if a > bestauc:
            best, bestauc = C, a
    return best


def select_mlp(Xtr, ytr, Xtu, ytu, alphas=(1e-4, 1e-3, 1e-2)):
    """Pick MLP width and L2 penalty by AUC on the train-tune slice."""
    best, bestauc = None, -1
    for hid in [(32, 16), (64, 32)]:
        for alpha in alphas:
            m = MLPClassifier(hidden_layer_sizes=hid, alpha=alpha, max_iter=250,
                              early_stopping=True, n_iter_no_change=8,
                              random_state=SEED).fit(Xtr, ytr)
            a = roc_auc_score(ytu, m.predict_proba(Xtu)[:, 1])
            if a > bestauc:
                best, bestauc = (hid, alpha), a
    return best


def fit_logistic(X, y, C, fit):
    """Refit on the rows in `fit` (all of train) and score every row."""
    return LogisticRegression(max_iter=4000, C=C).fit(X[fit], y[fit]).predict_proba(X)[:, 1]


def fit_mlp(X, y, cfg, fit):
    """Refit the selected MLP on the rows in `fit` and score every row."""
    hid, alpha = cfg
    return MLPClassifier(hidden_layer_sizes=hid, alpha=alpha, max_iter=250,
                         early_stopping=True, n_iter_no_change=8,
                         random_state=SEED).fit(X[fit], y[fit]).predict_proba(X)[:, 1]


def pairwise(Zl):
    """Append every pairwise product of the logits (construction C+)."""
    L = Zl.shape[1]
    i, j = np.triu_indices(L, k=1)
    return np.hstack([Zl, Zl[:, i] * Zl[:, j]])


# ----------------------------------------------------------------- data
def load(key):
    """Label list, corrected probabilities, features and split table for one backbone."""
    meta = json.loads((OUT / f"labels_{key}.json").read_text())
    paths = meta["pathologies"]
    R = np.load(OUT / f"probsraw_{key}.npy").astype(np.float32)
    Z = np.load(OUT / f"feats_{key}.npy").astype(np.float32)
    idx = np.load(OUT / "valid_idx.npy")
    sp = pd.read_csv(OUT / "splits.csv").iloc[idx].reset_index(drop=True)
    return paths, R, Z, sp


def build_scores(key, labels, tag, with_features=True, with_dpp=True):
    """Every construction for one backbone and one estimand. Cached to disk."""
    cache = OUT / f"rev3_scores_{key}_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "_hp"}, json.loads(str(z["_hp"]))

    paths, R, Z, sp = load(key)
    cols = [paths.index(l) for l in labels]
    p = np.clip(R[:, cols], 1e-6, 1 - 1e-6)
    y = (sp[labels].sum(axis=1) == 0).astype(int).values
    tr = (sp["split"] == "train").values
    pid = sp["patient_id"].values

    # A patient-level 15% slice of train, used only to pick hyper-parameters.
    # Calib is left for thresholds and test is left alone.
    trp = np.unique(pid[tr])
    rng = np.random.default_rng(SEED)
    rng.shuffle(trp)
    tune_p = set(trp[: int(0.15 * len(trp))].tolist())
    tu = tr & np.isin(pid, list(tune_p))
    fit = tr & ~tu

    # One isotonic calibrator per label, fitted on train.
    q = np.zeros_like(p)
    for j, lab in enumerate(labels):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p[tr, j], sp[lab].values[tr])
        q[:, j] = iso.predict(p[:, j])
    qc = np.clip(q, 1e-6, 1 - 1e-6)

    lg = np.log(p / (1 - p))
    mu, sd = lg[tr].mean(0), lg[tr].std(0) + 1e-6
    L = (lg - mu) / sd
    hp = {}
    S = {"A": 1.0 - p.max(1),
         "B": np.exp(np.log1p(-p).sum(1)),
         "A_cal": 1.0 - qc.max(1),
         "B_cal": np.exp(np.log1p(-qc).sum(1))}

    t0 = time.time()
    hp["C"] = select_logistic(L[fit], y[fit], L[tu], y[tu], [0.01, 0.1, 1.0, 10.0])
    S["C"] = fit_logistic(L, y, hp["C"], tr)
    hp["Cpp"] = select_mlp(L[fit], y[fit], L[tu], y[tu])
    S["C++"] = fit_mlp(L, y, hp["Cpp"], tr)
    if with_features:
        Lp = pairwise(lg)
        Lp = (Lp - Lp[tr].mean(0)) / (Lp[tr].std(0) + 1e-6)
        hp["Cplus"] = select_logistic(Lp[fit], y[fit], Lp[tu], y[tu], [0.01, 0.1, 1.0, 10.0])
        S["C+"] = fit_logistic(Lp, y, hp["Cplus"], tr)
        F = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)
        hp["D"] = select_logistic(F[fit], y[fit], F[tu], y[tu], [0.01, 0.03, 0.1, 0.3])
        S["D"] = fit_logistic(F, y, hp["D"], tr)
        if with_dpp:
            hp["Dpp"] = select_mlp(F[fit], y[fit], F[tu], y[tu], alphas=(1e-4, 1e-2))
            S["D++"] = fit_mlp(F, y, hp["Dpp"], tr)
    hp = {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()}
    hp["_fit_seconds"] = round(time.time() - t0)
    np.savez(cache, _hp=json.dumps(hp), **{k: v.astype(np.float32) for k, v in S.items()})
    return S, hp


# ================================================================== main analysis, per backbone
MAIN_CONS = ["A", "B", "A_cal", "B_cal", "C", "C++", "D"]
RC_CONS = ["A", "B", "B_cal", "C++", "D"]          # risk-control table
COUNT_CONS = ["A", "B", "C++", "D"]                 # the four constructions behind the headline counts

main_rows, rc_rows, rs_rows, bn_rows, dc_rows, dep_rows = [], [], [], [], [], []
label_meta = {}

for key in ALL:
    say("\n" + "=" * 100)
    say(f"{key}   common-7 estimand")
    say("=" * 100)
    paths, R, Z, sp = load(key)
    S, hp = build_scores(key, COMMON7, "common7")
    say(f"  selected hyper-parameters (on train-tune slice): {hp}")
    cols = [paths.index(l) for l in COMMON7]
    p = np.clip(R[:, cols], 1e-6, 1 - 1e-6)
    y = (sp[COMMON7].sum(axis=1) == 0).astype(int).values
    tr = (sp["split"] == "train").values
    ca = (sp["split"] == "calib").values
    te = (sp["split"] == "test").values
    pid = sp["patient_id"].values
    yt, yc = y[te], y[ca]
    boots = cluster_resamples(pid[te], BOOT)

    # How coarse the float16 storage is, per label.
    nuniq = [int(len(np.unique(R[:, c]))) for c in cols]
    label_meta[key] = {"distinct_values_per_label": dict(zip(COMMON7, nuniq))}

    # ---------------- ranking, calibration, safety, and empirical-target thresholds
    say(f"\n  {'construction':8s} {'AUC':>7s} {'[95% CI]':>17s} {'ECE':>7s} {'FCR@5%':>7s} {'[95% CI]':>17s}"
        + "".join(f"{f'cov@{t}':>9s}{'NPV':>7s}{'n':>6s}{'[CI]':>16s}" for t in NAIVE_TARGETS))
    for nm in MAIN_CONS:
        s = S[nm]
        st, sc = s[te], s[ca]
        r = {"model": key, "construction": nm, "auc": roc_auc_score(yt, st),
             "ece": ece(yt, st), "brier": float(np.mean((st - yt) ** 2)), "fcr05": fcr(yt, st)}
        ba = [roc_auc_score(yt[b], st[b]) for b in boots]
        bf = [fcr(yt[b], st[b]) for b in boots]
        r["auc_lo"], r["auc_hi"] = ci(ba)
        r["fcr_lo"], r["fcr_hi"] = ci(bf)
        line = f"  {nm:8s} {r['auc']:7.4f} [{r['auc_lo']:.4f},{r['auc_hi']:.4f}] {r['ece']:7.4f} {r['fcr05']:7.4f} [{r['fcr_lo']:.4f},{r['fcr_hi']:.4f}]"
        for t in NAIVE_TARGETS:
            th = naive_t(yc, sc, t)
            cov, npv, n = apply_t(yt, st, th)
            bn = [apply_t(yt[b], st[b], th)[1] for b in boots]
            lo, hi = ci(bn)
            r[f"cov@{t}"], r[f"npv@{t}"], r[f"n@{t}"] = cov, npv, n
            r[f"npv_lo@{t}"], r[f"npv_hi@{t}"] = lo, hi
            r[f"holds@{t}"] = bool(np.isfinite(npv) and npv >= t)
            line += f"{100*cov:8.2f}%{npv:7.3f}{n:6d} [{lo:.3f},{hi:.3f}]"
        main_rows.append(r)
        say(line)

    # ---------------- risk control: naive vs Bonferroni vs fixed-sequence
    say(f"\n  risk control (grid & sequence from TRAIN quantiles; FST starts at q={FST_START_Q})")
    say(f"  {'target':6s} {'cons':6s} | {'naive cov':>9s} {'NPV':>6s} {'ok':>3s} | {'bonf cov':>9s} {'NPV':>6s} {'ok':>3s} | {'FST cov':>9s} {'NPV':>6s} {'ok':>3s}")
    for nm in RC_CONS:
        s = S[nm]
        st, sc, s_tr = s[te], s[ca], s[tr]
        grid = np.quantile(s_tr, np.linspace(0.50, 0.9995, GRID))
        seq = np.quantile(s_tr, np.linspace(FST_START_Q, 0.50, GRID))
        for t in CTRL_TARGETS:
            res = {"model": key, "construction": nm, "target": t}
            for lab, th in [("naive", naive_t(yc, sc, t)),
                            ("bonf", bonf_t(yc, sc, t, grid)),
                            ("fst", fst_t(yc, sc, t, seq))]:
                cov, npv, n = apply_t(yt, st, th)
                res[f"{lab}_cov"], res[f"{lab}_npv"], res[f"{lab}_n"] = cov, npv, n
                res[f"{lab}_holds"] = bool(np.isfinite(npv) and npv >= t)
            rc_rows.append(res)
            f = lambda l: ("yes" if res[f"{l}_holds"] else ("NO" if res[f"{l}_n"] else "--"))
            say(f"  {t:<6.2f} {nm:6s} | {100*res['naive_cov']:8.2f}% {res['naive_npv']:6.3f} {f('naive'):>3s}"
                f" | {100*res['bonf_cov']:8.2f}% {res['bonf_npv']:6.3f} {f('bonf'):>3s}"
                f" | {100*res['fst_cov']:8.2f}% {res['fst_npv']:6.3f} {f('fst'):>3s}")

    # ---------------- diagnostic: re-split calib+test by patient, and swap the halves
    pool = ca | te
    pool_p = np.unique(pid[pool])
    n_ca_p = len(np.unique(pid[ca]))
    say(f"\n  diagnostic: naive threshold at 0.95, {RESPLITS} patient-level re-splits of calib+test (NOT a held-out result)")
    say(f"  {'cons':6s} {'orig NPV':>9s} {'swap NPV':>9s} {'fail rate':>10s} {'mean short':>11s} {'SD':>7s}")
    for nm in RC_CONS:
        s = S[nm]
        _, orig, _ = apply_t(yt, s[te], naive_t(yc, s[ca], 0.95))
        _, swap, _ = apply_t(yc, s[ca], naive_t(yt, s[te], 0.95))
        real = []
        for _ in range(RESPLITS):
            perm = RNG.permutation(pool_p)
            ma = pool & np.isin(pid, perm[:n_ca_p])
            mb = pool & ~ma
            real.append(apply_t(y[mb], s[mb], naive_t(y[ma], s[ma], 0.95))[1])
        real = np.array(real)
        ok = np.isfinite(real)          # a half may admit no threshold at all; counted separately
        r_ok = real[ok]
        rs_rows.append({"model": key, "construction": nm, "orig_npv": orig, "swap_npv": swap,
                        "fail_rate": float(np.mean(r_ok < 0.95)) if ok.any() else np.nan,
                        "mean_shortfall": float(np.mean(r_ok - 0.95)) if ok.any() else np.nan,
                        "sd": float(np.std(r_ok)) if ok.any() else np.nan,
                        "no_threshold_frac": float(1 - ok.mean())})
        say(f"  {nm:6s} {orig:9.4f} {swap:9.4f} {rs_rows[-1]['fail_rate']:10.3f} {rs_rows[-1]['mean_shortfall']:+11.4f} "
            f"{rs_rows[-1]['sd']:7.4f}   no-threshold {100*(1-ok.mean()):.1f}%")

    # ---------------- 2x2: representation x function class, with paired cluster CIs
    say(f"\n  bottleneck 2x2 (all fit on train, selected on train-tune, thresholds on calib)")
    variants = [("C", "linear(per-finding)"), ("C+", "linear(+pairwise)"),
                ("C++", "MLP(per-finding)"), ("D", "linear(features)"), ("D++", "MLP(features)")]
    say(f"  {'variant':22s} {'AUC':>7s} {'ECE':>7s} {'Brier':>7s} {'FCR@5%':>7s} {'cov@NPV95':>10s} {'NPV':>6s}")
    for nm, desc in variants:
        s = S[nm]
        st, sc = s[te], s[ca]
        cov, npv, n = apply_t(yt, st, naive_t(yc, sc, 0.95))
        r = {"model": key, "variant": f"{nm} {desc}", "auc": roc_auc_score(yt, st), "ece": ece(yt, st),
             "brier": float(np.mean((st - yt) ** 2)), "fcr05": fcr(yt, st),
             "cov_npv95_calib_chosen": cov, "npv_at_cov95": npv}
        bn_rows.append(r)
        say(f"  {r['variant']:22s} {r['auc']:7.4f} {r['ece']:7.4f} {r['brier']:7.4f} {r['fcr05']:7.4f} {100*cov:9.2f}% {npv:6.3f}")
    a = {nm: roc_auc_score(yt, S[nm][te]) for nm, _ in variants}
    gap, rec, resid = a["D"] - a["C"], a["C++"] - a["C"], a["D++"] - a["C++"]
    bg, br, bfr, bres, bdd = [], [], [], [], []
    for b in boots:
        ab = {nm: roc_auc_score(yt[b], S[nm][te][b]) for nm in ["C", "C++", "D", "D++"]}
        g = ab["D"] - ab["C"]
        bg.append(g); br.append(ab["C++"] - ab["C"]); bres.append(ab["D++"] - ab["C++"])
        bdd.append(ab["D++"] - ab["D"])
        bfr.append((ab["C++"] - ab["C"]) / g if g > 0 else np.nan)
    d = {"model": key, "gap_D_minus_C": gap, "recovered_Cpp_minus_C": rec,
         "recovered_frac": rec / gap if gap > 0 else np.nan,
         "residual_Dpp_minus_Cpp": resid, "Dpp_minus_D": a["D++"] - a["D"]}
    for nmv, v in [("gap", bg), ("recovered", br), ("recovered_frac", bfr), ("residual", bres), ("Dpp_minus_D", bdd)]:
        d[f"{nmv}_lo"], d[f"{nmv}_hi"] = ci(v)
    dc_rows.append(d)
    say(f"  AUC gap D-C {gap:+.4f} [{d['gap_lo']:+.4f},{d['gap_hi']:+.4f}]   recovered by C++ {rec:+.4f} "
        f"= {100*d['recovered_frac']:.0f}% [{100*d['recovered_frac_lo']:.0f},{100*d['recovered_frac_hi']:.0f}]   "
        f"residual D++-C++ {resid:+.4f} [{d['residual_lo']:+.4f},{d['residual_hi']:+.4f}]   "
        f"D++-D {a['D++']-a['D']:+.4f} [{d['Dpp_minus_D_lo']:+.4f},{d['Dpp_minus_D_hi']:+.4f}]")

    # ---------------- dependence on raw vs train-calibrated marginals
    Y = sp[COMMON7].values.astype(np.float32)
    q = np.zeros_like(p)
    for j, lab in enumerate(COMMON7):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr, j], Y[tr, j])
        q[:, j] = iso.predict(p[:, j])
    for tag, pp in [("raw", p), ("calibrated", q)]:
        Res = Y[te] - pp[te]
        Cm = np.corrcoef(Res.T)
        off = Cm[~np.eye(len(COMMON7), dtype=bool)]
        pairs = sorted([(COMMON7[i], COMMON7[j], Cm[i, j]) for i in range(7) for j in range(i + 1, 7)],
                       key=lambda t: -abs(t[2]))
        dep_rows.append({"model": key, "version": tag, "mean_abs_r": float(np.abs(off).mean()),
                         "max_abs_r": float(np.abs(off).max()),
                         "top_pair": f"{pairs[0][0]}~{pairs[0][1]}", "top_r": float(pairs[0][2])})
    # Direct check: among the 10% of images B_cal is most confident about, how many are normal?
    s = S["B_cal"][te]
    top = s >= np.quantile(s, 0.90)
    dep_rows[-1]["Bcal_top_decile_mean_score"] = float(s[top].mean())
    dep_rows[-1]["Bcal_top_decile_realised"] = float(yt[top].mean())
    say(f"  dependence: raw mean|r|={dep_rows[-2]['mean_abs_r']:.4f}  calibrated mean|r|={dep_rows[-1]['mean_abs_r']:.4f}  "
        f"| B_cal top decile: mean score {s[top].mean():.4f} vs realised {yt[top].mean():.4f}")

# ================================================================== 14-label secondary
say("\n" + "=" * 100)
say("SECONDARY: 14-label estimand (pc, nih). Naive thresholds only; n and cluster CI shown.")
say("=" * 100)
n14_rows = []
for key in ["pc", LEAKY_MODEL]:
    paths, R, Z, sp = load(key)
    S, hp = build_scores(key, NIH14, "nih14", with_features=True, with_dpp=False)
    y = (sp[NIH14].sum(axis=1) == 0).astype(int).values
    ca = (sp["split"] == "calib").values
    te = (sp["split"] == "test").values
    pid = sp["patient_id"].values
    yt, yc = y[te], y[ca]
    boots = cluster_resamples(pid[te], BOOT)
    say(f"\n  {key}  base rate normal (test) = {yt.mean():.4f}   hp={hp}")
    for nm in MAIN_CONS:
        s = S[nm]
        st, sc = s[te], s[ca]
        r = {"model": key, "construction": nm, "auc": roc_auc_score(yt, st), "ece": ece(yt, st), "fcr05": fcr(yt, st)}
        line = f"  {nm:8s} AUC {r['auc']:.4f} ECE {r['ece']:.4f} FCR {r['fcr05']:.4f}"
        for t in NAIVE_TARGETS:
            th = naive_t(yc, sc, t)
            cov, npv, n = apply_t(yt, st, th)
            lo, hi = ci([apply_t(yt[b], st[b], th)[1] for b in boots])
            r[f"cov@{t}"], r[f"npv@{t}"], r[f"n@{t}"], r[f"npv_lo@{t}"], r[f"npv_hi@{t}"] = cov, npv, n, lo, hi
            line += f" | @{t}: cov {100*cov:.2f}% NPV {npv:.3f} n={n} [{lo:.3f},{hi:.3f}]"
        n14_rows.append(r)
        say(line)

# ================================================================== write
main = pd.DataFrame(main_rows); rc = pd.DataFrame(rc_rows); rs = pd.DataFrame(rs_rows)
bn = pd.DataFrame(bn_rows); dc = pd.DataFrame(dc_rows); dep = pd.DataFrame(dep_rows); n14 = pd.DataFrame(n14_rows)
main.to_csv(OUT / "rev3_main.csv", index=False)
rc.to_csv(OUT / "rev3_riskcontrol.csv", index=False)
rs.to_csv(OUT / "rev3_resplit_diagnostic.csv", index=False)
bn.to_csv(OUT / "rev3_bottleneck.csv", index=False)
dc.to_csv(OUT / "rev3_bottleneck_decomp.csv", index=False)
dep.to_csv(OUT / "rev3_dependence.csv", index=False)
n14.to_csv(OUT / "rev3_nih14.csv", index=False)
(OUT / "rev3_label_meta.json").write_text(json.dumps(label_meta, indent=1))

# ================================================================== summary counts
say("\n" + "=" * 100)
say("SUMMARY COUNTS  (primary = pc, chex, mimic_ch; nih shown separately)")
say("=" * 100)
prim = main[main.model != LEAKY_MODEL]
for t in NAIVE_TARGETS:
    sub = prim[prim.construction.isin(COUNT_CONS)]
    say(f"  naive @{t}: primary {int((~sub[f'holds@{t}']).sum())}/{len(sub)} below target on test "
        f"(all constructions incl. A_cal/B_cal: {int((~prim[f'holds@{t}']).sum())}/{len(prim)}; "
        f"nih: {int((~main[(main.model==LEAKY_MODEL)&main.construction.isin(COUNT_CONS)][f'holds@{t}']).sum())}/{len(COUNT_CONS)})")
rsp = rs[rs.model != LEAKY_MODEL]
say(f"  re-split diagnostic (primary, {RESPLITS}x): mean fail rate {rsp.fail_rate.mean():.3f}, "
    f"mean shortfall {rsp.mean_shortfall.mean():+.4f}, SD range {rsp.sd.min():.4f}-{rsp.sd.max():.4f}; "
    f"swap direction below target: {int((rsp.swap_npv<0.95).sum())}/{len(rsp)} (orig {int((rsp.orig_npv<0.95).sum())}/{len(rsp)})")
rsn = rs[rs.model == LEAKY_MODEL]
say(f"  nih: orig below target {int((rsn.orig_npv<0.95).sum())}/{len(rsn)}, swap below target {int((rsn.swap_npv<0.95).sum())}/{len(rsn)}")
for t in CTRL_TARGETS:
    for cons_set, lab in [(COUNT_CONS, "four R2 constructions"), (RC_CONS, "incl. B_cal")]:
        d = rc[(rc.target == t) & (rc.model != LEAKY_MODEL) & rc.construction.isin(cons_set)]
        say(f"  @{t} primary {lab} (n={len(d)}):")
        for proc in ["naive", "bonf", "fst"]:
            live = d[d[f"{proc}_cov"] > 0]
            say(f"      {proc:6s} mean cov {100*d[f'{proc}_cov'].mean():6.2f}%   zero-coverage {int((d[f'{proc}_cov']==0).sum())}/{len(d)}   "
                f"held on test {int(live[f'{proc}_holds'].sum())}/{len(live)}")
    dn = rc[(rc.target == t) & (rc.model == LEAKY_MODEL) & rc.construction.isin(COUNT_CONS)]
    say(f"      nih ({len(dn)}): naive {100*dn.naive_cov.mean():.2f}%  bonf {100*dn.bonf_cov.mean():.2f}%  fst {100*dn.fst_cov.mean():.2f}%")
say("\n  bottleneck decomposition (primary):")
for _, d in dc[dc.model != LEAKY_MODEL].iterrows():
    say(f"    {d.model:9s} recovered {100*d.recovered_frac:.0f}% [{100*d.recovered_frac_lo:.0f},{100*d.recovered_frac_hi:.0f}]  "
        f"D-C {d.gap_D_minus_C:+.4f} [{d.gap_lo:+.4f},{d.gap_hi:+.4f}]  D++-C++ {d.residual_Dpp_minus_Cpp:+.4f} [{d.residual_lo:+.4f},{d.residual_hi:+.4f}]")
say("\n  calibrated product vs learned linear (primary):")
for key in PRIMARY_MODELS:
    m = main[main.model == key].set_index("construction")
    say(f"    {key:9s} B ECE {m.loc['B','ece']:.4f} -> B_cal ECE {m.loc['B_cal','ece']:.4f}  (C {m.loc['C','ece']:.4f})   "
        f"AUC B_cal {m.loc['B_cal','auc']:.4f} vs C {m.loc['C','auc']:.4f}   FCR B_cal {m.loc['B_cal','fcr05']:.4f} vs C {m.loc['C','fcr05']:.4f}")
say("\n  fp16 distinct values per label (probsraw):")
for k, v in label_meta.items():
    say(f"    {k:9s} min {min(v['distinct_values_per_label'].values())}  max {max(v['distinct_values_per_label'].values())}")

(OUT / "_rev3_numbers.txt").write_text("\n".join(LOG), encoding="utf-8")
print(f"\nwrote out/rev3_*.csv and out/_rev3_numbers.txt")
