"""
Step 16 - fit and check what the demo needs. No earlier step depends on it or
reads what it writes. Step 18 uses its output.

13_analysis.py fits the isotonic calibrators and the learned combiners inside
build_scores and keeps only their scores. The demo has to score an image it
has never seen, so this step refits the same components on train with the
hyper-parameters 13_analysis.py selected (read from its score cache). It then
checks three things and refuses to save if any fails:

  1. the refitted components, run through demo_core.construction_scores,
     reproduce 13_analysis.py's cached scores for all 112,120 images exactly,
     for every checkpoint and all seven constructions;
  2. every operating threshold, recomputed with naive_t, bonf_t and fst_t from
     13_analysis.py, gives on the test split the coverage, NPV and n that
     out/rev3_main.csv and out/rev3_riskcontrol.csv report;
  3. each sample study, run live through the demo's own inference path on
     this machine, gets the same verdict as the stored research scores under
     every rule and every checkpoint.

Needs out/probsraw_*.npy, out/feats_*.npy, out/rev3_scores_*_common7.npz and
data/images. Runs on the CPU in a few minutes.

Writes out/demo_models.pkl and demo/samples/<Image Index>.png
"""
import json
import pickle
import shutil
import time
from datetime import date

import numpy as np
import pandas as pd
import sklearn
import torch
import torchxrayvision as xrv
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import demo_core as dc
from cxrcfg import OUT, IMAGES, MODELS, SEED, HEADLINE_MODEL

A13 = dc.analysis()
CANDIDATES_PER_SAMPLE = 25      # live-checked in turn until one agrees everywhere


def fit_components(key, R, sp, Z, hp):
    """Refit what build_scores in 13_analysis.py fits for the common-7
    estimand, line for line, and keep the fitted objects. Hyper-parameters
    come from its cache, so no selection is rerun."""
    labels = A13.COMMON7
    paths = json.loads((OUT / f"labels_{key}.json").read_text())["pathologies"]
    cols = [paths.index(l) for l in labels]
    p = np.clip(R[:, cols], 1e-6, 1 - 1e-6)
    y = (sp[labels].sum(axis=1) == 0).astype(int).values
    tr = (sp["split"] == "train").values

    iso = []
    for j, lab in enumerate(labels):
        m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        m.fit(p[tr, j], sp[lab].values[tr])
        iso.append(m)

    lg = np.log(p / (1 - p))
    mu, sd = lg[tr].mean(0), lg[tr].std(0) + 1e-6
    L = (lg - mu) / sd
    fmu, fsd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
    F = (Z - fmu) / fsd
    hid, alpha = hp["Cpp"]
    return {
        "cols": cols, "iso": iso, "mu": mu, "sd": sd, "fmu": fmu, "fsd": fsd,
        "C": LogisticRegression(max_iter=4000, C=hp["C"]).fit(L[tr], y[tr]),
        "C++": MLPClassifier(hidden_layer_sizes=tuple(hid), alpha=alpha, max_iter=250,
                             early_stopping=True, n_iter_no_change=8,
                             random_state=SEED).fit(L[tr], y[tr]),
        "D": LogisticRegression(max_iter=4000, C=hp["D"]).fit(F[tr], y[tr]),
    }


def operating_rules(S, y, sp):
    """Every threshold the study uses, computed as the loops in
    13_analysis.py compute them: the empirical target for all seven
    constructions, and both Learn-Then-Test procedures for the five
    risk-control constructions, with grid and sequence from train quantiles.
    Returns the thresholds and what each one did on the test split."""
    tr = (sp["split"] == "train").values
    ca = (sp["split"] == "calib").values
    te = (sp["split"] == "test").values
    rules, test = {}, {}
    for nm in A13.MAIN_CONS:
        s = S[nm]
        sc, st = s[ca], s[te]
        r = {"empirical": {t: A13.naive_t(y[ca], sc, t) for t in A13.NAIVE_TARGETS}}
        if nm in A13.RC_CONS:
            grid = np.quantile(s[tr], np.linspace(0.50, 0.9995, A13.GRID))
            seq = np.quantile(s[tr], np.linspace(A13.FST_START_Q, 0.50, A13.GRID))
            r["bonferroni"] = {t: A13.bonf_t(y[ca], sc, t, grid) for t in A13.CTRL_TARGETS}
            r["fixed_sequence"] = {t: A13.fst_t(y[ca], sc, t, seq) for t in A13.CTRL_TARGETS}
        rules[nm] = r
        test[nm] = {proc: {t: A13.apply_t(y[te], st, th) for t, th in by_t.items()}
                    for proc, by_t in r.items()}
    return rules, test


def same(a, b):
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return abs(float(a) - float(b)) <= 1e-12


def check_tables(key, test):
    """Compare each threshold's test-split result with the published tables."""
    main = pd.read_csv(OUT / "rev3_main.csv")
    rc = pd.read_csv(OUT / "rev3_riskcontrol.csv")
    checked, bad = 0, []
    for nm in A13.MAIN_CONS:
        row = main[(main.model == key) & (main.construction == nm)].iloc[0]
        for t in A13.NAIVE_TARGETS:
            cov, npv, n = test[nm]["empirical"][t]
            for got, col in [(cov, f"cov@{t}"), (npv, f"npv@{t}"), (n, f"n@{t}")]:
                checked += 1
                if not same(got, row[col]):
                    bad.append(f"{nm} empirical {t} {col}: {got} vs {row[col]}")
    for nm in A13.RC_CONS:
        for t in A13.CTRL_TARGETS:
            row = rc[(rc.model == key) & (rc.construction == nm) & (rc.target == t)].iloc[0]
            for proc, tag in [("empirical", "naive"), ("bonferroni", "bonf"), ("fixed_sequence", "fst")]:
                cov, npv, n = test[nm][proc][t]
                for got, col in [(cov, f"{tag}_cov"), (npv, f"{tag}_npv"), (n, f"{tag}_n")]:
                    checked += 1
                    if not same(got, row[col]):
                        bad.append(f"{nm} {proc} {t} {col}: {got} vs {row[col]}")
    return checked, bad


def all_verdicts(scores, rules):
    return {nm: dc.verdicts(scores[nm], rules[nm]) for nm in A13.MAIN_CONS}


# ---------------------------------------------------------------- sample studies
# Each sample is a test-split image (never used to fit anything or to choose a
# threshold), picked by a fixed rule on the headline checkpoint's stored
# scores under the empirical target at 0.95. Candidates are shuffled with the
# project seed, studies with fewer listed findings go first, and the first one
# whose live verdicts match the stored ones under every rule and checkpoint is
# kept.
SAMPLE_SPECS = [
    ("normal_cleared", "No finding, cleared",
     "The report lists no finding. All seven constructions clear it under the empirical rule at 0.95."),
    ("normal_referred", "No finding, referred",
     "The report lists no finding, yet all seven constructions send it to a radiologist at 0.95."),
    ("cardiomegaly", "Cardiomegaly",
     "The report lists cardiomegaly and nothing else. All seven constructions refer it."),
    ("effusion", "Effusion",
     "The report lists an effusion. All seven constructions refer it."),
    ("pneumothorax", "Pneumothorax",
     "The report lists a pneumothorax. All seven constructions refer it."),
    ("bound_clears_abnormal", "Cleared by the bound",
     "The report lists {labels}. 1 - max(p) clears it at 0.95, while the calibrated product and the direct classifier refer it."),
    ("mass_only", "Mass only",
     "The report lists a mass. None of the seven scored findings is present, so the study counts as normal here, and most constructions clear it."),
]


def sample_candidates(spec, sp, y7, clear, te):
    lab = sp[A13.NIH14].values.astype(bool)
    names = np.array(A13.NIH14)
    only = lambda f: lab[:, list(names).index(f)] & (lab.sum(1) == 1)
    has = lambda f: lab[:, list(names).index(f)]
    nf = sp["no_finding"].values == 1
    n_clear = clear.sum(1)
    k = A13.MAIN_CONS.index
    m = {
        "normal_cleared": nf & (n_clear == 7),
        "normal_referred": nf & (n_clear == 0),
        "cardiomegaly": only("Cardiomegaly") & (n_clear == 0),
        "effusion": has("Effusion") & (n_clear == 0),
        "pneumothorax": has("Pneumothorax") & (n_clear == 0),
        "bound_clears_abnormal": (y7 == 0) & clear[:, k("A")] & ~clear[:, k("B_cal")] & ~clear[:, k("D")],
        "mass_only": only("Mass") & (n_clear >= 5),
    }[spec]
    idx = np.flatnonzero(m & te)
    np.random.default_rng(SEED).shuffle(idx)
    idx = idx[np.argsort(lab[idx].sum(1), kind="stable")]     # fewest listed findings first
    return idx[:CANDIDATES_PER_SAMPLE]


def main():
    dc.utf8_console()
    t_start = time.time()
    bundle = {
        "format": 1,
        "built": str(date.today()),
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__,
                     "torch": torch.__version__, "torchxrayvision": xrv.__version__},
        "common7": A13.COMMON7, "nih14": A13.NIH14,
        "main_cons": A13.MAIN_CONS, "rc_cons": A13.RC_CONS, "count_cons": A13.COUNT_CONS,
        "naive_targets": A13.NAIVE_TARGETS, "ctrl_targets": A13.CTRL_TARGETS,
        "checkpoints": {}, "samples": [],
    }
    stored = {}      # per checkpoint: what the research pipeline stored for every image

    for key in dc.CHECKPOINTS:
        t0 = time.time()
        paths, R, Z, sp = A13.load(key)
        cache = np.load(OUT / f"rev3_scores_{key}_common7.npz", allow_pickle=True)
        hp = json.loads(str(cache["_hp"]))
        comp = fit_components(key, R, sp, Z, hp)

        # check 1: every image, every construction, bit for bit
        S, _, _ = dc.construction_scores(comp, R, Z)
        for nm in A13.MAIN_CONS:
            if not np.array_equal(S[nm], cache[nm]):
                d = np.abs(S[nm].astype(np.float64) - cache[nm])
                raise SystemExit(f"{key}/{nm}: refit differs from the 13_analysis.py cache "
                                 f"(max |diff| {d.max():.3e}); nothing saved")
        # the same, one image at a time, the way the demo scores an upload
        rows = np.random.default_rng(SEED).choice(len(R), 300, replace=False)
        worst = 0.0
        for i in rows:
            s1, _, _ = dc.score_image(comp, R[i], Z[i])
            worst = max(worst, max(abs(s1[nm] - float(cache[nm][i])) for nm in A13.MAIN_CONS))
        if worst > 0:
            raise SystemExit(f"{key}: one-image scoring differs from the cache by {worst:.3e}; nothing saved")

        # check 2: thresholds against the published tables
        y = (sp[A13.COMMON7].sum(axis=1) == 0).astype(int).values
        Sc = {nm: cache[nm] for nm in A13.MAIN_CONS}
        rules, test = operating_rules(Sc, y, sp)
        checked, bad = check_tables(key, test)
        if bad:
            raise SystemExit(f"{key}: {len(bad)} threshold results differ from the tables:\n  "
                             + "\n  ".join(bad[:10]))

        meta = json.loads((OUT / f"labels_{key}.json").read_text())
        te = (sp["split"] == "test").values
        bundle["checkpoints"][key] = {
            "weights": MODELS[key], "pathologies": paths, "trained_nih14": meta["usable_nih14"],
            "op_threshs": dc.op_threshs(key).tolist(), "hp": hp, "comp": comp,
            "rules": rules, "test": test,
            "test_images": int(te.sum()), "test_normal_rate": float(y[te].mean()),
            "checks": {"images_scored_exactly": int(len(R)), "one_image_rows": int(len(rows)),
                       "table_values_matched": checked},
        }
        stored[key] = {"S": Sc, "y": y}
        print(f"{key:9s} scores exact on {len(R):,} images, one-image path exact on {len(rows)} rows, "
              f"{checked} table values matched  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------ samples
    sp = A13.load(HEADLINE_MODEL)[3]
    te = (sp["split"] == "test").values
    head = bundle["checkpoints"][HEADLINE_MODEL]
    Sh = stored[HEADLINE_MODEL]["S"]
    clear = np.column_stack([Sh[nm] >= head["rules"][nm]["empirical"][0.95] for nm in A13.MAIN_CONS])
    y7 = stored[HEADLINE_MODEL]["y"]

    device = torch.device("cpu")
    models = {k: dc.load_checkpoint(k, device) for k in dc.CHECKPOINTS}
    dc.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for spec, title, caption in SAMPLE_SPECS:
        chosen = None
        for i in sample_candidates(spec, sp, y7, clear, te):
            name = sp["Image Index"].iloc[i]
            img = dc.load_one(IMAGES / name)
            agree = True
            for key in dc.CHECKPOINTS:
                ck = bundle["checkpoints"][key]
                P16, Z16 = dc.forward(models[key], img, device)
                R1 = dc.corrected(P16, np.asarray(ck["op_threshs"]))
                live, _, _ = dc.score_image(ck["comp"], R1, Z16)
                ref = {nm: float(stored[key]["S"][nm][i]) for nm in A13.MAIN_CONS}
                if all_verdicts(live, ck["rules"]) != all_verdicts(ref, ck["rules"]):
                    agree = False
                    break
            if agree:
                chosen = i
                break
        if chosen is None:
            print(f"  [skip] no candidate for '{spec}' agreed under every rule")
            continue
        i = chosen
        name = sp["Image Index"].iloc[i]
        shutil.copy2(IMAGES / name, dc.SAMPLES_DIR / name)
        present = [l for l in A13.NIH14 if sp[l].iloc[i] == 1]
        nice = ", ".join(l.replace("_", " ").lower() for l in present)
        bundle["samples"].append({
            "file": name, "kind": spec, "title": title,
            "caption": caption.format(labels=nice or "no finding"),
            "patient_id": int(sp["patient_id"].iloc[i]), "split": "test", "findings": present,
        })
        print(f"  sample {spec:22s} {name}  findings: {present or ['none']}")

    with open(dc.BUNDLE, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nwrote {dc.BUNDLE} ({dc.BUNDLE.stat().st_size / 1e6:.1f} MB) and "
          f"{len(bundle['samples'])} samples in {dc.SAMPLES_DIR}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
