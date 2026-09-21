"""
Regression tests for the parts of the pipeline that are easy to get silently
wrong. Run from project/ with the venv active:

    python -m pytest tests -q

The first three tests need only out/splits.csv and the step 9 outputs, which
ship with the repository. None of them needs the images or the features.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
OUT = PROJ / "out"

C7 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Pneumonia", "Pneumothorax"]


def _load_naive_t():
    """The threshold rule from 13_analysis.py, without running the analysis."""
    src = (PROJ / "13_analysis.py").read_text(encoding="utf-8")
    block = src.split("# ----------------------------------------------------------------- metrics")[1]
    block = block.split("# ----------------------------------------------------------------- models")[0]
    ns = {}
    exec(block, {"np": np, "beta": None}, ns)
    return ns["naive_t"], ns["apply_t"]


# ------------------------------------------------------------------ 1. splits
def test_no_patient_crosses_splits():
    sp = pd.read_csv(OUT / "splits.csv")
    per_patient = sp.groupby("patient_id")["split"].nunique()
    assert per_patient.max() == 1, "a patient appears in more than one split"
    assert set(sp["split"].unique()) == {"train", "calib", "test"}
    assert len(sp) == 112120


def test_no_finding_matches_the_fourteen_labels():
    sp = pd.read_csv(OUT / "splits.csv")
    labels = [c for c in sp.columns if c not in ("Image Index", "patient_id", "split", "no_finding")]
    assert len(labels) == 14
    derived = (sp[labels].sum(axis=1) == 0).astype(int)
    assert (derived.values == sp["no_finding"].values).all()


# ------------------------------------------------------------------ 2. op_norm inversion
def _op_norm(raw, t):
    """torchxrayvision's warp: threshold t maps to 0.5, linear on each side."""
    return np.where(raw < t, raw / (2 * t), 1 - (1 - raw) / (2 * (1 - t)))


def test_op_norm_round_trip():
    # only the inversion function is needed; take it from the source without running the script body
    src = (PROJ / "09_fix_opnorm.py").read_text(encoding="utf-8")
    fn_src = src[src.index("def invert_op_norm"):src.index("rows = []")]
    ns = {}
    exec(fn_src, {"np": np}, ns)
    invert = ns["invert_op_norm"]
    rng = np.random.default_rng(0)
    raw = rng.uniform(1e-4, 1 - 1e-4, size=(5000, 3)).astype(np.float32)
    th = np.array([0.00126, 0.03, 0.5])
    warped = np.stack([_op_norm(raw[:, j], th[j]) for j in range(3)], axis=1).astype(np.float32)
    back = invert(warped, th)
    assert np.allclose(back, raw, atol=2e-6), "inverse of op_norm does not recover the raw probabilities"


def test_untrained_label_stays_nan():
    src = (PROJ / "09_fix_opnorm.py").read_text(encoding="utf-8")
    fn_src = src[src.index("def invert_op_norm"):src.index("rows = []")]
    ns = {}
    exec(fn_src, {"np": np}, ns)
    back = ns["invert_op_norm"](np.full((10, 2), 0.5, dtype=np.float32), np.array([np.nan, 0.1]))
    assert np.isnan(back[:, 0]).all() and np.isfinite(back[:, 1]).all()


# ------------------------------------------------------------------ 3. threshold rule
def test_empirical_threshold_meets_target_on_the_set_it_admits():
    """The set {score >= t} on calibration must reach the target, ties included.
    Cutting inside a float16 tie group fails this."""
    naive_t, _ = _load_naive_t()
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = 4000
        s = rng.choice(np.linspace(0, 1, 40), size=n).astype(np.float16).astype(np.float32)  # heavy ties
        y = (rng.uniform(size=n) < 0.6 + 0.35 * s).astype(int)
        for target in (0.90, 0.95):
            t = naive_t(y, s, target)
            if np.isfinite(t):
                assert y[s >= t].mean() >= target - 1e-12


def test_empirical_threshold_on_the_shipped_scores():
    """Same check on the real calibration split, for every cached construction."""
    caches = sorted(OUT.glob("rev3_scores_*_common7.npz"))
    if not caches:
        pytest.skip("score caches are not shipped; run 13_analysis.py first")
    naive_t, _ = _load_naive_t()
    idx = np.load(OUT / "valid_idx.npy")
    sp = pd.read_csv(OUT / "splits.csv").iloc[idx].reset_index(drop=True)
    y = (sp[C7].sum(axis=1) == 0).astype(int).values
    ca = (sp["split"] == "calib").values
    for c in caches:
        z = np.load(c, allow_pickle=True)
        for k in z.files:
            if k == "_hp":
                continue
            s = z[k][ca]
            t = naive_t(y[ca], s, 0.95)
            if np.isfinite(t):
                assert y[ca][s >= t].mean() >= 0.95, f"{c.name}/{k}"


# ------------------------------------------------------------------ 4. estimand and constructions
def test_constructions_on_synthetic_marginals():
    p = np.array([[0.1, 0.2, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    A = 1 - p.max(axis=1)
    B = np.exp(np.log1p(-p).sum(axis=1))
    assert np.allclose(A, [0.8, 0.5, 1.0])
    assert np.allclose(B, [0.72, 0.125, 1.0])
    assert (B <= A + 1e-12).all(), "the product can never exceed the Fréchet upper bound"


def test_common_seven_are_predicted_by_every_backbone():
    for key in ["pc", "chex", "mimic_ch", "nih"]:
        meta = json.loads((OUT / f"labels_{key}.json").read_text())
        assert set(C7) <= set(meta["pathologies"]), key
