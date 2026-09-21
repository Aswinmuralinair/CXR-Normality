"""
Step 14 - four summary figures, drawn from the step 13 tables.

Every number plotted here is read from out/rev3_*.csv, so a figure cannot
disagree with the table it sits next to. Nothing is recomputed.

  figs/rev3_fig1_swap.png          the swap diagnostic: realised NPV on the
                                   original split against the swapped split
  figs/rev3_fig2_coverage.png      coverage at NPV >= 0.95 under the three
                                   threshold procedures
  figs/rev3_fig3_calibration.png   ECE of the product before and after
                                   calibrating the marginals, against the
                                   learned combiners
  figs/rev3_fig4_decomposition.png the 2x2: how much of the derived-versus-
                                   direct gap is function class

The NIH backbone was trained on this dataset. It is drawn hollow or hatched
wherever it appears and is never part of a count.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cxrcfg import OUT, FIGS, PRIMARY_MODELS, LEAKY_MODEL

FIGS.mkdir(exist_ok=True)
NAME = {"pc": "PadChest", "chex": "CheXpert", "mimic_ch": "MIMIC-CXR", "nih": "NIH (leaky)"}
INK, MUTED, GREY = "#15181c", "#5b6570", "#b9c4cc"
C_NAIVE, C_BONF, C_FST = "#9aa5ad", "#0e7490", "#a25708"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "figure.dpi": 100,
})


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ figure 1
def fig_coverage():
    rc = pd.read_csv(OUT / "rev3_riskcontrol.csv")
    rc = rc[rc.target == 0.95]
    cons = ["A", "B", "B_cal", "C++", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    w = 0.26
    for ax, key in zip(axes, PRIMARY_MODELS):
        d = rc[rc.model == key].set_index("construction").loc[cons]
        x = np.arange(len(cons))
        ax.bar(x - w, 100 * d.naive_cov, w, color=C_NAIVE, label="empirical target")
        ax.bar(x, 100 * d.bonf_cov, w, color=C_BONF, label="Learn-Then-Test, Bonferroni")
        ax.bar(x + w, 100 * d.fst_cov, w, color=C_FST, label="Learn-Then-Test, fixed-sequence")
        # mark the controlled thresholds that admitted coverage but fell short on test
        for j, c in enumerate(cons):
            for off, proc in [(0, "bonf"), (w, "fst")]:
                if d.loc[c, f"{proc}_n"] > 0 and not d.loc[c, f"{proc}_holds"]:
                    ax.text(x[j] + off, 100 * d.loc[c, f"{proc}_cov"] + 0.6, "x",
                            ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(cons)
        ax.set_title(NAME[key]); ax.grid(axis="y", color=GREY, lw=0.5, alpha=0.6)
    axes[0].set_ylabel("studies cleared on the locked test split (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Coverage at a target NPV of 0.95. An x marks a controlled threshold whose "
                 "realised test NPV fell below the target.", fontsize=9, color=MUTED, y=1.02)
    save(fig, "rev3_fig2_coverage.png")


# ------------------------------------------------------------------ figure 2
def fig_swap():
    rs = pd.read_csv(OUT / "rev3_resplit_diagnostic.csv")
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    lo, hi = 0.925, 0.972
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, zorder=1)
    ax.axhline(0.95, color=INK, lw=0.8, ls=":", zorder=1)
    ax.axvline(0.95, color=INK, lw=0.8, ls=":", zorder=1)
    marks = {"A": "o", "B": "s", "B_cal": "D", "C++": "^", "D": "v"}
    for _, r in rs.iterrows():
        leaky = r.model == LEAKY_MODEL
        ax.scatter(r.orig_npv, r.swap_npv, marker=marks[r.construction], s=42,
                   facecolor="white" if leaky else C_BONF, edgecolor=C_BONF, lw=1.1, zorder=3)
    for c, m in marks.items():
        ax.scatter([], [], marker=m, s=42, facecolor=C_BONF, edgecolor=C_BONF, label=c)
    ax.scatter([], [], marker="o", s=42, facecolor="white", edgecolor=C_BONF, label="NIH (leaky), hollow")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("realised NPV, threshold chosen on calib, applied to test")
    ax.set_ylabel("realised NPV, threshold chosen on test, applied to calib")
    ax.set_title("The swap diagnostic at a 0.95 target")
    ax.text(0.9265, 0.9695, "below target one way,\nabove it the other", fontsize=7.5, color=MUTED, va="top")
    save(fig, "rev3_fig1_swap.png")


# ------------------------------------------------------------------ figure 3
def fig_calibration():
    m = pd.read_csv(OUT / "rev3_main.csv")
    cons = ["B", "B_cal", "C", "C++"]
    labels = ["B  product", "B_cal  product,\ncalibrated marginals", "C  learned linear", "C++  learned MLP"]
    cols = ["#9aa5ad", C_FST, C_BONF, "#15706b"]
    keys = PRIMARY_MODELS + [LEAKY_MODEL]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.19
    x = np.arange(len(keys))
    for i, (c, lab, col) in enumerate(zip(cons, labels, cols)):
        vals = [m[(m.model == k) & (m.construction == c)].ece.iloc[0] for k in keys]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=lab)
        bars[-1].set_hatch("////"); bars[-1].set_edgecolor("white")   # NIH, leaky
    ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k in keys])
    ax.set_ylabel("expected calibration error (test split)")
    ax.set_title("Calibrating the marginals removes most of the product construction's error")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color=GREY, lw=0.5, alpha=0.6)
    save(fig, "rev3_fig3_calibration.png")


# ------------------------------------------------------------------ figure 4
def fig_decomposition():
    dc = pd.read_csv(OUT / "rev3_bottleneck_decomp.csv")
    keys = PRIMARY_MODELS + [LEAKY_MODEL]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    fig.subplots_adjust(wspace=0.32)
    x = np.arange(len(keys))
    for i, k in enumerate(keys):
        r = dc[dc.model == k].iloc[0]
        leaky = k == LEAKY_MODEL
        face = "white" if leaky else C_BONF
        a1.errorbar(i, 100 * r.recovered_frac,
                    yerr=[[100 * (r.recovered_frac - r.recovered_frac_lo)], [100 * (r.recovered_frac_hi - r.recovered_frac)]],
                    fmt="o", color=C_BONF, mfc=face, capsize=3, lw=1.1)
        a2.errorbar(i - 0.12, r.gap_D_minus_C, yerr=[[r.gap_D_minus_C - r.gap_lo], [r.gap_hi - r.gap_D_minus_C]],
                    fmt="s", color=C_FST, mfc="white" if leaky else C_FST, capsize=3, lw=1.1,
                    label="AUC gap, direct minus linear on logits (D − C)" if i == 0 else None)
        a2.errorbar(i + 0.12, r.residual_Dpp_minus_Cpp,
                    yerr=[[r.residual_Dpp_minus_Cpp - r.residual_lo], [r.residual_hi - r.residual_Dpp_minus_Cpp]],
                    fmt="^", color=C_BONF, mfc="white" if leaky else C_BONF, capsize=3, lw=1.1,
                    label="residual under equal function class (D++ − C++)" if i == 0 else None)
    for ax in (a1, a2):
        ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k in keys], fontsize=8)
        ax.grid(axis="y", color=GREY, lw=0.5, alpha=0.6)
    a1.set_ylabel("share of the D − C gap recovered by C++ (%)"); a1.set_ylim(0, 60)
    a1.set_title("Function class explains about a third of the gap")
    a2.axhline(0, color=INK, lw=0.8); a2.set_ylabel("AUC difference"); a2.set_ylim(-0.016, 0.054)
    a2.set_title("What remains is the representation")
    a2.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.text(0.5, -0.04, "Bars are 95% patient-cluster bootstrap intervals. NIH is hollow and not counted.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "rev3_fig4_decomposition.png")


fig_coverage()
fig_swap()
fig_calibration()
fig_decomposition()
