"""Shared code for the demo: 16_demo_build.py fits and checks, 17_demo.py serves.

The demo calls the pipeline's own functions:

  02_build_cache.load_one        decode and resize, the way the image cache was built
  03_extract.get_features        the 1024-d features that construction D reads
  09_fix_opnorm.invert_op_norm   undoes torchxrayvision's op_norm warp
  13_analysis                    COMMON7, NIH14, the construction lists, the targets,
                                 naive_t, bonf_t, fst_t, apply_t and load

09 and 13 do their work at module level, so importing them would rerun the
whole step. `pipeline_defs` executes only their imports, constants and
function definitions.

Two things in 13_analysis.py cannot be imported: build_scores computes the
seven constructions inline and does not return the fitted calibrators and
combiners. `construction_scores` below repeats that arithmetic for any rows,
and 16_demo_build.py refits the components. The build step refuses to save
unless both reproduce 13_analysis.py's cached scores for every image exactly.
"""
import ast
import sys
import types
from pathlib import Path

import numpy as np

from cxrcfg import ROOT, OUT, MODELS, PRIMARY_MODELS, LEAKY_MODEL

BUNDLE = OUT / "demo_models.pkl"
DEMO_DIR = ROOT / "demo"
SAMPLES_DIR = DEMO_DIR / "samples"
CHECKPOINTS = list(PRIMARY_MODELS) + [LEAKY_MODEL]
PROCEDURES = ["empirical", "bonferroni", "fixed_sequence"]

_loaded = {}


def pipeline_defs(script):
    """Run a pipeline script's imports, constants and function definitions,
    stopping at the first loop, `if` or `with` block, which is where each
    script starts its actual work. Bare calls before that point (such as
    warnings.filterwarnings in 13_analysis.py) are skipped."""
    if script in _loaded:
        return _loaded[script]
    path = ROOT / script
    keep = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
            break
        if isinstance(node, ast.Expr):
            continue
        keep.append(node)
    mod = types.ModuleType(path.stem)
    mod.__file__ = str(path)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), mod.__dict__)
    _loaded[script] = mod
    return mod


def analysis():
    """13_analysis.py without the analysis."""
    m = pipeline_defs("13_analysis.py")
    for name in ["COMMON7", "NIH14", "MAIN_CONS", "RC_CONS", "COUNT_CONS", "NAIVE_TARGETS",
                 "CTRL_TARGETS", "GRID", "FST_START_Q", "naive_t", "bonf_t", "fst_t", "apply_t", "load"]:
        if not hasattr(m, name):
            raise RuntimeError(f"13_analysis.py no longer defines {name}; the demo needs updating")
    return m


def load_one(path):
    """02_build_cache.load_one: grayscale decode, 224 x 224 with INTER_AREA."""
    return pipeline_defs("02_build_cache.py").load_one(path)


def invert_op_norm(P, th):
    """09_fix_opnorm.invert_op_norm."""
    return pipeline_defs("09_fix_opnorm.py").invert_op_norm(P, th)


def op_threshs(key):
    """The operating thresholds step 9 inverted with, read from the library the
    same way 09_fix_opnorm.py reads them."""
    import torchxrayvision as xrv
    return np.asarray(xrv.models.model_urls[MODELS[key]]["op_threshs"], dtype=np.float64)


def load_checkpoint(key, device):
    import torchxrayvision as xrv
    return xrv.models.DenseNet(weights=MODELS[key]).to(device).eval()


def forward(model, img, device):
    """One 224 x 224 uint8 image through one backbone, as 03_extract.py ran the
    dataset: the same normalisation, autocast to float16 on a GPU, and outputs
    and features rounded to float16 as they were stored. On a CPU the network
    runs in float32, so values can differ from the stored ones in the third
    decimal place."""
    import torch
    import torchxrayvision as xrv
    get_features = pipeline_defs("03_extract.py").get_features
    x = torch.from_numpy(xrv.datasets.normalize(img[None].astype(np.float32), 255)).unsqueeze(1).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(x)
        z = get_features(model, x)
    out = out.float()
    if out.min() < 0 or out.max() > 1:      # same defensive branch as 03_extract.py; never fires
        out = torch.sigmoid(out)
    return out.cpu().numpy().astype(np.float16)[0], z.float().cpu().numpy().astype(np.float16)[0]


def corrected(P16, th):
    """Undo op_norm for one image. 09_fix_opnorm.py saved the result as float16
    and 13_analysis.py read it back as float32; both roundings are kept so a
    stored image gives the stored number."""
    return invert_op_norm(P16[None].astype(np.float32), th).astype(np.float16).astype(np.float32)[0]


def construction_scores(comp, R, Z):
    """The seven constructions for rows of corrected probabilities R (every
    label the checkpoint has, float32) and features Z (float32). Same
    arithmetic and dtypes as build_scores in 13_analysis.py; float32 output
    like its cache. Also returns the clipped marginals p and the train-fitted
    isotonic marginals q for display."""
    p = np.clip(R[:, comp["cols"]], 1e-6, 1 - 1e-6)
    q = np.zeros_like(p)
    for j, iso in enumerate(comp["iso"]):
        q[:, j] = iso.predict(p[:, j])
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p))
    L = (lg - comp["mu"]) / comp["sd"]
    F = (Z - comp["fmu"]) / comp["fsd"]
    S = {"A": 1.0 - p.max(1),
         "B": np.exp(np.log1p(-p).sum(1)),
         "A_cal": 1.0 - qc.max(1),
         "B_cal": np.exp(np.log1p(-qc).sum(1)),
         "C": comp["C"].predict_proba(L)[:, 1],
         "C++": comp["C++"].predict_proba(L)[:, 1],
         "D": comp["D"].predict_proba(F)[:, 1]}
    return {k: v.astype(np.float32) for k, v in S.items()}, p, q


BLOCK = 16


def score_image(comp, r, z):
    """The seven scores for one image, from its corrected probabilities r and
    features z. BLAS takes a different code path for a single row, which moves
    the last bit of C, C++ and D away from the batch values 13_analysis.py
    cached, so the row is repeated to a block of 16 first. 16_demo_build.py
    checks on 300 images per checkpoint that the result is exact."""
    S, p, q = construction_scores(comp, np.repeat(r[None], BLOCK, 0),
                                  np.repeat(z[None].astype(np.float32), BLOCK, 0))
    return {k: float(v[0]) for k, v in S.items()}, p[0], q[0]


def verdicts(score, rules):
    """Apply every operating threshold to one score. A study is cleared when
    score >= t, the same test apply_t in 13_analysis.py uses. An infinite
    threshold means the procedure found no threshold that passes, so it clears
    nothing."""
    st = np.asarray([score], dtype=np.float32)       # a float32 array, like the cached scores
    out = {}
    for proc, by_target in rules.items():
        out[proc] = {}
        for target, t in by_target.items():
            out[proc][target] = bool(np.isfinite(t) and (st >= t)[0])
    return out


def utf8_console():
    """torchxrayvision's download bar prints characters cp1252 cannot encode."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
