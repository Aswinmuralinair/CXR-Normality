# Can a chest X-ray AI safely say "normal"?

Calibration and finite-sample risk control of normality scores built from
per-finding deep learning outputs, on NIH ChestX-ray14 (112,120 images,
30,805 patients) with four pretrained DenseNet-121 models from torchxrayvision.

This repository holds the code and every result table behind the paper
"Certified Auto-Clearing of Normal Chest Radiographs: Calibration and
Finite-Sample Risk Control of Per-Finding Deep Learning Scores" (under review).

## Start here

| file | what it is |
|---|---|
| `project/README.md` | the pipeline, step by step, with setup and run order |
| `project/out/_rev3_numbers.txt` | every number the main analysis prints |
| `project/17_demo.py` | the demo: upload a chest X-ray and see every score's verdict |

## Where each part of the paper comes from

| paper | script | output |
|---|---|---|
| Undoing the library's rescaling, Eq. (2) and Section IV-A | `09_fix_opnorm.py` | `out/probsraw_*.npy`, `out/op_threshs.json`, `out/opnorm_fix.csv` |
| Scores, thresholds and the main results (Sections III and IV-B to IV-E) | `13_analysis.py` | `out/rev3_*.csv`, `out/_rev3_numbers.txt` |
| Patient clustering (Section IV-F) | `19_design_effect.py` | `out/design_effect.csv` |
| The Indiana University check (Section IV-F) | `18_outside_check.py` | `out/outside_*.csv` |
| Variants of the guarantee and the inversion check (Section IV-G) | `20_guarantee_variants.py` | `out/variants_*.csv` |

## What the study found

The score hospitals use to auto-clear a chest X-ray is built from per-finding
probabilities. The published construction, 1 − max(p), is the Fréchet upper
bound on the joint probability, the most optimistic value the marginals allow. A threshold
chosen to hit an empirical NPV target on held-out data is a point estimate and
misses its target about half the time. Attaching a finite-sample guarantee at
NPV 0.95 cuts mean coverage from 15.5% to between 3.3% and 8.3%, depending on the
multiplicity procedure, and at NPV 0.98 nothing can be cleared. That guarantee
treats every image as independent, though one patient can have many images.
Step 19 measures that clustering inflates the variance it relies on 1.26 to
1.47 times, so its nominal 95% confidence is optimistic. Once the per-finding
probabilities are calibrated, the plain product comes within 0.012 AUC of a
learned linear combiner (and matches it on PadChest) and trails a small MLP by
0.006 to 0.023 AUC. Direct estimation from image features still ranks best, and the MLP closes
about a third of the gap between the linear combiner and that direct model.

A check on 340 radiologist-reported radiographs from another hospital system
(the Indiana University collection, which none of the models saw) found that
the scores still separated normal from abnormal studies well (AUC 0.89 to 0.91
with PadChest). The same empirical-target thresholds, though, cleared far more of the normal
studies there (61.0% to 81.5% with PadChest, against 20.2% to 36.9% on the
NIH test split), and pneumothorax was a blind spot. The exams were chosen by
label, so the check covers ranking and clearance rates only. It cannot measure
the NPV.

## Try the demo

The demo runs the study's own models and decision rules on any chest X-ray you
give it. It needs the Setup below but not the dataset.

```
cd project
python 17_demo.py
```

It opens http://127.0.0.1:8765. Upload an X-ray or pick one of the seven sample
studies. The simple view shows in plain words whether three of the scores
would auto-clear it or send it to a radiologist, and for sample studies
whether that matches the report labels. The detailed view adds the 14
per-finding probabilities, all seven normality scores, all four models and
every threshold rule in the study. The demo is a research demonstration and
not a clinical tool, and its thresholds do not carry over to another
hospital's images. `project/README.md` says how the demo was checked against
the study's outputs.

## What is in this repository

```
project/
  00_download.py … 03_extract.py   data download, patient-level splits, image cache, feature extraction
  09_fix_opnorm.py                 inverts torchxrayvision's op_norm rescaling (mandatory)
  13_analysis.py                   the whole analysis
  14_figures.py                    four summary figures, from the step 13 tables
  15_manifest.py                   writes out/RUN_MANIFEST.json (versions, seed, weights, file hashes)
  16_demo_build.py, 17_demo.py     the demo: builds and checks its fitted components; serves the page
  18_outside_check.py              340 radiographs from another hospital through the demo's analysis
  19_design_effect.py              how much patient clustering weakens the risk-control bounds
  20_guarantee_variants.py         variants of the guarantee (grids, starts, patient-level, a pneumothorax
                                   limit, subgroups, Platt calibration) and a check of the op_norm inversion
  demo_core.py                     the demo's shared code, which calls the pipeline's own functions
  demo/                            the demo page and seven sample images
  tests/                           eight regression tests; python -m pytest tests -q
  requirements-lock.txt            the exact package versions behind the reported numbers
  cxrcfg.py, extract_zip.py        configuration and the zip extractor
  out/rev3_*.csv                   every result table of the main analysis
  out/outside_*.csv                step 18's tables
  out/design_effect.csv            step 19's measurements
  out/variants_*.csv               step 20's measurements
  out/_rev3_numbers.txt            every number step 13 prints
  out/probsraw_*.npy               corrected per-finding probabilities, float16
  out/probs_*.npy                  raw torchxrayvision outputs (rescaled, not probabilities)
  out/labels_*.json, splits.csv, valid_idx.npy
  out/op_threshs.json, RUN_MANIFEST.json   the thresholds step 9 inverted with; what the run was made with
  out/demo_models.pkl              the demo's fitted calibrators and combiners (written by step 16)
  figs/                            step 14's four figures
```

## What is not in this repository, and why

| item | size | how to get it |
|---|---|---|
| `project/data/` | 7.8 GB | Kaggle mirror `khanfashee/nih-chest-x-ray-14-224x224-resized`, via `00_download.py --resized` |
| `project/out/feats_*.npy` | 876 MB | run `03_extract.py` (needs a GPU) |
| `project/out/rev3_scores_*.npz` | caches | `13_analysis.py` rebuilds them |
| `project/test images/` | 105 MB | the outside check's 340 radiographs carry a non-commercial licence; `18_outside_check.py --download` fetches the same ones |
| `project/.venv/` | 5.1 GB | see Setup in `project/README.md` |

`09_fix_opnorm.py` runs from what is here. `13_analysis.py` needs the feature
files for constructions D and D++; everything else in it runs from
`probsraw_*.npy` and `splits.csv`.

## Setup

Three things to know before you start: the pinned torch 2.5.1 has no Python
3.14 wheels, torchxrayvision ships a module called `config` that shadows a
local `config.py` (ours is named `cxrcfg.py` for that reason), and on Windows
you must set `PYTHONUTF8=1`. `project/README.md` covers all three.
