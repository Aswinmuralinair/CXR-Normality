# Pipeline: is "this chest X-ray is normal" a number you can trust?

Hospitals auto-clear chest X-rays as normal so that no radiologist reads them.
The verdict is built by combining a model's per-finding probabilities into one
score. The individual probabilities get validated, but the combined number does not,
and its operating threshold carries no stated guarantee. This pipeline
measures both, on seven ways of building the score, four pretrained DenseNet-121
checkpoints ("backbones" below), 112,120 images and 30,805 patients.

The main `README.md` maps each section of the paper to the step behind it.

## Setup (once)

The pinned torch 2.5.1 has no wheels for Python 3.14, so install 3.12 alongside it.

```
winget install -e --id Python.Python.3.12
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements-lock.txt --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

`requirements-lock.txt` pins the exact versions the reported numbers were
produced with (Python 3.12.10, torch 2.5.1+cu121, torchxrayvision 1.5.4). If a
pinned wheel is unavailable on your platform, install torch and torchvision
from the cu121 index first and then the rest without version pins.

Set `PYTHONUTF8=1` before every run. torchxrayvision's progress bar prints a
character the cp1252 console cannot encode, the crash interrupts the weight
download, and the half-written `.pt` file then fails to load.

Kaggle token: kaggle.com, Settings, API, Create New API Token. Save `kaggle.json`
to `%USERPROFILE%\.kaggle\kaggle.json`.

## Run order

| step | script | what it does |
|---|---|---|
| 0 | `00_download.py --resized` | ChestX-ray14 at 224 px (about 2.5 GB) from Kaggle |
| 1 | `01_make_splits.py` | patient-level 60/15/25 train/calib/test split |
| 2 | `02_build_cache.py` | decode every image once into a 5.6 GB uint8 memmap |
| 3 | `03_extract.py --batch 32 --throttle 0.01` | four pretrained models: raw outputs plus 1024-d features (needs a GPU) |
| 9 | `09_fix_opnorm.py` | inverts torchxrayvision's op_norm warp, writes `probsraw_*.npy` (mandatory) |
| 13 | `13_analysis.py` | the whole analysis, writes `out/rev3_*.csv` and `out/_rev3_numbers.txt` |
| 14 | `14_figures.py` | four summary figures, drawn from the step 13 tables, into `figs/` |
| 15 | `15_manifest.py` | `out/RUN_MANIFEST.json`: versions, seed, model weights, SHA-256 of the main inputs and outputs |
| 16 | `16_demo_build.py` | optional: fits what the demo needs and checks it against step 13, writes `out/demo_models.pkl` |
| 17 | `17_demo.py` | optional: the demo page at http://127.0.0.1:8765 |
| 18 | `18_outside_check.py` | optional: 340 radiographs from another hospital through the demo, writes `out/outside_*.csv` (`--download` fetches them first) |
| 19 | `19_design_effect.py` | optional: how much patient clustering weakens the risk-control bounds, writes `out/design_effect.csv` (needs the step 13 score caches) |
| 20 | `20_guarantee_variants.py` | optional: variants of the Learn-Then-Test guarantee (1,000 candidates, several fixed-sequence starts, one image per patient, sample sizes divided by the design effect, a pneumothorax limit, subgroups by sex, age and view, Platt calibration) and a check that step 9's inversion exactly undoes the library's op_norm, writes `out/variants_*.csv` (needs the step 13 score caches and `data/Data_Entry_2017.csv`) |

Step numbers skip 4 to 8 and 10 to 12; those steps are not part of this
release. `figs/` holds the four figures step 14 writes.

Smoke-test step 3 before the full run: `python 03_extract.py --limit 6400 --models pc --batch 32`

After step 3 nothing touches the images again. Steps 9, 13 and 14 re-run from
the cached outputs. `out/probs_*.npy` holds the raw library outputs and is not a
probability array; see step 9.

## Tests

```
python -m pytest tests -q
```

Eight checks that need no images: no patient crosses a split, the derived
"no finding" flag matches the fourteen labels, the op_norm inversion recovers
raw probabilities exactly and leaves untrained labels as NaN, the
empirical-target threshold reaches its target on the set it admits (ties
included, on synthetic scores and on the stored ones), the closed-form
constructions compute what their formulas say, and the seven primary labels exist
in every backbone's label list. Step 9's exact thresholds are in
`out/op_threshs.json`. The check on the stored scores reads the step 13 score
caches, which are not in the repository, so in a fresh clone it is skipped
(7 pass, 1 skipped) until `13_analysis.py` has run.

## The demo (steps 16 and 17)

Upload one chest X-ray and the page shows what every normality score in the
study would decide about it. It is a research demonstration and not a clinical
tool: the thresholds were chosen on ChestX-ray14 and do not carry over to
another hospital's images.

```
python 17_demo.py           # http://127.0.0.1:8765, opens the browser
python 17_demo.py --cpu     # stay on the CPU even when a GPU is present
```

The repository ships `out/demo_models.pkl` and seven test-split sample images
in `demo/samples/`, so the demo runs without the dataset. The four
torchxrayvision weights download on first use. It needs no packages beyond
`requirements-lock.txt`; the page is plain HTML served by the standard library.

The page opens in a simple view: the PadChest model, three of
the scores (A, B_cal and D), two rules (the empirical target as the "quick
rule" and Learn-Then-Test with Bonferroni as the "safe rule"), targets of 0.95
and 0.98, plain words, and for dataset images a note on whether each decision
matches the report labels. The switch at the top opens the detailed view with every
checkpoint, construction and procedure. `?view=detailed` opens it directly.

For each image the server runs all four checkpoints the way step 3 ran the
dataset, undoes op_norm with step 9's own function, scores the seven
constructions, and applies every threshold the study uses: the empirical
target at 0.95, 0.98 and 0.99, and both Learn-Then-Test procedures at 0.95 and
0.98 on the five risk-control constructions. When the file name is a
ChestX-ray14 image index, the page also shows the image's split, its report
labels and the stored research scores next to the live ones.

`16_demo_build.py` rebuilds `out/demo_models.pkl`. Step 13 fits the isotonic
calibrators and the learned combiners inside `build_scores` and keeps only
their scores, so this step refits them on train with the hyper-parameters step
13 selected. It saves nothing unless three checks pass: the refitted components
reproduce step 13's cached scores for all 112,120 images exactly, every
threshold gives the coverage, NPV and n in `rev3_main.csv` and
`rev3_riskcontrol.csv`, and each sample image gets the same verdict live as
from the stored scores. `13_analysis.py` is not modified. `demo_core.py` loads
the pipeline's own functions for preprocessing, the op_norm inversion and the
threshold rules.

`python 17_demo.py --check 200` runs 200 test images through the code the page
uses and compares pixels, model outputs, probabilities, scores and verdicts
with the stored files (it needs the images and the step 13 score caches). On a
CPU the network runs in float32, while the stored outputs came from a float16
GPU run, so scores can differ by a few hundredths and a study that close to a
threshold can change verdict. On 500 test images that happened in 9 of 82,000
verdicts.

To share the demo outside your own machine, tunnel port 8765 with a tool such
as ngrok.

## The outside check (step 18)

```
python 18_outside_check.py --download   # once: picks and fetches the 340 radiographs
python 18_outside_check.py              # scores them and writes the tables
```

Step 18 runs radiographs from a hospital none of the four checkpoints was
trained on through the demo's own analysis, with every threshold unchanged,
and compares how often each threshold clears normal and abnormal studies there
with the same rates on the ChestX-ray14 test split. The radiographs come from
the Indiana University chest X-ray collection (Open-i, US National Library of
Medicine; Demner-Fushman et al., JAMIA 2016), read from the Hugging Face copy
`ykumards/open-i`. With the project seed it draws 200 exams whose report is
coded only "normal" and 20 for each of the seven findings, leaving out codes
marked borderline. The images carry a non-commercial licence, so they stay in
`test images/`, which git ignores, together with `labels.csv` (each report)
and `results.csv` (every score and decision). The copy's server limits how
often it answers, so the download goes one request at a time and resumes if
stopped. The tables go to `out/outside_rules.csv`, `out/outside_auc.csv` and
`out/outside_by_finding.csv`, which the paper's Indiana results (Section IV-F)
come from.

The exams were chosen by label and are not a consecutive series, so the step
cannot measure the NPV a threshold delivers at that hospital. It shows whether
the ranking carries over and how the share of studies a threshold clears
changes at a new site.

## The seven constructions (common seven-label estimand)

| | name | formula | fitted on |
|---|---|---|---|
| A | Fréchet upper bound | `1 - max(p)` | nothing. This is the published construction (Schalekamp 2024). |
| B | Independence | `prod(1 - p)` | nothing |
| A_cal | Bound on calibrated marginals | `1 - max(q)`, `q` = per-label isotonic(p) | isotonic on train |
| B_cal | Product on calibrated marginals | `prod(1 - q)` | isotonic on train |
| C | Learned linear | logistic on the 7 logits | train; C chosen on a train-tune slice |
| C++ | Learned MLP | MLP on the 7 logits | train; width and alpha chosen on train-tune |
| D | Direct | logistic on the 1024 frozen features | train; C chosen on train-tune |

Calib is used for one thing, choosing thresholds. Test is touched once per
reported number, except for the swap/re-split diagnostic, which chooses thresholds on test by design and is labelled as a diagnostic.

## Design rules

1. Split by patient, never by image. One patient contributes 184 images.
   `01_make_splits.py` asserts that no patient crosses splits.
2. Never treat a library's output as a probability without checking. Print the
   per-column distribution. A mean far above the finding's prevalence means
   op_norm rescaling, and a column fixed at exactly 0.5 is a label the model was
   not trained on. Run step 9.
3. Do not lead with the NIH-trained model. It was trained on this dataset. It
   stays in every table as a labelled row and is excluded from every count.
4. Choose thresholds on calib and report on a locked test set. A threshold chosen
   on test is an oracle.
5. A threshold that hits an empirical target is a point estimate. It misses its
   target about half the time by construction. Say so, or use Learn-Then-Test
   and pay the coverage.
6. Fix the multiple-testing procedure in advance and report every one you fixed.
   Bonferroni and fixed-sequence give coverage that differs by about 2.5x. Picking
   the better one after seeing it is the post-hoc selection the study is about.
7. Bootstrap by patient. `13_analysis.py` resamples patients.
8. A diagnostic that pools calib and test is a diagnostic. Never present it as a
   held-out result.

## What "normal" means here

Primary estimand: none of the seven labels every backbone predicts (atelectasis,
cardiomegaly, consolidation, edema, effusion, pneumonia, pneumothorax). That is
72% of images and it counts a lung mass as normal. Secondary estimand: none of
the 14 NIH labels (54%). Neither is "clinically healthy chest".

## Reading the output

`out/rev3_main.csv`: AUC, ECE, Brier and FCR@5% with patient-cluster confidence
intervals, plus coverage, realised NPV and n at empirical-target thresholds for
0.95, 0.98 and 0.99.

`out/rev3_riskcontrol.csv`: naive vs Bonferroni vs fixed-sequence coverage on the
locked test split. Zero coverage is a real result.

`out/rev3_resplit_diagnostic.csv`: the swap and re-split diagnostic. Not a
held-out result.

`out/rev3_bottleneck*.csv`: the 2x2 (representation x function class) with paired
cluster-bootstrap intervals.

`out/rev3_dependence.csv`: residual correlation raw vs calibrated, and the
top-decile check on B_cal.

`out/_rev3_numbers.txt`: everything above as printed.
