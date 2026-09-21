"""Shared paths and constants for every pipeline script.

Import from here rather than hard-coding paths, so the project can be moved
as a whole. (The file is not called config.py because torchxrayvision ships a
module of that name which would shadow it.)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"          # put Data_Entry_2017.csv + images/ here
OUT = ROOT / "out"            # every result table and array
FIGS = ROOT / "figs"          # step 14's figures; created on demand
IMAGES = DATA / "images"      # all 112,120 PNGs, flat
CSV = DATA / "Data_Entry_2017.csv"

for d in (DATA, OUT):
    d.mkdir(exist_ok=True)

# The 14 ChestX-ray14 findings. "No Finding" means none of these was extracted
# from the report text. That is a weaker statement than "clinically normal
# chest", and the paper says so.
NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

IMG_SIZE = 224
CACHE = DATA / f"cache_{IMG_SIZE}_uint8.npy"   # N x 224 x 224 uint8 memmap
CACHE_INDEX = DATA / "cache_index.csv"          # row order of the memmap

# Backbones. We evaluate on ChestX-ray14, so the primary models must have been
# trained elsewhere. The NIH-trained weights are kept as a leaky contrast: they
# appear in every table and in no summary count. Label overlap with the 14 NIH
# findings, measured from the models' own pathology lists:
#   pc        14 labels   PadChest, the headline backbone
#   chex       7 labels   CheXpert
#   mimic_ch   7 labels   MIMIC-CXR
#   nih       14 labels   trained on this dataset, contrast only
#
# The backbones expose different label sets, so the primary estimand is "none
# of the 7 labels every backbone predicts" (see 13_analysis.py). That way every
# backbone estimates the same event. The 14-label event is a secondary analysis
# on pc and nih only. The product construction shrinks with label count on its
# own (0.99^14 = 0.87 against 0.99^7 = 0.93); that is arithmetic and says nothing
# about dependence.
MODELS = {
    "pc":       "densenet121-res224-pc",        # PadChest, 14 labels, headline
    "chex":     "densenet121-res224-chex",      # CheXpert, 7 labels
    "mimic_ch": "densenet121-res224-mimic_ch",  # MIMIC-CXR, 7 labels
    "nih":      "densenet121-res224-nih",       # leaky contrast only
}
HEADLINE_MODEL = "pc"
PRIMARY_MODELS = ["pc", "chex", "mimic_ch"]
LEAKY_MODEL = "nih"

# Each torchxrayvision model has its own pathology list, which only partly
# overlaps the NIH 14. 03_extract.py writes the resolved list per model to
# out/labels_<model>.json, and every later script reads that file, so the
# "normal" event is always defined over labels the model can predict.

SEED = 0                  # every random choice in the pipeline derives from this
NUM_WORKERS = 12          # decode threads for 02_build_cache.py; lower it on a CPU with fewer threads
