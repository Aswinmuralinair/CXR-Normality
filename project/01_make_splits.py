"""
Step 1 - patient-level splits. Needs only Data_Entry_2017.csv, no images, so it
can run before the image download finishes.

Every image of one patient lands in exactly one of train / calib / test
(60 / 15 / 25 of patients). Splitting by image would leak, because most
images are follow-ups of the same person and one patient has 184 of them.

Writes out/splits.csv with columns: Image Index, patient_id, split, <14 labels>, no_finding
"""
import numpy as np
import pandas as pd
from cxrcfg import CSV, OUT, NIH14, SEED

RNG = np.random.default_rng(SEED)


def main():
    """Read the label CSV, multi-hot the 14 findings, assign each patient to one
    split, and write out/splits.csv with a few sanity figures printed."""
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}. Download Data_Entry_2017.csv first (it is ~7 MB).")

    df = pd.read_csv(CSV)
    df = df.rename(columns=lambda c: c.strip())

    # Patient id: the CSV has it directly, but derive from filename as a fallback.
    if "Patient ID" in df.columns:
        df["patient_id"] = df["Patient ID"].astype(int)
    else:
        df["patient_id"] = df["Image Index"].str.split("_").str[0].astype(int)

    # ---- multi-hot the 14 findings from the "Finding Labels" column
    labels = df["Finding Labels"].fillna("")
    for lab in NIH14:
        df[lab] = labels.str.contains(rf"(?:^|\|){lab}(?:\||$)", regex=True).astype(np.int8)

    # "No Finding" == none of the 14 present. Derive it rather than trusting the
    # string, so the definition is explicit and auditable.
    df["no_finding"] = (df[NIH14].sum(axis=1) == 0).astype(np.int8)

    string_nf = labels.str.strip().eq("No Finding").astype(np.int8)
    mismatch = int((df["no_finding"] != string_nf).sum())
    if mismatch:
        print(f"[warn] {mismatch} rows where derived no_finding != literal 'No Finding' string")

    # ---- patient-level split. Never split by image: 112k images come from only
    # ~30k patients and most are follow-ups, so an image split leaks.
    patients = np.array(sorted(df["patient_id"].unique()))
    RNG.shuffle(patients)
    n = len(patients)
    n_tr, n_va = int(0.60 * n), int(0.15 * n)
    assign = {}
    for p in patients[:n_tr]:
        assign[p] = "train"          # fits the calibrators and the learned constructions
    for p in patients[n_tr:n_tr + n_va]:
        assign[p] = "calib"          # chooses the thresholds, nothing else
    for p in patients[n_tr + n_va:]:
        assign[p] = "test"           # never touched until the final numbers
    df["split"] = df["patient_id"].map(assign)

    keep = ["Image Index", "patient_id", "split"] + NIH14 + ["no_finding"]
    df[keep].to_csv(OUT / "splits.csv", index=False)

    # ---- sanity numbers
    print(f"images            : {len(df):,}")
    print(f"patients          : {n:,}")
    print(f"images/patient    : mean {len(df)/n:.2f}  median {df.groupby('patient_id').size().median():.0f}"
          f"  max {df.groupby('patient_id').size().max()}")
    print(f"no_finding rate   : {df['no_finding'].mean():.4f}")
    abn = df[df["no_finding"] == 0]
    print(f"of abnormal, >=2 findings : {(abn[NIH14].sum(axis=1) >= 2).mean():.4f}")
    print()
    print(df.groupby("split").agg(images=("Image Index", "size"),
                                  patients=("patient_id", "nunique"),
                                  no_finding=("no_finding", "mean")))

    # no patient may appear in two splits
    overlap = df.groupby("patient_id")["split"].nunique().max()
    assert overlap == 1, "patient leaked across splits"
    print("\nOK - no patient appears in more than one split.")
    print(f"wrote {OUT/'splits.csv'}")


if __name__ == "__main__":
    main()
