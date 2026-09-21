"""
Step 2 - decode every PNG once into a uint8 memmap at 224x224.

112,120 x 224 x 224 x 1 byte = 5.6 GB, small enough to sit in RAM on an
ordinary machine, so no later pass decodes an image again. Decoding once is
the biggest speed-up in the pipeline.

The script resumes. Rerun it after an interruption and it continues where it
stopped.
"""
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from cxrcfg import IMAGES, OUT, CACHE, CACHE_INDEX, IMG_SIZE, NUM_WORKERS

cv2.setNumThreads(1)          # the thread pool below does the parallelism


def load_one(path):
    """One PNG to a 224x224 uint8 grayscale array. cv2 decodes much faster than PIL."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # INTER_AREA is the right filter for shrinking an image
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def main():
    """Decode every image listed in splits.csv into the memmap, resuming from the
    index file if a previous run was interrupted."""
    splits = pd.read_csv(OUT / "splits.csv")
    names = splits["Image Index"].tolist()
    n = len(names)

    # Resume support: cache_index.csv records which rows are already decoded.
    # If it matches the current split file row for row, continue from there.
    done_from = 0
    if CACHE.exists() and CACHE_INDEX.exists():
        prev = pd.read_csv(CACHE_INDEX)
        if len(prev) == n and (prev["Image Index"] == splits["Image Index"]).all():
            done_from = int(prev["cached"].sum())
            print(f"resuming at {done_from:,}/{n:,}")

    arr = np.lib.format.open_memmap(
        CACHE, mode="r+" if CACHE.exists() else "w+",
        dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE))

    cached = np.zeros(n, dtype=np.int8)
    if done_from:
        cached[:done_from] = 1

    # Decode in chunks of 2048 with a thread pool, flush the memmap and the
    # index after every chunk so an interruption loses at most one chunk.
    missing, CHUNK = [], 2048
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for start in range(done_from, n, CHUNK):
            batch = names[start:start + CHUNK]
            for j, img in enumerate(ex.map(load_one, [IMAGES / b for b in batch])):
                idx = start + j
                if img is None:
                    missing.append(batch[j])
                    continue
                arr[idx] = img
                cached[idx] = 1
            arr.flush()
            pd.DataFrame({"Image Index": names, "cached": cached}).to_csv(CACHE_INDEX, index=False)
            done = min(start + CHUNK, n)
            print(f"\r{done:,}/{n:,}  ({100*done/n:5.1f}%)  missing={len(missing)}", end="")

    print()
    if missing:
        pd.Series(missing).to_csv(OUT / "missing_images.csv", index=False, header=["Image Index"])
        print(f"[warn] {len(missing)} images missing -> out/missing_images.csv")
        print("       These are dropped downstream. If the count is large, your download is incomplete.")
    print(f"wrote {CACHE}  ({arr.nbytes/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
