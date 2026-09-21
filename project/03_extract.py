"""
Step 3 - one pass over the cached images per backbone.

Crash safe. Results stream straight to disk memmaps and progress is
checkpointed, so after a crash the same command continues from the last
checkpoint instead of starting over.

  python 03_extract.py                     # all models, resumes automatically
  python 03_extract.py --models pc         # one model
  python 03_extract.py --throttle 0.05     # gentler on a machine that runs hot
  python 03_extract.py --limit 5000        # smoke test

Writes per model:
  out/probs_<model>.npy    N x L float16   raw model outputs. torchxrayvision
                                            applies op_norm() inside forward(), so
                                            these are not probabilities. Step 9
                                            inverts the warp into probsraw_*.npy.
  out/feats_<model>.npy    N x 1024 float16
  out/labels_<model>.json  label names + which NIH-14 are usable
  out/_ckpt_<model>.json   progress marker (delete to force a clean rerun)
"""
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from cxrcfg import OUT, CACHE, CACHE_INDEX, MODELS, NIH14

CKPT_EVERY = 100          # batches between checkpoints (~3200 images)


def get_features(model, x):
    """Penultimate 1024-d DenseNet features: conv trunk, ReLU, global average pool."""
    f = model.features(x)
    f = F.relu(f, inplace=True)
    return F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)


def open_out(path, shape, dtype=np.float16):
    """Reopen an existing memmap or create it. Resuming costs nothing because of this."""
    if path.exists():
        m = np.lib.format.open_memmap(path, mode="r+")
        if m.shape == shape:
            return m
        del m
        path.unlink()
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def run_model(key, weights, arr, idx, device, batch, throttle):
    """One backbone over every cached image, streaming outputs to disk memmaps
    and checkpointing progress so a crash costs at most CKPT_EVERY batches."""
    ck = OUT / f"_ckpt_{key}.json"
    start = json.loads(ck.read_text())["done"] if ck.exists() else 0
    if start >= len(idx):
        print(f"=== {key}: already complete ({start:,}) ===")
        return

    print(f"\n=== {key} ({weights}) ===")
    if start:
        print(f"  resuming from image {start:,}/{len(idx):,}")

    model = xrv.models.DenseNet(weights=weights).to(device).eval()
    paths = list(model.pathologies)
    P = open_out(OUT / f"probs_{key}.npy", (len(idx), len(paths)))
    Z = open_out(OUT / f"feats_{key}.npy", (len(idx), 1024))

    t0, last, nb = time.time(), start, 0
    for s in range(start, len(idx), batch):
        rows = idx[s:s + batch]
        x = torch.from_numpy(
            xrv.datasets.normalize(arr[rows].astype(np.float32), 255)
        ).unsqueeze(1).to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x)
            z = get_features(model, x)
        out = out.float()
        # Defensive only: pretrained torchxrayvision models already return values
        # in [0, 1] (op_norm-warped), so this branch does not fire for them.
        if out.min() < 0 or out.max() > 1:
            out = torch.sigmoid(out)

        P[s:s + len(rows)] = out.cpu().numpy().astype(np.float16)
        Z[s:s + len(rows)] = z.float().cpu().numpy().astype(np.float16)

        nb += 1
        if nb % CKPT_EVERY == 0 or s + batch >= len(idx):
            P.flush(); Z.flush()
            done = min(s + batch, len(idx))
            ck.write_text(json.dumps({"done": int(done), "total": int(len(idx))}))
            torch.cuda.empty_cache()
            dt = time.time() - t0
            rate = (done - last) / max(dt, .01)
            eta = (len(idx) - done) / max(rate, 1)
            print(f"  {done:,}/{len(idx):,}  {rate:6.1f} img/s  eta {eta/60:5.1f} min", flush=True)
            t0, last = time.time(), done      # rate is per interval, so a stall shows up at once
        if throttle:
            time.sleep(throttle)

    # Which of the NIH 14 does this backbone predict? A column that never moves
    # belongs to a label the weights were not trained on.
    usable = []
    for i, p in enumerate(paths):
        if p not in NIH14:
            continue
        col = np.asarray(P[:, i], dtype=np.float32)
        if np.nanstd(col) < 1e-4 or np.all(np.isnan(col)):
            print(f"  [skip] '{p}' untrained (std={np.nanstd(col):.2e})")
            continue
        usable.append(p)

    (OUT / f"labels_{key}.json").write_text(json.dumps(
        {"pathologies": paths, "usable_nih14": usable, "n_rows": int(len(idx))}, indent=2))
    print(f"  done. usable NIH-14 ({len(usable)}): {usable}")
    del model, P, Z
    torch.cuda.empty_cache()


def main():
    """Open the image cache, pick the rows to process, and run each requested
    backbone over them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--batch", type=int, default=32, help="32 needs less GPU memory than 64")
    ap.add_argument("--throttle", type=float, default=0.0, help="seconds slept per batch")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    print(f"batch={a.batch}  throttle={a.throttle}s  checkpoint every {CKPT_EVERY} batches")

    arr = np.load(CACHE, mmap_mode="r")
    cached = pd.read_csv(CACHE_INDEX)["cached"].values
    idx = np.flatnonzero(cached == 1)
    if a.limit:
        idx = idx[:a.limit]
    np.save(OUT / "valid_idx.npy", idx)
    print(f"images: {len(idx):,}")

    for k in a.models:
        run_model(k, MODELS[k], arr, idx, device, a.batch, a.throttle)


if __name__ == "__main__":
    main()
