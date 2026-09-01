#!/usr/bin/env python3
"""
cg_decode.py — Phase A gate: determine the encoding of liquidation_leverage_data.

The 672 KB payload cannot be inspected from a chat session (it exceeds context,
the sandbox cannot reach api.apify.com, and the signed dataset URL is refused by
the fetcher). So the encoding must be established here, on the VM, before any
number downstream can be trusted.

Every downstream metric depends on getting this right. If this script cannot
classify the encoding confidently it says so and exits non-zero. Do not proceed
past a non-zero exit by guessing.

Usage:
    python3 cg_decode.py snapshots/<cycle>/BTC_model1_24h.json
"""

import json, sys
from collections import Counter
import numpy as np


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d[0] if isinstance(d, list) else d


def classify(grid, n_y):
    """Return (encoding, detail). Encodings: dense_ty, dense_yt, sparse_triplet."""
    if not grid:
        return "empty", {}
    lens = Counter(len(r) if isinstance(r, (list, tuple)) else -1 for r in grid)
    if len(lens) != 1:
        return "ragged", {"row_length_histogram": dict(lens)}
    row_len = next(iter(lens))
    n_rows = len(grid)

    if row_len == n_y:
        return "dense_ty", {"n_time": n_rows, "n_price": row_len}
    if n_rows == n_y:
        return "dense_yt", {"n_price": n_rows, "n_time": row_len}
    if row_len == 3:
        a = np.asarray(grid, dtype=float)
        return "sparse_triplet", {
            "n_cells": n_rows,
            "col0_range": [float(a[:, 0].min()), float(a[:, 0].max())],
            "col1_range": [float(a[:, 1].min()), float(a[:, 1].max())],
            "col2_range": [float(a[:, 2].min()), float(a[:, 2].max())],
            "col0_unique": int(np.unique(a[:, 0]).size),
            "col1_unique": int(np.unique(a[:, 1]).size),
        }
    return "unknown", {"n_rows": n_rows, "row_len": row_len, "n_y": n_y}


def current_column(grid, enc, n_y):
    """Extract the latest time slice — the profile that gets frozen as forecast."""
    g = np.asarray(grid, dtype=float)
    if enc == "dense_ty":
        return g[-1]
    if enc == "dense_yt":
        return g[:, -1]
    if enc == "sparse_triplet":
        # Whichever of col0/col1 has ~n_y distinct values is the price index.
        c0u, c1u = np.unique(g[:, 0]).size, np.unique(g[:, 1]).size
        t_col, y_col = (0, 1) if abs(c1u - n_y) < abs(c0u - n_y) else (1, 0)
        tmax = g[:, t_col].max()
        prof = np.zeros(n_y)
        sel = g[g[:, t_col] == tmax]
        for _, r in enumerate(sel):
            yi = int(r[y_col])
            if 0 <= yi < n_y:
                prof[yi] = r[2]
        return prof
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: cg_decode.py <snapshot.json>")
    item = load(sys.argv[1])
    y = np.asarray(item["y_axis"], dtype=float)
    grid = item["liquidation_leverage_data"]
    enc, detail = classify(grid, y.size)

    step = np.diff(y)
    print("=== Y AXIS ===")
    print(f"  levels        : {y.size}")
    print(f"  range         : {y[0]:.2f} .. {y[-1]:.2f}  (span {y[-1]-y[0]:.2f})")
    print(f"  step          : {step[0]:.4f}  uniform={bool(np.allclose(step, step[0], rtol=1e-6))}")
    mid = (y[0] + y[-1]) / 2
    print(f"  span as % mid : {(y[-1]-y[0])/mid*100:.2f}%   (i.e. +/-{(y[-1]-y[0])/mid*50:.2f}%)")

    print("\n=== GRID ===")
    print(f"  encoding      : {enc}")
    for k, v in detail.items():
        print(f"  {k:<14}: {v}")

    if enc in ("ragged", "unknown", "empty"):
        print("\nGATE FAILED — encoding not classified. Do not proceed to scoring.")
        return 2

    prof = current_column(grid, enc, y.size)
    if prof is None or prof.sum() <= 0:
        print("\nGATE FAILED — could not extract a usable current-column profile.")
        return 2

    nz = prof > 0
    p = prof / prof.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    print("\n=== CURRENT COLUMN (the frozen forecast) ===")
    print(f"  non-zero cells: {int(nz.sum())} / {y.size}  ({nz.mean()*100:.1f}%)")
    print(f"  intensity     : min={prof[nz].min():.4g} med={np.median(prof[nz]):.4g} max={prof.max():.4g}")
    print(f"  peak at price : {y[int(np.argmax(prof))]:.2f}")
    print(f"  entropy       : {ent:.4f} nats  (uniform would be {np.log(y.size):.4f})")
    print(f"  concentration : {np.exp(ent)/y.size:.3f}  (1.0 = flat, ->0 = spiked)")
    print("\nGATE PASSED — encoding classified, profile extractable.")
    print("Record this block verbatim in the study's data dictionary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
