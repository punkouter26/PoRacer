#!/usr/bin/env python3
"""
decimate_meshes.py - build a low-poly LOD of the Isaac visual meshes.

The meshes `extract_meshes.py` pulls out of `robot/usd/instanceable_meshes.usd` are the
full-fat CAD geometry: 1 603 512 vertices / 534 504 triangles per creature, fully
un-welded (`verts == 3 * tris`, i.e. flat-shaded). That is fine for one hero robot and
far too much for a grid of racers - PoRacer's own perf rung measures 8 H1s sitting right
on the 60 FPS budget, with the meshes and inference dominating, not the solver.

This writes a decimated copy of every blob, same format, into `Meshes/decimated/`.
The originals are left untouched, so `IsaacH1 > Import Original Meshes` always restores
full detail.

Method: **vertex clustering**. Snap every vertex to a uniform grid, weld the cells,
drop triangles that collapse to a degenerate, then re-split the survivors so each
triangle keeps its own three vertices - which preserves the crisp flat-shaded look of a
machined part instead of smoothing it into a blob. Cell size is binary-searched per link
to hit that link's share of the triangle budget.

Vertex clustering rather than quadric edge collapse because it needs nothing but numpy,
runs in seconds, and is deterministic. QEM would hold silhouettes better at the same
budget; at these viewing distances it is not worth a new dependency.

Usage:  python decimate_meshes.py [--budget 20000] [--in-dir Meshes] [--out-dir Meshes/decimated]
Requires: numpy.
"""
from __future__ import annotations

import argparse
import glob
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC = b"IH1M"
VERSION = 1

# Below this a link is already cheap and further loss only costs silhouette.
MIN_TRIS_PER_LINK = 150


def read_blob(path):
    with open(path, "rb") as f:
        magic, version, nv, nt = struct.unpack("<4siii", f.read(16))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version != VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        verts = np.frombuffer(f.read(nv * 12), dtype="<f4").reshape(nv, 3).astype(np.float64)
        f.read(nv * 12)          # normals are recomputed after decimation
        tris = np.frombuffer(f.read(nt * 12), dtype="<i4").reshape(nt, 3).astype(np.int64)
    return verts, tris


def write_blob(path, verts, normals, tris):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iii", VERSION, len(verts), len(tris)))
        f.write(verts.astype("<f4").tobytes())
        f.write(normals.astype("<f4").tobytes())
        f.write(tris.astype("<i4").tobytes())


def cluster(verts, tris, cell):
    """Weld vertices onto a `cell`-sized grid; return (positions, triangles)."""
    keys = np.floor(verts / cell).astype(np.int64)
    # unique cell -> representative position = mean of the vertices that landed in it,
    # which sits closer to the original surface than the cell centre does.
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n = counts.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64)
    np.add.at(acc, inverse, verts)
    pos = acc / counts[:, None]

    t = inverse[tris]
    # a triangle whose corners fell into fewer than three cells has no area left
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return pos, t[keep]


def solve_cell(verts, tris, target, lo, hi, iterations=24):
    """Binary-search the smallest cell size whose triangle count is <= target."""
    best = None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        _, t = cluster(verts, tris, mid)
        if len(t) <= target:
            best = mid
            hi = mid          # try to keep more detail
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return best if best is not None else hi


def flat_split(pos, tris):
    """Give every triangle its own three vertices and a true face normal."""
    corners = pos[tris]                                     # (nt, 3, 3)
    verts = corners.reshape(-1, 3)
    fn = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = np.divide(fn, np.maximum(ln, 1e-12))
    normals = np.repeat(fn, 3, axis=0)
    out_tris = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    return verts, normals, out_tris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=20000,
                    help="total triangles across all links (default 20000)")
    ap.add_argument("--in-dir", default=os.path.join(HERE, "Meshes"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "Meshes", "decimated"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.ih1mesh")))
    if not files:
        raise SystemExit(f"no .ih1mesh files in {args.in_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    meshes = [(os.path.splitext(os.path.basename(p))[0], *read_blob(p)) for p in files]
    total_in = sum(len(t) for _, _, t in meshes)

    # Budget split by original triangle count, floored so small links stay recognisable.
    raw = {name: max(MIN_TRIS_PER_LINK, int(round(args.budget * len(t) / total_in)))
           for name, _, t in meshes}
    overshoot = sum(raw.values()) / args.budget
    if overshoot > 1.0:      # floors pushed us over; scale the non-floored ones back
        for name in raw:
            raw[name] = max(MIN_TRIS_PER_LINK, int(raw[name] / overshoot))

    rows, out_v, out_t = [], 0, 0
    for name, verts, tris in meshes:
        target = raw[name]
        extent = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
        cell = solve_cell(verts, tris, target, lo=extent * 1e-4, hi=extent)
        pos, t = cluster(verts, tris, cell)
        v, n, t = flat_split(pos, t)
        write_blob(os.path.join(args.out_dir, f"{name}.ih1mesh"), v, n, t)
        rows.append((name, len(tris), len(t), cell))
        out_v += len(v)
        out_t += len(t)

    w = max(len(r[0]) for r in rows)
    print(f"wrote {len(rows)} decimated meshes to {args.out_dir}")
    print(f"{'link'.ljust(w)}   tris in   tris out   ratio   cell (mm)")
    for name, ti, to, cell in rows:
        print(f"{name.ljust(w)}   {ti:7d}   {to:8d}   {ti / max(to, 1):5.1f}x   {cell * 1000:7.2f}")
    print(f"{'TOTAL'.ljust(w)}   {total_in:7d}   {out_t:8d}   {total_in / max(out_t, 1):5.1f}x")
    print(f"\nvertices {out_v:,} (was {sum(len(v) for _, v, _ in meshes):,})")
    print("Now run  IsaacH1 > Import Decimated Meshes  in the editor.")


if __name__ == "__main__":
    main()
