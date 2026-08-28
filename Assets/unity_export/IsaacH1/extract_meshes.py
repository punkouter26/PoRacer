#!/usr/bin/env python3
"""
extract_meshes.py - pull the ORIGINAL Isaac visual meshes out of the export and
write them in Unity coordinates for IsaacH1MeshImporter to turn into Mesh assets.

The vendor URDF points at `package://h1_description/meshes/*.STL`, and those STL
files are NOT in the export (nor anywhere in the Isaac Lab tree). The real visual
geometry lives only inside `robot/usd/instanceable_meshes.usd`, referenced by
`h1_minimal.usd` - 39 Mesh prims, ~1.6M vertices before de-duplication.

What this does per link:
  * collects every Mesh prim under `<link>/visuals` (through instance proxies),
  * bakes each prim's transform into its vertices, relative to the LINK's frame at
    the zero-joint pose - the same frame IsaacH1RigBuilder places the link object in,
  * applies the Isaac -> Unity frame map, and REVERSES triangle winding, because the
    map has det = -1 and would otherwise turn every surface inside out,
  * triangulates n-gons as a fan,
  * writes one little-endian binary per link.

Format (`Meshes/<link>.ih1mesh`), all little-endian:
    magic   4s   b"IH1M"
    version i32  1
    nVerts  i32
    nTris   i32
    verts   f32 * 3 * nVerts    (Unity coords, metres, link-local)
    normals f32 * 3 * nVerts    (Unity coords, unit)
    tris    i32 * 3 * nTris     (Unity winding)

Usage: python extract_meshes.py [--export-dir ../../h1] [--out-dir Meshes]
Requires: usd-core, numpy.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPORT = os.path.normpath(os.path.join(HERE, "..", "..", "h1"))

MAGIC = b"IH1M"
VERSION = 1


def to_unity_pos(v: np.ndarray) -> np.ndarray:
    """M : isaac (x, y, z) -> unity (-y, z, x). Matches IsaacH1FrameMap.Pos."""
    out = np.empty_like(v)
    out[:, 0] = -v[:, 1]
    out[:, 1] = v[:, 2]
    out[:, 2] = v[:, 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default=DEFAULT_EXPORT)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "Meshes"))
    args = ap.parse_args()

    try:
        from pxr import Usd, UsdGeom, Gf  # noqa: F401
    except ImportError:
        sys.exit("ERROR: usd-core is required.  pip install usd-core")
    from pxr import Usd, UsdGeom

    usd_path = os.path.join(args.export_dir, "robot", "usd", "h1_minimal.usd")
    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        sys.exit("ERROR: expected a Z-up stage")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        sys.exit("ERROR: expected metersPerUnit == 1.0")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # link world transforms at the zero-joint pose
    link_world = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Xform" and "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas():
            link_world[prim.GetName()] = cache.GetLocalToWorldTransform(prim)

    # collect visual meshes per link, de-duplicated by prim path (instance proxies
    # surface the same source geometry once per instance)
    per_link = {}
    seen = set()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.GetTypeName() != "Mesh":
            continue
        path = str(prim.GetPath())
        if "/visuals" not in path or not path.startswith("/h1/"):
            continue
        link = path.split("/")[2]
        if link not in link_world:
            continue
        key = (link, path)
        if key in seen:
            continue
        seen.add(key)
        per_link.setdefault(link, []).append(prim)

    os.makedirs(args.out_dir, exist_ok=True)
    total_v = total_t = 0
    rows = []

    for link, prims in sorted(per_link.items()):
        L_inv = link_world[link].GetInverse()
        verts_all, tris_all = [], []
        base = 0

        for prim in prims:
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts or not indices:
                continue

            P = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)

            # bake this prim's transform, expressed in the LINK's frame
            M = cache.GetLocalToWorldTransform(prim) * L_inv
            m = np.array([[M[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
            # USD is row-vector: v' = v * M
            P = P @ m[:3, :3] + m[3, :3]

            # triangulate as a fan
            tris = []
            k = 0
            for c in counts:
                if c >= 3:
                    for i in range(1, c - 1):
                        tris.append((indices[k], indices[k + i], indices[k + i + 1]))
                k += c
            if not tris:
                continue
            T = np.array(tris, dtype=np.int64) + base

            verts_all.append(P)
            tris_all.append(T)
            base += len(P)

        if not verts_all:
            continue

        V = np.concatenate(verts_all, axis=0)
        T = np.concatenate(tris_all, axis=0)

        V = to_unity_pos(V)
        # det(M) < 0 flips handedness, so reverse winding to keep faces outward
        T = T[:, ::-1].copy()

        # smooth vertex normals from the triangles (the USD normals would need the
        # same handedness fix and are face-varying; recomputing is simpler and safe)
        N = np.zeros_like(V)
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        fn = np.cross(b - a, c - a)
        for i in range(3):
            np.add.at(N, T[:, i], fn)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        N = np.divide(N, np.maximum(ln, 1e-12))

        out = os.path.join(args.out_dir, f"{link}.ih1mesh")
        with open(out, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<iii", VERSION, len(V), len(T)))
            f.write(V.astype("<f4").tobytes())
            f.write(N.astype("<f4").tobytes())
            f.write(T.astype("<i4").tobytes())

        total_v += len(V)
        total_t += len(T)
        rows.append((link, len(prims), len(V), len(T), os.path.getsize(out)))

    w = max(len(r[0]) for r in rows)
    print(f"wrote {len(rows)} meshes to {args.out_dir}")
    print(f"{'link'.ljust(w)}  prims    verts     tris     bytes")
    for link, np_, nv, nt, sz in rows:
        print(f"{link.ljust(w)}  {np_:5d}  {nv:7d}  {nt:7d}  {sz:9d}")
    print(f"{'TOTAL'.ljust(w)}  {sum(r[1] for r in rows):5d}  {total_v:7d}  {total_t:7d}  "
          f"{sum(r[4] for r in rows):9d}")
    if total_v > 65535:
        print("\nNOTE: some meshes exceed 65535 vertices; the importer sets "
              "IndexFormat.UInt32 on every mesh it builds.")


if __name__ == "__main__":
    main()
