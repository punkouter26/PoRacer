"""Skeleton extraction from a glTF-binary rig, in MuJoCo's frame.

Reads Assets/Boy_Character_mujoco.glb, recovers the rest-pose world transform of
every bone two independent ways, cross-checks them, and hands back joint
positions already converted to MuJoCo's Z-up right-handed frame.

Skin weights and meshes are ignored on purpose: the visuals stay in Unity and
only the skeleton crosses into MJCF.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# glTF is Y-up right-handed, MuJoCo is Z-up right-handed: (x, y, z) -> (x, -z, y).
_GLTF_TO_MJC = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]]
)

# The authored character faces glTF +Z, which the frame change alone would land on
# MuJoCo -Y. Measured, not assumed: the Shoes mesh extends 0.168 m one way from the
# ankle bone and 0.062 m the other, and the long side is the toes. A 180 deg yaw
# turns him to face MuJoCo +Y, which org.mujoco maps to Unity +Z -- the race
# direction -- so the prefab needs no yaw correction the way Fido's does.
_YAW_180 = np.array(
    [[-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, 1.0]]
)

GLTF_TO_MJC = _YAW_180 @ _GLTF_TO_MJC

# Torso-local axes at the rest pose, shared by the MJCF, the trainer and Unity.
FORWARD = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
LATERAL = np.array([1.0, 0.0, 0.0])

_COMPONENT_DTYPE = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass(frozen=True)
class Bone:
    """One rig bone, resolved to its rest pose in MuJoCo coordinates."""

    name: str
    parent: str | None
    pos: np.ndarray  # world rest position, MuJoCo frame, metres


class Rig:
    def __init__(self, bones: dict[str, Bone], order: list[str], residual: float):
        self.bones = bones
        self.order = order
        # Max |inverse(IBM) - forward-kinematics| over all bones, metres. A large
        # value means the two derivations disagree and the rig is not trustworthy.
        self.residual = residual

    def pos(self, name: str) -> np.ndarray:
        return self.bones[name].pos

    def local(self, name: str) -> np.ndarray:
        """Position of a bone relative to its parent, MuJoCo frame."""
        bone = self.bones[name]
        if bone.parent is None:
            return bone.pos.copy()
        return bone.pos - self.bones[bone.parent].pos

    def seg_len(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.pos(b) - self.pos(a)))


def _read_glb(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as handle:
        magic, version, _total = struct.unpack("<III", handle.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a GLB (bad magic {magic:#x})")
        if version != 2:
            raise ValueError(f"{path} is glTF {version}, expected 2")
        gltf: dict | None = None
        binary = b""
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            chunk_len, chunk_type = struct.unpack("<II", header)
            payload = handle.read(chunk_len)
            if chunk_type == 0x4E4F534A:
                gltf = json.loads(payload.decode("utf-8"))
            elif chunk_type == 0x004E4942:
                binary = payload
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk")
    return gltf, binary


def _accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    if "sparse" in acc:
        raise NotImplementedError("sparse accessors are not supported")
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPE[acc["componentType"]])
    ncomp = _TYPE_COUNT[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * ncomp
    rows = []
    for element in range(acc["count"]):
        offset = start + element * stride
        rows.append(np.frombuffer(binary, dtype=dtype, count=ncomp, offset=offset))
    return np.array(rows, dtype=np.float64)


def _trs(node: dict) -> np.ndarray:
    """Local matrix of a glTF node, column-vector convention (M @ v)."""
    if "matrix" in node:
        # glTF stores matrices column-major.
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    mat = np.eye(4)
    if "scale" in node:
        mat[:3, :3] = mat[:3, :3] @ np.diag(node["scale"])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        mat[:3, :3] = rot @ mat[:3, :3]
    if "translation" in node:
        mat[:3, 3] = node["translation"]
    return mat


def load(path: str | Path = "Assets/Boy_Character_mujoco.glb", skin_index: int = 0) -> Rig:
    gltf, binary = _read_glb(Path(path))
    nodes = gltf["nodes"]
    skin = gltf["skins"][skin_index]
    joints = skin["joints"]

    parent_of: dict[int, int] = {}
    for node_index, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = node_index

    # Derivation A: forward kinematics down the node hierarchy from the scene root.
    world: dict[int, np.ndarray] = {}

    def resolve(node_index: int) -> np.ndarray:
        if node_index in world:
            return world[node_index]
        local = _trs(nodes[node_index])
        parent = parent_of.get(node_index)
        mat = local if parent is None else resolve(parent) @ local
        world[node_index] = mat
        return mat

    # Derivation B: invert the inverse bind matrices.
    ibm = _accessor(gltf, binary, skin["inverseBindMatrices"])
    bind: dict[int, np.ndarray] = {}
    for slot, node_index in enumerate(joints):
        bind[node_index] = np.linalg.inv(ibm[slot].reshape(4, 4).T)

    residual = 0.0
    bones: dict[str, Bone] = {}
    order: list[str] = []
    joint_set = set(joints)
    for node_index in joints:
        name = nodes[node_index].get("name") or f"node{node_index}"
        fk = resolve(node_index)[:3, 3]
        bm = bind[node_index][:3, 3]
        residual = max(residual, float(np.max(np.abs(fk - bm))))
        parent_index = parent_of.get(node_index)
        parent_name = None
        if parent_index in joint_set:
            parent_name = nodes[parent_index].get("name") or f"node{parent_index}"
        bones[name] = Bone(name=name, parent=parent_name, pos=GLTF_TO_MJC @ bm)
        order.append(name)

    return Rig(bones=bones, order=order, residual=residual)


def mesh_points(path: str | Path, mesh_name: str) -> np.ndarray:
    """Every vertex of a named mesh, in MuJoCo coordinates. Used to size colliders."""
    gltf, binary = _read_glb(Path(path))
    for mesh in gltf["meshes"]:
        if mesh.get("name") != mesh_name:
            continue
        chunks = [_accessor(gltf, binary, prim["attributes"]["POSITION"]) for prim in mesh["primitives"]]
        return np.vstack(chunks) @ GLTF_TO_MJC.T
    raise KeyError(f"no mesh named {mesh_name!r}")


if __name__ == "__main__":
    rig = load()
    print(f"bones={len(rig.order)}  fk-vs-ibm residual = {rig.residual:.3e} m")
    print(f"{'bone':<14}{'parent':<14}{'x':>9}{'y':>9}{'z':>9}{'|local|':>10}")
    for name in rig.order:
        bone = rig.bones[name]
        x, y, z = bone.pos
        span = float(np.linalg.norm(rig.local(name))) if bone.parent else 0.0
        print(f"{name:<14}{str(bone.parent):<14}{x:9.4f}{y:9.4f}{z:9.4f}{span:10.4f}")
