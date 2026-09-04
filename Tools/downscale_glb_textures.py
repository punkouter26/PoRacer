#!/usr/bin/env python3
"""
Downscale the images embedded in a .glb, in place.

WHY THIS EXISTS
---------------
glTFast imports a .glb through a ScriptedImporter, which builds its Texture2D
sub-assets itself. Those textures never pass through a TextureImporter, so they
have no platform overrides and no compression settings, and setting
EditorUserBuildSettings.androidBuildSubtarget to ASTC cannot reach them. They
ship as raw RGBA with mipmaps no matter what the build settings say.

That is not a theory. Assets/Boy_Character_mujoco.glb is 30.6 MB on disk and
expanded to 250.2 MB in the Android build - 82% of a 306 MB payload - and the
commit that turned on ASTC moved the texture total from 261.8 MB to only
261.4 MB. Nothing else in the build pipeline can shrink these bytes, so the one
lever left is the source resolution inside the file itself.

Caps are per-image because the authored art is not uniform, and they follow the
reasoning already written down in IsaacBoxMaterials.cs for this same character:
metallic/roughness is low-frequency data that is indistinguishable at 512, while
the base colour on the head is the one map a player actually looks at.

The .glb keeps its GUID, so every prefab and scene reference survives. The
original is recoverable with: git checkout -- <path>
"""

import argparse
import io
import json
import struct
import sys
from pathlib import Path

from PIL import Image

# Cap by image name. The first pattern that is a substring of the name wins, so
# order matters - the more specific keys are listed first.
DEFAULT_CAPS = [
    ("Metallic", 512),          # low-frequency gloss data; 4096 of it is wasted
    ("Roughness", 512),
    ("_Normal", 512),           # tangent-space detail, seen at phone scale
    ("Head_BaseColor", 1024),   # the map a player actually looks at
    ("_BaseColor", 1024),
]
FALLBACK_CAP = 1024

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


def cap_for(name, caps):
    for pattern, cap in caps:
        if pattern in name:
            return cap
    return FALLBACK_CAP


def gpu_bytes(width, height):
    """Uncompressed RGBA32 with a full mip chain, which is what Unity ships."""
    return width * height * 4 * 4 / 3


def read_glb(path):
    raw = path.read_bytes()
    magic, version, _length = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC:
        raise ValueError(str(path) + " is not a binary glTF (bad magic)")
    offset = 12
    doc = None
    blob = b""
    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        data = raw[offset:offset + chunk_len]
        if chunk_type == CHUNK_JSON:
            doc = json.loads(data)
        elif chunk_type == CHUNK_BIN:
            blob = data
        offset += chunk_len
        offset += (-chunk_len) % 4
    if doc is None:
        raise ValueError(str(path) + " has no JSON chunk")
    return version, doc, blob


def write_glb(path, version, doc, blob):
    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    blob += b"\x00" * ((-len(blob)) % 4)

    total = 12 + 8 + len(json_bytes) + (8 + len(blob) if blob else 0)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, version, total)
    out += struct.pack("<II", len(json_bytes), CHUNK_JSON)
    out += json_bytes
    if blob:
        out += struct.pack("<II", len(blob), CHUNK_BIN)
        out += blob
    path.write_bytes(bytes(out))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args()

    if not args.glb.exists():
        print("ERROR: " + str(args.glb) + " not found", file=sys.stderr)
        return 1

    version, doc, blob = read_glb(args.glb)
    views = doc.get("bufferViews", [])
    images = doc.get("images", [])
    if not images:
        print("No embedded images; nothing to do.")
        return 0

    # bufferView index -> replacement bytes, for the image views only.
    replacements = {}
    before_gpu = 0.0
    after_gpu = 0.0
    rows = []

    for image in images:
        if "bufferView" not in image:
            continue
        view_index = image["bufferView"]
        view = views[view_index]
        start = view.get("byteOffset", 0)
        data = blob[start:start + view["byteLength"]]

        name = image.get("name", "image" + str(view_index))
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            width, height = source.size
            cap = cap_for(name, DEFAULT_CAPS)
            before_gpu += gpu_bytes(width, height)

            if max(width, height) <= cap:
                after_gpu += gpu_bytes(width, height)
                rows.append((name, width, height, width, height,
                             len(data), len(data), "kept"))
                continue

            scale = cap / float(max(width, height))
            new_size = (max(1, int(round(width * scale))),
                        max(1, int(round(height * scale))))
            resized = source.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            # Keep the original mime type. Every image in these files is PNG, and
            # switching to JPEG would destroy the alpha the base-colour maps carry.
            resized.save(buffer, format="PNG", optimize=True)
            new_bytes = buffer.getvalue()

        replacements[view_index] = new_bytes
        after_gpu += gpu_bytes(new_size[0], new_size[1])
        rows.append((name, width, height, new_size[0], new_size[1],
                     len(data), len(new_bytes), "resized"))

    name_width = max(len(r[0]) for r in rows)
    header = "image".ljust(name_width) + "       was          now          png            action"
    print(header)
    for name, ow, oh, nw, nh, ob, nb, action in rows:
        print("%s  %5dx%-5d  %5dx%-5d  %6.2f->%6.2fMB  %s"
              % (name.ljust(name_width), ow, oh, nw, nh,
                 ob / 1048576.0, nb / 1048576.0, action))
    print("")
    print("GPU payload (RGBA32 + mips, which is what Unity ships):")
    print("  before %8.1f MB" % (before_gpu / 1048576.0))
    print("  after  %8.1f MB" % (after_gpu / 1048576.0))
    print("  saved  %8.1f MB" % ((before_gpu - after_gpu) / 1048576.0))

    if args.dry_run:
        print("")
        print("--dry-run: nothing written.")
        return 0
    if not replacements:
        print("")
        print("Every image is already within its cap; nothing written.")
        return 0

    # Re-lay-out the whole binary chunk. Every bufferView offset shifts once an
    # image shrinks, and the mesh/accessor views must be carried across intact,
    # so walk them in their existing storage order and re-pack with the 4-byte
    # alignment glTF requires for accessor offsets.
    order = sorted(range(len(views)), key=lambda i: views[i].get("byteOffset", 0))
    packed = bytearray()
    for view_index in order:
        view = views[view_index]
        if view_index in replacements:
            data = replacements[view_index]
        else:
            start = view.get("byteOffset", 0)
            data = blob[start:start + view["byteLength"]]
        packed += b"\x00" * ((-len(packed)) % 4)
        view["byteOffset"] = len(packed)
        view["byteLength"] = len(data)
        packed += data

    packed += b"\x00" * ((-len(packed)) % 4)
    doc["buffers"][0]["byteLength"] = len(packed)

    before_file = args.glb.stat().st_size
    write_glb(args.glb, version, doc, bytes(packed))
    after_file = args.glb.stat().st_size
    print("")
    print("Wrote %s: %.1f MB -> %.1f MB on disk"
          % (args.glb, before_file / 1048576.0, after_file / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
