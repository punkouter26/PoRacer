# Mesh assets

There are none. The robot is built entirely from MuJoCo primitive geoms
(capsules, boxes, a sphere and a ground plane), so `biped.xml` has no
`<mesh>` assets and no linked .stl/.obj files to ship.

Collision geoms live in **group 3**, which MuJoCo hides by default. Sites
named `bone_*` in group 4 mark hip, knee, ankle and toe positions for
binding a skinned mesh.

Primitive sizes are in `../robot_spec.json` under `geoms`, which is enough
to rebuild the collision volumes as Unity colliders:
capsule `size = [radius, half_length]`, box `size = [hx, hy, hz]`
(Unity box colliders take full extents, so double these).
