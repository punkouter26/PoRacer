# Bug creatures in MuJoCo (Quad, Hexapod, Crab)

MuJoCo models of the three ML-Agents "bug" creatures, built straight from their Unity
prefabs so that method C (MuJoCo / mujoco_warp training) trains **the same animal** that
races in Unity (Plan-TrainingMethodComparison.md, Phase 2).

| File | What it is |
|---|---|
| `prefab_to_mjcf.py` | Reads a Unity prefab (read-only) and writes the models below |
| `<name>.xml` | The MJCF to train with |
| `<name>_literal.xml` | Same model with Unity's damping inside the force clamp. For comparison only: unstable at 5 ms (see "Damping") |
| `<name>_rig.json` | Single source of truth for twin checks: bodies, masses, geoms, joint order, limits, gains, action scaling, gait tables, notes |
| `check_bug_mjcf.py` | Validation (counts, mass, heights, drop-and-hold, gait replay, fidelity, mujoco_warp on cuda) |

## Regenerate and check

```powershell
.\.venv-mjwarp\Scripts\python.exe training\bugs\prefab_to_mjcf.py            # all three (or name one: Quad_v01)
.\.venv-mjwarp\Scripts\python.exe training\bugs\check_bug_mjcf.py            # add --no-warp to skip the GPU step
```

It needs `pyyaml` in `.venv-mjwarp` (`uv pip install --python .venv-mjwarp\Scripts\python.exe pyyaml`).
Nothing under `Assets/` is written. The Unity editor is not used.

Sources that are read: `Assets/Prefabs/{Quad,Hexapod,Crab}_v01.prefab`,
`Assets/Settings/CreatureCatalog.asset` (spawn heights), and any `.physicMaterial` a
collider points to. None of the three prefabs points to one.

## Frames

Unity is left-handed, Y-up, +X right, +Z forward. MuJoCo is right-handed, Z-up, +X forward, +Y left.
This is the same map as `Assets/unity_export/MujocoBiped/CONTRACT.md`:

| Quantity | Unity → MuJoCo |
|---|---|
| positions, offsets (true vectors) | `(x, y, z) → (z, -x, y)`, det = -1 |
| hinge axes, angular velocity, torque (pseudovectors) | `(x, y, z) → (-z, x, -y)` |
| quaternions | `(x, y, z, w) → (w, -z, x, -y)` |
| box half-sizes | `(sx, sy, sz) → (sz, sx, sy)` |

Because the axes pick up the extra sign flip, **a positive Unity `jointPosition` or `xDrive.target` is a
positive MuJoCo `qpos` or `ctrl` for the same physical motion.** No sign flips are needed when you
replay Unity actions. Worked example: Quad hip, Unity axis +X. +20° swings the foot toward Unity -Z
(backward). The MuJoCo axis is +Y, and +20° swings the foot toward MuJoCo -X (backward). They match.

Side check: `prefab_to_mjcf.py` confirms that every link with Unity x < 0 (the creature's left)
ends up at MuJoCo y > 0 (left). It passes for all three.

## Units: ArticulationDrive

The Unity 6 Scripting API is explicit about these units:
- `ArticulationDrive.stiffness`: "N/m for linear and Nm/radian for angular motion"
- `ArticulationDrive.damping`: "Ns/m for linear and Nms/rad for angular motion"
- `ArticulationDrive.target`: "meters for linear and degrees for angular motion". Limits are also in degrees.

So stiffness is **per radian, not per degree**. Only the target and limits are in degrees.
`kp = stiffness` and damping pass through unchanged. Targets and limits are converted from degrees
to radians. `Agent_Creature` agrees with this: it reads `jointPosition` in radians
(`jointPosition / (_jointDriveScale * Deg2Rad)`) and writes `xDrive.target = action * _jointDriveScale` in degrees.

## How each Unity field maps

| Unity | MJCF |
|---|---|
| Root ArticulationBody (joint type Fixed, not immovable) | `<freejoint>` |
| Revolute ArticulationBody | `<joint type="hinge">`. Axis = `anchorRotation * X` in the child frame. Pivot = `anchorPosition` scaled by the link's lossy scale |
| `m_Twist` Limited, `m_XDrive.lower/upperLimit` | `range` in radians |
| `m_XDrive.stiffness` | `<position kp>` |
| `m_XDrive.forceLimit` | `forcerange = ±forceLimit` |
| `m_XDrive.damping` | `<joint damping>` (see "Damping") |
| `_jointDriveScale` = 45° | `ctrlrange = ±45°` in radians. Unity never clamps the target to the joint limit: the Hexapod/Crab hips are limited to ±40° and can be driven at 45°, which presses into the limit. The ctrl range matches that |
| `_joints` order | actuator order. `ctrl[i]` is Unity action `i` × 0.785 rad |
| `m_Mass` + implicit COM/tensor | geom `mass`. MuJoCo computes COM and inertia from the collider shape, as PhysX does for a solid shape. Mass is split across colliders by volume when a body has several (none here do) |
| CapsuleCollider | Radius × max(the two non-axis lossy scales); total height × the axis scale, clamped to ≥ 2r. When the height collapses to 2r, PhysX makes a sphere, and so does the MJCF (all 6 Hexapod/Crab upper links) |
| BoxCollider | Box with half-sizes = size × lossy scale / 2 |
| Parent/child collision filtering | MuJoCo `filterparent` (default). All non-adjacent link pairs collide (rule M) |
| Gravity, timestep | -9.81, 0.005 s (`TimeManager` Fixed Timestep 0.005) |

The converter also checks that the parent and child anchors meet at the same world point (worst
mismatch: 1e-7 m). It checks that every joint is at zero in the prefab pose, so no `ref` offsets are needed.

## Damping: the one deliberate change from the literal model

The literal twin puts Unity's damping inside the torque clamp: `<position kp kv forcerange>`.
PhysX treats a drive as `clamp(k·(target−q) − d·q̇, ±forceLimit)` and solves it implicitly.
MuJoCo integrates `kp` explicitly. With these gains (kp 3500–4500 N·m/rad on 2–4 kg links,
kv 233–300), the literal form is **numerically unstable at 5 ms**. Replaying each creature's own
coded gait gives joint speeds of 61–242 rad/s, limit overshoot of up to 50°, and legs passing
28–33 mm into each other (Quad, Hexapod). These artifacts shrink with the step size, but at 1 ms the
model is still 5–7° RMS away from the 0.5 ms result.

`<name>.xml` therefore keeps `kp` and `forcerange` on the actuator and moves Unity's damping to
`<joint damping>`. MuJoCo integrates joint damping implicitly, so this is stable at 5 ms.
`check_bug_mjcf.py` measures the cost against the literal model run at 0.5 ms, where it has
converged, over 1 s of the coded gait:

| | Quad | Hexapod | Crab |
|---|---|---|---|
| this model @ 5 ms, joint RMS / root-height RMS | 2.0° / 12 mm | 2.0° / 8 mm | 1.0° / 26 mm |
| literal model @ 5 ms | 18° / 215 mm | 27° / 104 mm | 8° / 44 mm |
| max joint speed: reference / this model | 11 / 3.8 rad/s | 23 / 7.1 rad/s | 7.1 / 3.1 rad/s |

The remaining gap: damping torque sits outside the force clamp here. When a drive saturates, the
joint moves more slowly than in Unity (see the max-speed row). If a later twin check against Unity
shows this matters, the options are substeps (1 ms physics, 5 ms control) with the literal model,
or a custom clamped implicit PD.

## Friction and contact

- **Friction μ = 0.6 on every geom, floor included.** None of the creature colliders has a material
  (`m_Material {fileID: 0}`). `DynamicsManager.m_DefaultMaterial` is also empty, so Unity uses its
  built-in default: static 0.6, dynamic 0.6, bounce 0, combine Average. The ML-Agents training floor
  is `new PhysicsMaterial("TrainingGround")` (`Systems_TrainingArea.cs`), which has the same values.
  The race tracks in `SCN_RACE_FLAT` / `SCN_TRAIN_*` have `_physicsMaterial: {fileID: 0}`.
  MuJoCo has one sliding coefficient and combines by max. With every geom at 0.6, the combine rule
  makes no difference.
  **This is below the AGENTS §2E target** (μs 0.8–1.0, μd 0.6–0.8) and below the Plan's 0.8 / 0.6.
  The twin copies what Unity actually does. `Assets/Art/Materials/PM_Worm.physicsMaterial`
  (0.9 / 0.8, combine Maximum) matches AGENTS, but nothing references it.
- Contacts use `solref="0.01 1"` (time constant = 2 × step, the stiffest stable setting) on geoms
  and joint limits. PhysX contacts and articulation limits are rigid. With MuJoCo's default 0.02,
  the Hexapod sank 10 mm when it landed after spawn. With 0.01 it sinks 2.2 mm.
- Unity also uses a 0.01 m contact offset, the TGS solver (12 position / 4 velocity iterations) and
  enhanced determinism. MuJoCo uses its own soft-contact solver, `implicitfast`, and a pyramidal cone.
  These solvers are not the same, and only the Phase 2 twin check can measure the difference.

## Not modelled (on purpose, or no MuJoCo equivalent)

| Unity | Value | Why it is left out |
|---|---|---|
| `m_JointFriction` | 0.05 | A PhysX coefficient on the joint constraint force, with no unit match to MuJoCo `frictionloss` (N·m) |
| `m_LinearDamping` / `m_AngularDamping` | 0.05 /s | Per-body velocity drag, about 5 %/s. Negligible next to the drives and contacts |
| `maxAngularVelocity` | 20 rad/s | Set at runtime by `Agent_Creature.Initialize` (root and every link). MuJoCo has no clamp. The model peaks at 3–7 rad/s during the coded gait, but a trained policy may go faster |
| Fatigue | stiffness and forceLimit × 0.55–1.0 | Runtime rule in `Agent_Creature.ApplyFatigueToDrives`. It belongs in the training env, not the model |
| Race quirks, training friction randomisation | per spawn | Runtime randomisation in `Systems_Spawn` and `Systems_TrainingArea` |
| Visual meshes | Unity capsule/cube meshes, non-uniformly scaled | Collision shapes only. In Unity, the Hexapod/Crab upper-leg mesh is an ellipsoid while its collider is a sphere of r = 0.18 m |

## Creature notes

- **Quad_v01**: 90 kg (body 60, uppers 4 × 4, lowers 4 × 3.5). Eight fore-aft (pitch) hinges.
  Hips ±45°, knees ±60°. kp 4500, kd 300, 900 N·m.
- **Hexapod_v01**: 70 kg (body 40, uppers 6 × 3, lowers 6 × 2). Twelve pitch hinges. Hips ±40°,
  knees ±50°. kp 3500, kd 233.3, 700 N·m. The legs sit at Unity z = ±0.817 m, well outside the
  body box (half-length 0.35 m). They are attached only through the joints. That is faithful to the prefab.
- **Crab_v01**: the same body, masses and gains as the Hexapod. The difference is the hips:
  `anchorRotation` is 90° about Z, so the hips turn about the leg's own vertical axis (yaw) instead of pitching.
- **Heights.** The catalog `spawnHeight` is described as "root height at rest". That is true for the
  Quad: 0.85 + the spawner's 0.05 equals exactly 0.900, where the feet touch. It is **not** true for the
  Hexapod and Crab. They rest at a root height of **0.526 m**, but they spawn at 0.80 m and fall
  0.27 m every time. `Editor_WalkExam` uses `spawnHeight` as the leg length for its Froude speed
  target, so their target speeds are about 19 % too high (√(0.75 / 0.526)). The MJCF has both
  keyframes: `spawn` (the Unity spawn height) and `rest` (feet on the floor).

## Replaying Unity actions (for the twin check)

`ctrl[i] = clip(action[i], −1, 1) × radians(45)`, with actuators in `_joints` order. Unity holds
actions for `DecisionPeriod` = 20 physics steps (0.1 s) with `TakeActionsBetweenDecisions`.
`Agent_Creature.Heuristic` uses the absolute `Time.fixedTime`, so align t = 0 when you compare
gaits. `check_bug_mjcf.py` replays that gait from the `rest` keyframe for 5 s and reports forward
and lateral travel as the MuJoCo side of the twin check. The Unity side has not been measured yet.
