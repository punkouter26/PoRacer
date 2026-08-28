# CONTRACT — Isaac Lab spider → Unity (PhysX 4 / Inference Engine)

Source of truth: `Assets/spider/checkpoint/params/env.yaml`, `Assets/spider/export_report.json`,
`Assets/spider/source/spider_env.py`. Where the URDF and the Isaac cfg disagree, the **Isaac cfg wins**
(noted below). Everything here is implemented in `Runtime/IsaacSpiderAgent.cs`.

## ONNX

| | |
|---|---|
| file | `spider.onnx` (231 988 bytes, weights embedded, opset 15, IR 8) |
| input | `obs` float32 `[1, 59]` |
| output | `actions` float32 `[1, 16]` — Gaussian **mean**, unclamped (values up to ±5 occur); clamp to [−1, 1] before use |
| normaliser | observation normalisation is baked into the graph (RSL-RL `obs_normalization: true`) |
| check | `check_onnx.py`: 200 recorded steps, max \|onnxruntime − Isaac\| = **6.2e-6** (PASS < 1e-4); in-engine (`RunReferenceCheck`, CPU backend) **5.96e-6** |
| network | MLP 256-128-64, ELU |

## Observation (59 floats) — `SpiderEnv._get_observations()`

| idx | n | content | Unity source |
|---|---|---|---|
| 0–15 | 16 | joint position [rad], policy joint order | `ArticulationBody.jointPosition[0]` |
| 16–31 | 16 | joint velocity [rad/s] × **0.1** | `jointVelocity[0] * 0.1` |
| 32–34 | 3 | root linear velocity in the body frame [m/s] | `UnityToRos(body.InverseTransformDirection(root.linearVelocity))` |
| 35–37 | 3 | root angular velocity in the body frame [rad/s] × **0.2** | `-UnityToRos(body.InverseTransformDirection(root.angularVelocity)) * 0.2` (pseudo-vector: sign flips with handedness) |
| 38–40 | 3 | gravity direction in the body frame (unit; (0,0,−1) upright) | `UnityToRos(body.InverseTransformDirection(Vector3.down))` |
| 41–42 | 2 | target − root, in the body's **yaw-only** frame: x forward, y left [m] | forward = body `transform.forward` flattened; left = (−fwd.z, 0, fwd.x) |
| 43–58 | 16 | previous action (clamped values applied last step) | cached `_prevAction` |

At reset everything is 0 except gravity (0,0,−1) and the target offset.

## Action (16 floats)

`q_target[j] = action_scale · clamp(a[j], −1, 1)` with `action_scale = 0.8` rad. Same joint order as obs 0–15.

## Joint order and sign convention

`L1_hip, L1_knee, L2_hip, L2_knee, L3_hip, L3_knee, L4_hip, L4_knee, R1_hip, R1_knee, R2_hip, R2_knee, R3_hip, R3_knee, R4_hip, R4_knee`

Joint `X_hip` drives link `X_femur`, `X_knee` drives `X_tibia` (URDF child links; the ArticulationBody lives on the child).
Limits: hips ±0.8 rad, knees ±1.0 rad (URDF = Isaac; `soft_joint_pos_limit_factor = 1.0`).

Hip axis = URDF `(0,0,1)` (body Z, Unity Y); knee axis = URDF `(0,1,0)` in the femur frame (Unity −X after the map).
The revolute ArticulationBody rotates about its anchor **X**; the anchor X is set to **−M·axis**, so a positive
Unity joint angle is the same physical rotation as a positive Isaac joint angle — no sign flip anywhere in the
agent. Verified by `Kinematics_JointSignAndFrameMapMatchIsaacFk` (three poses, 0.0 mm against an independent
Python FK).

## Frame map (URDF Importer convention)

| | Isaac / ROS (Z-up, RH) | Unity (Y-up, LH) |
|---|---|---|
| vectors | (x, y, z) | (−y, z, x) |
| inverse | (u.z, −u.x, u.y) | (x, y, z) |
| quaternion | (x, y, z, w) | (y, −z, −x, w) |
| pseudo-vectors (ω) | ω | −(−ω.y, ω.z, ω.x) — the extra minus is the handedness flip |
| body forward | +x | +z (`transform.forward`) |
| body left | +y | −x |

`isaac_reference.json` quaternions are **xyzw** (the export labelled them wxyz; `rig_audit.py` proves xyzw against obs[41:42], max err 1.5e-6).

## Control rate and physics

| | Isaac (`env.yaml`) | Unity |
|---|---|---|
| physics dt | 1/120 s | project: **0.02 s** (see deviations) |
| decimation | 4 | `round(policyDt / Time.fixedDeltaTime)`; error logged if not integer |
| policy dt | 1/30 s | 1/30 s |
| solver | PhysX 5 TGS (`solver_type: 1`), pos 8 / vel 0 iterations | PhysX 4.1 TGS (project `m_SolverType: 1`), **per body** pos 8 / vel 0 |
| gravity | (0, 0, −9.81) | (0, −9.81, 0) — same |
| actuator | `ImplicitActuator` = PhysX joint drive, kp 25 N·m/rad, kd 1 N·m·s/rad, maxForce 15 N·m, drive type force | default `ArticulationDrive` (same PhysX mechanism, stiffness 25 / damping 1 / forceLimit 15 / Force); optional `TorqueCSharp` explicit PD via `jointForce` |
| velocity limit | `velocity_limit_sim = 12` → USD `maxJointVelocity = 687.5°/s`, **not effective** (reference shows \|q̇\| up to 89.8 rad/s, p99 up to 45) | `maxJointVelocity = 100` (link cap), `enforceVelocityLimit = false` — matches what Isaac actually did |
| link caps | maxLinearVelocity 100, maxAngularVelocity 100, maxDepenetrationVelocity 10 | same, per body |
| damping | linear 0, angular 0.05 (PhysX default; cfg null), joint friction 0, armature 0 | same, per body |
| CCD / gyroscopic | ccd off, `enable_gyroscopic_forces: true` | Discrete; PhysX 4 articulations have no gyroscopic toggle (deviation, minor at these speeds) |
| self-collision | off | `Physics.IgnoreCollision` across all 17 colliders |
| masses | body 2.0, femur 0.1, tibia 0.12 kg (URDF = USD) | raw, `massFloor = 0` |
| inertia | URDF diagonal in the inertial frame | same axes; `inertiaFloor = 1e-4` raises only the roll (long-axis) component 2e-5 / 1.35e-5 → 1e-4 |
| colliders | body sphere r 0.1; cylinders femur r 0.02 L 0.16, tibia r 0.015 L 0.26 (USD convex) | sphere; capsules height L + r on unscaled child objects |
| contact offset | PhysX 5 default 0.02 m, rest 0 | 0.02 m per collider (project default is 0.01) |
| spawn | (0, 0, 0.18) m, joints 0 | (0, 0.18, 0) |
| standing height | 0.141 m | 0.142 m measured (rung 1) |

## Target

Isaac: ring sampler r ∈ [1.5, 3.5] m around the env origin, z = 0.12, reach when horizontal distance < 0.3 m.
Unity: `Transform target` / `ITargetProvider` first; the ring sampler is only the fallback. The policy only sees
obs 41–42, so any world position works without re-export.

## Unity deviations (this project)

| item | Isaac | Unity (PoRacer) | effect / mitigation |
|---|---|---|---|
| fixed step | 1/120 | 0.02 (not a divisor of 1/30) | policy runs every 2 steps = 40 ms → walked 0.91 m/s in one run, fell in another. **Recommended: 1/60** (decimation 2, exact) — a project-wide change, see README table |
| ground friction | 0.5 / 0.5, restitution 0, combine average | race ground has **no physics material** → Unity default 0.6 / 0.6 | `PM_IsaacSpider` on the spider colliders uses **Minimum** combine → pair friction stays 0.5 |
| ground geometry | infinite plane | flat map: plane + 1 m box slab, top y = 0; hills map: mesh terrain (sine hills) — policy never saw slopes | flat map only for parity; expect falls on hills |
| contact offset | 0.02 | project default 0.01 | set to 0.02 on the spider colliders (per collider, not global) |
| solver iterations | 8 / 0 | project default 12 / 4 | set per body on the spider only |
| solver | PhysX 5 TGS | PhysX 4.1 TGS | articulation drives are implicit in both; PhysX 4 bang-bang drives pump CoM momentum in zero-g (rung 3: 0.40 m/s at 0.02, 0.21 at 1/60, 0.12 at 1/120, 0.015 torque at 1/480) |
| other agents | none | ML-Agents racers: `Agent_Creature` sets `maxAngularVelocity = 20` on their bodies, race spawner stacks racers; their drives are their own | untouched; the spider's per-body values do not affect them. Racers run at 0.02 s — changing the fixed step changes **their** dynamics (CLAUDE.md) |
| layer | filter_collisions across envs | project defines no `IsaacSpider` layer; spider stays on Default and collides with racers | add the layer (TagManager change) to filter — requires confirmation |
| gyroscopic forces | on | not available | minor |
| joint velocity | reached 90 rad/s | reached 75–83 rad/s in rung 5 | consistent |
| speed | 2.9 m/s mean (eval), 2.76 m/s (reference) | 0.5–1.06 m/s (rung 5, 8 s window, all substeps/actuators) | main open sim-to-sim gap; see README |
