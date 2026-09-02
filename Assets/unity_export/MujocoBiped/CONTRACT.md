# MujocoBiped contract

Everything the Unity port has to reproduce exactly, and everything it deliberately does
not. If the creature misbehaves, this file plus the triage ladder at the bottom is the
fastest route to the cause.

Source: `Assets/biped_sentis/` — a PPO policy trained for 14,499,768 steps in MuJoCo
3.12.0. Port: `Assets/unity_export/MujocoBiped/`. Nothing outside that folder is written.

---

## 1. Frames

MuJoCo is right-handed, Z-up, X-forward. Unity is left-handed, Y-up, Z-forward. One
linear operator does all of it:

```
M : (x, y, z)_mujoco  ->  (-y, z, x)_unity        det(M) = -1
```

`det(M) < 0` — the map reverses handedness — and that single fact drives everything else.

| Quantity | Transforms by | `MujocoBipedFrameMap` |
| --- | --- | --- |
| Position, linear velocity, force, gravity direction | `M` | `Pos` / `PosToMujoco` |
| Angular velocity, torque, rotation axis, quaternion vector part | `-M` | `Axis` / `AxisToMujoco` |
| Quaternion `(w, x, y, z)` | `(y, -z, -x, w)` | `RotFromWxyz` / `RotToMujocoWxyz` |
| Diagonal inertia | permute to `(Iyy, Izz, Ixx)` | `InertiaDiag` |

Pseudovectors pick up an extra sign flip under an orientation-reversing map, which is why
they use `-M` and not `M`. A rotation about axis `a` by angle `t` in MuJoCo is the same
physical motion as a rotation about `-M*a` by `+t` in Unity — so **every revolute anchor
is built with its local +X at `-M*axis`**, and a positive Unity joint angle then *is* a
positive MuJoCo joint angle. The agent never flips a sign reading `qpos` or writing a
torque; the sign convention lives entirely in the anchor.

The quaternion map is the one the export's own README arrived at after an exhaustive
search over all 24 signed axis permutations. The form originally requested there,
`(-x, z, -y, w)`, is **not a rotation map under any position convention** — do not
reintroduce it.

Verified by `RungK_ForwardKinematicsMatchesIndependentPythonFk` against
`kinematics_reference.json`, which `gen_kinematics_reference.py` computes straight from
the MJCF — never from `MujocoBiped_rig.json`, so an error shared by the extraction and the
builder cannot hide in both.

---

## 2. Observation — 49 floats

Assembled index by index in exactly the order `env.py`'s `_get_obs()` concatenates its
terms. **Every clip is part of the trained contract, not a safety measure.**

| Index | Size | Field | Source | Unity |
| --- | --- | --- | --- | --- |
| 0 | 1 | `torso_height` | `qpos[2]` | `root.transform.position.y` |
| 1–3 | 3 | `projected_gravity` | `R^T (0,0,-1)` | `PosToMujoco(invRot * gravity.normalized)` |
| 4–6 | 3 | `linear_velocity` | `clip(R^T qvel[0:3], ±10)` | origin velocity, see below |
| 7–9 | 3 | `angular_velocity` | `clip(R^T qvel[3:6], ±10)` | **double-rotated**, see below |
| 10–21 | 12 | `joint_positions` | `qpos[7:]` | `jointPosition[0]`, radians |
| 22–33 | 12 | `joint_velocities` | `clip(qvel[6:], ±20)` | `jointVelocity[0]`, radians/s |
| 34–35 | 2 | `target_direction` | unit vector rotated by **−yaw only** | planar XZ |
| 36 | 1 | `target_distance` | `min(planar distance, 10)` | planar XZ |
| 37–48 | 12 | `last_action` | previous network output | zeros after a reset |

### The two traps

**`obs[7:10]` is rotated into the torso frame twice.** MuJoCo stores a free joint's
`qvel[3:6]` in the **body-local** frame, and `_get_obs` applies `rot.T` to it regardless:

```python
local_angvel = rot.T @ qvel[3:6]      # qvel[3:6] is ALREADY body-local
```

That is not a defensible modelling choice, but the policy trained on it for 14.5M steps,
so Unity reproduces it exactly — `Quaternion.Inverse(rootRotation)` is applied **twice**.
Feeding the singly-rotated value instead fails silently: the vector has the same
magnitude and differs only when the torso is tilted, which is precisely when the policy
needs it. Proven in `RIG_AUDIT.md` section D (frame-to-frame rotation axis sits 20° from
ω under the body-local hypothesis, 62° under the world one, n=146).

**`obs[4:7]` measures the body-frame ORIGIN, not the centre of mass.** `qvel[0:3]` is
`d/dt qpos[0:3]`, in the world frame. `ArticulationBody.linearVelocity` reports the
*centre of mass* velocity, and the torso's CoM sits 0.185 m above its origin — so at
5 rad/s of pitch the two readings differ by nearly 1 m/s. The agent subtracts `ω × r` to
get back to the origin, with `r` written as `rotation * centerOfMass` rather than as
`worldCenterOfMass − transform.position`. The two are equal in steady state, but
`worldCenterOfMass` is a physics-side property that lags a step behind a teleport — so the
first observation after every fall recovery would carry a wrong velocity, silently.
`LinearVelocityReference.CenterOfMass` exists only so the sweep can measure the difference.

### Clipping is live

33 of the 1800 recorded joint-velocity samples (1.8%) exceed the ±20 rad/s clip — the
ankles routinely swing past it, peaking at 37.14 rad/s. The policy has only ever seen the
clipped value. Removing the clip feeds the network numbers it was never trained on.

---

## 3. Action — 12 floats

The MJCF actuators are `<motor>` elements: **direct torque, no position servo, no PD loop,
no target integration anywhere in the policy.**

```
torque_i (N.m) = action_i * gear_i
```

| Index | Joint | Gear = peak torque | Limits |
| --- | --- | ---: | ---: |
| 0 | `hip_z_l` | 60 N.m | −45° … 45° |
| 1 | `hip_x_l` | 80 N.m | −30° … 15° |
| 2 | `hip_y_l` | 130 N.m | −110° … 45° |
| 3 | `knee_l` | 110 N.m | −150° … −2° |
| 4 | `ankle_y_l` | 70 N.m | −45° … 35° |
| 5 | `ankle_x_l` | 40 N.m | −25° … 25° |
| 6 | `hip_z_r` | 60 N.m | −45° … 45° |
| 7 | `hip_x_r` | 80 N.m | −15° … 30° |
| 8 | `hip_y_r` | 130 N.m | −110° … 45° |
| 9 | `knee_r` | 110 N.m | −150° … −2° |
| 10 | `ankle_y_r` | 70 N.m | −45° … 35° |
| 11 | `ankle_x_r` | 40 N.m | −25° … 25° |

The joint order is identical for the action vector, `obs[10:22]` and `obs[22:34]`.

Torque is written to `ArticulationBody.jointForce` **every physics tick**, not only on
policy ticks, because MuJoCo holds `ctrl` constant across all 5 `frame_skip` substeps.

**`xDrive.stiffness` must stay 0.** Leaving Unity's default position drive alive is the
single most effective way to break this port: it fights every torque the policy emits and
the creature folds up. The drive carries *only* MuJoCo's passive joint damping
(1.0 N·m·s/rad), with a large finite `forceLimit` of 1e6 — MuJoCo never clips passive
damping, and a limit of `gear` would saturate it exactly when the joint is moving fastest.
`float.MaxValue` is **not** a safe stand-in for "no limit": at a 1/120 s step it produced
8.05e6 rad/s of joint velocity where 1e6 gives 65 rad/s at the same amplitude.

### Drive damping is a per-DEGREE gain

Measured, not assumed — `RungD_ActuatorReachesTheSolverAndDampingIsPerRadian`. Unity
documents neither convention, and the drive straddles both: `jointPosition` and
`jointVelocity` are radians while `xDrive.target` and the limits are degrees.

Writing MuJoCo's `damping = 1.0` straight into the drive behaves as **98 N·m·s/rad**.
At 37 rad/s — the fastest joint velocity MuJoCo ever recorded — that is 3600 N·m of
damping against a 110 N·m actuator, and the creature wades instead of walking. Scaling by
`Mathf.Deg2Rad` brings the effective gain to **3.0 N·m·s/rad** (the residual is coupling
through the parent chain, not a units error) and roughly doubles the measured gait speed.
`MujocoBipedAgent.gainUnits` ships as `Degrees` for that reason; `Radians` exists so the
calibration test has something to contrast against.

### Which API delivers the torque

`ArticulationBody.jointForce` writes into the joint's **reduced space**, which is exactly
MuJoCo's actuator semantics, and it ships. The alternative
`ActuatorMode.TorquePairImplicitDamping` applies an equal-and-opposite `AddTorque` on the
child and its parent — the route the export's README recommends. Measured equal on speed
(rung 6: 0.250 vs 0.247 m/s) and worse under extreme actuation (rung 3), so it is the
fallback, not the default.

A warning about measuring this: with joint limits freed, a joint driven at full torque
swings its own shin into the pelvis inside 50 ms and **jams**. Every reading then collapses
to a number indistinguishable from "the torque never arrived" — which is exactly the wrong
conclusion this ladder drew once before `Diag_SingleJointTorqueTimeSeries` separated the
two. Turn self-collision off for any actuator measurement.

---

## 4. Timing

| | MuJoCo | This project |
| --- | --- | --- |
| Physics step | 0.005 s | **0.005 s** — already identical |
| Frame skip / decimation | 5 | **5** |
| Control rate | 40 Hz | 40 Hz |

`policy_dt / Time.fixedDeltaTime = 0.025 / 0.005 = 5` **exactly**. No project setting had
to change, and the agent gets MuJoCo's own substep count per control step, not merely the
right control rate. `MujocoBipedAgent.ComputeDecimation` logs a `LogError` naming the exact
ratio if that ever stops being true, and ships anyway with rounded decimation rather than
silently changing a project-wide setting.

---

## 5. Rig construction

### Multi-hinge bodies become chains

MuJoCo puts three hinges on one `thigh` body and two on one `foot`, composing them
sequentially: `R = R_j1 R_j2 R_j3`, each axis expressed in the frame left by the previous
joint — an intrinsic Z-X-Y Euler chain for the hip.

PhysX cannot do that. Its spherical articulation joint is a single 3-DOF quaternion joint
whose composition is not MuJoCo's, and whose `jointPosition` does not map back to `qpos`.
So each extra hinge gets its own single-DOF link:

```
torso                       (free joint = articulation root)
 +- j_hip_z_l               placeholder, offset (0, 0.09, -0.02)
     +- j_hip_x_l           placeholder, zero offset
         +- thigh_l         mass 5.204, hip_y_l
             +- shin_l      mass 3.359, knee_l
                 +- j_ankle_y_l   placeholder, zero offset
                     +- foot_l    mass 1.380, ankle_x_l
```

13 links = 7 real bodies + 6 placeholders, 12 revolute DOF. The body's own offset goes on
the **first** link of its chain and the rest sit at zero offset, so all of a body's joints
share one anchor point — exactly where MuJoCo puts them.

Placeholders carry 0.01 kg and an explicit 1e-4 inertia floor. Total mass 40.346 kg
against MuJoCo's 40.286 kg: **+0.15%**.

### Armature

MuJoCo's `armature = 0.02` adds to the joint-space mass-matrix **diagonal**,
`H[i][i] += A`. Unity has no such field, so it can only be bought with link inertia — and
link inertia is spatial, so it accumulates up the tree. Adding `A*a*a^T` to every jointed
link (the obvious move) over-counts every parallel-axis run: `hip_y -> knee -> ankle_y`
all turn about the same axis, so the hip would see 3A.

Placing `c_k*a_k*a_k^T` on link k contributes `c_k*(a_i·a_k)^2` to `H[i][i]` for every
ancestor i, which at the zero pose is a triangular system in c. Solved leaf-upward it is
**exact** and needs less than half the added inertia:

| Joint | Naive | Exact |
| --- | ---: | ---: |
| `hip_z` | 0.02 | 0.02 |
| `hip_x` | 0.02 | 0.00 |
| `hip_y` | 0.02 | 0.00 |
| `knee` | 0.02 | 0.00 |
| `ankle_y` | 0.02 | 0.02 |
| `ankle_x` | 0.02 | 0.02 |

`ArmatureMode.Exact` ships. `None` and `Naive` exist for the rung-6 sweep.

### Colliders

Each collider sits on its own **unscaled** child object, so a tilted capsule can be rotated
rather than approximated — `thigh_l`'s capsule runs along `(0, 0.01, -0.38)`, which no
axis-aligned `CapsuleCollider` can express. Nothing in the hierarchy has a scale other
than 1, because PhysX cooks collider geometry through the transform.

MuJoCo box `size` is **half-extents**; Unity's `BoxCollider.size` is full extents. Unity's
capsule `height` includes both hemispherical caps; MuJoCo's `fromto` spans only the
cylinder.

### Self-collision is not optional

MuJoCo excludes **direct parent-child geom pairs and nothing else** — leg-vs-leg and
thigh-vs-its-own-foot contacts were live during training.

That exclusion has to be applied explicitly here. The pelvis capsule (r = 0.085, spanning
y = −0.09…0.09) and each thigh capsule (r = 0.06, starting 0.02 m away at the hip) overlap
by more than 0.12 m at the spawn pose. PhysX would normally suppress that as an adjacent
articulation pair — but **the two links are not adjacent in Unity**: the placeholder chain
carrying `hip_z` and `hip_x` sits between them. Without the filtering the creature
detonates on the first physics tick. `MujocoBipedAgent.ApplySelfCollisionFiltering` walks
the *MuJoCo* body tree, not the Unity link chain.

### Inertia frames

`robot_spec.json` ships each body's inertia as a diagonal, which is the tensor in MuJoCo's
*inertial* frame. The builder writes `inertiaTensorRotation = identity`, so the two frames
have to coincide. Measured (`RIG_AUDIT.md` section E): the largest body-axis misalignment
is 14.31° on the thighs, but because a capsule has two equal principal moments, treating
every tensor as body-diagonal costs at most **0.676%** on any moment. Justified by
measurement, not by inspection — a future rig with a genuinely asymmetric link must export
`iquat` and set `inertiaTensorRotation` from it.

---

## 6. Deliberate differences from MuJoCo

These are the places where Unity cannot reproduce MuJoCo, listed so nobody has to
rediscover them.

### Joint limits are hard here, soft there

MuJoCo's joint limits are soft constraints it may violate and then relax. Its own
`init_qpos` puts every hinge at 0 — but the knee range is `[-150°, -2°]`, so zero is **2°
outside it**, and the reset noise pushes further (the recording starts with `knee_r` at
+0.0138 rad). Unity's `ArticulationDrive` limits are hard and the solver fights a
`jointPosition` set outside them from the first tick.

So the spawn pose is `init_qpos` **clamped into each joint's own range**. For the knees
that is 2° of bend, costing 0.24 mm of standing height.

### Friction combines differently

MuJoCo takes the **elementwise maximum** of two geoms' friction, so the foot/floor pair ran
at `max(foot 1.2, floor 1.0) = 1.2` during training.

Unity's combine modes are Average, Multiply, Minimum, Maximum, and the **higher enum value
wins** a mismatched pair. `PM_MujocoBiped.physicMaterial` ships **1.2 / 1.2 with Minimum**,
which is the conservative choice: it can never produce more grip than the scene's own
ground offers, so dropping this creature into someone else's level cannot make that level
unexpectedly sticky.

The cost is real. Against a ground with no physics material at all, Unity's default is
0.6, so the effective pair friction is **0.6 rather than MuJoCo's 1.2** — and
`SCN_RACE_FLAT` builds its ground with a null material, so that is exactly the case there.
`Maximum` at 1.2 would reproduce MuJoCo's rule against any ground. Rung 6 measures both;
`README_UNITY.md` carries the result and the one-line change.

### Everything else

| MuJoCo | Unity | Consequence |
| --- | --- | --- |
| Newton solver, 50 iterations, `implicitfast` | PhysX TGS, 12/4 per-body iterations | Different contact resolution; the reason rung 6 exists |
| Contacts are soft (`solref`/`solimp`) | Rigid contacts + `contactOffset` | Landing impulses differ |
| No joint velocity limit | `maxJointVelocity` = 200 rad/s | Never reached; recorded peak is 37.14 |
| Episode terminates on falling | `autoRecoverFromFalls` stands it back up | Unity has no episode; off in tests |
| `armature` on the mass-matrix diagonal | folded into link inertia | Exact at the zero pose only |

### The `.physicMaterial` extension

`.physicMaterial`, with **one** 's', is the only extension Unity 6000.5 imports as a
`PhysicsMaterial`. A file written to `.physicsMaterial` is byte-for-byte correct YAML and
gets a `.meta`, but the importer does not claim the extension, so it lands as a
`DefaultAsset` and every `LoadAssetAtPath<PhysicsMaterial>` returns null — silently, with
the colliders falling back to PhysX defaults.

---

## 7. Triage ladder

Work **down** the list. Each rung isolates one layer, so the first failure names its own
cause instead of leaving you to guess.

| Rung | Test | Asks | Gate |
| --- | --- | --- | --- |
| 0 | `Rung0_InferenceMatchesRecordedActions` | Is the model intact in Unity? | < 1e-4 vs recording |
| K | `RungK_ForwardKinematicsMatchesIndependentPythonFk` | Is the frame map right? | < 1 mm |
| O | `RungO_ObservationMatchesRecordedObservations` | Is the policy fed what it was trained on? | < 1e-3 per term |
| 1 | `Rung1_StandsAtTheMujocoSpawnHeightAndRestsOnTheGround` | Right geometry, does contact hold? | spawn exact, rests on the floor |
| 2 | `Rung2_ZeroGSingleJointConservesLinearMomentum` | Is PhysX inventing momentum? | \|v_CoM\| < 0.02 m/s |
| D | `RungD_ActuatorReachesTheSolverAndDampingIsPerRadian` | Does torque arrive, in what units? | kd_eff < 5 N·m·s/rad |
| 3 | `Rung3_ZeroGSquareWaveStabilityAcrossTimesteps` | How much headroom has the step? | shipped config, project step |
| 4 | `Rung4_ZeroGPolicyActuationStaysFinite` | Does the full loop run clean? | finite, \|action\| ≤ 1 |
| 5 | `Rung5_WalksTowardTheTargetUnderGravity` | Does it walk? | > 1 m closed, upright |
| 6 | `Rung6_SpeedParityAgainstMujocoBaseline` | How fast, against 1.15 m/s? | ≥ 50% parity |
| — | `Perf_EightCreaturesHoldSixtyFps` | Does the budget hold? | 8 creatures under 16.67 ms |

Three shortcuts worth internalising:

* **Rung 0 passes but the creature walks wrong** → the problem is physics, and only
  physics. No amount of re-exporting the model will help.
* **Rung 0 and K pass but rung O fails** → the rig is built right and the model is right,
  but the policy is being handed something it never trained on. Rung O reports per term,
  and the term names the bug.
* **Rungs 0, K and O all pass but rung 5 or 6 is weak** → nothing is *wrong*; you are
  looking at the PhysX-versus-MuJoCo gap. Read the rung-6 sweep table in
  `README_UNITY.md` before changing a gain, because it already says which knobs move the
  number and which do not.

`check_onnx.py` is the out-of-engine twin of rung 0 and reports 7.749e-07 under
onnxruntime. A materially larger number in rung 0 is an Inference Engine difference, not a
model one.
