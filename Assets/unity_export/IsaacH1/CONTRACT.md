# IsaacH1 - contract

The exact interface between the Isaac Lab policy and this Unity rig. Every number
here was read out of `checkpoint/params/env.yaml`, `robot/usd/h1_minimal.usd`,
`export_report.json` or `isaac_reference.json`, or measured in the engine. Where a
source disagrees with another, the winner and the reason are stated.

Creature `IsaacH1` · task `Isaac-Velocity-Flat-H1-Play-v0` · Unitree H1 · 20 bodies,
19 joints, **3 collision shapes** · 51.437 kg.

---

## 1. Policy I/O

| | |
|---|---|
| file | `IsaacH1.onnx` (copy of `h1_policy.onnx`, single file, 178 875 bytes) |
| input | `obs` `float32[1, 69]` |
| output | `actions` `float32[1, 19]` |
| network | MLP [128, 128, 128], `elu`; ops used: `Elu`, `Gemm` only |
| opset / IR | ai.onnx 15 / IR 8 |
| normaliser | **none** - `obs_normalization: false`, so raw observations are fed in. Verified: the graph contains no `BatchNormalization` / `LayerNormalization` / `InstanceNormalization` node, and only 8 initializers (weight+bias x 4 layers). |
| determinism | verified bit-identical across repeated runs |
| runtime | Inference Engine `com.unity.ai.inference` 2.6.1, namespace `Unity.InferenceEngine`, `BackendType.CPU` |

**Not ML-Agents.** This RSL-RL export has no `obs_0`, `continuous_actions`,
`version_number` or `memory_size` tensors, so it cannot be attached to
`BehaviorParameters`. It is driven directly by `ModelLoader.Load` → `new Worker(model,
BackendType.CPU)` → `Schedule` → `PeekOutput`.

Verified equality against the 250-step recording:

| check | max abs diff | gate |
|---|---|---|
| `check_onnx.py`, onnxruntime | 2.384e-06 | 1e-4 |
| rung 0, Inference Engine CPU, in-editor | **1.907e-06** | 1e-4 |

---

## 2. Observation vector - 69 floats

Built index-by-index in `IsaacH1Agent.BuildObservations()`, in the order Isaac's
`_get_observations()` concatenates its terms. `enable_corruption: false` in the play
task, and every term has `scale: null` and `clip: null`, so **no noise and no scaling
are applied**.

| idx | term | meaning | Unity source |
|---|---|---|---|
| 0-2 | `base_lin_vel` | root linear velocity in the base frame [m/s] | `PosToIsaac(inv(rot) * root.linearVelocity)` |
| 3-5 | `base_ang_vel` | root angular velocity in the base frame [rad/s] | `AxisToIsaac(inv(rot) * root.angularVelocity)` |
| 6-8 | `projected_gravity` | gravity **direction** in the base frame, unit | `PosToIsaac(inv(rot) * Physics.gravity.normalized)` |
| 9-11 | `velocity_commands` | `vx`, `vy`, `wz` in Isaac convention | derived from the target, see §6 |
| 12-30 | `joint_pos` | `q - q_default` [rad] (`joint_pos_rel`) | `body.jointPosition[0] - default` |
| 31-49 | `joint_vel` | `qd` [rad/s] (`joint_vel_rel`; default velocity is 0) | `body.jointVelocity[0]` |
| 50-68 | `actions` | the previous **raw** policy output, before scale/offset | cached |

Two details that are easy to get wrong:

* `ArticulationBody.jointPosition` / `jointVelocity` are in **radians** for a revolute
  joint, while `xDrive.target` and `xDrive.lowerLimit/upperLimit` are in **degrees**.
  The observation therefore needs no conversion; the action does.
* `base_lin_vel` is the **centre-of-mass** velocity, not the link-origin velocity. Not
  assumed - measured against the recording by differencing `root_pos_w` and adding
  `omega x r`: the CoM reading fits with a 0.0089 m/s mean residual versus 0.0179 m/s
  for link-origin, and `|omega x r|` averages 0.028 m/s. `ArticulationBody.linearVelocity`
  is already the CoM velocity, so it is used directly. `IsaacH1Agent.baseVelocityReference`
  exposes the other choice.

Verified at a known state (test `Observations_AreCorrectAtTheSpawnPose`): standing at
the default pose, `projected_gravity = (0.0000, 0.0000, -1.0000)`, `max |joint_pos| =
0.0000`, all velocities `0.0000`.

---

## 3. Actions → joint targets

```
joint_position_target[i] = default_joint_pos[i] + 0.5 * action[i]      # radians
```

`scale = 0.5`, `use_default_offset = true`, `clip: null` (actions are **not** clipped;
the articulation's joint limits do the clamping, exactly as in Isaac).

Applied once per policy step and held across the decimation window, as Isaac does:

```csharp
d.target = target_rad * Mathf.Rad2Deg;   // Unity drive targets are DEGREES
```

An implicit PD then tracks it every physics tick:
`tau[i] = kp[i] * (target[i] - q[i]) - kd[i] * qd[i]`, clamped to the effort limit.

---

## 4. Joint order, signs, limits and gains

Isaac's order is breadth-first from the articulation root. It is **not** the URDF's
declaration order, and Isaac's names drop the `_joint` suffix.

| # | joint | default [rad] | lower | upper | kp | kd | effort [N·m] |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `left_hip_yaw` | 0.00 | -0.43 | 0.43 | 150 | 5 | 300 |
| 1 | `right_hip_yaw` | 0.00 | -0.43 | 0.43 | 150 | 5 | 300 |
| 2 | `torso` | 0.00 | -2.35 | 2.35 | 200 | 5 | 300 |
| 3 | `left_hip_roll` | 0.00 | -0.43 | 0.43 | 150 | 5 | 300 |
| 4 | `right_hip_roll` | 0.00 | -0.43 | 0.43 | 150 | 5 | 300 |
| 5 | `left_shoulder_pitch` | 0.28 | -2.87 | 2.87 | 40 | 10 | 300 |
| 6 | `right_shoulder_pitch` | 0.28 | -2.87 | 2.87 | 40 | 10 | 300 |
| 7 | `left_hip_pitch` | -0.28 | **-1.57** | **1.57** | 200 | 5 | 300 |
| 8 | `right_hip_pitch` | -0.28 | **-1.57** | **1.57** | 200 | 5 | 300 |
| 9 | `left_shoulder_roll` | 0.00 | -0.34 | 3.11 | 40 | 10 | 300 |
| 10 | `right_shoulder_roll` | 0.00 | -3.11 | 0.34 | 40 | 10 | 300 |
| 11 | `left_knee` | 0.79 | -0.26 | 2.05 | 200 | 5 | 300 |
| 12 | `right_knee` | 0.79 | -0.26 | 2.05 | 200 | 5 | 300 |
| 13 | `left_shoulder_yaw` | 0.00 | -1.30 | 4.45 | 40 | 10 | 300 |
| 14 | `right_shoulder_yaw` | 0.00 | -4.45 | 1.30 | 40 | 10 | 300 |
| 15 | `left_ankle` | -0.52 | -0.87 | 0.52 | 20 | 4 | **100** |
| 16 | `right_ankle` | -0.52 | -0.87 | 0.52 | 20 | 4 | **100** |
| 17 | `left_elbow` | 0.52 | -1.25 | 2.61 | 40 | 10 | 300 |
| 18 | `right_elbow` | 0.52 | -1.25 | 2.61 | 40 | 10 | 300 |

Bold = the vendor URDF says something different and loses (see §8).

### Sign convention - no flips in the agent

Every revolute anchor is built with its local **+X at `-M * axis`** (§5). A positive
Unity joint angle is therefore a positive Isaac joint angle, and the agent reads
`jointPosition` and writes `xDrive.target` with no sign handling anywhere.

Proven by `Kinematics_MatchesIndependentUrdfForwardKinematics`: three joint poses
(zero, the Isaac default, and an asymmetric pose that drives every joint to a distinct
value of alternating sign) compared against a Python URDF FK built from the URDF's own
joint origins and axes - i.e. from a different source than the rig.

| pose | worst link-origin disagreement |
|---|---:|
| zero | 0.0005 mm |
| default | 0.0007 mm |
| asymmetric | 0.0007 mm |
| **gate** | **1.000 mm** |

### Drive gain units - measured

Unity's drive **target** is unambiguously in degrees while `jointPosition` is in
radians, so the gain convention had to be measured rather than assumed. Test
`Rung2b_GainUnitsCalibration` applies a known position error from rest and reads the
resulting angular acceleration:

| | |
|---|---:|
| measured `kp` | **13.83** |
| expected if gains are **radian**-based | 40.00 |
| expected if gains are **degree**-based | 2291.83 |

Radian-based, by three orders of magnitude. **Isaac's `kp`/`kd` are used raw.** The
bundled `h1/README.md` advises dividing them by `Mathf.Rad2Deg`; that advice is wrong
for Unity 6 and would make every joint 57.3x too soft. `IsaacH1Agent.gainUnits` keeps
the other option available.

(The measured 13.83 vs 40 gap is chain reaction - the elbow's parent is not infinitely
massive - which is irrelevant to a 57x discrimination.)

---

## 5. Frame map

Isaac Lab is right-handed, Z-up, X-forward. Unity is left-handed, Y-up, Z-forward. One
linear operator drives everything:

```
M : (x, y, z)_isaac  ->  (-y, z, x)_unity          det(M) = -1
```

`det(M) < 0` because the map reverses handedness, and that single fact splits the rules:

| quantity | rule | code |
|---|---|---|
| position, linear velocity, force, gravity direction (true vectors) | `M` | `IsaacH1FrameMap.Pos` |
| angular velocity, torque, rotation axes, quaternion vector part (pseudovectors) | `-M` | `IsaacH1FrameMap.Axis` |
| quaternion | vector part by `-M`, scalar unchanged | `RotFromXyzw` / `RotFromWxyz` |

```csharp
Pos(i)          => new Vector3(-i.y,  i.z,  i.x);
PosToIsaac(u)   => new Vector3( u.z, -u.x,  u.y);
Axis(i)         => new Vector3( i.y, -i.z, -i.x);
AxisToIsaac(u)  => new Vector3(-u.z,  u.x, -u.y);
RotFromXyzw(x,y,z,w) => new Quaternion(y, -z, -x, w);
```

Isaac forward `+X` is Unity forward `+Z`. Isaac up `+Z` is Unity up `+Y`. A positive
Isaac yaw rate `wz` (counter-clockwise from above) is a **negative** rotation about
Unity `+Y`.

Diagonal inertia permutes with `M`: Isaac `(Ixx, Iyy, Izz)` → Unity `(Iyy, Izz, Ixx)`.
`principalAxes` is identity in this export, so no inertia rotation is needed.

The map is applied in exactly two places - `IsaacH1RigBuilder` (rig construction) and
`IsaacH1Agent.BuildObservations` - so one test proves it for the whole rig.

### Reference quaternion order

`isaac_reference.json` names its root orientation field `root_quat_w_wxyz`. **It is
XYZW.** Tested against `obs[6:9]` (`projected_gravity`, which is `R^T * (0,0,-1)` and
therefore pins the order over all 250 steps):

| interpretation | max abs error | mean abs error |
|---|---:|---:|
| as `wxyz` | 1.92 | 1.73 |
| as `xyzw` | **0.0083** | **3.3e-05** |

Read as `wxyz` the robot comes out upside down. The copy in this folder renames the
field `root_quat_w_xyzw` so the name states the real order; nothing else was changed.

---

## 6. Control rate

| | |
|---|---:|
| Isaac `policy_dt` | 0.02 s (50 Hz) |
| Isaac `physics_dt` | 0.005 s (200 Hz) |
| Isaac `decimation` | 4 |
| this project's `Time.fixedDeltaTime` | **0.02 s** |
| `policy_dt / fixedDeltaTime` | **1.000000 - an exact integer** |
| shipped decimation | **1** |

The ratio is exact, so the control rate is a true 50 Hz and the agent logs **no
`LogError`**. It does log one `LogWarning`, because decimation 1 is not Isaac's 4: the
PD drive gets 1 physics tick per policy step where Isaac gave it 4. That is a fidelity
gap, not a rate error, and it is compensated per-body (§7).

If the step is ever set to something that does not divide `policy_dt`, the agent logs
one `LogError` naming the exact ratio and the nearest exact divisor, then runs anyway
with rounded decimation. Verified at `fdt = 0.03`
(`Decimation_LogsAnErrorWhenTheStepDoesNotDividePolicyDt`, matched with
`LogAssert.Expect` - note that `LogAssert.ignoreFailingMessages` does **not** silence a
`LogError`).

### Velocity command

Isaac's `UniformVelocityCommand` with `heading_command: true`,
`heading_control_stiffness: 0.5`, ranges `lin_vel_x [0, 1]`, `lin_vel_y [0, 0]`,
`ang_vel_z [-1, 1]`. Reproduced as:

```
vx = commandSpeed                                          # clamped to [0, 1]
vy = 0                                                     # Isaac trained [0, 0]
wz = clamp(0.5 * heading_error, -1, 1)                     # Isaac convention, CCW +
```

`heading_error` is computed in Unity and negated: `Vector3.SignedAngle` is positive
when the target lies to the creature's right, and turning right is a negative `wz` in
a right-handed Z-up frame.

Target priority: explicit `Transform target` → an enabled `ITargetProvider` component →
`IsaacH1RingTargetSampler` (resamples a heading every 10 s, matching Isaac's
`resampling_time_range`).

---

## 7. Physics map

Everything below is applied **per body or per collider**. No project-wide setting is
written by any shipped code.

| quantity | Isaac | Unity project default | shipped | where |
|---|---|---|---|---|
| gravity | `(0,0,-9.81)` Z-up | `(0,-9.81,0)` Y-up | unchanged - same magnitude | project |
| `max_linear_velocity` | 1000 | - | **1000** | per body |
| `max_angular_velocity` | 1000 | **50** (`m_DefaultMaxAngularSpeed`) | **1000** | per body |
| `max_depenetration_velocity` | 1.0 | 10 | **1.0** | per body |
| `linear_damping` / `angular_damping` | 0 / 0 | 0 / 0.05 | **0 / 0** | per body |
| joint friction | null → 0 | 0 | **0** | per body |
| solver iterations | 4 / 4 | **12 / 4** | **adaptive, see below** | per body |
| contact offset | 0.02 (PhysX default, not overridden) | **0.01** | **0.02** | per collider |
| rest offset | 0.0 | not exposed by Unity | - | deviation |
| `enabled_self_collisions` | **false** | n/a | `Physics.IgnoreCollision` over all own collider pairs | runtime |
| joint armature | **0.1 kg·m² every joint** | not exposed by Unity | **not applied** - see §9 | deviation |
| solver type | **TGS** | **TGS** (`m_SolverType: 1`) | TGS - **matches Isaac, no deviation** | project (already set) |
| enhanced determinism | n/a | **On** (`m_EnableEnhancedDeterminism: 1`) | left On | see §9.11 |
| friction pair | 0.8 static / 0.6 dynamic | 0.6 / 0.6 | **0.8 / 0.6** | `PM_IsaacH1` |
| restitution | 0.0 | 0.0 | 0.0 | `PM_IsaacH1` |

All confirmed live at runtime by `PerBodyOverrides_AreLiveAtRuntime`, which exists
because **Unity does not serialise most of these onto a prefab**: `m_Mass`,
`m_InertiaTensor`, `m_InertiaRotation` and `m_CenterOfMass` are serialised, but
`contactOffset`, `solverIterations`, `maxJointVelocity`, `maxAngular/LinearVelocity`
and `maxDepenetrationVelocity` are runtime-only. Inspecting the prefab proves nothing
about them; they exist only because `Awake` re-applies them.

### Friction

`env.yaml`'s startup `physics_material` event sets the **robot's own shapes** to
`static_friction_range: [0.8, 0.8]`, `dynamic_friction_range: [0.6, 0.6]` - a
degenerate range, so a fixed value, not a random draw. The **ground** stays at the
`sim.physics_material` 1.0/1.0, and PhysX combined them by `multiply`:

```
pair friction = 0.8 * 1.0 = 0.8 static,  0.6 * 1.0 = 0.6 dynamic
```

`export_report.json`'s `physics.ground_material` reports only the ground's 1.0/1.0 and
does not mention the robot's 0.8/0.6, so the pair value is **not** what that field
suggests.

`PM_IsaacH1` carries 0.8/0.6 with **Minimum** combine. Against the 1.0/1.0-equivalent
ground this tool creates, Minimum gives exactly 0.8/0.6 - the Isaac pair value. Against
an arbitrary existing ground it degrades to `min(0.8, mu_ground)`, i.e. never above the
Isaac value. In Unity 6 the combine enum is `Average=0, Multiply=1, Minimum=2,
Maximum=3` and the **higher** value wins a mismatched pair, so Minimum also overrides a
scene material asking for Average or Multiply; only a `Maximum` material beats it.
Measured here: Average and Multiply combine both give 1.025 m/s - identical to Minimum -
so this choice is not load-bearing on a matched ground.

#### What the H1 actually stands on in `SCN_RACE_FLAT`

PoRacer's race ground is **not in the scene**. `Systems_TrackBuilder` builds it at
runtime when the race starts, and `RaceTrackView._physicsMaterial` is `{fileID: 0}` -
**null** - so the built ground gets Unity's implicit default material: 0.6 static /
0.6 dynamic, `Average` combine. Against `PM_IsaacH1` (0.8/0.6, `Minimum`), `Minimum`
outranks `Average`, so the pair resolves to:

```
static  = min(0.8, 0.6) = 0.6      (Isaac's pair was 0.8)
dynamic = min(0.6, 0.6) = 0.6      (Isaac's pair was 0.6)
```

Static friction is therefore **0.6 instead of Isaac's 0.8** on the real track;
dynamic friction is exact. Every number in this document and in `README_UNITY.md` was
measured on the test ladder's own plane, which *does* carry `PM_IsaacH1` and so gets
the full 0.8/0.6. Assigning `PM_IsaacH1` (or any 0.8-static material) to
`RaceTrackView._physicsMaterial` would close the gap; that is a scene edit and is left
for you to confirm.

Two more things about that ground: it only exists **after the race starts**, so an H1
sitting in the edit-mode scene has nothing under it until then; and
`IsaacH1Agent` has no `holdUntilGrounded`, so it will fall for those frames.

### Solver iterations - adaptive, and NOT load-bearing in this project

`env.yaml` says 4/4, which is exactly right **at Isaac's own 0.005 s step**. This
project runs a 4x coarser 0.02 s step, so the shipped `AutoScaleWithStep` mode raises
the count. Measured here by `Diag_ProjectStepRescueAttempts`, 20 s runs at `fdt = 0.02`
(same run at Isaac's 1/200 for reference: 1.001 m/s, upright 0.999):

| solver iterations | speed | upright | pelvis h | parity | verdict |
|---|---:|---:|---:|---:|---|
| **4 / 4 (Isaac's own)** | **0.969 m/s** | **0.992** | 0.908 m | 108 % | **walks** |
| 16 / 16 | 0.991 m/s | 0.993 | 0.918 m | 111 % | walks |
| 32 / 32 | 0.978 m/s | 0.999 | 0.933 m | 109 % | walks |
| 48 / 48 | 0.978 m/s | 0.994 | 0.929 m | 109 % | walks |
| **64 / 64 (shipped here)** | **0.969 m/s** | **0.992** | 0.908 m | 108 % | **walks** |
| 96 / 96 | 0.660 m/s | 0.990 | 0.905 m | 74 % | walks, slower |
| 128 / 128 | **NaN** | 1.000 | NaN | - | **diverges** |

> **This table is specific to PoRacer.** The same rig in the project it was first built
> in - PGS, 6/1 defaults - fell over at 4, 16 and 32 and needed >= 48. PoRacer already
> runs **TGS** (`m_SolverType: 1`) with 12/4 defaults, which conditions the articulation
> well enough that Isaac's own 4/4 walks. The count is no longer load-bearing here;
> 4 through 64 are all within noise of each other.
>
> Two consequences. First, `AutoScaleWithStep` costs nothing in correctness but is not
> buying anything either - and it is **not** worth switching to `IsaacExact` for
> performance: 8 creatures measured 17.15 ms/frame at 64/64 and 18.40 ms at 4/4, i.e.
> solver iterations are not the bottleneck (inference plus the 534k-triangle visual
> meshes are). Second, more is not safer: **128/128 diverges to NaN**, so do not raise
> the count past 96 looking for stability.

`IsaacH1Agent.solverIterationMode = AutoScaleWithStep` (the default) resolves to
Isaac's 4/4 when the step is at or finer than 0.005 s, and otherwise to
`clamp(max(48, 4 * ratio²), 4, 96)` - which is 64 at this project's step. Solver
iterations are a **per-body** property, so this makes the creature work at the
project's own `fixedDeltaTime` **without any project-wide change**.

### Collision shapes - only three

The USD Isaac simulated has `PhysicsCollisionAPI` on exactly three prims:
`torso_link`, `left_ankle_link`, `right_ankle_link`. The legs, arms and pelvis have
**no colliders at all** - they pass through each other and through the ground. The rig
reproduces that exactly. Giving the legs colliders because the URDF lists them would
create contacts Isaac never had.

| link | Isaac shape | Unity | size (Unity, m) | centre (Unity, m) |
|---|---|---|---|---|
| `torso_link` | convex hull, 140 358 verts | `BoxCollider` | 0.344 × 0.754 × 0.198 | (0, 0.377, 0.026) |
| `left_ankle_link` | convex hull, 12 996 verts | `BoxCollider` | 0.080 × 0.081 × 0.240 | (0, -0.030, 0.055) |
| `right_ankle_link` | convex hull, 12 972 verts | `BoxCollider` | 0.080 × 0.081 × 0.240 | (0, -0.030, 0.055) |

Boxes are the axis-aligned bounds of the hulls in each link's frame. Colliders sit on
the link objects themselves, which are **unscaled** (`localScale == 1` on every
`ArticulationBody`, asserted at build), so nothing is cooked at a scale.

The `Visual/` children are non-colliding render proxies built from the URDF's
collision primitives, purely so the creature is visible. They carry no `Collider` and
no mass.

---

## 8. Where the sources disagree - Isaac wins

The vendor URDF (`robot/h1.urdf`) and the USD Isaac actually simulated
(`robot/usd/h1_minimal.usd`) are not the same robot.

| | vendor URDF | Isaac (USD / env.yaml) - **used** |
|---|---|---|
| total mass | 59.338 kg | **51.437 kg** |
| `pelvis` | 5.983 kg | 5.390 kg |
| `left_knee_link` | 2.824 kg | 1.721 kg |
| `left_ankle_link` | 0.725 kg | 0.446 kg |
| inertia tensors | its own | the USD's (differ throughout) |
| `hip_pitch` limits | -3.14 / 2.53 | **±1.57** |
| effort limits | 200 hip, 300 knee, 40 ankle, 18 elbow | **300 legs+arms, 100 ankles** |
| collision shapes | 20 (every link) | **3** |
| joint armature | absent | **0.1 kg·m²** |
| extra links | `imu`, `logo`, `d435_*`, `mid360` | merged away - not in Isaac's 20 bodies |

Two more that `export_report.json` does not surface:

* **Robot friction 0.8/0.6.** See §7.
* **`torso_link` mass is randomised.** `env.yaml`'s `add_base_mass` startup event
  scales it by `logU(0.8, 1.25)`. The USD nominal is **17.789 kg**;
  `export_report.json`'s `bodies.masses_kg` reports **15.333 kg**, which is the draw
  env 0 happened to get (0.8619x) and is therefore the value the reference recording
  was made with. The prefab ships **nominal**; `IsaacH1Agent.torsoMassScale = 0.8619`
  reproduces the recording. Measured difference in locomotion: 1.046 → 1.035 m/s.
  This is also why USD total 51.437 kg minus report total 48.981 kg = 2.456 kg exactly.

`extract_rig.py` cross-checks the USD's joint limits against `export_report.json`'s and
aborts if any disagree by more than 2e-4 rad, so this reconciliation cannot rot.

---

## 9. Unity deviations

Ordered by how much they matter.

1. **Joint armature (0.1 kg·m²) is not applied.** PhysX's articulation armature adds to
   the joint-space mass-matrix diagonal (`H[i][i] += armature`). Unity exposes no such
   field. The obvious approximation - adding `armature * a*aᵀ` to each link's inertia -
   is **wrong here**, because link inertia is *spatial*: `hip_pitch`, `knee` and `ankle`
   all rotate about the same axis, so each joint accumulates the armature of every
   descendant and leg-swing inertia comes out ~3x too high. Measured:

   | armature mode | speed | upright | parity |
   |---|---:|---:|---:|
   | **`None` (shipped)** | **1.046 m/s** | **0.994** | **117%** |
   | `FoldIntoInertia` (naive) | 0.094 m/s | 0.070 | 11% |
   | `FoldDistalOnly` (exact for parallel runs) | 0.210 m/s | 0.062 | 23% |

   `FoldDistalOnly` folds the armature only into the distal end of each parallel-axis
   run, which is the exact triangular solve of the naive over-count. It is better than
   naive and still fails, because putting 0.1 kg·m² on the ankle link inflates that
   link's own 0.000214 kg·m² by 476x and wrecks foot dynamics - something joint-space
   armature does not do. All three modes ship; `None` is the default because it is the
   one that measures best. The cost is that joints are under-damped relative to Isaac,
   which only affects the `ExplicitTorquePD` diagnostic path (§10).

2. **~~PGS instead of TGS~~ - not a deviation in PoRacer.** Isaac ran `solver_type: 1`
   (TGS) and this project's `DynamicsManager.m_SolverType` is **already `1` (TGS)**, so
   the solver matches. Nothing to compensate for and nothing to confirm. (The rig was
   first built in a PGS project, where this *was* a deviation and drove the raised
   per-body iteration counts of §7; those counts are now belt-and-braces here.)

3. **Decimation 1 instead of 4.** The control rate is exactly right (§6); the drive
   just gets fewer ticks per policy step. Compensated by the adaptive solver count.

4. **No `restOffset`.** Unity exposes `contactOffset` only. Isaac used PhysX's default
   `restOffset = 0`, which is also Unity's behaviour, so this is a non-issue in
   practice but is not settable.

5. **No `friction_offset_threshold` / `friction_correlation_distance`.** Isaac ran
   0.04 / 0.025; Unity does not expose them. `Physics.improvedPatchFriction` is `False`
   in this project and is project-wide.

6. **`bounce_threshold_velocity`.** Isaac 0.5, project `m_BounceThreshold` 2.
   Irrelevant here: restitution is 0.

7. **Physics material file extension.** Unity 6.5 renamed the class
   `PhysicMaterial` → `PhysicsMaterial` but kept the legacy **asset** extension. A file
   named `*.physicsMaterial` is given `DefaultImporter` and never loads (verified in
   this editor: `.physicsMaterial` → `loaded=False`, `.physicMaterial` → `loaded=True`).
   The deliverable is therefore `PM_IsaacH1.**physicMaterial**`.

8. **PoRacer's ground is PROCEDURAL, and was not created by this tool.**
   `SCN_RACE_FLAT` holds Main Camera, Directional Light, RaceTrack (`TrackRoot`,
   `FinishLine`, `FinishBar`, `Lane_0`), CameraRig, RaceHud, Input, GameLifetimeScope,
   Menu, WinFx, AudioDirector, DebugOverlay - and **no ground collider at all** in edit
   mode. `Systems_TrackBuilder.Build(...)` raises the track when the race starts, sizing
   it from the rolled map, and moves `FinishLine.z` to `map.LengthMeters - 2`. The spawn
   step therefore ran with `createEssentials: false`: no ground, light or camera was
   added, and `IsaacH1_Ground` does **not** exist in this project. See §7's
   "What the H1 actually stands on" for the friction consequence.

9. **No creature layer.** `TagManager.asset` defines no free named layer (6-31 are all
   empty), so the creature stays on `Default` and the agent logs one line saying so.
   Adding a layer is a project-settings change and is left for confirmation. It matters
   more here than in a single-creature project: `Physics.IgnoreCollision` is applied
   *within* one creature, so two H1s racing side by side in the same lane block **do**
   collide with each other.

10. **Other agents' settings - read, not touched.** Unlike the project this rig was
    first built in, PoRacer *does* contain ML-Agents (`com.unity.ml-agents` 4.1.0),
    `Agent_Creature` / `ICreatureAgent` in
    `Assets/Scripts/Agents/`, 10+ trained brains under `Assets/Agents/*_v01/`, a
    `CreatureCatalog` of 8 racers. **None of it was read from or written to.** The
    relevant shared surface is `Time.fixedDeltaTime = 0.02`, which those brains were
    trained at and which this integration does not change (§6: the ratio is exactly 1).
    IsaacH1 is not registered in `CreatureCatalog` and is not an `ICreatureAgent`; it
    does not participate in a race unless you wrap it in an `Agent_*` adapter.

11. **Enhanced determinism is On, and this rig is less forgiving because of it.**
    `m_EnableEnhancedDeterminism: 1` here (it was `0` in the project this rig was first
    built in). Isaac has no equivalent knob, so there is no "correct" value to match.
    Left alone. It is recorded because the two divergence diagnostics behave visibly
    worse under this project's physics configuration than under the previous one:

    | diagnostic | previous project (PGS, 6/1, determinism off) | PoRacer (TGS, 12/4, determinism on) |
    |---|---|---|
    | rung 3 bang-bang @ `fdt 0.02` | 1.808 m/s peak, 1.286 m drift | **NaN** |
    | rung 3 bang-bang @ `fdt 0.01` | 0.776 m/s, 0.811 m | 3.688 m/s, 1.085 m |
    | rung 3 bang-bang @ `fdt 0.005` | 0.306 m/s, 0.512 m | 0.373 m/s, 0.372 m |
    | solver 128/128 @ `fdt 0.02` | not run | **NaN** |

    Both are pathological by construction - rung 3 slams all 19 joints between the
    limits at 2.5 Hz in zero gravity with no contacts, which the policy never does. The
    policy-driven paths are all finite and healthy here (rung 4 zero-g `|vCoM|` 0.348
    m/s; rung 5/5b/6/6b all walk upright). The honest reading is that **at the 0.02 s
    step this articulation has little margin against pathological actions**, so clamp or
    smooth anything you feed `actionOverride` in production. The three project settings
    above differ together, so no single one is isolated as the cause.

---

## 10. Actuator models

| | `ArticulationDrive` (shipped) | `ExplicitTorquePD` (diagnostic) |
|---|---|---|
| model | PhysX implicit spring-damper = Isaac's `ImplicitActuator` | `tau = clip(kp*(q*-q) - kd*qd, ±effort)` via `ArticulationBody.jointForce` |
| stability | unconditional | conditional - see below |
| gains | `xDrive.stiffness/damping` = Isaac's kp/kd raw, `forceLimit` = `effort_limit_sim` | same numbers, applied by hand; drive zeroed so it cannot fight |
| measured at 1/50 | **1.153 m/s, upright 0.994** | n/a |
| measured at 1/200 | **1.001 m/s, upright 0.999** | n/a |
| measured at 1/500 | - | **diverges** (zero-g bang-bang max abs vCoM 34.2 m/s, 58.7 m drift in 3 s) |
| measured at 1/1000 | - | **0.956 m/s, upright 0.991** |

The audit's single-joint bound `kd*dt/I < 2` predicts 1/480 with no armature; the
engine says 1/1000 is what actually works, which is the parent-recoil caveat being
real. The agent logs the measured figure and compares it against the live step; it
never sets `Time.fixedDeltaTime` itself.

`SetJointAccelerations` is not used anywhere. `linearVelocity` (not the obsolete
`velocity`) and `PhysicsMaterial` (not `PhysicMaterial`) are used throughout.
