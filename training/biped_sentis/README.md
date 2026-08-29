# Biped Walk-to-Target — Unity Sentis Export

Exported 2026-08-28 18:26 UTC from a PPO policy
trained for 14,499,768 steps in MuJoCo 3.12.0.

The policy takes a 49-dimensional egocentric observation and emits 12 joint torques
that walk a biped toward a goal marker, turning as the goal moves.

## Contents

| Path | What it is |
| --- | --- |
| `policy.onnx` | The exported policy. Opset 15, IR 8, batch 1, no dynamic axes |
| `reference_trajectory.json` | 150 steps of ground-truth I/O and state for parity testing |
| `robot_spec.json` | Full joint / body / actuator / simulation specification |
| `evaluation.json` | Performance over 10 deterministic rollouts |
| `training_config.json` | Hyperparameters and reward terms |
| `model/biped.xml` | The MJCF source (primitive geoms only — see `model/MESHES.md`) |
| `checkpoint/` | Raw SB3 PyTorch checkpoint + VecNormalize statistics |
| `src/` | Environment source, so observations can be reproduced exactly |

## The ONNX model

| Property | Value |
| --- | --- |
| Input | `obs` float32 `[1, 49]` |
| Output | `action` float32 `[1, 12]` |
| Opset / IR | 15 / 8 |
| Dynamic axes | none — fixed batch 1 |
| Operators | Clip, Constant, Gemm, Mul, Sub, Tanh |
| Size | 321 KB, 8 initializers (weights embedded) |
| `onnx.checker` | passed (full_check) |
| ONNX vs PyTorch | max abs diff **7.451e-07** over 256 real observations (tolerance 1e-04) |

All operators are Sentis-supported.

**The graph is self-contained.** Observation normalisation and the action clamp are
baked in, so you feed raw observations and read final actions. Do not apply
normalisation yourself — it is already inside the network.

```csharp
using Unity.Sentis;

var model  = ModelLoader.Load(policyAsset);
var worker = new Worker(model, BackendType.CPU);

using var input = new Tensor<float>(new TensorShape(1, 49), observation);
worker.Schedule(input);
var action = worker.PeekOutput() as Tensor<float>;   // shape (1, 12), already in [-1, 1]
```

## Observation layout (49 floats)

Assemble in exactly this order. Every quantity is in **MuJoCo** coordinates
(right-handed, Z-up) — convert Unity state into MuJoCo space *before* filling the
vector, not after.

| Index | Size | Field | Description | Units |
| --- | --- | --- | --- | --- |
| `0` | 1 | `torso_height` | qpos[2], torso height above ground | metres |
| `1-3` | 3 | `projected_gravity` | gravity unit vector rotated into the torso frame; encodes roll/pitch | unit vector |
| `4-6` | 3 | `linear_velocity` | torso linear velocity in the torso frame | m/s, clipped +/-10 |
| `7-9` | 3 | `angular_velocity` | torso angular velocity in the torso frame | rad/s, clipped +/-10 |
| `10-21` | 12 | `joint_positions` | qpos[7:], hinge angles in JOINT_ORDER | radians |
| `22-33` | 12 | `joint_velocities` | qvel[6:], hinge rates in JOINT_ORDER | rad/s, clipped +/-20 |
| `34-35` | 2 | `target_direction` | unit vector to the goal, rotated into the torso yaw frame (x, y) | unit vector |
| `36` | 1 | `target_distance` | planar distance to the goal | metres, min(d, 10) |
| `37-48` | 12 | `last_action` | action emitted on the previous control step | [-1, 1] |

Notes:

* `projected_gravity` is the world down-vector `(0, 0, -1)` rotated into the torso
  frame: `R_torsoᵀ · (0, 0, -1)`. It encodes roll and pitch while deliberately
  leaking no heading, which is what makes the policy heading-invariant.
* `linear_velocity` / `angular_velocity` are also expressed in the torso frame
  (`R_torsoᵀ · v_world`), not the world frame.
* `target_direction` is the unit vector to the goal rotated by **negative torso yaw**
  only — yaw, not full orientation. With the goal dead ahead it reads `(1, 0)`.
* `last_action` is the previous step's network output, post-clamp. Feed zeros on the
  first step after a reset.
* Clipping is part of the observation definition and is applied *before* the
  network's own normalisation.

## Action → joint mapping

The network outputs 12 values already clamped to `[-1, 1]`. These are **normalised
torques**, not positions or velocities:

```
torque_i (N·m) = action_i × gear_i
```

There is no target integration and no PD loop in the policy — it is direct torque
control at 40 Hz.

| Index | Actuator | Joint | Gear | ctrlrange | Peak torque | Joint limits |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `hip_z_l` | `hip_z_l` | 60 | [-1, 1] | 60 N·m | [-45, 45]° |
| 1 | `hip_x_l` | `hip_x_l` | 80 | [-1, 1] | 80 N·m | [-30, 15]° |
| 2 | `hip_y_l` | `hip_y_l` | 130 | [-1, 1] | 130 N·m | [-110, 45]° |
| 3 | `knee_l` | `knee_l` | 110 | [-1, 1] | 110 N·m | [-150, -2]° |
| 4 | `ankle_y_l` | `ankle_y_l` | 70 | [-1, 1] | 70 N·m | [-45, 35]° |
| 5 | `ankle_x_l` | `ankle_x_l` | 40 | [-1, 1] | 40 N·m | [-25, 25]° |
| 6 | `hip_z_r` | `hip_z_r` | 60 | [-1, 1] | 60 N·m | [-45, 45]° |
| 7 | `hip_x_r` | `hip_x_r` | 80 | [-1, 1] | 80 N·m | [-15, 30]° |
| 8 | `hip_y_r` | `hip_y_r` | 130 | [-1, 1] | 130 N·m | [-110, 45]° |
| 9 | `knee_r` | `knee_r` | 110 | [-1, 1] | 110 N·m | [-150, -2]° |
| 10 | `ankle_y_r` | `ankle_y_r` | 70 | [-1, 1] | 70 N·m | [-45, 35]° |
| 11 | `ankle_x_r` | `ankle_x_r` | 40 | [-1, 1] | 40 N·m | [-25, 25]° |

Joint order for observation indices 10–21 and 22–33 is identical to this action
order: `hip_z_l, hip_x_l, hip_y_l, knee_l, ankle_y_l, ankle_x_l, hip_z_r, hip_x_r, hip_y_r, knee_r, ankle_y_r, ankle_x_r`.

## Bodies

| Body | Parent | Mass (kg) | Offset from parent (m) |
| --- | --- | --- | --- |
| `torso` | `world` | 20.40 | (+0.000, +0.000, +0.880) |
| `thigh_l` | `torso` | 5.20 | (+0.000, +0.090, -0.020) |
| `shin_l` | `thigh_l` | 3.36 | (+0.000, +0.010, -0.400) |
| `foot_l` | `shin_l` | 1.38 | (+0.000, +0.000, -0.400) |
| `thigh_r` | `torso` | 5.20 | (+0.000, -0.090, -0.020) |
| `shin_r` | `thigh_r` | 3.36 | (+0.000, -0.010, -0.400) |
| `foot_r` | `shin_r` | 1.38 | (+0.000, +0.000, -0.400) |

Total mass 40.29 kg. Inertia tensors and centres of
mass are in `robot_spec.json`.

## Coordinate conversion

MuJoCo is **right-handed, Z-up**: X forward, Y left, Z up.
Unity is **left-handed, Y-up**: X right, Y up, Z forward.

Positions and direction vectors:

```
unity.x = -mujoco.y      // MuJoCo +Y (left)    -> Unity -X
unity.y =  mujoco.z      // MuJoCo +Z (up)      -> Unity +Y
unity.z =  mujoco.x      // MuJoCo +X (forward) -> Unity +Z
```

MuJoCo stores quaternions as `(w, x, y, z)`; Unity uses `(x, y, z, w)`.

> **Correction, verified numerically.** The requested mapping
> `(x, y, z, w) -> (-x, z, -y, w)` is **not** consistent with the position mapping
> `(x, y, z) -> (-y, z, x)` above. Using the two together yields rotations that disagree
> with the positions: the root translates correctly while the limbs point the wrong way.
>
> An exhaustive search over all 24 signed axis permutations that flip handedness found **no** position convention under which the requested form is a valid rotation map, so this is not a mismatch between two conventions -- the mapping is simply not a rotation. It is one sign away from two rules that *are* valid, listed in the table; `(-x, -z, -y, w)` is likely the intended one, but it belongs to the Y/Z-swap convention `unity = (x, z, y)`, not to the X-forward convention specified here.
>
> The correct quaternion mapping for the stated axes is **`( y, -z, -x,  w)`**, derived from the
> requirement `R_unity == P · R_mujoco · P⁻¹` and confirmed to machine precision (max
> error 0.0) over 512 random rotations. That is what `MuJoCoToUnity.Rotation` below
> implements. Every candidate tested is in the table.

| Candidate mapping | Max error | Valid with our axes? | Position convention it *is* valid for |
| --- | --- | --- | --- |
| `requested (-x,  z, -y,  w)` | 2.00e+00 | wrong here | **none - unusable with any axis convention** |
| `verified  ( y, -z, -x,  w)` | 0.00e+00 | **correct** | `unity = (-mujoco.y, +mujoco.z, +mujoco.x)` |
| `sign-fix  (-x, -z, -y,  w)` | 1.97e+00 | wrong here | `unity = (+mujoco.x, +mujoco.z, +mujoco.y)` |
| `w-negated (-x,  z, -y, -w)` | 1.98e+00 | wrong here | `unity = (-mujoco.x, +mujoco.z, -mujoco.y)` |

```csharp
public static class MuJoCoToUnity
{
    // MuJoCo (x_fwd, y_left, z_up) -> Unity (x_right, y_up, z_fwd)
    public static Vector3 Position(double x, double y, double z)
        => new Vector3((float)(-y), (float)z, (float)x);

    public static Vector3 Direction(double x, double y, double z)
        => Position(x, y, z);

    // MuJoCo quaternion (w, x, y, z) -> Unity Quaternion (x, y, z, w)
    public static Quaternion Rotation(double w, double x, double y, double z)
        => new Quaternion((float)y, (float)(-z), (float)(-x), (float)w);

    // Inverse: Unity -> MuJoCo, needed when building the observation vector
    public static (double x, double y, double z) ToMuJoCo(Vector3 v)
        => (v.z, -v.x, v.y);
}
```

Angular velocity is a pseudo-vector: convert with `Direction()`, then negate the
result, because the handedness flip reverses the sense of rotation.

## PD / motor drive setup in Unity

The MuJoCo model uses **direct-torque motors**, not position servos. Every joint also
carries passive damping and rotor inertia that you must reproduce or the gait will not
transfer.

| MuJoCo property | Value | Unity equivalent |
| --- | --- | --- |
| `timestep` | 0.005 s | `Time.fixedDeltaTime = 0.005` |
| `frame_skip` | 5 | run inference every 5 physics steps (40 Hz) |
| `gravity` | [0.0, 0.0, -9.81] | `Physics.gravity = new Vector3(0, -9.81f, 0)` |
| joint `damping` | 1.00 N·m·s/rad | `ArticulationDrive.damping = 1.00`, `stiffness = 0`, `target = 0` |
| joint `armature` | 0.020 kg·m² | add to the child body's inertia about the joint axis |
| geom `friction` | 0.9 (sliding) | `PhysicMaterial.dynamicFriction`, combine = Multiply |
| `condim` | 3 | standard Unity friction cone |

Recommended `ArticulationBody` configuration per joint:

```csharp
var drive = body.xDrive;
drive.stiffness      = 0f;                     // no position servo: policy outputs torque
drive.damping        = 1.00f;                  // matches MuJoCo joint damping
drive.forceLimit     = gear;                   // per-joint, from the table above
drive.target         = 0f;
drive.targetVelocity = 0f;
drive.lowerLimit     = jointLowerDeg;          // from the table above
drive.upperLimit     = jointUpperDeg;
body.xDrive          = drive;

// Each control tick, apply the network output as a torque:
body.AddRelativeTorque(jointAxis * (action[i] * gear[i]), ForceMode.Force);
```

Two traps worth calling out:

1. **Stiffness must be zero.** If you leave Unity's default position drive active it
   fights the policy's torques and the robot collapses. The policy was never trained
   against a servo.
2. **Armature matters.** MuJoCo's `armature = 0.020` adds
   rotor inertia that stabilises the joints. Unity has no direct equivalent; without
   compensating inertia the joints are effectively lighter than in training and the gait
   becomes jittery.

If the gait still differs, replay `reference_trajectory.json`: feed each recorded
observation to your Sentis worker and compare the action against the recorded one. That
isolates a model/plumbing problem (actions differ) from a physics problem (actions match
but the robot moves differently).

## Measured performance

Over 10 deterministic episodes:

| Metric | Value |
| --- | --- |
| Targets reached per episode | **4.00** (max 9) |
| Episode length | 516 steps (12.9 s) |
| Return | 3015 ± 2267 |
| Mean closing speed | 1.15 m/s |
| Survived the full 25 s episode | 30% |

Reaching more than one target per episode requires turning: each new goal spawns up to
±138° from the current heading.

**Known limitation:** the policy still falls before the 25 s timeout in most episodes.
Training was stopped at 14,499,768 of a scheduled 20.5M steps while episode
length was still improving.

## Task definition

Goals spawn 3–6 m away, within
±138° of the current heading. A goal counts as reached within
0.6 m, at which point a new one spawns. An episode terminates when the
torso leaves the height band [0.55, 1.1] m or tips past
66° from vertical.
