# Biped walk-to-target — Unity Sentis export bundle

Trained in Isaac Lab with RSL-RL PPO (2026-08-28_12-30-02, checkpoint
`model_1300.pt`). Measured over 256 robots x 30 s:
**35.4 targets/robot/minute at 2.80 m/s
with 0 falls**.

The rig is primitives only — it is meant to have a skinned mesh bound to it.

## Contents

| path | what |
|---|---|
| `biped.onnx` | the policy: single self-contained file, opset 15, IR 8, ops {Div, Elu, Gemm, Sub} |
| `export_report.json` | every number below, machine readable |
| `isaac_reference.json` | 250-step recording of one robot, to validate your port |
| `robot/biped.urdf` | robot description (metres, kg, radians) |
| `robot/biped_usd/` | the same robot in Isaac Sim USD form |
| `checkpoint/` | raw RSL-RL checkpoint plus the exact env/agent configs used |
| `source/` | the Isaac Lab task code, including the URDF generator and this exporter |

## The ONNX policy

Input `obs` float32 `[1, 42]` -> output `actions` float32 `[1, 10]`.
Batch size is fixed at 1. Observation normalisation is **baked into the graph**, so feed raw
observations. The output is the Gaussian **mean** action — clamp it to `[-1, 1]` before use.
Verified against PyTorch on 250 real observations: max abs diff **9.54e-06**.

## Observation layout (42 floats)

Everything is in **Isaac's Z-up right-handed** frame; convert Unity state into it before filling
the vector (see the conversion section).

| idx | n | name | content |
|---|---|---|---|
| 0-9 | 10 | `joint_pos` | joint positions minus the standing pose [rad], joint_order below |
| 10-19 | 10 | `joint_vel` | joint velocities [rad/s] x 0.1 |
| 20-22 | 3 | `root_lin_vel_b` | torso linear velocity in the torso frame [m/s] |
| 23-25 | 3 | `root_ang_vel_b` | torso angular velocity in the torso frame [rad/s] x 0.25 |
| 26-28 | 3 | `projected_gravity_b` | gravity unit vector in the torso frame; (0,0,-1) upright |
| 29-30 | 2 | `target_dir_b` | unit vector to the target in the torso yaw-only frame; x fwd, y left |
| 31 | 1 | `target_dist` | horizontal distance to the target [m] / 5, clamped to 1 |
| 32-41 | 10 | `prev_actions` | previous action, clamped to [-1,1] |

## Actions to joint targets

The 10 outputs are **not** torques and **not** absolute angles. They are offsets
around the standing pose:

```
joint_target[i] = default_joint_pos[i] + 0.5 * clamp(action[i], -1, 1)   // [rad]
```

An implicit PD drive tracks that target every physics tick:

```
tau[i] = kp[i] * (joint_target[i] - q[i]) - kd[i] * qd[i]      // clamped to effort_limit[i]
```

| # | joint | default [rad] | lower | upper | kp [N·m/rad] | kd [N·m·s/rad] | effort [N·m] | vel [rad/s] |
|---|---|---|---|---|---|---|---|---|
| 0 | `L_hip_yaw` | +0.0000 | -0.6000 | +0.6000 | 100 | 4 | 100 | 12 |
| 1 | `L_hip_roll` | +0.0000 | -0.5000 | +0.5000 | 100 | 4 | 100 | 12 |
| 2 | `L_hip_pitch` | -0.2500 | -1.5000 | +1.0000 | 150 | 5 | 150 | 15 |
| 3 | `L_knee` | +0.5000 | +0.0000 | +2.2000 | 150 | 5 | 150 | 15 |
| 4 | `L_ankle` | -0.2500 | -0.8000 | +0.8000 | 80 | 3 | 80 | 15 |
| 5 | `R_hip_yaw` | +0.0000 | -0.6000 | +0.6000 | 100 | 4 | 100 | 12 |
| 6 | `R_hip_roll` | +0.0000 | -0.5000 | +0.5000 | 100 | 4 | 100 | 12 |
| 7 | `R_hip_pitch` | -0.2500 | -1.5000 | +1.0000 | 150 | 5 | 150 | 15 |
| 8 | `R_knee` | +0.5000 | +0.0000 | +2.2000 | 150 | 5 | 150 | 15 |
| 9 | `R_ankle` | -0.2500 | -0.8000 | +0.8000 | 80 | 3 | 80 | 15 |

Joint order in the observation and action vectors is exactly the table above (left leg first,
then right, proximal to distal).

## Bodies

| link | mass [kg] |
|---|---|
| `torso` | 6.000 |
| `L_hip_yaw_link` | 0.150 |
| `R_hip_yaw_link` | 0.150 |
| `L_hip_roll_link` | 0.150 |
| `R_hip_roll_link` | 0.150 |
| `L_thigh` | 2.000 |
| `R_thigh` | 2.000 |
| `L_shank` | 1.500 |
| `R_shank` | 1.500 |
| `L_foot` | 0.600 |
| `R_foot` | 0.600 |

Total 14.80 kg. Standing hip height 0.660 m — note the
torso link origin sits *on the hip line*, so the root z coordinate is the hip height.

## Physics settings to match

| setting | value |
|---|---|
| physics timestep | 0.005000 s (200 Hz) |
| policy timestep | 0.020000 s (50 Hz) |
| decimation | 4 physics ticks per action |
| gravity | [0.0, 0.0, -9.81] m/s² (Isaac Z-up) |
| ground static friction | 0.5 |
| ground dynamic friction | 0.5 |
| ground restitution | 0.0 |
| spawn root position | [0.0, 0.0, 0.68] (Isaac) |
| episode length | 20.0 s |

## Isaac (Z-up) to Unity (Y-up)

Isaac Lab is **right-handed, Z-up, X-forward**. Unity is **left-handed, Y-up, Z-forward**.

```csharp
// positions and linear velocities
Vector3 ToUnity(Vector3 isaac) => new Vector3(-isaac.y, isaac.z, isaac.x);
Vector3 ToIsaac(Vector3 unity) => new Vector3( unity.z, -unity.x, unity.y);

// quaternions are stored WXYZ in isaac_reference.json; Unity uses XYZW
Quaternion QuatToUnity(float w, float x, float y, float z) => new Quaternion(y, -z, -x, w);

// angular velocity flips sign with handedness
Vector3 AngVelToUnity(Vector3 isaac) => new Vector3(isaac.y, -isaac.z, -isaac.x);
```

`projected_gravity_b` is the world gravity direction expressed in the torso frame, normalised —
`(0, 0, -1)` while standing upright. In Unity compute it as
`ToIsaac(Quaternion.Inverse(torsoRotation) * Vector3.down)`.

`target_dir_b` is the unit vector from the robot to the target in the torso's **yaw-only** frame
(roll and pitch removed): x forward, y to the robot's left. `target_dist` is the horizontal
distance in metres divided by 5 and clamped to 1.

Spawn pose: Isaac `[0.0, 0.0, 0.68]` = Unity `(0.000, 0.680, 0.000)`.

Joint axes follow the URDF: hip_yaw about Isaac Z (Unity Y), hip_roll about Isaac X (Unity Z),
and hip_pitch / knee / ankle about Isaac Y (Unity -X). Positive pitch swings the segment
**backward**; the knee bends one way only, `[0, 2.20]` rad.

## Verifying the port

`isaac_reference.json` is the contract, and separates two failure modes:

1. **Inference path** — feed each recorded `obs` through the ONNX model in Unity and compare
   against the recorded `action`. Should match to ~1e-6. A mismatch is a Sentis or tensor-layout
   problem, not a physics problem.
2. **Physics path** — replay the recorded actions from the recorded spawn pose and watch
   `root_pos_w` / `joint_pos` drift over the 250 steps. Divergence here means gains,
   timestep, masses or friction differ.

## The task

A target is placed 2.0–4.5 m away at a random
bearing; reaching within 0.4 m scores it and immediately spawns a new one, so a
single episode chains many targets. The robot starts each episode facing a random direction, so the
policy has to turn as well as walk. An episode ends if the torso drops below
0.4 m or leans past a projected-gravity z of -0.4.

| metric | value |
|---|---|
| targets / robot / minute | 35.41 |
| reached <0.5 s after assignment (lucky spawns) | 2.7% |
| falls in 30 s x 256 robots | 0 |
| mean speed | 2.80 m/s |
