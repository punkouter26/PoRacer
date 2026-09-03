# Boy - contract

The exact interface between the Isaac Lab policy and the Unity rig. Every number here was
read from the LIVE simulation by `ISAAC/scripts/export_bundle.py`, not from documentation.
Checkpoint `model_2000.pt`, task `Isaac-Chase-Flat-Boy-Play-v0`.

## 1. Policy I/O

| | |
|---|---|
| file | `Boy.onnx`, single file, 184,076 bytes |
| input | `obs` `float32[1, 75]` |
| output | `actions` `float32[1, 21]` |
| opset / IR | ai.onnx 15 / IR 8 |
| operators | Concat, Div, Elu, Gemm, Sub |
| normaliser | **baked into the graph** (yes): feed RAW observations |
| onnxruntime vs PyTorch | max abs diff 2.384e-06 over 250 real observations (gate 1e-4) |
| runtime | Unity Inference Engine `com.unity.ai.inference`, `BackendType.CPU`, driven directly (NOT ML-Agents) |

## 2. Observation vector - 75 floats

| idx | term | size | meaning |
|---|---|---|---|
| 0-2 | `base_lin_vel` | 3 | root linear velocity in the base frame [m/s] |
| 3-5 | `base_ang_vel` | 3 | root angular velocity in the base frame [rad/s] |
| 6-8 | `projected_gravity` | 3 | gravity DIRECTION in the base frame, unit length; (0,0,-1) upright |
| 9-11 | `target_pos_b` | 3 | chase target minus root position, rotated into the base frame, norm-clipped to 5.0 m |
| 12-32 | `joint_pos` | 21 | q - q_default [rad], simulator joint order |
| 33-53 | `joint_vel` | 21 | qd [rad/s] |
| 54-74 | `actions` | 21 | the previous RAW policy output |

All in Isaac's frame. `target_pos_b` is the ONLY task input: Unity computes
`PosToIsaac(inv(rootRot) * (target - rootPos))`, then scales the vector down to length
5.0 m if it is longer. In training the target was resampled on a ring of radius
[3.0, 10.0] m whenever the robot came within 0.5 m of it or after
[8.0, 12.0] s. No velocity command exists; the policy's pace is whatever the reward
taught it (target speed 1.0 m/s).

## 3. Actions -> joint targets

```
joint_position_target[i] = default_joint_pos[i] + 0.5 * action[i]   # rad
tau[i] = kp[i] * (joint_position_target[i] - q[i]) - kd[i] * qd[i]                          # clamped to effort[i]
```

Physics 200 Hz, decimation 4, policy 50 Hz. Unity:
`Time.fixedDeltaTime = 0.005`, inference every 4 fixed steps. Unity's `ArticulationDrive`
target and limits are in DEGREES, `jointPosition`/`jointVelocity` in radians; gains are radian-based
(measured by the rung-2b test).

| # | joint | axis (child frame) | default [rad] | lower | upper | kp | kd | effort |
|---|---|---|---|---|---|---|---|---|
| 0 | `spine_pitch` | [0, 1, 0] | +0.000 | -0.500 | +0.500 | 400 | 15.0 | 200 |
| 1 | `hip_yaw_L` | [0, 0, 1] | +0.000 | -0.600 | +0.600 | 80 | 3.0 | 150 |
| 2 | `hip_yaw_R` | [0, 0, 1] | +0.000 | -0.600 | +0.600 | 80 | 3.0 | 150 |
| 3 | `shoulder_roll_L` | [1, 0, 0] | -1.350 | -1.750 | +0.600 | 30 | 3.0 | 40 |
| 4 | `shoulder_roll_R` | [1, 0, 0] | +1.350 | -0.600 | +1.750 | 30 | 3.0 | 40 |
| 5 | `hip_roll_L` | [1, 0, 0] | +0.000 | -0.600 | +0.600 | 100 | 4.0 | 150 |
| 6 | `hip_roll_R` | [1, 0, 0] | +0.000 | -0.600 | +0.600 | 100 | 4.0 | 150 |
| 7 | `shoulder_pitch_L` | [0, 0, 1] | +0.000 | -2.500 | +1.000 | 30 | 3.0 | 40 |
| 8 | `shoulder_pitch_R` | [0, 0, 1] | +0.000 | -1.000 | +2.500 | 30 | 3.0 | 40 |
| 9 | `hip_pitch_L` | [0, 1, 0] | -0.200 | -2.000 | +0.800 | 150 | 5.0 | 150 |
| 10 | `hip_pitch_R` | [0, 1, 0] | -0.200 | -2.000 | +0.800 | 150 | 5.0 | 150 |
| 11 | `shoulder_yaw_L` | [0, 1, 0] | +0.000 | -1.000 | +1.000 | 30 | 3.0 | 40 |
| 12 | `shoulder_yaw_R` | [0, 1, 0] | +0.000 | -1.000 | +1.000 | 30 | 3.0 | 40 |
| 13 | `knee_L` | [0, 1, 0] | +0.400 | +0.000 | +2.400 | 150 | 5.0 | 150 |
| 14 | `knee_R` | [0, 1, 0] | +0.400 | +0.000 | +2.400 | 150 | 5.0 | 150 |
| 15 | `elbow_L` | [0, 0, 1] | -0.400 | -2.300 | +0.000 | 20 | 2.0 | 40 |
| 16 | `elbow_R` | [0, 0, 1] | +0.400 | +0.000 | +2.300 | 20 | 2.0 | 40 |
| 17 | `ankle_pitch_L` | [0, 1, 0] | -0.200 | -0.900 | +0.900 | 30 | 3.0 | 50 |
| 18 | `ankle_pitch_R` | [0, 1, 0] | -0.200 | -0.900 | +0.900 | 30 | 3.0 | 50 |
| 19 | `ankle_roll_L` | [1, 0, 0] | +0.000 | -0.500 | +0.500 | 20 | 2.0 | 50 |
| 20 | `ankle_roll_R` | [1, 0, 0] | +0.000 | -0.500 | +0.500 | 20 | 2.0 | 50 |

## 4. Frames

Isaac: right-handed, Z-up, X-forward, Y-left. Unity: left-handed, Y-up, Z-forward.

```
M : (x, y, z)_isaac -> (-y, z, x)_unity           true vectors (position, velocity, gravity)
-M: (x, y, z)_isaac -> ( y,-z,-x)_unity           pseudovectors (angular velocity, rotation axes)
quaternion (x, y, z, w)_isaac -> (y, -z, -x, w)_unity
```

Every revolute anchor's local +X is built at `-M * axis`, so a positive Unity joint angle IS a
positive Isaac joint angle and no sign is flipped anywhere.

## 5. Zero pose vs default pose

The articulation's zero pose is the authored T-POSE (all link frames world-aligned). The
default pose is the joint-angle offset in the table above (arms hang, knees bent). Unity
attaches the skinned-mesh bones in the T-pose and lets the drives take the rig to the default
pose, so the mesh follows for free.

## 6. Reference recording (`isaac_reference.json`)

Env 0, target fixed 8.0 m straight ahead, 250 policy steps. Quaternions stored XYZW
(verified against projected_gravity: residual 0.0000 vs 494.7230 for the other order).
Mean forward speed after the first second: **0.962 m/s**, 4.76 m travelled.

## 7. Isaac evaluation (64 robots, 30 s, random targets)

| metric | value |
|---|---|
| mean speed | 0.992 m/s |
| mean speed toward target | 0.949 m/s |
| mean \|v_along - 1.0\| | 0.071 m/s |
| falls | 0 (0.000 per robot per minute) |
| targets reached | 253 (7.91 per robot per minute) |
