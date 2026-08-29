# Spider walk-to-target — Unity export bundle

Trained in Isaac Lab (RSL-RL PPO, 1500 iterations, 2026-08-27). Eval: ~49 targets/spider/minute, mean speed ~2.9 m/s.

## Contents
- `spider.onnx`            policy, single file, ONNX opset 15, IR 8 (Unity Sentis-compatible). Input `obs` float32 [1,59] -> output `actions` float32 [1,16] (mean action; clamp to [-1,1]).
                           Observation normalisation is baked into the graph. Verified vs PyTorch: max |diff| 6e-6.
- `export_report.json`     joint order, joint limits (rad), body names/masses, PD gains, dt, gravity, ground friction, spawn height, eval numbers.
- `isaac_reference.json`   200-step recording of one spider (obs, action, root pose, joint positions, target) for validating your Unity port step-by-step.
- `robot/spider.urdf`      robot description (metres, kg, radians). Import with a URDF importer, or rebuild from the numbers in `source/make_urdf.py`.
- `robot/spider_usd/`      same robot as USD (Isaac Sim format), for reference.
- `checkpoint/model_1499.pt` + `params/`  raw RSL-RL checkpoint and the exact env/agent configs used.
- `source/`                the Isaac Lab task code (env, config, URDF generator, exporter).

## Running the policy in Unity (e.g. Barracuda / Sentis)
Policy runs at 30 Hz (`policy_dt` = 1/30 s); physics at 120 Hz (4 substeps per action).
Each step build the 59-float observation in this order (all in the body frame unless noted):

| idx    | n  | content                                                                              |
|--------|----|--------------------------------------------------------------------------------------|
| 0-15   | 16 | joint positions [rad], order = `joint_order` in export_report.json (L1_hip, L1_knee, L2_hip, ... R4_knee) |
| 16-31  | 16 | joint velocities [rad/s] x 0.1                                                       |
| 32-34  | 3  | body linear velocity [m/s] in body frame                                             |
| 35-37  | 3  | body angular velocity [rad/s] in body frame x 0.2                                    |
| 38-40  | 3  | gravity direction in body frame (unit vector; (0,0,-1) when upright)                 |
| 41-42  | 2  | target position relative to body, in the body's yaw-only frame, x = forward, y = left [m] |
| 43-58  | 16 | previous action (the clamped [-1,1] values sent last step)                           |

Action -> joint target: `q_target = 0.8 * clamp(action, -1, 1)` [rad], applied with a PD drive
(stiffness 25 N·m/rad, damping 1 N·m·s/rad, torque limit 15 N·m, velocity limit 12 rad/s).

Coordinate conventions: Isaac is Z-up, right-handed, metres; quaternions in the JSON are (w, x, y, z).
Unity is Y-up, left-handed: map Isaac (x, y, z) -> Unity (x, z, y) and flip rotation handedness.
Hip joints rotate about the body Z (Unity Y) axis; knees about each leg's local Y (Unity: local Z after the axis swap).
Target reached when horizontal distance < 0.3 m (then a new target is sampled 1.5-3.5 m away).
