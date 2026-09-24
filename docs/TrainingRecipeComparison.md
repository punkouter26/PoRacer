# How the three trainers train today (2026-09-24)

Input for Phase 3 of [Plan-TrainingMethodComparison.md](Plan-TrainingMethodComparison.md).
Read off the code on 2026-09-24. File references are to the repo.

**A** = Unity ML-Agents (Quadruped, Hexapod, Crab). **B1** = Isaac Lab "Boy" chase task
(IsaacBox, `ISAAC/boy_tasks`). **B2** = Isaac Lab Unitree H1 (`training/h1`, imported).
**C** = MuJoCo Warp (MojucuBoy, `training/mojucuboy`).

## 1. What the policy sees

| | A | B1 | B2 | C |
|---|---|---|---|---|
| Size | 3N + 19 (Quad 43, Hexapod/Crab 55) | 75 | 69 | 75 |
| Joints | pos/45°, vel/10, contact flag | pos relative to default, vel | same as B1 | **absolute** pos, vel |
| Body | **world** up vector, **absolute world height**, local vel, ang vel | lin vel, ang vel, projected gravity | same | gravity in body frame, lin vel, ang vel |
| Previous action | no | yes | yes | yes |
| Goal | direction + distance to a goal point; **no speed** | target position, clipped 5 m; speed 1.0 is **not observed** | velocity command vx, vy, wz | heading error (cos, sin) + commanded speed (constant 1.5 by default) |
| Extras | stamina, 4 terrain probes | — | — | — |
| Obs noise / normalisation | none / on | noise / on (baked) | noise / **off** | none / running normaliser |

## 2. Actions and timing

| | A | B1 | B2 | C |
|---|---|---|---|---|
| Joint target | ±45° around joint zero, no stance offset | default + 0.5·a rad | default + 0.5·a rad | stance + 0.6·half-range·tanh(a) |
| PD gains | prefab ArticulationDrive (Quad 4500/300/900); **fatigue** weakens to 0.55× | rig JSON table | Unitree cfg | MJCF servos |
| Physics dt | 0.005 | 0.005 | 0.005 | 0.005 |
| Policy rate | **10 Hz** | 50 Hz | 50 Hz | 50 Hz |

## 3. Rewards

Scaling differs: A adds terms every physics step × 0.25; Isaac multiplies weight × value ×
0.02 s; C sums raw values per 0.02 s step. So the raw weights are not comparable.

| Term | A | B1 | B2 | C |
|---|---|---|---|---|
| Speed / progress | Δ distance to goal (potential), **no target speed** | exp kernel at **1.0 m/s**, w 1.5; progress 0.5 | velocity-command tracking, w 1.0 | exp kernel at **1.5 m/s, one-sided** (overshoot free), w 2.0 |
| Heading | — | cos to target, 0.3 | yaw-rate tracking, 1.0 | facing, 0.4 |
| Upright | small penalty | flat orientation −1.0 | −1.0 | +0.05 bonus |
| Alive | — | — | — | +0.10 / step |
| Get-up | — | — | — | +0.6 × standing |
| Effort | applied torque² −0.05 | torque² −1e-5 | **off** | action² −0.005 (torque logged, not charged) |
| Action rate | −0.01 | −0.005 | −0.005 | −0.01 |
| Foot slip | all grounded links, −0.025 | feet −0.25 | feet −0.25 | **none** |
| Feet air time | — | +1.0 | +1.0 | — |
| Posture / contacts | — | joint-deviation and body-contact penalties | joint deviation only | impact penalty |
| Terminal | goal +10, out of bounds −1, stall −0.5 | fall −200, target +5 | fall −200 | — |

## 4. Episodes

| | A | B1 | B2 | C |
|---|---|---|---|---|
| Fall ends episode | **no** | yes (hip/spine contact) | yes (torso contact) | no (optional flag: height < 0.45 or up < 0.3) |
| Length | **15 s** (comment says 60 s; `Systems_TrainingArea.cs:18` counts 0.005 s steps as 0.02 s) | 20 s | 20 s | 20 s |
| Reset | spawn pose, goal 3–20 m | ±0.5 m, any yaw, pushes every 10–15 s | similar | stance + noise, any yaw, **30 % sprawled starts**, pushes |
| Randomisation | quirks (power/mass/friction) | friction, torso mass | torso mass | gains ±30 %, mass ±10 %, friction ×0.6–1.4 |
| Timeouts | bootstrapped | bootstrapped | bootstrapped | **treated as terminal** |

## 5. Curriculum and demos

| | A | B1 | B2 | C |
|---|---|---|---|---|
| Curriculum | goal distance, angle, roughness, track kind, hazards; gated by **training time** (Quad) or reward (Crab) | none | none | sprawl ramp; hand-staged runs |
| Demos | GAIL 0.05 + BC 0.3 from scripted sine gaits | none | none | none |

## 6. PPO

| | A | B1 | B2 | C |
|---|---|---|---|---|
| LR | 3e-4 linear | 1e-3 adaptive KL | 1e-3 adaptive | 3e-4 constant |
| Epochs | 3 | 5 | 5 | 4 |
| γ | 0.995 per 0.1 s (≈20 s horizon) | 0.99 per 0.02 s (≈2 s) | same | same |
| Entropy | 0.005 | 0.01 | 0.01 | 0.002 |
| Network | 2×256 Swish | 3×128 ELU | 3×128 ELU | 3×128 ELU, σ₀ ≈ 0.22 |
| Parallel envs | 2 areas × 4 instances | 4096 | 4096 | 8192 |

## 7. Defects found on the way

- **A's episodes are 15 s, not 60 s** (verified). A 20 m goal needs 1.33 m/s; these
  creatures manage 0.26 m/s. The 20 s stall rule can never fire.
- **`Config/AcrobatLoco01.yaml` does not parse** (verified): it uses `*id001`-style
  aliases with no anchors. The same failure was recorded before for FocusedLoco01.
- A's curriculum for the Quadruped advances on elapsed training, not on skill.
- C's shipped brain was trained on code the committed env no longer reproduces
  (reward sign bug M9, fixed since).
- C's speed metric always compares against 1.5 m/s even when the command varies.
