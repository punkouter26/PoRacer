# MujocoBiped — Unity integration

A MuJoCo-trained PPO biped (`Assets/biped_sentis/`) running on a Unity ArticulationBody
rig through Inference Engine. It walks toward a target and turns to chase it.

Everything lives under `Assets/unity_export/MujocoBiped/`. No project setting, layer, build
setting, scene asset or other creature was modified.

- **`CONTRACT.md`** — what the port must reproduce exactly, and where it deliberately
  cannot. Read this before changing anything.
- **`RIG_AUDIT.md`** — the measurements the rig is built on: mass conditioning, the
  armature solve, explicit-PD stability, the two conventions, and a geometry cross-check.

---

## Quick start

1. `MujocoBiped ▸ Rebuild Rig Asset From JSON` — `MujocoBiped_rig.json` → `MujocoBipedRig.asset`
2. `MujocoBiped ▸ Build Prefab` — → `MujocoBiped.prefab` + `PM_MujocoBiped.physicMaterial`
3. `MujocoBiped ▸ Spawn Into Open Scene` — places it, leaves the scene dirty, saves nothing
4. `MujocoBiped ▸ Run Reference Check` — edit-mode ONNX check against the recording

Give it a target by assigning `MujocoBipedAgent.target`, or implement `ITargetProvider`.
With neither, `MujocoBipedTargetSampler` reproduces `env.py`'s own goal sampling — 3–6 m
away, within ±138° of the current heading, respawning inside 0.6 m.

**One instance is already spawned in `SCN_RACE_FLAT`** at `(-7, 0.88, -2)` — lane 0, two
metres behind the start grid, at the export's `init_qpos` height — chasing `FinishLine`.
**The scene is dirty and has not been saved.**

## Regenerating the data files

```
python extract_rig.py               # robot_spec.json  -> MujocoBiped_rig.json
python make_reference.py            # recorded trajectory -> mujoco_reference.json (verified)
python gen_kinematics_reference.py  # MJCF -> kinematics_reference.json (independent FK)
python rig_audit.py                 # -> RIG_AUDIT.md + rig_audit.json
python check_onnx.py                # onnxruntime vs the recording, exits non-zero on failure
```

Needs `numpy`, `onnx`, `onnxruntime`. Not `mujoco` — the forward kinematics is written from
the MJCF directly so it stays an *independent* check of the rig, not a restatement of it.

---

## Results

Run: `unity cmd run_tests --mode playmode --filter MujocoBiped.Tests --async_tests true`.
**12 of 14 pass** in 179 s. Unity 6000.5.8f1, `Time.fixedDeltaTime = 0.005`, CPU backend.
(The 14th is `Diag_SingleJointTorqueTimeSeries`, a diagnostic that always passes; the two
decimation tests are counted in the table's footer rows.)

| Rung | Result | Measured |
| --- | --- | --- |
| 0 inference | **pass** | max abs action error **7.15e-07** over 150 recorded steps (gate 1e-4) |
| K kinematics | **pass** | worst body position error **0.0052 mm** over 3 poses (gate 1 mm) |
| O observations | **pass** | worst term error **1.9e-06** over 149 recorded states (gate 1e-3) |
| 1 statics | **pass** | spawns at 0.8800 m exactly; lowest point −7.16 mm; rests on the floor |
| 2 momentum | **pass** | zero-g single-joint \|v_CoM\| **0.00000 m/s** (gate 0.02) |
| D actuator | **pass** | torque reaches the solver; kd_effective **3.02** N·m·s/rad (gate < 5) |
| 3 stability | **fail** | project step diverges under 12-joint full-torque square wave |
| 4 policy sanity | **pass** | 5 s zero-g, peak 42.96 rad/s, \|action\| ≤ 1, 201 policy steps |
| 5 locomotion | **pass** | closes **2.45 m** of 10 m, upright 0.92, falls at 1.7 s |
| 6 speed parity | **fail** | **0.250 m/s** vs MuJoCo's 1.15 m/s = **22%** (gate 50%) |
| perf | **pass** | 8 creatures at **7.23 ms / 138 FPS**, 2.3× headroom |

### What passes is worth stating plainly

The **port itself is exact**. The model reproduces MuJoCo's actions to 7e-07, the frame map
places every body to 5 µm against an independently computed forward kinematics, and — the
one that matters most — all nine observation terms reproduce MuJoCo's own recorded
observations to **float precision** when the rig is placed in MuJoCo's recorded state.

That last check is what makes the rest of this report trustworthy. It means the double
rotation in `obs[7:10]`, the body-origin linear velocity in `obs[4:7]`, and the yaw-only
target direction in `obs[34:36]` are all right — none of which is visible by inspection,
and each of which fails silently.

### Rung 6: 22% of MuJoCo's speed

The creature walks, turns toward its goal and closes ground, at roughly a fifth of
MuJoCo's pace. The sweep, 15 s per configuration:

| Configuration | Speed | Parity |
| --- | ---: | ---: |
| **baseline (shipped)** | **0.250 m/s** | **22%** |
| ground friction 1.2, Maximum combine (MuJoCo's own rule) | 0.248 m/s | 22% |
| ground friction 0.6 (Unity default, no material) | 0.227 m/s | 20% |
| contact offset 0.02 | 0.248 m/s | 22% |
| armature `None` | 0.085 m/s | 7% |
| armature `Naive` (over-counts parallel runs) | 0.194 m/s | 17% |
| explicit torque damping (`jointForce`) | 0.136 m/s | 12% |
| `AddTorque` pair + implicit damping | 0.247 m/s | 21% |

Read it for what it rules out as much as for what it finds:

* **Friction and contact offset are not the problem.** 0.6 to 1.2 moves the number by
  10%. The friction-combine question in `CONTRACT.md` is real but it is not this.
* **The armature solve is load-bearing.** Dropping it costs **two thirds** of the speed
  (22% → 7%), and the naive fold — which over-counts every parallel-axis run — costs a
  quarter. The exact solve in `RIG_AUDIT.md` section A earns its complexity.
* **Both actuator APIs are equivalent.** `jointForce` and the `AddTorque` pair land within
  1% of each other, which is why the semantically correct one ships.

What remains is the PhysX-versus-MuJoCo gap itself: rigid contacts against MuJoCo's soft
`solref`/`solimp` ones, hard joint limits against MuJoCo's soft constraints, and a TGS
solver at 12/4 iterations against Newton at 50. The policy was trained for 14.5M steps
against those specific dynamics and is not robust to them changing — which is also why
MuJoCo's own evaluation only survives its full 25 s episode 30% of the time.

If you want to close the gap further, the honest options are to retrain with domain
randomisation over contact stiffness, or to accept 22% and tune the game around it. Ranking
gains against a policy that never saw them is not a promising direction — the sweep above
is the evidence for that.

### Rung 3: the project step diverges under saturating actuation

Twelve joints square-waving at full torque in free fall, 3 s per configuration:

| Configuration | peak \|v_CoM\| | peak \|qd\| | |
| --- | ---: | ---: | --- |
| project 0.005, shipped (`jointForce`) | 3.7e8 m/s | 7.7e8 rad/s | **diverged** |
| 1/120, shipped | 36.0 m/s | 563 rad/s | **diverged** |
| 1/240, shipped | 6.5 m/s | 103 rad/s | stable |
| project 0.005, `AddTorque` pair | 94.8 m/s | 849 rad/s | **diverged** |
| 1/480, explicit torque | 1.3 m/s | 72.8 rad/s | stable |
| project 0.005, explicit torque | 3.8 m/s | 68.5 rad/s | stable |

This is a real limit, not a passing note, and it is left failing rather than tuned away.

Its practical severity is low: under the actual policy the peak joint velocity is
**42.96 rad/s** (rung 4), against MuJoCo's own recorded 37.14 — nowhere near the 500 rad/s
divergence threshold, and rungs 4, 5 and the perf test all run for tens of seconds without
incident. The stress case is 12 joints slamming hard limits at full torque simultaneously,
with no gravity to bleed energy, which the policy never does.

Three mitigations, in order of preference:

1. Leave it. The shipped configuration is stable everywhere the policy actually goes.
2. `ActuatorMode.DirectTorqueExplicitDamping` is stable at the project step in this test
   (68 rad/s peak) — at the cost of half the gait speed (12% vs 22% parity).
3. `Time.fixedDeltaTime = 1/240` is stable, still divides `policy_dt` exactly (decimation
   6), and is a **project-wide change this port will not make on its own**.

---

## Performance

| | |
| --- | --- |
| 8 creatures | 7.23 ms mean, 12.33 ms worst → **138 FPS** |
| 60 FPS budget | 16.67 ms — within by 9.43 ms, **2.3× headroom** |
| Extrapolated | about **18 creatures** at 60 FPS |

Measured in the editor, which carries its own overhead — a player build is faster. Physics
runs at 200 Hz (5 ticks per 60 Hz frame) and inference at 40 Hz, so the frame cost is
dominated by PhysX articulation solving, not by the network: 600 policy evaluations across
those 120 frames cost a small fraction of the total.

---

## Settings that matter

All on `MujocoBipedAgent`. The defaults are what the ladder measured as best; each
alternative exists because a specific test needed the contrast.

| Field | Default | Why |
| --- | --- | --- |
| `actuatorMode` | `DirectTorqueImplicitDamping` | `jointForce` is MuJoCo's actuator semantics; the drive carries the passive damping implicitly |
| `gainUnits` | `Degrees` | **Measured.** The drive is a per-degree gain; unscaled it over-damps by ~57× |
| `armatureMode` | `Exact` | Worth 15 points of parity over `None`. `RIG_AUDIT.md` section A |
| `linearVelocityReference` | `BodyFrameOrigin` | What `qvel[0:3]` actually is |
| `selfCollisionMode` | `MujocoFaithful` | Excludes MuJoCo's parent-child pairs only — **not optional**, see below |
| `enforceVelocityLimit` | `false` | No MJCF joint has a velocity limit |
| `autoRecoverFromFalls` | `true` | Unity has no episode; without it one fall is permanent |

### Diagnostics

`debugLogObservations`, `actionOverride` (None/Constant/SquareWave), `zeroGravity` (per
body — never touches `Physics.gravity`), `showOnGuiReadout` for a live CoM-velocity and
uprightness overlay, and `RunReferenceCheck` for in-engine ONNX parity.

---

## Three traps worth knowing before you touch this

**Self-collision filtering is load-bearing.** The pelvis capsule and each thigh capsule
overlap by more than 0.12 m at the spawn pose. PhysX would normally suppress that as an
adjacent articulation pair — but the two links are *not* adjacent here, because the
single-DOF placeholder chain carrying `hip_z` and `hip_x` sits between them. Without the
explicit filtering the creature detonates on the first physics tick.

**A jammed joint reads exactly like a dead actuator.** With limits freed and full torque, a
joint swings its own shin into the pelvis inside 50 ms and stops. Every velocity reading
then collapses to something indistinguishable from "the torque never arrived" — which is
the wrong conclusion this ladder drew once, before `Diag_SingleJointTorqueTimeSeries` gave
a trace instead of a mean. Turn self-collision off for any actuator measurement.

**`.physicsMaterial` does not import.** Only `.physicMaterial`, with one `s`, is claimed by
Unity 6000.5's importer. A file written to the plural extension is byte-correct YAML, gets
a `.meta`, and still loads as `DefaultAsset` — so every
`LoadAssetAtPath<PhysicsMaterial>` returns null and the colliders silently fall back to
PhysX defaults.

---

## What this integration does not do

* It does not route through ML-Agents. The ONNX takes a bare `obs[1,49]` and returns a bare
  `action[1,12]`, with no `obs_0` / `continuous_actions` / `version_number` tensors, so it
  cannot bind to `BehaviorParameters` at all. The project's ML-Agents creatures are untouched.
* It does not change `Time.fixedDeltaTime`. It does not need to — the project already runs
  MuJoCo's own 0.005 s, and `0.025 / 0.005 = 5` exactly, so the agent gets MuJoCo's frame
  skip and not merely the right control rate.
* It does not add a layer, a tag, a build setting or a package. `com.unity.ai.inference`
  2.6.1 was already present as a dependency of ML-Agents 4.1.0.
* It does not save the scene.
