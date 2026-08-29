# Isaac Lab spider in PoRacer — integration notes

Everything lives in `Assets/unity_export/spider/`. Nothing outside it was modified: no manifest,
ProjectSettings, layer, build-settings or scene change was applied (see "Global settings" and the
final report for the changes that *are* recommended and need your confirmation).

## Runtime: Inference Engine, not ML-Agents

The policy runs through `com.unity.ai.inference` **2.6.1**, which the project already had (indirect
dependency of `com.unity.ml-agents` 4.1.0, and used directly by `CreatureCatalog`/`Editor_AutoBake`).
It is reused, not added, and not downgraded. ML-Agents is left alone: the RSL-RL ONNX has none of the
tensors `BehaviorParameters` needs (`obs_0`, `continuous_actions`, `version_number`, `memory_size`) and
cannot be attached to a behaviour. No training in Unity is planned, so deliverable 8 (ML-Agents wrapper)
is not shipped; if that changes, the wrapper is an `Agent` subclass forwarding `BuildObservation()` /
`ComputeAction()` to `VectorSensor`/`ActionBuffers`, and the raw ONNX must be re-wrapped to the
ML-Agents schema (or re-exported after a fine-tune in Unity).

| package | status |
|---|---|
| `com.unity.ai.inference` 2.6.1 | already present — reused (`Unity.InferenceEngine` assembly) |
| `com.unity.ml-agents` 4.1.0 | untouched, not used by the spider |
| URDF Importer (`https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer`) | **not added.** The editor script uses it when present (`versionDefines` → `URDF_IMPORTER`, assemblies `Unity.Robotics.UrdfImporter` / `Unity.Robotics.UrdfImporter.Editor`), and otherwise builds the rig with the built-in URDF parser (`IsaacSpiderRigBuilder`, primitives only, no scaled parents). The shipped prefab was built with the built-in path. Add the importer only if you want its path; it is not needed. |

Assemblies: `IsaacSpider.Runtime` (refs `Unity.InferenceEngine`), `IsaacSpider.Editor`, `IsaacSpider.Tests`
(PlayMode, `UNITY_INCLUDE_TESTS`). The project compiles with or without the URDF Importer.

## Global project settings the spider needs (NOT applied — your call)

| setting | current | required | what else it affects | fallback if it cannot change |
|---|---|---|---|---|
| Fixed Timestep (`TimeManager.asset`) | **0.02 s** | **1/60 s (0.0166667)** for `ArticulationDrive` mode — the coarsest step that divides the 1/30 s policy step exactly; **1/480 s** for `TorqueCSharp` mode | every racer (`Agent_Creature`) was trained at 0.02 — CLAUDE.md warns that changing it silently changes their fitted dynamics; 1/60 is 1.2× physics cost, 1/480 is 9.6× (16 steps per policy step) | the agent runs at 0.02 with decimation 2 (40 ms policy period, error logged); rung 5 walked 0.91 m/s upright in one run and fell in another — usable for a demo, not for parity |
| Solver type | TGS (`m_SolverType: 1`) | TGS | — | matches Isaac already |
| Solver iterations | 12 / 4 project default | 8 / 0 | — | applied **per body** on the spider only; nothing global |
| Gravity | (0, −9.81, 0) | same | — | — |
| Default contact offset | 0.01 | 0.02 (Isaac/PhysX 5 default) | — | applied per collider on the spider only |
| Layers (`TagManager.asset`) | no `IsaacSpider` layer | optional layer `IsaacSpider` | collision matrix; lets you stop spiders colliding with each other (Isaac `filter_collisions`) or with racers | agent logs an info line and stays on Default |
| Ground physics material | race ground has none (0.6/0.6) | 0.5/0.5 | — | `PM_IsaacSpider` (Minimum combine) on the spider colliders makes the pair 0.5 |

## Menu items

* **PoRacer ▸ Isaac Spider ▸ Build Prefab** — parses `robot/spider.urdf` (or runs `UrdfRobotExtensions.Create`
  when the importer exists, then strips `Controller`/`JointControl`/`FKRobot`/`UrdfRobot` and replaces the scaled
  convex-mesh colliders with unscaled capsules), applies the env.yaml values + floors, assigns `spider.onnx`
  and `isaac_reference.json`, saves **`IsaacSpider.prefab`**. Also creates `PM_IsaacSpider.physicsMaterial`
  if missing. Idempotent; overwrites only its own prefab.
* **PoRacer ▸ Isaac Spider ▸ Spawn Into Open Scene (target = selection)** — window: ground position, spawn
  height (0.18 m), target (defaults to the selection). Instantiates the prefab into the *active* scene and
  wires `_target`. Creates a ground / light / camera only if the scene has **none** of each. Never calls
  `NewScene`, never saves the scene for you.
* **PoRacer ▸ Isaac Spider ▸ Run Isaac Reference Check** — edit-mode: 200 recorded obs through the Worker.

Target hook: `IsaacSpiderAgent.SetTarget(Transform)` or `agent.TargetProvider = myProvider`
(`ITargetProvider.TryGetTargetPosition`). In `SCN_RACE_FLAT` the natural target is **`FinishLine`** at
(0, 0.5, 20) — select it, open the spawn window, spawn at e.g. (0, 0, −2). With no target the Isaac ring
sampler (1.5–3.5 m) runs and resamples on reach. Spawning into `SCN_RACE_FLAT` modifies that scene — do it
yourself or confirm it; the race track ground is built at runtime by `Systems_TrackBuilder` (flat map:
plane + slab, top y = 0), so in the editor the scene has no ground until Play; the spawner will therefore add
`IsaacSpider_Ground` unless you spawn after entering Play. Racers are spawned by `Systems_Spawn` from the
catalog; the spider is not part of the race model and does not need `GameLifetimeScope`.

## What "working" looks like

Isaac eval (256 spiders × 30 s): **48.8 targets/spider/minute, 2.9 m/s mean speed**, reference recording 2.76 m/s,
walking body height 0.13–0.20 m (mean 0.167), joint speeds up to 90 rad/s.

Measured in Unity (PlayMode tests, editor, prefab as shipped):

| rung | result |
|---|---|
| 0 ONNX in-engine vs recording | **5.96e-6** (PASS) |
| kinematics (3 poses vs Python FK) | **0.0 mm** |
| 1 rest, drives holding q = 0, gravity, dt 1/60 | **0.142 m** (Isaac 0.141) ; free joints: 0.100 m (belly on ground) |
| 2 zero-g, single knee step to 0.64 rad | drive 1/60: \|vCoM\| 0.0016 m/s ; torque 1/480: 0.0045 ; **torque 1/240: 1.44 m/s, knee −0.62 (diverged, joint cap hit)** |
| 3 zero-g, bang-bang ±0.8 all joints, 0.5 s period | drive 0.02: 0.40 m/s ; drive 1/60: 0.21 ; drive 1/120: 0.12 ; torque 1/480: 0.015 |
| 4 zero-g, full policy 3 s | \|vCoM\| 0.44 m/s, height 0.73 (no divergence) |
| 5 gravity, policy, target 10 m away, 8 s | drive 1/60: **0.50 m/s** ; drive 1/120: 0.94 ; drive 0.02: 0.91 (fell in an earlier run) ; torque 1/480: 1.06 — all upright, heights 0.13–0.18 |
| perf | 8 spiders, drive, policy, 1/60: **5.7 ms/frame** in the editor incl. test-runner overhead (≈0.5 ms/spider) |

Performance budget: the inputs left `<N>`/`<FPS>` unfilled; measured cost supports **8 spiders at 60 FPS** on
this desktop at 1/60. At 1/480 (torque mode) the physics cost is 8× that and shares the step with the racers.

**Open gap:** the Unity spider walks at roughly a third of Isaac's speed with every substep/actuator tried,
while being stable and upright. Candidates, in the order to test: capsule tip vs convex cylinder rim (foot
contact patch), friction combine on the runtime ground, the 0.02 m contact offset, PhysX 4 momentum pumping
from bang-bang drives (rung 3 numbers), and the missing gyroscopic term. Retraining knobs that make both sims
agree: domain-randomise ground friction (0.4–0.8), link mass (±20 %) and gains (±30 %); make the femurs
heavier (0.1 → 0.2 kg halves the body:femur ratio and doubles the knee inertia, loosening the PD bound);
lower kd or enforce `velocity_limit_sim` in the actuator model so it is actually a limit.

## Stability-triage ladder (the tests are `Tests/IsaacSpiderPlayModeTests.cs`)

0. **ONNX in-engine vs recording** (`Rung0`). Fail → the ModelAsset import (check `spider.onnx` importer) or the
   obs layout; nothing physical is involved.
1. **Rest with drives holding zero** (`Rung1`) → height 0.141 ± 0.03. Fail → colliders (scale/offset), masses,
   ground collider; then `inertiaFloor`.
2. **Zero-g, single-joint constant target, no ground** (`Rung2`) → \|vCoM\| ≈ 0. Fail (torque mode) → the step
   is too coarse: rig audit says kd·dt/I_knee = 7.4 at 0.02, 3.1 at 1/120, **1.54 at 1/240 (still diverges —
   parent recoil)**, 0.77 at 1/480. Physics substep first, then `inertiaFloor`/`massFloor`, then `_damping`.
3. **Zero-g, square wave on all joints** (`Rung3`). Momentum appears → coarser steps pump it (numbers above);
   substep first, then floors, then action scale.
4. **Full policy, zero-g** (`Rung4`). Divergence here with 2/3 clean → observation bug (frame/sign).
5. **Gravity in the real level** (`Rung5`). Falls → decimation error in the console? (fixed step), ground
   friction/material, contact offset, then gains.

Knob order for every rung: physics substep → link inertia/mass floors → damping → action scale.

## Driving the live editor from the CLI (`com.unity.pipeline` is installed)

```bash
unity status                                   # state "ready"
unity command recompile && unity command recompile_status
unity command menu -- --path "PoRacer/Isaac Spider/Build Prefab"
unity command run_tests -- --mode playmode --filter IsaacSpider --async_tests true
unity command test_status                      # poll until status != running
unity command eval -- --code 'return IsaacSpider.Editor.IsaacSpiderSetup.Spawn(UnityEngine.Vector3.zero, 0.18f, null).name;'
unity command console
```

* `unity test`, `unity run`, `unity open` spawn a **second editor** and fail while the project is open — use the
  `command` forms above.
* `--filter` matches test **names** (`Rung5` runs the four rung-5 tests).
* Editing a `.cs` file during a PlayMode run aborts it.
* PlayMode runs enter Play mode; if the open scene is dirty, Unity raises the modal **"Scene(s) Have Been
  Modified"** and every pipeline call times out until it is answered. Save or discard the scene *before*
  `run_tests`.
