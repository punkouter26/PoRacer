# IsaacH1 in PoRacer

A Unitree H1 driven by its Isaac Lab RSL-RL locomotion policy, running on Unity
`ArticulationBody` physics through Inference Engine, with the **original Isaac USD
visual meshes**. Everything lives in `Assets/unity_export/IsaacH1/`.

**It works.** At PoRacer's own `Time.fixedDeltaTime` of 0.02 s it walks at
**0.914 - 0.969 m/s** upright (**102 - 108 %** of the Isaac reference), and at Isaac's
own 0.005 s step **0.979 - 1.025 m/s** (**109 - 114 %**). **No project setting was
changed to get there** - and none needs to be. 20/20 PlayMode tests pass.

The policy step and the project step happen to line up exactly here:
`policy_dt / fixedDeltaTime = 0.02 / 0.02 = 1`, an **exact integer**, so the control
rate is a true 50 Hz with decimation 1 and no rounding error. There is no fixed-step
change to confirm.

---

## Quick start

1. `IsaacH1 ▸ Build Prefab` - writes `IsaacH1.prefab` and `PM_IsaacH1.physicMaterial`.
   (Already built; only needed after editing the rig.)
2. `IsaacH1 ▸ Import Original Meshes` - rebuilds `Meshes/IsaacH1Meshes.asset` from the
   `.ih1mesh` blobs. (Already built.)
3. `IsaacH1 ▸ Spawn Into Open Scene` - position, height, target = current selection.
   Spawns into the scene that is already open, leaves it **dirty and unsaved**.
4. Press Play. Give it a `Transform target`, or leave it empty and the
   `IsaacH1RingTargetSampler` will steer it.

Menu items: `Rebuild Rig Asset From JSON`, `Build Prefab`, `Import Original Meshes`,
`Spawn Into Open Scene`, `Run Reference Check` (edit mode).

**Already spawned:** one `IsaacH1` sits in `SCN_RACE_FLAT` at **(0, 1.05, -2)** - 2 m
behind the `Lane_0` start grid at the export's own `init_pos` height of 1.05 m - with
`target` wired to the scene's **`FinishLine`** at (0, 0.5, 20), 22.01 m away. The scene
was left **dirty and was not saved**.

---

## Packages

| package | version | status |
|---|---|---|
| `com.unity.ai.inference` | 2.6.1 | **already present** - pulled in by `com.unity.ml-agents` 4.1.0. Reused as-is. Provides `Unity.InferenceEngine`. |
| `com.unity.test-framework` | 1.7.0 | already present, reused |
| `com.unity.pipeline` | 0.5.0-exp.1 | already present, reused |

**Nothing was added to `Packages/manifest.json`, nothing was removed, nothing was
downgraded.** `com.unity.ai.inference` sits at `depth: 1` in `packages-lock.json` as an
indirect dependency of ML-Agents; the asmdefs reference `Unity.InferenceEngine` directly
and gate on it with a `versionDefines` entry, which works for an indirect dependency.
The URDF Importer was **not** added; the rig is built directly from the export.

**ML-Agents is present in this project and was left completely alone.** The H1 is *not*
routed through it - the RSL-RL ONNX has no `obs_0` / `continuous_actions` /
`version_number` / `memory_size` and cannot be attached to `BehaviorParameters`. It runs
on Inference Engine directly. See `CONTRACT.md` §1 and §9.10.

---

## Global settings

**Nothing in this table has been applied, and nothing in it is required.** Every row is
either already correct in PoRacer or handled per-body.

| setting | current | change needed? | why it matters | what is in use instead |
|---|---|---|---|---|
| `Time.fixedDeltaTime` | **0.02** | **No** | `policy_dt / fixedDeltaTime` is exactly **1**, so the control rate is a true 50 Hz. Isaac ran 0.005 s with decimation 4; here the drive gets 1 tick per policy step instead of 4. Other PoRacer brains were trained at 0.02 and it must not move. | Nothing. Setting it to 0.005 would reproduce Isaac exactly and also divides `policy_dt` - measured 0.979-1.025 m/s - but is a project-wide change and is **not** recommended. |
| `DynamicsManager.m_SolverType` | **1 = TGS** | **No - already matches Isaac** | Isaac ran TGS (`solver_type: 1`). | Nothing needed. This is the single biggest reason the H1 behaves better here than in the project it was first built in. |
| `m_DefaultSolverIterations` | 12 / 4 | No | Isaac used 4/4. | Set **per body**. Measured here: 4, 16, 32, 48 and 64 all walk (0.969-0.991 m/s); **128 diverges to NaN**. |
| `m_DefaultContactOffset` | 0.01 | No | Isaac ran PhysX's 0.02. | Set **per collider** to 0.02. Measured neutral: 0.01 gives 1.018 m/s vs 1.025. |
| `m_DefaultMaxAngularSpeed` | 50 | No | Isaac allowed 1000 rad/s. | Set **per body** to 1000. |
| `m_EnableEnhancedDeterminism` | **1 = On** | No | Isaac has no equivalent, so there is no value to match. | Left alone - but see `CONTRACT.md` §9.11: the divergence diagnostics are visibly harsher under it. |
| A creature layer | **none free** in `TagManager.asset` | Optional | Would let you stop two H1s colliding with each other. `Physics.IgnoreCollision` is applied *within* a creature only. | Stays on `Default`; the agent logs one line saying so. |
| `RaceTrackView._physicsMaterial` | **null** | Optional | The procedural race ground gets Unity's default 0.6/0.6 Average, so the H1's pair static friction is **0.6, not Isaac's 0.8**. | Nothing. Assigning `PM_IsaacH1` (or any 0.8-static material) closes the gap. This is a scene edit, not a project setting. |
| `m_BounceThreshold` | 2 | No | Isaac used 0.5. | Irrelevant - restitution is 0. |

---

## What working looks like

Isaac numbers come from the 250-step recording at a constant `(1, 0, 0)` command and
from `export_report.json`'s own 64-robot evaluation. Unity numbers are measured by the
PlayMode suite in this folder, in this project, in the editor.

| | Isaac | Unity @ project 1/50 | Unity @ Isaac's 1/200 |
|---|---:|---:|---:|
| forward speed at command `vx = 1` | 0.895 m/s | **0.914 - 0.969 m/s** | **0.979 - 1.025 m/s** |
| speed parity | 100 % | **102 - 108 %** | **109 - 114 %** |
| pelvis height, walking | 0.954 m (range 0.913-1.050) | 0.908 - 0.959 m | 0.951 - 0.963 m |
| torso upright (`dot(up, +Y)`) | ~1.0 (max tilt 5.4°) | 0.992 - 1.000 | 0.996 - 0.998 |
| pelvis height, standing still | - | - | 0.921 m (CoM 0.895 m) |
| ONNX vs recorded actions | 2.384e-06 (onnxruntime) | **1.907e-06** (Inference Engine CPU) | same |
| falls, shipped config | 0.125 /robot/min | - | **none in 60 s**, 52.5 m travelled |
| 8 creatures, physics per 60 FPS frame | - | **16.66 - 17.15 ms** vs 16.67 ms budget, 8/8 upright | 19.70 ms, 8/8 upright |

Unity runs slightly *faster* than Isaac at the same command. The likely cause is the
missing joint armature (`CONTRACT.md` §9.1): without 0.1 kg·m² of rotor inertia the
joints accelerate a little more freely. It is a 2-14 % overshoot in the direction of
"more lively", not a failure, and `commandSpeed` scales it directly.

### The visual meshes

The creature renders with the **real Isaac geometry**, not primitive proxies: 20
`MeshRenderer`s, **1 603 512 vertices / 534 504 triangles** per creature, extracted from
`robot/usd/instanceable_meshes.usd` (the vendor URDF points at
`package://h1_description/meshes/*.STL`, and those STL files ship nowhere in the export).

That is heavy, and it is the dominant cost in the perf figures above - not the solver
(4/4 measured *slower* per frame than 64/64, because iterations are not the bottleneck).
`Meshes/IsaacH1Meshes.asset` is an **86 MB** text-serialised asset and the folder is
**131 MB** on disk, with no Git LFS configured in this repo. If 8 H1s ever race at once
on a phone, decimating these meshes is the first thing to do; the geometry is also fully
un-welded (`verts == 3 × tris`, i.e. flat-shaded), so welding alone is a large win. The
importer falls back to the URDF primitive proxies if the library is absent, so a
decimated library is a drop-in replacement.

---

## Triage ladder

Work down it. Each rung isolates one failure mode, and each has a knob.

| rung | what it proves | if it fails |
|---|---|---|
| **0 - ONNX in engine** | Inference Engine reproduces the recorded actions | A tensor-layout or backend problem, not physics. Check the input shape is `[1, 69]`, that you `Upload` a 69-float array, and that you call `CompleteAllPendingOperations()` before indexing. Compare against `python check_onnx.py`. |
| **observations** | every term of the 69-vector is right at a known state | `projected_gravity` should be `(0,0,-1)` upright. If it is `(0,0,+1)` the frame map is inverted; if `x`/`y` are swapped, `Pos` vs `Axis` has been used for the wrong quantity. |
| **kinematics** | the rig matches an independent URDF FK | A frame-map, anchor-axis or joint-sign error. Do not trust any dynamics result until this passes. Re-run `gen_kinematics_reference.py` and check `axisInChild` in `IsaacH1_rig.json`. |
| **per-body overrides** | the runtime-only physics values are actually live | Unity does **not** serialise `contactOffset`, `solverIterations`, `maxJointVelocity`, `maxAngular/LinearVelocity` or `maxDepenetrationVelocity` onto a prefab. If you changed a field on the component after `Instantiate`, call `agent.Reconfigure()`. |
| **1 - rest height** | the legs hold the body up | Collapsing = drive gains or effort limits wrong. Floating = spawn height or collider offsets wrong. Compare against the FK rest height 0.978 m. |
| **2 - zero-g single joint** | no momentum pumping | `\|vCoM\|` must stay ~0 with no external force. Growth means the articulation solver is pumping - lower the step or raise solver iterations. |
| **2b - gain units** | drive gains are radian-based | If the measured `kp` lands near `kp * 57.3`, flip `gainUnits` to `Degrees`. |
| **3 - zero-g square wave** | how bad the pumping gets vs step | Bang-bang on all 19 joints. Scales strongly with the step - and in **this** project it reaches NaN at 0.02 s. See the knob list. |
| **5 - locomotion** | it actually walks | See the knob list below. |
| **6 / 6b - speed parity + sweep** | which variable matters | The sweep changes one thing at a time. On this rig only the foot box mattered, and only mildly. |

### Knobs, in the order worth trying

1. **`commandSpeed`** - trained range `[0, 1]`. Scales speed directly. The first thing
   to reach for.
2. **`turnSlowdown`** (shipped 0.5) and **`autoRecoverFromFalls`** (shipped on) - these
   are what keep it upright against the ring sampler's near-max yaw demands. Measured
   over 60 s: shipped = no fall, 52.5 m; `turnSlowdown 0` = first fall at 43.5 s.
3. **`armatureMode`** - unlike in the project this rig was first built in, all three
   modes walk here: `None` (shipped) 1.025 m/s, `FoldIntoInertia` 1.164 m/s,
   `FoldDistalOnly` 1.141 m/s. `None` stays the default as the closest to Isaac.
4. **`solverIterationMode`** - **not load-bearing in PoRacer.** 4/4 through 64/64 all
   walk (see `CONTRACT.md` §7). Do not raise past 96: 128/128 goes NaN.
5. **`Time.fixedDeltaTime`** - 0.005 reproduces Isaac exactly. Global; **do not change
   it** - PoRacer's other brains were trained at 0.02.
6. **`torsoMassScale`** - 0.8619 reproduces the reference recording's randomised draw.
7. **`actuatorMode`** - `ExplicitTorquePD` is a diagnostic; it needs a 1/1000 s step
   (measured: 0.947 m/s at 1/1000, 0.047 m/s at 1/500). It is not a fix for anything.
8. **Foot box, ground combine mode, contact offset, inertia floor** - measured
   near-neutral. The one that moves is the foot box: `sole-only thin box` drops speed to
   0.837 m/s (93 %). Keep the shipped box.

### If speed parity were below 50 %

It is not (114 %), but the sweep to run is `Rung6b_ConfigurationSweep`, and the
retraining knobs, in order, would be: friction randomisation (Isaac's was degenerate at
0.8/0.6 - widen it), mass randomisation beyond the current torso-only `logU(0.8, 1.25)`,
gain randomisation around the shipped kp/kd, heavier hinge links so the light
`shoulder_pitch` (5.8 % of `torso_link`) conditions better, and `velocity_limit_sim`
actually enforced in the actuator model so the policy cannot rely on 20 rad/s knee
transients.

---

## CLI recipe

The Unity CLI (`unity`, v1.0.0-beta.6) drives the already-open editor. **Never** use
`unity test`, `unity run` or `unity open` while the project is open - they spawn a
second editor. With more than one editor running, always pass `--project-path`.

```bash
unity pipeline list                      # Pipeline / Server Reachable must be true

# compile
unity command --project-path <PoRacer> recompile
unity command --project-path <PoRacer> recompile_status    # {"status":"completed","failed":false}

# build the rig, meshes and prefab
unity --json command --project-path <PoRacer> menu -- --path "IsaacH1/Rebuild Rig Asset From JSON"
unity --json command --project-path <PoRacer> menu -- --path "IsaacH1/Import Original Meshes"
unity --json command --project-path <PoRacer> menu -- --path "IsaacH1/Build Prefab"
unity --json command --project-path <PoRacer> menu -- --path "IsaacH1/Run Reference Check"

# run the ladder (async, then poll) - takes ~10.5 minutes
unity --json command --project-path <PoRacer> run_tests -- --mode playmode --filter "IsaacH1" --async_tests true
unity --json command --project-path <PoRacer> test_status

# a single rung
unity --json command --project-path <PoRacer> run_tests -- --mode playmode \
  --filter "IsaacH1.Tests.IsaacH1PlayModeTests.Rung5_WalksTowardsADistantTarget" \
  --async_tests true
```

Numbers printed by the rungs land in the editor log, **not** in the command result, and
**not** in `PoRacer/Logs/Editor.log` (that file is written only by CLI-launched
editors). A hub-launched editor writes to:

```bash
grep -a "\[rung " "$LOCALAPPDATA/Unity/Editor/Editor.log" | tail -40
```

`test_status` can time out while PlayMode holds the main thread even though the run is
progressing; `Temp/pipeline_test_status.json` is authoritative and is written directly:

```bash
python -c "import json;d=json.load(open('Temp/pipeline_test_status.json'));print(d['status'],d['summary'])"
```

Three traps worth knowing:

* **An open menu in the editor freezes every pipeline call.** A dropdown left open (the
  `PoRacer` menu, say) blocks Unity's main message loop; commands time out and CPU sits
  at zero while the window still reports "responding". Send `{ESC}` to the window.
* **Before `run_tests`, make sure the open scene is not dirty.** A dirty scene raises
  the "Scene(s) Have Been Modified" modal about 10 s after launch and every subsequent
  pipeline call times out until it is answered.
* **Editing a `.cs` file during a PlayMode run aborts it.** Let the run finish.

Python side (no Unity needed):

```bash
cd Assets/unity_export/IsaacH1
python extract_rig.py                # URDF + USD + env.yaml -> IsaacH1_rig.json
python extract_meshes.py             # USD -> Meshes/*.ih1mesh
python rig_audit.py                  # -> RIG_AUDIT.md
python gen_kinematics_reference.py   # -> kinematics_reference.json
python check_onnx.py                 # onnxruntime vs the recording, gate 1e-4
```

`extract_rig.py` and `extract_meshes.py` need `usd-core`; `check_onnx.py` needs
`onnxruntime`. All of them read `../../h1`, which resolves to `Assets/h1` in this
project. Re-running `extract_rig.py` and `rig_audit.py` here reproduced
`IsaacH1_rig.json` and `RIG_AUDIT.md` **byte-identically**.

---

## Rung table

All measured in **PoRacer**, in the editor, with the test runner attached.
**20/20 tests pass** (623 s for the full suite).

| rung | gate | measured | verdict |
|---|---|---|---|
| **0** ONNX in engine | max abs diff < 1e-4 vs 250 recorded actions | **1.907e-06** | **PASS** (strict) |
| **observations** | `projected_gravity = (0,0,-1)`, `joint_pos = 0` at the default pose | `(0.0000, 0.0000, -1.0000)`, `max\|joint_pos\| 0.0000` | **PASS** (strict) |
| **kinematics** | < 1 mm vs independent Python URDF FK, 3 poses × 20 links | **0.0008 mm** | **PASS** (strict) |
| **per-body overrides** | every runtime-only value live | maxAngVel 1000, maxJointVel 1000, contactOffset 0.020, solver 4/4 @1/200, material 0.80/0.60 Minimum, 0 un-ignored self pairs | **PASS** (strict) |
| **1** rest height | pelvis within [0.80, 1.05] m after 3 s standing | **0.9207 m** (CoM 0.8950; FK rest 0.978, Isaac walking 0.954) | **PASS** (strict) |
| **2** zero-g single joint | `max \|vCoM\| < 0.02` m/s | **0.00083 m/s** | **PASS** (strict) |
| **2b** gain units | measured kp must match the shipped convention | measured **28.26**; radian 40.00, degree 2291.83 → **radian** | **PASS** (strict) |
| **decimation @ project step** | exact integer ratio, no `LogError` | ratio **1.0000**, decimation 1 | **PASS** (strict) |
| **decimation @ 0.03** | one `LogError` naming the exact ratio | error logged via `LogAssert.Expect` | **PASS** (strict) |
| **5** locomotion | upright > 0.5, closed > 1 m, target 20 m | **11.809 m closed, 0.984 m/s, upright 0.996, h 0.951 m** | **PASS** (strict) |
| **3** zero-g square wave | informative | 1/50: **NaN** · 1/100: 3.689 m/s, 1.085 m · 1/200: 0.373, 0.372 · 1/500 torque: 33.52, 2.763 | informative |
| **4** zero-g policy | informative | 251 policy steps, decimation 4, max\|action\| 3.688, max\|jointVel\| 2.108 rad/s, `\|vCoM\|` 0.348 m/s | informative |
| **5b** steps × actuators | informative | 1/50 drive **0.914 m/s** up 1.000 · 1/200 drive **0.979** up 0.998 · 1/500 torque 0.047 up 0.974 · **1/1000 torque 0.947** up 0.993 | informative |
| **6** speed parity | < 50 % triggers a sweep | **1.023 m/s vs Isaac 0.895 = 114.2 %** | informative |
| **6b** config sweep | one variable at a time | see below | informative |
| **project-step rescue** | informative | 4/4 **0.969** · 16/16 0.991 · 32/32 0.978 · 48/48 0.978 · 64/64 0.969 · 96/96 0.660 · **128/128 NaN**, all upright ≥ 0.990 over 20 s | informative |
| **turning / sustained run** | informative | shipped: **no fall in 60 s**, 52.5 m, 0 recoveries · `turnSlowdown 0`: first fall 43.5 s · fixed target: no fall, 61.1 m | informative |
| **perf** | 8 creatures at 60 FPS, CPU | 1/50 solver 64: **16.66-17.15 ms/frame** (0.97-1.00× budget), **8/8 upright** · 1/50 solver 4: 18.40 ms (0.91×), 8/8 · 1/200: 19.70 ms (0.85×), 8/8 | see note |

### Sweep (10 s each at 1/200, Isaac reference 0.895 m/s)

| configuration | speed | upright | parity |
|---|---:|---:|---:|
| **shipped** (box foot, Minimum combine, offset 0.02, no floor, drive, armature None) | **1.025 m/s** | 0.997 | **114 %** |
| foot box → sole-only thin box | 0.837 | 0.998 | 93 % |
| ground material: Average combine | 1.025 | 0.997 | 114 % |
| ground material: Multiply combine (Isaac's own) | 1.025 | 0.997 | 114 % |
| contact offset 0.01 (Unity project default) | 1.018 | 0.996 | 114 % |
| inertia floor ON (1e-4) | 1.025 | 0.997 | 114 % |
| armature: `FoldIntoInertia` | 1.164 | 0.986 | 130 % |
| armature: `FoldDistalOnly` | 1.141 | 0.991 | 127 % |
| torso mass = the recording's 15.333 kg | 1.009 | 0.994 | 113 % |

Nothing here is load-bearing except the foot box. In the PGS project this rig was first
built in, the two `armature` folding modes **collapsed** (0.094 and 0.210 m/s, upright
0.06-0.07); under PoRacer's TGS solver they walk. That is the clearest single measure of
how much better-conditioned this project's physics is for articulations.

### Performance note

The 8-creature budget is **met, but with no margin**: ~16.7-17.2 ms of physics per
60 FPS frame against a 16.67 ms budget, i.e. **0.97-1.00× budget**, with **8/8 still
upright** after 8 s. Measured inside the editor with the test runner attached and the
full 534k-triangle meshes rendering, so treat it as a floor - a player build will do
better, and decimated meshes would do much better.

Raising or lowering solver iterations does **not** help: 4/4 measured 18.40 ms/frame,
*slower* than 64/64's 17.15 ms. Iterations are not the bottleneck; inference and the
visual meshes are.

Note that `Physics.IgnoreCollision` is applied **within** a creature, so two H1s walking
toward the same target do collide with each other. Give them separate targets, or add a
creature layer with self-collision disabled (a project-settings change - see the global
settings table).

---

## Files

| file | what |
|---|---|
| `IsaacH1.onnx` | the policy, single file, normaliser baked in (there is none - `obs_normalization: false`) |
| `IsaacH1_rig.json` | the rig in the **Isaac** frame, generated from URDF + USD + env.yaml |
| `IsaacH1Rig.asset` | the same, as a Unity `ScriptableObject` |
| `IsaacH1.prefab` | the built creature: 20 `ArticulationBody`, 3 `BoxCollider`, 20 `MeshRenderer` |
| `Meshes/IsaacH1Meshes.asset` | the 20 original Isaac visual meshes as sub-assets (86 MB) |
| `Meshes/*.ih1mesh` | the extracted geometry blobs `extract_meshes.py` writes (43 MB) |
| `Meshes/M_IsaacH1.mat` | URP Lit, the material every link renders with |
| `PM_IsaacH1.physicMaterial` | 0.8 / 0.6, Minimum combine (note the extension - `CONTRACT.md` §9.7) |
| `isaac_reference.json` | the 250-step recording, quaternion field renamed to its real order |
| `kinematics_reference.json` | independent URDF FK for 3 poses, the kinematics gate's ground truth |
| `RIG_AUDIT.md` | light links, inertia floors, joint velocities, explicit-PD bounds, quaternion order |
| `CONTRACT.md` | obs/action tables, joint order and signs, frame map, physics map, deviations |
| `extract_rig.py` · `extract_meshes.py` · `rig_audit.py` · `gen_kinematics_reference.py` · `check_onnx.py` | the generators, all re-runnable |
| `Runtime/` · `Editor/` · `Tests/` | 10 + 3 + 1 C# files and three `.asmdef`s |
