# Worm race (SCN_WORM_RACE): how to install, build, run and check it

Everything in this folder mirrors where it goes under the project root. Nothing here is
under `Assets/` yet, on purpose: a long exam was running in the Unity editor, and any new
file under `Assets/` triggers a recompile that would have killed it.

Two worms race down two straight lanes, 20 m long and 2 m apart:

| Lane | Racer | Colour | Brain | Physics inside Unity |
|---|---|---|---|---|
| 0 (x = -1 m) | **MuJoCo worm** | BLUE | `worm_mujoco.onnx` | MuJoCo, through the `org.mujoco` plug-in (the simulator it trained in) |
| 1 (x = +1 m) | **Isaac worm** | ORANGE | `worm_isaac.onnx` | PhysX `ArticulationBody` (Unity's own physics) |

Both have the same body (`worm_rig.json`), the same 35-number observation and the same
8 actions (`training/worm/WORM_SPEC.md`). Red and green are not used (AGENTS rule D).

---

## 1. Install (after the exam has finished)

Run from the project root in PowerShell.

```powershell
# 1. Code: copy the staged scripts into Assets/ (creates Assets/Scripts/WormRace and
#    Assets/Scripts/Editor/WormRace; touches no existing file).
robocopy training\worm\unity_staging\Assets Assets /E

# 2. The body: both worms are built from this file at race start.
New-Item -ItemType Directory -Force Assets\WormRace\Brains | Out-Null
Copy-Item training\worm\worm_rig.json Assets\WormRace\worm_rig.json

# 3. The brains (once the two training runs have exported them).
Copy-Item training\worm\export\worm_mujoco.onnx Assets\WormRace\Brains\worm_mujoco.onnx
Copy-Item training\worm\export\worm_isaac.onnx  Assets\WormRace\Brains\worm_isaac.onnx
```

Then let Unity import and compile. The console must show **no errors**. Unity creates the
`.meta` files itself; do not write any by hand.

What the copy adds:

- `Assets/Scripts/WormRace/` - runtime code, its own assembly `PoRacer.WormRace`
  (`allowUnsafeCode` is on, because the MuJoCo worm reads MuJoCo's state directly).
- `Assets/Scripts/Editor/WormRace/` - the scene builder and the CLI driver, assembly
  `PoRacer.WormRace.Editor` (editor only).

Re-copy `worm_rig.json` every time `build_worm.py` is re-run. The rig changed on
2026-09-24 (force limit 12 -> 6 N·m, joint damping 1 -> 2); the Unity worms read those
numbers from this file, so a stale copy races a different animal from the one trained.

If a brain is missing, nothing crashes: that worm lies still (straight) for the whole
race, the HUD says "NO BRAIN", and the console and the results file say exactly which
file to copy where.

## 2. Build the scene

```powershell
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build();"
```

(Also in the menu: **PoRacer > Worm Race > Build Scene**.) Save or discard any unsaved
scene first; the builder refuses to run over one.

Expected output, roughly:

```
materials: blue (MuJoCo), orange (Isaac), ground, line white/black
physics material: Assets/WormRace/PM_WormRace.physicMaterial (0.90 static/dynamic, no bounce)
panel settings: Assets/WormRace/UI/WormRacePanelSettings.asset
settings: Assets/WormRace/WormRaceSettings.asset
  rig:          Assets/WormRace/worm_rig.json
  MuJoCo brain: Assets/WormRace/Brains/worm_mujoco.onnx
  Isaac brain:  Assets/WormRace/Brains/worm_isaac.onnx
scene objects: WormRaceLifetimeScope, Directional Light, Main Camera, Track, WormRaceHud
saved Assets/Scenes/SCN_WORM_RACE.unity
```

A line saying `MISSING - copy ...` means that file is not in place yet. Copy it and run the
builder again; it is safe to re-run and keeps every asset's GUID.

What it creates (all authored once, AGENTS rule G): the ground (the only track collider,
its top at y = 0), three white lane lines, a white start line at z = 0, a chequered
black-and-white finish line at z = 20, a follow camera, a light, the HUD, and the
`WormRaceLifetimeScope`. The worms and the MuJoCo world are created at the start of each
race and removed after it.

The scene is not added to Build Settings; it is meant to be played in the editor.

## 3. Self-tests - run these before trusting any race

Each one enters play mode, runs, writes `Logs/wormrace_selftest_<mode>_<stamp>.json`,
and leaves play mode. Poll with `Status()` until it says `done`.

```powershell
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"zero\");"
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Status();"

unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"yaw\");"
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"pitch\");"
```

Each self-test spawns both worms, holds them straight for 1 s so they settle, then
replaces both brains with a fixed action and measures. The same code measures both worms.

| Test | Fixed action | Must be true for BOTH worms (the file's `passed` / `verdict`) |
|---|---|---|
| `zero` (5 s) | all zero | Lies still: nose moves < 2 cm and speed < 1 cm/s. Stays straight: every joint within 0.05 rad. Lies on the floor: segment 2's centre at 0.045 m (the radius) ± 1 cm |
| `yaw` (2 s) | j0_yaw target = +0.5 rad | WORM_SPEC sign test: j0_yaw reads more than +0.3 rad, and segment 1 is more than 2 cm to the worm's **right = Unity +x**. Expect about +0.45 to +0.5 rad and about +4.5 cm |
| `pitch` (2 s) | j0_pitch target = +0.5 rad | j0_pitch reads more than +0.3 rad, and segment 1 is more than 2 cm **above** the head's plane. Expect about +4 cm |

`"allPassed": true` in each file is the bar. If a test fails:

- **yaw or pitch fails on the Isaac worm only** (segment 1 swings left, or the angle reads
  negative): the PhysX axis map is off. Check `WormFrames.UnityFromSpecAxial` and the
  anchor rotation in `PhysxWormBuilder`. The MuJoCo worm's signs come from MuJoCo itself.
- **yaw or pitch fails on the MuJoCo worm only**: the plug-in built a different body.
  Diff the generated MJCF (section 5).
- **zero fails on the Isaac worm** by creeping or twitching: the passive joint damping
  mode. Try the other settings in `WormRaceSettings > Passive Damping` (section 6).
- **zero fails with the worm sitting 1-2 cm into the floor on MuJoCo**: soft-contact
  sinking. It is not a sign error; widen the tolerance or look at `solref`.

## 4. Race

```powershell
# Five races back to back (the default); any count works.
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Start(5);"
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Status();"
# Abandon a run and leave play mode:
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Stop();"
```

Or open `Assets/Scenes/SCN_WORM_RACE.unity` and press Play: it runs the default series of
5 by itself (`WormRaceSettings > Auto Start Series`).

Each race: both worms appear straight on the start line; **3-2-1** with both brains held
(physics runs, so both settle onto the floor the same way); **GO**; the first nose (the tip
of segment 0) across the finish line wins. If a worm has not finished after **60 s** it is
ranked by distance. Between races the results stay up for 4 s, then the worms are removed
and respawned.

The HUD (top left) shows, per worm: name, training method, physics, distance (m), speed
(m/s) and state; the centre shows the countdown; the bottom-right panel shows each race's
places, finish times (or "time limit"), distances, average speeds and the running score.

Results: `Logs/wormrace_<stamp>.json`, rewritten after every race, so a run that is cut
short still has every finished race in it. `Status()` prints its path and contents.
Shape:

```json
{
  "kind": "wormrace_series",
  "plannedRaces": 5, "completedRaces": 5,
  "trackLengthMeters": 20.0, "timeLimitSeconds": 60.0, "physicsDtSeconds": 0.005,
  "races": [
    { "raceNumber": 1, "winner": "MuJoCo worm", "winnerMethod": "MuJoCo", "endReason": "finish",
      "racers": [
        { "lane": 0, "name": "MuJoCo worm", "method": "MuJoCo", "physics": "MuJoCo (org.mujoco plug-in)",
          "brain": "worm_mujoco", "brainLoaded": true, "status": "Finished", "place": 1,
          "finishTimeSeconds": 31.42, "distanceMeters": 20.0, "averageSpeedMps": 0.637, "failReason": "" },
        { "lane": 1, "name": "Isaac worm", "method": "Isaac Lab", "physics": "PhysX ArticulationBody",
          "status": "TimedOut", "place": 2, "finishTimeSeconds": -1.0, "distanceMeters": 14.2,
          "averageSpeedMps": 0.237 } ] } ],
  "summary": [
    { "name": "MuJoCo worm", "wins": 5, "finishes": 5, "meanFinishTimeSeconds": 31.5,
      "bestFinishTimeSeconds": 30.9, "meanDistanceMeters": 20.0, "meanAverageSpeedMps": 0.635 },
    { "name": "Isaac worm", "wins": 0, "finishes": 0, "meanFinishTimeSeconds": -1.0, "...": "..." } ]
}
```

(The numbers are illustrative.) `-1` means "does not apply". `endReason` is `finish`,
`timeLimit` (nobody finished) or `failed` (both out). `status` is `Finished`, `TimedOut` or
`Failed`; a failed worm has a `failReason` (non-finite physics, a speed above 500 per
WORM_SPEC detail 12, or MuJoCo not available on this platform).

What to expect: the trainers' own reports (`training/worm/export/*_report.json`) give each
brain's mean speed in its own simulator. **The MuJoCo worm races in the simulator it
trained in, so it should roughly match its report.** The Isaac worm trained in Isaac
Sim's PhysX and races in Unity's PhysX; a gap between its report speed and its Unity speed
is a sim-to-sim gap, not a race result, and is worth writing down on its own.

## 5. Check the MuJoCo worm is worm.xml

With `WormRaceSettings > Dump Mujoco Mjcf` on (the default), every race writes the model
the plug-in actually compiled to `Application.temporaryCachePath/wormrace_mujoco_scene.xml`
(the console prints `MJCF saved to <path>`, usually under
`%LOCALAPPDATA%\Temp\<company>\<product>\`). Compare it with `training/worm/worm.xml`.
Names get a `_<n>` suffix and every attribute is written out explicitly; ignore that.
What must match:

- 5 capsule geoms, `size="0.045 0.075"`, `mass="1.335962"`, friction `0.9 0.005 0.0001`,
  `condim="3"`, `solref="0.01 1"`, `solimp="0.9 0.95 0.001"`.
- 4 link bodies at `pos="-0.1 0 0"` with `<inertial mass="0.1" diaginertia="0.0002 0.0002 0.0002">`.
- 8 hinges: pitch `axis="0 1 0"` at the link origin; yaw `axis="0 0 1"` at `pos="0.1 0 0"`;
  `range="-45 45"` (degrees - the plug-in's compiler uses MuJoCo's degree default),
  `limited="true"`, `damping="2"`, `armature="0.01"`.
- 8 `<position>` actuators in action order: `kp="30"`, `ctrlrange="-0.785398 0.785398"`,
  `forcerange="-6 6"`, `ctrllimited="true"`, `forcelimited="true"`.
- 4 `<exclude>` pairs seg0-seg1, seg1-seg2, seg2-seg3, seg3-seg4.
- `<option>`: `integrator="implicitfast"`, `solver="Newton"`, `iterations="10"`,
  `cone="pyramidal"`, `timestep="0.005"`, `gravity="0 0 -9.81"`.
- seg0's `quat` is a +90° turn about z (`0.7071 0 0 0.7071`, or its negative - the
  same rotation). That is the lane direction: the plug-in maps Unity +z to MuJoCo +y.
- Extra, and expected: the floor plane and five `mocap="true"` bodies (the Isaac worm's
  collision stand-ins, section 7).

## 6. The knobs (`Assets/WormRace/WormRaceSettings.asset`)

The body itself (masses, gains, limits, friction) is **not** here; it comes from
`worm_rig.json`, so an Inspector edit cannot make the two worms different animals.

| Setting | Default | What it does |
|---|---|---|
| Track Length / Lane Spacing / Start Line Z | 20 / 2 / 0 | Race geometry. Re-run the scene builder after changing these so the painted lines move too |
| Countdown Seconds / Time Limit Seconds / Results Hold Seconds | 3 / 60 / 4 | Race timing |
| Series Length / Auto Start Series | 5 / on | What pressing Play does. The CLI overrides both |
| Passive Damping | Joint Force Outside Limit | PhysX worm only, see below |
| Solver Iterations / Velocity Iterations | 16 / 4 | PhysX worm, per body; no project setting is touched |
| Fold Armature Into Inertia | on | PhysX has no armature; adds MuJoCo's 0.01 to each hinge's child inertia about its axis |
| Mujoco Solver Iterations | 10 | worm.xml's value |
| Cross Simulator Proxies | on | Collision stand-ins between the two simulators (section 7) |
| Previous Action Clipped | on | WORM_SPEC detail 5: the previous-action observation uses clipped actions |

**Passive Damping.** WORM_SPEC detail 16 says the joint damping is a passive
`-c·q̇` that the servo's force limit does not cap, and the servo is `kp` with no damping.
Unity's `ArticulationBody` has no viscous joint friction, so by default the Isaac worm's
view applies `-c·q̇` through `ArticulationBody.jointForce` every physics step (drive
damping 0). That matches the trained model but is explicit; at c = 2 and 5 ms it is
comfortably stable for these link inertias. If the zero test shows creeping or buzzing,
the other two options put the damping inside the drive (implicit, but capped together
with the servo by the force limit), read per radian (Unity's documentation) or per degree
(what `MujocoBiped` measured on this Unity version).

## 7. How it is put together

Architecture (`.claude/rules`): plain models, plain systems in VContainer, thin views,
MessagePipe messages, UniTask for the countdown and the race loop.

| Kind | Types |
|---|---|
| Model | `WormRaceModel`, `WormRacerModel` |
| System | `WormRaceSystem` (entry point: series, race, self-test flow), `WormSpawnSystem` (worms, MuJoCo world, stand-ins) |
| Per-racer logic | `WormPilot` (hold, decimation, observation, inference, clipping, race clock), `WormPolicy` (Inference Engine), `WormObservation` (the 35 numbers, shared by both worms) |
| View | `MujocoWormView`, `PhysxWormView` (physics adapters, no decisions), `WormRaceHudView`, `WormRaceCameraView`, two proxy followers |
| Messages | `WormCountdownMessage`, `WormRaceFinishedMessage`, `WormSeriesFinishedMessage` |
| Scope | `WormRaceLifetimeScope` |

**How the MuJoCo worm gets into Unity.** Not through the MJCF importer. `MujocoWormBuilder`
adds the plug-in's own components (`MjBody`, `MjGeom`, `MjHingeJoint`, `MjInertial`,
`MjFreeJoint`, `MjExclude`, `MjActuator`) straight from `worm_rig.json` at race start.
Why:

- The importer reads a missing hinge axis as Unity +X, and MuJoCo's own writer drops
  axes equal to its default (0, 0, 1). Any round trip (`ImportFile` does one) silently
  turns every yaw hinge into a pitch hinge - the defect `training/mojucuboy/make_unity_mjcf.py`
  had to work around for MojucuBoy.
- The importer assigns the built-in Standard shader (magenta in URP) and brings in
  worm.xml's own floor, which would then have to be found and removed.
- An editor-time import would bake a prefab that has to be kept in step with worm.xml by
  hand, and could not sit in the scene anyway: MjScene compiles once, from whatever Mj
  components exist when it starts, so the world and the worm must be created together.
- Building from `worm_rig.json` means both worms come from the same file.

Section 5's dump is the check that the result is worm.xml.

**How MuJoCo and PhysX share one scene.** They do not share anything physical. MuJoCo
steps inside `MjScene.FixedUpdate` and writes the MuJoCo worm's transforms; PhysX steps
the Isaac worm after all `FixedUpdate`s. Both run at `Time.fixedDeltaTime` = 0.005 s, one
step per `FixedUpdate`, and each worm's pilot counts its own physics steps, so the race
clock is fair whichever simulator Unity runs first. Each has its own floor at y = 0: an
infinite MuJoCo plane, and the track's box collider (friction 0.9, `Maximum` combine,
like MuJoCo's max rule).

Because neither simulator can see the other, each worm gets a copy of the other's five
capsules as a collision stand-in (AGENTS rule M): five MuJoCo mocap bodies following the
Isaac worm's segments, and five kinematic PhysX capsules following the MuJoCo worm's.
Both are one-way (each worm sees the other as an unstoppable moving object) and one 5 ms
step late. With the lanes 2 m apart a contact would be an accident; the stand-ins make it
a collision instead of the worms passing through each other.

**Observation frames.** The MuJoCo worm reads everything from MuJoCo's own data, in the
plug-in's MuJoCo frame (Unity (x, y, z) = MuJoCo (x, z, y)), including segment 2's
velocity via `mj_objectVelocity` - the same quantities the trainer computes. The Isaac
worm reads `ArticulationBody` state and converts with WORM_SPEC's Unity mapping
(positions (z, -x, y), axes (-z, x, -y)). Both then go through `WormObservation`, which only
projects onto segment 2's axes, so the two worlds' different orientation (the MuJoCo worm's
lane is its world +y, the Isaac worm's is its world +x) is invisible to both policies,
exactly like the ±45° spawn yaw they trained with.

**Timing.** Policy every 4 physics steps (50 Hz), action clipped to [-1, 1], target =
action × 45°. The MuJoCo worm writes `ctrl` inside the plug-in's control callback, between
`mj_step1` and `mj_step2`, so there is no one-step lag. The Isaac worm writes drive
targets in `FixedUpdate`, before PhysX steps.

**No rescues (AGENTS rule H).** A worm that fails is held where it lies until the race is
torn down. Nothing ever stands it up or moves it.

## 8. Known risks

1. **Compiled, but never run in Unity.** Both assemblies were compiled outside Unity with
   `dotnet build` against this project's own Unity 6000.6 engine DLLs and the package DLLs
   in `Library/ScriptAssemblies`, restricted to exactly the references in the two asmdefs:
   0 errors (the only warnings are CS0649 on `[SerializeField]` fields, the same as the
   rest of the project). So every API exists with the signature used. What is unproven is
   behaviour: the four `ArticulationBody` anchor properties written right after
   `AddComponent` at runtime with `matchAnchors = false` (MujocoBipedRigBuilder does the
   same with `matchAnchors = true`; if the joints come out misplaced, try that); explicit
   `inertiaTensor`/`centerOfMass` sticking after a collider is added (they are set after
   the collider for that reason); the sign and per-step persistence of
   `ArticulationBody.jointForce`; `Object.Instantiate` of a PanelSettings asset;
   `JsonUtility` reading `"parent": null` into private nested DTO classes; and the
   self-tests themselves, which are the proof the frames are right.
2. **The Isaac worm is not in its training simulator.** Isaac Sim's PhysX 5 and Unity's
   PhysX differ (solver, contact offsets, no viscous joint friction, no armature). Expect
   it to race below its report speed; the self-tests only prove the wiring is right.
3. **Damping model** (section 6). The default explicit damping is the trained model; if it
   misbehaves the implicit options are one setting away.
4. **MuJoCo solver.** worm.xml's `ls_iterations="8"` has no plug-in field, so MuJoCo uses
   its default 50 (slightly more accurate than training). Training used MuJoCo Warp; this
   is C MuJoCo 3.x. Small numerical differences are possible.
5. **Stand-ins are one-way and a step late**, and a mocap body carries no velocity into
   MuJoCo's friction. Fine for accidental bumps; not a model of worm-to-worm wrestling.
6. **MuJoCo is Windows-only here** (`mujoco.dll` only, AGENTS rule F). Elsewhere the MuJoCo
   worm is marked failed with that reason and the Isaac worm races alone.
7. **If the generated MJCF fails to compile**, the MuJoCo plug-in itself stops play mode
   (`MjScene.CreateScene`). The CLI then reports "play mode exited before the job finished";
   the console has the MuJoCo error.
8. **Finish rule** uses the nose (the tip of segment 0's capsule, 0.12 m ahead of its
   centre), the same point for both worms, with the crossing interpolated inside the
   physics step.

## 9. Removing it

Delete `Assets/Scripts/WormRace`, `Assets/Scripts/Editor/WormRace`, `Assets/WormRace` and
`Assets/Scenes/SCN_WORM_RACE.unity` (with their `.meta` files) in the Unity editor.
Nothing else in the project refers to them.

---

**TL;DR:** copy the staged folder into Assets plus the rig and two ONNX files, run the scene builder, pass the zero/yaw/pitch self-tests, then `Editor_WormRace.Start(5)`; results land in `Logs/wormrace_*.json`.
