# Worm race (SCN_WORM_RACE): how to install, build, run and check it

Everything in this folder mirrors where it goes under the project root. Nothing here is
under `Assets/` yet, on purpose: a long exam was running in the Unity editor, and any new
file under `Assets/` triggers a recompile that would have killed it.

Three worms race down three straight lanes, 20 m long and 2 m apart, centred on x = 0:

| Lane | Racer | Method | Colour | Brain | Physics inside Unity (default) |
|---|---|---|---|---|---|
| 0 (x = -2 m) | **MuJoCo worm** | MuJoCo | BLUE | `worm_mujoco.onnx` | MuJoCo, through the `org.mujoco` plug-in (the simulator it trained in) |
| 1 (x = 0 m) | **Isaac worm** | Isaac Lab | ORANGE | `worm_isaac.onnx` | PhysX `ArticulationBody` (Unity's own physics) |
| 2 (x = +2 m) | **Isaac3Worm** | Isaac Lab 3 | PURPLE | `worm_isaaclab3.onnx` | MuJoCo plug-in (it trains on Isaac Lab 3's Newton backend with the MuJoCo-Warp solver); switch to PhysX if that trainer falls back to PhysX (section 6) |

All have the same body (`worm_rig.json`), the same 35-number observation and the same
8 actions (`training/worm/WORM_SPEC.md`). Red and green are not used (AGENTS rule D).

The racers are a list in `WormRaceSettings` (section 6), one entry per lane, so a fourth
racer is one more entry plus a scene rebuild; nothing in the code knows how many lanes
there are or which lane runs in which simulator.

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

# 3. The brains (once the training runs have exported them).
Copy-Item training\worm\export\worm_mujoco.onnx     Assets\WormRace\Brains\worm_mujoco.onnx
Copy-Item training\worm\export\worm_isaac.onnx      Assets\WormRace\Brains\worm_isaac.onnx
Copy-Item training\worm\export\worm_isaaclab3.onnx  Assets\WormRace\Brains\worm_isaaclab3.onnx
```

After copying a brain, run the scene builder again (section 2) so the settings pick it up.

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
race, the HUD says "NO BRAIN", and the console and the results file (`brainError`) say
exactly which file to copy where. As of 2026-09-24 `worm_isaaclab3.onnx` does not exist
yet, so lane 2 races that way.

## 2. Build the scene

```powershell
unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build();"
```

(Also in the menu: **PoRacer > Worm Race > Build Scene**.) Save or discard any unsaved
scene first; the builder refuses to run over one.

Expected output, roughly:

```
materials: blue (MuJoCo), orange (Isaac), purple (Isaac Lab 3), ground, line white/black
physics material: Assets/WormRace/PM_WormRace.physicMaterial (0.90 static/dynamic, no bounce)
panel settings: Assets/WormRace/UI/WormRacePanelSettings.asset
settings: Assets/WormRace/WormRaceSettings.asset
  rig: Assets/WormRace/worm_rig.json
  lane 0: MuJoCo worm (MuJoCo, MuJoCo plug-in) brain: Assets/WormRace/Brains/worm_mujoco.onnx
  lane 1: Isaac worm (Isaac Lab, PhysX ArticulationBody) brain: Assets/WormRace/Brains/worm_isaac.onnx
  lane 2: Isaac3Worm (Isaac Lab 3, MuJoCo plug-in) brain: MISSING - copy training/worm/export/worm_isaaclab3.onnx to Assets/WormRace/Brains/worm_isaaclab3.onnx (the worm lies still, HUD says NO BRAIN)
scene objects: WormRaceLifetimeScope, Directional Light, Main Camera, Track (3 lanes), WormRaceHud
saved Assets/Scenes/SCN_WORM_RACE.unity (checked on disk: scope -> settings, HUD -> panel settings)
```

A line saying `MISSING - copy ...` means that file is not in place yet. Copy it and run the
builder again; it is safe to re-run and keeps every asset's GUID. A re-run refreshes each
known racer's name, method, colour, material and brain, but **keeps its Physics choice**
(a racer's first run adds it with the default in the table above, marked `new`). Entries
added by hand after the three known ones are kept and get a lane.

The last line is a check, not a claim: after saving, the builder reads the `.unity` file
back and looks for the scope's `_settings` link and the HUD's `PanelSettings` link by GUID.
If either is missing it ends with `ERROR` instead of `saved`.

*Fixed 2026-09-24:* earlier builds saved `WormRaceLifetimeScope._settings` (and the HUD's
`PanelSettings`) as `{fileID: 0}`. The builder loaded those assets first and then called
`EditorSceneManager.NewScene(..., Single)`, which unloads every asset nothing in memory
references - the settings and panel assets were held only in local variables, so they were
destroyed under their C# references, and assigning a destroyed object through
`SerializedObject` stores nothing. The builder now creates the new scene first and loads
the assets after it.

What it creates (all authored once, AGENTS rule G): the ground (the only track collider,
its top at y = 0), one white lane line per lane edge (four for three lanes), a white start line at z = 0, a chequered
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

Run them one at a time: wait until `Status()` says `done` (and `unity cmd editor_status`
shows `"playMode":"stopped"`) before starting the next. Entering play mode can time out on
the CLI bridge; if a launch fails, wait and retry.

Each self-test spawns every racer, holds them straight for 1 s so they settle, then
replaces every brain with a fixed action and measures. The same code measures every worm,
each in its own physics. The tests do not use the brains, so a racer with NO BRAIN (lane 2
today) is tested exactly like the others and must pass too.

| Test | Fixed action | Must be true for EVERY worm (the file's `passed` / `verdict`) |
|---|---|---|
| `zero` (5 s) | all zero | Lies still: nose moves < 2 cm and speed < 1 cm/s. Stays straight: every joint within 0.05 rad. Lies on the floor: segment 2's centre at 0.045 m (the radius) ± 1 cm |
| `yaw` (2 s) | j0_yaw target = +0.5 rad | WORM_SPEC sign test: j0_yaw reads more than +0.3 rad, and segment 1 is more than 2 cm to the worm's **right = Unity +x**. Expect about +0.45 to +0.5 rad and about +4.5 cm |
| `pitch` (2 s) | j0_pitch target = +0.5 rad | j0_pitch reads more than +0.3 rad, and segment 1 is more than 2 cm **above** the head's plane. Expect about +4 cm |

`"allPassed": true` in each file is the bar. Since 2026-09-24 (the creature template, section 10)
each racer entry reports `probeOffsetMeters` (segment 1 minus segment 0 in segment 0's frame,
MuJoCo x forward, y left, z up), `probeAlongAxisMeters` (the tested direction: -y for yaw, +z
for pitch; it replaces `secondSegmentLateralMeters` / `secondSegmentVerticalMeters`),
`leadDisplacementMeters` (was `noseDisplacementMeters`), `referenceHeightMeters` and
`referenceUpright`; the zero test's verdict reads `still= atRestPose= atHeight=` (was
`still= straight= onFloor=`). The numbers and pass rules are unchanged: re-run on the template,
zero/yaw/pitch gave exactly the values below (yaw +0.4834 rad / 4.65 cm MuJoCo, +0.4388 rad /
4.25 cm PhysX; pitch +0.4609 rad / 4.45 cm, +0.4638 rad / 4.47 cm). If a test fails:

- **yaw or pitch fails on the PhysX worms only** (segment 1 swings left, or the angle reads
  negative): the PhysX axis map is off. Check `CreatureFrames.UnityFromSpecAxial` and the
  anchor rotation in `PhysxWormBuilder`. The MuJoCo worm's signs come from MuJoCo itself.
- **yaw or pitch fails on the MuJoCo worms only**: the plug-in built a different body.
  Diff the generated MJCF (section 5). **On one MuJoCo worm only**: its ids or qpos/ctrl
  addresses are crossed with the other MuJoCo worm's (section 7, "Several MuJoCo worms").
- **zero fails on a PhysX worm** by creeping or twitching: the passive joint damping
  mode. Try the other settings in `WormRaceSettings > Passive Damping` (section 6).
- **zero fails with the worm sitting 1-2 cm into the floor on MuJoCo**: soft-contact
  sinking. It is not a sign error; widen the tolerance or look at `solref`.

Results on 2026-09-24 (three lanes, lane 2 on the MuJoCo plug-in with no brain): all
three passed in all three tests. Yaw: j0_yaw +0.483 rad and segment 1 +4.65 cm right on
both MuJoCo worms, +0.439 rad and +4.25 cm on the PhysX worm. Pitch: +0.461 rad and
+4.45 cm up (MuJoCo), +0.464 rad and +4.47 cm (PhysX). With lane 2 switched to PhysX,
zero and yaw passed as well (lane 2 then matched lane 1 exactly).

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

Each race: every worm appears straight on the start line; **3-2-1** with every brain held
(physics runs, so all settle onto the floor the same way); **GO**; the first nose (the tip
of segment 0) across the finish line wins. If a worm has not finished after **60 s** it is
ranked by distance. The race ends when every worm has finished, timed out or failed.
Between races the results stay up for 4 s, then the worms are removed and respawned.

The HUD (top left) shows one row per lane: name, training method, physics, distance (m),
speed (m/s) and state, plus "NO BRAIN (holds straight)" for a racer without a brain; the
centre shows the countdown; the bottom-right panel shows each race's places, finish times
(or "time limit"), distances, average speeds and the running score. The camera follows the
average nose of the worms that have a brain, so a NO BRAIN worm on the start line does not
drag the frame back.

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
          "brain": "worm_mujoco", "brainLoaded": true, "brainError": "", "status": "Finished", "place": 1,
          "finishTimeSeconds": 44.15, "distanceMeters": 20.0, "averageSpeedMps": 0.453, "failReason": "" },
        { "lane": 1, "name": "Isaac worm", "method": "Isaac Lab", "physics": "PhysX ArticulationBody",
          "status": "TimedOut", "place": 2, "finishTimeSeconds": -1.0, "distanceMeters": 1.07,
          "averageSpeedMps": 0.018, "...": "..." },
        { "lane": 2, "name": "Isaac3Worm", "method": "Isaac Lab 3",
          "physics": "MuJoCo (org.mujoco plug-in)", "brain": "missing", "brainLoaded": false,
          "brainError": "Isaac3Worm: no brain assigned. Copy the exported ONNX to Assets/WormRace/Brains/worm_isaaclab3.onnx and re-run ...",
          "status": "TimedOut", "place": 3, "distanceMeters": -0.01, "...": "..." } ] } ],
  "summary": [
    { "lane": 0, "name": "MuJoCo worm", "method": "MuJoCo", "physics": "MuJoCo (org.mujoco plug-in)",
      "brain": "worm_mujoco", "brainLoaded": true, "brainError": "", "wins": 1, "finishes": 1,
      "meanFinishTimeSeconds": 44.15, "bestFinishTimeSeconds": 44.15, "meanDistanceMeters": 20.0,
      "meanAverageSpeedMps": 0.453 },
    { "lane": 1, "name": "Isaac worm", "...": "..." },
    { "lane": 2, "name": "Isaac3Worm", "brainLoaded": false, "brainError": "...", "...": "..." } ]
}
```

(From a real 1-race run.) One `racers` entry per lane in finishing order, and one
`summary` entry per lane in lane order. `-1` means "does not apply". `endReason` is `finish`,
`timeLimit` (nobody finished) or `failed` (every racer out). The self-test files carry the
same `method`, `physics`, `brain`, `brainLoaded` and `brainError` per racer. `status` is `Finished`, `TimedOut` or
`Failed`; a failed worm has a `failReason` (non-finite physics, a speed above 500 per
WORM_SPEC detail 12, or MuJoCo not available on this platform).

What to expect: the trainers' own reports (`training/worm/export/*_report.json`) give each
brain's mean speed in its own simulator. **The MuJoCo worm races in the simulator it
trained in, so it should roughly match its report** (0.45 m/s, 20 m in about 44 s). The
Isaac worm trained in Isaac Sim's PhysX and races in Unity's PhysX; a gap between its
report speed and its Unity speed is a sim-to-sim gap, not a race result, and is worth
writing down on its own.

**The Isaac worm's distance depends on the track layout.** Its Unity gait barely moves and
is chaotic: tiny numerical differences grow into very different races. With the same code
it covered 2.95 m in 60 s in the old two-lane layout (lanes at x = +/-1, 7 m wide ground;
reproduced exactly), -2.52 m in a two-lane test on the wider three-lane ground, and 1.07 m
in the three-lane layout (lane 1 at x = 0; repeated exactly). It never diverged. Compare it
only within one layout.

## 5. Check the MuJoCo worm is worm.xml

With `WormRaceSettings > Dump Mujoco Mjcf` on (the default), every race writes the model
the plug-in actually compiled to `Application.temporaryCachePath/wormrace_mujoco_scene.xml`
(the console prints `MJCF saved to <path>`, usually under
`%LOCALAPPDATA%\Temp\<company>\<product>\`). Compare it with `training/worm/worm.xml`.
Names get a `_<n>` suffix and every attribute is written out explicitly; ignore that.
What must match:

There is one MJCF for the whole race: every MuJoCo racer is in it (two with the default
roster), so everything below appears once **per MuJoCo worm**. The plug-in orders the
`<actuator>` section by each actuator's position under its own parent, so the worms'
actuators come interleaved (`j0_pitch_13`, `j0_pitch_40`, `j0_yaw_16`, ...). That is
harmless: each worm's view looks its actuators up by their generated names, never by
position.

- 5 capsule geoms, `size="0.045 0.075"`, `mass="1.335962"`, friction `0.9 0.005 0.0001`,
  `condim="3"`, `solref="0.01 1"`, `solimp="0.9 0.95 0.001"`.
- 4 link bodies at `pos="-0.1 0 0"` with `<inertial mass="0.1" diaginertia="0.0002 0.0002 0.0002">`.
- 8 hinges: pitch `axis="0 1 0"` at the link origin; yaw `axis="0 0 1"` at `pos="0.1 0 0"`;
  `range="-45 45"` (degrees - the plug-in's compiler uses MuJoCo's degree default),
  `limited="true"`, `damping="2"`, `armature="0.01"`.
- 8 `<position>` actuators in action order: `kp="30"`, `ctrlrange="-0.785398 0.785398"`,
  `forcerange="-6 6"`, `ctrllimited="true"`, `forcelimited="true"`.
- 4 `<exclude>` pairs seg0-seg1, seg1-seg2, seg2-seg3, seg3-seg4 per worm, each inside one
  worm (8 in total with two MuJoCo worms). Nothing excludes one worm from another: MuJoCo
  worms collide with each other natively.
- `<option>`: `integrator="implicitfast"`, `solver="Newton"`, `iterations="10"`,
  `cone="pyramidal"`, `timestep="0.005"`, `gravity="0 0 -9.81"`.
- seg0's `quat` is a +90° turn about z (`0.7071 0 0 0.7071`, or its negative - the
  same rotation). That is the lane direction: the plug-in maps Unity +z to MuJoCo +y.
  Each worm's seg0 `pos` x is its lane (-2 and +2 with the default roster).
- Extra, and expected: the floor plane and five `mocap="true"` bodies per PhysX worm,
  named `MocapProxy_L<lane>_Seg<n>` (their collision stand-ins, section 7).

## 6. The knobs (`Assets/WormRace/WormRaceSettings.asset`)

Since the creature template (section 10) everything except the PhysX knobs sits under the
asset's **Race** block (the template's `CreatureRaceConfig`): racers, track, race timing,
observation, self-tests, MuJoCo options and labels. `Editor_BuildWormRaceScene` writes all of
it; re-run it rather than editing by hand, except for a racer's Physics choice, which it keeps.

The body itself (masses, gains, limits, friction) is **not** here; it comes from
`worm_rig.json`, so an Inspector edit cannot make the worms different animals.

**Racers** - the list of lanes, lane 0 first. Each entry:

| Field | What it is |
|---|---|
| Name / Method | HUD and results labels ("Isaac3Worm" / "Isaac Lab 3") |
| **Physics** | **`Mujoco Plugin`** or **`Physx Articulation`**: the simulator that steps this worm in Unity |
| Brain / Brain File | the ONNX, and its expected file name under `Assets/WormRace/Brains/` (used to find it and for the missing-brain message) |
| Material / Color | the segment material and the HUD swatch (never red or green, rule D) |

**Choosing lane 2's physics.** The Isaac3Worm defaults to **Mujoco Plugin**, because
it trains on Isaac Lab 3's Newton backend with the MuJoCo-Warp solver, whose contact and
joint model is MuJoCo's. If that trainer falls back to PhysX, set lane 2's Physics to
**Physx Articulation** (Inspector: `Assets/WormRace/WormRaceSettings.asset > Race >
Racers > Element 2 > Physics`). It takes effect on the next play; the scene builder keeps the choice
on re-runs. Any lane can be switched the same way. PhysX racers use the PhysX settings
below (passive damping, solver iterations); MuJoCo racers use the MuJoCo ones.

| Setting | Default | What it does |
|---|---|---|
| Track Length / Lane Spacing / Start Line Z | 20 / 2 / 0 | Race geometry; lanes are Lane Spacing apart and centred on x = 0. Re-run the scene builder after changing these, or the number of racers, so the painted lines move too |
| Countdown Seconds / Time Limit Seconds / Results Hold Seconds | 3 / 60 / 4 | Race timing |
| Series Length / Auto Start Series | 5 / on | What pressing Play does. The CLI overrides both |
| Passive Damping | **Drive Damping Per Radian** | PhysX worms only, see below. Keep it: Joint Force Outside Limit diverged |
| Solver Iterations / Velocity Iterations | 16 / 4 | PhysX worms, per body; no project setting is touched |
| Fold Armature Into Inertia | on | PhysX has no armature; adds MuJoCo's 0.01 to each hinge's child inertia about its axis |
| Mujoco Solver Iterations | 10 | worm.xml's value |
| Cross Simulator Proxies | on | Collision stand-ins between MuJoCo worms and PhysX worms (section 7) |
| Previous Action Clipped | on | WORM_SPEC detail 5: the previous-action observation uses clipped actions |

**Passive Damping.** WORM_SPEC detail 16 says the joint damping is a passive
`-c·q̇` that the servo's force limit does not cap, and the servo is `kp` with no damping.
Unity's `ArticulationBody` has no viscous joint friction. *Joint Force Outside Limit*
applies `-c·q̇` through `ArticulationBody.jointForce` every physics step (drive damping 0):
that matches the trained model, but it is explicit, and on these light links the Isaac
worm diverged in every race with it (RESULTS.md, problem 3). *Drive Damping Per Radian*
(the setting in use, and now also the code default) puts the damping inside the drive:
implicit and stable, but capped together with the servo by the force limit. *Per Degree*
is what `MujocoBiped` measured drive damping to behave like on this Unity version.

## 7. How it is put together

Architecture (`.claude/rules`): plain models, plain systems in VContainer, thin views,
MessagePipe messages, UniTask for the countdown and the race loop.

| Kind | Types |
|---|---|
Since 2026-09-24 the worm runs on the creature template (section 10); only the worm-specific
pieces are still in `Assets/Scripts/WormRace`.

| Kind | Types |
|---|---|
| Model | `CreatureRaceModel`, `CreatureRacerModel` (template) |
| Config | `WormRaceSettings` (asset): its `Race` block is the template's `CreatureRaceConfig` (racers as `CreatureRacerDefinition`, `CreaturePhysicsKind` selects the simulator; self-tests as `CreatureSelfTestDefinition` data), plus the PhysX knobs |
| System | `CreatureRaceSystem` (template entry point: series, race, self-test flow, one pilot per lane), `WormSpawnSystem` (the worm's `ICreatureSpawner`: MuJoCo worms via the template builder, PhysX worms, the one MuJoCo world, stand-ins) |
| Per-racer logic | `CreaturePilot` (hold, decimation, observation, inference, clipping, race clock), `CreaturePolicy`, `CreatureObservation` (the 35 numbers, per the settings' observation definition) |
| Body | `WormRig` (worm_rig.json) -> `WormRigAdapter` -> `CreatureRig` for MuJoCo; `PhysxWormBuilder` still builds from `WormRig` |
| View | `MujocoCreatureView` (template), `PhysxWormView` (physics adapters, no decisions), `CreatureRaceHudView`, `CreatureRaceCameraView`, two proxy followers |
| Messages | `CreatureCountdownMessage`, `CreatureRaceFinishedMessage`, `CreatureSeriesFinishedMessage` |
| Scope | `WormRaceLifetimeScope` (`CreatureRaceInstaller` + `WormSpawnSystem`) |

**How the MuJoCo worm gets into Unity.** Not through the MJCF importer. `MujocoCreatureBuilder` (fed by `WormRigAdapter`)
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
- Building from `worm_rig.json` means every worm comes from the same file.

**Several MuJoCo worms.** All MuJoCo racers of a race live in the race's one `MjScene`:
`WormSpawnSystem` creates the world, then every MuJoCo worm and every mocap stand-in in the
same frame, so the plug-in compiles them into one model at the start of the next frame
(a worm created later would never be simulated). The plug-in names every element
`<GameObject>_<counter>`, so the worms' `seg0`s become e.g. `seg0_8` and `seg0_35`. Each
worm's view keeps its own `MjBody`/`MjHingeJoint`/`MjActuator` components and reads their
`MujocoId`, `QposAddress` and `DofAddress` after the compile, so body ids, qpos/qvel
addresses and ctrl indices are per worm and never shared; each view subscribes its own
control callback and writes only its own ctrl entries. Excludes are only between adjacent
segments of one worm; two MuJoCo worms collide with each other natively. The yaw and pitch
self-tests prove this per worm (a crossed index would move the wrong worm).

Section 5's dump is the check that the result is worm.xml.

**How MuJoCo and PhysX share one scene.** They do not share anything physical. MuJoCo
steps inside `MjScene.FixedUpdate` and writes the MuJoCo worms' transforms; PhysX steps
the PhysX worms after all `FixedUpdate`s. Both run at `Time.fixedDeltaTime` = 0.005 s, one
step per `FixedUpdate`, and each worm's pilot counts its own physics steps, so the race
clock is fair whichever simulator Unity runs first. Each has its own floor at y = 0: an
infinite MuJoCo plane, and the track's box collider (friction 0.9, `Maximum` combine,
like MuJoCo's max rule).

Because neither simulator can see the other, every worm gets a copy of its five capsules
in the other simulator as a collision stand-in (AGENTS rule M): five MuJoCo mocap bodies
per PhysX worm (`MocapProxy_L<lane>_Seg<n>`), and five kinematic PhysX capsules per MuJoCo
worm (`KinematicProxy_L<lane>_Seg<n>`). So every MuJoCo/PhysX pair is covered, whatever
the lane count and whichever physics each lane picks. Worms in the same simulator need no
stand-ins: MuJoCo worms share the MjScene, PhysX worms share the PhysX scene. Stand-ins are
one-way (each worm sees the other as an unstoppable moving object) and one 5 ms step late.
With the lanes 2 m apart a contact would be an accident; the stand-ins make it a collision
instead of the worms passing through each other.

**Observation frames.** A MuJoCo worm reads everything from MuJoCo's own data, in the
plug-in's MuJoCo frame (Unity (x, y, z) = MuJoCo (x, z, y)), including segment 2's
velocity via `mj_objectVelocity` - the same quantities the trainer computes. A PhysX
worm reads `ArticulationBody` state and converts with WORM_SPEC's Unity mapping
(positions (z, -x, y), axes (-z, x, -y)). Both then go through `CreatureObservation`, which only
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

1. **Run in Unity since.** The self-tests pass for all three lanes and races run (sections
   3 and 4); the notes below are from before the first run and are kept for the record.
   Both assemblies were first compiled outside Unity with
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
6. **MuJoCo is Windows-only here** (`mujoco.dll` only, AGENTS rule F). Elsewhere every MuJoCo
   racer is marked failed with that reason and the PhysX racers race alone.
7. **If the generated MJCF fails to compile**, the MuJoCo plug-in itself stops play mode
   (`MjScene.CreateScene`). The CLI then reports "play mode exited before the job finished";
   the console has the MuJoCo error.
8. **Finish rule** uses the nose (the tip of segment 0's capsule, 0.12 m ahead of its
   centre), the same point for every worm, with the crossing interpolated inside the
   physics step.
9. **Countdown timing is wall-clock**, so the number of physics steps a worm spends settling
   before GO varies a little between runs; the MuJoCo worm's finish time moves by a few
   tenths of a second (44.1-44.7 s seen) for that reason.
10. **The HUD writes glyphs into `Assets/UI/PoRacerFont SDF.asset`** (a dynamic font atlas)
    the first time it draws new characters in play mode. Revert that file if it shows up
    in `git status` after a race and nothing else changed it.

## 9. Removing it

Delete `Assets/Scripts/WormRace`, `Assets/Scripts/Editor/WormRace`, `Assets/WormRace` and
`Assets/Scenes/SCN_WORM_RACE.unity` (with their `.meta` files) in the Unity editor.
Nothing else in the project refers to them.

---

## 10. The creature template, and the quad race

The worm race was turned into a template any MuJoCo-Warp-trained creature can race on
(`Assets/Scripts/CreatureRace`, assembly `PoRacer.CreatureRace`; editor side
`Assets/Scripts/Editor/CreatureRace`). A creature brings a rig JSON in the trainers' format
(`training/quad/quad_rig.json`: bodies with parent/pos/quat/mass/inertiaDiag/ipos/iquat,
geoms capsule|box|sphere with size and pos/quat or fromto and contact, hinge joints with
axis/pos/range/damping/armature/solreflimit, position actuators with kp/ctrlrange/forcerange,
optional excludes, `actionOrder`, `restPose`, `actionScaleRad`, `physics.decimation`,
`torso`, `spawnRootHeight`, `task.targetSpeed`), a settings asset and a scene builder. The
template supplies the MuJoCo builder and view, the observation (reference body, goal,
optional target speed / divisor; worm 35, quad 36), the pilot (target = rest + a x scale,
decimation from the rig), the race system, HUD, camera, reports and a data-driven self-test
list. `CreatureRaceSceneKit` and `CreatureRaceHarness` are the shared builder and CLI pieces.

**Quad race** (`Assets/Scenes/SCN_QUAD_RACE.unity`, `Assets/QuadRace/`): two lanes 3 m apart,
30 m, 60 s. "Quad (MuJoCo)" BLUE with `quad_mujoco.onnx`, "Quad (Isaac Lab 3)" PURPLE with
`quad_isaaclab3.onnx`, both on the MuJoCo plug-in. The builder copies
`training/quad/quad_rig.json` and any `training/quad/export/quad_*.onnx` into
`Assets/QuadRace/` when they change, so re-run it after the trainers export. No righting
(rule H): a fallen quad keeps its own policy; the HUD shows `RACING (DOWN)` and the report
`fellOver` / `falls`.

```powershell
unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_BuildQuadRaceScene.Build();"
unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.SelfTest(\"zero\");"   # also "hip", "knee"
unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.Start(5);"
unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.Status();"
```

Self-tests: `zero` (every servo at rest: stands still, joints within 0.1 rad, torso at the
rig's rest height +/- 5 cm, upright >= 0.9), `hip` / `knee` (+0.5 rad on the front-left hip /
knee must read positive and move the front-left foot > 2 cm backward, MuJoCo -x, relative to
the torso: training/bugs/README.md's sign convention). Results in `Logs/quadrace_*.json`.
First run, 2026-09-24, no brains yet: zero passed (torso 0.8998 m, upright 1.0, no drift),
hip passed (+0.61 rad, foot 0.38 m back), knee passed (+0.505 rad, foot 0.142 m back), and a
1-race run completed with both quads standing at the line (NO BRAIN).

**Superseded worm files.** The worm's own copies of what moved into the template are compiled
out (`#if PORACER_WORMRACE_LEGACY`, content unchanged) and marked `// SUPERSEDED by ...` in
their first line; delete them and their `.meta` files when convenient.

**TL;DR:** three lanes (MuJoCo, Isaac, Isaac Lab 3 with a MuJoCo/PhysX switch in the settings); copy brains, run the builder, pass zero/yaw/pitch, then `Editor_WormRace.Start(5)`.
