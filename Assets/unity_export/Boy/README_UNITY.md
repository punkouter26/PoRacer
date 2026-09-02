# Boy - Unity side

The Isaac Lab port of the authored `Boy_Character` (FBX in `Assets/Art/Models/`, GLB twin
in `ISAAC/boy_rig/`). Same shape as the IsaacH1 port: Inference Engine, no ML-Agents, one
rig JSON as the source of truth, a rung ladder of play-mode tests.

## Files

| Path | What |
|---|---|
| `boy_rig.json` | The rig in the Isaac frame, from `ISAAC/boy_rig/build_boy_rig.py` (rewritten by `export_bundle.py` after training). |
| `kinematics_reference.json` | Independent Python FK of the rig, 3 poses. |
| `BoyRig.asset` | `boy_rig.json` as a ScriptableObject (**Boy > Rebuild Rig Asset From JSON**). |
| `Boy.prefab` | The articulation + skin (**Boy > Build Prefab**). |
| `PM_Boy.physicMaterial` | 0.8 / 0.6, Minimum combine. |
| `Boy.onnx`, `isaac_reference.json`, `export_report.json` | Arrive from `ISAAC/scripts/export_bundle.py`. Overwrite in place; the `.meta` GUIDs must survive. |
| `CONTRACT.md` | The index-by-index contract. |
| `Runtime/BoyAgent.cs` | The controller. |
| `Runtime/BoyRigAsset.cs`, `BoyPaths.cs`, `BoyTargetSampler.cs` | Rig data, paths + loaders, training-style ring target. |
| `../Editor/BoyRigBuilder.cs`, `BoySetup.cs` | Prefab builder, menu items, spawn window, reference check. |
| `../Tests/BoyPlayModeTests.cs` | The rung ladder. |
| `Assets/Scripts/Agents/Agent_Boy.cs` | Race adapter (`ICreatureAgent`). |
| `Assets/Scripts/Editor/Editor_RegisterBoyRacer.cs` | **PoRacer > Creatures > Register Boy Racer**: `Assets/Prefabs/Boy_v01.prefab` + catalog entry. |

## Workflow

1. `python ISAAC/boy_rig/build_boy_rig.py` (already run).
2. **Boy > Rebuild Rig Asset From JSON**, **Boy > Build Prefab**.
3. Test Runner > PlayMode > `Boy.Tests`. Everything that does not need a brain must be green.
4. Train and export (`ISAAC/README.md`). The exporter drops `Boy.onnx`, the recording and the
   report here and rewrites `boy_rig.json`.
5. Repeat step 2 (the joint order may have changed), rerun the tests: rung 0 and rung 5 now run.
6. **PoRacer > Creatures > Register Boy Racer** to put him on the grid.

## Design notes

* **Zero pose = T-pose.** All link frames are world-aligned at zero; the standing pose is a
  joint-angle offset (`defaultPosRad`). The skinned mesh bones are attached in the T-pose
  and the drives take the rig to the default pose.
* **Bones ride on links.** The physics rig is built from JSON as clean empties; the FBX
  bones are re-parented under them. The builder checks every bone against the JSON to
  5 mm after trying the four importer yaws and aborts on a mismatch, printing the hips
  height so an FBX scale problem is obvious.
* **No target = hold pose.** `BoyAgent.holdPoseWithoutTarget` keeps the drives at the
  default pose instead of feeding the policy a zero target it never saw. The race sets the
  finish line through `Agent_Boy.SetGoal`.
* **Fall recovery** is on for standalone scenes and switched off by `Agent_Boy` (RacerView
  owns rescue and retire in a race).
* **Per-body physics** (contact offset, solver iterations, velocity caps, damping) is
  re-applied every `Awake`, because Unity does not serialise it. The project's step and
  solver settings are never touched.
* The head mesh is 53k vertices and the whole skin is ~70k; eight racers is ~560k skinned
  vertices. Fine on desktop; a decimated head is the first thing to do for Android.
