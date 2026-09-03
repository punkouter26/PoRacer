# IsaacBox - Unity side

The Isaac Lab port of the authored `IsaacBox_Character` (FBX in `Assets/Art/Models/`, GLB twin
in `ISAAC/boy_rig/`). Same shape as the IsaacH1 port: Inference Engine, no ML-Agents, one
rig JSON as the source of truth, a rung ladder of play-mode tests.

## Files

| Path | What |
|---|---|
| `isaacbox_rig.json` | The rig in the Isaac frame, from `ISAAC/boy_rig/build_boy_rig.py` (rewritten by `export_bundle.py` after training). |
| `kinematics_reference.json` | Independent Python FK of the rig, 3 poses. |
| `IsaacBoxRig.asset` | `isaacbox_rig.json` as a ScriptableObject (**IsaacBox > Rebuild Rig Asset From JSON**). |
| `IsaacBox.prefab` | The articulation + skin (**IsaacBox > Build Prefab**). |
| `PM_IsaacBox.physicMaterial` | 0.8 / 0.6, Minimum combine. |
| `IsaacBox.onnx`, `isaac_reference.json`, `export_report.json` | Arrive from `ISAAC/scripts/export_bundle.py`. Overwrite in place; the `.meta` GUIDs must survive. |
| `CONTRACT.md` | The index-by-index contract. |
| `Runtime/IsaacBoxAgent.cs` | The controller. |
| `Runtime/IsaacBoxRigAsset.cs`, `IsaacBoxPaths.cs`, `IsaacBoxTargetSampler.cs` | Rig data, paths + loaders, training-style ring target. |
| `../Editor/IsaacBoxRigBuilder.cs`, `IsaacBoxSetup.cs` | Prefab builder, menu items, spawn window, reference check. |
| `../Tests/IsaacBoxPlayModeTests.cs` | The rung ladder. |
| `Assets/Scripts/Agents/Agent_IsaacBox.cs` | Race adapter (`ICreatureAgent`). |
| `Assets/Scripts/Editor/Editor_RegisterIsaacBoxRacer.cs` | **PoRacer > Creatures > Register IsaacBox Racer**: `Assets/Prefabs/IsaacBox_v01.prefab` + catalog entry. |

## Workflow

1. `python ISAAC/boy_rig/build_boy_rig.py` (already run).
2. **IsaacBox > Rebuild Rig Asset From JSON**, **IsaacBox > Build Prefab**.
3. Test Runner > PlayMode > `IsaacBox.Tests`. Everything that does not need a brain must be green.
4. Train and export (`ISAAC/README.md`). The exporter drops `IsaacBox.onnx`, the recording and the
   report here and rewrites `isaacbox_rig.json`.
5. Repeat step 2 (the joint order may have changed), rerun the tests: rung 0 and rung 5 now run.
6. **PoRacer > Creatures > Register IsaacBox Racer** to put him on the grid.

## Design notes

* **Zero pose = T-pose.** All link frames are world-aligned at zero; the standing pose is a
  joint-angle offset (`defaultPosRad`). The skinned mesh bones are attached in the T-pose
  and the drives take the rig to the default pose.
* **Bones ride on links.** The physics rig is built from JSON as clean empties; the FBX
  bones are re-parented under them. The builder checks every bone against the JSON to
  5 mm after trying the four importer yaws and aborts on a mismatch, printing the hips
  height so an FBX scale problem is obvious.
* **No target = hold pose.** `IsaacBoxAgent.holdPoseWithoutTarget` keeps the drives at the
  default pose instead of feeding the policy a zero target it never saw. The race sets the
  finish line through `Agent_IsaacBox.SetGoal`.
* **Fall recovery** is on for standalone scenes and switched off by `Agent_IsaacBox` (RacerView
  owns rescue and retire in a race).
* **Per-body physics** (contact offset, solver iterations, velocity caps, damping) is
  re-applied every `Awake`, because Unity does not serialise it. The project's step and
  solver settings are never touched.
* The head mesh is 53k vertices and the whole skin is ~70k; eight racers is ~560k skinned
  vertices. Fine on desktop; a decimated head is the first thing to do for Android.
