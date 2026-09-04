# Driving the Editor from the CLI

Every authoring action in this project is reachable from the command line against
a **running** Editor — no menu clicking, no batch-mode restart. The Unity Pipeline
package (`com.unity.pipeline`) serves a bridge; `unity status` shows it:

    unity status
    # Port  State  Project                            Version      PID
    # 7800  ready  C:\Users\punko\Downloads\PoRacer   6000.5.8f1   31300

`unity list` prints every command the bridge exposes (~150: scene, GameObject,
prefab, asset, package, build, test, profiling and play-mode control).

## The two ways in

**1. Run a menu action by path.** Every `[MenuItem]` below is invocable directly:

    unity cmd menu --path "PoRacer/Creatures/Rebuild Fido Prefab"

**2. Call the method.** Every menu action is now `public static`, so it also works
through `-executeMethod` (batch mode) and through `eval` / `eval_file` on the live
Editor. Prefer this when you want a return value or need to pass arguments.

    unity cmd eval_file --file build.cs

> **Arguments use `--name value`, never `name=value`.** The bridge rejects the
> latter with "is not a parameter". Parameter names are not always what the docs
> imply — `move_asset` takes `--asset`, `eval_file` takes `--file`,
> `set_autotick` takes `--enable`. `unity cmd <name>` with no arguments prints the
> required ones.

## Action catalogue

| Menu path | Method |
|---|---|
| PoRacer/Build Android APK | `PoRacer.EditorTools.Editor_BuildAndroid.Build` |
| PoRacer/Build Android AAB (Play release) | `PoRacer.EditorTools.Editor_BuildAndroidAAB.Build` |
| PoRacer/Configure Android Release Settings | `PoRacer.EditorTools.Editor_ConfigureAndroidRelease.Apply` |
| PoRacer/Build FX Particle Materials | `PoRacer.EditorTools.Editor_BuildFxMaterials.Build` |
| PoRacer/Configure Rendering | `PoRacer.EditorTools.Editor_ConfigureRendering.Configure` |
| PoRacer/Sync Agent Observation Sizes | `PoRacer.EditorTools.Editor_SyncObservationSizes.Sync` |
| PoRacer/Report Orphaned Creature Brains | `PoRacer.EditorTools.Editor_ReportOrphanedBrains.Report` |
| PoRacer/Creatures/Rebuild Fido Prefab | `PoRacer.EditorTools.Editor_BuildFidoPrefab.Build` |
| PoRacer/Creatures/Register IsaacBox Racer | `PoRacer.EditorTools.Editor_RegisterIsaacBoxRacer.Register` |
| PoRacer/Creatures/Register MojucuBoy Racer | `PoRacer.EditorTools.Editor_RegisterMojucuBoyRacer.Register` |
| PoRacer/Creatures/Build Boy Race Scene | `CreatureEditor.MojucuBoySetup.Build` |
| PoRacer/Build Shared Training Scene (SCN_TRAIN_ALL) | `PoRacer.EditorTools.Editor_BuildSharedTrainingScene.BuildScene` |
| PoRacer/Build Humanoid Training Scene | `…Editor_BuildSharedTrainingScene.BuildHumanoidScene` |
| PoRacer/Build Focused Training Scene (SCN_TRAIN_FOCUSED) | `…Editor_BuildSharedTrainingScene.BuildFocusedScene` |
| PoRacer/Bake Authored Tracks into SCN_RACE_FLAT | `PoRacer.EditorTools.Editor_BakeAuthoredTrack.Bake` |
| PoRacer/Smoke-race every map in play mode | `PoRacer.EditorTools.Editor_SmokeRace.Start` / `.Status` |
| PoRacer/Smoke-play one scene | `PoRacer.EditorTools.Editor_SmokeRace.StartScene` |
| PoRacer/Build All-Creatures Training Env | `…Editor_BuildSharedTrainingScene.BuildEnv` |
| PoRacer/Build Humanoid Training Env | `…Editor_BuildSharedTrainingScene.BuildHumanoidEnv` |
| PoRacer/Training/Enable Demo Recorders In Open Scene | `…Editor_RecordDemos.EnableRecorders` |
| PoRacer/Training/Disable Demo Recorders In Open Scene | `…Editor_RecordDemos.DisableRecorders` |
| PoRacer/Training/Gauntlet - Add Selected Brains | `…Editor_EloGauntlet.AddSelectedBrains` ¹ |
| PoRacer/Training/Gauntlet - Remove Gauntlet Entries | `…Editor_EloGauntlet.RemoveGauntletEntries` |
| MuJoCo Creature/Build Verification Scene | `CreatureEditor.CreatureSceneBuilder.Build` ² |
| IsaacBox/Rebuild Rig Asset From JSON | `IsaacBox.EditorTools.IsaacBoxSetup.RebuildRigAsset` |
| IsaacBox/Build Prefab | `…IsaacBoxSetup.BuildPrefab` |
| IsaacBox/Rebuild Materials From GLB Textures | `…IsaacBoxMaterials.RebuildMenu` |
| IsaacBox/Spawn Into Open Scene (Defaults) | `…IsaacBoxSetup.SpawnIntoOpenSceneDefaults` |
| IsaacBox/Run Reference Check | `…IsaacBoxSetup.RunReferenceCheckMenu` |
| IsaacH1/Rebuild Rig Asset From JSON | `…IsaacH1Setup.RebuildRigAsset` |
| IsaacH1/Build Prefab | `…IsaacH1Setup.BuildPrefab` |
| IsaacH1/Import Original Meshes | `…IsaacH1MeshImporter.ImportMeshes` |
| IsaacH1/Import Decimated Meshes | `…IsaacH1MeshImporter.ImportDecimatedMeshes` |
| IsaacH1/Restore Full-Detail Meshes | `…IsaacH1MeshImporter.RestoreFullDetailMeshes` |
| IsaacH1/Run Reference Check | `…IsaacH1Setup.RunReferenceCheckMenu` |
| MujocoBiped/Rebuild Rig Asset From JSON | `…MujocoBipedSetup.RebuildRigAsset` |
| MujocoBiped/Build Prefab | `…MujocoBipedSetup.BuildPrefab` |
| MujocoBiped/Run Reference Check | `…MujocoBipedSetup.RunReferenceCheck` |

¹ Reads the Project-window selection. Set it first with `unity cmd find_assets`
plus a selection call, or call the method from `eval` with an explicit list.

² `Build` is the headless entry (logs, no dialogs). `BuildFromMenu` is the
interactive wrapper — it prompts to save the open scene and reports in a dialog,
so **do not** drive `BuildFromMenu` from the CLI.

`IsaacH1/Spawn Into Open Scene` and `MujocoBiped/Spawn Into Open Scene` open an
`EditorWindow` and have no headless equivalent yet; IsaacBox has one
(`SpawnIntoOpenSceneDefaults`). Add matching `…Defaults` entry points if those two
ever need to run unattended.

## Authoring a scene by hand, from the CLI

The bridge does full scene authoring, so a scene can be built and then tuned by
hand in the Editor rather than regenerated by a script every play:

    unity cmd open_scene       --path Assets/Scenes/SCN_RACE_FLAT.unity
    unity cmd create_gameobject --name Ground --primitive plane
    unity cmd add_component     --target Ground --type BoxCollider
    unity cmd get_scene_hierarchy
    unity cmd save_scene

Useful neighbours: `create_prefab`, `instantiate_prefab`, `save_prefab_contents`
(nested-prefab safe), `set_component_properties`, `find_gameobjects`,
`add_scene_to_build`, `capture_game_view`, `run_tests`, `build`.

## Watching a scene run

    unity cmd set_autotick --enable true      # keeps ticking while unfocused
    unity cmd clear_console
    unity cmd editor_play
    unity cmd get_performance_stats           # draw calls, memory, frame timing
    unity cmd console --level warning --tail 80
    unity cmd capture_game_view --width 720 --height 1544 --source screen \
        --save_path Assets/Temp/shot.png
    unity cmd editor_stop

`--source screen` captures the composited backbuffer including overlay UI;
`--source camera` misses Screen Space - Overlay canvases. Pass `--width`/`--height`
or the capture comes back 1280x720 regardless of the Game view aspect. `save_path`
must be **inside the project root** — an absolute temp path is rejected.

## Gotchas that cost time

- **Compile before trusting anything.** `unity cmd recompile`, then poll
  `unity cmd recompile_status` until `completed`, and check `failed`.
- **A clean console is not proof a scene ran.** `SCN_RACE_FLAT` boots to the menu
  and spawns nothing until a race starts; assert on `get_scene_hierarchy` or
  `get_performance_stats`, not on the absence of errors.
- **`eval` takes a bare statement body**, not a file with `using` directives at
  the top — those parse as using-*statements* and fail. Use fully-qualified type
  names instead.
- The Editor warns `Editor is not in automated mode` on every command. Harmless
  for these actions; it matters only if something opens a modal.
