# Feature Brief: PoRacer — Self-Running Creature Race

> **Status header added 2026-09-03 by the docs regeneration.** This is the *original* feature brief
> from the 2026-08-14 interview, preserved as the record of intent. It has **not** been rewritten.
> Verified against the current tree:
>
> | Then | Now |
> |---|---|
> | Unity 6000.5.6f1, Windows standalone | **Unity 6000.5.8f1**, and the shipping target is **Android** — signed AAB to Play internal testing |
> | Δt locked 0.02 s | **Δt locked 0.005 s (200 Hz)**, 12 solver iterations |
> | MVS packages "NOT yet installed — step one of implementation" | VContainer, MessagePipe and UniTask are all installed and load-bearing |
> | P1: worm reaches a goal on flat ground | **Done.** P2 (self-running race, ELO, HUD, camera director) and P3 (more morphologies) are done too |
> | P3: "second morphology (spider), then more bodies" | **9 racers on the grid** across three inference paths; 6 more brains trained but uncatalogued |
> | P4: combat | **Not started.** Still the next unbuilt phase |
> | P5: terrain variety | **Done** — 5 player-selectable maps with mud, boost pads, gusts and gates |
> | P6: GLB scene import | **Not started**, but glTFast is installed and already loads rigged creature models |
> | "up to 8 racers" | The grid is 10 wide and **stacks in layers past 30 racers** |
>
> **Acceptance criteria status:** 3, 7, 8 and 9 hold as written. Criterion 4 now asserts
> `fixedDeltaTime == 0.005`, not 0.02. Criterion 2 (a goal 20 m away within 60 s, 8 of 10 episodes) is
> **not met by four creatures** — Worm, Spider, Centipede and Crab do not finish the 22 m Flat race,
> which is why `Config/BrokenLoco01.yaml` exists. Criterion 1's evidence is gone: `results/` is no
> longer in the working tree.
>
> Current state lives in [`creatures_dashboard.html`](creatures_dashboard.html),
> [`architecture_pipeline.html`](architecture_pipeline.html),
> [`scenes_layout.html`](scenes_layout.html) and [`onnx_summary.md`](onnx_summary.md).

---

> Produced by /unity-interview on 2026-08-14. Confirmed answers: worm first slice, combat later,
> portrait 9:16, watch+camera+stats viewer, up to 8 racers, ELO + race history persisted.

## Scope

**Does:**
- Physics creatures race from a start point to a goal in 3D scenes.
- Every creature is torque-driven (joints + motors). ML-Agents brains learn locomotion. No keyframe animation.
- The app is self-running: trained `.onnx` brains race each other in a loop; the viewer watches.
- Viewer can switch camera targets and see a live leaderboard (ELO + race results).
- Creature roster grows over time: worm → spider/quadruped → half-biped → biped → more.
- Later phases: combat (projectiles, limb swings that break balance), varied terrains, runtime GLB scene import with start/end/checkpoints.

**Does NOT (v1):**
- No combat in the first training phases (combat is a later curriculum phase).
- No GLB import in v1 (later; will need `com.unity.cloud.gltfast` + a checkpoint path tool).
- No player-controlled creatures. No multiplayer/networking. No mobile builds yet (Windows standalone, portrait window).
- No UGUI/IMGUI — UI Toolkit only, per project rules.

**Trigger:** app launch → auto-loads race loop. Training is launched from CLI (`mlagents-learn` + headless env builds).

**Output:** on-screen race + leaderboard; `elo.json` + `race-history.json` in `Application.persistentDataPath`; trained `.onnx` per creature version under `Assets/Agents/<Name>_v<NN>/`.

## Roadmap (phases)

| Phase | Deliverable |
|-------|-------------|
| P1 (this slice) | Worm learns to reach a goal on flat ground. Training pipeline proven end-to-end. |
| P2 | Self-running race app: 8 worms, finish line, ELO, leaderboard HUD, camera director. |
| P3 | Second morphology (spider), then more bodies over time. |
| P4 | Combat phase: projectiles / limb strikes, knockdown detection, self-play + ELO. |
| P5 | Terrain variety (hills, gaps, obstacles) as training curriculum + race tracks. |
| P6 | GLB scene import with start/end/checkpoints; creatures race imported scenes. |

## Technical Requirements

- **Unity:** 6000.5.6f1 | **Pipeline:** URP | **Platform:** Windows standalone, portrait 9:16, 60 FPS (`vSyncCount=0`, `targetFrameRate=60`)
- **ML:** com.unity.ml-agents 4.1.0 (C#) / mlagents 1.1.0 (Python, `.venv`); headless env builds to `Builds/WormEnv/`, 4–8 envs, explicit `--base-port`
- **Physics:** Δt locked 0.02 s, gravity −9.81, SI units, actions in `FixedUpdate` only, actions normalized [−1,1] → joint limits, fatigue reads applied torque (later phases)
- **Architecture:** MVS + VContainer + MessagePipe + UniTask per `.claude/rules/` — **these packages are NOT yet installed; adding them is step one of implementation**
- **Persistence:** JSON (`elo.json`, `race-history.json`) in `Application.persistentDataPath`, written by a `Systems_Persistence` class
- **Naming:** `Assets/Agents/Worm_v01/Worm_v01.onnx`; scenes `SCN_RACE_*`, `SCN_TRAIN_WORM`; config `WormLoco01.yaml` ↔ run-id `worm_loco01`; script prefixes `Agent_`, `Sensor_`, `Reward_`, `Systems_`

## Edge Cases

| Case | Expected behavior |
|------|-------------------|
| Creature leaves track / falls out of bounds | Training: end episode, negative reward. Race: mark DNF, remove from ranking for that race. |
| Creature makes no progress for 20 s | Training: end episode. Race: DNF. |
| `.onnx` missing or observation shape mismatch | Skip that racer, log warning, race continues with the rest. Never crash. |
| All racers DNF | Race ends by timeout with no winner; history records it; next race starts. |
| Physics NaN / explosion | Actions clamped; joint drives limited; episode/race resets the creature cleanly (fatigue cleared before motors restored). |
| Corrupt save JSON | Back up bad file, recreate defaults, keep running. |
| Window not 9:16 (resized) | Panel UI scales on width; layout stays usable. |
| Trainer port collision | Consecutive `--base-port` allocation per env; documented in run scripts. |

## Integration Points (all new systems — fresh project)

| System | Owns / Direction | Messages (MessagePipe) |
|--------|------------------|------------------------|
| `Systems_Race` | owns `RaceModel`; write | publishes `RaceStartedMessage`, `CheckpointPassedMessage`, `RaceFinishedMessage` |
| `Systems_Spawn` | builds racers from creature catalog (ScriptableObjects) | subscribes `RaceFinishedMessage` |
| `Systems_Elo` | owns `EloModel`; write | subscribes `RaceFinishedMessage` |
| `Systems_Persistence` | read/write JSON | subscribes `RaceFinishedMessage` |
| `Systems_CameraDirector` | owns `CameraModel` | subscribes `RaceStartedMessage`, `CheckpointPassedMessage` |
| `Agent_Worm` (ML-Agents `Agent`) | reads joints, applies torques | none (training reward only) |
| Views: `RaceHudView`, `InputView` | UI Toolkit HUD + camera input | observe Models via `ReactiveProperty` |

## Assembly Placement

- `Assets/Scripts/PoRacer.Runtime.asmdef` — Models, Systems, Views, Agents (references ML-Agents, VContainer, MessagePipe, UniTask, InputSystem)
- `Assets/Scripts/Editor/PoRacer.Editor.asmdef` — editor tooling
- `Assets/Tests/PoRacer.Tests.asmdef` — EditMode tests for Systems/Models (input-agnostic, no mocks needed)

## Acceptance Criteria (P1 slice + app skeleton)

1. [ ] `SCN_TRAIN_WORM` contains ≥8 parallel training areas; `mlagents-learn WormLoco01.yaml --run-id=worm_loco01` runs against a headless build with `--no-graphics` and mean reward rises over the run.
2. [ ] `Assets/Agents/Worm_v01/Worm_v01.onnx` in inference mode reaches a goal 20 m away on flat ground within 60 s in ≥8 of 10 episodes.
3. [ ] The worm moves only through its own joint motor torques — no `AddForce` on the root body (negative test: zeroing all joint drives leaves the worm unable to advance).
4. [ ] Physics settings verify at runtime: `fixedDeltaTime == 0.02`, gravity −9.81, agent actions applied only in `FixedUpdate`.
5. [ ] Race scene runs 8 worm instances at ≥60 FPS in a portrait 9:16 window (verified with performance stats capture).
6. [ ] Finish trigger produces `RaceFinishedMessage`; ELO updates; `elo.json` and `race-history.json` exist and survive an app restart.
7. [ ] A racer with a missing `.onnx` is skipped with a logged warning and the race still completes.
8. [ ] HUD (UI Toolkit) shows leaderboard and `Application.version` anchored top-left; zero UGUI/IMGUI in the project.
9. [ ] Corrupting `elo.json` by hand then launching does not crash; file is regenerated.

## Estimated Complexity

**Complex.** Physics-learned locomotion is research-adjacent; the race app itself is moderate. Risk concentrates in P1 training quality (reward shaping) — which is exactly why the worm slice comes first.

## Recommended Approach

1. Install VContainer, MessagePipe, UniTask (UPM/openupm) — rules require them and nothing is wired yet.
2. Build P1 with `/unity-workflow`: worm prefab (segments + configurable joints) via MCP, `Agent_Worm` + `Reward_WormLoco`, training scene, YAML config, headless build, train, bake `.onnx`.
3. Then P2 race skeleton with `unity-prototyper` + `unity-scene-builder` agents.
