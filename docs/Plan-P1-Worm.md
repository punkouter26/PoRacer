# P1 Implementation Plan — Worm Slice + App Skeleton (v2, post-critic)

> **Status header added 2026-09-03 by the docs regeneration.** This is a *historical* design
> document from 2026-08-14, preserved because decisions D1–D14 are still load-bearing and the
> reasoning behind them exists nowhere else. It has **not** been rewritten to today's numbers.
> Verified against the current tree, here is what changed:
>
> | Then | Now | Where the current truth lives |
> |---|---|---|
> | Δt 0.02 s, DecisionPeriod 5 | **Δt 0.005 s (200 Hz), DecisionPeriod 20** — same 0.1 s decision interval | [`creatures_dashboard.html`](creatures_dashboard.html) Tier 2 |
> | D3: "no per-step time penalty" | A **−0.002 time penalty exists**, plus jerk and skate penalties, all step-scaled | [`creatures_dashboard.html`](creatures_dashboard.html) Tier 3 reward tree |
> | D3: obs = joint pos/vel + root up/height + goal | obs = **3N + 19** — adds per-limb contact, root angular velocity, stamina, 4 terrain probes | [`onnx_summary.md`](onnx_summary.md) §3.3 |
> | One creature (Worm), 8 racers | **9 racers in the catalog, 15 ONNX brains on disk**, three inference paths | [`creatures_dashboard.html`](creatures_dashboard.html) roster |
> | `Agent_Worm.cs` | Generalised to **`Agent_Creature.cs`**, morphology-agnostic via `ICreatureAgent` | `Assets/Scripts/Agents/` |
> | 12 position / 4 velocity solver iterations | **12 solver iterations**, locked | [`scenes_layout.html`](scenes_layout.html) |
> | Fatigue "later phases" | **Shipped** — reads applied `jointForce`, drains stamina, scales drives | [`creatures_dashboard.html`](creatures_dashboard.html) Tier 3 |
> | VContainer/MessagePipe/UniTask "not yet installed" | All three installed and load-bearing | [`architecture_pipeline.html`](architecture_pipeline.html) Tier 3 |
>
> **Still exactly true:** D1 (ArticulationBody, reset via `TeleportRoot`, never `transform.position`),
> D5 (no hand-rolled ReactiveProperty), D6 (the naming amendment and the ML-Agents MVS carve-out),
> D7 (pairwise ELO, K = 32), D10 (`MaxStep = 0` at race time, `DecisionStep` staggering),
> D13 (scene-anchor Views exception), D14 (one frame between despawn and respawn, `Safe()` on every
> observation), and every line under **Verified non-issues** — including that C# ML-Agents 4.1.0
> against Python mlagents 1.1.0 is the *correct* pairing.

---

> Amends docs/FeatureBrief-CreatureRace.md after unity-critic review on 2026-08-14.
> Complexity: complex. Executed by the orchestrator with MCP for scenes/settings; verify loop at the end.

## Decisions locked by the critic review

| # | Decision |
|---|----------|
| D1 | **ArticulationBody, not Rigidbody+HingeJoint.** Stable joint chains, no stretching, and exposes `jointForce`/drive torque — required later by the fatigue rule. Reset via `TeleportRoot` + `SetJointPositions` + zeroed velocities, never `transform.position`. |
| D2 | **Inchworm gait (pitch-axis joints).** Lateral undulation cannot move on Unity's isotropic friction. Joints bend in the vertical plane. Validated with a scripted sinusoidal gait BEFORE any PPO run. |
| D3 | **Reward:** potential-based progress (delta distance × 1.0), goal terminal +10, out-of-bounds terminal −1 via `EndEpisode()`, timeout/no-progress via `EpisodeInterrupted()` (bootstraps value fn). No per-step time penalty. Reward code accounts for `OnActionReceived` firing every FixedUpdate (`TakeActionsBetweenDecisions=true`, `DecisionPeriod=5`). |
| D4 | **Training budget:** 10M steps ceiling, checked at 500k/1M/2M via TensorBoard; goal distance randomized 3–20 m per episode (natural curriculum). YAML: `normalize: true`, `time_horizon: 1000`, hidden 256×2, `learning_rate_schedule: linear`, buffer/batch sized for 64 concurrent agents (8 areas × 8 envs). |
| D5 | **No hand-rolled ReactiveProperty.** Models expose plain C# `event Action`; HUD refreshes via UI Toolkit `schedule.Execute().Every(250)` reading the Model. Rules amendment recorded in CLAUDE.md. |
| D6 | **Naming amendment (recorded in CLAUDE.md):** `Agent_`/`Sensor_`/`Reward_`/`Systems_` prefixes apply to those four folders; Views/Models follow architecture.md (`*View`, `*Model`). ML-Agents `Agent` subclasses get an explicit MVS carve-out (they hold obs/action/reward logic; `Reward_WormLoco` stays a plain C# class, unit-testable; no VContainer in the training scene). |
| D7 | **ELO = pairwise expansion:** 28 pairs for 8 racers, K=32/(N−1); DNF loses to finishers, DNF vs DNF = draw; all-DNF race = no rating change. |
| D8 | **Persistence:** JsonUtility wrapper classes (no Dictionary), write `*.tmp` then `File.Replace`, race history capped at last 200 races. |
| D9 | **Cinemachine 3.x API** (`CinemachineCamera`, `Target.TrackingTarget`, priority switching). |
| D10 | **Race prefab:** `MaxStep=0`, `InferenceDevice` = Burst/CPU, `DecisionStep = spawnIndex % 5` (staggers inference), training-area reset script exists only in the training scene. |
| D11 | **Cut:** `CheckpointPassedMessage` (P6), CompositeDisposable, per-creature SO types (one catalog SO with `List<CreatureEntry>`), action-map switching (single `Camera` map: Next/Prev/Free). |
| D12 | **Worm self-collision:** worm segments on a dedicated `Creature` layer; non-adjacent self-collision allowed but guarded: `maxAngularVelocity` capped, NaN check in `OnActionReceived` → end episode/DNF. |
| D13 | **Scene-anchor Views exception (post-review):** `Systems_Spawn` and `Systems_CameraDirector` may depend on `RaceTrackView`/`CameraRigView` — logic-free scene anchors exposing Inspector-wired refs. Accepted deviation from "Systems never reference Views"; anchors must stay data-only. |
| D14 | **Respawn discipline (post-review):** always one frame between Despawn() and respawn (deferred Destroy leaves colliders live → overlap → physics NaN explosion). All observations pass a NaN/Infinity `Safe()` filter; a failed racer is deactivated so it cannot break other agents' Academy step. Large spawns yield every 25 instances. |

## Settings pass (before any code)

- Physics: Δt 0.02, gravity −9.81, **solver TGS (type 1), 12 position / 4 velocity iterations, enhanced determinism on** — locked now, asserted at runtime (extends acceptance criterion 4).
- Player: `runInBackground = 1` (CRITICAL for --num-envs), `fullscreenMode = Windowed`, `defaultIsNativeResolution = 0`, 540×960 default, `resizableWindow = 1`.
- Display config (`vSyncCount=0`, `targetFrameRate=60`) applied ONLY when `!Academy.Instance.IsCommunicatorOn`.
- Delete `Assets/TutorialInfo/` and `Assets/Scenes/SampleScene.unity`; build list managed per-purpose (see build script).
- Editor script `Editor_BuildWormEnv` builds `Builds/WormEnv/` from an explicit scene array (`SCN_TRAIN_WORM` only), **Mono backend** for fast rebuilds.

## Files

```
Assets/Scripts/PoRacer.Runtime.asmdef        (refs: Unity.ML-Agents, VContainer, MessagePipe,
                                              MessagePipe.VContainer, UniTask, Unity.InputSystem,
                                              Unity.Cinemachine)
Assets/Scripts/Editor/PoRacer.Editor.asmdef  + Editor_BuildWormEnv.cs
Assets/Tests/PoRacer.Tests.asmdef            (Editor platform, nunit precompiled ref,
                                              UNITY_INCLUDE_TESTS constraint)

Assets/Scripts/Models/RaceModel.cs            plain C#, event-based
Assets/Scripts/Models/EloModel.cs
Assets/Scripts/Models/Messages.cs             RaceStartedMessage, RacerFinishedMessage,
                                              RacerDnfMessage, RaceFinishedMessage (readonly structs)
Assets/Scripts/Systems/Systems_Race.cs        RegisterEntryPoint (ITickable) — finish order, DNF timeout
Assets/Scripts/Systems/Systems_Spawn.cs       catalog SO → spawn 8; skip+warn on missing/mismatched onnx
Assets/Scripts/Systems/Systems_Elo.cs         pairwise ELO (D7)
Assets/Scripts/Systems/Systems_Persistence.cs elo.json + race-history.json (D8)
Assets/Scripts/Systems/Systems_CameraDirector.cs  Cinemachine 3.x priority switching
Assets/Scripts/Systems/Systems_TrainingArea.cs    training-scene-only reset/goal (also used per-area)
Assets/Scripts/Agents/Agent_Worm.cs           ArticulationBody obs/actions; root-local goal dir,
                                              normalized joint pos/vel, root up-vector + height
Assets/Scripts/Rewards/Reward_WormLoco.cs     plain C# (D3), EditMode-tested
Assets/Scripts/Views/RaceHudView.cs           UI Toolkit hierarchy in C#; leaderboard + version stamp
                                              (non-pickable, top-left)
Assets/Scripts/Views/InputView.cs             PlayerControls (generated) — Camera map only
Assets/Scripts/GameLifetimeScope.cs           race scene only
Assets/Input/PlayerControls.inputactions
Assets/Settings/CreatureCatalog.asset         (SO: id, name, prefab, onnx ModelAsset)
Config/WormLoco01.yaml
scripts/train_worm.ps1                        preflight asserts runInBackground; builds env;
                                              .venv mlagents-learn --env --no-graphics --base-port 5005
                                              --num-envs 8; teardown order trainer→envs→TensorBoard
```

## Scenes & assets (via MCP)

- Worm prefab: 6 capsule segments, ArticulationBody chain (pitch revolute joints), shared URP Lit
  material with GPU instancing; per-racer tint via MaterialPropertyBlock. `PhysicsMaterial` (Unity 6 name)
  with explicit frictionCombine.
- `SCN_TRAIN_WORM`: 8 self-contained areas (area-local positions only), ground, goal marker per area.
- `SCN_RACE_FLAT`: 8 lanes, start/finish trigger, GameLifetimeScope, HUD UIDocument, CinemachineBrain +
  overview/chase cameras, ground static-batched.
- UI Toolkit needs `PanelSettings` + default runtime theme (.tss): attempt via MCP asset creation; if not
  possible, STOP and give the user click-by-click editor steps (per performance.md Developer Action Items).
  Scale mode: scale-with-screen-size, match width.

## Rendering strategy (performance.md compliance)

8 worms × 6 capsules = 48 instanced-capsule renderers, 1 shared material (+MPB tint), ground static,
UI on a single UIDocument panel. Expected draw calls: <20. No transparency except HUD.

## Order of execution

1. Settings pass + deletions (MCP) → 2. asmdefs + all C# (file tools) → compile check →
3. worm prefab + gait validation script (MCP scene, scripted sine gait must advance ≥0.5 m/10 s BEFORE PPO) →
4. SCN_TRAIN_WORM + build + smoke `mlagents-learn` 20k steps (handshake + reward wiring proof) →
5. full training run (background, monitored) → 6. race scene + systems wiring while training runs →
7. bake Worm_v01.onnx into Assets/Agents/Worm_v01/ + catalog → 8. tests + verify loop + deslop →
9. acceptance criteria sweep (FPS capture, restart persistence, corrupt-file test, missing-onnx test).

## Verified non-issues (do not "fix")

- C# 4.1.0 vs Python 1.1.0 is CORRECT pairing — both speak comm API 1.5.0 (verified in source).
- Do not enable "Disable Domain Reload" (breaks Academy + MessagePipe statics).
- `com.unity.ai.inference` comes in transitively; never pin it in manifest.json.
