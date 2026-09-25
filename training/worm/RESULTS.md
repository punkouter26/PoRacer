# Worm5 results: MuJoCo vs Isaac Lab (2026-09-24)

## Round 2: a third worm, Isaac3Worm (Isaac Lab 3)

Isaac Lab v3.0.0-beta2.patch1, kit-less, on the **Newton** backend with the **MuJoCo-Warp**
solver (`ISAAC/worm_tasks_v3`). Same body, contract and 30-minute budget.

| | MuJoCo worm | Isaac worm (Lab 2.3, PhysX) | **Isaac3Worm** (Lab 3, Newton) |
|---|---|---|---|
| Training throughput | 192k steps/s | 52k steps/s | ~125k steps/s |
| Own-simulator test (100 × 20 s) | **0.450 m/s** | 0.178 m/s | 0.326 m/s |
| Same physics (in MuJoCo 3.12) | **0.450 m/s** | 0.142 m/s | 0.322 m/s |
| Unity race (3-lane track, 5 races) | **0.451 m/s, won 5/5, 44.3 s** | 0.018 m/s (1.1 m) | 0.322 m/s (19.3 m at the 60 s limit) |

- **Isaac Lab 3 beats Isaac Lab 2.3** by 1.8× in its own simulator, 2.3× in identical
  physics and 18× in the Unity race. It also transfers into Unity with no loss, because
  it trains on MuJoCo-style physics and the game runs the MuJoCo plug-in.
- **MuJoCo is still best:** 1.4× Isaac3Worm, with 1.5× the training throughput.
- **Isaac3Worm's training collapsed** at 18.2–18.5 minutes: 0.32 m/s → 0 in about 30
  iterations, and it never recovered (the final checkpoint scores 0.0006 m/s). The race
  and tests use **model_1350**, the last checkpoint before the collapse (17.8 min, 133 M
  steps). The report file's `stepsTrained` (218 M) is the whole run, not this checkpoint.
- In the Unity race Isaac3Worm keeps its trained speed exactly (0.322 vs 0.326) and misses
  the 20 m finish by 0.7 m.

## Round 1: MuJoCo vs Isaac Lab 2.3

One worm body and one training contract ([WORM_SPEC.md](WORM_SPEC.md)). 30 minutes of
training each on an RTX 5070 Ti Laptop, one run after the other, Unity closed.

## Headline

**The MuJoCo-trained worm is about 3× faster, and it is the only one that races well
in Unity.** It won all 7 races.

## Training (30 min each, headless, 4096 envs)

| | MuJoCo (Warp + PPO) | Isaac Lab (RSL-RL) |
|---|---|---|
| Experience in 30 min | **346 M steps** (192k steps/s) | 94 M steps (52k steps/s) |
| Speed at the end of training | 0.44 m/s | 0.18 m/s |
| Physics divergences | 0 | 0 |

MuJoCo simulates this small worm about 3.7× faster, so in the same time its policy
gets 3.7× the practice.

## Same test, 100 episodes of 20 s, deterministic

| Brain | In its own simulator | In MuJoCo (identical physics for both) | In the Unity race |
|---|---|---|---|
| **MuJoCo-trained** | **0.450 ± 0.007 m/s** (9.0 m) | 0.450 m/s | **0.451 m/s**; finished 20 m in 44.2–44.5 s, 7 of 7 races |
| Isaac-trained | 0.178 ± 0.011 m/s (3.6 m) | 0.142 ± 0.056 m/s (2.8 m) | 0.049 m/s (2.95 m in 60 s), after the damping fix below |

- **Which tool trained better:** MuJoCo. In identical physics its brain is 3.2× faster.
  This comparison is per 30 minutes of wall clock; MuJoCo's head start in simulation
  speed is part of the result. A per-step comparison (Isaac trained to 346 M steps) has
  not been run.
- **Which worm is fastest:** the MuJoCo worm, in every setting.
- **Transfer into the game:** the MuJoCo worm races on the MuJoCo plug-in inside Unity
  and keeps its trained speed exactly (0.451 vs 0.450). The Isaac worm races on Unity's
  PhysX, which is not Isaac Sim's PhysX, and keeps only about a quarter of its speed.

## Realism (rule I), from the evaluations

| | MuJoCo-trained | Isaac-trained |
|---|---|---|
| Mean / peak roll rate of the middle segment | 0.49 / 5.5 rad/s | 0.31 / 4.9 rad/s |
| Peak joint speed | 6.7 rad/s | 5.3 rad/s |
| Mean segment height (lying flat = 0.045 m) | 0.054 m | 0.053 m |
| Some actuator at its 6 N·m limit | ~100 % of the time | ~100 % of the time |

Both crawl; neither corkscrews or hops. Both keep at least one muscle at full force
nearly all the time, so the effort penalty may be too weak to shape an efficient gait.

## Problems found and fixed on the way

1. **The first body could cheat.** With 12 N·m servos the MuJoCo smoke run learned a
   hopping corkscrew at 3 m/s, with the head spinning 31 rad/s. Fixed for both sides:
   servos at 6 N·m, joint damping 2.0, and a roll-rate penalty plus a belly-down term.
2. **Sideways drift.** The first Isaac smoke run sidewound 6 m sideways for 5 m forward.
   The lateral penalty went from −0.1 to −0.5 for both sides.
3. **The Isaac worm blew up in Unity.** In the first 5 races it diverged within the
   first metre every time. The cause was the explicit per-step joint damping
   (`ArticulationBody.jointForce`) on these light links. Switching
   `WormRaceSettings > Passive Damping` to *Drive Damping Per Radian* makes it stable.
4. **Scene builder bug (fixed):** `Editor_BuildWormRaceScene` left
   `WormRaceLifetimeScope._settings` (and the HUD's panel settings) empty in the saved scene.
   `EditorSceneManager.NewScene` unloaded the assets it had loaded into local variables
   before the scene was built. The builder now creates the scene first, then loads the
   assets, and reads the saved file back to check both links.
5. **The Isaac worm's Unity distance depends on the track layout.** Its crawl on
   Unity's PhysX is chaotic: 2.95 m on the two-lane track and 1.07 m on the three-lane
   track, from the same code. Only compare it within one layout.

## Files

- Brains: `export/worm_mujoco.onnx`, `export/worm_isaac.onnx` (copied to
  `Assets/Races/WormRace/Brains/`).
- Reports: `export/worm_mujoco_report.json`, `export/worm_isaac_report.json`,
  `export/worm_isaac_in_mujoco_report.json`.
- Race logs: `Logs/wormrace_*.json` (not in git).
- Scene: `Assets/Scenes/SCN_WORM_RACE.unity`; code in `Assets/Scripts/Races/WormRace/` and
  `Assets/Scripts/Editor/Races/WormRace/`.
