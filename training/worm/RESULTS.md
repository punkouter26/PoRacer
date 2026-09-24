# Worm5 results: MuJoCo vs Isaac Lab (2026-09-24)

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
4. **Scene builder bug:** `Editor_BuildWormRaceScene` left `WormRaceLifetimeScope._settings`
   empty in the saved scene. It was set by hand; the builder itself still needs fixing.

## Files

- Brains: `export/worm_mujoco.onnx`, `export/worm_isaac.onnx` (copied to
  `Assets/WormRace/Brains/`).
- Reports: `export/worm_mujoco_report.json`, `export/worm_isaac_report.json`,
  `export/worm_isaac_in_mujoco_report.json`.
- Race logs: `Logs/wormrace_*.json` (not in git).
- Scene: `Assets/Scenes/SCN_WORM_RACE.unity`; code in `Assets/Scripts/WormRace/` and
  `Assets/Scripts/Editor/WormRace/`.
