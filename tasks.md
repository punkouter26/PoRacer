# PoRacer: remaining tasks

Written 2026-09-24 when the plan was paused.

- **Goal:** every creature walks to one realistic standard, trained in MuJoCo and in
  Isaac Lab, and races in Unity.
- **Full plan:** [docs/Plan-TrainingMethodComparison.md](docs/Plan-TrainingMethodComparison.md).
- **Rules that apply throughout:** AGENTS.md. In particular:
  - J: train only in MuJoCo/Newton or Isaac Lab.
  - I: movement must be realistic.
  - L: close Unity for runs of 30 minutes or more.
  - C: start TensorBoard first.

**Direction decided so far:** train on **MuJoCo Warp** physics (plain MuJoCo Warp, or Isaac
Lab 3 with Newton and its MuJoCo Warp solver) and race on Unity's **MuJoCo plug-in**. The Worm
experiment ([training/worm/RESULTS.md](training/worm/RESULTS.md)) showed this keeps 100 % of
the trained speed in Unity. PhysX-trained brains kept about 10 %.

---

## 0. Things only you can do

- [ ] **Delete the 24 superseded worm-race files** (the agent's delete was blocked). They are
      compiled out behind `PORACER_WORMRACE_LEGACY` and marked `// SUPERSEDED`. Delete these
      in `Assets/Scripts/WormRace/`, each with its `.meta`:
      - `Models/`: all 6 files
      - `Systems/WormRaceSystem.cs`
      - `Physics/`: `MujocoWorldBuilder.cs`, `MujocoWormBuilder.cs`, `WormCapsuleMesh.cs`
      - `Views/`: `MujocoWormView.cs`, `WormRaceHudView.cs`, `WormRaceCameraView.cs`
      - `WormBodyState`, `WormFrames`, `WormObservation`, `WormPhysicsKind`, `WormPilot`,
        `WormPolicy`, `WormProbe`, `WormRaceReport`, `WormRaceRequest`, `WormRacerDefinition`,
        `WormReportWriter` (all `.cs`)

      Then run `unity cmd recompile` and check the console has no errors.
- [ ] **Stop the two leftover TensorBoard processes** holding port 6006:
      `Stop-Process -Id 22304, 32980` (the PIDs change after a reboot; find them with
      `Get-NetTCPConnection -LocalPort 6006`). The trainers refuse a busy port, so
      use `--tensorboard-port 6008` or similar until 6006 is free.
- [ ] Optional: delete `C:\Users\punko\.ai-game-dev` (the removed IvanMurzak plug-in's
      downloaded server and credentials).
- [ ] Optional: decide on `Assets/Temp/SCN_RACE_FLAT_unsaved_backup.unity`. It is an in-memory
      copy of the race scene that was flagged as changed with nothing visibly different;
      it's git-ignored.

---

## 1. Finish the Quadruped pilot (the current blocker)

**Where it stands.** Both trainers are built and train fast and stably:

| Trainer | Result |
|---|---|
| MuJoCo Warp (`training/quad/mujoco`) | 1.25–1.33 m/s, no falls |
| Isaac Lab 3 (`ISAAC/quad_tasks_v3`) | 1.39–1.44 m/s, no falls |

After 8 rounds of reward tuning **the gait still fails the realism stop rule**: feet slip,
landings are hard, and stances are short. The history of every round is in
[training/quad/QUAD_SPEC.md](training/quad/QUAD_SPEC.md).

**Paused mid-round-9.** The round-9 changes:
- **Soft floor instead of soft feet:** floor solref 0.03, priority 1, friction randomised per
  world. Soft feet made the legs overlap each other by up to 65 mm, breaking rule M.
- **Stronger trot reference:** contact_phase 2.0, gait_ref σ 0.2.
- **A new stance rule:** duty factor ≥ 0.35 and time-weighted stance ≥ 0.15 s.

The last reports from the two agents say how far the regeneration got. Check `git status` for
uncommitted files under `training/quad`, `training/creature` and `ISAAC/quad_tasks_v3`.

- [ ] Check `training/quad/quad.xml` and `quad_rig.json` have the soft **floor**, and that the
      body geoms are back to solref 0.01 / priority 0. Run
      `.venv-mjwarp\Scripts\python.exe training\quad\check_quad.py`. Expect:
      - stand at about 0.90 m;
      - a 2 cm drop peaking at about 1.3 body weights;
      - leg-leg overlap back to about 1–2 cm.
- [x] Isaac side: done 2026-09-24. Re-converted the soft-floor quad.xml (21:31:03); 79/79
      checks pass:
      - every foot–floor contact uses the floor's 0.03 and per-world friction 0.9·s
        (`randomize_floor_friction` event);
      - the spawn drop gives 1.334 body weights on both sides;
      - leg overlap is 11 mm (CPU MuJoCo 18 mm);
      - the reference replay matches CPU MuJoCo.
- [ ] **Round-9 smoke on the MuJoCo side** (4 min, 2048 envs):
      `train_quad_mujoco.py --minutes 4 --num-envs 2048 --tensorboard-port 6013`, then
      `eval_quad_mujoco.py` on the final and best checkpoints. The stop rule:
      | Measure | Target |
      |---|---|
      | Speed | ≥ 1.0 m/s |
      | Falls | ≤ 5 % |
      | Flight | ≤ 40 % |
      | Foot slip | ≤ 0.3 m/s |
      | Peak impact (after 0.5 s) | ≤ ~4 body weights |
      | Action rate (50 Hz equivalent) | ≤ 0.1 |
      | Duty factor | ≥ 0.35 |
      | Time-weighted stance | ≥ 0.15 s |
- [ ] If it still fails, the next options in order:
      1. anneal the speed reward in only after the gait forms;
      2. a landing-force penalty;
      3. accept a stylised gait for this bug creature.
- [ ] Once it passes: the same smoke on the Isaac side (`train_quad_v3.py`, port 6014).
- [ ] **30-minute runs, one after the other, with Unity closed** (rule L). TensorBoard
      first; keep checkpoints; if a run collapses, use the best checkpoint from before it
      (the adaptive-KL guard should prevent that).
- [ ] Export both (`quad_mujoco.onnx`, `quad_isaaclab3.onnx`, with obs **38** = the clock
      input), evaluate 100 × 20 s, and cross-check each brain in plain MuJoCo.
- [ ] **Unity:** the quad scene reads the rig and brains from `training/quad` when rebuilt.
      - The observation is now **38** floats: add the clock (sin φ, cos φ) at 1.5 Hz,
        advancing 0.075 rad per decision, to the quad's observation spec in the creature
        template.
      - The rig now carries per-geom/floor solref and priority. Make sure
        `CreatureRigParser` / `MujocoCreatureBuilder` and the MuJoCo floor build them.
      - Rebuild `SCN_QUAD_RACE` (`Editor_BuildQuadRaceScene`), then run the self-tests
        (`Editor_QuadRace.SelfTest`) and 5 races (`Editor_QuadRace.Start(5)`).
- [ ] Write `training/quad/RESULTS.md` (like the worm's) and commit.

---

## 2. Remaining creatures

Use the Quadruped as the template, including its final reward set. Each new body still needs
its own short smoke-and-check loop before the 30-minute runs.

- [ ] **Hexapod** and **Crab**:
      - MuJoCo bodies exist in `training/bugs/` (spawn heights fixed to 0.526 m).
      - Apply the same realism fixes: realistic force limits, kp/damping scaled to match,
        friction 0.9, soft floor.
      - The Crab's hips turn about the vertical axis, so it needs its own reference gait
        (sideways scuttle).
      - Train with both tools, then race.
- [ ] **MojucuBoy:** already MuJoCo. Put him on the creature template and retrain with the
      shared rules if his current brain doesn't pass the exam. It currently runs 143–207 % of
      his target speed and falls on courses.
- [ ] **Isaac H1** and **IsaacBox**: they need MuJoCo bodies first.
      - H1: from its URDF, or MuJoCo Menagerie's H1 model.
      - IsaacBox: from `ISAAC/boy_rig`; `build_boy_rig.py` works from the same GLB.
      - Then train in MuJoCo Warp and in Isaac Lab 3 (Newton). They are humanoids, so rule I
        requires human-like joint speeds and forces.
- [ ] Rule K: any *new* creature needs a skinned mesh from you first. Primitive test
      creatures are allowed only with your explicit approval.

---

## 3. Getting up after a fall (plan step 5)

No creature has ever got up after a fall. Rule H forbids rescues, so this must be trained.

- [ ] Add a stage 2 to each creature's training:
      - start 30 % of episodes lying down;
      - reward standing back up;
      - stop ending the episode on a fall (keep the 12 s knockdown window in the game).
- [ ] MojucuBoy's history applies: see `rl_optimization_log.md`. Two full runs failed to learn
      get-up, and the M10 finding explains why.
- [ ] Re-run the walking exam; W7 (get-up) must pass.

---

## 4. Walking exam (`Assets/Scripts/Editor/Editor_WalkExam.cs`)

- [ ] Add **W5 foot skating, W6 smoothness and W8 torque/speed limits**. These need per-agent
      data; the creature template's MuJoCo view can supply it.
- [ ] Point the exam at the **MuJoCo-plug-in racers** (the creature template) as they replace
      the old PhysX/ML-Agents brains.
- [ ] Run the full exam with **nothing else on the GPU**. Both earlier runs stopped at about
      26 of 54 trials with no error while other GPU jobs were running.
- [ ] Leg length for the speed target comes from the catalogue `spawnHeight`. Keep it equal
      to each creature's real standing root height.

---

## 5. Android MuJoCo library (rule F), now a priority

Every MuJoCo-trained racer is missing on phones until this is done.

- [ ] Build or obtain an arm64 `libmujoco.so` from <https://github.com/joanllobera/mujoco-bin/>.
      Match the plug-in's MuJoCo version (3.12).
- [ ] Add it to `Packages/org.mujoco`'s plug-ins for Android arm64. Build the APK
      (`PoRacer/Build Android APK`) and test on the Pixel 9 Pro: the MuJoCo racers must
      appear and race.

---

## 6. Put the new brains into the main game

- [ ] Register the MuJoCo-trained creatures in `CreatureCatalog`, racing on the MuJoCo plug-in
      through the creature template, and retire the ML-Agents bug brains once their
      replacements pass the exam.
- [ ] **Rule D (colours):** heuristic bots red, the baseline RL racer green. Variants need
      **textures from you**; the plain blue, orange and purple test colours are only for
      the test scenes.
- [ ] Update the DOCS: `onnx_summary.md`, the creatures dashboard, and the scene layout.

---

## 7. Optional experiments

- [ ] **Worm, equal practice:** train the Isaac Lab 2.3 worm to 346 M steps (about 80 more
      minutes) to compare learning per step rather than per hour.
- [ ] **Worm, natural gait:** remove the sideways-drift penalty and widen the lanes, to see
      whether the worms discover sidewinding and whether it's faster.
- [ ] **Repeats:** train each tool 2–3 times with different seeds, so the comparisons have
      error bars.
- [ ] Isaac Lab 3 worm: investigate or retrain with the adaptive-KL guard, which fixes its
      18-minute collapse.

---

## Housekeeping notes

- `Assets/UI/PoRacerFont SDF.asset` gains glyphs whenever a race plays. Revert it
  (`git checkout -- "Assets/UI/PoRacerFont SDF.asset"`) unless you want the new glyphs.
- The `ISAAC/install.ps1` exit code is 1 only because `pip check` reports two known version
  clashes; installs are fine. The rig rebuild step is opt-in (`-RebuildRig`).
- Isaac Lab 3 lives in `ISAAC/isaaclab3` (Python 3.12, kit-less). Its installer has a USD
  packaging bug: uninstall `usd-exchange` and reinstall `usd-core==25.11`.
- IvanMurzak's Unity-MCP was removed; `.mcp.json` is git-ignored because that plug-in wrote a
  cloud token into it.

## Handy commands

| What | Command |
|---|---|
| Unity bridge status | `unity status` / `unity cmd editor_status` |
| Close / open Unity | `unity close "C:\Users\punko\Downloads\PoRacer"` / `unity open "C:\Users\punko\Downloads\PoRacer"` |
| MuJoCo worm training | `.venv-mjwarp\Scripts\python.exe training\worm\mujoco\train_worm_mujoco.py --minutes 30 --tensorboard-port 6008` |
| Isaac Lab 3 training | `ISAAC\isaaclab3\.venv\Scripts\python.exe ISAAC\scripts\train_quad_v3.py --minutes 30 --num_envs 4096 --tensorboard_port 6008` |
| Worm race | `unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Start(5);"` |
| Walking exam | `unity cmd eval --code "return PoRacer.EditorTools.Editor_WalkExam.Start(\"\", \"0,1,2\", 3);"` |
