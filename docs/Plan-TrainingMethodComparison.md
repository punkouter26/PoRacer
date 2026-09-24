# Plan — Training Methods, One Walking Standard

> Written 2026-09-24. Goal: every creature in the race walks equally well, and the
> app can show which training tool got it there best.
>
> **Scope change (2026-09-24, AGENTS rule J):** training now happens **only in
> MuJoCo/Newton (C) and Isaac Lab (B)**. Method A (ML-Agents) is no longer trained.
> Its three brains keep racing until the bugs are retrained in B or C (their MuJoCo
> bodies already exist, see Phase 2). Wherever this plan says "three methods" or
> "18 brains", read "two tools" and "12 brains". The first head-to-head is the Worm5
> experiment (`training/worm/WORM_SPEC.md`).

## Status and direction (updated 2026-09-24, after the Worm5 experiments)

**What the worms proved** ([training/worm/RESULTS.md](../training/worm/RESULTS.md)):
training on **MuJoCo Warp** physics and racing on Unity's **MuJoCo plug-in** keeps 100 %
of the trained speed. Training on PhysX and racing on Unity's PhysX kept about 10 %, and
the physics blew up until damping was reworked. Two tools reach MuJoCo Warp:

| Tool | Worm speed in Unity | Notes |
|---|---|---|
| **Plain MuJoCo Warp** (`training/worm/mujoco`) | **0.451 m/s** | 192k steps/s; best final worm; no collapse |
| **Isaac Lab 3 + Newton, MuJoCo Warp solver** (`ISAAC/worm_tasks_v3`) | 0.322 m/s | ~125k steps/s; fastest learner in the first 10 min; beta; collapsed at 18 min |
| Isaac Lab 2.3 + PhysX | 0.018 m/s | retired for new training |

**New direction:** every creature trains on MuJoCo Warp, with **plain MuJoCo Warp** as
the default and **Isaac Lab 3 (Newton, MuJoCo Warp solver)** as the Isaac Lab entry in
comparisons. Every creature **races on the MuJoCo plug-in** in Unity. Consequences:

- **Phase 2 gets simpler.** The body check is MuJoCo against MuJoCo (the same engine),
  not MuJoCo against Unity PhysX. The PhysX damping mismatch in `training/bugs` stops
  mattering.
- **Android becomes a priority (rule F).** MuJoCo creatures are missing on phones until
  `libmujoco.so` (arm64) from joanllobera/mujoco-bin is added to `Packages/org.mujoco`.
- **One reusable pipeline instead of one-off code:** rig → MJCF → trainer → ONNX →
  Unity MuJoCo builder → self-tests → exam. The worm code is the template.

| Phase | State |
|---|---|
| 0 · Training setups | ✅ MuJoCo Warp, Isaac Lab 2.3, Isaac Lab 3 (Newton) all run here |
| 1 · Walking exam | ✅ Built and run twice; W5/W6/W8 still to add. No creature meets the standard |
| 2 · Bodies in MuJoCo | 🟡 Worm ✅, bugs ✅ (MJCF), MojucuBoy ✅ (already MuJoCo). **H1 and IsaacBox still need MJCF** |
| 3 · Shared recipe | ✅ Written; used for the worm. Needs a legged version (Froude speed, uprightness, get-up stage) |
| 4 · Pilot | 🟡 Worm done. **Next: Quadruped**, trained with plain MuJoCo Warp and Isaac Lab 3 |
| 5 · All creatures | ⬜ |
| 6 · Everyone to the standard, including get-up | ⬜ No creature has ever got up after a fall |
| 7 · In-app head-to-head | 🟡 Worm race scene (N lanes, selectable physics) is the template |

**Next steps, in order:**
1. Generalise the worm pipeline into a **creature template**: an MJCF-driven trainer,
   exporter and evaluator, and a Unity MuJoCo builder that works for any rig.
2. **Quadruped pilot:** legged recipe, trained with plain MuJoCo Warp and Isaac Lab 3,
   raced in Unity on the MuJoCo plug-in, then the walking exam.
3. **Android MuJoCo library** (rule F), so MuJoCo racers work on phones.
4. Roll out to Hexapod, Crab and MojucuBoy; convert H1 and IsaacBox to MJCF.
5. Add get-up training, and re-run the exam until each creature passes.

## The three methods

| | Method | Physics it trains on | Runs where in the game |
|---|---|---|---|
| **A** | Unity ML-Agents (PPO + GAIL/BC demos) | Unity PhysX, `ArticulationBody` | Same physics — no transfer gap |
| **B** | Isaac Lab (RSL-RL PPO) | Isaac Sim PhysX (GPU) | Unity PhysX — small transfer gap |
| **C** | MuJoCo / Newton (MuJoCo Warp + torch PPO) | MuJoCo | MuJoCo plug-in inside Unity — no gap on desktop, **absent on Android** until rule F is done |

## Where we start (2026-09-24)

| Creature | A · ML-Agents | B · Isaac Lab | C · MuJoCo |
|---|:-:|:-:|:-:|
| Quadruped | ✅ racing | — | — |
| Hexapod | ✅ racing | — | — |
| Crab | ✅ racing | — | — |
| Isaac H1 | — | ✅ racing (imported) | — |
| IsaacBox | — | ✅ racing (imported) | — |
| MojucuBoy | — | — | ✅ racing |

We have 6 of 18 possible creature × method brains, each trained with a different
reward, budget and yardstick. As things stand, nobody can say which is the better walker.

### What the smoke race showed (2026-09-24, `Logs/smoke_20260924_1228.json`)

One race per map with all 6 current brains. It ran clean: 0 errors.

| Creature (method) | Flat, first 21 s | Acrobat course | Apartment course |
|---|---|---|---|
| Quadruped (A) | 9.6 m, cut off | 44.5 m, out | 18.1 m, out |
| Hexapod (A) | 7.1 m, cut off | 43.5 m, out | 3.7 m, out |
| Crab (A) | **1.2 m**, cut off | 3.3 m, out | 25.5 m, **still going at full time (1st)** |
| Isaac H1 (B) | 19.7 m, **finished 3rd** | 47.4 m, out (1st) | 9.1 m, out |
| IsaacBox (B) | 19.4 m, **finished 2nd** | 18.9 m, out | 53.9 m, out (2nd) |
| MojucuBoy (C) | 20.0 m, **finished 1st** | 18.2 m, out | 24.2 m, out (3rd) |

**What it means for this plan:**

1. **The spread is huge.** On flat ground the three bugs cover 1–10 m while the
   three bipeds cover ~20 m in the same time. The Crab barely moves on flat
   (0.06 m/s), yet it outlasts everyone in the Apartment. Today's brains are
   nowhere near "the same ability", so the Phase 1 baseline will show large gaps.
2. **Nobody finishes a course.** On Acrobat all six were knocked out, on Apartment
   five of six. **Staying up and getting up (W2, W3, W7) is the biggest gap, not
   speed.** Even MojucuBoy's shipped brain, which is documented as able to get up,
   went out on both courses. → The get-up lesson (stage 2) is required for every
   brain, and the exam must include **course terrain** (ramps, curves), not only a
   flat straight.
3. **One race proves nothing.** The winner changes on every map (MojucuBoy, H1, Crab).
   → The exam runs each brain **10 times per terrain** and reports the average;
   ELO (W9) needs many races before it counts.
4. **The report doesn't say *why* a racer went out** (fell and stayed down, got
   stuck, or left the track). → `Editor_WalkExam` must record the reason for every
   knockout. That is the most useful detail for fixing a brain.
5. **The flat race ended at 21 s of 120 s** because three racers finished and the
   podium rule knocked out the rest. That hides how good the slower brains are.
   → The exam and the Method Shootout **turn the podium cutoff off** and run to
   full time or full distance.
6. **Frame rate dropped to 2–5 fps at moments** during every race. → Exam speeds
   are measured on the **physics clock, not frames**, so a stutter can't change a
   score.

---

## 1 · The Walking Standard (the one yardstick)

"Same ability" has to survive very different body sizes. A crab and a human cannot
share a speed target, but they can share a *gait*: every creature is asked to walk at
the same **Froude number** (the speed-for-size measure biologists use to compare
walking animals).

**Target speed** `v = √(Fr · g · L)`, with `Fr = 0.25` (a relaxed walk) and `L` =
hip height. For MojucuBoy (L ≈ 0.9 m) this gives **1.49 m/s**, which is the
1.5 m/s command he already trains on.

| # | Test | Pass mark | Comes from |
|---|---|---|---|
| W1 | **Speed**: holds the target speed | within ±10 % | K2 |
| W2 | **Uptime**: upright and at stance height | > 90 % of the time | K1 |
| W3 | **Falls**: time spent fallen | < 10 % | K3 |
| W4 | **Steering**: heading error to the next waypoint | < 15° | K4 |
| W5 | **No skating**: feet slide while planted | < 0.1 m/s mean | AGENTS §2E |
| W6 | **Smoothness**: action jerk | ≤ the median of the 18 brains | K6/K7 |
| W7 | **Get-up**: recovers unaided from a knockdown | ≥ 50 % within 12 s | K10, rule H |
| W8 | **Plausible**: no joint over its torque or speed limit | 0 violations | rule I |
| W9 | **Race result**: ELO from real races | within 100 of the best brain for that creature | `Systems_Elo` |

A brain **meets the standard** when it passes W1–W8 over 10 back-to-back tests.
W9 is the tiebreak.

**Measured in Unity, not in the trainer.** Each trainer reports its own numbers
while it runs, which is fine for watching progress. The verdict, though, comes from
one test run inside Unity, the same for all 18 brains. That way method B's transfer
gap counts against it, as it should: it is part of what that method costs.

---

## 2 · Phases

### Phase 0: Setup and decisions (½ day)
- [x] **Checked the machine (2026-09-24). None of the three training setups is
      installed right now.** All three worked here once, and all three were removed:
      | Method | What's needed | State |
      |---|---|---|
      | A · ML-Agents | `.venv` on **Python 3.10** with the pins in commit `058ea25` (torch 2.4.1, numpy 1.23.5, protobuf 3.20.3, onnx 1.15.0, setuptools<81, mlagents 1.1.0) | `.venv` gone, and no Python 3.10 on the machine (only 3.11.9) |
      | B · Isaac Lab | `ISAAC/install.ps1`: clones Isaac Lab into `ISAAC/isaaclab` and installs Isaac Sim ≥ 5.0 (**~10 GB**, NVIDIA licence to accept). IsaacBox was trained here this way | `ISAAC/isaaclab` gone; the installer and the Boy task code are still there |
      | C · MuJoCo | `mujoco`, `mujoco_warp`, `warp`, torch with CUDA 12.8 (the RTX 5070 Ti is Blackwell / sm_120 and needs cu128) | Packages missing |
      Hardware is fine: RTX 5070 Ti Laptop 12 GB, 1.2 TB free.
- [x] **A · ML-Agents rebuilt (2026-09-24):** `.venv` on Python 3.10.11 with the
      documented pins. `mlagents-learn` starts. torch is `2.4.1+cpu` on purpose:
      2.4.1 has no Blackwell (sm_120) kernels, and ML-Agents trains its small MLP
      on the CPU anyway.
- [x] **C · MuJoCo rebuilt (2026-09-24):** `.venv-mjwarp` on Python 3.11.9 with
      torch `2.11.0+cu128`, `mujoco 3.12.0`, `mujoco-warp 3.12.0` (now on PyPI,
      and the same version as `Packages/org.mujoco`, so what trains is what races),
      `warp-lang 1.17.0`, tensorboard, onnx and onnxruntime. The MojucuBoy env builds
      and steps on the GPU (3,268 steps/s at 256 worlds while Unity was also busy).
- [x] **B · Isaac Lab installed (2026-09-24)** with `ISAAC/install.ps1`: torch
      2.7.0+cu128 running real kernels on sm_120. A 64-env, 2-iteration IsaacBox training
      run started TensorBoard first, trained, saved checkpoints and shut down cleanly.
      The installer exits 1 only because `pip check` reports two known version clashes
      (fastapi/starlette upstream, ipython/psutil); training is unaffected. The installer's rig-rebuild step is now opt-in
      (`-RebuildRig`): it used to overwrite `Assets/unity_export/IsaacBox/isaacbox_rig.json`
      with a provisional joint order, which is the file the IsaacBox racer reads.
- [ ] Decide how a viewer tells the three methods apart in a race (see *Open
      questions*).
- [x] AGENTS.md rule J rewritten: three methods, one standard.

**Two fairness problems found in the race code while building the exam:**
- **The Isaac robots are knocked out after 1 second tipped over.** `Agent_IsaacH1`
  and `Agent_IsaacBox` report `Failed` after `_fallenGraceSeconds = 1` s past 60°,
  and `RacerView` retires any racer whose agent reports `Failed`. Every other
  creature gets the 12 s knockdown window of rule H to get back up. So the Isaac
  robots can never show a get-up, and method B is penalised by the referee rather
  than by its training. **Fixed 2026-09-24:** their `Failed` now means only a
  non-finite pose (as for `Agent_MojucuBoy`), so falls go to the same 12 s
  knockdown referee as everyone else. The baseline exam started before this fix,
  so its Isaac knockouts show up as `AgentFailed`.
- **Random quirks** (TURBO +12 % drive, HEAVY +12 % mass...) are rolled per racer.
  Now switchable (`RaceConfigModel.QuirksEnabled`); the exam turns them off.

### Phase 1: Build the walk exam (1–2 days)

**Slice 1 done (2026-09-24):** `Editor_WalkExam` runs every creature **alone** on the
three race maps, N trials each, quirks off, speed on the physics clock. It writes
`Logs/walkexam_<stamp>.json` and a `.md` table. It measures W1 speed (Froude 0.25
target), W2 uptime, W3 time fallen, W7 get-up (from real falls), completion, and
the **knockout reason**, which the referee now records for every DNF
(`KnockoutReason`: KnockedDown, Stalled, LeftTrack, Diverged, AgentFailed,
PodiumCutoff). **W4 steering added** (after the baseline run started): the
distance-weighted angle between where the body travels and where the track runs,
in 0.5 s windows. Travel, not facing, because the Crab is built to scuttle sideways.
W5, W6 and W8 still need per-agent data and are marked "not measured yet".

    unity cmd eval --code "return PoRacer.EditorTools.Editor_WalkExam.Start(\"\", \"0,1,2\", 3);"
    unity cmd eval --code "return PoRacer.EditorTools.Editor_WalkExam.Status();"

Still to do for Phase 1:
- [ ] `SCN_WALK_EXAM`, a new scene: 30 m flat straight, then a 90° waypoint turn,
      then a scripted shove (knockdown) at a fixed moment, then a **course section
      (ramp and banked curve) copied from Acrobat and Apartment**. It is authored in
      the scene (rule G), not spawned from code.
- [ ] `Editor_WalkExam`, which runs a brain through 10 tests per section and writes
      a report card (W1–W8) to `Logs/walkexam_<creature>_<method>.json`. It is
      modelled on `Editor_SmokeRace`, so it can be driven from the CLI.
      **Requirements:** it records the reason for every knockout, times on the
      physics clock, and has no podium cutoff.
- [ ] Port the same W1–W8 maths into each trainer's own eval script
      (`gate4_eval.py` for C, an Isaac Lab play script for B, an ML-Agents stats
      side-channel for A).
- [ ] **Run the exam on the 6 current brains.** That gives the baseline and shows
      how far apart they really are.

### Baseline exam result (2026-09-24, `Logs/walkexam_20260924_1404.md`)

The run stopped at 24 of 54 trials (play mode ended, most likely an editor recompile),
after one complete round on every map plus a second round on Flat. It predates the
Isaac fall fix and W4. **Not one brain passes W1 (speed) on Flat**, and none finishes
a course.

| Creature (method) | Flat speed vs target | Courses | How they go out on courses |
|---|---|---|---|
| Quadruped (A) | 20 % | 7–24 %, upright 16–94 % | fell and stayed down; left the track |
| Hexapod (A) | 17 % | 11–13 % | stalled once on Flat |
| Crab (A) | **3 %**, stalls | 3–11 % | stalled (barely moves; travels 75° off the track) |
| Isaac H1 (B) | 63 % | 71–73 % | old 1 s fall rule; left the track |
| IsaacBox (B) | 75 % | 85–93 % (**passes W1 on Apartment**) | old 1 s fall rule |
| MojucuBoy (C) | **143 % (too fast)** | 154–207 %, upright 74–88 % | left the track |

- **Getting up:** nobody got up after any fall (0 recoveries).
- **Uprightness on Flat:** fine for everyone (100 %).
- **The spread:** from 3 % to 207 % of target. The standard is far from met by any
  method today.

A check run after the fixes (IsaacBox and Crab on Acrobat, 1 trial) confirmed W4 works
(IsaacBox 7°, Crab 75°), and that IsaacBox now lies down and is judged by the normal
referee (Stalled) instead of being retired after 1 s.

**Speed targets for Hexapod and Crab are about 19 % too high.** The exam takes leg
length from the catalogue `spawnHeight` (0.75 m), but the Phase 2 models show both
stand with the root at 0.526 m. They spawn 0.27 m in the air and drop every race.
Fix the catalogue values before the next exam.

### Exam re-run under fair rules (2026-09-24, `Logs/walkexam_20260924_1504.md`)

This run had the 12 s get-up window for the Isaac robots, W4 steering, and the
corrected Hexapod/Crab spawn heights. It stopped at 26 of 54 trials: play mode ended
with no compile and no error, both times while other heavy GPU work was running.
Re-run the rest with nothing else on the GPU.

| Creature | Flat speed | Flat steering | Courses | Out by |
|---|---|---|---|---|
| Quadruped | 23 % | 42° ✗ | 12–24 %, fallen 51 % on Acrobat | stalled |
| Hexapod | 33 % | 30° ✗ | 11–15 % | stalled |
| Crab | 5 % | 60° ✗ | 4–13 % | stalled |
| Isaac H1 | 63 % | 2° ✓ | 41–79 %, steering ✓ | stalled; left track |
| IsaacBox | 75 % | 2° ✓ | 31–91 % (**W1 ✓ on Apartment**), steering ✓ | stalled; left track |
| MojucuBoy | 143 % (too fast) | 7° ✓ | 154–207 %, steering 21–34° ✗ | left track |

- **Getting up:** still 0 recoveries from any fall, including the Isaac robots now
  that they have the full 12 s. The old 1 s rule was not what stopped them.
- **Steering:** the bugs don't head down the track (30–79° off); the bipeds do on Flat.
- **Nothing meets the standard.** The closest is IsaacBox on Apartment, which fails
  only on get-up.

### Phase 2: One body, three simulators (3–5 days)

**Bugs → MuJoCo done (2026-09-24):** `training/bugs/` has `Quad_v01.xml`, `Hexapod_v01.xml`
and `Crab_v01.xml`, generated from the prefabs by `prefab_to_mjcf.py`, plus a
`<name>_rig.json` each and a README of the conventions. The axis mapping is the same as
MujocoBiped's CONTRACT.md. Stiffness is N·m/rad, so kp = Unity stiffness. All three
pass `check_bug_mjcf.py`:
- body and joint counts match the prefabs, and masses match exactly;
- each stands for 5 s with no drift;
- each loads in mujoco_warp and matches the CPU result.

Open points from the conversion:
- Damping is on the joint rather than in the drive, because the literal Unity setup
  is unstable at 5 ms in MuJoCo. Joints are about 1–2° off a fine-step reference, but
  a full-torque joint moves slower than in Unity. The twin check decides whether that
  is acceptable.
- Ground friction is Unity's default 0.6, below the 0.8–1.0 in AGENTS §2E. That gap
  is in Unity itself.
- The Quadruped is 90 kg at 0.9 m tall with 900 N·m joints. Worth a physical-plausibility
  look (rule I).
Each creature must be *the same animal* in all three sims, or the comparison is
meaningless.
- [ ] Export every rig to all three formats: Unity prefab ↔ MJCF (MuJoCo) ↔
      URDF/USD (Isaac). Matching items: masses, joint limits, PD gains, torque
      limits, friction (µ 0.8 / 0.6, no bounce) and full collision shapes (rule M).
- [ ] **Twin check** per rig: drop it, and play the same 5 s of fixed joint commands
      in all three sims. Pass mark: body height and forward travel agree within 10 %.
      The MujocoBiped port shows the kind of gap this catches (it only ever reached 22 %
      speed parity).
- [ ] Fix the biped fatigue scale bug noted in `onnx_summary.md` before it can skew
      any results.

### Phase 3: One shared training recipe (1 day)

**Where the three stand today:** see [TrainingRecipeComparison.md](TrainingRecipeComparison.md).
They differ in the task itself (reach a point / chase a target / follow a velocity
command / hold a heading), the speed target (none / 1.0 / 0–1 / 1.5 one-sided), what
counts as a fall, policy rate (10 Hz vs 50 Hz), episode length (15 s vs 20 s), how
rewards scale with time, and PPO settings. Only C trains get-up, and only A uses demos.

**Shared recipe (approved 2026-09-24):**

| Item | Shared value |
|---|---|
| Task | follow waypoints along the track; the policy sees the direction to the next waypoint **and** its target speed |
| Target speed | the creature's Froude 0.25 speed, **two-sided** exp kernel (overshoot costs too) |
| Policy rate | 50 Hz everywhere (ML-Agents: DecisionPeriod 4) |
| Episode | 20 s; timeouts bootstrapped in all three |
| Rewards | same terms, weight × value × dt: speed tracking, heading, uprightness, torque² (applied torque, normalised), action rate, foot slip. No alive bonus |
| Falls | one definition: body up < 0.5 **or** torso below 50 % of stance height. Stage 1 ends the episode on a fall; stage 2 starts 30 % of episodes sprawled and rewards standing |
| Actions | rest pose + action × 0.5 × joint half-range; PD gains from the shared rig file |
| Randomisation | gains ±20 %, mass ±10 %, friction 0.6–1.2, a 0.5 m/s push every 10–15 s |
| PPO | 3×128 hidden units, γ 0.99 per 0.02 s, λ 0.95, 5 epochs, LR 3e-4 constant, entropy 0.005, initial σ 0.5 |
| Known unavoidable gap | ML-Agents fixes its activation (Swish) and has no adaptive-KL schedule; B and C use ELU. Recorded, not hidden |

**Fixes needed before any A run counts:** `TRAINING_MAX_STEP` must be 4000 for 20 s at
0.005 s (it is 3000 = 15 s), and `AcrobatLoco01.yaml` must be rewritten without its
dangling aliases.
Only the trainer changes between methods. Everything else is fixed:
- **Same inputs**: body state, target direction and target speed.
- **Same outputs**: `target = rest + action × range`, driven by the joint PD drives.
- **Same reward**: the AGENTS §3A terms, with identical weights. The uprightness sign
  bug fix (M9) is included.
- **Same lesson plan**: stage 1 ends an attempt on a fall (the E11 recipe, which is
  what worked). Stage 2 teaches getting up from a knockdown. Stage 3 adds the harder
  tracks.
- **Same budget**: **3 hours of wall-clock time on this laptop** per brain, one run
  at a time, 2 seeds each.
- **Demos are allowed only where every method can use them.** GAIL/BC is currently
  ML-Agents-only, so method A runs both **with** and **without** demos. The version
  without demos is the fair comparison; the one with them shows what demos add.

### Phase 4: Pilot, 2 creatures × 3 methods (about 2 days of machine time, mostly unattended)
- [ ] **Quadruped** (simplest, 8 joints) and **MojucuBoy** (hardest, 21 joints).
- [ ] 6 brains (plus 2 extra for A without demos), each exam-tested.
- [ ] **Checkpoint with you:** review the report cards and watch a shootout race,
      then decide whether the recipe is fair before spending the next ~75 hours.

### Phase 5: The full grid (about 6–8 days, mostly unattended)
- [ ] The remaining 4 creatures × 3 methods = 12 brains.
- [ ] Unity is closed during every 30 min+ run (rule L). You'll be told when it's
      safe to reopen it.
- [ ] TensorBoard goes up first for every run, and old runs are pruned (rule C).

### Phase 6: Bring every creature up to the standard
- [ ] Any brain that failed the exam gets extra training with **its own method**,
      in 1-hour top-ups, until it passes or hits 3 extra hours.
- [ ] The extra time is recorded, because **"hours to reach the standard" is the
      headline comparison number.**
- [ ] The game ships the best brain per creature (by W9 among those that pass W1–W8).

### Phase 7: Show it in the app
- [ ] **Method Shootout** race mode: one creature, three brains (A, B, C), same track,
      with the podium cutoff off, so all three race to full distance.
- [ ] A results screen with the report card side by side, plus hours to standard and
      ELO.
- [ ] A dashboard page in `DOCS/`: an 18-cell grid, pass/fail, and cost per method.

---

## 3 · What the comparison will answer

| Question | Measured by |
|---|---|
| Which method reaches the standard fastest? | Hours to pass W1–W8 |
| Which walks best in the actual game? | Exam scores and ELO, measured in Unity |
| How much is lost moving a brain into Unity? | Trainer score vs Unity exam score |
| Do example walks (demos) help? | Method A with vs without demos |
| Which works on phones? | A and B yes. C only after the Android MuJoCo library (rule F) |

## 4 · Open questions for you

1. **Isaac Lab**: is it installed somewhere, or were H1 and IsaacBox trained on
   another machine? This decides whether method B can run here.
2. **Telling methods apart in a race**: rule D reserves red for hand-coded bots and
   green for the baseline RL racer. Method variants count as "variations", which need
   textures from you. Options: (a) you supply 3 method textures, (b) a small A/B/C
   badge floats over each racer with the colours untouched, (c) both.
3. **Scope**: all 6 creatures, or pilot on 2 and decide after Phase 4? (The plan
   assumes the pilot first.)
4. **Budget**: is 3 hours per brain × 2 seeds acceptable? The full grid is roughly
   **110–130 hours of machine time**.

## 5 · Rough timeline

| Phase | Your time | Machine time |
|---|---|---|
| 0–3 Setup, exam, rig twins, recipe | a few check-ins | ~1 week of build work |
| 4 Pilot | 1 review session | ~48 h |
| 5 Full grid | none | ~75 h |
| 6 Top-ups | none | 0–36 h |
| 7 In-app shootout | 1 review | ~2 days build |
