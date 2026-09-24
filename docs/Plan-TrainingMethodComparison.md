# Plan — Three Training Methods, One Walking Standard

> Written 2026-09-24. Goal: every creature in the race walks equally well, and the
> app can show which of the three training methods got it there best.

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
- [ ] Reinstall all three environments (needs your go-ahead for the Isaac download
      and licence).
- [ ] Decide how a viewer tells the three methods apart in a race (see *Open
      questions*).
- [x] AGENTS.md rule J rewritten: three methods, one standard.

**Two fairness problems found in the race code while building the exam:**
- **The Isaac robots are knocked out after 1 second tipped over.** `Agent_IsaacH1`
  and `Agent_IsaacBox` report `Failed` after `_fallenGraceSeconds = 1` s past 60°,
  and `RacerView` retires any racer whose agent reports `Failed`. Every other
  creature gets the 12 s knockdown window of rule H to get back up. So the Isaac
  robots can never show a get-up, and method B is penalised by the referee rather
  than by its training. **Needs a decision:** give them the same 12 s window.
- **Random quirks** (TURBO +12 % drive, HEAVY +12 % mass...) are rolled per racer.
  Now switchable (`RaceConfigModel.QuirksEnabled`); the exam turns them off.

### Phase 1: Build the walk exam (1–2 days)

**Slice 1 done (2026-09-24):** `Editor_WalkExam` runs every creature **alone** on the
three race maps, N trials each, quirks off, speed on the physics clock. It writes
`Logs/walkexam_<stamp>.json` and a `.md` table. It measures W1 speed (Froude 0.25
target), W2 uptime, W3 time fallen, W7 get-up (from real falls), completion, and
the **knockout reason**, which the referee now records for every DNF
(`KnockoutReason`: KnockedDown, Stalled, LeftTrack, Diverged, AgentFailed,
PodiumCutoff). W4, W5, W6 and W8 still need per-agent data and are marked "not
measured yet".

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

### Phase 2: One body, three simulators (3–5 days)
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
