# AGENTS.md — Unity ML-Agents Biomechanical Humanoid & Creature Physics Simulation

> Specialization context and operational specification for AI agents developing biologically plausible terrestrial motion (1G gravity) with Unity ML-Agents.

---

## 1. PROJECT DOMAIN & BIOMECHANICAL OBJECTIVES

* **Domain:** Active ragdolls, bipedal humanoid locomotion, quadruped and multi-limbed creature motor control.
* **Core Objective:** Synthesize natural, energy-efficient, grounded locomotion mimicking real-world musculoskeletal physiology. Eliminate artificial artifacts: foot skating, micro-jitter, high-frequency twitching, rigid/stiff spines, and physics solver torque exploitation.
* **Tech Stack:**
  * **Engine & Physics:** Unity 6000.5+ (C#), Universal Render Pipeline (URP), PhysX / ArticulationBody & ConfigurableJoint dynamics.
  * **ML Framework:** Unity ML-Agents Toolkit (C# `com.unity.ml-agents` 4.1.0+), Python `mlagents` (1.1.0+), PyTorch (2.4.1+).
  * **Tooling & Orchestration:** Unity CLI Pipeline (`com.unity.pipeline`), TensorBoard telemetry.

---

## 2. C# SIMULATION & BIOMECHANICAL CONSTRAINTS

### A. Joint Control & Drive Mechanics
* **Drive Mechanism:** Drive motion exclusively through joint drive target rotations and velocities using PD controllers (Proportional $K_p$ / Position Spring and Derivative $K_d$ / Damper).
  * For `ConfigurableJoint`: configure `slerpDrive` / `angularXDrive` / `angularYZDrive` with `targetRotation` and `targetAngularVelocity`.
  * For `ArticulationBody`: set `xDrive` / `yDrive` / `zDrive` targets within anatomical limits.
* **Prohibition:** NEVER modify `Transform.position`, `Transform.rotation`, or tele-transport physics bodies directly during active simulation steps. Physics bodies must respond solely to joint drives and contact forces.

### B. Muscle & Torque Capacity Limits
* **Proportional Limits:** Enforce maximum torque ($T_{max}$ / `forceLimit`) and spring-damper values calibrated proportionally to body segment mass ($m_i$) and physiological muscle strength (Hill-type muscle envelopes).
* **Anti-Exploit:** Reject infinite-force or unnaturally stiff motors ($K_p \gg 10^4$) that enable superhuman leverage or induce solver explosion/twitching.

### C. Anatomical Limits & Action Normalization
* **Continuous Action Space:** Map continuous policy actions $a_t \in [-1, 1]$ directly into biological degrees of freedom (DoF):
  $$\theta_{target} = \theta_{rest} + a_t \cdot \Delta \theta_{range}$$
* **Joint Bounds:** Enforce hard hinge (e.g. knee flexion $[0^\circ, 140^\circ]$) and ball-and-socket constraints (e.g. hip abduction/adduction/flexion) matching biological range of motion without breaking joint anchors.

### D. Energy & Effort Penalties (Muscle Fatigue)
* **Torque Squared ($\tau^2$) Penalty:** Penalize instantaneous applied joint effort:
  $$R_{\tau} = -w_{\tau} \sum_{j} \left( \frac{\tau_j}{\tau_{max, j}} \right)^2$$
* **Action Jerk ($\Delta a$) Penalty:** Penalize high-frequency action changes to suppress motor twitching and enforce smooth swing/stance phases:
  $$R_{jerk} = -w_{jerk} \sum_{j} (a_{t, j} - a_{t-1, j})^2$$
* **Fatigue Mechanics:** Track accumulated joint strain to scale motor force limits during sustained maximal contraction.

### E. Ground Interaction & Anti-Skating
* **PhysicMaterial:** Apply realistic friction coefficients ($\mu_s \approx 0.8\text{--}1.0$, $\mu_d \approx 0.6\text{--}0.8$) without artificial bounce ($e = 0$).
* **Stance Contact Tracking:** Maintain limb ground sensors (`Sensor_LimbContact`). When a foot/limb is in ground contact (stance phase), penalize horizontal relative linear velocity:
  $$R_{skate} = -w_{skate} \cdot \mathbb{I}_{contact} \cdot \|\mathbf{v}_{foot, xz} - \mathbf{v}_{ground, xz}\|$$
* Eliminate artificial sliding/skating, demanding true friction-based thrust.

### F. Upright Posture & Center of Mass (CoM)
* **Center of Mass Tracking:** Continually compute composite system CoM and ground reaction force vectors.
* **Organic Balance:** Align pelvis/torso upright vector with world up ($[0, 1, 0]$) via reward shaping rather than freezing Rigidbody rotation axes.

---

## 3. REWARD ENGINEERING & MOTION SHAPING

### A. Dense vs Sparse Reward Hierarchy
* **Dense Incremental Rewards (`AddReward()` in `OnActionReceived()`):**
  * **Target Velocity Matching:** $R_{vel} = \exp\left( - \frac{\|\mathbf{v}_{root} - \mathbf{v}_{target}\|^2}{\sigma_v} \right)$
  * **Torso Upright Alignment:** $R_{upright} = (\mathbf{u}_{torso} \cdot \hat{\mathbf{y}}_{world})$
  * **Facing Direction:** $R_{heading} = (\mathbf{f}_{pelvis} \cdot \hat{\mathbf{d}}_{target})$
  * **Gait Cadence & Stance Regularity:** Symmetric limb phase progression.
  * **Effort Penalties:** Subtraction of $R_{\tau}$, $R_{jerk}$, and $R_{skate}$.
* **Sparse / Terminal Events (`SetReward()` & `EndEpisode()`):**
  * **Terminal Success:** Reaching checkpoint/target destination ($+1.0$).
  * **Terminal Failure:** Catastrophic head/torso ground impact, excessive spine inversion, or structural divergence ($-1.0$).

### B. Reference Trajectories & Imitation (GAIL / BC)
* **Pose Delta Metrics:** When cloning MoCap or kinematic reference animation clips:
  * Compute delta joint rotations in parent-local space: $\|\mathbf{q}_{sim} \ominus \mathbf{q}_{ref}\|$.
  * Compute end-effector (feet/hands) relative position errors relative to pelvis root.
  * Do NOT match absolute global coordinates to prevent brittleness on uneven terrain.

### C. Episode Reset Hygiene (`OnEpisodeBegin()`)
* Cleanly reset linear velocity ($\mathbf{v} = \mathbf{0}$) and angular velocity ($\boldsymbol{\omega} = \mathbf{0}$) for all rigid segments.
* Zero out internal PD target accumulators, previous action histories ($a_{t-1} = \mathbf{0}$), and ground contact states.
* Re-initialize joint drives and transform positions to valid rest poses with slight domain randomization (stochastic initial perturbation).

---

## 4. BUILD & TRAINING COMMANDS

### A. Python Environment
```powershell
# Verify Python virtual environment & dependencies
.\.venv\Scripts\python.exe -m pip list
```

### B. Training Launch Workflow
* **Rule:** Always launch TensorBoard **before** starting `mlagents-learn`.
```powershell
# 1. Start TensorBoard in background
Start-Process -FilePath ".\.venv\Scripts\tensorboard.exe" -ArgumentList "--logdir", "results", "--port", "6006"

# 2. Launch ML-Agents Headless Training (standardized 4 envs)
.\.venv\Scripts\mlagents-learn.exe Config/Humanoids01.yaml --run-id=humanoid_loco_01 --num-envs=4 --no-graphics

# Resume training from checkpoint
.\.venv\Scripts\mlagents-learn.exe Config/Humanoids01.yaml --run-id=humanoid_loco_01 --resume

# Force overwrite existing run
.\.venv\Scripts\mlagents-learn.exe Config/Humanoids01.yaml --run-id=humanoid_loco_01 --force
```

---

# OPERATING RULES — non-negotiable checklist

Read this section before doing anything in the project. These are the rules that
must hold on every session, on every commit, on every training run. The detailed
spec behind each one lives above in sections 1–4; this block is the short form.

## A. Source control
* **`master` only, no other branches.** Commit directly to `master` — no feature
  branches, no `cleanup/*`, no work branches, no matter how tidy the intent. The
  single exception is an explicit request for a branch; absent that, do not
  create one, and any branch that does get made must be merged back and deleted
  in the same session.
* Commit real, working increments. Compile clean, and for anything touching
  runtime, do a play-mode check as well.
* **Before any git sync (pull/push), commit all local changes first.** Never
  sync with a dirty tree — uncommitted work can be lost or force-conflicted.

## B. Project context — read first, every time
* At the start of every task, check the repo root for a `DOCS/` folder and
  **read it before touching code**. It is the project's own summary: roster
  state, scene layout, brain catalogue, agent motivation, worked example plan,
  the lot. Use it as the source of truth over memory whenever the two disagree.

## C. Training — TensorBoard is mandatory
* **Always start TensorBoard before `mlagents-learn`**, every run, no exceptions.
  Run it in the background on port `6006` with `--logdir results`:
  ```powershell
  Start-Process -FilePath ".\.venv\Scripts\tensorboard.exe" `
    -ArgumentList "--logdir","results","--port","6006"
  ```
  Then launch the trainer. A training run without TensorBoard going up first is
  a blind run — do not ship it.
* **Prune obsolete TensorBoard runs before starting a new one.** Dead
  experiments crowd the active curve and make the useful one unreadable. Use
  `scripts/Clean-TrainingArtifacts.ps1` (it knows both run-id conventions and
  protects the newest run of each prefix). Confirm port `6006` is free first —
  a crashed run leaks its TensorBoard listener, and the next launch then trains
  blind.
* **Training new agents means clearing old behaviours.** When a new set of
  agents starts training, remove old TensorBoard behaviours/runs that are no
  longer going to be used, so only live curves remain.

## D. Racer colours are a legend, not decoration
A viewer must be able to tell what is driving a racer at a glance. Colour
encodes the controller, never the creature.

| Racer kind | Colour |
|---|---|
| Heuristic / hand-coded bots | **RED**, always |
| The standard RL policy (the baseline, before per-creature variations) | **GREEN**, always, **untextured** |
| RL variations derived from that baseline | custom textures and meshes, supplied by the user |

* Red and green are reserved: do not spend them on a variation, a highlight or
  a team tint, and do not recolour a heuristic bot or the baseline RL racer to
  fit a palette.
* A new variation without a supplied texture **waits for one** rather than
  borrowing either colour.
* **"Standard RL" means the baseline policy only, not every RL racer.** A
  creature that came in with its own authored look is a variation and keeps
  that look.
* Every RL app carries a **heuristic coded bot, a reference bot, and zero to
  many custom bots**, the custom ones often with their own textures and skinned
  meshes.

---

## E. Simulators are for watching, not just for curves

* **Show the MuJoCo / Isaac Lab UI while training and after it**, so the
  creature's motion can be observed directly instead of inferred from reward
  curves. Use **Newton** to visualise training where it is the better viewer.
  Existing viewers: `training/mojucuboy/view_mojucuboy.py`,
  `training/fido/view_creature.py`.
* This deliberately trades throughput for observability. Headless is faster;
  faster is not the point when the question is *how does it move*. Reserve
  `--no-graphics` for runs whose only purpose is throughput, and say so.

## F. MuJoCo on Android

* Build the native library from <https://github.com/joanllobera/mujoco-bin/>.
  `Packages/org.mujoco` ships `mujoco.dll` only, so every MuJoCo call throws
  `DllNotFoundException` on a phone and the MuJoCo racers are simply absent from
  the roster there. An arm64 `libmujoco.so` from that repo, added to the
  plug-in, is what lifts that restriction.

## G. Scene objects belong in the scene

* **Create props, markers, spawn points and track furniture as real
  GameObjects/prefabs via MCP**, not by instantiating them from code at runtime.
  Anything a human might want to nudge should be draggable in the Scene view.
* Code-generated scenery is acceptable only where it is genuinely procedural and
  re-rolled per run. Everything else is authored once and committed, so that
  what is tuned in the editor is what ships.

## H. A fallen racer gets up on its own, or not at all

* **Never stand a knocked-down racer back up.** No marshal, no rescue flip, no
  righting torque, no snap to an upright pose mid-race. Enforced by
  `RacerView.MAX_RESCUES = 0`.
* The knockdown referee stays: on its back and going nowhere for
  `KNOCKDOWN_SECONDS` (12 s) is a DNF. That window is the racer's chance to
  recover under its own policy, **not** a countdown to being rescued.
* Snapping a rig to its trained stance is standing it up. That belongs at spawn
  only (`Agent_MojucuBoy.SnapToTrainedStance`), never during a race.
* **Why:** a policy that falls over and waits for help scores identically to one
  that genuinely recovers, and only one of them is racing. Recovery has to be
  trained — see the get-up curriculum notes in `rl_optimization_log.md`, where
  two full runs failed to learn it and that failure was worth knowing about.

## I. Physical plausibility is not optional

* Earth gravity (−9.81 m/s²), SI units, anatomically plausible joint ranges and
  motion, and **mass scaled to the creature's size**. A creature that moves in a
  way a real animal of that size and weight could not is a bug, however good its
  reward curve looks.
* **Joint speed and force must resemble real humans** when the trained agent
  is a human: no superhuman angular velocities, no torque beyond human muscle
  capability.

## J. Three training methods, compared side by side

* This app **compares three ways of training locomotion**. None of them is the
  default and none is being phased out:
  1. **Unity ML-Agents** — PPO (+ GAIL / BC demos) on Unity physics.
  2. **Isaac Lab** — RSL-RL PPO on Isaac Sim / PhysX GPU.
  3. **MuJoCo / Newton** — MuJoCo Warp + torch PPO.
* Unity remains the host app and the viewer for all three.
* A comparison is only fair if every method is judged by the **same walking
  standard, measured the same way** — see `DOCS/Plan-TrainingMethodComparison.md`.
  Never declare a method better on its own reward curve; reward scales differ
  between the three trainers.
* Every creature trained outside Unity needs its rig imported into that
  simulator first (see rule K).

## K. Skinned mesh first, then training

* **Before attempting to train a creature, ask the user for its skinned mesh.**
  The rig structure (bone hierarchy, joint anchors, mass distribution) is
  extracted from that model and imported into whichever simulator is training it.
* The user will supply additional creature/human models over time. For now,
  **focus on training the initial model with all the behaviours it needs**;
  do not start new creatures without an explicit request and a supplied mesh.

## L. Long training runs take over the machine

* When an RL run (any of the three methods) will take **30+ minutes**, first **save and close the
  Unity editor** to free the machine and avoid stalls. Tell the user when the
  run starts and explicitly tell them when they can reopen the editor
  (i.e., training is over).

## M. Collisions are complete

* **Every body part of every creature must collide accurately with everything
  else.** Creatures never pass through each other, and never pass through
  environment geometry. Missing collider = bug, no matter how stable the sim
  looks.

## N. Unity tooling — pick the best tool for the job

* Use whichever of these gives the best results for the task at hand:
  * **Unity CLI Pipeline** (`com.unity.pipeline`) for builds/automation.
  * **MCP plugins:** <https://github.com/AnkleBreaker-Studio/unity-mcp-plugin>,
    <https://github.com/CoplayDev/unity-mcp>,
    <https://github.com/IvanMurzak/Unity-MCP>.

## O. Unity stall prevention — set once, verify per session

* Set these in Unity to avoid editor stalling:
  * **Editor preferences:** Interaction Mode → **No Throttling**
    (`EditorPrefs` key `InteractionMode` = 1).
  * **Player settings:** **Run In Background** enabled
    (`PlayerSettings.runInBackground = true`).
* Do this via MCP when an editor session is connected; verify before long
  training runs.

## P. How agents write to the user

* Output plain, non-technical language the user can act on — decide follow-ups
  without decoding jargon.
* Any answer longer than 100 words ends with a **TL;DR of ~20 words**.

---

## 5. FILE INTEGRITY & EXCLUSIONS

* **Critical Exclusion List:** NEVER edit, format, corrupt, or manually regenerate:
  * Unity `.meta` files (GUID preservation is paramount).
  * Binary assets: `.onnx`, `.fbx`, `.glb`, `.prefab`, `.unity`, `.mat`.
  * Ephemeral directories: `Library/`, `Logs/`, `Temp/`, `Build/`, `Builds/`, `results/`, `.utmp/`.
* **Biomechanical Parameter Serialization:** Expose all physical tuning constants (PD gains $K_p, K_d$, torque limits $\tau_{max}$, penalty weights $w_{jerk}, w_{skate}, w_{\tau}$) via `[SerializeField]` or `ScriptableObject` assets for non-destructive runtime tuning and domain randomization.
