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

## D. Racer colours are a legend, not decoration
A viewer must be able to tell what is driving a racer at a glance. Colour
encodes the controller, never the creature.

| Racer kind | Colour |
|---|---|
| Heuristic / hand-coded bots | **RED**, always |
| The standard RL policy (the baseline, before per-creature variations) | **GREEN**, always |
| RL variations derived from that baseline | custom textures, supplied by the user |

* Red and green are reserved: do not spend them on a variation, a highlight or
  a team tint, and do not recolour a heuristic bot or the baseline RL racer to
  fit a palette.
* A new variation without a supplied texture **waits for one** rather than
  borrowing either colour.
* **"Standard RL" means the baseline policy only, not every RL racer.** A
  creature that came in with its own authored look is a variation and keeps
  that look.

---

## 5. FILE INTEGRITY & EXCLUSIONS

* **Critical Exclusion List:** NEVER edit, format, corrupt, or manually regenerate:
  * Unity `.meta` files (GUID preservation is paramount).
  * Binary assets: `.onnx`, `.fbx`, `.glb`, `.prefab`, `.unity`, `.mat`.
  * Ephemeral directories: `Library/`, `Logs/`, `Temp/`, `Build/`, `Builds/`, `results/`, `.utmp/`.
* **Biomechanical Parameter Serialization:** Expose all physical tuning constants (PD gains $K_p, K_d$, torque limits $\tau_{max}$, penalty weights $w_{jerk}, w_{skate}, w_{\tau}$) via `[SerializeField]` or `ScriptableObject` assets for non-destructive runtime tuning and domain randomization.
