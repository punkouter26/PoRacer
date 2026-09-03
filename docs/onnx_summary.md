# PoRacer — ONNX & Rig Model Inventory

> **Regenerated 2026-09-03 by `/unity-docs`, read directly off disk.** Every file size, tensor
> shape, parameter count, layer width and PD gain in this document was extracted from the
> actual artifacts (`onnx.load` over `Assets/**/*.onnx`, YAML scrape over `Assets/Prefabs/*.prefab`,
> `isaacbox_rig.json` / `IsaacH1_rig.json` / `MujocoBiped_rig.json`, `Assets/Creature/policy.json`).
> Nothing here is estimated unless the row says so.
>
> Companion documents: [`creatures_dashboard.html`](creatures_dashboard.html) ·
> [`architecture_pipeline.html`](architecture_pipeline.html) · [`scenes_layout.html`](scenes_layout.html)
>
> Three progressive tiers: **Tier 1 (Executive, <30 s)** → **Tier 2 (Overview)** → **Tier 3 (Complete Specification)**.

---

## Tier 1 · Executive summary

| | |
|---|---|
| **ONNX files on disk** | **15** — 12 under `Assets/Agents/<Name>_v01/`, 3 under `Assets/unity_export/<Port>/` |
| **Total ONNX weight** | **12,262,121 B (11.69 MB)**, 3,047,616 parameters |
| **Brains that actually race** | **9** — the 8 wired into `CreatureCatalog.asset` with a `ModelAsset`, plus Fido, whose policy is a JSON MLP inside his prefab. IsaacBox's is `model_2999.pt` as of commit `afa3eeb` |
| **Brains on disk but not racing** | **6** — Blob, Kangaroo, Grandma, Grandpa, Matt, Nick (trained, exported, never added to the catalog) + MujocoBiped (ported, verified, not registered) |
| **Shipped race payload** | **2,353,477 B (2.24 MB)** of ONNX + 851,712 B of `policy.json` = **~3.06 MB** for the whole starting grid |
| **Three inference paths, not one** | ML-Agents `Agent` + Inference Engine (9 creatures) · raw Inference Engine, no ML-Agents (IsaacBox, IsaacH1) · hand-rolled MLP over MuJoCo (Fido) |
| **Promotion rule** | ELO from actual races (`Systems_Elo`, K = 32, base 1200) — **never** mean reward. A potential-based reward scales with episode length, so a longer-surviving brain outscores a faster one. |
| **Biggest correction vs. the previous edition of this doc** | The old file claimed 9 brains at ~2.79 MB with DOF counts up to 32 and `forceLimit = 100 N·m`. All three were wrong: there are 15 files, the real DOF ceiling is 21 (IsaacBox), and `forceLimit` runs 500–1350 N·m. |

---

## Tier 2 · Basic overview

### Matrix A — Models

`obs` and `act` are read from the ONNX graph itself; the `Prefab obs/act` column is the
`BehaviorParameters` value scraped from the prefab. **They agree on every row** — that is the
check `PoRacer ▸ Sync Agent Observation Sizes` exists to keep true.

#### A1 · ML-Agents creatures — `Assets/Agents/<Name>_v01/<Name>_v01.onnx`

| Creature | Size (B) | Input tensor | Action tensor | Params | Hidden | Prefab obs/act | Behavior | In catalog |
|---|---:|---|---|---:|---|:-:|---|:-:|
| **Worm** | 309,738 | `obs_0 (batch, 34)` | `continuous_actions (batch, 5)` | 76,113 | 256 × 2 | 34 / 5 ✓ | `Worm` | ✅ `spawnHeight 0.15` |
| **Blob** | 309,738 | `obs_0 (batch, 34)` | `continuous_actions (batch, 5)` | 76,113 | 256 × 2 | 34 / 5 ✓ | `Blob` | ❌ |
| **Kangaroo** | 313,866 | `obs_0 (batch, 37)` | `continuous_actions (batch, 6)` | 77,145 | 256 × 2 | 37 / 6 ✓ | `Kangaroo` | ❌ |
| **Quad** | 322,122 | `obs_0 (batch, 43)` | `continuous_actions (batch, 8)` | 79,209 | 256 × 2 | 43 / 8 ✓ | `Quad` | ✅ `0.85` |
| **Spider** | 322,122 | `obs_0 (batch, 43)` | `continuous_actions (batch, 8)` | 79,209 | 256 × 2 | 43 / 8 ✓ | `Spider` | ✅ `0.5` |
| **Crab** | 338,634 | `obs_0 (batch, 55)` | `continuous_actions (batch, 12)` | 83,337 | 256 × 2 | 55 / 12 ✓ | `Crab` | ✅ `0.75` |
| **Hexapod** | 338,634 | `obs_0 (batch, 55)` | `continuous_actions (batch, 12)` | 83,337 | 256 × 2 | 55 / 12 ✓ | `Hexapod` | ✅ `0.75` |
| **Centipede** | 359,276 | `obs_0 (batch, 70)` | `continuous_actions (batch, 17)` | 88,497 | 256 × 2 | 70 / 17 ✓ | `Centipede` | ✅ `0.25` |
| **Grandma** | 2,239,066 | `obs_0 (batch, 52)` | `continuous_actions (batch, 11)` | 558,209 | 512 × 3 | 52 / 11 ✓ | `Grandma` | ❌ |
| **Grandpa** | 2,239,066 | `obs_0 (batch, 52)` | `continuous_actions (batch, 11)` | 558,209 | 512 × 3 | 52 / 11 ✓ | `Grandpa` | ❌ |
| **Matt** | 2,239,066 | `obs_0 (batch, 52)` | `continuous_actions (batch, 11)` | 558,209 | 512 × 3 | 52 / 11 ✓ | `Matt` | ❌ |
| **Nick** | 2,239,066 | `obs_0 (batch, 52)` | `continuous_actions (batch, 11)` | 558,209 | 512 × 3 | 52 / 11 ✓ | `Nick` | ❌ |

> **Byte-identical groups.** Worm/Blob (309,738 B, 76,113 params), Quad/Spider (322,122 B),
> Crab/Hexapod (338,634 B) and all four bipeds (2,239,066 B, 558,209 params) share exact
> sizes because they share exact *shapes* — same obs, same act, same network. The **weights
> differ**; the file layout does not.
>
> **The observation formula is exact:** `obs = 3N + 19`, where `N` = joint count = action count.
> Verified on all twelve: Worm 3(5)+19 = 34 · Kangaroo 3(6)+19 = 37 · Spider 3(8)+19 = 43 ·
> Crab 3(12)+19 = 55 · Centipede 3(17)+19 = 70 · bipeds 3(11)+19 = 52.

#### A2 · Ports — `Assets/unity_export/<Port>/`

| Port | Size (B) | Input | Output | Params | Hidden | Activation | Producer | In catalog |
|---|---:|---|---|---:|---|---|---|:-:|
| **IsaacBox** | 184,076 | `obs (1, 75)` | `actions (1, 21)` | 45,611 | 128 × 3 | ELU | Isaac Lab / RSL-RL | ✅ `spawnHeight 0.714` |
| **IsaacH1** | 178,875 | `obs (1, 69)` | `actions (1, 19)` | 44,435 | 128 × 3 | ELU | Isaac Lab / RSL-RL | ✅ `1.0` |
| **MujocoBiped** | 328,776 | `obs (1, 49)` | `action (1, 12)` | 81,774 | 256 × 2 | Tanh | MuJoCo 3.12 / PPO | ❌ ported, not registered |

#### A3 · Fido — the one racer with no ONNX

| | |
|---|---|
| **File** | `Assets/Creature/policy.json` — 851,712 B, format `mjx-unity-policy/1`, run `walk03` |
| **Shape** | 33 → 128 → 128 → 128 → 16, activation **SiLU** |
| **Parameters** | **39,506** (39,440 weights+biases, plus a 33-wide `mean` and 33-wide `std` normaliser stored in the same JSON) |
| **16 outputs for 8 actions** | mean(8) ‖ std(8) — a Gaussian head, same as the ML-Agents brains, just not in ONNX form |
| **Export fidelity** | `verifiedMaxAbsError: 2.03e-06` against the Python reference |
| **Why not ONNX** | He is a MuJoCo creature. `CreatureEntry.brainInPrefab = true` exists for exactly this row; the menu gates on `HasBrain`, not on `model != null`. |

### Matrix B — Physics & Actuators

All ML-Agents creature rigs are `ArticulationBody` chains driven through `xDrive` (PhysX
implicit PD). **No `ConfigurableJoint` and no SLERP drive appears anywhere in the project** —
an earlier edition of this document claimed both; neither is in the prefabs.

| Creature | Bodies | Joints (= DOF = actions) | Joint type | Kp (stiffness) | Kd (damping) | forceLimit / `_maxJointTorque` | `_jointDriveScale` | Gait freq | Behavioral purpose |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| **Worm** | 6 | 5 | Revolute, ±45° | 6750 | 450 | **1350 N·m** | 45° | 0.796 Hz | Serpentine body wave. First slice; the curriculum clock every lesson gates on. |
| **Blob** | 6 | 5 | **Prismatic** | 2500 | 166.67 | 500 N | **0.3 m** | 0.6 Hz | Soft-body pulse — five extending lobes, not rotating limbs. The only prismatic rig; its drive scale is **metres**, not degrees. |
| **Kangaroo** | 7 | 6 | Revolute, ±60° | 4250 | 283.33 | 850 N·m | 60° | 1.2 Hz | Two-legged hop with a counterweight tail. |
| **Quad** | 9 | 8 | Revolute, ±45° | 4500 | 300 | 900 N·m | 45° | 1.0 Hz | Quadruped trot — 4 legs × 2 joints. Fastest on flat, least stable on rough. |
| **Spider** | 9 | 8 | Revolute, ±40° | 3000 | 200 | 600 N·m | 40° | 0.796 Hz | Octopedal crawl — 8 legs × 1 joint. Alternating-tetrapod phase table. |
| **Crab** | 13 | 12 | Revolute, ±45° | 3500 | 233.33 | 700 N·m | 45° | 0.9 Hz | Sideways scuttle, wide low stance. |
| **Hexapod** | 13 | 12 | Revolute, ±45° | 3500 | 233.33 | 700 N·m | 45° | 0.9 Hz | Insect alternating-tripod — 6 legs × 2 joints. Same rig budget as Crab, different phase table. |
| **Centipede** | 18 | 17 | Revolute, ±35° | 4000 | 266.67 | 800 N·m | 35° | 0.9 Hz | Many-leg coordination; the widest ML-Agents control problem. **Authored lying along its body axis (90° on X)** — dropping that rest rotation stands the capsule chain into a tower that collapses to NaN. |
| **Grandma / Grandpa / Matt / Nick** | 12 | 11 | Revolute, ±60° | 1000 & 837.5 | 66.67 & 55.83 | 200 & 167.5 N·m | 60° | 1.1 Hz | The four `.glb` bipeds. Eleven joints of balance — hence 512 × 3, and no imitation warm start. |

> **A consistent gain law, and one exception.** Every creature rig satisfies **Kp : Kd = 15 : 1**
> and **forceLimit = Kp / 5** — Worm 6750/450/1350, Quad 4500/300/900, Blob 2500/166.67/500,
> and so on. The bipeds keep the 15:1 ratio (1000/66.67, 837.5/55.83) but carry two gain tiers
> and a `_maxJointTorque` of **670** that matches *neither* drive tier (200 or 167.5). Since
> `_maxJointTorque` is only the denominator in `ComputeNormalizedTorque()`, this makes biped
> fatigue read **~3× lower** than the other creatures at the same real load — the four bipeds
> effectively never tire. Worth fixing before they are ever added to the catalog.
>
> Every rig also carries a root body at `ArticulationJointType.Fixed`, which is why `Bodies`
> is always `Joints + 1`.

#### B2 · Ports — different actuator model, different frame

| Port | Rig source | Bodies / joints | Drive | Gains | Frame & timing | Notes |
|---|---|---:|---|---|---|---|
| **IsaacBox** | `isaacbox_rig.json`, built at runtime by `IsaacBoxAgent` (the prefab holds no `ArticulationBody`) | 22 / 21 revolute | `ArticulationDrive`, gains read from the rig JSON per joint | 11 families: spine 400/15/200 · hip_pitch & knee 150/5/150 · hip_roll 100/4/150 · hip_yaw 80/3/150 · ankle_pitch 30/3/50 · ankle_roll 20/2/50 · shoulder ×3 30/3/40 · elbow 20/2/40 | Isaac Z-up → Unity Y-up; physics 0.005 s, decimation 4 → **50 Hz policy** | 45.0 kg, 4/4 solver iterations, self-collisions off. `target = default + 0.5 · action`. |
| **IsaacH1** | `IsaacH1_rig.json` + USD meshes | 20 / 19 revolute | same | from the rig JSON | physics 0.005 s, decimation 4 → **50 Hz** | 51.44 kg (USD mass wins over the vendor URDF's 59.34 kg). `velocity_commands(3)` replaces `target_pos_b(3)`. |
| **MujocoBiped** | `MujocoBiped_rig.json` | 13 links / 12 revolute | ported PD | — | MuJoCo Z-up; physics 0.005 s, frame skip 5 → **40 Hz policy** | Plumbing exact to float precision; ~22% speed parity. The residual gap is physics, not tuning. |
| **Fido** | `Assets/Creature/creature.xml` (MJCF) | quadruped, 8 actuated | **MuJoCo `MjActuator`, not PhysX at all** | in the MJCF | `ctrlDt 0.02`, `nSubsteps 5`, prefab `actionDecimation 4` → exact 0.02 s at the project's 0.005 s | Has no `ArticulationBody` — the reason `ICreatureAgent.Body` exists. Cannot steer: his 33 observations describe only his own body. |

---

## Tier 3 · Complete specification

### 3.1 · Export provenance

Every ML-Agents brain is the frozen-policy output of an `mlagents-learn` run, exported through
`mlagents-export`. The graph metadata is identical across all twelve:

```
producer_name    : pytorch
producer_version : 2.4.1
ir_version       : 4
opset_import     : ai.onnx v9
metadata_props   : (empty — no param_count, no run-id, no training date)
```

That empty `metadata_props` block is why run provenance cannot be recovered from the `.onnx`
itself and must be tracked in `Config/*.yaml` headers and `results/<run-id>/` instead. As of
this regeneration **`results/` does not exist in the working tree**, so the mean-reward and
run-id columns that earlier editions of this document filled in were not verifiable and have
been removed rather than restated. The live provenance that *does* survive on disk is the
Isaac side, in §3.5.

`Assets/Agents/<Name>_v01/<Name>_v01.onnx.meta` carries the stable `ModelAsset` GUID.
**Overwrite `.onnx` files in place** — replacing the file loses the GUID and silently unwires
`CreatureCatalog.asset`.

### 3.2 · Per-model ONNX graph node inventory

The twelve ML-Agents graphs are node-for-node identical apart from tensor widths. One
representative dump (`Worm_v01.onnx`, 12 initializers):

| Op | Count | Role |
|---|---:|---|
| `Sub`, `Div`, `Clip` | 1 / 3 / 3 | The baked observation normaliser (`normalize: true`) plus the action clamp and the ÷3 rescale |
| `Gemm` | 3 | `seq_layers.0` W(256, obs) · `seq_layers.2` W(256, 256) · `mu` W(N, 256) |
| `Sigmoid`, `Mul` | 2 / 4 | Two swish activations — `x · sigmoid(x)` |
| `Exp` | 1 | `log_sigma` → `sigma` |
| `RandomNormalLike` | 1 | The stochastic sample. **This is the node that makes `continuous_actions` non-deterministic**; race-time inference reads `deterministic_continuous_actions` instead. |
| `Add`, `Concat`, `Constant`, `Identity` | 2 / 1 / 3 / 3 | Bias adds, output plumbing |

Five outputs per ML-Agents graph:

| Output | Shape | Used by |
|---|---|---|
| `continuous_actions` | `(batch, N)` | Training rollouts — sampled |
| `deterministic_continuous_actions` | `(batch, N)` | **Race inference** — the mean, no sampling |
| `version_number` | `(1)` | ML-Agents API compatibility check |
| `memory_size` | `(1)` | 0 — **no LSTM in any shipped brain.** An earlier edition of this document claimed "optional LSTM(128)"; there is none. |
| `continuous_action_output_shape` | `(1)` | Runtime shape assertion |

The three ports are much plainer graphs, which is the whole point of them:

| Port | Ops | Why it differs |
|---|---|---|
| **IsaacH1** | `Gemm`, `Elu` only | The RSL-RL normaliser was **not** baked in; the Unity agent normalises before the call |
| **IsaacBox** | `Concat`, `Sub`, `Div`, `Gemm`, `Elu` | The fitted `obs_normalization` **is** baked in (`obs_normalization_baked_into_onnx: true`), so Unity feeds raw observations — deliberately unlike IsaacH1 |
| **MujocoBiped** | `Gemm`, `Tanh`, `Clip`, `Mul`, `Sub`, `Constant` | Tanh-squashed policy with an explicit output rescale |

None of the three has a `RandomNormalLike`: they export the deterministic actor directly.

### 3.3 · Observation vector layouts, index by index

**ML-Agents creatures** — `Agent_Creature.CollectObservations`, `obs = 3N + 19`. Every value
passes through `Safe()` / `SafeVector()`, because one NaN aborts the Academy step for *every*
agent in the scene.

| Range | Width | Contents | Normalisation |
|---|---:|---|---|
| `[0, 3N)` | 3 per joint | `jointPosition[0]`, `jointVelocity[0]`, `IsGrounded` | ÷ (`_jointDriveScale` · Deg2Rad), clamp ±1 · ÷ 10 rad·s⁻¹, clamp ±1 · 1.0 or 0.0 |
| `[3N, 3N+3)` | 3 | `root.up` | world-space unit vector, raw |
| `[3N+3]` | 1 | `root.position.y` | **raw metres, unnormalised** |
| `[3N+4, 3N+7)` | 3 | goal direction | `InverseTransformDirection(toGoal.normalized)` — root-local |
| `[3N+7]` | 1 | goal distance | `clamp01(|toGoal| / 20 m)` |
| `[3N+8, 3N+11)` | 3 | root linear velocity | root-local, ÷ 2 m·s⁻¹, `ClampMagnitude 1` |
| `[3N+11, 3N+14)` | 3 | root angular velocity | root-local, ÷ 20 rad·s⁻¹, `ClampMagnitude 1` |
| `[3N+14]` | 1 | stamina | 0..1 |
| `[3N+15, 3N+19)` | 4 | terrain look-ahead | Raycast down from +3 m, range 6 m, at 0.5 / 1.2 / 2.2 / 3.5 m along flattened forward; `(hit.y − root.y) / 2`, clamp ±1 |

**IsaacBox** — 75 floats. This ordering is a contract with `chase_env_cfg.py`; change one side and you must change both.

| Term | Start | End | Dim |
|---|---:|---:|---:|
| `base_lin_vel` | 0 | 3 | 3 |
| `base_ang_vel` | 3 | 6 | 3 |
| `projected_gravity` | 6 | 9 | 3 |
| `target_pos_b` | 9 | 12 | 3 |
| `joint_pos` (relative to default) | 12 | 33 | 21 |
| `joint_vel` (relative) | 33 | 54 | 21 |
| `actions` (previous) | 54 | 75 | 21 |

**IsaacH1** — 69 floats, the same seven terms with `velocity_commands(3)` where IsaacBox has
`target_pos_b(3)`, and 19-wide joint blocks: `[0,3) [3,6) [6,9) [9,12) [12,31) [31,50) [50,69)`.

**Fido** — 33 floats: `gravity_local(3) · linvel_local(3) · angvel_local(3) · joint_pos(8) ·
joint_vel(8) · last_action(8)`. **There is no goal term anywhere in that list** — which is
precisely why he walks robustly at ~1.6 m/s and still curves away from the finish line. Fixing
it means retraining with a heading observation; `OBS_LAYOUT` in `training/fido/creature_env.py`
is a contract with `CreatureAgent.BuildObservation`.

**MujocoBiped** — 49 floats, with clip constants in the rig JSON (`clipLinVel 10`,
`clipAngVel 10`, `clipJointVel 20`, `maxTargetDistance 10`). One deliberate bug is preserved:
MuJoCo's free joint stores `qvel[3:6]` in the *body-local* frame and `env.py` applies `rotᵀ`
to it anyway, so `obs[7:10]` is `Rᵀ Rᵀ ω_world`. **Reproduce it; do not fix it** — the policy
was trained on it.

### 3.4 · Actuator specification — IsaacBox, per joint family

Joint order is the ONNX action order, rewritten from the live sim by `export_bundle.py`.

| # | Joint | Family | Kp | Kd | Effort limit | Range (rad) |
|---:|---|---|---:|---:|---:|---|
| 0 | `spine_pitch` | spine | 400 | 15 | 200 | −0.5 … 0.5 |
| 1–2 | `hip_yaw_L/R` | legs | 80 | 3 | 150 | from rig JSON |
| 3–4 | `shoulder_roll_L/R` | arms | 30 | 3 | 40 | " |
| 5–6 | `hip_roll_L/R` | legs | 100 | 4 | 150 | " |
| 7–8 | `shoulder_pitch_L/R` | arms | 30 | 3 | 40 | " |
| 9–10 | `hip_pitch_L/R` | legs | 150 | 5 | 150 | " |
| 11–12 | `shoulder_yaw_L/R` | arms | 30 | 3 | 40 | " |
| 13–14 | `knee_L/R` | legs | 150 | 5 | 150 | " |
| 15–16 | `elbow_L/R` | arms | 20 | 2 | 40 | " |
| 17–18 | `ankle_pitch_L/R` | feet | 30 | 3 | 50 | " |
| 19–20 | `ankle_roll_L/R` | feet | 20 | 2 | 50 | " |

All armatures are 0.0. Twenty-two links, 45.0 kg total, hips at 0.76 m at zero pose and
0.744 m at the default pose; spawn is `posIsaac [0, 0, 0.764]`, and the catalog carries
`spawnHeight 0.714`. Solver: 4 position / 4 velocity iterations, TGS, self-collisions **off**.

### 3.5 · Live training provenance — the only run with surviving telemetry

`ISAAC/logs/rsl_rl/boy_chase_flat/` holds two runs, both read out of their TFEvents files:

| Run | Iteration range | Final mean reward | Peak | Final episode length | Base-contact terminations | Targets reached / episode | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| `2026-09-02_21-32-53_full` | 0 → 2,015 | **24.37** | 25.50 | 986.4 / 1000 steps | 2.1% | 2.42 | 59,583 steps/s (peak 67,163) |
| `2026-09-03_09-03-04_finish` | 2,000 → 2,999 | **22.68** | 25.34 | 995.9 / 1000 steps | 2.1% | 2.48 | 61,609 steps/s (peak 75,968) |

The `_finish` run **resumes from `model_2000.pt`** — its TFEvents begin at global iteration 2000 —
and runs the experiment out to its configured `max_iterations` of 3,000, ending at `model_2999.pt`.
Its first ~30 iterations show a steep climb from reward −0.06 to 17.5; that is RSL-RL's running-mean
buffers refilling after the resume, not the policy relearning.

Reward rose from 0.003 to 1.32 on `target_speed_exp` alone over the full run; `distance_to_target`
fell from 6.55 m to 4.05 m. `Policy/mean_std` settled at 0.86 (full) and 0.91 (finish) — still
exploring, not collapsed.

> **`model_2999.pt` was promoted, and it is the best argument in this document for not trusting mean
> reward.** Commit `afa3eeb` (2026-09-03 10:13) re-exported the shipped ONNX from the finish run's
> final checkpoint. Its **mean reward is lower** — 22.68 against the incumbent's 24.37 — and it is
> marginally *slower*: 0.9869 m/s against 0.9916. On the metric the task is actually scored by it is
> **better**: **8.09 targets reached per minute, up from 7.91**, with falls still at exactly zero and
> episode length up from 986.4 to 995.9 steps. A brain that closes on more goals while scoring less
> reward is the whole reason promotion goes through racing and not through the reward curve. The graph
> is structurally identical — 184,076 bytes, 45,611 parameters, `obs (1, 75)` → `actions (1, 21)` —
> only the weights moved.

Independently measured evaluation figures, from the rig JSON `eval` blocks:

| Port | Measured |
|---|---|
| **IsaacBox** (`model_2999.pt`, shipped) | mean speed 0.987 m/s · toward target 0.939 m/s · target-speed error 0.079 · **falls per robot-minute 0.0** · **8.09 targets/min** · reference forward speed 0.953 m/s |
| *IsaacBox* (`model_2000.pt`, superseded 2026-09-03) | mean speed 0.992 · toward target 0.949 · error 0.071 · falls 0.0 · 7.91 targets/min · reference 0.962 |
| **IsaacH1** | mean speed 0.510 m/s · linear-velocity tracking error 0.117 · falls per robot-minute 0.125 |
| **MujocoBiped** | 4.0 targets/episode · 516 steps/episode · mean closing speed 1.15 m/s · **survived a full episode only 30% of the time** |
| **Fido** | ~1.6 m/s sustained walk, but no heading observation, so he traces a circle over a long straight |

### 3.6 · Runtime inference configuration

| Path | Component | Backend | Rate |
|---|---|---|---|
| 9 ML-Agents creatures | `Agent_Creature` + `BehaviorParameters` | Inference Engine, CPU | `DecisionPeriod 20` × 0.005 s = **0.1 s**, `TakeActionsBetweenDecisions = 1` |
| IsaacBox | `IsaacBoxAgent` → `Agent_IsaacBox` adapter | `new Worker(model, BackendType.CPU)` | decimation 4 × 0.005 s = **0.02 s (50 Hz)** |
| IsaacH1 | `IsaacH1Agent` → `Agent_IsaacH1` adapter | same | **0.02 s (50 Hz)** |
| Fido | `Agent_Fido` reading `policy.json` | hand-rolled float MLP, no Inference Engine | `actionDecimation 4` × 0.005 s = **0.02 s** |

**Δt = 0.005 s (200 Hz), solver iterations 12, is load-bearing.** It is an exact multiple of
0.02 s (IsaacH1, IsaacBox, Fido) and 0.025 s (MujocoBiped). Moving to 1/240 s breaks IsaacH1's
exact step; moving to 0.004 s breaks MujocoBiped's and forces `DecisionPeriod` 20 → 25 on all
twelve ML-Agents prefabs. If Δt ever changes, **`DecisionPeriod` on every prefab and Fido's
`actionDecimation` must change with it.**

> **Known upstream noise, do not vendor the package to fix it.** `TensorProxy`'s finalizer
> dereferences `data.dataOnBackend` with no null check. On a Pixel 9 Pro that is ~3,553
> `NullReferenceException`s in two minutes, ~30/sec. It is harmless — `SentisModelInfo` builds
> its `Worker` with `DeviceType.CPU`, so the guarded body is a no-op — so filter the log
> (`adb logcat | grep -v TensorProxy`) and report upstream. Embedding the package to patch it
> costs 40 MB and a fork; that was tried on 2026-08-29 and reverted.

### 3.7 · Orphans and gaps

| Artifact | State | What it would take to race |
|---|---|---|
| `Blob_v01.onnx`, `Kangaroo_v01.onnx` | Trained, exported, prefab wired, **absent from `CreatureCatalog.asset`** | One catalog entry each with a `spawnHeight`. Blob is the only prismatic rig, so check its 0.3 m drive scale survives spawn quirk scaling. |
| `Grandma/Grandpa/Matt/Nick_v01.onnx` | Trained (512 × 3, no BC/GAIL), absent from the catalog | Catalog entries — **and fix `_maxJointTorque = 670` first**, or they race without fatigue (§Matrix B note). |
| `MujocoBiped.onnx` | Ported, verified to float precision, no adapter registered | An `Agent_MujocoBiped` adapter implementing `ICreatureAgent`, plus a catalog entry. |
| `results/` | **Does not exist** | Nothing to recover; the ML-Agents run history is gone from the working tree. Future runs must be preserved, or promotion decisions cannot be audited. |
| Demonstrations | `AllLoco8h02.yaml` references `Assets/Demonstrations/<Name>.demo` for BC + GAIL | Re-record via `PoRacer/Training/Enable Demo Recorders In Open Scene` **whenever the observation size changes** — an old `.demo` carries the old obs size and mlagents rejects it. |

---

*Regenerated 2026-09-03. Re-run the extraction after any retrain, re-export, prefab gain
change or catalog edit; every number above is mechanically checkable against the files.*
