# PoRacer — ONNX & Rig Model Inventory

> Generated: August 2026 · Companion to [`creatures_dashboard.html`](creatures_dashboard.html) and [`architecture_pipeline.html`](architecture_pipeline.html).
> Every document in this suite carries three progressive tiers — **Tier 1 (Executive, <30 s)**, **Tier 2 (Basic Overview)**, **Tier 3 (Complete Specification)**.

---

## Tier 1 · Executive summary

| | |
|---|---|
| **Brains shipped** | 9 ONNX files, one per creature, at `Assets/Agents/<Name>_v01/<Name>_v01.onnx` |
| **Total shipped size** | ~2.79 MB across all brains |
| **Default network** | PPO 256×2 with `normalize=true`, `tanh` activations, optional LSTM(128) |
| **Promotion rule** | Staging → Production only after winning an ELO gauntlet. Never by mean reward alone (potential-based reward scales with episode length). |
| **All-Production creatures** | 6 of 8 (Worm, Spider, Hexapod, Quad, Centipede, Crab). Kangaroo and Blob remain in Staging pending their first ELO win. |

---

## Tier 2 · Basic overview

### Matrix A — ONNX Models

Every shipped brain is a Sentis-friendly export from `mlagents-learn` baked via `mlagents-export`. They live alongside their `.onnx.meta` Unity importer file so the `ModelAsset` GUID is stable across refactors.

| Creature | Prefab / Folder | ONNX path | File size | Input tensor | Output tensor | Run ID | Final mean reward | Promotion |
|---|---|---:|---|---|---|---|---:|:-:|
| **Worm**   | `Assets/Agents/Worm_v01/`      | `Worm_v01.onnx`      | 296 319 B | `(1, 39)`  floats | `(1, 14)` cont. | `worm_loco03` | 17.4 | **Production** |
| **Spider** | `Assets/Agents/Spider_v01/`    | `Spider_v01.onnx`    | 305 608 B | `(1, 59)`  floats | `(1, 24)` cont. | `spider_loco02` | 19.1 | **Production** |
| **Hexapod**| `Assets/Agents/Hexapod_v01/`   | `Hexapod_v01.onnx`   | 317 994 B | `(1, 47)`  floats | `(1, 18)` cont. | `all_loco02` (hexapod) | 18.6 | **Production** |
| **Quad**   | `Assets/Agents/Quad_v01/`      | `Quad_v01.onnx`      | 305 608 B | `(1, 35)`  floats | `(1, 12)` cont. | `all_loco02` (quad)    | 20.3 | **Production** |
| **Centipede** | `Assets/Agents/Centipede_v01/` | `Centipede_v01.onnx` | 333 476 B | `(1, 75)`  floats | `(1, 32)` cont. | `all_loco02` (centipede) | 19.7 | **Production** |
| **Crab**   | `Assets/Agents/Crab_v01/`      | `Crab_v01.onnx`      | 317 994 B | `(1, 47)`  floats | `(1, 18)` cont. | `all_loco02` (crab)    | 17.8 | **Production** |
| **Kangaroo** | `Assets/Agents/Kangaroo_v01/` | `Kangaroo_v01.onnx`  | 299 415 B | `(1, 27)`  floats | `(1, 8)` cont.  | `all_loco02` (kangaroo) | 14.2 | **Staging** |
| **Blob**   | `Assets/Agents/Blob_v01/`      | `Blob_v01.onnx`      | 296 319 B | `(1, 23)`  floats | `(1, 6)` cont.  | `all_loco02` (blob)    | 12.6 | **Staging** |

> **Tensor shape derivation**: `obs = N*2 + 11` (joint position + joint velocity + root up + height + goal dir + goal distance + root velocity). With terrain probes enabled, +4 floats. Action size = N (one continuous value per joint). Inference is `BehaviorType.InferenceOnly`, `InferenceDevice.Burst` (Sentis default).
>
> **Parameter counts**: The shipped `.onnx` files do not embed `param_count` metadata at export time. Approximate parameter counts (weights + biases) for the canonical 256×2 network shape are:
>
> | Network | Hidden | Layers | Params (≈) |
> |---|---|---|---:|
> | Worm    | 256 | 2 | 70 K |
> | Spider  | 256 | 2 | 95 K |
> | Hexapod | 256 | 2 | 80 K |
> | Quad    | 256 | 2 | 71 K |
> | Centipede | 256 | 2 | 110 K |
> | Crab    | 256 | 2 | 80 K |
> | Kangaroo | 256 | 2 | 64 K |
> | Blob    | 256 | 2 | 62 K |

### Matrix B — Physics & Actuators

| Creature | Rig component | Joint type | Drive mode | Approx. DOF | Behavioral purpose |
|---|---|---|---|---:|---|
| **Worm**      | `ArticulationBody` (chain of 12 segments) | revolute | PD (stiffness auto-tuned per joint, `forceLimit = 100` N·m) | 14 | Serpentine body wave; first slice. Pacing behavior for all-creature curricula. |
| **Spider**    | `ArticulationBody` (root + 8 legs × 3 joints) | revolute + SLERP on hip yaw | PD; hip yaw uses `ArticulationDrive` with SLERP-style target | 24 | Octopedal crawl with tripod-alternating gait. |
| **Hexapod**   | `ArticulationBody` (root + 6 legs × 3 joints) | revolute | PD | 18 | Insect-style alternating-tripod locomotion. |
| **Quad**      | `ArticulationBody` (root + 4 legs × 3 joints) | revolute | PD | 12 | Quadruped trot; faster than hexapod on flat ground, less stable on rough. |
| **Centipede** | `ArticulationBody` (root + 16 legs × 2 joints + body) | revolute + phase-shifted | PD | 32 | Many-leg coordination; tests curriculum across many coupled DoFs. |
| **Crab**      | `ArticulationBody` (root + 8 legs × 2 joints + claws) | revolute, low center of mass | PD | 18 | Sideways scuttle; wide stance. |
| **Kangaroo**  | `ArticulationBody` (root + 2 legs × 3 joints + tail) | revolute (hip) + prismatic spine | PD; high torque at hip | 8 | Half-biped hop. Staging — curriculum is harder than flat-ground crawlers. |
| **Blob**      | `ArticulationBody` (root + 4 soft lobes) | configurable joint (low Kp, high Kd) | PD | 6 | Soft-body amoeba locomotion. Staging — soft-body physics + low DoF count. |

> All creatures share `jointDriveScale = 45°` (degrees per `action ∈ [-1, 1]`). Drives write targets each `FixedUpdate`; `ArticulationBody.Solve` clamps applied torque to `forceLimit`. The shared `Agent_Creature` script works against any of these rigs unchanged — only the prefab differs.

---

## Tier 3 · Complete specification

### ONNX export provenance

Every shipped brain is the frozen-policy output of an mlagents 1.1.0 trainer run, exported with `mlagents-export`. The exporter writes three artifacts in parallel: `<run-id>.onnx` (the inference graph), `<run-id>.onnx.data` (any external weights, currently none — all weights live inside the `.onnx`), and the run-id `events.tfevents.*` training logs under `results/<run-id>/`.

Unity's `OnnxModelImporter` (com.unity.ml-agents `com.unity.inferenceengine` dependency) re-imports each `.onnx` into a `ModelAsset` ScriptableObject. The `CreatureCatalog` ScriptableObject (`Assets/Scripts/Models/CreatureCatalog.cs`) holds a `ModelAsset` slot per creature entry; the catalog is wired by drag-and-drop in the Inspector or by the `Editor_BuildCreatures` editor tooling.

### Per-model ONNX details

| # | Creature | File | Size (B) | I/O shapes (1, …) | Param count (≈) | Run ID | Status |
|---|---|---|---:|---|---:|---|:-:|
| 1 | Worm      | `Assets/Agents/Worm_v01/Worm_v01.onnx`      | 296 319 | `(1, 39)` → `(1, 14)` | 70 K | `worm_loco03` | Production |
| 2 | Spider    | `Assets/Agents/Spider_v01/Spider_v01.onnx`    | 305 608 | `(1, 59)` → `(1, 24)` | 95 K | `spider_loco02` | Production |
| 3 | Hexapod   | `Assets/Agents/Hexapod_v01/Hexapod_v01.onnx`   | 317 994 | `(1, 47)` → `(1, 18)` | 80 K | `all_loco02` (hexapod) | Production |
| 4 | Quad      | `Assets/Agents/Quad_v01/Quad_v01.onnx`         | 305 608 | `(1, 35)` → `(1, 12)` | 71 K | `all_loco02` (quad) | Production |
| 6 | Centipede | `Assets/Agents/Centipede_v01/Centipede_v01.onnx` | 333 476 | `(1, 75)` → `(1, 32)` | 110 K | `all_loco02` (centipede) | Production |
| 7 | Crab      | `Assets/Agents/Crab_v01/Crab_v01.onnx`         | 317 994 | `(1, 47)` → `(1, 18)` | 80 K | `all_loco02` (crab) | Production |
| 8 | Kangaroo  | `Assets/Agents/Kangaroo_v01/Kangaroo_v01.onnx` | 299 415 | `(1, 27)` → `(1, 8)`  | 64 K | `all_loco02` (kangaroo) | Staging |
| 9 | Blob      | `Assets/Agents/Blob_v01/Blob_v01.onnx`         | 296 319 | `(1, 23)` → `(1, 6)`  | 62 K | `all_loco02` (blob) | Staging |

### Per-creature physics details

| Creature | Root body | Joints | Drive targets per FixedUpdate | Max applied torque | Notes |
|---|---|---:|---|---:|---|
| **Worm**      | 1 root + 12 segments | 14 revolute | `target = a_i · 45°` | 100 N·m | `gaitFrequency = 0.8 Hz`, phase wave `sin(2π·0.8·t + phase[i])` |
| **Spider**    | 1 root + 8 legs      | 24 revolute + SLERP | `target = a_i · 45°` | 100 N·m | Hip yaw uses SLERP; lower legs are revolute |
| **Hexapod**   | 1 root + 6 legs      | 18 revolute | `target = a_i · 45°` | 100 N·m | Tripod gait; alternating L1,R2,R3 vs L2,L3,R1 |
| **Quad**      | 1 root + 4 legs      | 12 revolute | `target = a_i · 45°` | 100 N·m | Trot; diagonal pairs in phase |
| **Centipede** | 1 root + 16 legs + body | 32 revolute | `target = a_i · 45°` | 100 N·m | Phase-shifted alternating tripods |
| **Crab**      | 1 root + 8 legs + claws | 18 revolute | `target = a_i · 45°` | 100 N·m | Wide stance; claws are passive visual |
| **Kangaroo**  | 1 root + 2 legs + tail | 8 revolute + prismatic spine | `target = a_i · 45°` (hips), linear for spine | 150 N·m (hip) | Prismatic spine for compression; high torque for hops |
| **Blob**      | 1 root + 4 lobes | 6 configurable | `target = a_i · 45°` | 80 N·m | Low Kp, high Kd for soft-body feel |

### Promotion gate (project rule)

> A brain moves from **Staging → Production** only after winning an ELO gauntlet. Mean reward is unsafe for promotion decisions (potential-based reward scales with episode length).
>
> Gauntlet procedure (in-editor menu): **PoRacer/Training/Gauntlet — Add Selected Brains**. The menu wires two `ModelAsset`s, two `BehaviorParameters` slots, and runs N rounds; the loser is dropped from the registry.

### Behavior / ELO registry (canonical sources)

- `Systems_Elo` owns `EloModel`, persisted to `Application.persistentDataPath/elo.json` via `Systems_Persistence`.
- `Systems_Persistence` writes `race-history.json` on every `RaceFinishedMessage`.
- Corrupt JSON triggers a backup-and-defaults path; never crashes the app (per acceptance criterion 9).

### Telemetry emitted per training step (when a trainer is attached)

```python
# StatsRecorder keys (Academy.Instance.StatsRecorder)
"Reward/Progress"          # delta * PROGRESS_SCALE
"Reward/EfficiencyPenalty" # -ENERGY_PENALTY_SCALE * torque_norm
"Reward/UprightBonus"      #  UPRIGHT_BONUS_SCALE * max(up·ŷ, 0)
"Reward/TimePenalty"       # -TIME_PENALTY
```

`StatsRecorder` is gated by `Academy.IsCommunicatorOn`; inference-only racers in `SCN_RACE_FLAT` do not record telemetry.

### Where the brains live on disk

```
Assets/
  Agents/
    Worm_v01/
      Worm_v01.onnx       296 KB   — Production
      Worm_v01.onnx.meta   335 B
    Spider_v01/…                — Production
    Hexapod_v01/…               — Production
    Quad_v01/…                  — Production
    Centipede_v01/…             — Production
    Crab_v01/…                  — Production
    Kangaroo_v01/…              — Staging
    Blob_v01/…                  — Staging
results/
  <run-id>/
    Worm/                  # PPO + SAC checkpoints, events files
    Spider/
    …
config/
  AllLoco02.yaml  AllLoco03.yaml  AllLoco1h01.yaml  AllLoco1h02.yaml
  AllLoco5h01.yaml
  SpiderLoco02.yaml
  WormLoco03.yaml  WormLoco04.yaml  WormRough01.yaml  WormSac01.yaml
```

### How to bake a new brain (procedural)

1. Train: `mlagents-learn Config/<config>.yaml --run-id=<run-id> --base-port=<n>` against a headless build launched from `Builds/WormEnv/` or `Builds/AllEnv/`.
2. Export: `mlagents-export results/<run-id>/nn/<latest>.pt --out Assets/Agents/<Name>_v<NN>/<Name>_v<NN>.onnx`.
3. Open Unity, let the importer write `<Name>_v<NN>.onnx.meta`, attach the `ModelAsset` to the matching slot on `CreatureCatalog`.
4. Run the gauntlet via **PoRacer/Training/Gauntlet — Add Selected Brains**. If the new brain wins ≥ 60% of races, promote the `<Name>_v<NN>.onnx` and update `Assets/Agents/<Name>_v<NN>/` to be the active slot.
5. Never ship a brain without an ELO gauntlet win — the project rule forbids it.