# Fido — a trained MuJoCo quadruped, ready to drop into a Unity project

Fido is a four-legged creature trained with MuJoCo MJX + Brax PPO. He walks
forward at **~1.1–1.3 m/s** and stays upright for a full 20-second episode
(989/1000 steps). The trained brain is a 33→128→128→128→16 MLP that runs at
50 Hz on the CPU — no ML runtime, no inference package, just C#.

Everything here has been run and measured, not assumed. See *Verification* below.

---

## Install into a Unity project (3 steps, ~2 minutes)

**Requires Unity 6000.0+ (developed and verified on 6000.5.8f1).**

### 1. Add the MuJoCo plug-in

Copy `Packages/org.mujoco/` into your project's `Packages/` folder.

Use **this** copy, not a fresh clone from GitHub — it carries two fixes that
upstream does not have (details in *Why the bundled plug-in* below). Unity will
show it in the Package Manager as **MuJoCo 3.12.0**.

### 2. Import Fido

**Assets ▸ Import Package ▸ Custom Package…** → `Fido.unitypackage` → Import.

This lands in `Assets/Creature/` and preserves asset GUIDs, so the included
scene keeps its references. (`RawAssets/Creature/` is the same content as loose
files if you would rather copy it in — copy the `.meta` files too, or every
reference breaks.)

### 3. Build the scene

**MuJoCo Creature ▸ Build Verification Scene**, then press **Play**.

You get Fido walking across a 1 m grid, a follow camera, and an on-screen
readout of speed, distance, and torso height.

---

## Using Fido in your own scene

1. **Assets ▸ Import MuJoCo Scene** → `Assets/Creature/creature.xml`
2. Add **CreatureAgent** to the imported root, drag `policy.json` into its
   *Policy Json* slot.
3. Add a **CreatureSceneBootstrap** to any GameObject. **This is not optional** —
   see *The timestep trap*.
4. Press Play. `torso`, `joints` and `actuators` auto-resolve by name.

Optional: `CreatureCameraFollow` (Fido outruns a fixed camera) and `CreatureHud`.

### The timestep trap

The MuJoCo plug-in **ignores** `<option timestep>` in the MJCF and steps once per
`FixedUpdate`, so the real physics rate is `Time.fixedDeltaTime` — a **project**
setting. At Unity's 0.02 default, a policy trained at 0.004 runs at 10 Hz instead
of 50 Hz and Fido just flails, with nothing in the console to explain it.

`CreatureSceneBootstrap` sets it at load, and `CreatureAgent` logs a loud error if
the rate is ever wrong. Keep both.

---

## Contents

```
Fido/
├─ Fido.unitypackage      import this (scene, model, policy, scripts, materials)
├─ Packages/org.mujoco/   MuJoCo Unity plug-in 3.12.0, patched (required)
├─ RawAssets/Creature/    same assets as loose files, .meta included
├─ Training/              retrain or re-export Fido (needs WSL2/Linux)
├─ screenshot.png         what a working scene looks like
└─ README.md
```

**Scripts** (namespace `Creature`, assembly `Creature`):

| Script | Role |
| --- | --- |
| `CreatureAgent` | builds observations, runs the policy, drives the actuators |
| `CreaturePolicy` | MLP inference; mirrors the Python exporter exactly |
| `CreatureSceneBootstrap` | forces the 0.004 s physics timestep |
| `CreatureCameraFollow` | keeps the camera on a creature moving 1.3 m/s |
| `CreatureHud` | live speed / distance / height readout |
| `CreatureSceneBuilder` *(Editor)* | one-click verification scene |

The assembly is named `Creature`, not `Fido` — renaming it would break the
scene's script references. Fido is the creature; `Creature` is the code.

---

## Why the bundled plug-in

The stock MuJoCo 3.12.0 Unity plug-in has two problems on Unity 6.5. Both are
fixed in `Packages/org.mujoco/` here, and neither is fixed upstream as of
2026-08-29:

1. **It does not compile.** `MjMeshFilter.cs:61` calls `Mesh.GetInstanceID()`,
   which Unity 6.5 marks obsolete-**as-error** (CS0619). One line is wrapped in
   `#if UNITY_6000_5_OR_NEWER` to use `GetEntityId()`. This failure takes the
   whole `Mujoco.Runtime` assembly down with it.
2. **`mujoco.dll` may not register as a native plug-in.** Unity can generate a
   meta with only a GUID and no `PluginImporter`, giving `DllNotFoundException`
   at Play (upstream issue #1146). `mujoco.dll.meta` here is authored explicitly
   for Editor + Standalone Win64.

**`mujoco.dll` here is Windows x86_64.** For macOS or Linux, take the matching
`libmujoco` from the [MuJoCo 3.12.0 release](https://github.com/google-deepmind/mujoco/releases),
rename it as the MuJoCo docs describe, and drop it beside `package.json`.

MuJoCo is Apache-2.0; `Packages/org.mujoco/` keeps its `LICENSE`.

---

## Two more things the importer silently does

- **`<keyframe>` is dropped.** Fido would otherwise start with straight legs, a
  stance he never trained from. His home pose therefore travels inside
  `policy.json` as `homeJointPos`, and `CreatureAgent` applies it on the first
  step. Turn it off with `applyHomePose` if you want the imported pose.
- **`MjScene.CreateScene()` renames GameObjects** — `fl_hip` becomes `fl_hip_30`.
  `CreatureAgent` strips a trailing `_<digits>` when binding, so keep digits out
  of the tail of MJCF names if you edit the model.

---

## Verification

Measured on an RTX 5070 Ti, Unity 6000.5.8f1:

| Check | Result |
| --- | --- |
| Export vs Brax's own inference, 256 observations | max diff **2.0e-06** |
| `CreaturePolicy.cs` vs numpy, 252 observations | max diff **3.2e-06** |
| Policy in native MuJoCo, 4 s | **+4.44 m**, 1.11 m/s, never fell |
| Policy in the built Unity scene, 4 s | **+5.16 m**, 1.29 m/s, min height 0.245 |
| Live Play mode in the Editor | walked continuously, no errors |
| Ground grid | **1.000 m** per square (measured, not eyeballed) |
| **Installed into a brand-new empty project** | **+5.16 m, 1.29 m/s, ALL PASS** |

That last row is the one that matters for reuse: a fresh Unity project, plug-in
copied in, assets added, scene built, policy run — all four assemblies compiled
and Fido walked, with numbers identical to the source project. That project had
**no URP**, and the materials correctly fell back to the `Standard` shader, so the
pipeline detection works in both directions.

**What was verified how:** the fresh-project install was tested via
`RawAssets/Creature/` (batch mode cannot complete an async `.unitypackage`
import). `Fido.unitypackage` holds byte-identical files with the same GUIDs —
its contents were verified by extraction — but the interactive import itself is
the one step I could not exercise headlessly. If it ever misbehaves, copying
`RawAssets/Creature/` into `Assets/` is the tested equivalent.

Unity runs ~16% faster than native MuJoCo. That is expected: the plug-in does not
serialize `ls_iterations` when regenerating the MJCF, so Unity solves with
MuJoCo's default 50 where training used 8 — i.e. *more* accurately. Both gaits are
stable; the policy transfers.

---

## Retraining Fido

`Training/` has the full pipeline. It needs **WSL2 or Linux**: JAX publishes no
CUDA wheels for Windows, and MJX on CPU is slower than plain MuJoCo.

```bash
pip install -r requirements-wsl.txt
python check_env.py                                    # confirm GPU
python train.py --name fido2 --num_timesteps 60000000 --num_envs 4096
python export_policy.py --run fido2                    # rewrites policy.json
```

~53 min for 60M steps. `walk03-params/` holds the current brain, so you can
re-export without retraining. Checkpoints are written every eval, so a run can be
stopped at any time without losing progress.

Two settings that matter more than they look:

- **Keep the survival bonus small.** An early run reached reward 710 while
  standing perfectly still at 0.04 m/s, because staying alive paid ~10x more than
  moving. Falling already ends the episode; a big alive bonus double-counts it.
- **Keep `batch_size * num_minibatches == num_envs`.** Setting `batch_size` 8x too
  high cut gradient updates 8x and roughly halved learning speed per env step.

Use the default `--impl jax`. MuJoCo Warp is **1.85x slower** on a creature this
small and cannot be driven correctly through `jax.vmap` — it silently drops
contacts and Fido sinks through the floor.

**If you change the model**, the observation layout is a contract between
`creature_env.py` (`OBS_LAYOUT`) and `CreatureAgent.BuildObservation`. Change one,
change both, retrain, and re-export.
