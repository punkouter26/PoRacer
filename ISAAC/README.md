# ISAAC/ - the Boy's Isaac Lab side

Everything Python for the Boy character lives here, in the folder you asked for. Nothing in
`Assets/` references it; only the exported bundle (`Boy.onnx`, `boy_rig.json`,
`isaac_reference.json`, `export_report.json`, `CONTRACT.md`, `kinematics_reference.json`)
crosses into `Assets/unity_export/Boy/`.

| Path | What |
|---|---|
| `boy_rig/Boy_Character.glb` | The authored character. The Python side reads its skeleton from the glTF JSON; the FBX twin lives in `Assets/Art/Models/` for Unity. |
| `boy_rig/build_boy_rig.py` | GLB -> `out/boy.usda` (headless articulation, primitives only), `out/boy_rig.json` (the single source of truth for gains, limits, masses, spawn), `out/kinematics_reference.json`. Also copies the two JSONs into `Assets/unity_export/Boy/`. Pure Python; validates the USD if `usd-core` is installed. |
| `boy_tasks/` | pip package registering `Isaac-Chase-Flat-Boy-v0` and `-Play-v0`: articulation cfg (reads the rig JSON), manager-based env cfg, the `TargetPositionCommand` term, chase rewards, RSL-RL PPO cfg. |
| `scripts/train.py` | Headless RSL-RL training. Starts TensorBoard on :6006 BEFORE the simulator; tears down trainer -> env -> sim -> TensorBoard. |
| `scripts/export_bundle.py` | Checkpoint -> `Boy.onnx` (opset 15, IR 8, batch 1, normaliser baked) + onnxruntime check (< 1e-4) + reference trajectory + evaluation + `export_report.json` + `CONTRACT.md`; rewrites `boy_rig.json` with the simulator's joint order and drops the bundle into `Assets/unity_export/Boy/`. |
| `install.ps1` | Clones Isaac Lab into `ISAAC/isaaclab`, builds its venv with Isaac Sim, installs `rsl_rl` and `boy_tasks`. |

## The pipeline

```powershell
# 0. rig (already run; re-run after editing build_boy_rig.py)
python ISAAC\boy_rig\build_boy_rig.py

# 1. one-time setup (Isaac Sim is a ~10 GB download)
powershell -File ISAAC\install.ps1

# 2. train - TensorBoard comes up first on http://localhost:6006
ISAAC\isaaclab\isaaclab.bat -p ISAAC\scripts\train.py --num_envs 2048 --max_iterations 3000

# 3. export + validate + bundle into Assets/unity_export/Boy/
ISAAC\isaaclab\isaaclab.bat -p ISAAC\scripts\export_bundle.py --num_envs 64

# 4. Unity (Editor): Boy > Rebuild Rig Asset From JSON, Boy > Build Prefab,
#    then Test Runner > PlayMode > Boy (rung 0 checks the ONNX against the recording)
```

## Task summary

* **Morphology** - 22 links, 21 revolute joints, 45 kg. Hip yaw/roll/pitch, knee, ankle
  pitch/roll, spine pitch, shoulder roll/pitch/yaw, elbow. Neck, head, hands and clavicles
  are welded into their parents. Multi-DoF joints are chains of single-axis joints through
  0.2 kg intermediate links so PhysX 5 and Unity's PhysX 4.1 build the same tree.
* **Zero pose = the authored T-pose**; the default (standing) pose is an offset:
  shoulder roll -/+1.35 rad, elbow -/+0.4, hip pitch -0.2, knee 0.4, ankle -0.2.
* **Task** - target chasing on flat ground. A goal is sampled on a 3-10 m ring, resampled
  on reach (0.5 m) or after 8-12 s. Observation = the goal in the base frame, norm-clipped
  to 5 m, plus the usual proprioception (75 floats). Reward tracks 1 m/s toward the goal,
  faces it, rewards each reach, with the H1 gait/regularisation terms.
* **Control** - joint position targets, `target = default + 0.5 * action`, implicit PD
  drives, 200 Hz physics / decimation 4 / 50 Hz policy. Same timing as the IsaacH1 port and
  the PoRacer project's locked `Time.fixedDeltaTime`.
* **Network** - MLP 3 x 128 ELU, empirical observation normalisation ON (baked at export).

## Things to know before training

* **Joint order.** PhysX decides the articulation's joint order when it builds the USD;
  `boy_rig.json` ships a provisional breadth-first guess. `export_bundle.py` reads the real
  order from the simulator, rewrites the JSON, and aborts if the joint *set* differs. Unity
  reads whatever order the JSON says, so nothing else needs to change.
* **GPU.** This machine's RTX 2060 (6 GB) is below Isaac Sim's stated minimum. Try
  `--num_envs 512`; if the app fails to start, train on another machine and copy
  `logs/rsl_rl/boy_chase_flat/<run>/` back here before exporting.
* **API drift.** `chase_env_cfg.py` carries small shims for the physics-backend split
  (`isaaclab_physx`) and the RSL-RL model-cfg rename (`RslRlMLPModelCfg`). If your
  checkout is older or newer than what the shims cover, the import error will name the
  symbol.
* **The reward for pace is a target speed, not a command.** There is no velocity command
  in the observation; a race just hands the finish line to `BoyAgent.target` and the
  policy walks at the pace it was rewarded for.
