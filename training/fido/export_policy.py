"""Export a trained Brax PPO policy to a Unity-readable JSON file.

  python export_policy.py --run <run_name>

Writes Assets/Creature/policy.json, which Assets/Creature/Scripts/CreaturePolicy.cs
loads at runtime. Before writing, the exported weights are re-evaluated in pure
numpy and compared against Brax's own deterministic inference function -- if the
two disagree the export fails rather than shipping a silently wrong policy.

Inference contract (mirrored exactly in CreaturePolicy.cs):
    x = (obs - mean) / std              # running observation statistics
    x = silu(x @ W_i + b_i)             # for each hidden layer
    out = x @ W_last + b_last           # size 2 * action_size
    action = tanh(out[:action_size])    # NormalTanhDistribution.mode()
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".70")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # export needs no GPU

import numpy as np
import jax, jax.numpy as jp
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

import mujoco

from creature_env import ACTION_SIZE, OBS_LAYOUT, OBS_SIZE, XML_PATH

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DEST = HERE.parent / "Assets" / "Creature" / "policy.json"


def _find_layers(tree) -> list[tuple[np.ndarray, np.ndarray]]:
  """Pull ordered (kernel, bias) pairs out of a flax MLP param tree."""
  flat = {}

  def walk(node, path=()):
    if isinstance(node, dict):
      if "kernel" in node and "bias" in node:
        flat[path] = (np.asarray(node["kernel"]), np.asarray(node["bias"]))
        return
      for k, v in node.items():
        walk(v, path + (k,))

  walk(tree)
  if not flat:
    raise SystemExit("no kernel/bias pairs found in policy params")

  def order(path):
    name = path[-1]
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0

  return [flat[p] for p in sorted(flat, key=order)]


def _find_normalizer(params):
  """Locate the RunningStatisticsState in the saved params tuple."""
  for p in (params if isinstance(params, (tuple, list)) else [params]):
    if isinstance(p, running_statistics.RunningStatisticsState):
      return np.asarray(p.mean), np.asarray(p.std)
    if isinstance(p, dict) and "mean" in p and "std" in p:
      return np.asarray(p["mean"]), np.asarray(p["std"])
  return np.zeros(OBS_SIZE, np.float32), np.ones(OBS_SIZE, np.float32)


def _find_policy_params(params):
  """The policy tree is the first non-normalizer entry holding kernels."""
  cands = params if isinstance(params, (tuple, list)) else [params]
  for p in cands:
    if isinstance(p, running_statistics.RunningStatisticsState):
      continue
    try:
      layers = _find_layers(p)
    except SystemExit:
      continue
    # policy head emits 2 * action_size (mean and std); the value head emits 1
    if layers[-1][1].shape[-1] == 2 * ACTION_SIZE:
      return p
  raise SystemExit(
      f"could not find a policy head emitting {2*ACTION_SIZE} outputs")


def numpy_policy(obs, mean, std, layers):
  x = (obs - mean) / std
  for w, b in layers[:-1]:
    x = x @ w + b
    x = x / (1.0 + np.exp(-x))  # silu / swish
  w, b = layers[-1]
  out = x @ w + b
  return np.tanh(out[..., :ACTION_SIZE])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True, help="run name under mjx_training/runs/")
  ap.add_argument("--out", default=str(DEST))
  ap.add_argument("--tol", type=float, default=1e-4)
  args = ap.parse_args()

  run_dir = RUNS / args.run
  cfg = json.loads((run_dir / "config.json").read_text())
  params = model.load_params(str(run_dir / "params"))

  mean, std = _find_normalizer(params)
  policy_tree = _find_policy_params(params)
  layers = _find_layers(policy_tree)

  print(f"run: {run_dir}")
  print(f"normalizer: mean{mean.shape} std{std.shape}")
  print("layers: " + " -> ".join(
      [f"{layers[0][0].shape[0]}"] + [str(w.shape[1]) for w, _ in layers]))

  # --- verify the export against Brax's own inference function -----------
  nets = ppo_networks.make_ppo_networks(
      observation_size=OBS_SIZE,
      action_size=ACTION_SIZE,
      preprocess_observations_fn=running_statistics.normalize,
      policy_hidden_layer_sizes=tuple(cfg["policy_hidden"]),
      value_hidden_layer_sizes=tuple(cfg["value_hidden"]),
  )
  brax_policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)

  rng = np.random.default_rng(0)
  probe = rng.normal(size=(256, OBS_SIZE)).astype(np.float32)
  ref, _ = jax.jit(brax_policy)(jp.asarray(probe), jax.random.PRNGKey(0))
  ref = np.asarray(ref)
  ours = numpy_policy(probe, mean, std, layers)

  max_diff = float(np.abs(ref - ours).max())
  print(f"max |brax - exported| over 256 random observations: {max_diff:.3e}")
  if max_diff > args.tol:
    raise SystemExit(
        f"EXPORT REJECTED: mismatch {max_diff:.3e} > tol {args.tol:.0e}. "
        "The JSON would not reproduce the trained policy in Unity.")

  mj_model = mujoco.MjModel.from_xml_path(XML_PATH.as_posix())
  if mj_model.nkey < 1:
    raise SystemExit("creature.xml has no <keyframe>; Unity needs one to match training")
  home_joint_pos = [float(x) for x in mj_model.key_qpos[0][7:]]
  if len(home_joint_pos) != ACTION_SIZE:
    raise SystemExit(f"home pose has {len(home_joint_pos)} joints, expected {ACTION_SIZE}")
  print(f"home joint pose: {[round(v, 3) for v in home_joint_pos]}")

  # Flat, JsonUtility-friendly layout: kernels are row-major [inSize*outSize].
  payload = {
      "format": "mjx-unity-policy/1",
      "run": args.run,
      "obsSize": OBS_SIZE,
      "actionSize": ACTION_SIZE,
      "activation": "silu",
      "ctrlDt": cfg["ctrl_dt"],
      "nSubsteps": cfg["n_substeps"],
      "obsLayout": [f"{name}:{n}" for name, n in OBS_LAYOUT],
      # Unity's MJCF importer drops <keyframe> entirely, so the home pose
      # has to travel with the policy or the creature starts in a stance it
      # never trained from.
      "homeJointPos": home_joint_pos,
      "verifiedMaxAbsError": max_diff,
      "mean": mean.astype(np.float32).tolist(),
      "std": std.astype(np.float32).tolist(),
      "layers": [
          {
              "inSize": int(w.shape[0]),
              "outSize": int(w.shape[1]),
              "kernel": w.astype(np.float32).reshape(-1).tolist(),
              "bias": b.astype(np.float32).tolist(),
          }
          for w, b in layers
      ],
  }
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(payload))
  size_kb = out_path.stat().st_size / 1024
  print(f"wrote {out_path}  ({size_kb:,.0f} KB)")
  print("Unity: put a CreatureAgent on the imported creature root and press Play.")


if __name__ == "__main__":
  main()
