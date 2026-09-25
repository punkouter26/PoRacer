"""Policy network, checkpoint loading and ONNX export shared by every creature.

ONNX interface (WORM_SPEC "Export", QUAD_SPEC "Export"): input `obs` float[1, obs],
output `actions` float[1, act] = the deterministic actor mean, NOT clipped (the
runtime clips); the observation normaliser is baked into the graph.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HIDDEN = (128, 128, 128)
INIT_STD = 0.5
NORM_EPS = 1e-2


class EmpiricalNormalization(nn.Module):
    """RSL-RL's observation normaliser: running mean/var, (x - mean) / (std + 0.01),
    starts at mean 0 / var 1. Buffers travel with the checkpoint and into the ONNX."""

    def __init__(self, size: int, eps: float = NORM_EPS):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("std", torch.ones(size))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        self.count += n
        rate = n / self.count
        var_x = x.var(dim=0, unbiased=False)
        mean_x = x.mean(dim=0)
        delta = mean_x - self.mean
        self.mean += rate * delta
        self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))
        self.std.copy_(torch.sqrt(self.var))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def mlp(inp: int, out: int, hidden=HIDDEN) -> nn.Sequential:
    layers, last = [], inp
    for size in hidden:
        layers += [nn.Linear(last, size), nn.ELU()]
        last = size
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate 3x128 ELU actor and critic, one state-independent log-std per action."""

    def __init__(self, obs_size: int, action_size: int, init_std: float = INIT_STD):
        super().__init__()
        self.obs_size, self.action_size = obs_size, action_size
        self.norm = EmpiricalNormalization(obs_size)
        self.actor = mlp(obs_size, action_size)
        self.critic = mlp(obs_size, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), float(np.log(init_std))))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the mean), unclipped. Same as the ONNX graph."""
        return self.actor(self.norm(obs))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(self.norm(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean),
                                          validate_args=False)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.norm(obs)).squeeze(-1)


class OnnxPolicy(nn.Module):
    """obs [1, O] -> actions [1, A]: normaliser + actor mean, nothing else."""

    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.norm = copy.deepcopy(policy.norm).cpu()
        self.actor = copy.deepcopy(policy.actor).cpu()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs))


def save_checkpoint(path: Path, policy: ActorCritic, optimiser, iteration: int,
                    total_steps: int, seconds: float, extra: dict | None = None) -> Path:
    torch.save({"model": policy.state_dict(), "optimiser": optimiser.state_dict(),
                "iteration": iteration, "total_steps": total_steps, "wall_seconds": seconds,
                "obs_size": policy.obs_size, "action_size": policy.action_size,
                **(extra or {})}, path)
    return path


def load_policy(checkpoint: Path, obs_size: int | None = None, action_size: int | None = None,
                device="cpu") -> tuple[ActorCritic, dict]:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    obs_size = ck.get("obs_size", obs_size)
    action_size = ck.get("action_size", action_size)
    policy = ActorCritic(obs_size, action_size).to(device)
    policy.load_state_dict(ck["model"])
    policy.eval()
    return policy, ck


def export_onnx(policy: ActorCritic, path: Path, metadata: dict,
                parity_obs: torch.Tensor) -> float:
    """Write the ONNX (with string metadata) and return max |onnx - torch| over parity_obs."""
    import onnx
    import onnxruntime as ort

    path.parent.mkdir(parents=True, exist_ok=True)
    module = OnnxPolicy(policy).eval()
    torch.onnx.export(module, (torch.zeros(1, policy.obs_size),), str(path),
                      input_names=["obs"], output_names=["actions"],
                      opset_version=15, dynamo=False, dynamic_axes=None)
    model = onnx.load(str(path))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs_np = parity_obs.detach().cpu().float().numpy()
    with torch.no_grad():
        ref = module(torch.from_numpy(obs_np)).numpy()
    err = 0.0
    for row in range(obs_np.shape[0]):
        out = session.run(["actions"], {"obs": obs_np[row:row + 1]})[0]
        err = max(err, float(np.abs(out - ref[row:row + 1]).max()))
    return err


# ---------------------------------------------------------------- CLI: export a checkpoint
def pick_checkpoint(runs_dir: Path, run_id: str | None, pick: str,
                    best_from_fraction: float = 1.0 / 3.0) -> Path:
    """`final` = model_final.pt of the run (newest run if none named); `best` = the
    checkpoint with the highest mean episode return in checkpoints.jsonl among those saved
    after `best_from_fraction` of the run's iterations (default: the last 2/3), the one to
    report beside the final if a run collapses."""
    import json

    if run_id:
        run = runs_dir / run_id
    else:
        runs = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "checkpoints.jsonl").exists()]
        if not runs:
            raise FileNotFoundError(f"no finished run with checkpoints under {runs_dir}")
        run = max(runs, key=lambda p: (p / "checkpoints.jsonl").stat().st_mtime)
    if pick == "final":
        return run / "model_final.pt"
    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()
            if line.strip()]
    last_it = max((r["it"] for r in rows), default=0)
    rows = [r for r in rows if r.get("ep_return") is not None and (run / r["checkpoint"]).exists()
            and r["it"] >= best_from_fraction * last_it]
    if not rows:
        raise FileNotFoundError(f"no checkpoint with episode metrics in {run}")
    return run / max(rows, key=lambda r: r["ep_return"])["checkpoint"]


def export_main(name: str, runs_dir: Path, onnx_out: Path, obs_description: str,
                trainer_script: str, tool: str = "mujoco", argv=None) -> int:
    import argparse

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=f"Export a {name} checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit .pt")
    parser.add_argument("--run-id", type=str, default=None, help="run under the runs dir")
    parser.add_argument("--pick", choices=("final", "best"), default="final")
    parser.add_argument("--best-from", type=float, default=1.0 / 3.0,
                        help="best = highest mean episode return after this fraction of the run")
    parser.add_argument("--onnx-out", type=Path, default=onnx_out)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or pick_checkpoint(runs_dir, args.run_id, args.pick, args.best_from)
    policy, ck = load_policy(checkpoint)
    # Parity inputs shaped like real observations: the normaliser's own statistics.
    g = torch.Generator().manual_seed(0)
    parity = policy.norm.mean + policy.norm.std * torch.randn(512, policy.obs_size, generator=g)
    try:
        ck_rel = checkpoint.resolve().relative_to(repo).as_posix()
    except ValueError:
        ck_rel = str(checkpoint)
    metadata = {"tool": tool, "creature": name, "trainer": trainer_script,
                "run_id": checkpoint.parent.name, "stepsTrained": ck.get("total_steps"),
                "iterations": ck.get("iteration"),
                "wallMinutes": round(float(ck.get("wall_seconds", 0.0)) / 60.0, 3),
                "checkpoint": ck_rel, "obs": obs_description,
                "actions": f"{policy.action_size}, unclipped mean"}
    err = export_onnx(policy, args.onnx_out, metadata, parity)
    print(f"ONNX: {args.onnx_out}  from {ck_rel} (it {ck.get('iteration')}, "
          f"{ck.get('total_steps')} steps); onnx vs torch max |err| {err:.2e}")
    return 0
