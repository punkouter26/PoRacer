"""Batched MuJoCo Warp creature environment (the generic half of training/worm/mujoco/worm_env.py).

Everything runs on the GPU: MJWarp integrates `num_worlds` copies of the MJCF, and
observation, reward, termination and reset logic are torch ops over zero-copy views of
MJWarp's arrays (wp.to_torch). The only host sync per step is the index of worlds to
reset.

Step order (WORM_SPEC "Details resolved" 13, Isaac Lab ManagerBasedRLEnv order):
    clip action -> ctrl = rest + action * scale -> `decimation` physics substeps
    -> refresh kinematics -> episode counter += 1 -> reward (post-step state)
    -> task termination -> health guard -> timeouts -> reset finished worlds
    -> pushes (worlds whose push timer elapsed) -> observation (post-reset/post-push)

A creature subclasses CreatureEnv and implements:
    compute_observation() -> (N, obs_size)
    compute_reward()      -> (reward (N,), terms {name: (N,)})   reward already x dt
    task_terminated()     -> (N,) bool, a TERMINAL ending (no bootstrap); default: never
and may override on_reset(index). `term_names` lists the terms the trainer logs;
the term named by CreatureConfig.speed_term feeds the episode-speed statistics.

Shared rules carried over from the worm (binding on both trainers of a creature):
  * Actions clipped to [-1, 1] before the env; the previous-action observation and the
    action-rate penalty use the clipped values (item 5).
  * Timeouts bootstrap; task terminations and health-guard endings do not. A world that
    both times out and terminates on the same step counts as terminated.
  * Health guard (item 12): non-finite state or any |qvel| > divergence_qvel ends the
    world as TERMINAL, reward 0 on that step, terms zeroed, counted in info["diverged"].
  * Reset (item 6): root at (x0, y0, spawn_height), yaw U(+-reset_yaw) about the root,
    joints = rest + U(+-reset_joint_noise), all velocities 0, previous action 0.
  * Randomisation (items 7, 14, 15), redrawn at every reset, per world: one friction
    factor on every geom incl. the floor (MuJoCo combines by max); independent mass
    factor per randomised body with inertia scaled alike; independent kp factor per
    actuator (force limit and joint damping unchanged).
  * Training starts every world at a random point of its first episode (item 8).
  * Pushes (optional): every U(push_interval) seconds of episode time, add push_speed
    m/s in a uniformly random horizontal direction to the root's linear velocity. The
    timer is redrawn at every reset and after every push.

MJWarp 3.12 batches geom_friction, body_mass, body_inertia, actuator_gainprm and
actuator_biasprm per world via put_model(batch_sizes=...); checked at start-up.
body_invweight0 / dof_invweight0 are not recomputed after a mass change (neither does
CPU MuJoCo without mj_setConst); contact-softness scaling moves by <= ~10 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp


@dataclass
class CreatureConfig:
    mjcf: Path
    action_order: Sequence[str]            # joint == actuator names, in action order
    reference_body: str                    # the body the task is written in (worm seg2, quad torso)
    obs_size: int
    action_scale: float                    # rad per unit action
    spawn_height: float                    # root body z at reset
    rest_pose: Sequence[float] | None = None   # joint targets at action 0 (default all 0)
    decimation: int = 4                    # physics substeps per policy step (per creature)
    episode_steps: int = 1000              # policy steps; ignored when episode_seconds is set
    episode_seconds: float | None = None   # if set, episode_steps = round(seconds / policy dt)
    reset_yaw: float = math.radians(45.0)
    reset_joint_noise: float = 0.05
    friction_range: tuple[float, float] = (0.85, 1.15)
    mass_range: tuple[float, float] = (0.9, 1.1)
    kp_range: tuple[float, float] = (0.8, 1.2)
    randomized_bodies: Sequence[str] | None = None   # None = every body of the creature
    friction_randomized_geoms: Sequence[str] | None = None   # None = every geom incl. the floor
    push_speed: float = 0.0                # m/s, 0 disables pushes
    push_interval: tuple[float, float] = (10.0, 15.0)   # s of episode time
    floor_geom: str = "floor"
    nconmax: int = 32                      # per world
    njmax: int = 128                       # per world
    divergence_qvel: float = 500.0
    speed_term: str = "speed_x"
    tracked_contact_geoms: Sequence[str] = ()   # geoms (feet) tracked every substep, in the graph
    air_time_target: float = 0.25               # s, for the tracker's touchdown sum
    air_time_cap: float = 1e9                   # s, t_air is clamped to this before - target
    contact_debounce: int = 1                   # substeps a new contact state must persist
    contact_reset_state: int = 0                # debounced state at reset (1 = in contact)
    # Reference gait (round 8): q_ref_i = rest_i + scale * amp_i * shape_i(phi + phase_i), with a
    # per-world clock phi (random at reset, += 2 pi freq dt per policy step). Keys: freq (Hz),
    # phases, amps (per actuator, action units), scale (rad per unit; default action_scale),
    # shapes (per actuator: "sin" = sin, "lift" = max(0, -cos); default all "sin"),
    # foot_hips (actuator index of each tracked foot's hip) and stance_signal ("sin" or
    # "neg_cos"): a foot should be in stance while that signal of its hip's phase is < 0.
    reference_gait: dict | None = None
    terminal_contact_geoms: Sequence[str] = ()  # a raw floor contact of any of these in ANY
                                                # substep of a policy step ends it as terminal
    extra: dict = field(default_factory=dict)


def uniform(generator, shape, low, high, device):
    return low + torch.rand(shape, generator=generator, device=device) * (high - low)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, (w, x, y, z), batched over the leading dim."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


class CreatureEnv:
    """Vectorised creature. step() returns (obs, reward, done, timeouts, info)."""

    term_names: tuple[str, ...] = ()

    def __init__(self, cfg: CreatureConfig, num_worlds: int, device: str = "cuda:0",
                 seed: int = 0, randomize: bool = True, random_initial_episode: bool = True,
                 pushes: bool | None = None):
        self.cfg = cfg
        self.num_worlds = num_worlds
        self.device = torch.device(device)
        self.randomize = randomize
        self.pushes = (randomize if pushes is None else pushes) and cfg.push_speed > 0.0
        self.obs_size = cfg.obs_size
        self.action_size = len(cfg.action_order)
        self.episode_steps = cfg.episode_steps
        self.substep_callback = None           # eval diagnostics: called after EVERY substep

        # Warp must run on torch's stream, or a torch write to ctrl can race the physics
        # kernels that read it. A dedicated (non-legacy) stream also allows graph capture.
        wp.config.quiet = True
        wp.init()
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        wp.set_stream(wp.stream_from_torch(self.stream))

        self.mjm = mujoco.MjModel.from_xml_path(str(cfg.mjcf))
        self.dt = float(self.mjm.opt.timestep) * cfg.decimation
        if cfg.episode_seconds is not None:
            self.episode_steps = int(round(cfg.episode_seconds / self.dt))
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)

        batched = ("geom_friction", "body_mass", "body_inertia",
                   "actuator_gainprm", "actuator_biasprm")
        self.wm = mjw.put_model(self.mjm, batch_sizes={k: num_worlds for k in batched})
        self.wm.opt.warn_overflow = 0   # contact overflow is checked explicitly by the trainer
        self.wd = mjw.put_data(self.mjm, mjd, nworld=num_worlds,
                               nconmax=cfg.nconmax, njmax=cfg.njmax)

        self.ref_body = self.body_id(cfg.reference_body)
        root = int(self.mjm.body_rootid[self.ref_body])
        self.root_body = root
        jnt = int(self.mjm.body_jntadr[root])
        if jnt < 0 or self.mjm.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("the creature's root body needs a <freejoint>")
        self.root_qadr = int(self.mjm.jnt_qposadr[jnt])
        self.root_dadr = int(self.mjm.jnt_dofadr[jnt])

        qadr, dadr = [], []
        for index, name in enumerate(cfg.action_order):
            jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid != index:
                raise RuntimeError(f"joint/actuator {name!r} must exist and be actuator #{index}")
            qadr.append(int(self.mjm.jnt_qposadr[jid]))
            dadr.append(int(self.mjm.jnt_dofadr[jid]))
        self.qpos_addr = torch.tensor(qadr, device=self.device)
        self.dof_addr = torch.tensor(dadr, device=self.device)
        rest = cfg.rest_pose if cfg.rest_pose is not None else [0.0] * self.action_size
        self.rest_pose = torch.tensor(rest, dtype=torch.float32, device=self.device)

        franges = self.mjm.actuator_forcerange
        if not (self.mjm.actuator_forcelimited.all() and (franges[:, 0] == -franges[:, 1]).all()):
            raise RuntimeError(f"actuators need symmetric force ranges, got {franges}")
        self.force_limits = torch.tensor(franges[:, 1], dtype=torch.float32, device=self.device)
        self.force_limit = float(franges[:, 1].max())

        names = cfg.randomized_bodies
        bodies = ([self.body_id(n) for n in names] if names is not None
                  else [b for b in range(1, self.mjm.nbody) if self.mjm.body_rootid[b] == root])
        self.randomized_bodies = torch.tensor(bodies, device=self.device)
        self.floor_geom = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, cfg.floor_geom)
        names = cfg.friction_randomized_geoms
        self.friction_geoms = (torch.tensor([self.geom_id(g) for g in names], device=self.device)
                               if names is not None else None)
        from .contacts import FloorContactTracker
        self.contact_tracker = None
        if cfg.tracked_contact_geoms:
            self.contact_tracker = FloorContactTracker(
                self.mjm, self.wd, [self.geom_id(g) for g in cfg.tracked_contact_geoms],
                self.floor_geom, str(self.device), air_time_target=cfg.air_time_target,
                air_time_cap=cfg.air_time_cap, debounce=cfg.contact_debounce,
                reset_state=cfg.contact_reset_state)
        self.terminal_tracker = None
        if cfg.terminal_contact_geoms:
            self.terminal_tracker = FloorContactTracker(
                self.mjm, self.wd, [self.geom_id(g) for g in cfg.terminal_contact_geoms],
                self.floor_geom, str(self.device))
        self._trackers = [t for t in (self.contact_tracker, self.terminal_tracker) if t is not None]

        # Zero-copy views. MJWarp never reallocates these arrays.
        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.qacc_warmstart = wp.to_torch(self.wd.qacc_warmstart)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xmat = wp.to_torch(self.wd.xmat)
        self.cvel = wp.to_torch(self.wd.cvel)
        self.subtree_com = wp.to_torch(self.wd.subtree_com)
        self.actuator_force = wp.to_torch(self.wd.actuator_force)
        self.nacon = wp.to_torch(self.wd.nacon)
        self.contact_geom = wp.to_torch(self.wd.contact.geom)
        self.contact_dist = wp.to_torch(self.wd.contact.dist)
        self.contact_world = wp.to_torch(self.wd.contact.worldid)
        self._contact_index = torch.arange(self.contact_dist.shape[0], device=self.device)
        self._contact_tables: dict[tuple[int, ...], torch.Tensor] = {}

        self.friction = wp.to_torch(self.wm.geom_friction)
        self.mass = wp.to_torch(self.wm.body_mass)
        self.inertia = wp.to_torch(self.wm.body_inertia)
        self.gain = wp.to_torch(self.wm.actuator_gainprm)
        self.bias = wp.to_torch(self.wm.actuator_biasprm)
        for name, t in (("geom_friction", self.friction), ("body_mass", self.mass),
                        ("body_inertia", self.inertia), ("actuator_gainprm", self.gain),
                        ("actuator_biasprm", self.bias)):
            if t.shape[0] != num_worlds:
                raise RuntimeError(f"MJWarp did not batch {name} per world "
                                   f"(shape {tuple(t.shape)}); randomisation would be global")
        self.nominal_friction = self.friction.clone()
        self.nominal_mass = self.mass.clone()
        self.nominal_inertia = self.inertia.clone()
        self.nominal_gain = self.gain.clone()
        self.nominal_bias = self.bias.clone()
        self.nominal_qpos = self.qpos[0].clone()

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        n = num_worlds
        self.action = torch.zeros(n, self.action_size, device=self.device)
        self.prev_action = torch.zeros_like(self.action)
        self.episode_step = torch.zeros(n, device=self.device, dtype=torch.long)
        self.push_timer = torch.zeros(n, device=self.device, dtype=torch.long)
        self.gait_phase = None
        gait = cfg.reference_gait
        if gait is not None:
            a = self.action_size
            self.gait_phase = torch.zeros(n, device=self.device)
            self.gait_omega = 2.0 * math.pi * float(gait["freq"])
            self.gait_offsets = torch.tensor(gait["phases"], dtype=torch.float32, device=self.device)
            self.gait_amps = torch.tensor(gait["amps"], dtype=torch.float32, device=self.device)
            self.gait_scale = float(gait.get("scale", cfg.action_scale))
            shapes = gait.get("shapes", ["sin"] * a)
            self.gait_lift = torch.tensor([s_ == "lift" for s_ in shapes], device=self.device)
            self.gait_foot_hips = torch.tensor(gait.get("foot_hips", []), dtype=torch.long,
                                               device=self.device)
            self.gait_stance_signal = gait.get("stance_signal", "sin")
        self.start_x = torch.zeros(n, device=self.device)
        self.ep_return = torch.zeros(n, device=self.device)
        self.ep_speed_sum = torch.zeros(n, device=self.device)
        self.ep_len = torch.zeros(n, device=self.device)

        # Capture one policy step's physics (substeps + kinematics refresh) as a CUDA
        # graph; warm up first so every kernel is compiled. reset() undoes the warm-up.
        self._graph = None
        self._physics_eager()
        try:
            with wp.ScopedCapture() as capture:
                self._physics_eager()
            self._graph = capture.graph
        except Exception as exc:  # pragma: no cover - driver without graph support
            print(f"[creature_env] CUDA graph capture unavailable ({exc}); stepping eagerly")

        self.reset(torch.arange(n, device=self.device))
        if random_initial_episode:
            self.episode_step[:] = torch.randint(0, self.episode_steps, (n,),
                                                 generator=self.generator, device=self.device)

    # ---- lookups -----------------------------------------------------------------
    def body_id(self, name: str) -> int:
        b = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise KeyError(f"no body {name!r} in {self.cfg.mjcf.name}")
        return b

    def geom_id(self, name: str) -> int:
        g = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise KeyError(f"no geom {name!r} in {self.cfg.mjcf.name}")
        return g

    # ---- physics -----------------------------------------------------------------
    def _substep(self) -> None:
        mjw.step(self.wm, self.wd)
        for tracker in self._trackers:
            tracker.after_substep()

    def _physics_eager(self) -> None:
        for tracker in self._trackers:
            tracker.begin_step()
        for _ in range(self.cfg.decimation):
            self._substep()
        self._refresh()

    def _physics(self) -> None:
        if self.substep_callback is not None:
            # Diagnostics path (evaluation): step eagerly and let the callback sample each
            # substep. After mjw.step, poses, velocities, contacts and constraint forces all
            # describe the same (pre-integration) state of that substep.
            for tracker in self._trackers:
                tracker.begin_step()
            for _ in range(self.cfg.decimation):
                self._substep()
                self.substep_callback(self)
            self._refresh()
        elif self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._physics_eager()

    def _refresh(self) -> None:
        """Kinematics + com velocities for the CURRENT qpos/qvel (after mjw.step the body
        poses still describe the state before the last integration)."""
        mjw.kinematics(self.wm, self.wd)
        mjw.com_pos(self.wm, self.wd)
        mjw.com_vel(self.wm, self.wd)

    # ---- state helpers -----------------------------------------------------------
    def body_state(self, body: int):
        """(R body->world (N,3,3), v of the body ORIGIN in world (N,3), w world (N,3), pos (N,3))."""
        rot = self.xmat[:, body]
        w = self.cvel[:, body, 0:3]
        v_com = self.cvel[:, body, 3:6]                     # at subtree_com of the root
        offset = self.xpos[:, body] - self.subtree_com[:, int(self.mjm.body_rootid[body])]
        return rot, v_com + torch.cross(w, offset, dim=1), w, self.xpos[:, body]

    def ref_state(self):
        return self.body_state(self.ref_body)

    def point_velocity(self, bodies: torch.Tensor, local_points: torch.Tensor):
        """World velocity (N,k,3) and position (N,k,3) of body-fixed points.
        bodies (k,) long; local_points (k,3) in each body's frame. Bodies must share a root."""
        rot = self.xmat[:, bodies]                                   # (N,k,3,3)
        w = self.cvel[:, bodies, 0:3]
        v_com = self.cvel[:, bodies, 3:6]
        root = int(self.mjm.body_rootid[int(bodies[0])])
        world_offset = torch.einsum("nkij,kj->nki", rot, local_points)
        point = self.xpos[:, bodies] + world_offset
        v = v_com + torch.cross(w, point - self.subtree_com[:, root].unsqueeze(1), dim=2)
        return v, point

    def floor_contacts(self, geoms: Sequence[int]) -> torch.Tensor:
        """(N, k) bool: geom k touches the floor with a force-carrying contact (dist < 0)
        after the last substep's collision detection. No host sync."""
        key = tuple(int(g) for g in geoms)
        table = self._contact_tables.get(key)
        if table is None:
            table = torch.full((self.mjm.ngeom,), -1, dtype=torch.long, device=self.device)
            table[torch.tensor(key, device=self.device)] = torch.arange(len(key), device=self.device)
            self._contact_tables[key] = table
        k = len(key)
        g = self.contact_geom.long()
        other = torch.where(g[:, 0] == self.floor_geom, g[:, 1],
                            torch.where(g[:, 1] == self.floor_geom, g[:, 0], torch.full_like(g[:, 0], -1)))
        slot = table[other.clamp_min(0)]
        ok = ((self._contact_index < self.nacon[0]) & (other >= 0) & (slot >= 0)
              & (self.contact_dist < 0.0))
        flat = torch.where(ok, self.contact_world.long() * k + slot,
                           torch.full_like(slot, self.num_worlds * k))
        out = torch.zeros(self.num_worlds * k + 1, dtype=torch.bool, device=self.device)
        out.index_fill_(0, flat, True)
        return out[:-1].view(self.num_worlds, k)

    def floor_contact_forces(self, geoms: Sequence[int]) -> torch.Tensor:
        """(N, k) floor NORMAL force (N) on each geom, summed over its floor contacts, from
        the last substep's constraint solve (mjw.contact_force, contact frame). For
        diagnostics (e.g. peak foot impact); one extra kernel, no host sync."""
        if not hasattr(self, "_contact_force_buf"):
            size = self.contact_dist.shape[0]
            self._contact_ids_wp = wp.array(list(range(size)), dtype=int, device=str(self.device))
            self._contact_force_wp = wp.zeros(size, dtype=wp.spatial_vector, device=str(self.device))
            self._contact_force_buf = wp.to_torch(self._contact_force_wp)
        self._contact_force_buf.zero_()
        mjw.contact_force(self.wm, self.wd, self._contact_ids_wp, False, self._contact_force_wp)
        key = tuple(int(g) for g in geoms)
        self.floor_contacts(key)                           # builds the lookup table
        table = self._contact_tables[key]
        k = len(key)
        g = self.contact_geom.long()
        other = torch.where(g[:, 0] == self.floor_geom, g[:, 1],
                            torch.where(g[:, 1] == self.floor_geom, g[:, 0], torch.full_like(g[:, 0], -1)))
        slot = table[other.clamp_min(0)]
        ok = (self._contact_index < self.nacon[0]) & (other >= 0) & (slot >= 0)
        flat = torch.where(ok, self.contact_world.long() * k + slot,
                           torch.full_like(slot, self.num_worlds * k))
        out = torch.zeros(self.num_worlds * k + 1, device=self.device)
        out.index_add_(0, flat, torch.where(ok, self._contact_force_buf[:, 0], 0.0))
        return out[:-1].view(self.num_worlds, k)

    def reference_pose(self) -> torch.Tensor:
        """(N, A) reference joint angles at the current gait phase."""
        psi = self.gait_phase.unsqueeze(1) + self.gait_offsets
        shape = torch.where(self.gait_lift, torch.clamp(-torch.cos(psi), min=0.0), torch.sin(psi))
        return self.rest_pose + self.gait_scale * self.gait_amps * shape

    def reference_stance(self) -> torch.Tensor:
        """(N, feet) bool: the reference says this foot should be on the ground."""
        psi = self.gait_phase.unsqueeze(1) + self.gait_offsets[self.gait_foot_hips]
        signal = torch.sin(psi) if self.gait_stance_signal == "sin" else -torch.cos(psi)
        return signal < 0.0

    def joint_pos(self) -> torch.Tensor:
        return self.qpos[:, self.qpos_addr]

    def joint_vel(self) -> torch.Tensor:
        return self.qvel[:, self.dof_addr]

    # ---- hooks -------------------------------------------------------------------
    def compute_observation(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    def task_terminated(self) -> torch.Tensor:
        return torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)

    def on_reset(self, index: torch.Tensor) -> None:
        pass

    def observation(self) -> torch.Tensor:
        return torch.nan_to_num(self.compute_observation(), nan=0.0, posinf=0.0, neginf=0.0)

    # ---- reset -------------------------------------------------------------------
    def reset(self, index: torch.Tensor) -> None:
        if index.numel() == 0:
            return
        n = index.numel()
        g, dev, cfg = self.generator, self.device, self.cfg
        qpos = self.nominal_qpos.unsqueeze(0).repeat(n, 1)
        a = self.root_qadr
        yaw = uniform(g, (n,), -cfg.reset_yaw, cfg.reset_yaw, dev)
        q_yaw = torch.stack((torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw),
                             torch.sin(0.5 * yaw)), dim=1)
        qpos[:, a + 2] = cfg.spawn_height
        qpos[:, a + 3:a + 7] = quat_mul(q_yaw, qpos[:, a + 3:a + 7])
        qpos[:, self.qpos_addr] = self.rest_pose + uniform(
            g, (n, self.action_size), -cfg.reset_joint_noise, cfg.reset_joint_noise, dev)
        self.qpos[index] = qpos
        self.qvel[index] = 0.0
        self.qacc_warmstart[index] = 0.0
        self.ctrl[index] = self.rest_pose
        self.action[index] = 0.0
        self.prev_action[index] = 0.0
        self.episode_step[index] = 0
        self.ep_return[index] = 0.0
        self.ep_speed_sum[index] = 0.0
        self.ep_len[index] = 0.0
        if self.pushes:
            self.push_timer[index] = self._draw_push_steps(n)
        if self.gait_phase is not None:
            self.gait_phase[index] = uniform(self.generator, (n,), 0.0, 2.0 * math.pi, self.device)
        if self.randomize:
            self._randomise(index)
        for tracker in self._trackers:
            tracker.reset(index)
        self.on_reset(index)
        self._refresh()
        self.start_x[index] = self.xpos[index, self.ref_body, 0]

    def _draw_push_steps(self, n: int) -> torch.Tensor:
        lo, hi = self.cfg.push_interval
        return torch.round(uniform(self.generator, (n,), lo, hi, self.device) / self.dt).long()

    def _randomise(self, index: torch.Tensor) -> None:
        n = index.numel()
        g, dev, cfg = self.generator, self.device, self.cfg
        f = uniform(g, (n,), *cfg.friction_range, dev)
        friction = self.nominal_friction[index].clone()
        if self.friction_geoms is None:
            friction[:, :, 0] *= f.unsqueeze(1)             # sliding coefficient, every geom
        else:
            friction[:, self.friction_geoms, 0] *= f.unsqueeze(1)   # only the listed geoms
        self.friction[index] = friction

        bodies = self.randomized_bodies
        k = uniform(g, (n, bodies.numel()), *cfg.mass_range, dev)
        mass = self.nominal_mass[index].clone()
        inertia = self.nominal_inertia[index].clone()
        mass[:, bodies] *= k
        inertia[:, bodies] *= k.unsqueeze(-1)
        self.mass[index] = mass
        self.inertia[index] = inertia

        s = uniform(g, (n, self.action_size), *cfg.kp_range, dev)
        gain = self.nominal_gain[index].clone()
        bias = self.nominal_bias[index].clone()
        gain[:, :, 0] *= s                                   # kp
        bias[:, :, 1] *= s                                   # -kp
        self.gain[index] = gain
        self.bias[index] = bias

    # ---- step --------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        action = torch.clamp(action, -1.0, 1.0)
        self.prev_action = self.action
        self.action = action
        self.ctrl[:] = self.rest_pose + action * self.cfg.action_scale

        self._physics()                       # includes the kinematics refresh
        self.episode_step += 1
        if self.gait_phase is not None:          # the clock of the post-step state
            self.gait_phase = torch.remainder(self.gait_phase + self.gait_omega * self.dt,
                                              2.0 * math.pi)

        reward, terms = self.compute_reward()
        terminated = self.task_terminated()
        if self.terminal_tracker is not None:
            illegal_contact = self.terminal_tracker.any_raw > 0.0
            terminated = terminated | illegal_contact
        else:
            illegal_contact = torch.zeros_like(terminated)
        _, v_ref, _, pos_ref = self.ref_state()
        finite = (torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1)
                  & torch.isfinite(pos_ref).all(dim=1) & torch.isfinite(v_ref).all(dim=1)
                  & torch.isfinite(reward))
        healthy = finite & (torch.nan_to_num(self.qvel, nan=0.0).abs().amax(dim=1)
                            <= self.cfg.divergence_qvel)
        terminated = terminated & healthy
        timeouts = (self.episode_step >= self.episode_steps) & healthy & ~terminated
        diverged = ~healthy
        done = timeouts | terminated | diverged
        reward = torch.where(healthy, reward, torch.zeros_like(reward))
        for key, value in terms.items():
            terms[key] = torch.where(healthy, torch.nan_to_num(value), torch.zeros_like(value))

        self.ep_return += reward
        self.ep_speed_sum += terms[self.cfg.speed_term]
        self.ep_len += 1
        info = dict(terms)
        info["diverged"] = diverged
        info["terminated"] = terminated
        info["illegal_contact"] = illegal_contact & healthy
        info["timeouts"] = timeouts
        # Per-episode statistics of the worlds that end on this step (pre-reset).
        info["ep_return"] = self.ep_return.clone()
        info["ep_speed"] = self.ep_speed_sum / self.ep_len.clamp_min(1.0)
        info["ep_distance"] = torch.nan_to_num(pos_ref[:, 0] - self.start_x)
        info["ep_length"] = self.ep_len.clone()

        push = None
        if self.pushes:
            self.push_timer -= 1
            push = (self.push_timer <= 0) & ~done
        info["pushed"] = push if push is not None else torch.zeros_like(done)

        idx = done.nonzero(as_tuple=True)[0]      # the one host sync per step
        if idx.numel() > 0:
            self.reset(idx)
        if push is not None:
            self._apply_pushes(push)
        return self.observation(), reward, done, timeouts, info

    def _apply_pushes(self, push: torch.Tensor) -> None:
        """Add push_speed m/s in a random horizontal direction to the root's linear velocity
        (MuJoCo free-joint qvel[0:3] is the world-frame linear velocity). No host sync."""
        n = self.num_worlds
        phi = uniform(self.generator, (n,), 0.0, 2.0 * math.pi, self.device)
        dv = self.cfg.push_speed * torch.stack((torch.cos(phi), torch.sin(phi)), dim=1)
        d = self.root_dadr
        self.qvel[:, d:d + 2] += dv * push.unsqueeze(1).float()
        self.push_timer = torch.where(push, self._draw_push_steps(n), self.push_timer)
        mjw.com_vel(self.wm, self.wd)          # observation reads cvel

    # ---- diagnostics -------------------------------------------------------------
    def contact_peak(self) -> int:
        return int(self.nacon.max().item())

    @property
    def naconmax(self) -> int:
        return int(self.wd.naconmax)
