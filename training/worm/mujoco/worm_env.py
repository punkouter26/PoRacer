"""Batched MuJoCo Warp environment for Worm5 -- method C of the worm shoot-out.

The contract is training/worm/WORM_SPEC.md. The Isaac Lab trainer implements the
same contract, so every number here is the spec's number; where the spec was
silent, the choice made is written down next to the code AND in the header of
this file so the Isaac side can copy it.

Everything runs on the GPU: MJWarp integrates `num_worlds` copies of worm.xml and
the observation, reward and reset logic are torch ops over zero-copy views of
MJWarp's arrays (wp.to_torch). Nothing is read back to the host inside step().

Step order matches Isaac Lab's ManagerBasedRLEnv, so the two trainers see the
same sequence:
    clip action -> write targets -> 4 physics steps -> refresh kinematics
    -> episode counter += 1 -> timeouts -> reward (post-step state)
    -> reset timed-out worlds -> observation (post-reset for reset worlds)

Resolved ambiguities (the Isaac side must match these):
  A1  Segment-2 velocity = velocity of seg2's body origin, which IS its centre of
      mass (the capsule is centred on the body frame, ipos = 0). World frame for
      the reward terms, rotated into seg2's frame for the observation.
  A2  "Goal direction in B, horizontal" = world +x rotated into seg2's FULL
      orientation (g_B = R_B^T [1, 0, 0], i.e. the first row of R_B), keep its
      (x, y) and renormalise to unit length. The heading reward is separate:
      seg2's local +x axis in world, projected onto the horizontal plane and
      normalised; reward = its world-x component (cos of the angle to +x).
  A3  "Previous action" (obs 25..32) and the action-rate term use the CLIPPED
      action actually applied. Both are zero after a reset.
  A4  Effort torque = the actuator (servo) force of the LAST of the 4 physics
      substeps: clip(kp * (target - q), +-F), F = the XML forcerange (6 N*m).
      The joint damping (2.0 N*m*s/rad in the XML)
      is a passive joint term in MuJoCo, NOT part of the actuator force, and is
      not capped by the force limit. Effort = mean((torque / F)^2).
  A5  Reset pose: the free-joint root is segment 0 (the head). Head at
      (0, 0, 0.05); yaw ~ U(-45 deg, +45 deg) rotates the whole straight worm about
      the head; each of the 8 joints gets U(-0.05, +0.05) rad; all velocities 0.
  A6  Domain randomisation, re-sampled at every reset, independent per world:
        friction: ONE factor f ~ U(0.85, 1.15) per world. MuJoCo combines two
                  geoms' friction with max(), so the factor is applied to the
                  sliding friction of EVERY geom including the floor: the
                  effective worm-floor (and worm-worm) coefficient is 0.9 * f.
                  MuJoCo has one coefficient, so static = dynamic = 0.9 * f.
        mass:     independent factor ~ U(0.9, 1.1) for each of the 5 SEGMENTS;
                  the 4 intermediate 0.1 kg links are not randomised. The body
                  inertia is scaled by the same factor (as Isaac Lab's
                  randomize_rigid_body_mass with recompute_inertia=True).
        kp:       independent factor ~ U(0.8, 1.2) for each of the 8 actuators;
                  the force limit and the joint damping are unchanged.
  A7  Episode length 1000 policy steps; timeout when the counter reaches 1000.
      Timeouts are the ONLY ending (no fall termination). Training starts every
      world at a random point of its first episode (Isaac Lab's
      init_at_random_ep_len=True) so the 4096 worlds do not all reset together.
  A8  Simulator-health guard (not a task rule): a world whose state is non-finite
      or has any |qvel| > 500 (rad/s or m/s) is ended as TERMINAL (not a timeout,
      no bootstrap), gets reward 0 on that step and is reset. It is counted and
      logged; a healthy policy never trips it.

Per-world field support in MJWarp 3.12 (checked, not assumed): geom_friction,
body_mass, body_inertia, actuator_gainprm and actuator_biasprm are all batched
per world via put_model(batch_sizes=...). The constraint-regularisation weights
body_invweight0 / dof_invweight0 are NOT recomputed after a mass change (neither
does CPU MuJoCo unless mj_setConst is called); that affects only contact
softness scaling by at most ~10 % and is left at nominal.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "worm.xml"

OBS_SIZE = 35
ACTION_SIZE = 8
ACTION_ORDER = ("j0_pitch", "j0_yaw", "j1_pitch", "j1_yaw",
                "j2_pitch", "j2_yaw", "j3_pitch", "j3_yaw")
SEGMENTS = ("seg0", "seg1", "seg2", "seg3", "seg4")
REFERENCE_BODY = "seg2"

JOINT_RANGE = 0.785398            # rad, 45 deg; action 1.0 -> 45 deg
DECIMATION = 4
EPISODE_STEPS = 1000              # 20 s at 0.02 s

# Observation scales (spec table).
SCALE_LIN_VEL = 0.5
SCALE_ANG_VEL = 0.25
SCALE_JOINT_POS = 1.0 / JOINT_RANGE
SCALE_JOINT_VEL = 0.1

# Reward weights, per second (multiplied by dt = 0.02).
W_PROGRESS = 1.0
W_HEADING = 0.1
W_ACTION_RATE = -0.02
W_EFFORT = -0.01                 # of mean((torque / force limit)^2); limit read from the XML
W_ROLL_RATE = -0.1               # abs(seg2 angular velocity about its own long axis), rad/s
W_BELLY_DOWN = 0.2               # seg2 z axis . world z
W_LATERAL = -0.5               # spec update: was -0.1 (Isaac smoke run sidewound 6 m sideways)
PROGRESS_CLIP = (-1.0, 2.0)

# Reset and randomisation (spec).
SPAWN_Z = 0.05
RESET_YAW = math.radians(45.0)
RESET_JOINT_NOISE = 0.05
FRICTION_RANGE = (0.85, 1.15)
MASS_RANGE = (0.9, 1.1)
KP_RANGE = (0.8, 1.2)

# Contact budget. Worst case: 5 capsule-plane pairs x 2 + 6 non-adjacent
# capsule-capsule pairs x 2 = 22 contacts. Pyramidal condim 3 = 4 rows each,
# plus 8 joint limits -> 96 rows. The trainer checks for overflow every
# iteration: dropped contacts would let the worm sink through the floor.
NCONMAX = 32
NJMAX = 128

# Health guard (see step()): any |qvel| above this, rad/s or m/s, is a runaway.
# ~3x the largest spike reproduced in float64 CPU MuJoCo (180 rad/s), ~10x the
# largest seen in 24 closed-loop CPU episodes (55 rad/s).
DIVERGENCE_QVEL = 500.0


def _uniform(generator, shape, low, high, device):
    return low + torch.rand(shape, generator=generator, device=device) * (high - low)


class WormEnv:
    """Vectorised Worm5. step() returns (obs, reward, done, timeouts, info)."""

    def __init__(self, num_worlds: int, device: str = "cuda:0", seed: int = 0,
                 randomize: bool = True, random_initial_episode: bool = True):
        self.num_worlds = num_worlds
        self.device = torch.device(device)
        self.randomize = randomize

        # Warp must run on torch's stream, or a torch write to ctrl can race the
        # physics kernels that read it. A dedicated (non-legacy) stream also lets
        # the substep loop be captured as a CUDA graph.
        wp.config.quiet = True
        wp.init()
        self.stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.stream)
        wp.set_stream(wp.stream_from_torch(self.stream))

        self.mjm = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.dt = float(self.mjm.opt.timestep) * DECIMATION      # 0.02 s
        # Force limit for the effort term, from the XML (not hard-coded): every
        # actuator must share one symmetric forcerange.
        franges = self.mjm.actuator_forcerange
        if not (self.mjm.actuator_forcelimited.all() and (franges[:, 1] == franges[0, 1]).all()
                and (franges[:, 0] == -franges[:, 1]).all()):
            raise RuntimeError(f"worm.xml actuators need one symmetric forcerange, got {franges}")
        self.force_limit = float(franges[0, 1])
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)

        batched = ("geom_friction", "body_mass", "body_inertia",
                   "actuator_gainprm", "actuator_biasprm")
        self.wm = mjw.put_model(self.mjm, batch_sizes={k: num_worlds for k in batched})
        self.wm.opt.warn_overflow = 0   # checked explicitly by the trainer
        self.wd = mjw.put_data(self.mjm, mjd, nworld=num_worlds, nconmax=NCONMAX, njmax=NJMAX)

        def body(name):
            return mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, name)

        self.ref_body = body(REFERENCE_BODY)
        self.root_of_ref = int(self.mjm.body_rootid[self.ref_body])
        self.segment_bodies = torch.tensor([body(n) for n in SEGMENTS], device=self.device)
        qadr, dadr = [], []
        for name in ACTION_ORDER:
            jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr.append(int(self.mjm.jnt_qposadr[jid]))
            dadr.append(int(self.mjm.jnt_dofadr[jid]))
            aid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid == len(qadr) - 1, "actuator order must equal the action order"
        self.qpos_addr = torch.tensor(qadr, device=self.device)
        self.dof_addr = torch.tensor(dadr, device=self.device)

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
        self.action = torch.zeros(n, ACTION_SIZE, device=self.device)
        self.prev_action = torch.zeros_like(self.action)
        self.episode_step = torch.zeros(n, device=self.device, dtype=torch.long)
        self.start_x = torch.zeros(n, device=self.device)
        self.ep_return = torch.zeros(n, device=self.device)
        self.ep_speed_sum = torch.zeros(n, device=self.device)
        self.ep_len = torch.zeros(n, device=self.device)

        # Capture the policy step's physics (4 substeps + kinematics refresh) as
        # one CUDA graph: for a model this small, per-kernel launch overhead is
        # most of the step time. Warm-up first so every kernel is compiled; the
        # warm-up's state change is undone by the reset below.
        self._graph = None
        self._physics_eager()
        try:
            with wp.ScopedCapture() as capture:
                self._physics_eager()
            self._graph = capture.graph
        except Exception as exc:  # pragma: no cover - driver without graph support
            print(f"[worm_env] CUDA graph capture unavailable ({exc}); stepping eagerly")

        self.reset(torch.arange(n, device=self.device))
        if random_initial_episode:
            self.episode_step[:] = torch.randint(0, EPISODE_STEPS, (n,), generator=self.generator,
                                                 device=self.device)

    # ---- physics ------------------------------------------------------------
    def _physics_eager(self) -> None:
        for _ in range(DECIMATION):
            mjw.step(self.wm, self.wd)
        self._refresh()

    def _physics(self) -> None:
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._physics_eager()

    def _refresh(self) -> None:
        """Kinematics + com velocities for the CURRENT qpos/qvel. After mjw.step the
        body poses in d still describe the state before the last integration."""
        mjw.kinematics(self.wm, self.wd)
        mjw.com_pos(self.wm, self.wd)
        mjw.com_vel(self.wm, self.wd)

    # ---- reference-body state ---------------------------------------------
    def ref_state(self):
        """(R, v_world, w_world, pos) of segment 2. R: body->world (N, 3, 3)."""
        b = self.ref_body
        rot = self.xmat[:, b]
        w = self.cvel[:, b, 0:3]
        v_com = self.cvel[:, b, 3:6]                          # at subtree_com of the root
        offset = self.xpos[:, b] - self.subtree_com[:, self.root_of_ref]
        v = v_com + torch.cross(w, offset, dim=1)             # at seg2's origin = its CoM
        return rot, v, w, self.xpos[:, b]

    # ---- observation ------------------------------------------------------
    def observation(self) -> torch.Tensor:
        rot, v, w, _ = self.ref_state()
        rot_t = rot.transpose(1, 2)
        obs = torch.empty(self.num_worlds, OBS_SIZE, device=self.device)
        obs[:, 0:3] = -rot[:, 2, :]                                          # R^T (0,0,-1)
        obs[:, 3:6] = torch.bmm(rot_t, v.unsqueeze(2)).squeeze(2) * SCALE_LIN_VEL
        obs[:, 6:9] = torch.bmm(rot_t, w.unsqueeze(2)).squeeze(2) * SCALE_ANG_VEL
        obs[:, 9:17] = self.qpos[:, self.qpos_addr] * SCALE_JOINT_POS
        obs[:, 17:25] = self.qvel[:, self.dof_addr] * SCALE_JOINT_VEL
        obs[:, 25:33] = self.action
        goal = rot[:, 0, 0:2]                              # (R^T [1,0,0]).xy = row 0 of R
        obs[:, 33:35] = goal / goal.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- reset -------------------------------------------------------------
    def reset(self, index: torch.Tensor) -> None:
        if index.numel() == 0:
            return
        n = index.numel()
        g, dev = self.generator, self.device
        qpos = self.nominal_qpos.unsqueeze(0).repeat(n, 1)
        yaw = _uniform(g, (n,), -RESET_YAW, RESET_YAW, dev)
        qpos[:, 0:3] = torch.tensor([0.0, 0.0, SPAWN_Z], device=dev)
        qpos[:, 3] = torch.cos(0.5 * yaw)
        qpos[:, 4] = 0.0
        qpos[:, 5] = 0.0
        qpos[:, 6] = torch.sin(0.5 * yaw)
        qpos[:, self.qpos_addr] = _uniform(g, (n, ACTION_SIZE), -RESET_JOINT_NOISE,
                                           RESET_JOINT_NOISE, dev)
        self.qpos[index] = qpos
        self.qvel[index] = 0.0
        self.qacc_warmstart[index] = 0.0
        self.ctrl[index] = 0.0
        self.action[index] = 0.0
        self.prev_action[index] = 0.0
        self.episode_step[index] = 0
        self.ep_return[index] = 0.0
        self.ep_speed_sum[index] = 0.0
        self.ep_len[index] = 0.0
        if self.randomize:
            self._randomise(index)
        self._refresh()
        self.start_x[index] = self.xpos[index, self.ref_body, 0]

    def _randomise(self, index: torch.Tensor) -> None:
        n = index.numel()
        g, dev = self.generator, self.device
        f = _uniform(g, (n,), *FRICTION_RANGE, dev)
        friction = self.nominal_friction[index].clone()
        friction[:, :, 0] *= f.unsqueeze(1)                    # sliding coefficient, all geoms
        self.friction[index] = friction

        seg = self.segment_bodies
        k = _uniform(g, (n, seg.numel()), *MASS_RANGE, dev)
        mass = self.nominal_mass[index].clone()
        inertia = self.nominal_inertia[index].clone()
        mass[:, seg] *= k
        inertia[:, seg] *= k.unsqueeze(-1)
        self.mass[index] = mass
        self.inertia[index] = inertia

        s = _uniform(g, (n, ACTION_SIZE), *KP_RANGE, dev)
        gain = self.nominal_gain[index].clone()
        bias = self.nominal_bias[index].clone()
        gain[:, :, 0] *= s                                     # kp
        bias[:, :, 1] *= s                                     # -kp
        self.gain[index] = gain
        self.bias[index] = bias

    # ---- step --------------------------------------------------------------
    def step(self, action: torch.Tensor):
        action = torch.clamp(action, -1.0, 1.0)
        self.prev_action = self.action
        self.action = action
        self.ctrl[:] = action * JOINT_RANGE

        self._physics()                       # includes the kinematics refresh
        self.episode_step += 1

        reward, terms = self._reward()
        # Simulator-health guard, NOT a task termination. The spec body can whip
        # itself into ~100-180 rad/s head roll (reproduced in CPU MuJoCo, float64,
        # from the same state), and in MJWarp's float32 a few such worlds then run
        # away to inf/NaN. A world that is non-finite or past DIVERGENCE_QVEL on
        # any DoF is ended as TERMINAL (no bootstrap), earns 0 on that step and is
        # reset; its terms are zeroed so one runaway cannot poison the batch stats.
        _, v_ref, _, pos_ref = self.ref_state()
        finite = (torch.isfinite(self.qpos).all(dim=1) & torch.isfinite(self.qvel).all(dim=1)
                  & torch.isfinite(pos_ref).all(dim=1) & torch.isfinite(v_ref).all(dim=1)
                  & torch.isfinite(reward))
        healthy = finite & (torch.nan_to_num(self.qvel, nan=0.0).abs().amax(dim=1)
                            <= DIVERGENCE_QVEL)
        timeouts = (self.episode_step >= EPISODE_STEPS) & healthy
        diverged = ~healthy
        done = timeouts | diverged
        reward = torch.where(healthy, reward, torch.zeros_like(reward))
        for key, value in terms.items():
            terms[key] = torch.where(healthy, torch.nan_to_num(value), torch.zeros_like(value))

        self.ep_return += reward
        self.ep_speed_sum += terms["speed_x"]
        self.ep_len += 1
        pos = pos_ref
        info = dict(terms)
        info["diverged"] = diverged
        # Per-episode statistics of the worlds that end on this step (pre-reset).
        info["ep_return"] = self.ep_return.clone()
        info["ep_speed"] = self.ep_speed_sum / self.ep_len.clamp_min(1.0)
        info["ep_distance"] = torch.nan_to_num(pos[:, 0] - self.start_x)
        info["ep_length"] = self.ep_len.clone()

        idx = done.nonzero(as_tuple=True)[0]      # one host sync per step, unavoidable
        if idx.numel() > 0:
            self.reset(idx)
        return self.observation(), reward, done, timeouts, info

    def _reward(self):
        rot, v, w, _ = self.ref_state()
        speed_x = v[:, 0]
        progress = speed_x.clamp(*PROGRESS_CLIP)
        fx, fy = rot[:, 0, 0], rot[:, 1, 0]
        heading = fx / torch.sqrt(fx * fx + fy * fy).clamp_min(1e-6)
        action_rate = (self.action - self.prev_action).pow(2).mean(dim=1)
        torque = self.actuator_force
        effort = (torque / self.force_limit).pow(2).mean(dim=1)
        roll_rate = (w * rot[:, :, 0]).sum(dim=1).abs()      # (R^T w).x, B-frame x
        belly_down = rot[:, 2, 2]                            # (B z axis) . world z
        lateral = v[:, 1].abs()
        reward = self.dt * (W_PROGRESS * progress + W_HEADING * heading
                            + W_ACTION_RATE * action_rate + W_EFFORT * effort
                            + W_ROLL_RATE * roll_rate + W_BELLY_DOWN * belly_down
                            + W_LATERAL * lateral)
        terms = {"speed_x": speed_x, "progress": progress, "heading": heading,
                 "action_rate": action_rate, "effort": effort, "lateral": lateral,
                 "roll_rate": roll_rate, "belly_down": belly_down,
                 "torque_abs": torque.abs().mean(dim=1)}
        return reward, terms

    # ---- diagnostics -------------------------------------------------------
    def contact_peak(self) -> int:
        return int(self.nacon.max().item())

    @property
    def naconmax(self) -> int:
        return int(self.wd.naconmax)
