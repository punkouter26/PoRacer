"""Quad pilot environment on MuJoCo Warp: training/quad/QUAD_SPEC.md on the creature template.

Body, action order, action scale, rest height, target speed and foot points are read
from training/quad/quad_rig.json (the single source of truth, written by build_quad.py).
Everything generic (step order, resets, randomisation, pushes, health guard) lives in
training/creature/env.py; this file only adds the quad's observation, reward, fall
termination and plausibility metrics.

Interpretations the Isaac Lab 3 side must copy (QUAD_SPEC + worm items 1-16, "segment 2"
read as "torso"; the torso is the root body `Quad_v01`, its origin is its CoM):
  Q1  Observation (36): [0:3] gravity direction in the torso frame, R^T (0,0,-1);
      [3:6] torso origin linear velocity, torso frame, x0.5; [6:9] torso angular velocity,
      torso frame, x0.25; [9:17] (q - rest) / 0.785398 in action order (rest = 0, so knees
      read up to +-1.33 at their +-60 deg limits); [17:25] joint velocities x0.1;
      [25:33] previous (clipped) action; [33:35] world +x rotated into the torso's full
      orientation, keep (x, y), renormalise; [35] target speed / 2 = 0.745 (constant).
  Q2  Target speed 1.49 m/s exactly (the spec's printed value; sqrt(0.25*9.81*0.9) = 1.486).
      speed = exp(-((v_x - 1.49) / 0.5)^2), v_x = torso origin world velocity along +x.
  Q3  heading = torso +x axis projected on the ground plane, normalised, world-x component.
      upright = torso z axis . world z (R[2,2]). lateral = |torso world velocity . y|.
  Q4  effort = mean over the 8 actuators of (actuator force of the LAST substep / 300)^2;
      actuator force = clip(kp (target - q), +-300), passive joint damping excluded.
  Q5  action rate = mean over 8 of (a_t - a_{t-1})^2, clipped actions; a_{-1} = 0 after reset.
  Q6  foot slip = sum over the 4 lower legs IN GROUND CONTACT of the horizontal (world x, y)
      speed of the foot point = the bottom end-cap centre of the lower capsule, (0, 0, -0.108)
      in the lower-leg frame (quad_rig.json footPointsLocal). In contact = a force-carrying
      floor contact (MuJoCo contact with dist < 0) in the last substep's collision pass.
      Isaac: ground contact force on the lower-leg body > 1 N.
  Q7  Fall = torso R[2,2] < 0.5 or torso origin z < 0.45 m, checked on the post-step state.
      It is TERMINAL (no bootstrap). The falling step keeps its ordinary reward (the spec
      has no fall penalty). If a world falls on its timeout step it counts as a fall.
  Q8  Reset: torso at (0, 0, 0.92), yaw U(+-45 deg) about the torso, each joint U(+-0.05)
      rad, zero velocity. Randomised at every reset: friction x U(0.85, 1.15), one factor
      per world on every geom incl. the floor; mass x U(0.9, 1.1) independently for ALL 9
      bodies (torso, 4 upper, 4 lower) with inertia scaled alike; kp x U(0.8, 1.2) per
      actuator (force limit, damping unchanged).
  Q9  Pushes (training only): per world, a timer U(10, 15) s drawn at reset and after each
      push; when it expires, +0.5 m/s is added to the torso's linear velocity in a
      uniformly random horizontal direction (fixed magnitude). Applied after resets and
      before the observation, like an Isaac Lab interval event. With 20 s episodes a full
      episode gets exactly one push, at 10-15 s (the next would fall at >= 20 s, i.e. at
      the timeout). Worlds that just reset or ended on this step are not pushed.
      Evaluation: no pushes, no randomisation.
  Q10 Health guard: non-finite or any |qvel| > 500 -> terminal, reward 0, excluded.
  Q12 Round 4: progress = clip(v_x, 0, 1.49) / 1.49 per second (weight 0.5). Feet air
      time (weight 1.0, per footfall, NOT x dt): contacts are tracked at EVERY 5 ms physics
      substep (Warp kernels inside the step's CUDA graph) with the same contact test as Q6;
      a touchdown is the first substep in contact after >= 1 airborne substep and pays
      (t_air - 0.25 s), t_air = airborne substeps x 0.005 s of the swing that just ended
      (negative for short swings). All 4 lower legs. The timers are zeroed at reset, so the
      2 cm spawn drop's first touchdowns pay about (0.06 - 0.25) each. The touchdowns of
      the 10 substeps of a policy step are summed into that step's reward.
  Q13 Round 5: (a) a fall is ALSO a floor contact (raw test of Q6, dist < 0) of the torso
      geom or of any of the 4 upper-leg geoms in ANY of the 10 substeps of the policy step,
      terminal like the other falls. (b) Feet contact is debounced: a foot's state flips only
      after 3 consecutive substeps (15 ms) of the opposite raw contact; air/stance timers
      follow the debounced state; touchdown = debounced flip to contact, paying
      min(t_air, 0.5) - 0.25 with t_air the debounced swing that just ended. The touchdowns
      of a policy step are summed and paid x 0.5 only if the POST-STEP torso v_x > 0.3 m/s
      (the same v_x as the speed term), not x dt. Foot slip keeps the raw last-substep contact.
  Q14 Round 6: alive = 1.0 per second on every policy step that does not end in a fall
      (any of the three fall rules), 0 on the falling step; health-guard steps earn 0 as
      always. Progress weight 1.0; speed tracking sigma 1.0: exp(-((v_x - 1.49) / 1.0)^2).
  Q15 Round 7: flight = (substeps of this policy step in which all 4 feet are in the
      DEBOUNCED airborne state, evaluated after each substep's debounce update) / 10,
      weight -1.0 per second (x dt). The debounced state starts as "in contact" at reset
      (so the 2 cm spawn drop shows up as flight once the 15 ms debounce has passed).
      Alive and termination share one fall test (Q7 + Q13a, any substep).
  Q16 Round 8 (MuJoCo side): reference trot. Per-world clock phi ~ U(0, 2 pi) at reset,
      phi += 2 pi 1.5 Hz x 0.05 s after the physics of each step (post-step clock; reset
      worlds get a fresh random phi); obs[36:38] = (sin phi, cos phi) -> obs [1, 38].
      q_ref: hip_i = 0.3 x 0.785398 x sin(phi + ph_i), knee_i = 0.5 x 0.785398 x
      max(0, -cos(phi + ph_i)), ph = (0, pi, pi, 0) for RL, FL, RR, FR (the prefab's diagonal
      pairs). gait_ref = exp(-mean_8 (q - q_ref)^2 / 0.3^2), weight 2.0 per second.
      contact_phase = mean over the 4 feet of [debounced contact state == (-cos(phi + ph_foot's
      hip) < 0)], weight 0.5 per second. Both on the post-step state.
  Q11 Round 3: policy at 20 Hz (decimation 10, physics dt 0.005 unchanged), reward x 0.05,
      episode 20 s = 400 policy steps, push timer in seconds (200-300 policy steps).
      vertical bounce = (torso origin world v_z)^2, weight -2.0 per second.
      PPO unchanged (gamma 0.99, 24 steps per env per rollout): the rollout now spans
      1.2 s of sim time instead of 0.48 s, and gamma 0.99 per step now means a ~5 s
      horizon instead of ~2 s.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
QUAD_DIR = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "training"))

from creature.env import CreatureConfig, CreatureEnv  # noqa: E402

RIG = json.loads((QUAD_DIR / "quad_rig.json").read_text(encoding="utf-8"))
MODEL_PATH = QUAD_DIR / "quad.xml"

# Round 8: the reference-gait clock [sin phi, cos phi] is appended -> 38 floats. This
# changes the ONNX interface (obs [1, 38]); Unity must run the same clock.
OBS_SIZE = 38
ACTION_ORDER = tuple(RIG["actionOrder"])
ACTION_SIZE = len(ACTION_ORDER)
ACTION_SCALE = float(RIG["actionScaleRad"])          # 0.785398
TARGET_SPEED = float(RIG["task"]["targetSpeed"])     # 1.49 m/s
TORSO = RIG["torso"]
LOWER_LEGS = tuple(RIG["lowerLegs"])
FOOT_POINTS = [RIG["footPointsLocal"][b] for b in LOWER_LEGS]
FOOT_GEOMS = tuple(RIG["footGeoms"])
# Round 5: a floor contact of the torso or of any upper leg is a fall (catches kneeling).
TERMINAL_CONTACT_GEOMS = tuple(g["name"] for g in RIG["geoms"]
                               if g["body"] == RIG["torso"] or g["body"].startswith("Upper_"))
MASS = float(RIG["totalMass"])
OBS_DESCRIPTION = "QUAD_SPEC.md 38-float observation (36 + gait clock sin, cos; round 8)"

# Round 8 reference trot (validated open-loop, see QUAD_SPEC round 8). Diagonal pairs from
# the prefab's coded gait (hip phases 0, pi, pi, 0 for RL, FL, RR, FR); the prefab's knee
# sine (a quarter period behind the hip) is replaced by a one-sided lift during the forward
# swing: knee = amp * max(0, -cos(psi_hip)). Stance is expected while -cos(psi_hip) < 0,
# i.e. while the hip swings the foot backward.
_BUG_AGENT = json.loads((REPO / "training" / "bugs" / "Quad_v01_rig.json").read_text())["agent"]
_HIP_IDX = [0, 2, 4, 6]
_HIP_PHASES = [float(_BUG_AGENT["gaitPhases"][i]) for i in _HIP_IDX]
REFERENCE_GAIT = {
    "freq": 1.5,
    "phases": [ph for ph in _HIP_PHASES for _ in (0, 1)],   # knee uses its own hip's phase
    "amps": [0.3, 0.5] * 4,                                 # hip, knee (action units x 45 deg)
    "shapes": ["sin", "lift"] * 4,
    "foot_hips": _HIP_IDX,                                  # lowerLegs order RL, FL, RR, FR
    "stance_signal": "neg_cos",
}
GAIT_REF_SIGMA = 0.3     # rad

# Observation scales.
SCALE_LIN_VEL = 0.5
SCALE_ANG_VEL = 0.25
SCALE_JOINT_VEL = 0.1

# Reward weights per second (x dt 0.02): QUAD_SPEC.md "Task and reward". The ONE place
# they live; compute_reward() sums weight * term over this dict. Effort, action rate and
# foot slip were raised 10x / 25x / 10x after the first smoke run learned a 50 Hz buzz-hop.
REWARD_WEIGHTS = {
    "alive": 1.0,            # round 6: 1 on every step that did not end in a fall, else 0
    "speed_track": 2.0,
    "progress": 1.0,         # clip(v_x, 0, 1.49) / 1.49 (round 4: 0.5; round 6: 1.0)
    "heading": 0.1,          # round 4, was 0.3
    "upright": 0.2,          # round 4, was 0.5
    "lateral": -0.5,
    "effort": -0.2,          # was -0.02
    "action_rate": -0.2,     # round 4, was -0.5 (and -0.02 before that)
    "foot_slip": -1.0,       # was -0.1
    "vertical_bounce": -2.0, # (torso world v_z)^2, round 3: stops the bounding gait
    "flight": -1.0,          # round 7: fraction of the step's substeps with all feet up (debounced)
    "gait_ref": 2.0,         # round 8: exp(-mean_i (q_i - q_ref_i)^2 / 0.3^2)
    "contact_phase": 0.5,    # round 8: share of feet whose debounced contact matches the reference
    "feet_air_time": 0.5,    # PER FOOTFALL: sum of (min(t_air, 0.5) - 0.25) at debounced
                             # touchdowns, only while v_x > 0.3 (round 5; round 4: 1.0, uncapped)
}
# Terms paid per event, NOT multiplied by the policy dt.
PER_EVENT_TERMS = frozenset({"feet_air_time"})
AIR_TIME_TARGET = 0.25   # s
AIR_TIME_CAP = 0.5       # s (round 5)
AIR_TIME_MIN_SPEED = 0.3 # m/s: feet air time is paid only while torso v_x > this (round 5)
CONTACT_DEBOUNCE = 3     # substeps (15 ms) before a foot's contact state flips (round 5)
SPEED_SIGMA = 1.0        # round 6 (was 0.5)

FALL_UP = 0.5
FALL_HEIGHT = 0.45
PUSH_SPEED = 0.5
PUSH_INTERVAL = (10.0, 15.0)

# Contact budget per world. A quad lying on its back: box-plane 4 + 8 capsules x 2 +
# leg-leg / leg-torso pairs; the trainer aborts if the pool overflows. Pyramidal
# condim 3 = 4 rows per contact, + 8 joint limits.
NCONMAX = 48
SETTLE_SECONDS = 0.5     # plausibility: foot impacts after the spawn drop only
NJMAX = 208


def quad_config() -> CreatureConfig:
    return CreatureConfig(
        mjcf=MODEL_PATH, action_order=ACTION_ORDER, reference_body=TORSO, obs_size=OBS_SIZE,
        action_scale=ACTION_SCALE, spawn_height=float(RIG["spawnRootHeight"]),
        rest_pose=[float(RIG["restPose"][j]) for j in ACTION_ORDER],
        decimation=int(RIG["physics"]["decimation"]),                     # 10 -> 20 Hz
        episode_seconds=float(RIG["task"]["episodeSeconds"]),             # 20 s = 400 steps
        reset_yaw=math.radians(45.0), reset_joint_noise=0.05,
        friction_range=(0.85, 1.15), mass_range=(0.9, 1.1), kp_range=(0.8, 1.2),
        randomized_bodies=None, push_speed=PUSH_SPEED, push_interval=PUSH_INTERVAL,
        nconmax=NCONMAX, njmax=NJMAX, divergence_qvel=500.0, speed_term="speed_x",
        tracked_contact_geoms=FOOT_GEOMS, air_time_target=AIR_TIME_TARGET,
        air_time_cap=AIR_TIME_CAP, contact_debounce=CONTACT_DEBOUNCE, contact_reset_state=1,
        terminal_contact_geoms=TERMINAL_CONTACT_GEOMS, reference_gait=REFERENCE_GAIT)


class QuadEnv(CreatureEnv):
    term_names = ("alive", "speed_x", "speed_track", "progress", "heading", "upright", "lateral",
                  "effort", "action_rate", "foot_slip", "vertical_bounce", "flight",
                  "gait_ref", "contact_phase",
                  "feet_air_time",
                  "touchdowns", "feet_in_contact", "torso_height", "torque_abs")

    def __init__(self, num_worlds: int, device: str = "cuda:0", seed: int = 0,
                 randomize: bool = True, random_initial_episode: bool = True,
                 pushes: bool | None = None):
        self._fallen = None
        super().__init__(quad_config(), num_worlds, device=device, seed=seed, randomize=randomize,
                         random_initial_episode=random_initial_episode, pushes=pushes)
        self.lower_bodies = torch.tensor([self.body_id(b) for b in LOWER_LEGS], device=self.device)
        self.foot_points = torch.tensor(FOOT_POINTS, dtype=torch.float32, device=self.device)
        self.foot_geoms = [self.geom_id(g) for g in FOOT_GEOMS]

    # ---- observation -------------------------------------------------------------
    def compute_observation(self) -> torch.Tensor:
        rot, v, w, _ = self.ref_state()
        rot_t = rot.transpose(1, 2)
        obs = torch.empty(self.num_worlds, OBS_SIZE, device=self.device)
        obs[:, 0:3] = -rot[:, 2, :]                                     # R^T (0, 0, -1)
        obs[:, 3:6] = torch.bmm(rot_t, v.unsqueeze(2)).squeeze(2) * SCALE_LIN_VEL
        obs[:, 6:9] = torch.bmm(rot_t, w.unsqueeze(2)).squeeze(2) * SCALE_ANG_VEL
        obs[:, 9:17] = (self.joint_pos() - self.rest_pose) / ACTION_SCALE
        obs[:, 17:25] = self.joint_vel() * SCALE_JOINT_VEL
        obs[:, 25:33] = self.action
        goal = rot[:, 0, 0:2]                                           # (R^T [1,0,0]).xy
        obs[:, 33:35] = goal / goal.norm(dim=1, keepdim=True).clamp_min(1e-6)
        obs[:, 35] = TARGET_SPEED * 0.5
        obs[:, 36] = torch.sin(self.gait_phase)
        obs[:, 37] = torch.cos(self.gait_phase)
        return obs

    # ---- reward ------------------------------------------------------------------
    def foot_state(self):
        """(in contact (N,4) bool, horizontal foot-point speed (N,4))."""
        contact = self.floor_contacts(self.foot_geoms)
        v_foot, _ = self.point_velocity(self.lower_bodies, self.foot_points)
        return contact, v_foot[..., 0:2].norm(dim=2)

    def compute_reward(self):
        rot, v, w, pos = self.ref_state()
        speed_x = v[:, 0]
        speed_track = torch.exp(-((speed_x - TARGET_SPEED) / SPEED_SIGMA) ** 2)
        progress = speed_x.clamp(0.0, TARGET_SPEED) / TARGET_SPEED
        fx, fy = rot[:, 0, 0], rot[:, 1, 0]
        heading = fx / torch.sqrt(fx * fx + fy * fy).clamp_min(1e-6)
        upright = rot[:, 2, 2]
        lateral = v[:, 1].abs()
        torque = self.actuator_force
        effort = (torque / self.force_limits).pow(2).mean(dim=1)
        action_rate = (self.action - self.prev_action).pow(2).mean(dim=1)
        contact, foot_speed = self.foot_state()
        foot_slip = (foot_speed * contact.float()).sum(dim=1)
        self._fallen = (upright < FALL_UP) | (pos[:, 2] < FALL_HEIGHT)
        # Round 6: no alive bonus on a falling step, whichever fall rule fired (the contact
        # rule is applied by the base env from the same substep tracker read here).
        fell = self._fallen | (self.terminal_tracker.any_raw > 0.0)
        tracker = self.contact_tracker
        terms = {"alive": (~fell).float(), "speed_x": speed_x, "speed_track": speed_track,
                 "progress": progress,
                 "heading": heading,
                 "upright": upright, "lateral": lateral, "effort": effort,
                 "action_rate": action_rate, "foot_slip": foot_slip,
                 "vertical_bounce": v[:, 2] * v[:, 2],
                 "flight": tracker.flight_substeps / float(self.cfg.decimation),
                 "gait_ref": torch.exp(-(self.joint_pos() - self.reference_pose()).pow(2).mean(dim=1)
                                       / GAIT_REF_SIGMA ** 2),
                 "contact_phase": (tracker.state.bool() == self.reference_stance()).float().mean(dim=1),
                 "feet_air_time": tracker.touchdown_sum * (speed_x > AIR_TIME_MIN_SPEED).float(),
                 "touchdowns": tracker.touchdown_count.clone(),
                 "feet_in_contact": contact.float().sum(dim=1), "torso_height": pos[:, 2],
                 "torque_abs": torque.abs().mean(dim=1)}
        per_second = torch.zeros_like(speed_x)
        per_event = torch.zeros_like(speed_x)
        for name, weight in REWARD_WEIGHTS.items():
            if name in PER_EVENT_TERMS:
                per_event = per_event + weight * terms[name]
            else:
                per_second = per_second + weight * terms[name]
        return self.dt * per_second + per_event, terms

    def task_terminated(self) -> torch.Tensor:
        return self._fallen            # set by compute_reward on the same post-step state


class QuadPlausibility:
    """Rule I metrics over each world's first episode (only while it is still running).
    Contact metrics (airtime, feet down, stance duration, slip in stance, impacts) are
    sampled at EVERY physics substep (5 ms) via substep(); the rest once per policy step."""

    def __init__(self, env: QuadEnv):
        self.env = env

    def reset(self, env: QuadEnv) -> None:
        n, dev = env.num_worlds, env.device
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.active = torch.ones(n, dtype=torch.bool, device=dev)
        self.steps = z()
        self.joint_speed_sum, self.joint_peak = z(), z()
        self.saturated = z()
        self.height_sum, self.height_peak = z(), z()
        self.height_min = torch.full((n,), 1e9, device=dev)
        self.up_sum = z()
        self.up_min = torch.ones(n, device=dev)
        self.ang_peak, self.roll_sum, self.pitch_sum, self.vz_sq_sum = z(), z(), z(), z()
        self.power_sum, self.speed_sum, self.lateral_sum = z(), z(), z()
        self.action_rate_sum = z()
        # substep-level contact statistics
        k = len(env.foot_geoms)
        self.substeps = z()
        self.contact_sum_db = z()
        self.slip_sum, self.contact_sum, self.airborne = z(), z(), z()
        self.impact_peak, self.impact_peak_spawn = z(), z()
        self.run_peak = torch.zeros(n, k, device=dev)
        self.footfall_peak_sum = z()
        self.settle_steps = int(round(SETTLE_SECONDS / env.dt))
        self.in_stance = torch.zeros(n, k, dtype=torch.bool, device=dev)
        self.stance_run = torch.zeros(n, k, device=dev)
        self.stance_sum, self.stance_count = z(), z()
        self.db_stance_sum, self.db_liftoffs, self.db_touchdowns = z(), z(), z()
        self.db_airborne = z()
        self.db_run = torch.zeros(n, k, device=dev)
        self.db_prev = torch.ones(n, k, dtype=torch.bool, device=dev)
        self.db_run_sum, self.db_run_sq, self.db_run_long, self.db_run_n = z(), z(), z(), z()
        self.flight_steps_sum = z()
        self.illegal = z()

    def substep(self, env: QuadEnv) -> None:
        a = self.active.float()
        contact, foot_speed = env.foot_state()
        cf = contact.float()
        self.substeps += a
        self.contact_sum_db += a * env.contact_tracker.state.float().sum(dim=1)
        self.db_airborne += a * (env.contact_tracker.state.sum(dim=1) == 0).float()
        st = env.contact_tracker.state.bool()
        lift = self.db_prev & ~st & self.active.unsqueeze(1) & (env.episode_step >= self.settle_steps).unsqueeze(1)
        d = self.db_run * lift.float() * float(env.mjm.opt.timestep)
        self.db_run_sum += d.sum(dim=1)
        self.db_run_sq += (d * d).sum(dim=1)
        self.db_run_long += (d * (d >= 0.15).float()).sum(dim=1)
        self.db_run_n += lift.float().sum(dim=1)
        self.db_run = torch.where(st, self.db_run + 1.0, torch.zeros_like(self.db_run))
        self.db_prev = st
        self.slip_sum += a * (foot_speed * cf).sum(dim=1)
        self.contact_sum += a * cf.sum(dim=1)
        self.airborne += a * (cf.sum(dim=1) == 0).float()
        force = env.floor_contact_forces(env.foot_geoms)
        settled = (env.episode_step >= self.settle_steps).float()   # skip the 2 cm spawn drop
        self.impact_peak_spawn = torch.maximum(self.impact_peak_spawn, a * force.amax(dim=1))
        self.impact_peak = torch.maximum(self.impact_peak, a * settled * force.amax(dim=1))
        # stance runs: a footfall ends when a foot that was in contact lifts off
        lift = self.in_stance & ~contact & self.active.unsqueeze(1)
        self.stance_sum += (self.stance_run * lift.float()).sum(dim=1)
        self.stance_count += lift.float().sum(dim=1)
        self.footfall_peak_sum += (self.run_peak * lift.float()).sum(dim=1)
        self.stance_run = torch.where(contact, self.stance_run + 1.0, torch.zeros_like(self.stance_run))
        self.run_peak = torch.where(contact, torch.maximum(self.run_peak, force),
                                    torch.zeros_like(self.run_peak))
        self.in_stance = contact

    def accumulate(self, env: QuadEnv, info: dict, active: torch.Tensor) -> None:
        a = active.float()
        rot, v, w, pos = env.ref_state()
        w_b = torch.bmm(rot.transpose(1, 2), w.unsqueeze(2)).squeeze(2)
        qd = env.joint_vel().abs()
        force = env.actuator_force
        self.steps += a
        self.joint_speed_sum += a * qd.mean(dim=1)
        self.joint_peak = torch.maximum(self.joint_peak, a * qd.amax(dim=1))
        self.saturated += a * (force.abs() >= env.force_limits * 0.999).any(dim=1).float()
        self.height_sum += a * pos[:, 2]
        self.height_peak = torch.maximum(self.height_peak, a * pos[:, 2])
        self.height_min = torch.where(active, torch.minimum(self.height_min, pos[:, 2]), self.height_min)
        self.up_sum += a * rot[:, 2, 2]
        self.up_min = torch.where(active, torch.minimum(self.up_min, rot[:, 2, 2]), self.up_min)
        self.ang_peak = torch.maximum(self.ang_peak, a * w.norm(dim=1))
        self.roll_sum += a * w_b[:, 0].abs()
        self.pitch_sum += a * w_b[:, 1].abs()
        self.vz_sq_sum += a * info["vertical_bounce"]
        self.power_sum += a * (force * env.joint_vel()).abs().sum(dim=1)
        self.speed_sum += a * info["speed_x"]
        self.lateral_sum += a * info["lateral"]
        self.action_rate_sum += a * info["action_rate"]
        tr = env.contact_tracker
        self.db_stance_sum += a * tr.stance_sum
        self.db_liftoffs += a * tr.liftoff_count
        self.db_touchdowns += a * tr.touchdown_count
        self.illegal += a * info["illegal_contact"].float()
        self.flight_steps_sum += a * info["flight"]
        # worlds whose first episode ended on this step stop contributing to substep stats
        self.active = active & ~(info["terminated"] | info["timeouts"] | info["diverged"])

    def summary(self, env: QuadEnv, kept: torch.Tensor) -> dict:
        k = kept
        steps = self.steps[k].clamp_min(1.0)
        total = self.steps[k].sum().clamp_min(1.0)
        sub_total = self.substeps[k].sum().clamp_min(1.0)
        sub_dt = float(env.mjm.opt.timestep)

        def per_step(x):                      # time-weighted mean over all kept policy steps
            return float(x[k].sum() / total)

        def per_substep(x):
            return float(x[k].sum() / sub_total)

        mean_speed = per_step(self.speed_sum)
        power = per_step(self.power_sum)
        footfalls = float(self.stance_count[k].sum())
        seconds = float(sub_total) * sub_dt
        impact = float(self.impact_peak[k].max())
        return {
            "meanJointSpeed": per_step(self.joint_speed_sum),
            "peakJointSpeed": float(self.joint_peak[k].max()),
            "fractionTimeAnyActuatorSaturated": per_step(self.saturated),
            "meanTorsoHeight": per_step(self.height_sum),
            "minTorsoHeight": float(self.height_min[k].min()),
            "peakTorsoHeight": float(self.height_peak[k].max()),
            "meanTorsoUp": per_step(self.up_sum),
            "minTorsoUp": float(self.up_min[k].min()),
            "peakTorsoAngularSpeed": float(self.ang_peak[k].max()),
            "meanAbsRollRate": per_step(self.roll_sum),
            "meanAbsPitchRate": per_step(self.pitch_sum),
            "rmsTorsoVerticalSpeed": per_step(self.vz_sq_sum) ** 0.5,
            "meanFootSlipSpeedInStance": float(self.slip_sum[k].sum()
                                               / self.contact_sum[k].sum().clamp_min(1.0)),
            "meanFeetInContact": per_substep(self.contact_sum),
            "airborneFraction": per_substep(self.airborne),
            "meanStanceDurationS": (float(self.stance_sum[k].sum()) / footfalls * sub_dt
                                    if footfalls > 0 else None),
            "footfallsPerSecond": footfalls / seconds if seconds > 0 else None,
            "meanStanceDurationDebouncedS": (float(self.db_stance_sum[k].sum())
                                             / float(self.db_liftoffs[k].sum())
                                             if float(self.db_liftoffs[k].sum()) > 0 else None),
            "footfallsPerSecondDebounced": float(self.db_touchdowns[k].sum()) / seconds
            if seconds > 0 else None,
            "airborneFractionDebounced": per_substep(self.db_airborne),
            "stanceTimeWeightedMeanDebouncedS": (float(self.db_run_sq[k].sum()) / float(self.db_run_sum[k].sum())
                                                 if float(self.db_run_sum[k].sum()) > 0 else None),
            "shareOfStanceTimeInStancesOver0.15s": (float(self.db_run_long[k].sum()) / float(self.db_run_sum[k].sum())
                                                     if float(self.db_run_sum[k].sum()) > 0 else None),
            "dutyFactorDebounced": per_substep(self.contact_sum_db) / 4.0,
            "flightFractionRewardTerm": per_step(self.flight_steps_sum),
            "episodesEndedByTorsoOrThighContact": int((self.illegal[k] > 0).sum()),
            "peakFootImpactN": impact,
            "peakFootImpactBodyWeights": impact / (MASS * 9.81),
            "peakFootImpactNote": f"max over feet and 5 ms substeps of one foot's floor normal force, "
                                  f"excluding the first {SETTLE_SECONDS:g} s after reset (the 2 cm "
                                  f"spawn drop alone gives ~4.5 body weights); standing = 0.25",
            "peakFootImpactBodyWeightsInclSpawn": float(self.impact_peak_spawn[k].max()) / (MASS * 9.81),
            "meanPeakForcePerFootfallBodyWeights": (float(self.footfall_peak_sum[k].sum()) / footfalls
                                                    / (MASS * 9.81) if footfalls > 0 else None),
            "contactSampling": f"every physics substep ({sub_dt * 1000:g} ms)",
            "meanForwardSpeed": mean_speed,
            "meanLateralSpeed": per_step(self.lateral_sum),
            "meanActionRate": per_step(self.action_rate_sum),
            "meanAbsMechanicalPowerW": power,
            "costOfTransport": power / (MASS * 9.81 * mean_speed) if mean_speed > 0.05 else None,
            "meanEpisodeSeconds": float(steps.mean() * env.dt),
            "forceLimit": env.force_limit,
        }
