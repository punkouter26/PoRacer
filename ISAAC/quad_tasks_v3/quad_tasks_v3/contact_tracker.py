"""Per-SUBSTEP foot contact tracking for the quad (QUAD_SPEC round 4 "feet air time" + substep metrics).

The policy runs at 20 Hz over 10 physics substeps of 5 ms. A foot tap shorter than 50 ms would be invisible
to a once-per-policy-step contact test, so touchdowns are detected at every substep, as on the MuJoCo side.

How it is wired into Newton's CUDA-graphed decimation loop:

* :class:`QuadArticulation` (the quad's Newton Articulation) creates a :class:`FootContactTracker` on
  PHYSICS_READY and registers :meth:`FootContactTracker.graph_step` with
  ``NewtonManager.register_post_actuator_callback``. Newton runs those callbacks inside the captured graph
  once per decimation iteration, right before that iteration's solver substep. So iteration i (i = 1..9)
  sees MuJoCo-Warp's contact buffer written by substep i of this policy step; iteration 0 sees the last
  substep of the PREVIOUS policy step and is skipped (``counter == 0``).
* The 10th substep's buffer is processed eagerly by the ``feet_air_time`` reward term after the step
  (:meth:`process_last_substep`), which then zeroes the counter. Every substep is therefore processed
  exactly once, whether the decimation loop is graphed (training) or run substep by substep.
* All buffers are allocated at PHYSICS_READY (before any graph capture); inside the graph only kernels
  run. Geom / body ids are host scalars read from the Newton-built mjModel at capture time.

Per substep and per foot (lower-leg geom), QUAD_SPEC round 5:
    raw contact c = a floor contact with dist < 0 (the Q6 rule; foot slip keeps using the raw state).
    DEBOUNCED state D flips to c only after c != D on 3 consecutive substeps (15 ms); the run counter
    resets whenever c == D.
    D flips to contact (a debounced touchdown): reward_acc += min(t_air, 0.5) - 0.25; t_air = 0.
    (Round 6: the v_x > 0.3 m/s gate is applied ONCE PER POLICY STEP by the reward term, on the post-step
    torso v_x - the same v_x as the speed term - to the sum of that step's touchdown terms.)
    after the flip logic, while D == airborne: t_air += 0.005.
    So t_air counts the substeps from the debounced lift-off to the debounced touchdown; both are 15 ms
    late, so it equals the raw swing time for a clean step.
Reset: t_air = 0, D = contact, run counter 0.

Round 7: the same pass counts the substeps (after the debounce update) in which all four DEBOUNCED states are
airborne; :meth:`finish_step` snapshots the count into ``flight_step`` (0..10 per policy step).

The same pass also records whether the torso or an upper leg has a floor contact with dist < 0 at ANY
substep of the policy step (round 5/6 fall rule); :meth:`finish_step` snapshots it into ``touch_step``.

With ``metrics`` on (evaluation only) the same pass also accumulates, per world and only while
``active[w] == 1``: substeps, feet in contact, all-feet-airborne substeps, stance length per footfall (closed
at lift-off), debounced paid touchdowns, foot-point horizontal speed while in contact, and the peak single-foot floor
normal force (mujoco_warp.contact_force, contact frame) - after the spawn drop (``settled[w] == 1``, set by the
evaluation after 0.5 s) and including it - and the mean per-footfall peak force (the MuJoCo side's keys).
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

AIR_TIME_OFFSET = 0.25  # s, QUAD_SPEC round 4
AIR_CAP = 0.5  # s, round 5: min(t_air, 0.5)
VX_MIN = 0.3  # m/s; unused in the kernel since round 6 (the reward term gates once per policy step)
DEBOUNCE_SUBSTEPS = 3  # round 5: contact state flips after 3 consecutive substeps (15 ms)


@wp.kernel
def _clear(flags: wp.array2d(dtype=wp.int32), force: wp.array2d(dtype=wp.float32)):
    w, k = wp.tid()
    flags[w, k] = 0
    force[w, k] = 0.0


@wp.kernel
def _scan(
    nacon: wp.array(dtype=wp.int32),
    geom: wp.array(dtype=wp.vec2i),
    world: wp.array(dtype=wp.int32),
    dist: wp.array(dtype=wp.float32),
    cforce: wp.array(dtype=wp.spatial_vector),
    use_force: wp.int32,
    floor: wp.int32,
    legs: wp.vec4i,
    nworld: wp.int32,
    flags: wp.array2d(dtype=wp.int32),
    force: wp.array2d(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    eager: wp.int32,
    touch_torso: wp.int32,
    touch_upper: wp.vec4i,
    touch_acc: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    other = wp.int32(-1)
    if g[0] == floor:
        other = g[1]
    elif g[1] == floor:
        other = g[0]
    if other < 0:
        return
    w = world[i]
    if w < 0 or w >= nworld:
        return
    # round 5/6 fall rule: torso or an upper leg on the floor at ANY substep of the policy step
    if dist[i] < 0.0 and (eager != 0 or counter[0] != 0):
        hit = other == touch_torso
        for k in range(4):
            if touch_upper[k] == other:
                hit = True
        if hit:
            touch_acc[w] = 1
    slot = wp.int32(-1)
    for k in range(4):
        if legs[k] == other:
            slot = wp.int32(k)
    if slot < 0:
        return
    if dist[i] < 0.0:
        flags[w, slot] = 1
    if use_force != 0:
        wp.atomic_add(force, w, slot, wp.spatial_top(cforce[i])[0])


@wp.kernel
def _update(
    counter: wp.array(dtype=wp.int32),
    eager: wp.int32,
    dt: wp.float32,
    offset: wp.float32,
    flags: wp.array2d(dtype=wp.int32),
    in_contact: wp.array2d(dtype=wp.int32),
    air: wp.array2d(dtype=wp.float32),
    reward_acc: wp.array(dtype=wp.float32),
    metrics: wp.int32,
    active: wp.array(dtype=wp.int32),
    force: wp.array2d(dtype=wp.float32),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    subtree_com: wp.array2d(dtype=wp.vec3),
    leg_bodies: wp.vec4i,
    root_body: wp.int32,
    foot_local: wp.vec3,
    m_sub: wp.array(dtype=wp.float32),
    m_feet: wp.array(dtype=wp.float32),
    m_air: wp.array(dtype=wp.float32),
    m_slip: wp.array(dtype=wp.float32),
    m_stance_run: wp.array2d(dtype=wp.float32),
    m_stance_sum: wp.array(dtype=wp.float32),
    m_stance_cnt: wp.array(dtype=wp.float32),
    m_touchdowns: wp.array(dtype=wp.float32),
    m_peak_force: wp.array(dtype=wp.float32),
    m_peak_spawn: wp.array(dtype=wp.float32),
    m_settled: wp.array(dtype=wp.int32),
    m_run_peak: wp.array2d(dtype=wp.float32),
    m_footfall_peak_sum: wp.array(dtype=wp.float32),
    db: wp.array2d(dtype=wp.int32),
    db_run: wp.array2d(dtype=wp.int32),
    qvel: wp.array2d(dtype=wp.float32),
    vx_min: wp.float32,
    air_cap: wp.float32,
    db_substeps: wp.int32,
    flight_acc: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    if eager == 0 and counter[0] == 0:
        return
    track = metrics != 0 and active[w] != 0
    n_down = float(0.0)
    for k in range(4):
        c = flags[w, k]
        # -- debounced state and the air-time reward (round 5)
        if c != db[w, k]:
            db_run[w, k] = db_run[w, k] + 1
        else:
            db_run[w, k] = 0
        if db_run[w, k] >= db_substeps:
            db[w, k] = c
            db_run[w, k] = 0
            if c != 0:
                reward_acc[w] = reward_acc[w] + wp.min(air[w, k], air_cap) - offset
                if track:
                    m_touchdowns[w] = m_touchdowns[w] + 1.0
                air[w, k] = 0.0
        if db[w, k] == 0:
            air[w, k] = air[w, k] + dt
        # -- raw state (metrics)
        if c != 0:
            in_contact[w, k] = 1
            n_down = n_down + 1.0
        else:
            if track and in_contact[w, k] != 0 and m_stance_run[w, k] > 0.0:
                m_stance_sum[w] = m_stance_sum[w] + m_stance_run[w, k]
                m_stance_cnt[w] = m_stance_cnt[w] + 1.0
                m_footfall_peak_sum[w] = m_footfall_peak_sum[w] + m_run_peak[w, k]
            in_contact[w, k] = 0
        if track:
            if c != 0:
                m_stance_run[w, k] = m_stance_run[w, k] + 1.0
                b = leg_bodies[k]
                p = xpos[w, b] + xmat[w, b] * foot_local
                v = wp.spatial_bottom(cvel[w, b]) + wp.cross(wp.spatial_top(cvel[w, b]), p - subtree_com[w, root_body])
                m_slip[w] = m_slip[w] + wp.sqrt(v[0] * v[0] + v[1] * v[1])
                m_run_peak[w, k] = wp.max(m_run_peak[w, k], force[w, k])
            else:
                m_stance_run[w, k] = 0.0
                m_run_peak[w, k] = 0.0
            m_peak_spawn[w] = wp.max(m_peak_spawn[w], force[w, k])
            if m_settled[w] != 0:
                m_peak_force[w] = wp.max(m_peak_force[w], force[w, k])
    # round 7: this substep is "in flight" when all four DEBOUNCED states are airborne
    if db[w, 0] == 0 and db[w, 1] == 0 and db[w, 2] == 0 and db[w, 3] == 0:
        flight_acc[w] = flight_acc[w] + 1.0
    if track:
        m_sub[w] = m_sub[w] + 1.0
        m_feet[w] = m_feet[w] + n_down
        if n_down == 0.0:
            m_air[w] = m_air[w] + 1.0


@wp.kernel
def _tick(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


class FootContactTracker:
    """Substep-resolution foot contact state; see the module docstring."""

    #: set True (before the env is built) to accumulate the evaluation metrics at every substep
    metrics_enabled: bool = False

    def __init__(self, num_envs: int, device: str, lower_legs: list[str], foot_local, torso: str, dt: float,
                 upper_legs: list[str]):
        self.n, self.device, self.dt = num_envs, device, float(dt)
        self.lower_legs, self.torso, self.upper_legs = list(lower_legs), torso, list(upper_legs)
        if len(self.upper_legs) != 4:
            raise ValueError("expected 4 upper legs")
        self.foot_local = wp.vec3(*[float(v) for v in foot_local])
        self.metrics = bool(FootContactTracker.metrics_enabled)
        z2i = lambda: wp.zeros((num_envs, 4), dtype=wp.int32, device=device)  # noqa: E731
        z2f = lambda: wp.zeros((num_envs, 4), dtype=wp.float32, device=device)  # noqa: E731
        z1 = lambda: wp.zeros(num_envs, dtype=wp.float32, device=device)  # noqa: E731
        self.counter = wp.zeros(1, dtype=wp.int32, device=device)
        self.touch_acc = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.flight_acc = wp.zeros(num_envs, dtype=wp.float32, device=device)
        self._last_step = None
        self.flags, self.force = z2i(), z2f()
        self.in_contact = wp.ones((num_envs, 4), dtype=wp.int32, device=device)
        self.air = z2f()
        self.db = wp.ones((num_envs, 4), dtype=wp.int32, device=device)
        self.db_run = z2i()
        self.reward_acc = z1()
        self.active = wp.ones(num_envs, dtype=wp.int32, device=device)
        self.m = {k: z1() for k in ("sub", "feet", "air", "slip", "stance_sum", "stance_cnt", "touchdowns", "peak_force",
                                    "peak_spawn", "footfall_peak_sum")}
        self.m_stance_run, self.m_run_peak = z2f(), z2f()
        self.settled = wp.zeros(num_envs, dtype=wp.int32, device=device)
        cap = 256 * num_envs if self.metrics else 1  # contact_force scratch (evaluation only)
        self._cf = wp.zeros(cap, dtype=wp.spatial_vector, device=device)
        self._cf_ids = wp.array(np.arange(cap, dtype=np.int32), dtype=wp.int32, device=device)
        self._bound = False
        # torch views
        t = wp.to_torch
        self.t_counter, self.t_in_contact, self.t_air, self.t_reward = t(self.counter), t(self.in_contact), t(self.air), t(self.reward_acc)
        self.t_touch_acc = t(self.touch_acc)
        self.t_flight_acc = t(self.flight_acc)
        self.flight_step = torch.zeros(num_envs, device=device)
        self.touch_step = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.t_active, self.t_stance_run, self.t_settled = t(self.active), t(self.m_stance_run), t(self.settled)
        self.t_run_peak = t(self.m_run_peak)
        self.t_db, self.t_db_run = t(self.db), t(self.db_run)
        self.t_m = {k: t(v) for k, v in self.m.items()}

    # ------------------------------------------------------------------------------------------------
    def _bind(self):
        import isaaclab_newton.physics.newton_manager as nm  # noqa: PLC0415

        solver = nm.NewtonManager._solver
        if solver is None or not hasattr(solver, "mjw_data"):
            raise RuntimeError("FootContactTracker needs Newton's SolverMuJoCo (MuJoCo-Warp)")
        mm = solver.mj_model
        floor = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == 0]
        if len(floor) != 1:
            raise RuntimeError(f"expected one world (floor) geom, found {floor}")

        def body(n):
            hits = [b for b in range(mm.nbody) if mm.body(b).name.endswith("_" + n)]
            if len(hits) != 1:
                raise RuntimeError(f"cannot map body {n!r} to one MuJoCo body: {hits}")
            return hits[0]

        leg_b = [body(n) for n in self.lower_legs]
        leg_g = []
        for b in leg_b:
            gs = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == b]
            if len(gs) != 1:
                raise RuntimeError(f"lower leg body {b} has {len(gs)} geoms")
            leg_g.append(gs[0])
        self.solver, self.floor = solver, int(floor[0])
        self.leg_geoms, self.leg_bodies = wp.vec4i(*leg_g), wp.vec4i(*leg_b)

        def geom_of(n):
            gs = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == body(n)]
            if len(gs) != 1:
                raise RuntimeError(f"body {n!r} has {len(gs)} geoms")
            return gs[0]

        self.touch_torso = int(geom_of(self.torso))
        self.touch_upper = wp.vec4i(*[geom_of(n) for n in self.upper_legs])
        self.root_body = int(mm.body_rootid[body(self.torso)])
        d = solver.mjw_data
        self.naconmax = int(d.contact.dist.shape[0])
        if self.metrics and self.naconmax > self._cf.shape[0]:
            raise RuntimeError(f"contact buffer {self.naconmax} > tracker scratch {self._cf.shape[0]}")
        if int(d.xpos.shape[0]) != self.n:
            raise RuntimeError("MuJoCo-Warp worlds != envs")
        self._bound = True

    def _run(self, eager: int):
        if not self._bound:
            self._bind()
        d, dev = self.solver.mjw_data, self.device
        c = d.contact
        wp.launch(_clear, dim=(self.n, 4), inputs=[self.flags, self.force], device=dev)
        if self.metrics:
            import mujoco_warp as mjw  # noqa: PLC0415

            mjw.contact_force(self.solver.mjw_model, d, self._cf_ids[: self.naconmax], False, self._cf[: self.naconmax])
        wp.launch(_scan, dim=self.naconmax,
                  inputs=[d.nacon, c.geom, c.worldid, c.dist, self._cf, int(self.metrics), self.floor, self.leg_geoms,
                          self.n, self.flags, self.force, self.counter, eager, self.touch_torso, self.touch_upper,
                          self.touch_acc], device=dev)
        m = self.m
        wp.launch(_update, dim=self.n,
                  inputs=[self.counter, eager, self.dt, AIR_TIME_OFFSET, self.flags, self.in_contact, self.air,
                          self.reward_acc, int(self.metrics), self.active, self.force, d.xpos, d.xmat, d.cvel,
                          d.subtree_com, self.leg_bodies, self.root_body, self.foot_local, m["sub"], m["feet"], m["air"],
                          m["slip"], self.m_stance_run, m["stance_sum"], m["stance_cnt"], m["touchdowns"],
                          m["peak_force"], m["peak_spawn"], self.settled, self.m_run_peak, m["footfall_peak_sum"],
                          self.db, self.db_run, d.qvel, VX_MIN, AIR_CAP, DEBOUNCE_SUBSTEPS, self.flight_acc],
                  device=dev)

    def graph_step(self):
        """Post-actuator callback (inside the graph): process the previous substep unless it belongs to the
        previous policy step (counter == 0), then advance the counter."""
        self._run(eager=0)
        wp.launch(_tick, dim=1, inputs=[self.counter], device=self.device)

    def process_last_substep(self):
        """Eager: process the policy step's last substep, then rewind the counter for the next step."""
        self._run(eager=1)
        self.t_counter.zero_()

    def finish_step(self, step_id) -> None:
        """Close the policy step ``step_id`` exactly once (whichever manager term asks first): process the last
        substep, snapshot "torso / upper leg touched the floor at any of the 10 substeps" into ``touch_step``,
        clear that accumulator."""
        if step_id == self._last_step:
            return
        self._last_step = step_id
        self.process_last_substep()
        self.touch_step = self.t_touch_acc.bool()
        self.t_touch_acc.zero_()
        self.flight_step = self.t_flight_acc.clone()  # substeps (of 10) with all feet airborne, debounced
        self.t_flight_acc.zero_()

    def take_reward(self) -> torch.Tensor:
        r = self.t_reward.clone()
        self.t_reward.zero_()
        return r

    def reset(self, env_ids):
        self.t_in_contact[env_ids] = 1
        self.t_air[env_ids] = 0.0
        self.t_reward[env_ids] = 0.0
        self.t_stance_run[env_ids] = 0.0
        self.t_run_peak[env_ids] = 0.0
        self.t_db[env_ids] = 1
        self.t_touch_acc[env_ids] = 0
        self.t_flight_acc[env_ids] = 0.0
        self.t_db_run[env_ids] = 0
