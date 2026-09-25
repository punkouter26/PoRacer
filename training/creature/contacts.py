"""Per-substep floor-contact tracking for chosen geoms, as Warp kernels so it can run
inside the env's captured CUDA graph after EVERY physics substep.

Raw contact: the geom has a force-carrying floor contact (dist < 0) in the latest
substep's collision pass (the same test as CreatureEnv.floor_contacts).

Debounced state (debounce = N substeps): a geom's state flips only after N consecutive
substeps whose raw contact disagrees with the current state (N = 1: no debounce). All
timers follow the DEBOUNCED state, so every swing and stance is measured between two
debounced flips (both shifted by the same N substeps, durations preserved).

Per world w and tracked geom k:
    raw[w, k]             raw contact in the latest substep (0/1)
    state[w, k]           debounced contact state (0/1)
    air_time[w, k]        seconds in the current debounced swing (0 while in stance)
    contact_time[w, k]    seconds in the current debounced stance (0 while in swing)
Per world, accumulated over one policy step (cleared by begin_step()):
    touchdown_sum[w]      sum over debounced touchdowns of (min(t_air, cap) - target),
                          t_air = the swing that just ended (legged_gym feet_air_time,
                          with Isaac Lab's clamp)
    touchdown_count[w]    number of debounced touchdowns
    stance_sum[w]         sum of the stance durations that ended (debounced lift-offs)
    liftoff_count[w]      number of debounced lift-offs
    any_raw[w]            1 if ANY tracked geom had a raw contact in ANY substep
    flight_substeps[w]    substeps in which EVERY tracked geom was in the debounced airborne
                          state (evaluated after that substep's debounce update)
At an episode reset: debounced state = reset_state (1 = "in contact", the quad's choice,
matching the Isaac side), pending counters and timers 0.
"""

from __future__ import annotations

from typing import Sequence

import torch
import warp as wp


@wp.kernel
def _clear_step(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float),
                d: wp.array(dtype=float), e: wp.array(dtype=float), f: wp.array(dtype=float)):
    w = wp.tid()
    a[w] = 0.0
    b[w] = 0.0
    c[w] = 0.0
    d[w] = 0.0
    e[w] = 0.0
    f[w] = 0.0


@wp.kernel
def _flight(state: wp.array2d(dtype=wp.int32), k: int, flight_substeps: wp.array(dtype=float)):
    w = wp.tid()
    down = int(0)
    for i in range(k):
        down = down + state[w, i]
    if down == 0:
        flight_substeps[w] = flight_substeps[w] + 1.0


@wp.kernel
def _clear_2d(a: wp.array2d(dtype=wp.int32)):
    w, k = wp.tid()
    a[w, k] = 0


@wp.kernel
def _mark(nacon: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i), dist: wp.array(dtype=float),
          worldid: wp.array(dtype=int), slot_of_geom: wp.array(dtype=int), floor: int,
          raw: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    other = int(-1)
    if g[0] == floor:
        other = g[1]
    elif g[1] == floor:
        other = g[0]
    if other < 0:
        return
    s = slot_of_geom[other]
    if s < 0:
        return
    if dist[i] >= 0.0:
        return
    raw[worldid[i], s] = 1


@wp.kernel
def _update(raw: wp.array2d(dtype=wp.int32), state: wp.array2d(dtype=wp.int32),
            pending: wp.array2d(dtype=wp.int32), debounce: int, sub_dt: float, target: float,
            cap: float, air_time: wp.array2d(dtype=float), contact_time: wp.array2d(dtype=float),
            touchdown_sum: wp.array(dtype=float), touchdown_count: wp.array(dtype=float),
            stance_sum: wp.array(dtype=float), liftoff_count: wp.array(dtype=float),
            any_raw: wp.array(dtype=float)):
    w, k = wp.tid()
    r = raw[w, k]
    if r == 1:
        any_raw[w] = 1.0
    st = state[w, k]
    if r != st:
        pending[w, k] = pending[w, k] + 1
        if pending[w, k] >= debounce:
            pending[w, k] = 0
            state[w, k] = r
            if r == 1:                                   # debounced touchdown
                wp.atomic_add(touchdown_sum, w, wp.min(air_time[w, k], cap) - target)
                wp.atomic_add(touchdown_count, w, 1.0)
                air_time[w, k] = 0.0
            else:                                        # debounced lift-off
                wp.atomic_add(stance_sum, w, contact_time[w, k])
                wp.atomic_add(liftoff_count, w, 1.0)
                contact_time[w, k] = 0.0
            st = r
    else:
        pending[w, k] = 0
    if st == 1:
        contact_time[w, k] = contact_time[w, k] + sub_dt
    else:
        air_time[w, k] = air_time[w, k] + sub_dt


class FloorContactTracker:
    def __init__(self, mjm, wd, geoms: Sequence[int], floor_geom: int, device: str,
                 air_time_target: float = 0.0, air_time_cap: float = 1e9, debounce: int = 1,
                 reset_state: int = 0):
        n, k = wd.nworld, len(geoms)
        self.wd, self.floor = wd, int(floor_geom)
        self.target, self.cap, self.debounce = float(air_time_target), float(air_time_cap), int(debounce)
        self.sub_dt = float(mjm.opt.timestep)
        self.reset_state = int(reset_state)
        slot = [-1] * mjm.ngeom
        for index, g in enumerate(geoms):
            slot[int(g)] = index
        self.slot_of_geom = wp.array(slot, dtype=int, device=device)
        self.raw_wp = wp.zeros((n, k), dtype=wp.int32, device=device)
        self.state_wp = wp.zeros((n, k), dtype=wp.int32, device=device)
        self.pending_wp = wp.zeros((n, k), dtype=wp.int32, device=device)
        self.air_wp = wp.zeros((n, k), dtype=float, device=device)
        self.stance_wp = wp.zeros((n, k), dtype=float, device=device)
        self.step_wp = [wp.zeros(n, dtype=float, device=device) for _ in range(6)]
        self.raw = wp.to_torch(self.raw_wp)
        self.state = wp.to_torch(self.state_wp)
        self.pending = wp.to_torch(self.pending_wp)
        self.air_time = wp.to_torch(self.air_wp)
        self.contact_time = wp.to_torch(self.stance_wp)
        (self.touchdown_sum, self.touchdown_count, self.stance_sum, self.liftoff_count,
         self.any_raw, self.flight_substeps) = (wp.to_torch(a) for a in self.step_wp)
        self.state[:] = self.reset_state
        self.n, self.k, self.device = n, k, device

    def begin_step(self) -> None:
        wp.launch(_clear_step, dim=self.n, inputs=self.step_wp, device=self.device)

    def after_substep(self) -> None:
        wp.launch(_clear_2d, dim=(self.n, self.k), inputs=[self.raw_wp], device=self.device)
        c = self.wd.contact
        wp.launch(_mark, dim=c.dist.shape[0],
                  inputs=[self.wd.nacon, c.geom, c.dist, c.worldid, self.slot_of_geom, self.floor,
                          self.raw_wp], device=self.device)
        wp.launch(_update, dim=(self.n, self.k),
                  inputs=[self.raw_wp, self.state_wp, self.pending_wp, self.debounce, self.sub_dt,
                          self.target, self.cap, self.air_wp, self.stance_wp, *self.step_wp[:4],
                          self.step_wp[4]], device=self.device)
        wp.launch(_flight, dim=self.n, inputs=[self.state_wp, self.k, self.step_wp[5]],
                  device=self.device)

    def reset(self, index: torch.Tensor) -> None:
        for t in (self.raw, self.pending):
            t[index] = 0
        self.state[index] = self.reset_state
        for t in (self.air_time, self.contact_time, self.touchdown_sum, self.touchdown_count,
                  self.stance_sum, self.liftoff_count, self.any_raw, self.flight_substeps):
            t[index] = 0.0
