"""Fast sanity check: does the creature env build, reset, step, and scan on GPU?"""
import os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".70")

import jax, jax.numpy as jp
from creature_env import Creature, OBS_SIZE, ACTION_SIZE, OBS_LAYOUT

print("jax", jax.__version__, "| backend", jax.default_backend(), "|", jax.devices())

env = Creature()
print(f"ctrl_dt={env.dt}s ({1/env.dt:.0f} Hz) | sim_dt={env.sim_dt}s | "
      f"n_substeps={env.n_substeps} | obs={env.observation_size} act={env.action_size}")
assert env.observation_size == OBS_SIZE, (env.observation_size, OBS_SIZE)

reset, step = jax.jit(env.reset), jax.jit(env.step)
s = reset(jax.random.PRNGKey(0))
assert s.obs.shape == (OBS_SIZE,), s.obs.shape
print("obs layout:", dict(OBS_LAYOUT))
print("reset gravity_local =", [round(float(x), 3) for x in s.obs[:3]], "(want ~[0,0,-1])")

for _ in range(5):
    s = step(s, jp.zeros(ACTION_SIZE))
print(f"5 zero-action steps -> reward={float(s.reward):.3f} done={float(s.done):.0f} "
      f"height={float(s.metrics['torso_height']):.3f}")

# batched scanned rollout -- the shape training actually uses
N, T = 2048, 200
@jax.jit
def rollout(keys):
    st = jax.vmap(env.reset)(keys)
    def body(st, _):
        return jax.vmap(env.step)(st, jp.zeros((N, ACTION_SIZE))), None
    st, _ = jax.lax.scan(body, st, None, length=T)
    return st.data.qpos[:, 2]

keys = jax.random.split(jax.random.PRNGKey(1), N)
t0 = time.time(); h = rollout(keys); h.block_until_ready()
print(f"compile: {time.time()-t0:.1f}s")
t0 = time.time(); h = rollout(keys); h.block_until_ready(); dt = time.time() - t0
print(f"{N} envs x {T} ctrl steps in {dt:.2f}s -> {N*T/dt:,.0f} env-steps/sec "
      f"({N*T*env.n_substeps/dt:,.0f} physics steps/sec)")
print("OK")
