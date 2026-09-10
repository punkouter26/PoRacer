# MuJoCo RL Optimization Log

Autonomous 2-phase optimization loop. Started **2026-09-09 ~22:35 local**, budget 8 h.

---

## Phase 1 — Metric discovery & baseline

### 1.0 Machine & toolchain constraints (established first, because they decided the target)

| Fact | Value | Consequence |
|---|---|---|
| GPU | RTX 2060, 6 GB, sm_75 (Turing), driver 610.74 | Enough for GPU-vectorized physics; 6 GB caps world count |
| Python | 3.10.11 via `py -3.10` | No system `python` on PATH (Windows Store stub only) |
| `.venv` (ml-agents) | **absent** | Every ML-Agents route in CLAUDE.md is dead until rebuilt |
| `.venv-mjwarp` | **absent** | The MojucuBoy trainer's expected venv; built this session |
| WSL2 | **broken** — `Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG` | Decisive: see below |
| `results/`, `ISAAC/logs/` | absent | No historical curves to compare against |

**Target selection.** The project holds two MuJoCo creatures. They are not
equally trainable *on this machine*:

- **Fido** (`training/fido/`, `Assets/Creature/creature.xml`) trains on **MJX +
  Brax PPO**, which needs `jax[cuda]`. `requirements-wsl.txt` says it plainly:
  *"JAX publishes no CUDA wheels for Windows, so GPU MJX must run under WSL."*
  WSL does not start on this machine. Fido is therefore **not trainable here**
  without either repairing WSL or accepting CPU-only MJX, which for a 1000-step
  8-DOF episode is orders of magnitude too slow to iterate on inside 8 h.
- **MojucuBoy** (`training/mojucuboy/`) trains on **MuJoCo Warp + torch**, and
  Warp *does* ship Windows CUDA. It is already GPU-vectorized (`NUM_WORLDS`
  worlds stepped on-device, `wp.to_torch` zero-copy views, no host readback in
  the rollout loop).

**Decision: MojucuBoy is the optimization target.** Fido is audited below anyway,
because the audit found the concrete cause of its documented steering failure and
that finding is worth recording even though I cannot act on it here.

---

### 1.1 Fido audit (recorded, not actioned — untrainable on this machine)

`Assets/Creature/creature.xml`, 8 DOF, 33 obs, `creature_env.py`.

**Reward:**
```
r = 2.0·min(vx, 1.5) + 0.10·healthy + 0.05·upright − 0.005·Σa² − 1e-4·Σq̇²
```

**Finding F1 — the steering failure is a reward defect, not only an observation
defect.** CLAUDE.md records Fido as unable to steer and attributes it to the
observation vector carrying no goal or heading. That is true but incomplete. The
reward pays for **world-frame +X velocity alone**. There is **no lateral (y)
penalty and no yaw-rate penalty**, so curving is free — the policy can spiral at
zero cost provided its x-component stays high. `angvel_local` *is* already in the
observation (qvel[3:6]), so the policy can sense yaw rate; nothing asks it to
keep it near zero.

**Consequence, and it matters:** a straight gait is reachable by a **reward-only
change** (add drift/yaw penalties), which leaves the 33-float observation
contract with `CreatureAgent.BuildObservation` **untouched** — no re-export of
the Unity side's layout, no C# change. Steering *to a goal* would still need the
contract change CLAUDE.md describes. Straightening does not.

**Finding F2 — `cost_jvel` is computed and charged but never logged.**
`creature_env.py` subtracts `cost_jvel` from the reward, then omits it from
`state.metrics`. The joint-velocity cost is invisible in TensorBoard, so its
weight cannot be tuned against evidence.

**Finding F3 — train/deploy substep mismatch.** MJCF and `default_config` use
`sim_dt = 0.004`; Unity runs `Time.fixedDeltaTime = 0.005` with
`actionDecimation 4`. The **policy rate matches at 0.02 s**, so the mismatch is
in the integrator substep only (5×0.004 vs 4×0.005), but it means contact
resolution differs between the rig that was trained and the rig that races.

**Finding F4 — the creature is anatomically incapable of steering.** All eight
joints are `axis="0 1 0"`: four hips and four knees, every one of them in the
sagittal plane. There is **no abduction/adduction DOF anywhere**. Fido cannot
steer by leg placement even in principle; any heading change has to come from
asymmetric gait timing and friction. A heading observation alone would not be
sufficient — the model needs a DOF it does not have.

**MJCF hygiene:** joint ranges are anatomical and limited (`hip ±50°`,
`knee −90..20°`), `damping 0.6`, `armature 0.02`, `gear 18`, `condim 3`,
pyramidal cone, Newton solver at 4 iterations / 8 line-search. No
`<contact><exclude>` tags, but none are needed: MuJoCo's default contact filter
already excludes parent–child pairs connected by a joint, and no sibling limbs
overlap in the home pose. **No action required on that item of the brief.**

---

### 1.2 MojucuBoy audit (the live target)

`training/mojucuboy/mojucuboy_roundtrip.xml`, 21 DOF, **75 obs**, 21 actions.
Timing 0.005 s × decimation 4 = 0.02 s policy — **exactly Unity's**, so no
train/deploy rate mismatch (contrast Finding F3).

**Reward terms already present** (all penalties saturating via `tanh` of a
normalized quantity — a deliberate design the file documents as replacing raw
quadratics that produced a −205 impact term against a +0.13 tracking term):

| Term | Weight | Purpose |
|---|---|---|
| `W_TRACK` | 2.0 | forward speed along commanded heading |
| `W_HEADING` | 0.4 | facing the commanded heading |
| `W_GETUP` | 0.60 | continuous, paid for being on its feet |
| `W_DRIFT` | 0.15 | uncommanded lateral + yaw motion |
| `W_ALIVE` | 0.10 | survival |
| `W_UPRIGHT` | 0.05 | torso uprightness |
| `W_IMPACT` | 0.05 | impact force beyond bodyweight |
| `W_ACCEL` | 0.03 | joint acceleration spikes |
| `W_ACTION_RATE` | 0.01 | action smoothness |
| `W_CTRL` | 0.005 | control effort |

This environment **already implements most of the Phase 2 reward-shaping brief**:
control cost, joint-acceleration penalty, target tracking, upright balance, and
impact/contact shaping are all present and bounded. It also already carries
domain randomization (gains ±30 %, masses ±10 %, friction 0.6–1.4, 1.1 m/s pushes
every 150 steps) and a 30 % fallen-reset fraction so recovery is trained directly.

**Finding M1 — instrumentation gap (the actionable Phase 1 item).** The trainer
logs only:
`rollout/mean_return`, `rollout/mean_episode_length`,
`rollout/mean_forward_speed`, `rollout/fall_rate`,
`loss/{policy,value,entropy}`, `perf/steps_per_second`.

**Not one of the ten reward terms is logged individually**, and none of the
physical-stability KPIs the brief asks for are visible: no heading error, no
lateral drift, no control effort, no joint acceleration, no impact force, no
torso pitch/roll. Ten weights are being tuned against a single scalar return.
This is the gap to close before any optimization run is meaningful.

**Finding M2 — contact buffers are sized against measurement, not guesswork.**
`NCONMAX = 32`, `NJMAX = 128` against a measured worst case of `ncon=11 /
nefc=46` (~2.9× headroom), and the trainer asserts on overflow rather than
trusting it. **No action required** on the brief's contact-budget item.

---

### 1.3 KPIs and convergence thresholds

Defined from the reward structure above. Survival and speed come from existing
logging; the rest require the M1 instrumentation.

| # | KPI | Definition | Threshold |
|---|---|---|---|
| K1 | **Uptime** — see M4 | mean `standing` per step (1 = upright at full height, 0 = flat) | **> 0.90** |
| K2 | Speed tracking | `\|mean_forward_speed − 1.5\| / 1.5` | **within ±10 %** |
| K3 | Fallen fraction | mean `fallen` per step (below `MIN_HEIGHT` or `MIN_UPRIGHT`) | **< 0.10** |
| K4 | Heading error | mean abs angle between torso facing and command | **< 15°** |
| K5 | Lateral drift | mean abs uncommanded lateral velocity | **< 0.25 m/s** |
| K6 | Control effort | mean `Σa²` per step, normalized by action count | **≤ baseline** |
| K7 | Joint accel | mean `Σq̈²`, the `W_ACCEL` quantity pre-weight | **≤ baseline** |
| K8 | Uprightness | mean torso up-axis · world up | **> 0.90** |
| K9 | Throughput | environment steps/second, headless | maximize; record baseline |

**Exit criteria:** K1–K8 satisfied across 10 consecutive evaluation episodes; or
improvement plateaus (< 3 % over 3 runs); or 8 h elapses.

#### Finding M4 — the project's convergence gate is two-thirds vacuous

`gate4_eval.py` states the agreed convergence metrics as:

```
mean episode length  >= 900 / 1000 steps
mean forward speed   >= 1.2 m/s
survival             >= 90 / 100 episodes
```

It computes survival as `alive &= ~done`, then `survived = length >=
EPISODE_STEPS`. But `done` is **timeout-only** (M3): no world ever terminates
early, so every world always reaches exactly 1000 steps. **`mean_episode_length`
is always 1000 and `survival_rate` is always exactly 1.0**, for any policy,
including an untrained one. Two of the three gates cannot fail.

These thresholds were written when a fall ended the episode. The env later
changed — deliberately, so the racer would learn to get up — and the gate was
never updated to match. Only `mean_forward_speed` still measures anything.

My own K1 had inherited exactly the same flaw and is corrected above: survival
is now **mean `standing` per step**, and K3 is **mean `fallen` per step**. Both
are per-step quantities that a lying-down policy genuinely fails, and both are
now logged (§1.5). This also means the baseline's `rollout/mean_episode_length`
and `kpi/survival_ratio` curves will be flat at 1000 and 1.0 respectively — that
is the bug, not a result.

**Caught live, and it is the clearest evidence in this log.** The baseline's
first episode statistics landed at iteration 45 — exactly the predicted
boundary, `ceil(1000/24) = 42` rounded up to the next multiple of 5:

```
iter   45  steps  8.85M  ret  75.06  len 1000.0  spd 0.00 m/s  fall 1.00
                         trk 0.17  hdg 90.7deg  lat 0.37  upr 0.27  std 0.10
```

`len 1000.0` and `fall 1.00` in the same line. Under the old gate that policy
scores a **perfect** mean episode length and a **perfect** survival rate while
**100 % of its worlds are lying on the floor** at the episode boundary, moving
at 0.00 m/s. Only `std 0.10` and `fall 1.00` — both newly added — say what is
actually happening.

---

### 1.4 Toolchain built

No requirements file existed for MojucuBoy, so `.venv-mjwarp` was constructed
from the imports:

| Package | Version | Note |
|---|---|---|
| torch | 2.6.0+**cu124** | CUDA verified live on the RTX 2060 |
| mujoco | 3.12.0 | matches the repo pin |
| mujoco-warp | 3.13.0 | **not on PyPI** — built from `github.com/google-deepmind/mujoco_warp` |
| warp-lang | 1.17.0 | |
| numpy | 1.26.4 | held `<2`; torch 2.6 warns and disables numpy interop otherwise |
| tensorboard | 2.21.0 | the trainer refuses to start without `tensorboard.exe` here |
| onnx | 1.22.0 | for the export step |

`mujoco-warp` 3.13 against `mujoco` 3.12 was a risk; verified compatible —
the env builds, steps, and returns a well-formed reward.

**Env smoke test passed:** 256 worlds built in 0.4 s (after a one-off ~25 s Warp
kernel compile), `obs (256, 75)`, step returns all fourteen terms including the
five added below, mean reward 0.149.

### 1.5 Instrumentation added (closes Finding M1)

`_reward` already *returned* a `terms` dict; the trainer simply discarded
everything except `speed_along` and `fallen`. Two changes:

1. **`mojucuboy_env.py`** — added `ctrl`, `action_rate`, `lateral`, `upright`
   and `height` to `terms`. All five were already being charged against the
   reward and none were observable.
2. **`train_mojucuboy.py`** — accumulate every term per rollout step and emit
   them as `kpi/*` scalars, plus derived `kpi/survival_ratio`,
   `kpi/speed_error_frac` and `kpi/heading_err_deg`. Console line now carries
   heading error, lateral drift and uprightness.

**Accumulators stay on the GPU** and are read once at logging time. A `.item()`
per step would have cost more than the physics — the same reason the env keeps
its rollout host-free. `heading_err_deg` is accumulated as an angle per world,
not as `arccos` of the mean cosine; those are not equal and only the former has
units.

### 1.5b Finding M3 — the trainer was blind for the first 45 iterations

Caught by the first baseline attempt producing **zero output in 10 minutes**
while the GPU sat at 59 % — it was training correctly and logging nothing.

`MojucuBoyEnv.step` sets `done = timeout`, and only timeout: a fall deliberately
does **not** end the episode, because "the racer is never picked up, in training
or in the race, so it has to learn to get itself back on its feet". Sound design
— but it means episodes are always exactly `EPISODE_STEPS = 1000` long, and with
`--rollout 24` the first `done` cannot arrive before iteration
`ceil(1000/24) = 42`.

The logging block was gated `if done_returns and iteration % 5 == 0`, so the
**first telemetry of any kind landed at iteration 45**. Every run shorter than
that — which is every 5–10 minute validation run the brief calls for — would
have produced an empty TensorBoard and an empty stdout.

**Fix:** per-step KPIs and losses now log unconditionally every 5 iterations;
episode statistics still wait for completed episodes and are appended to the
console line when present. Also switched the launch to `python -u`, since the
early prints were block-buffered and invisible.

**Related observation (not yet actioned):** `episode_step` starts at 0 for every
world, so all worlds time out on the *same* iteration. Episode statistics
therefore arrive in one lump every 42 iterations rather than continuously.
Staggering the initial `episode_step` per world would smooth both the statistics
and the GAE bootstrap. Logged as candidate **E7**.

### 1.5c Reference point — the shipped brain (`boy_chase01`)

`runs/boy_chase01/` has no `policy.pt` (only the exported ONNX), so it cannot be
re-evaluated, but its recorded `gate4_eval.json` is the reference:

| Metric | Randomised | Nominal |
|---|---|---|
| mean forward speed | **2.043 m/s** | 2.022 m/s |
| mean return | 2336.1 | 2341.9 |
| mean episode length | 1000.0 | 1000.0 |
| survival rate | 1.000 | 1.000 |

Trained at 1500 iterations × 196 608 = **295 M steps**, default hyperparameters.

**This confirms M4 empirically.** Episode length is exactly 1000.0 and survival
exactly 1.000 — not "excellent", *unconditional*. Those two numbers would read
identically for a policy that never stood up.

#### Finding M5 — the speed tracking kernel is one-sided, and the shipped brain overshoots by 36 %

The commanded speed is `TARGET_SPEED = 1.5` m/s. The shipped brain runs at
**2.04 m/s**, +36 %.

That is not misbehaviour, it is what the reward asks for:

```python
shortfall = (along - self.command_speed).clamp(max=0.0)
track = torch.exp(-(shortfall / SPEED_SIGMA) ** 2)
```

`clamp(max=0.0)` zeroes any positive error, so the kernel penalizes running
*slow* and is perfectly indifferent to running *fast*. The docstring says
exceeding the target "earns nothing extra, which stops the policy trading
stability for a sprint it cannot hold" — true, it earns nothing extra, but it
also costs nothing, so there is no gradient pulling the speed back down to the
command.

**Consequence for the brief's KPI.** "Target velocity tracking within ±10 %"
cannot be met by this reward as written: the shipped brain sits at +36 % and is
under no pressure to come down. Making the kernel two-sided (drop the `clamp`)
is therefore a *required* change if K2 is to be gated at ±10 %, not an optional
tuning idea. Promoted to experiment **E6a**, and it is evidence-backed rather
than speculative.

**Measured directly**, by driving the root at a known velocity and reading the
`track` term back (command 1.5 m/s):

| Racer speed | One-sided `track` | Two-sided `track` |
|---|---|---|
| 0.5 m/s | 0.3679 | 0.3679 — identical, undershoot is penalized either way |
| 1.5 m/s | 1.0000 | 1.0000 — on command |
| 2.0 m/s | **1.0000** | 0.7788 |
| 3.0 m/s | **1.0000** | 0.1054 |

The one-sided kernel pays **full marks at 3.0 m/s against a 1.5 m/s command** —
double the commanded speed, zero penalty. At 2.0 m/s it pays exactly what
running on command pays, which is precisely why the shipped brain settled at
2.04 and stayed there. There is no gradient to descend.

Open question this raises, worth flagging rather than assuming: for a *racer*,
overshooting the commanded speed may be desirable, in which case the correct fix
is to relax K2 to "≥ target" rather than to make the kernel symmetric. Both are
tested below; the reward change is the one that matches the brief as written.

### 1.6 Throughput baseline (KPI K9)

Pure environment stepping, zero-action, 50 steps after 10 warmup, headless:

| Worlds | Steps/s | Δ vs previous | VRAM used |
|---|---|---|---|
| 1 024 | 13 015 | — | — |
| 2 048 | 23 762 | +83 % | — |
| 4 096 | 40 877 | +72 % | — |
| 8 192 | 56 343 | +38 % | — (trainer default) |
| 12 288 | 62 381 | +11 % | 1.79 / 6.44 GB |
| 16 384 | **67 761** | **+9 %** | 1.97 / 6.44 GB |
| 24 576 | 70 765 | +4 % | 2.41 / 6.44 GB |

**Thermal note.** The 2060 sits at **84 °C** under sustained training at 8192
worlds (60 % utilisation, 2.4 GB). That is inside spec — Turing throttles around
88–93 °C — but it has no margin to spare, and it is a reason not to chase the
last 4 % by pushing world count: E1 raised utilisation to 71 %, and a
multi-hour run at that load is closer to the throttle point than this one. If a
long run's steps/s decays over time, check temperature before blaming the code.

**The knee is at 16 384 worlds** — +20 % over the trainer's default 8 192 for
1.97 GB of a 6.44 GB card. Past that the curve flattens to +4 %, which does not
pay for the larger rollout buffers. This is the first Phase 2 change to validate.

The brief's "vectorize environments (SubprocVecEnv / AsyncVectorEnv)" item is
**already satisfied and then some**: this env is GPU-vectorized with zero host
readback in the rollout loop, which is strictly better than subprocess vector
envs. No action.

### 1.7 Progress

- [x] Audit MJCF, env code, observation space, reward terms — findings F1–F4, M1–M2
- [x] Define KPIs and quantified thresholds — K1–K9
- [x] Instrument per-term reward + stability logging — closes M1
- [x] Build `.venv-mjwarp` and verify GPU stack end-to-end
- [x] Throughput baseline and world-count sweep — K9
- [x] Headless baseline training run; record SPS and initial KPI curves

### 1.8 Baseline results — `baseline_default`, 60 iterations, 11.80 M steps

Default hyperparameters (8192 worlds, rollout 24, lr 3e-4, entropy 2e-3,
γ 0.99, λ 0.95, clip 0.2, 4 epochs × 8 minibatches), seed 0, headless.

| Iter | Steps | track | speed m/s | heading err | lateral | upright | **uptime** | steps/s |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.98 M | 0.13 | −0.00 | 90.0° | 0.19 | 0.08 | **0.01** | 46 k |
| 15 | 2.95 M | 0.13 | 0.00 | 89.9° | 0.19 | 0.14 | **0.03** | 47 k |
| 25 | 4.92 M | 0.13 | 0.00 | 89.6° | 0.21 | 0.23 | **0.05** | 47 k |
| 35 | 6.88 M | 0.14 | 0.01 | 89.8° | 0.24 | 0.32 | **0.10** | 48 k |
| 45 | 8.85 M | 0.17 | −0.01 | 90.7° | 0.37 | 0.27 | **0.10** | 48 k |
| 55 | 10.81 M | 0.17 | 0.02 | 90.9° | 0.32 | 0.44 | **0.16** | 48 k |
| 60 | 11.80 M | 0.14 | 0.01 | 90.0° | 0.22 | 0.46 | **0.18** | 48 k |

**Baseline numbers for comparison:**

| KPI | Baseline @ 11.80 M | Threshold | Status |
|---|---|---|---|
| K1 uptime | **0.18** | > 0.90 | far off — expected at 4 % of the reference run |
| K2 speed | **0.01 m/s** | 1.5 ± 10 % | not walking yet |
| K4 heading err | **90.0°** | < 15° | **exactly** chance — see below |
| K5 lateral drift | **0.22** | < 0.25 | passes, but meaningless while stationary |
| K8 uprightness | **0.46** | > 0.90 | rising steadily |
| K9 throughput | **48 k steps/s** | maximize | 15 % below the 56 k pure-env rate → PPO overhead |

**Read:** the policy is learning to *stand up*, monotonically — uprightness
0.08 → 0.46 and uptime 0.01 → 0.18 over 11.8 M steps — but has not begun to
walk. That is the expected shape: `W_GETUP` (0.60) and the standing gate on the
tracking term mean the racer must get off the floor before any locomotion
reward is reachable at all. Nothing here is converged; 11.8 M steps is 4 % of
the 295 M the shipped brain took.

**Throughput:** 48 k steps/s effective against 56 k for pure environment
stepping at the same world count, so the PPO update costs ~15 % of wall-clock.
That is the headroom E1 and E3/E4 are competing for.

**K4 calibration — 90° is precisely the chance level, not a rough one.** On
reset the env commands *any* heading:

```python
self.command_heading[index] = yaw + (torch.rand(...) * 2 - 1) * torch.pi
```

Uniform over ±π, so for a policy with no heading control the expected mean
absolute error is `E|U(−180°, 180°)| = 90°` exactly. The baseline sits at
89.6–91.8° across every logged point, which is that value and nothing else. Any
sustained reading below ~85° is therefore the first real evidence of heading
learning, and the metric has a known floor to measure against. `command_speed`
is *not* randomised (fixed at `TARGET_SPEED`), so K2 needs no such correction.

---

## Phase 2 — results

**Methodology.** Runs are compared at a **matched step count (11.80 M)**, not at
matched iterations, because `--worlds` changes samples per iteration and equal
iterations would compare unequal experience. Wall-clock and steps/s are reported
separately.

**Caveat that applies to E1 specifically.** Holding steps constant while raising
`--worlds` also *halves the number of gradient updates* (30 PPO updates instead
of 60 over the same 11.8 M samples). E1 is therefore not a pure throughput test
— it is throughput *and* the large-batch/fewer-updates trade together. If its
KPIs come out worse at matched steps, the correct reading is that the extra
throughput was bought with sample efficiency, not that the GPU got slower.

### Finding M6 — the trainer deletes shipped brains as a side effect of starting

Caught mid-session, by noticing `runs/boy_chase01` had vanished from a directory
listing.

```python
def prune_runs(keep: int = 3) -> None:
    runs = sorted((p for p in RESULTS.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    for old in runs[:max(0, len(runs) - keep)]:
        shutil.rmtree(old, ignore_errors=True)
```

Oldest-by-mtime, no exemptions. Launching the **fourth** run of a session
therefore deleted `runs/boy_chase01`, which held:

- `mojucuboy_policy.onnx` — the brain shipping in `Assets/Agents/MojucuBoy_v01/`
- `gate4_eval.json`, `gate5_unity.json` — its recorded grades
- `mujoco_reference.json` — the parity reference

That run has **no `policy.pt`**, so the ONNX *is* the artifact; no rerun
reproduces it. It came back only because those five files are tracked in git.
Had they been gitignored like the other run outputs, the shipped brain would
have been gone.

This is worse than a housekeeping bug: it is a training script destroying a
release artifact as a side effect of an unrelated action, silently, with
`ignore_errors=True` guaranteeing that even a *failed* deletion would say
nothing.

**Fixed.** Runs containing `*.onnx`, `gate*.json` or `mujoco_reference.json` are
treated as graduated and never pruned; only training scratch is. `ignore_errors`
dropped so a failed wipe reports itself — precisely the Windows file-handle case
CLAUDE.md warns about for TensorBoard.

**Verified in production, not just in a unit check.** Launching the next run put
the directory count back over the threshold and the pruner fired for real:

```
pruned stale run baseline_default
```

It took the disposable experiment and left `boy_chase01` with all five files
intact — the exact situation that destroyed the shipped brain an hour earlier.
Experiment curves worth keeping were archived outside `runs/` first, since the
pruner is now doing its job correctly and *will* delete them.

### E1 — `--worlds 16384` (vs 8192). Throughput won, learning lost.

**Throughput hypothesis: confirmed exactly.** 56 k steps/s against the
baseline's 46–48 k, **+20 %**, which is precisely what the standalone sweep in
§1.6 predicted (56 343 → 67 761 pure-env, ~20 %). GPU utilisation rose from
59 % to 71 %, VRAM 2.50 GB of 6.44 GB.

**Learning hypothesis: refuted.** At matched step counts E1 is behind the
baseline everywhere:

| Steps | Baseline uprightness / uptime | E1 uprightness / uptime |
|---|---|---|
| 3.93 M | 0.19 / 0.04 | **0.11 / 0.02** |
| 5.90 M | 0.28 / 0.08 | **0.15 / 0.03** |
| 7.86 M | 0.34 / 0.11 | **0.20 / 0.04** |
| 9.83 M | 0.39 / 0.14 | **0.24 / 0.06** |
| **11.80 M** (final) | **0.46 / 0.18** | **0.28 / 0.08** |

E1 reaches roughly **half** the baseline's uptime at every matched point. The
20 % throughput gain does not come close to paying for it: to reach the
baseline's 7.86 M-step uptime of 0.11, E1 needs well past 11.8 M steps, so it is
slower in *wall-clock to a given competence* despite being faster in steps/s.

**Reading.** The cost is gradient updates, not physics. Holding steps fixed,
16384 worlds performs 30 PPO updates where 8192 performs 60. Larger batches make
each update better-estimated but there are half as many, and at this stage of
training — where the policy is still learning to stand at all — update *count*
dominates update *quality*.

**Verdict: rejected.** Keep `--worlds 8192`. Steps/s is the wrong objective to
maximize on its own; it is only worth having if sample efficiency holds, and
here it does not.

**Hypothesis this generates (E8, tested next):** if update count dominates, then
*increasing* updates per sample at fixed worlds should beat the baseline —
`--epochs 8` instead of 4, doubling gradient steps over the same experience.
This is the direct converse of E1 and the data, not the brief, is what suggests
it.

### E8 — `--epochs 8` (vs 4). No effect, and 8 % slower.

| Steps | Baseline uprightness / uptime | E8 uprightness / uptime |
|---|---|---|
| 1.97 M | 0.11 / 0.02 | 0.10 / 0.02 |
| 2.95 M | 0.14 / 0.03 | 0.14 / 0.03 |
| 3.93 M | 0.19 / 0.04 | 0.18 / 0.04 |
| 4.92 M | 0.23 / 0.05 | 0.22 / 0.05 |
| 7.86 M | 0.34 / 0.11 | 0.34 / 0.11 |
| 9.83 M | 0.39 / 0.14 | 0.37 / 0.13 |
| **11.80 M** (final) | **0.46 / 0.18** | **0.44 / 0.17** |

Throughput 43–44 k steps/s against the baseline's 47–48 k, so the extra four
epochs cost ~8 % of wall-clock and bought nothing measurable.

**Verdict: rejected.** Doubling gradient steps per sample is indistinguishable
from the baseline within noise.

### What E1 and E8 together mean

E1 halved the updates and got worse. E8 doubled them and changed nothing. Those
two results do not contradict each other — together they say the optimizer is
**not the binding constraint** at this stage. The baseline's 4 epochs already
extract what there is to extract from each batch; E1 lost ground because it fell
*below* that sufficiency, not because more updates are inherently better.

The policy is spending 11.8 M steps learning to *stand up*, and how it stands is
governed by `W_GETUP` and the `standing` gate on the tracking term, not by the
learning rate. **The remaining budget is therefore better spent on reward
structure than on hyperparameters** — which is where E6a already points, and
E6a targets K2, the one KPI that the current reward makes *unreachable* rather
than merely hard.

Accordingly E2/E3/E4/E5 (entropy, lr, rollout, λ) are **dropped**. Tuning them
further would be optimizing a constraint that is not binding, and the evidence
for that is above rather than assumed.

### E6a — `--two-sided-speed`, full-length run (in progress)

The remaining budget goes here, as one long controlled run rather than more
short ones.

**Why this and not more hyperparameters.** K2 asks for velocity tracking within
±10 %. Under the shipped reward that is not merely unachieved, it is
*unreachable*: the kernel pays full marks anywhere at or above the command
(1.0000 at 3.0 m/s against a 1.5 m/s command, measured above). No amount of
learning-rate tuning reaches a target the reward does not encode.

**The control is free.** `boy_chase01` was trained at 1500 iterations, 8192
worlds, rollout 24, 4 epochs, 8 minibatches, lr 3e-4, γ 0.99, λ 0.95, clip 0.2,
entropy 2e-3, seed 0 — and its evaluation is on disk. This run uses **that exact
configuration**, changing only the speed kernel. So the comparison is a genuine
one-variable experiment against a recorded 295 M-step reference, not against a
short proxy.

| | `boy_chase01` (shipped) | `e6a_twosided_long` |
|---|---|---|
| Speed kernel | one-sided | **two-sided** |
| Everything else | — | identical |
| Result | 2.043 m/s (**+36 %** vs command) | pending |

**Prediction, recorded before the result:** mean forward speed should land near
1.5 m/s rather than 2.04. The honest risk is that the symmetric kernel also
*slows learning* — capping the reward for speed removes a gradient the policy
was using to discover locomotion at all — in which case uptime and speed both
come in lower and the correct conclusion is that the one-sided kernel was a
deliberate, load-bearing choice rather than an oversight. Both outcomes are
informative; only one is an improvement.

---

## Phase 2 — planned experiment matrix

Ordered by expected value per unit of risk. Each is a 5–10 min validation run
against the baseline before anything longer is committed. One variable at a
time, same seed, same iteration count, compared on the K1–K9 dashboard.

| # | Change | Hypothesis | Risk |
|---|---|---|---|
| E1 | `--worlds 16384` | +20 % throughput at no learning cost — more samples/iter at the same wall-clock | Low; only VRAM |
| E2 | `--entropy` schedule (2e-3 → decay) | Fixed entropy keeps late-training exploration noise; decaying it should tighten heading error (K4) and control effort (K6) | Low |
| E3 | `--lr` 3e-4 → 1e-3 with more minibatches | Warp gives huge batches (197k–590k samples/iter); the default lr may be under-stepping | Medium; can destabilize |
| E4 | `--rollout` 24 → 48 | Longer horizon improves GAE credit assignment for a 1000-step episode; 24 steps is 0.48 s of a 20 s episode | Medium; doubles buffer VRAM |
| E5 | `--lam` 0.95 → 0.98 | Same reasoning as E4, cheaper | Low |
| E6 | Reward reweight, guided by baseline KPI | Only after the dashboard shows which term dominates. **Not speculative** — deferred until there is evidence | Medium |

**Explicitly NOT doing**, because the audit found them already satisfied:

- *Vectorize environments* — already GPU-vectorized, host-free rollout (§1.6).
- *Eliminate per-step Python allocations* — the rollout loop preallocates every
  buffer and the env returns zero-copy views; my own instrumentation was written
  to preserve this.
- *Contact `<exclude>` tags* — MuJoCo's default filter already excludes
  parent–child pairs; contact budget is measured with 2.9× headroom (M2).
- *Anatomical joint limits and damping* — present and sane in both MJCFs.
- *Run headless* — already headless.
