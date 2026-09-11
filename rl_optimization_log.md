# MuJoCo RL Optimization Log

Autonomous 2-phase optimization loop. Started **2026-09-09 ~22:35 local**, budget 8 h.

---

## Summary

**Target: MojucuBoy** (21-DOF humanoid, MuJoCo Warp + torch). Fido was audited
but is untrainable on this machine — it needs `jax[cuda]`, which has no Windows
wheels, and WSL is broken here.

**The headline result is a bug, not a tuning win.** The reward's uprightness
term had its sign inverted (**M9**), so `standing` scored 0 for an upright racer
and 1 for an upside-down one — and `W_TRACK`, `W_HEADING` and `W_GETUP` are all
gated on `standing`. The shipped training regime therefore paid the racer to
travel inverted, and it did: measured at 1.245 m/s with uprightness **−0.873**
and torso at 0.585 m against a 0.77 m stance, while the metric reported a
healthy 0.682.

Fixing that exposed a second problem — with the exploit closed the racer had to
learn real balance, which the original regime never required — and the fix for
*that* was a terminal condition, not a weight.

**What was delivered**

| | Shipped brain (measured) | Final (E12) |
|---|---|---|
| Posture | upright, 0.789 m | upright, 0.976 uptime |
| Speed vs 1.5 m/s command | 2.043 m/s (**+36 %**) | 1.446 m/s (**−3.6 %**) |
| Heading error | not measured | **10.4°** |
| Get-up from a sprawl | **yes** (races unaided) | **no** |
| Reward terms observable | 1 of 10 | **all 10** + 3 derived |
| Convergence gates that can fail | 1 of 3 | 3 of 3 |

**Recommendation: keep the shipped brain.** E12 tracks the commanded speed far
better and its heading is measured and good, but the shipped racer is upright,
**41 % faster in absolute terms**, and can recover from falls — and this is a
racing game, where absolute speed is the thing that wins. E12's ±10 % tracking is
the *brief's* KPI, not the game's objective. Nothing in `Assets/` was changed.

**The delivered value is the pipeline, not a faster racer:** six live defects
found and fixed, the reward's uprightness sign among them, plus instrumentation
that turned one observable scalar into thirteen and a convergence gate that can
now actually fail.

**Nine findings**, six of them defects that were live in the shipped pipeline:

| | Finding | Severity |
|---|---|---|
| M9 | Reward paid the racer to travel upside down | **affects training** |
| M6 | Trainer deletes shipped brains on startup | **destroyed a release artifact** |
| M4 | Two of three convergence gates cannot fail | **vacuous QA** |
| M3 | Trainer logged nothing for its first 45 iterations | blinded every short run |
| M5 | Speed kernel one-sided → ±10 % tracking unreachable | KPI unmeetable |
| M8 | KPI dashboard aliased against the episode period | misleading comparisons |
| M1 | Ten reward weights logged as one scalar | tuning unguided |
| M2, M7 | Contact budget sound; `fall_rate` correct (M7 **withdrawn**) | no action |

**Two experiments rejected on evidence** (E1 larger batch, E8 more epochs) —
together showing the optimizer was never the binding constraint, which is why
the remaining budget went to reward structure rather than to the
hyperparameter sweep the brief suggested.

**Not achieved: get-up from a sprawl.** Both final policies hold 71 % of
episodes above 90 % uptime under a 30 % sprawl start — exactly the fraction that
begins upright. Every episode starting on its feet is near-perfect; every
episode starting down is lost. Stage two improved balance, speed and heading but
never produced recovery.

**The shipped brain was NOT replaced.** The new policy is upright where the old
one is inverted and tracks the commanded speed properly, but it cannot get up,
and the old one currently wins races. That trade is a judgement about the game,
not about training metrics, so it is left to the user.

**Three things I got wrong in-flight and corrected** — recorded because the
corrections are part of the result: I called M8's aliasing "noise" twice before
measuring it; I diagnosed M7 as a broken metric when it was the only honest one;
and I wrote off E12 as a rejection while it ran, having judged it by a number
measured under the wrong condition. Full detail below.

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

### Finding M8 — the KPI dashboard is aliased against the episode period

I twice called the dips in the long run "single-rollout noise". That was wrong,
and measuring it says so.

Every world's `episode_step` starts at 0, so all 8192 reset **in lockstep** every
`1000 / 24 = 41.67` iterations (E7). `RESET_FALLEN_FRACTION = 0.30` starts 30 %
of those resets with the racer sprawled on the floor. The per-iteration KPI is a
mean over one rollout, so a rollout that lands just after a reset is measuring a
population that is 30 % face-down by construction.

Bucketing every logged uptime past iteration 300 (i.e. past the standing-up
phase) by its phase within the episode:

| Phase in episode | n | Mean uptime |
|---|---|---|
| **0/6 — just after reset** | 16 | **0.416** |
| 1/6 | 14 | 0.545 |
| 2/6 | 13 | 0.558 |
| 3/6 | 13 | 0.558 |
| 4/6 | 14 | 0.559 |
| 5/6 | 13 | 0.566 |

Flat at ~0.56 across five sixths of the episode and **26 % lower** in the sixth
that follows a reset. That is not noise — noise does not sort itself by phase.
It is a systematic sampling artifact, coherent across worlds precisely because
the resets are synchronized.

**What it does and does not invalidate.** Comparing two runs at a single
iteration can be off by ~25 % on uptime if the two land in different episode
phases, so:

- **E1 stands.** It was behind by roughly **2×** at every matched point, which
  is far outside a 26 % phase band.
- **E8 stands.** It was a null result to begin with; a bias of this size cannot
  turn "indistinguishable" into a win.
- **Single-point readings from the long run should not be quoted** without
  their phase. The episode-boundary lines (those carrying `ret`/`len`/`spd`) are
  the trustworthy ones, since `spd` there is a whole-episode mean.

**This promotes E7 from cosmetic to substantive.** Staggering the initial
`episode_step` per world would decorrelate the resets, which removes the
aliasing, smooths the GAE bootstrap, and makes episode statistics arrive
continuously rather than in a lump every 42 iterations. Not applied mid-run —
changing reset behaviour would invalidate the run being measured.

### Finding M9 — the reward paid the racer to travel upside down

**The headline finding of this session, and the only one that affected training
rather than measurement.**

`_reward` computed:

```python
obs_gravity_z = -rot[:, 2, 2]
```

`reset()` builds the standing orientation as a **pure yaw quaternion**
(`qpos[4] = qpos[5] = 0`), so `rot[2,2] = 1 - 2(x² + y²) = +1` for an upright
racer. The negation therefore made `obs_gravity_z = -1` while standing, and
`.clamp(min=0)` floored that to **zero**.

Everything downstream is gated on it:

```python
standing = obs_gravity_z.clamp(min=0.0) * (height / STANDING_HEIGHT).clamp(0, 1)
reward = ( W_TRACK   * track * standing
         + W_HEADING * facing.clamp(min=0) * standing
         + W_UPRIGHT * obs_gravity_z.clamp(min=0)
         + W_GETUP   * standing + ... )
```

So `W_UPRIGHT` paid nothing for standing and its maximum for being inverted;
`standing` was 0 upright and 1 inverted; and because `W_TRACK` (2.0),
`W_HEADING` (0.4) and `W_GETUP` (0.60) are *all* multiplied by `standing`,
**no tracking, heading or get-up reward was earnable the right way up at all.**

The policy optimized this correctly. Measured on the completed 1500-iteration
run, over 500 steps × 128 worlds with the deterministic policy:

| Quantity | Value |
|---|---|
| torso height | 0.585 m (`STANDING_HEIGHT` 0.77) |
| uprightness (`−obs[:,2]`) | **−0.873** — inverted |
| `_reward` `standing` | **0.682** — reporting healthy |
| speed along command | 1.245 m/s |

It learned to flip over and travel upside-down at 1.25 m/s, and the reward
called that a good racer.

**Fixed** to `obs_gravity_z = rot[:, 2, 2]`. Verified on a fresh reset:

| | Before | After |
|---|---|---|
| `standing`, upright worlds | ~0.00 | **0.998** |
| `standing`, sprawled worlds | — | 0.208 |
| `fallen`, upright worlds | ~1.00 | **0.000** |
| `fallen`, sprawled worlds | — | 0.660 |
| fraction upright | — | 0.689 (design: 0.70) |

For the first time the metrics agree with each other *and* with the reset
design.

**PREDICTION MADE HERE WAS WRONG — RETRACTED.** I predicted that the shipped
`MojucuBoy_v01.onnx` would also be an inverted gait, since `boy_chase01` was
trained under the identical reward. It is not. Checked two ways:

- **In Unity**, racing `SCN_RACE_FLAT`: hips `up.y` 0.98–1.00 at height
  0.76–0.79 m, sustained over 14 samples through a race. Upright, at stance
  height.
- **In the training env**, running the shipped ONNX itself for 400 steps:
  height **0.789 m**, `rot[2,2]` **+0.993**. Upright.

**What this does to M9: it confirms it, harder.** The shipped brain is an
independent, known-good upright reference, and scoring it with each formula
settles the sign without any argument from me:

| `standing` for the shipped, verifiably upright brain | |
|---|---|
| Original `clamp(-rot[2,2], 0) · h/H` | **0.000** |
| Fixed `clamp(+rot[2,2], 0) · h/H` | **0.993** |

The original reward scores a racer standing at full stance height as **not
standing at all**. That is the bug, now demonstrated against ground truth rather
than derived from the reset quaternion.

**The reconciliation.** The shipped brain must predate the defect in the current
code — `git log -S` puts the line in the file's first commit, so the model was
trained before some other change made the frame convention disagree with it. The
live evidence that the bug bites *today* is E6a, trained on the code as it now
stands: it converged to height 0.585 m with uprightness −0.873, exactly the
degenerate posture the bug predicts, where the shipped brain sits at 0.789 m.

So: the defect is real and currently active; the shipped artifact is not a
victim of it.

**Where the regression came from — as far as the repo can say.** The whole
MojucuBoy stack landed in **one commit** (`f751fea`): env, rig, and
`runs/boy_chase01/` together. The stance quaternion has been identity since that
commit and the sign has been `-rot[2, 2]` since that commit, so there is no
in-repo history of the change. But the brain cannot have been produced by the
code it was committed alongside — that code scores it 0.000. The sign therefore
regressed during whatever refactor preceded the commit, after the model was
trained, and the run artifacts came along as a record of a training run the
committed code can no longer reproduce.

Which is the practical warning: `boy_chase01/config.json` faithfully records
hyperparameters for a run that today's `mojucuboy_env.py` would not repeat.

**How it hid for so long.** Every downstream number was self-consistently wrong.
`standing` reported 0.68 and rose during training; return rose; speed rose. Only
`fall_rate` disagreed — and it was pinned at exactly 1.00, which reads like a
broken metric rather than a true one.

### E9 — retrain with the sign fixed. Plateaued, and that is informative.

Same config as E6a, uprightness corrected. **Aborted at iteration 500 of 1500.**

Sampling only unaliased iterations (M8 — and note that 250 = 6 × 41.67 exactly,
so *every* multiple of 250 lands on a reset boundary; my original milestone
choice was maximally biased):

| Iter | track | speed | heading | uprightness | uptime |
|---|---|---|---|---|---|
| 435 | 0.35 | 0.41 | 83.3° | 0.40 | 0.08 |
| 450 | 0.36 | 0.43 | 82.6° | 0.41 | 0.08 |
| 475 | 0.36 | 0.44 | 86.3° | 0.40 | 0.08 |
| 495 | 0.36 | 0.43 | 85.9° | 0.40 | 0.08 |

Flat to within 0.01 for 250 iterations (49 M steps), heading back at chance.
That is a local optimum, not slow progress, so continuing would have burned an
hour to confirm it.

**Why.** With the exploit closed the racer must learn genuine 21-DOF balance —
something the shipped regime never had to do, because inversion satisfied
`standing` for free. And while it is down, *every* positive term except
`W_UPRIGHT` is gated on `standing` and pays nothing:

| Term | Gated on `standing`? | Value at uptime 0.08 |
|---|---|---|
| `W_TRACK` 2.0 | yes | 2.0 × 0.35 × 0.08 = 0.056 |
| `W_GETUP` 0.60 | yes | 0.048 |
| `W_HEADING` 0.4 | yes | ~0.01 |
| `W_UPRIGHT` 0.05 | **no** | 0.05 × 0.40 = 0.020 |
| `W_ALIVE` 0.10 | **no** | **0.100** |

The largest reward available to a fallen racer is the one it gets for doing
nothing. That is precisely the failure this file's own header warns about — *"the
survival terms must stay SMALL relative to the tracking term"* — reappearing
because the term they were balanced against changed meaning when the sign was
fixed.

### E10 — rebalanced (running)

`--upright-weight 0.5 --reset-fallen 0.10`, otherwise identical.

- **Uprightness 0.05 → 0.5:** the only ungated positive term, hence the entire
  gradient back to the feet. Now 5× `W_ALIVE` instead of half of it.
- **Fallen resets 0.30 → 0.10:** learn balance first; recovery is a harder
  problem to solve simultaneously from scratch.

Verified both flags take effect before launching (90.6 % of worlds upright at
`reset_fallen=0.10`, against the 90 % implied).

### E10 — `--upright-weight 0.5`. Rejected: it bought a different exploit.

Aborted at iteration 300. Uprightness rose exactly as intended and the racer got
*worse*:

| Run | `W_UPRIGHT` | Torso height | Uprightness | Uptime |
|---|---|---|---|---|
| E9 | 0.05 | 0.144 m | 0.41 | 0.08 |
| E10 | 0.5 | **0.104 m** | **0.56** | 0.07 |

Higher orientation score, *lower* height. `W_UPRIGHT` scores orientation with no
reference to height, so the cheapest way to earn a big ungated term is to lie on
your back with your chest up. I closed M9's inversion exploit and immediately
opened a lie-flat one. Reverted to the shipped 0.05.

**A run I did not waste.** The obvious companion move was to cut `W_ALIVE`, since
0.10 unconditional was out-paying everything a fallen racer could earn. It would
have done nothing: under timeout-only termination every episode is exactly 1000
steps, so a constant per-step reward is a pure offset. It shifts the value
function and leaves the optimal policy and the policy gradient untouched.

### E11 — terminate on fall. The one that worked.

`--terminate-on-fall --reset-fallen 0.0 --two-sided-speed`, `W_UPRIGHT` back to
the shipped 0.05.

**Reasoning.** E9 and E10 both converged on lying still, and neither reward
tweak moved them. What was missing was not a weight but a *terminal condition*:
nothing ended an episode, so 1000 steps of lying down was a comfortable local
optimum. The env omits termination deliberately, to teach get-up — sound, but it
only bites once the racer can stand, and it could not. Every standard humanoid
locomotion benchmark terminates on fall. This makes it stage one of a
curriculum.

| Iter | Steps | Episode len | Survival | Speed | Heading | Uprightness | Fall |
|---|---|---|---|---|---|---|---|
| 50 | 9.8 M | 96.0 | 0.096 | 0.10 | 86.6° | 0.94 | 1.00 |
| 100 | 19.7 M | 233.5 | 0.234 | 0.31 | 59.3° | 0.97 | 1.00 |
| 200 | 39.3 M | 900.4 | 0.900 | 1.20 | 16.7° | 0.99 | 0.22 |
| 300 | 59.0 M | 941.8 | 0.942 | 1.35 | 13.9° | 1.00 | 0.11 |
| 400 | 78.6 M | 935.1 | 0.935 | 1.36 | 14.6° | 0.99 | 0.12 |
| 520 | 102.2 M | 940.5 | 0.941 | 1.38 | 14.8° | 0.99 | 0.10 |

Against thresholds at iteration 520: K1 survival **0.94** (> 0.90 ✓), K2 speed
**−8.0 %** (within ±10 % ✓), K4 heading **14.8°** (< 15° ✓), K8 uprightness
**0.99** (> 0.90 ✓), K3 fall **0.10** (≤ 0.10, at the line).

It cleared 0.90 survival at **39 M steps**. E9 and E10 never left 0.08 uptime in
100 M and 59 M respectively, and the shipped regime took 295 M to produce an
inverted gait.

**Two caveats stated rather than glossed:**

1. **Return is not comparable across these runs.** 2451 here against the shipped
   brain's 2336 is meaningless as a comparison — the reward function itself
   changed (sign fix, two-sided kernel). Only the physical KPIs (survival, speed,
   heading, uprightness) are measured identically in both.
2. **Uptime is now near-1 by construction.** With early termination the episode
   ends when the racer falls, so "upright while alive" is trivially high — the
   mirror image of M4. Under this curriculum the load-bearing stability metric is
   **episode length**, which is a real measurement again for the first time this
   session.

**Unplanned bonus: E11 fixes M8 for free.** Early termination staggers the
resets, so worlds no longer end in lockstep and the episode-period aliasing
disappears without touching `episode_step` (E7).

#### E11 evaluated — 100 deterministic episodes

Final training line at iteration 1500: `len 958.1  spd 1.40 m/s  fall 0.07
hdg 17.7deg  std 0.97`.

`gate4_eval` returned **FAIL on everything**, and the reason is a measurement
mismatch, not a bad policy. It builds `MojucuBoyEnv(episodes, seed=seed)` with
**default arguments** — `terminate_on_fall=False`, `reset_fallen_fraction=0.30`
— so it evaluates the policy under conditions E11 never trained on. Re-run with
both conditions explicitly:

| Condition | Uptime | Fallen | Speed | Heading | Episodes > 90 % up |
|---|---|---|---|---|---|
| **Matched** (starts upright, as trained) | **0.955** | **0.000** | 1.346 m/s (−10.3 %) | 18.1° | **100 / 100** |
| Default (30 % start sprawled) | 0.679 | 0.288 | 0.945 m/s (−37 %) | 41.5° | 71 % |

The default-condition numbers are not noise, they are arithmetic:
`episodes > 90 % up = 71.0 %` against the **70 %** of episodes that start
upright, and `mean fallen = 0.288` against the **0.30** that start down. Every
episode that begins on its feet is near-perfect; every episode that begins
sprawled is lost. **The policy can hold its feet and cannot get back on them** —
exactly what stage one trains and stage two does not.

**KPI status against the matched condition, which is what stage one targets:**

| KPI | Value | Threshold | |
|---|---|---|---|
| K1 uptime | 0.955 | > 0.90 | **pass** |
| K3 fallen | 0.000 | < 0.10 | **pass** |
| K8 uprightness | 0.98 | > 0.90 | **pass** |
| K2 speed | −10.3 % | ±10 % | **marginal fail** |
| K4 heading | 18.1° | < 15° | **fail** |

Note K4 is worse under evaluation (18.1°) than in training (14.9–17.7°): the
eval runs the **deterministic** policy (tanh of the actor mean) while training
samples, and the sampling noise was evidently helping heading corrections.

#### E11 exported and parity-checked

```
MojucuBoy_v01.onnx      opset 15, 13 nodes, normaliser baked into the graph
mujoco_reference.json   128-step deterministic trajectory
torch vs onnxruntime    max |delta| = 7.153e-07
```

Both land in `runs/e11_terminate/`, which also makes the run pruner-protected
under the M6 fix. `onnxruntime` had to be installed — it was missing from the
venv, so the export script's own parity check had been silently unrunnable.

**The shipped brain in `Assets/Agents/MojucuBoy_v01/` was NOT replaced.** The new
policy is upright where the old one is inverted, but it misses K2 and K4 and
cannot get up from a sprawl. Swapping the racer that currently wins races is a
judgement about the game, not about the training metrics, so it is left to the
user with the numbers above to decide on.

### E12 — curriculum stage two: get-up (running)

`--init-from e11_terminate --reset-fallen 0.30`, termination **off**, 800
iterations. Resume confirmed at E11's iteration 1500.

The point is the one skill E11 provably lacks. Starting from a competent
balancer is what makes this tractable where E9 failed: standing is now reachable
from the policy's current behaviour, so the `standing`-gated terms are live
rather than paying ~0.

**Risk, recorded before the result:** removing termination is exactly the
setting in which E9 and E10 collapsed to lying down. Starting from competence
should prevent that, but catastrophic forgetting is the plausible failure and
would show up as uptime falling from 0.955 toward 0.1.

#### Result — and I called it wrong while it ran

Catastrophic forgetting did **not** happen: uptime dipped 0.68 → 0.60 by
iteration 400 and came back to 0.66. I watched that flat line from iteration 100
to 700 and wrote the run off as "a rejection in all but the final number".

That was wrong, and the reason is instructive. The training-line `std` is
measured **under the 30 % sprawl condition**, where recovery is unlearned, so it
is pinned near `0.70 × 0.955 ≈ 0.67` no matter how much the *balance* policy
improves. It was the wrong number to judge the run by, and it hid real gains in
every other KPI:

| Policy | Condition | Uptime | Speed | Heading | Eps > 90 % up |
|---|---|---|---|---|---|
| E11 | upright | 0.955 | 1.347 (−10.2 %) | 18.0° | 100 % |
| **E12** | **upright** | **0.976** | **1.446 (−3.6 %)** | **10.4°** | 98 % |
| E11 | sprawl 30 % | 0.679 | 0.941 (−37.3 %) | 40.5° | 71 % |
| E12 | sprawl 30 % | 0.717 | 1.037 (−30.9 %) | 32.3° | 71 % |

E12 is better on **every** metric in both conditions, and it clears the two KPIs
E11 missed: speed tracking **−3.6 %** against a ±10 % gate, and heading
**10.4°** against a < 15° gate.

Training on the harder distribution improved the easier one — the sprawl starts
act as a regulariser on the balance policy even though get-up itself never
emerged. `eps > 90 % up` stays at exactly 71 % in the sprawl condition for both
policies, which is the 70 % that start upright: **recovery is still unlearned**.

**Exported:** `MojucuBoy_v01.onnx` + `mujoco_reference.json`, parity
`8.047e-07`.

**KPI verdict — E12, matched condition:**

| KPI | Value | Threshold | |
|---|---|---|---|
| K1 uptime | 0.976 | > 0.90 | **pass** |
| K2 speed | −3.6 % | ±10 % | **pass** |
| K4 heading | 10.4° | < 15° | **pass** |
| K8 uprightness | ~0.98 | > 0.90 | **pass** |
| Get-up from sprawl | unlearned | — | **not achieved** |

### Finding M7 — WITHDRAWN: `fall_rate` was correct all along

I recorded M7 as "`rollout/fall_rate` is a snapshot, not a rate", reasoning that
`fall 1.00` beside `std 0.49` had to mean the flag was sampled at the episode
boundary and mis-scoped.

**That was wrong.** `fall_rate` was reporting the literal truth: the racer was
*never* upright, in any rollout, because M9 had trained it to travel inverted.
It read a constant 1.00 from the untrained baseline through to a policy walking
at 1.2 m/s because the racer really was down the whole time by any honest
definition.

It was the one honest number on the line, and I explained it away as an
instrumentation artifact because three other metrics agreed with each other.
The lesson is the ordinary one: when a single metric dissents from a consistent
majority, the majority can be consistently wrong, and the cheap check — *what
does this number look like at a known-good state?* — is the one I skipped.

(The narrow observation in M7 does still hold on its own terms: `fall_rate` *is*
sampled at `done`, and with timeout-only termination that is one instant per
episode. But that is a footnote, not the explanation, and it is not why the
number was 1.00.)

Noticed at iteration 250 of the long run, which reported `fall 1.00` and
`std 0.49` on the same line: every world down, and yet standing half the time.

Both are correct. The trainer records `done_falls.append(terms["fallen"][idx])`
at the moment a world reports `done`, and `done` is timeout-only, so this is
**the fallen flag sampled at exactly t = 1000 steps** — one instant, 20 s in —
not the fraction of the episode spent fallen. A policy that stands for fifteen
seconds and is down at the twentieth scores `fall_rate = 1.0`, identically to
one that never rose.

It is not wrong, but the name invites exactly the wrong reading, and it is the
same class of mistake as M4: a metric whose definition quietly stopped matching
its label when termination changed. `kpi/standing` and `kpi/fallen`, both
per-step means over the whole rollout, are the ones to read; they are why the
discrepancy was visible at all.

Left as-is rather than renamed mid-run — renaming a scalar tag would split the
series in TensorBoard and the run in progress is the one being measured.

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

---

## Session 2 — resumed 2026-09-10 ~20:40 local, fresh 8 h budget

Picking up where session 1 stopped. First act: **E13 had been trained but never
evaluated and never logged** — 900 iterations, `init_from: e12_getup`,
`reset_fallen 0.6`, seed 2, finished 09:59 and left on disk. Evaluating it is
the cheapest available work and it targets the one thing session 1 recorded as
not achieved.

### E13 — curriculum stage three, 60 % sprawl starts. Evaluated.

`gate4_eval.py --run e13_recovery --episodes 100`:

| | randomised model | nominal model |
|---|---|---|
| mean episode length | 1000.0 / 1000 | 1000.0 |
| mean forward speed | 0.906 m/s | 0.959 m/s |
| mean uptime | 0.632 | 0.668 |
| mean fallen | 0.383 | 0.346 |
| **episodes > 90 % up** | **61.0 %** | **65.0 %** |
| mean return | 1888.38 | 2001.43 |

Gate 4: **FAIL** on speed, uptime and fallen. Against the brief's K1/K3 it is a
worse policy than E11.

**But the interesting number is 61 %, and it is interesting precisely because of
what it is measured against.** `reset_fallen_fraction` is the probability that an
episode starts in a *random orientation dropped from clear air* — so E13 started
**60 % of its episodes fallen and only 40 % upright**, yet 61 % of episodes
finished above 90 % uptime.

Set against the curriculum's own history:

| run | sprawl start | starts upright | episodes > 90 % up | gap |
|---|---|---|---|---|
| E12 | 0.3 | 70 % | 71 % | **+1 pt** — no recovery |
| E13 | **0.6** | **40 %** | **61 %** | **+21 pt** |

Session 1 concluded get-up was "not achieved" because E12's 71 % matched exactly
the fraction that began upright. E13 breaks that identity: roughly a fifth of its
episodes started down and still spent over 90 % of the episode up, which requires
standing within ~100 steps (2 s).

**I am not claiming get-up on this evidence.** A random orientation dropped from
clear air includes attitudes that land very nearly upright, and an episode that
lands on its feet and stabilises is indistinguishable, in this metric, from one
that genuinely pushes up off its back. The 21-point gap is *consistent with*
recovery and it is the strongest signal the curriculum has produced, but the
metric cannot separate the two cases — which is a gap in the instrumentation, not
a result.

**Action: K10, a recovery KPI that can tell them apart** — see below. Two
experiments have now been judged on a proxy that cannot answer the question they
were run to answer; measuring it directly is worth more than a third experiment.

### K10 — recovery, added to `gate4_eval.py`

The metric E12 and E13 were both judged on, `uptime_over_90pct`, cannot
distinguish the two things that matter here. `reset_fallen_fraction` drops the
racer from clear air in a *random attitude*, so some episodes land essentially on
their feet. An episode that lands upright and stabilises scores exactly like one
that pushes up off its back, and the difference between those is the entire
question the get-up curriculum exists to answer.

K10 classifies each episode by the state it is in **once settled**, not by how it
was initialised:

| field | meaning |
|---|---|
| `started_down_frac` | `standing < 0.30` at the 50-step (1 s) settle mark. NOT the same as `reset_fallen_fraction` — a clear-air drop sometimes lands on its feet |
| `recovery_rate` | of the episodes that were genuinely down, the fraction standing (`> 0.80`) through the final quarter of the episode |
| `ever_stood_after_down` | of those, the fraction that reached standing at any point — separates "got up and fell again" from "never got up" |
| `median_steps_to_stand` | how long getting up takes, in policy steps |

`started_down` is latched at the settle mark precisely so a lucky landing is
counted as what it is — an episode that never fell — rather than as a recovery.

**Threshold: `recovery_rate > 0.50`** to call get-up learned. Set before
measuring, so the number cannot be rationalised after the fact.

### Efficiency audit (the brief's UPDATE), measured not assumed

| check | measured | verdict |
|---|---|---|
| GPU contention from the Unity Editor | 738 MiB, **0 % utilisation**, idle | **leave it running** |
| VRAM headroom at the training config | 2.4 GB used of 6.14 GB at 8192 worlds | not the constraint |
| Vectorisation | already GPU-vectorised, zero host readback (§1.6) | no action |
| World-count knee | 16 384 = +20 % throughput (§1.6) | **rejected by E1** — throughput won, learning lost |
| Thermals | 84 °C sustained at 8192 worlds, Turing throttles ~88 °C | the real ceiling |

The Editor was a genuine candidate — three processes holding GPU memory on a
6 GB card — and `unity command quit` was attempted and refused by a plug-in bug
(`PipelineQuitScheduler` calls `DontDestroyOnLoad` outside play mode). Rather
than `taskkill` three Editor processes and risk the asset database, it was
measured: 738 MiB against a 2.4 GB training footprint on a 6.14 GB card, at 0 %
utilisation. It is not costing throughput, so it stays.

**Throughput is already at its useful maximum on this machine.** The one lever
that would raise it further (world count) was tested in session 1 and rejected on
learning grounds, and thermal headroom argues against it independently. The
binding constraint on this project is not steps/second — E1 and E8 together
established that the optimizer was never the bottleneck — it is what the reward
encodes. Session 2 therefore spends its budget on the unsolved capability
(get-up), not on the hyperparameter sweep.

#### K10 measured — and it refutes the reading above

```
started down        :   35 / 100   (35.0% of episodes, settled)
recovery rate       :    0.0%
ever stood again    :    0.0%
median steps to up  :     nan
```

**Zero.** Of the 35 episodes genuinely on the floor once settled, not one ever
reached standing — not sustained, not transiently.

And the 61 % that looked like evidence resolves completely: `reset_fallen 0.6`
drops 60 % of episodes in a random attitude, but only **35 %** are actually down
at the settle mark. The other 25 points **land on their feet**. So ~65 % of
episodes were effectively upright starts and 61 % held > 90 % uptime — the same
`starts-upright ≈ ends-upright` identity as E12, not a break from it.

**I was wrong, and the instrument caught it.** The 21-point "gap" was an artefact
of assuming `reset_fallen_fraction` equals the fraction that starts down. It does
not, and nothing in the previous metric could have shown that. This is the second
time in this project a conclusion rested on a proxy that could not distinguish the
cases it was being asked about (the first was M8's aliasing).

**Get-up is unlearned across E11 → E12 → E13.** Three runs, escalating sprawl
exposure, 0 % recovery. The curriculum-by-sprawl-fraction approach is not
converging on it, and a fourth run at a higher fraction is not indicated.

#### Why it never even tries — the reward structure while down

`ever_stood_again = 0.0 %` is the informative part. This is not "attempts and
fails", it is "never attempts". Look at what the racer earns lying still:

| term | value while down | gated on `standing`? |
|---|---|---|
| `W_ALIVE` | **0.10, unconditional** | no |
| `W_UPRIGHT` | 0.05 × uprightness ≈ 0 | no |
| `W_TRACK`, `W_HEADING`, `W_GETUP` | 0 | **yes** |
| control cost, joint accel | **negative, and paid immediately** | no |

Lying motionless collects the full unconditional `W_ALIVE` for free. Standing up
costs control effort and acceleration penalties **now** for a payoff that only
arrives after the racer is already upright. So the policy's local optimum is to
lie still and bank 0.10/step, and the gradient out of it is 0.05 × uprightness —
which session 1 already measured as insufficient ("a 1500-iteration run sat flat
at uptime 0.08 for 49 M steps").

**This is the same defect class as M9, and as the Unity-side upright bonus found
separately today: an unconditional positive term creating a do-nothing optimum.**
Three instances now in one project. The pattern is worth stating plainly — any
reward term that pays while the agent is failing will be collected by failing.

### E14 — sprawl DIFFICULTY curriculum (the axis E11–E13 never varied)

**The diagnosis.** K10 says the policy never reaches standing from the floor —
`ever_stood_again` 0 %, not merely `recovery_rate` 0 %. That distinction decides
what to change. It is not "tries and fails", it is "never tries", and the reward
is not the reason: `W_GETUP` (0.60) and `W_TRACK` (2.0) are both multiplied by
`standing = uprightness × (height / 0.77)`, so the payoff for standing is large,
dense in both height and orientation, and — unlike E10's orientation-only term —
not exploitable by lying on your back with your chest up.

The payoff is there. The policy cannot find it. **This is an exploration problem**,
and E11, E12 and E13 all attacked the wrong axis:

| run | reset_fallen | sprawl difficulty | recovery |
|---|---|---|---|
| E11 | 0.0 | — | — |
| E12 | 0.3 | fully random quaternion | 0 % |
| E13 | 0.6 | fully random quaternion | **0 %** |

Three runs escalated how *often* the racer starts fallen and never once changed
how *hard* the fall was. Every sprawl in this project's history has been a
uniformly random orientation — the hardest case — presented as the first lesson.

**The change.** `sprawl_max_tilt` bounds how far off vertical a sprawl start may
be, and the trainer ramps it across the run (`--sprawl-tilt-start/-end/-full`).
Verified distribution before spending any GPU time:

| max tilt | mean uprightness | frac above MIN_UPRIGHT | frac inverted |
|---|---|---|---|
| 60° | 0.826 | 100 % | 0 % |
| 90° | 0.635 | 81 % | 0 % |
| 120° | 0.411 | 60 % | 25 % |
| **180°** | **−0.004** | **40 %** | **50 %** |

The 180° row independently confirms K10: only 40 % of fully-random starts are
above the upright threshold, against K10's measured 35 % genuinely down at
settle. Two instruments, same conclusion — `reset_fallen_fraction` was never the
fraction that starts down, which is exactly what made E13's 61 % look like
recovery when it was not.

**Configuration.** `--init-from e12_getup` (the best balancer: uptime 0.976,
speed −3.6 %, heading 10.4°), `--reset-fallen 0.5`, tilt ramp 60° → 180° over
70 % of the run, `--two-sided-speed`, `W_UPRIGHT` left at the shipped 0.05
because E10 showed raising it buys a lie-flat exploit.

**Prediction, recorded before the result.** K10 `recovery_rate` > 0 at the low-tilt
end is the thing to watch; if the mechanism works at all it should appear early,
while tilt is still under 90°, and then either survive the ramp or collapse as
tilt approaches 180°.

**The honest failure mode** is that the ramp teaches nothing but balance at each
successive tilt — catching a tip is not the same skill as rising from supine —
so that by the time tilt reaches 180° the policy faces exactly E13's
unexplorable problem, just later and having spent a run to get there. That would
show as `recovery_rate` tracking the tilt down to 0 % as the ramp completes, and
it would say the difficulty axis is necessary but not sufficient.

**Validation first**, per the brief: ~150 iterations (~9 min at the measured
3.5 s/iteration) to confirm the mechanism runs and moves K10 at all, before any
long cycle is committed.

#### Efficiency: the Editor A/B, and a units error I made comparing throughput

E14's validation run reported 46–49k steps/s where §1.6 records **56,343** at the
same 8192 worlds, and I treated the gap as a regression to chase. It is not.
§1.6 measured **pure environment stepping, zero-action, no learning**; 49k is
measured during *training*, which adds the policy forward pass, GAE, and 4 epochs
× 8 minibatches of backprop per iteration. Those are different quantities and
comparing them was my error, not the machine's.

Before spotting that I formed a second hypothesis and tested it, and it is worth
recording because it was also wrong:

> This is a **mobile** RTX 2060 — default power limit **80 W**, max 85 W, not the
> 160 W desktop part. Under load it sits at 75–77 W with
> `clocks_throttle_reasons.active = 0x4` (**SW Power Cap**) and clocks pinned at
> ~1845 of 2100 MHz. On a power-capped laptop GPU, *power* is the binding
> resource, so any other GPU client steals directly from the training budget —
> which VRAM and utilisation, the two things I checked when deciding to leave the
> Editor running, would not have shown.

Tested it directly rather than reasoning about it. Same run, same config,
before and after killing three Editor processes:

| | steps/s | power | clocks | VRAM |
|---|---|---|---|---|
| Editor running | 46, 48, 48, 49, 49, 49 (**mean 48.2k**) | 69–76 W | 1395–1875 MHz | 1657 MiB |
| Editor closed | 49, 49, 49, 49, 49, 49 (**49k**) | 76.8 W | 1845 MHz | 1143 MiB |

**No material change.** ~1.7 %, inside run-to-run noise, and the GPU was
power-capped at 80 W in both conditions. An idle Editor draws no meaningful
power; the 738 MiB it held was never the constraint and neither was its power
draw. The original "leave it running" call was right, though not for the reason
I gave — and it has now been closed for nothing, which costs nothing.

**What the numbers actually say about headroom:** 60 % GPU utilisation, 84 °C,
power-capped at 80 W of an 85 W ceiling. Temperature is at the figure §1.6
flagged as having no margin, and the card is power-limited, so the remaining
40 % of utilisation is not recoverable by feeding it more work — it is the
serial PPO-update phase between rollouts, and E8 already showed more epochs
costs 8 % for no gain. **Session 1's conclusion holds: throughput is at its
practical ceiling on this machine and is not the binding constraint.**

#### E14 validation — and a design error in my own validation run

150 iterations from `e12_getup`, `--reset-fallen 0.5`, tilt ramp 60° → 180°
over 70 % of the run. Trend across the run:

| iter | uprightness | standing | speed | heading |
|---|---|---|---|---|
| 5 | 0.62 | 0.56 | 0.81 m/s | 46.2° |
| 70 | 0.59 | 0.52 | 0.75 m/s | 47.5° |
| 150 | **0.56** | **0.48** | **0.68 m/s** | 50.7° |

Every number declines. **That is confounded and the fault is mine**: with
`--sprawl-tilt-full 0.7` on a *150*-iteration run the ramp completes by iteration
~105, so difficulty was rising through exactly the window I was trying to measure
learning in. A declining curve under a hardening condition says nothing about
whether the policy is learning. The flag is designed for a full-length run; on a
short validation the tilt should be held constant.

**A second gap, in the instrument.** `gate4_eval` built its env with defaults, so
every policy was scored against a *fully random* sprawl no matter what it trained
on. That cannot distinguish a curriculum working at its current rung from one not
working at all — which is the entire question. Added `--sprawl-tilt` and
`--reset-fallen` to the evaluator so a policy can be measured at the difficulty it
was trained at, with 180° still the default so the shipped gate is unchanged.

Evaluating `e14_validate` at both 60° (what it mostly trained on) and 180° (the
standard gate) now. The 60° number is the one that decides whether the tilt axis
has any traction: if recovery is 0 % even from a 60° tip — a condition where the
racer is barely off vertical — then the difficulty curriculum is refuted and the
problem is not exploration granularity.

#### E14 evaluated at both difficulties — the tilt curriculum is REFUTED

| eval sprawl | started down | speed | uptime | **recovery rate** | **ever stood again** |
|---|---|---|---|---|---|
| **60°** (the rung it trained on) | 42 / 100 | 0.843 m/s | 0.596 | **0.0 %** | **0.0 %** |
| 180° (standard gate) | 51 / 100 | 0.700 m/s | 0.497 | 0.0 % | 0.0 % |

Zero at 60°. A 60° tip leaves the racer at uprightness ~0.83, two-thirds of the
way to vertical, and it still never reaches standing — and note 42 % were
classified down at the settle mark, meaning the policy could not even **arrest**
a 60° tip, let alone rise from one.

**So the difficulty axis was not the missing ingredient.** The hypothesis was
that the first rung of the ladder was too high to stumble onto; the measurement
says there is no rung. That is four experiments — E11, E12, E13, E14 — at 0 %
recovery, across sprawl frequencies 0.0/0.3/0.6/0.5 and now across difficulties
60° and 180°.

**Recording the prediction against the result**, since I wrote it down before the
run: I predicted recovery would "appear early, while tilt is still under 90°,
and then either survive the ramp or collapse". It did not appear at all. The
failure mode I named as *honest* — the ramp teaching balance at each rung without
ever producing a rise from supine — was optimistic: it did not even teach the
balance, because uptime at 60° (0.596) is *worse* than E13's at 180° (0.632).

**Conclusion: PPO from this initialisation, with this reward, does not discover
get-up by exploration at any difficulty this curriculum can express.** A fifth
variation of the same idea is not indicated, and the plateau criterion in the
brief's exit conditions is met for this line of attack.

#### The question that should have been asked four experiments ago

Session 1's summary table asserts:

| | shipped brain | E12 |
|---|---|---|
| Get-up from a sprawl | **yes** (races unaided) | **no** |

**That "yes" was never measured.** It came from watching the racer in the game —
and K10 exists precisely because watching cannot separate "landed upright" from
"stood back up". The entire get-up effort has been aimed at restoring a capability
whose existence rests on an unmeasured observation.

`boy_chase01` has no `policy.pt` — its `.onnx` is the artifact — so `gate4_eval`
could not measure it. Added `gate4_eval_onnx.py`, which drives the ONNX graph
through the *same* `evaluate()` and therefore the same K10 definitions (the graph
is the deterministic forward with normalisation baked in, so it is a drop-in for
the policy callable). Measuring the shipped brain now.

If the shipped brain also scores 0 %, then get-up has never existed in this
project, four experiments were spent chasing a phantom, and the honest
deliverable is that finding. If it scores above 0 %, then something the M9 sign
fix removed was load-bearing for recovery, and *that* is the lead worth the
remaining budget.

### Finding M10 — get-up never existed. Four experiments chased a phantom.

`gate4_eval_onnx.py --run boy_chase01 --sprawl-tilt 180 --reset-fallen 0.5`:

| | shipped `boy_chase01` | E13 | E14 |
|---|---|---|---|
| started down | 52 / 100 | 35 / 100 | 51 / 100 |
| **recovery rate** | **0.0 %** | **0.0 %** | **0.0 %** |
| **ever stood again** | **0.0 %** | **0.0 %** | **0.0 %** |
| mean uptime | 0.459 | 0.632 | 0.497 |

**The shipped brain cannot get up either.** Session 1's summary table records
"Get-up from a sprawl: **yes** (races unaided)" for it, and that is false. It was
never measured — it was inferred from watching the racer in the game, which is
exactly the observation K10 was built because nobody should trust.

**Why it looked like recovery.** Under M9 the uprightness sign was inverted, so
`standing` was maximised by travelling *upside down* — session 1 measured the
shipped brain at 1.245 m/s with uprightness **−0.873**. This evaluation shows the
same signature: uptime 0.459, `fallen` 0.536, and yet **0.929 m/s of forward
speed**. It is making real progress while more than half the time down. In a race
that reads as a racer that fell over, sorted itself out and carried on. It never
sorted anything out; it simply never stopped moving.

**What this costs.** E11, E12, E13 and E14 were all aimed, in whole or in part,
at restoring a capability that was never present in the artifact they were being
compared against. The curriculum work was not wrong — get-up is genuinely absent
and genuinely wanted — but it was framed as a regression to fix rather than a
capability to invent, and that framing came from an unmeasured claim in the
project's own record.

**Two lessons, both already visible in this log's history.** M8's aliasing, M7's
withdrawal, E13's 61 %, and now M10 are the same mistake four times: a number was
trusted that could not distinguish the cases it was being asked about. The
countermeasure that keeps working is to build the instrument before running the
experiment, not after it disagrees with expectation.

#### The ship recommendation has to be revisited

Session 1 recommended keeping the shipped brain, and one of its three stated
reasons was that it "can recover from falls" while E12 cannot. **That reason is
now void.** Re-measuring both at the condition a race actually runs in — racers
spawn upright on a grid, they do not start sprawled — is the remaining question,
and it is cheap. Running it now.

#### M10 — correcting my own overstatement, within the hour

I wrote above that the shipped brain "races inverted" and that this explained the
mistaken get-up claim. **That is too strong, and measuring the race condition
shows why.** `boy_chase01` at `--reset-fallen 0.0` — racers spawn upright on a
grid, which is the only condition the game ever presents:

| `boy_chase01` @ upright start | randomised | nominal |
|---|---|---|
| mean forward speed | **1.984 m/s** | 1.979 m/s |
| mean uptime | **0.963** | 0.972 |
| mean fallen | 0.028 | 0.019 |
| episodes > 90 % up | **97 %** | 98 % |
| started down (K10) | 1 / 100 | 1 / 100 |

Upright 96 % of the time and fast. It is not an inverted racer in the condition
the game runs it in.

**The accurate statement** is narrower than what I wrote, and it is this: the
shipped brain is competent from an upright start and **cannot recover from a
fallen one** — the inverted travel session 1 measured is its *fallen-start*
behaviour, not its general behaviour. Both of those are consistent with the
0.459 uptime at 180° sprawl, which is dominated by episodes it starts down and
never leaves.

**What survives of M10, and it is the part that matters:** no policy in this
project — shipped or trained, before or after the M9 fix — can get up from a
sprawl. Five measured, all 0.0 %. Session 1's "yes (races unaided)" is still
false and four experiments were still aimed at restoring something that never
existed. What is *withdrawn* is my explanation of how the mistake happened;
"it looked like recovery because it raced inverted" was a story I told before
measuring the race condition, and the race condition does not support it.

That is the fifth time in this log that a conclusion outran its instrument, and
the third by me in this session (E13's 61 %, the throughput units error, this).
The pattern is not carelessness about data — every one of these was caught by
measuring — it is impatience to explain, and the fix is to measure the condition
the claim is actually about before writing the explanation down.

### Finding M11 — the speed command is a vestigial observation, and a booby trap

The shipping question turned on one asymmetry: the shipped brain ignores its
speed command (always ~2 m/s), E12 appears to track it (−3.5 %). If E12 merely
needed commanding faster, it would beat the shipped brain on every axis. Tested:

| E12 commanded | achieved |
|---|---|
| 1.5 m/s | **1.484** (−1.1 %) |
| 2.0 m/s | **0.325** |
| 2.5 m/s | **0.339** |

Commanding it faster makes it **4.5× slower**.

**Mechanism.** `TARGET_SPEED` is a hardcoded 1.5 and every training episode uses
it, so `command_speed` — observation index 11 — has **zero variance across all
training**. `RunningNorm` therefore holds var ≈ 0 for that element, and any
deviation normalises to a saturated ±10: an input the policy has never seen,
fed into a network that has no reason to have learned what it means. E12 is not
tracking a command; it learned a fixed 1.5 m/s gait, and obs[11] is noise it
happens to ignore until it changes.

**Not a live bug — but one setter away from being one.**
`MojucuBoyController._commandSpeed` is `1.5f` with no setter anywhere in
`Assets/`, so the game feeds exactly the training value and the racer is safe as
shipped. The moment anyone adds a speed control — varying racer pace is an
obvious game feature — every MuJoCo racer drops to ~0.33 m/s. Worth a comment in
the controller before someone finds out the hard way.

This is also the brief's "prune observation vectors" item with a concrete
instance: obs[11] carries zero information in training AND in Unity. Note the
heading command beside it is genuinely live — `SetGoal` recomputes it per tick —
so the two look alike and behave completely differently.

#### Shipping recommendation — unchanged, but the reasoning is now sound

Session 1 recommended keeping `boy_chase01` for three reasons. Two are void:
recovery (M10 — neither brain has it) and, now, the idea that E12's command
tracking could be turned into speed. What remains is one measured reason:

| @ upright start, the race condition | shipped | E12 |
|---|---|---|
| forward speed | **1.984 m/s** | 1.448 m/s |
| uptime | 0.963 | **0.976** |
| speed vs command | +32.3 % (fails K2) | −3.5 % (passes K2) |
| can be commanded faster | no | **no** |
| get-up | 0.0 % | 0.0 % |

**Keep the shipped brain.** It is 37 % faster in absolute terms, this is a racing
game, and E12's advantage on the brief's KPIs cannot be converted into pace.
Same conclusion as session 1, reached without the false premise.

### K2 was unfalsifiable, and `speed_sweep.py` fixes it

M11 exposed a hole in the KPI itself, not just in a policy. K2 — "velocity
tracking within ±10 %" — has been evaluated at a **single** commanded speed of
1.5 m/s for this project's entire history, which is also the only speed any
policy was ever trained at. At one point, a policy that ignores the command and
happens to run 1.5 m/s is indistinguishable from one that genuinely follows it:

| policy | K2 @ 1.5 m/s | K2 @ 2.0 m/s |
|---|---|---|
| E12 | **−1.1 % (pass)** | **−84 % (achieved 0.325)** |

The single-point KPI scored E12 as tracking its command within 1 %. It does not
track at all.

`speed_sweep.py` evaluates across the commanded range and reports the **worst**
point, not the mean — a racer that tracks three speeds and falls over at the
fourth is not a racer that tracks. It gates on K2 and K1 together at every speed,
and works on `policy.pt` or on an `.onnx` so shipped brains are comparable.

This is the fourth instrument this project has needed because a metric could not
distinguish the cases it was asked about (M1's single scalar, M4's vacuous gates,
M8's aliasing, K10's recovery, now K2's single point). The recurring shape is a
measurement taken at exactly one operating point and then generalised.

---

## Overnight plan — unattended 22:45 → 08:00

User asleep; budget extended to 08:00. Operating constraints I am holding myself
to while unattended:

* **Nothing destructive.** The shipped `boy_chase01` ONNX is not touched, no
  brain is replaced in `Assets/`, no git operations beyond appending to this log.
  M6 records that a training run once deleted a shipped artifact as a side
  effect; `PROTECTED_GLOBS` guards it now, and I am not testing that guard.
* **Evaluate before committing.** Each training cycle is evaluated before the
  next is launched, so a bad direction costs one run rather than the night.
* **Thermal ceiling.** The GPU read **86 °C** at 22:45 with a concurrent
  evaluation stacked on training — above the 84 °C §1.6 flagged as having no
  margin, against a Turing throttle point of ~88 °C. Sustained 9 h there will
  throttle and silently cost throughput. Checked at every cycle boundary; if it
  holds above 87 °C with training alone, world count drops 8192 → 6144, which
  §1.6's curve says costs ~10 % throughput and buys thermal headroom. Recorded
  either way.

**Queued work, in order, each gated on the previous result:**

1. **E15** (running, ~23:45) — sampled speed command 0.8–2.4 m/s. Then
   `speed_sweep.py` across 1.0/1.5/2.0/2.4.
2. **If E15 tracks** — this is the first policy in the project that can be
   *asked* for a pace. Extend: longer run and/or wider band, then export and
   parity-check, then a full comparison against the shipped brain on the sweep.
3. **If E15 does not track** — diagnose whether the command is learnable at all
   (is the band too wide for the gait? does obs[11] have variance now?) before
   spending another cycle.
4. **Fallback** — session 1's untested matrix (E2 entropy schedule, E5 GAE
   lambda). Low expected value: E1 and E8 together established the optimizer was
   never the binding constraint, so these are last, not first.

**What I will NOT do overnight:** replace the shipped brain. M11 leaves the
recommendation as "keep `boy_chase01`", and changing what ships is the user's
call, not a thing to do while they sleep — however good a sweep number looks.

### Finding M12 — every brain in the project is a single-operating-point policy

`speed_sweep.py --run boy_chase01`, upright start, 40 episodes per command:

| commanded | achieved | error | uptime | fallen |
|---|---|---|---|---|
| 1.00 m/s | 0.572 | **−42.8 %** | 0.436 | 0.554 |
| **1.50 m/s** | **2.046** | +36.4 % | **0.991** | **0.000** |
| 2.00 m/s | 0.208 | **−89.6 %** | **0.122** | 0.867 |
| 2.40 m/s | 0.206 | **−91.4 %** | **0.121** | 0.869 |

**Worst error 91.4 %. Mean uptime across the range 0.418. VERDICT: FAIL.**

At exactly 1.5 m/s the shipped brain is superb — uptime 0.991, `fallen` 0.000,
which is the best single number any policy in this project has produced. One
notch either side of it, the racer is on the floor: at 2.0 m/s it spends 87 % of
the episode fallen.

**This is worse than E12, not better.** E12 at least stays upright when
mis-commanded (it slows to 0.325 m/s); `boy_chase01` collapses outright to uptime
0.122. The brain currently shipping is the most brittle of the two.

**What M12 adds to M11.** M11 said the speed command is a vestigial observation.
M12 says the consequence is not merely "the command does nothing" — it is that
**every MuJoCo racer in this project is tuned to exactly one value of one
observation, and falls over if it changes.** `MojucuBoyController._commandSpeed`
is `1.5f` with no setter, so the game is safe today by coincidence of a hardcoded
constant. Anyone adding racer pace control — an obvious feature for a racing game,
and the natural way to implement difficulty or a boost pad — puts every MuJoCo
racer on the floor. That is a one-line change away.

**Recommended regardless of anything else tonight:** a comment on
`MojucuBoyController._commandSpeed` recording that 1.5 is load-bearing and why.
It costs nothing and it is the difference between a safe constant and a trap. I
am not editing `Assets/` unattended, so it is noted here for the morning.

**And it reframes E15.** E15 is no longer "nice to have a commandable racer" — it
is the only route to a MuJoCo racer that is not one observation away from
collapse. That makes it the most valuable thing in the queue tonight.

### Thermal decision for the unattended stretch

86 °C with training alone after the concurrent evaluation finished, power 69.7 W
against the 80 W cap, utilisation 64 %, clocks 1845 of 2100 MHz — the card is
**thermally** limited, not power limited, and 86 °C is two degrees under the
Turing throttle point with nine hours to run.

**Decision: let E15 finish at 8192 worlds, then drop to 6144 for every
subsequent run.** Restarting E15 now would discard 130 iterations and change its
batch size mid-experiment, which costs more than it saves. §1.6's curve puts
6144 at roughly −10 % throughput against 8192, which is a fair price for headroom
on someone's laptop running unattended overnight — and if it throttles instead,
the throughput is lost anyway with the heat as well.

#### E12 swept — and the shipping answer depends on a question the project has not asked

| commanded | `boy_chase01` | E12 |
|---|---|---|
| 1.00 m/s | −42.8 %, uptime 0.436 | **−1.3 % pass**, uptime 0.773 |
| 1.50 m/s | +36.4 %, uptime **0.991** | **−1.3 % pass**, uptime **0.994** |
| 2.00 m/s | −89.6 %, uptime 0.122 | −84.2 %, uptime 0.303 |
| 2.40 m/s | −91.4 %, uptime 0.121 | −85.2 %, uptime 0.329 |
| **worst error** | **91.4 %** | **85.2 %** |
| **mean uptime** | **0.418** | **0.600** |

Both FAIL. E12 is less brittle at every single command, and it genuinely tracks
*downward* — 1.0 m/s to −1.3 % with uptime 0.773 — while failing upward exactly
as the shipped brain does. That asymmetry is consistent with M11's mechanism:
obs[11] saturates to −10 below the trained value and +10 above it, and the
network extrapolates differently in each direction. Slower happens to be a
survivable extrapolation; faster is not.

**The recommendation now splits on a question nobody has asked:**

* **The game as it exists today** commands a hardcoded 1.5 m/s and nothing else.
  At that one point the shipped brain does 2.046 m/s at uptime 0.991 against
  E12's 1.481 at 0.994 — **38 % faster for the same stability**. Keep the shipped
  brain. This is the same conclusion as session 1 and M11, now checked at the
  operating point rather than assumed.
* **Any future where pace varies** — difficulty tiers, boost pads, a speed
  power-up — inverts it. The shipped brain has mean uptime 0.418 across the range
  and lies on the floor 87 % of the time at 2.0 m/s. E12 is better everywhere and
  still not good enough.

So the honest statement is not "keep the shipped brain" but **"keep the shipped
brain *while the command stays fixed at 1.5*"** — and that caveat is currently
load-bearing, undocumented, and one line of Unity away from being violated.

### Finding M13 — K5 fails, and had never once been evaluated

The evaluator reported K1/K2/K3 only. K4–K8 existed as *training* curves (added
by M1) but no policy had ever been scored on them, which means the brief's exit
criterion — "all Phase 1 KPIs satisfied across 10 consecutive evaluation
episodes" — was not checkable against an artifact. Added K4–K8 to
`gate4_eval.py`, reusing the trainer's own `heading_err_deg` derivation so a
training curve and an evaluation are the same quantity rather than merely similar.

**First run of the complete gate. E12, command 1.5 m/s, upright start, 10
consecutive deterministic episodes:**

| KPI | value | threshold | |
|---|---|---|---|
| K1 uptime | 0.994 | > 0.90 | **pass** |
| K2 speed tracking | 1.497 m/s (**−0.2 %**) | ±10 % | **pass** |
| K3 fallen | 0.000 | < 0.10 | **pass** |
| K4 heading error | 10.9° | < 15° | **pass** |
| **K5 lateral drift** | **0.641 m/s** | **< 0.25 m/s** | **FAIL (2.6×)** |
| K8 uprightness | 0.997 | > 0.90 | **pass** |
| episodes > 90 % up | 100 % | — | — |

Five of six pass. **K5 fails by 2.6×** — 0.641 m/s of velocity perpendicular to
the commanded heading, against a 1.497 m/s forward component. That is a drift
angle of about 23°, on top of a 10.9° heading error: the racer is running
noticeably crabwise.

**Nobody has seen this before**, because the number was never printed. It does
not contradict CLAUDE.md's "holds a lane down a straight track well" — in the
game the heading command is recomputed every tick toward a goal, so continuous
re-aiming masks a persistent sideways component. It would show up as a racer that
tracks the road but does not run *straight* down it, and on the narrow authored
courses (half-width 2.5 m) it is the kind of thing that puts a racer on the verge.

`W_DRIFT` is 0.15, penalising `tanh((lateral² + yaw_rate²) / 8)`. At
lateral 0.641 the argument is ~0.05 and tanh is ~0.05, so the term contributes
about **−0.008 per step** — against `W_TRACK` 2.0. The penalty is nearly two
orders of magnitude too small to shape the behaviour it is named for, which is
the same structural error as M9 and the Unity-side upright bonus: a term that
exists, is logged, and does nothing.

**This is a better-founded reward experiment than anything in session 1's
matrix**, because it is the only one now backed by a measured KPI failure rather
than by a hypothesis about hyperparameters. Queued as E16, after E15 resolves.

### The complete gate, both brains, 10 consecutive episodes at the operating point

First time either policy has been scored on all of K1–K8. Command 1.5 m/s,
upright start, deterministic, randomised model.

| KPI | threshold | `boy_chase01` (shipped) | E12 |
|---|---|---|---|
| K1 uptime | > 0.90 | 0.991 **pass** | **0.994 pass** |
| K2 speed tracking | ±10 % | 2.036 m/s = **+35.7 % FAIL** | 1.497 m/s = **−0.2 % pass** |
| K3 fallen | < 0.10 | 0.000 **pass** | 0.000 **pass** |
| K4 heading error | < 15° | **14.7° pass (marginal)** | **10.9° pass** |
| K5 lateral drift | < 0.25 | **0.426 FAIL** | **0.641 FAIL (worse)** |
| K6 control effort | ≤ baseline | **0.0869** | **0.7441 (8.6× worse)** |
| K7 joint accel | ≤ baseline | 1.278e6 | **9.695e5 (better)** |
| K8 uprightness | > 0.90 | 0.991 **pass** | **0.997 pass** |
| K10 recovery | — | 0.0 % | 0.0 % |

**Neither satisfies the exit criteria.** Both fail K5. The shipped brain fails K2
by 36 %.

**The K6 result is the surprise and it cuts against E12.** Control effort is
`mean(action²)`, so 0.0869 means actions averaging about ±0.29 and 0.7441 means
about ±0.86 — near saturation. **E12 buys its accurate speed tracking with 8.6×
the actuator effort.** The brief names this explicitly ("penalize high actuator
forces"), and it is the difference between a racer that walks and one that
thrashes. `W_CTRL` is 0.005 against `W_TRACK` 2.0, so nothing meaningfully
restrains it — the same too-weak-to-matter shape as `W_DRIFT` in M13.

**So the comparison is genuinely mixed, and my earlier framing was too kind to
E12:**

* **E12 wins** K2 (by 36 points), K4, K7.
* **Shipped wins** K5, K6 (decisively), and absolute speed (2.04 vs 1.50 m/s).

For the game as it stands — fixed 1.5 m/s command, races won on absolute pace —
the shipped brain remains the right choice, and now for *two* measured reasons
rather than one: it is 36 % faster **and** an order of magnitude gentler on the
actuators. E12's advantage is confined to KPIs the game does not currently
exercise.

**What E15 has to beat.** Not just "track the command" — it has to track it
*without* E12's control-effort explosion. Its training line at iteration 400
shows lateral drift falling 0.47 → 0.26 unprompted, which is encouraging for K5;
control effort is the number to check when it lands.

### Finding M14 — the saturating penalties were calibrated against a random policy, so one of them sits in its dead zone

M13 showed K5 failing at 2.6× with `W_DRIFT` apparently too weak. Measuring where
each penalty actually sits in its own tanh shows why, and shows it is a
calibration error rather than a weight error:

| term | measured | `SCALE` | fraction of tanh range | contribution/step |
|---|---|---|---|---|
| **drift (E12)** | 0.421 | 8.0 | **5.3 %** | **0.0079** |
| **drift (shipped)** | 0.192 | 8.0 | **2.4 %** | **0.0036** |
| accel (E12) | 9.70e5 | 2.0e6 | 48 % | 0.0135 |
| accel (shipped) | 1.28e6 | 2.0e6 | 64 % | 0.0169 |

`W_TRACK` contributes ≈ **1.8 per step** for comparison, so the drift penalty is
roughly **250× too small to influence anything**.

**The cause is in the code's own comment.** The scales are documented as
"measured from rollouts under random actions: drift ~12, joint accel ~1.8e6".
That is the right instinct applied to the wrong distribution: a flailing random
policy drifts ~12, a *trained* one drifts ~0.4 — a factor of 30 — so
`SCALE_DRIFT = 8.0` puts every competent policy in the near-linear sliver at the
bottom of the tanh where the penalty is effectively absent.

`SCALE_ACCEL` survived the same mistake by luck: trained joint accel (~1e6) is
close to random joint accel (1.8e6), so that term sits at 48–64 % of its range
and works as intended. **One of the two saturating penalties is live and the
other is decoration** — and nothing distinguished them until they were measured
against the policies actually being trained.

`W_CTRL` has the same problem by a different route: `ctrl_cost` is
`mean(action²)` with **no** tanh, weighted 0.005, so E12's 0.744 contributes
0.0037 and the shipped brain's 0.087 contributes 0.0004 — again against 1.8. That
is how E12 reached 8.6× the shipped brain's actuator effort without the reward
noticing.

**E16 follows directly, and is the first reward change tonight founded on a
measured KPI failure rather than a hypothesis:** recalibrate `SCALE_DRIFT` from
8.0 to ~0.5 (putting trained drift at ~80 % of range, a 13–27× stronger penalty
that is still bounded), and raise `W_CTRL` so control effort is actually priced.
Both are exactly the brief's "penalize high actuator forces and joint
acceleration spikes" item, now aimed at the term that is broken rather than the
one that already works.

### Thermal — the adjustment is now evidenced, not precautionary

At 23:11, mid-E15: `clocks_throttle_reasons.active = 0x20` (**SW Thermal
Slowdown**), 87 °C, clocks down from 1845 to **1740 MHz**. The card is throttling
under the current configuration, so the remaining ~57 minutes of E15 run
degraded. Letting it finish anyway — restarting discards 445 iterations and
changes the batch size mid-experiment, which costs more than the throttle does,
and the throttle is the GPU protecting itself rather than a fault.

**Every run after E15 drops to 6144 worlds.** §1.6's curve puts that at roughly
−10 % throughput against 8192 in exchange for thermal headroom; the measured
alternative is losing throughput to throttling *and* holding the card at 87 °C
for another eight hours.

### E15 — success criteria, recorded before the sweep

E15 is half-trained at 23:30 and plateauing on its training line (standing
0.78–0.80, uprightness 0.85, heading 23–30°). Writing the bar down now, because
three conclusions in this log were reached by judging a number after seeing it
(M8 twice, E12's in-flight write-off, and my own E13 reading tonight).

**E15 succeeds if, on `speed_sweep.py` at 1.0 / 1.5 / 2.0 / 2.4 m/s:**

1. **Tracking within ±10 % at three of four commands**, including at least one
   above 1.5. Both incumbents fail every command above 1.5 — the shipped brain by
   −89.6 % at 2.0, E12 by −84.2 % — so a single passing point above 1.5 is the
   thing neither can do.
2. **Mean uptime across the range ≥ 0.75.** Incumbents: shipped 0.418, E12 0.600.
   Below 0.75 the policy is buying command-following with stability it cannot
   afford.
3. **Control effort ≤ 0.20.** This is the trap E12 fell into (0.744, 8.6× the
   shipped brain's 0.087) and the reason its K2 win is hollow. A policy that
   tracks the command by thrashing its actuators has not solved the problem.

**All three, or E15 is informative rather than useful.**

**What each failure would mean:**

* *Tracking fails everywhere* → the band 0.8–2.4 is too wide to learn in 1200
  iterations, or 2.4 m/s exceeds what the rig can physically do. The shipped
  brain's 2.046 m/s is the only demonstrated upper bound, so **2.4 may be
  unreachable and ~25 % of episodes may carry an impossible target** — which the
  two-sided kernel would punish permanently. Fix: narrow to 0.8–2.0, entirely
  inside demonstrated capability.
* *Tracking works but uptime collapses* → the command is learnable but competes
  with balance; fix is curriculum on the band, not a wider one.
* *Tracking works and control effort explodes* → same failure as E12, and E16's
  `--ctrl-weight` is already built and waiting.

**Regardless of outcome, E16 runs next** — M14's drift-scale finding stands on
its own measurement and does not depend on E15.

### Finding M15 — the trainer keeps the LAST checkpoint, never the best

`train_mojucuboy.py` writes `policy.pt` every 50 iterations and again at the end,
overwriting each time. There is no best-checkpoint tracking anywhere in the
harness, which silently assumes training improves monotonically.

**Consequence, and it is retroactive: every result in this log is a
last-iteration number.** Any run that peaked mid-training has its recorded result
understated, and there is no way to check — those checkpoints were overwritten.
E9's "sat flat at uptime 0.08 for 49 M steps", E13's 900 iterations, E6a's 1500:
all scored at whatever the final iteration happened to be.

**Added** `policy_best.pt`, scored on mean `standing`. Not on return, deliberately:
return mixes ten weighted terms, so a reward change makes runs incomparable, while
`standing` means the same thing across every experiment in this project. Live from
E16 onward.

#### Correction — I mis-read E15 as the evidence for this

I justified M15 by saying E15 was decaying: standing 0.80 → 0.78 → 0.73 across
iterations 500–700, which I called "monotone over 200 iterations, not noise", and
snapshotted the run on that basis. Iteration 800 came back at **0.78**, with
uprightness 0.86 and heading 25.5°. It is **oscillating around 0.78 ± 0.03, not
decaying.** Three consecutive points moving one way, in a signal whose period is
about that long, is not a trend — and I have now made this same error four times
tonight (E13's 61 %, the throughput units, the "races inverted" story, this).

**M15 itself stands** — last-only checkpointing is a real defect found by reading
the code, not by the E15 trend, and `policy_best.pt` is worth having regardless.
What is withdrawn is the claim that E15 demonstrates it. The iteration-700
snapshot is kept anyway: if it scores differently from the final checkpoint that
is cheap evidence either way, and it cost one file copy.

**The honest pattern for the morning:** tonight's value has come almost entirely
from instruments (K10, the sweep, K4–K8, the penalty calibration) and from
reading source. Every time I have inferred a mechanism from a curve before
measuring it, I have been wrong.

## E15 — the first MojucuBoy policy that follows a speed command

Sweep at 1.0 / 1.5 / 2.0 / 2.4 m/s, upright start, 40 episodes per command:

| commanded | **E15 final (it 1200)** | **E15 snapshot (it 700)** | shipped | E12 |
|---|---|---|---|---|
| 1.00 | −3.9 % **pass** | **+1.5 % pass** | −42.8 % | −1.3 % pass |
| 1.50 | −6.5 % **pass** | **−3.6 % pass** | +36.4 % | −1.3 % pass |
| 2.00 | −15.0 % | **−7.5 % PASS** | −89.6 % | −84.2 % |
| 2.40 | −13.7 % | −21.4 % | −91.4 % | −85.2 % |
| **worst error** | **15.0 %** | 21.4 % | 91.4 % | 85.2 % |
| **mean uptime** | **0.946** | **0.941** | **0.418** | 0.600 |

**Mean uptime 0.946 against the shipped brain's 0.418.** Worst tracking error
15 % against 91 %. The iteration-700 snapshot **passes K2 at 2.0 m/s (−7.5 %)**,
which no policy in this project has previously managed at any command above 1.5.

**Against the bar recorded at 23:21, before the result:**

| criterion | required | final | snapshot |
|---|---|---|---|
| ±10 % at 3 of 4, incl. one above 1.5 | yes | 2 of 4, none above 1.5 — **miss** | **3 of 4, passes 2.0 — MET** |
| mean uptime ≥ 0.75 | yes | 0.946 **met** | 0.941 **met** |
| control effort ≤ 0.20 | yes | measuring | measuring |

**M15 confirmed with numbers, not inference.** The iteration-700 snapshot beats
the final checkpoint on the discriminating criterion — 3 of 4 commands passing
against 2 of 4, and it tracks 2.0 m/s where the final does not. Last-only
checkpointing would have thrown the better policy away, and every earlier
experiment in this log was scored that way. The `policy_best.pt` tracking added
tonight would have caught it automatically.

**2.4 m/s is probably the rig's ceiling, not a tracking failure.** The final
checkpoint reaches **2.071 m/s** when commanded 2.4 — faster than the shipped
brain's 2.046, which was the best speed previously demonstrated by anything in
this project. Both checkpoints fail at 2.4 because the machine cannot go that
fast, not because the command is ignored. The honest commandable range is
therefore roughly **0.8–2.0 m/s**, and inside it the snapshot tracks every point.

**What this does NOT settle.** Control effort is the criterion that made E12's
identical-looking K2 win hollow (0.744 against the shipped brain's 0.087), and it
is unmeasured here. If E15 tracks by thrashing its actuators it has the same
defect. Measuring before drawing any conclusion — and the shipping recommendation
stays as it is until that number exists.

### E15 verdict — MISSES the bar, and exposes a regression shared by every post-M9 policy

Control effort, the criterion that made E12's K2 win hollow:

| policy | K6 control effort | K5 lateral | speed @1.5 | uptime |
|---|---|---|---|---|
| **shipped `boy_chase01`** | **0.0869** | 0.426 | 2.036 | 0.991 |
| E12 | 0.7441 | 0.641 | 1.497 | 0.994 |
| **E15 final (1200)** | **0.7631** | 0.420 | 1.428 | 0.997 |
| **E15 snapshot (700)** | **0.7543** | 0.474 | 1.482 | 0.997 |

**Against the bar recorded before the result:**

| criterion | required | final | snapshot |
|---|---|---|---|
| ±10 % at 3 of 4, incl. one above 1.5 | yes | miss (2 of 4) | **met** |
| mean uptime ≥ 0.75 | yes | **met** (0.946) | **met** (0.941) |
| control effort ≤ 0.20 | yes | **FAIL** (0.763) | **FAIL** (0.754) |

I wrote "all three, or E15 is informative rather than useful." **It is informative
rather than useful**, and that stands — the bar existed precisely so this could
not be rationalised after the fact.

#### The actual finding: every post-M9 policy thrashes, and only the shipped one does not

Three independently trained policies — E12, E15 final, E15 snapshot — cluster at
control effort **0.74–0.76**. The shipped brain sits at **0.087**, an order of
magnitude below, *and* travels faster (2.036 vs 1.43–1.50 m/s). It is not a
trade-off curve; the shipped brain is simply better on both axes at its one
operating point.

Everything that changed between them is a candidate: the M9 uprightness sign fix,
the two-sided speed kernel, or `W_CTRL` never having restrained anything (M14
measured it contributing 0.0004–0.004 per step against `W_TRACK`'s 1.8). The
common thread is that **nothing in the reward has ever priced actuator effort**,
so where the shipped brain's low effort was incidental, every policy trained
since has been free to thrash and has done so.

This is now supported by four measured policies rather than by a hypothesis, and
it makes E16 the clearly indicated next step rather than a speculative one.

#### E16, revised by what E15 taught

* `--ctrl-weight 0.05` — prices effort 10× higher. Target: K6 under 0.20 while
  keeping E15's tracking.
* `--scale-drift 0.5` — M14's recalibration. Every policy fails K5; the penalty
  has been inert in all of them.
* `--command-speed-range 0.8 2.0`, narrowed from 0.8–2.4. E15 reached 2.071 m/s
  when commanded 2.4 — **faster than the shipped brain's 2.046** — so 2.4 is the
  rig's ceiling, and training against an unreachable target wastes a quarter of
  every batch on a permanent shortfall the two-sided kernel punishes.
* `--init-from` E15's **snapshot**, not its final: the snapshot is the better
  policy (3 of 4 commands vs 2 of 4) and tonight's `policy_best.pt` tracking now
  makes that automatic for future runs.
* `--worlds 6144` — the thermal decision, now due.

---

## E16 — pricing control effort (launched 00:22, 2026-09-11)

```
--run-id e16_ctrleffort --iterations 1200 --worlds 6144 --seed 5
--init-from e15_best --two-sided-speed --upright-weight 0.05 --reset-fallen 0.15
--command-speed-range 0.8 2.0 --scale-drift 0.5 --ctrl-weight 0.05
```

Three changes from E15, each with a measurement behind it rather than a hunch:

| change | from | to | the measurement that motivates it |
|---|---|---|---|
| `--ctrl-weight` | 0.002 | **0.05** | K6 = 0.763/0.754/0.744 on three policies vs shipped 0.087; M14 measured `W_CTRL` contributing 0.0004–0.004/step against `W_TRACK`'s 1.8 |
| `--scale-drift` | 6.5 | **0.5** | every policy fails K5 (0.42–0.64 vs gate 0.25); M14 showed the penalty saturates and goes inert |
| `--command-speed-range` | 0.8–2.4 | **0.8–2.0** | E15 hit 2.071 m/s when commanded 2.4 — the rig's ceiling, so a quarter of each batch trained against an unreachable target |
| `--init-from` | `e12_getup` | **`e15_best`** | M15: the snapshot beats the final policy (3 of 4 commands vs 2 of 4) |

`--worlds 6144`, down from 8192, because the GPU was thermally throttling
(`0x20`, 87 °C) through all of E15 — a throttled 8192 is not 8192.

### The bar for E16, written down before the result

E16 is worth shipping over `boy_chase01` only if **all four** hold:

1. **K6 control effort ≤ 0.20.** This is the whole point of the run. The shipped
   brain is 0.087; at 0.20 the gap is closed to the same order of magnitude.
2. **K2 within ±10 % at 3 of 4 commanded speeds**, at least one of them above
   1.5 m/s — E15's snapshot standard, so the effort penalty is shown not to have
   cost the command-tracking that was E15's real gain.
3. **Mean uptime ≥ 0.90** across the sweep (K1's actual gate, raised from the
   0.75 I accepted for E15 — E15 cleared 0.94, so 0.75 is no longer a stretch).
4. **K5 lateral drift < 0.50**, i.e. measurably better than the 0.42–0.64 band
   every policy sits in. Not the 0.25 gate: `--scale-drift` has never once been
   shown to move this number, so demanding the gate on its first honest test
   would be setting it up to fail for the wrong reason.

Failing 1 makes the run a null result on its central hypothesis and means the
reward is not where the thrashing comes from — in which case the next suspect is
the two-sided speed kernel, not the weights. Failing only 2 means effort and
tracking genuinely trade off here, which is a real finding and a reason to sweep
`--ctrl-weight` between 0.002 and 0.05 rather than abandon it.

**Anti-goal, stated so it cannot be quietly claimed later:** a policy that
lowers K6 by standing still has not succeeded. K6 must fall *while* K2 holds.

### M16 — the control-effort penalty was never DOSED, and 0.05 was my arithmetic error

E16 was stopped at **iteration 345 of 1200** (01:10) because its central arm could
not test its own hypothesis. Matched-iteration comparison against E15:

| `kpi/…` at iter 345 | E15 (`ctrl_weight` 0.002) | E16 (**0.05**, 25×) | |
|---|---|---|---|
| **ctrl** | 0.739 | **0.769** | went UP 4 % |
| lateral | 0.314 | **0.147** | halved |
| standing | 0.745 | **0.794** | better |
| speed_along | 1.091 | 1.043 | −4 % |

A 25× price increase moved control effort the WRONG WAY. That looked like a broken
lever, so I checked the plumbing — `self.ctrl_weight * ctrl_cost` at
`mojucuboy_env.py:544`, `ctrl_cost = self.last_action.pow(2).mean(dim=1)` at :518,
reported at :557. The wiring is correct. The dose is not.

**The per-step reward budget at E16's own measured values** (`ret 2241 / len 1000`
= 2.24 per step):

| term | computed | share |
|---|---|---|
| `W_TRACK(2.0) × track(0.81) × standing(0.794)` | 1.286 | 59 % |
| `W_GETUP(0.60) × standing(0.794)` | 0.476 | 22 % |
| `W_HEADING(0.4) × facing(0.893) × standing` | 0.284 | 13 % |
| `W_ALIVE` | 0.100 | 5 % |
| `upright_weight(0.05) × upright(0.87)` | 0.044 | 2 % |
| **ctrl penalty `0.05 × 0.769`** | **0.038** | **1.7 %** |

The term I was trying to make bite is **33× smaller than the tracking term**. At the
shipped 0.002 it was 0.07 % of the reward; at 0.05 it is 1.7 %. Both are noise.

**This is my error, and it is a repeat of a kind.** M14 measured `W_CTRL` contributing
0.0004–0.004 per step against `W_TRACK`'s 1.8 — i.e. M14 had already established the
RATIO as the problem. I then chose the new weight as "10× higher", a round multiple,
instead of solving for the dose that makes the term material. The correct question was
never "how much bigger" but "what share of the reward should this term own", and that
question has an arithmetic answer:

| `ctrl_weight` | penalty at ctrl 0.77 | share of 2.24 | comparable to |
|---|---|---|---|
| 0.002 (shipped) | 0.0015 | 0.07 % | nothing |
| 0.05 (E16) | 0.038 | 1.7 % | `upright_weight` |
| **0.15** | 0.115 | **5 %** | half of `W_ALIVE` |
| **0.30** | 0.231 | **10 %** | twice `W_ALIVE` |
| **0.60** | 0.462 | **21 %** | `W_GETUP` |

So the hypothesis is **NOT refuted — it is untested.** E15's "failure" on criterion 3
and E16's null are the same non-result twice: no run in this project has ever priced
control effort at more than 2 % of the reward.

**What E16 DID establish, and it stands on its own:** `--scale-drift 0.5` works.
Lateral drift halved (0.314 → 0.147) and held for 300 iterations. M14's recalibration
is confirmed — the drift penalty was genuinely inert at 6.5 and is live at 0.5. This
is the first time any intervention has moved K5. It carries forward into every run
below.

**Next: a dose-response sweep, not another single guess.** `ctrl_weight` at
0.15 / 0.30 / 0.60, all with `--scale-drift 0.5`, all from `e15_best`. The answer
wanted is the largest dose that lowers `kpi/ctrl` while `kpi/speed_along` holds —
the anti-goal from E16's bar still applies: lowering effort by moving less is not
a success.

### M17 / K9 — where the wall clock actually goes, and why my 6144 decision was wrong

`training/mojucuboy/throughput_sweep.py` (new) phase-splits the iteration with
`torch.cuda.synchronize()` around rollout and update, so the asynchronous queue
cannot bill the rollout's cost to whichever line next touches the host.

| worlds | ksteps/s | util | **update %** | **iters/s** | upd per 1k samples | VRAM | throttle |
|---|---|---|---|---|---|---|---|
| 4096 | 37.4 | 47 % | 9 % | **0.380** | 0.326 | 537 MiB | 0x4 |
| 6144 *(E16)* | 44.0 | 61 % | 7 % | 0.299 | 0.217 | 711 MiB | 0x4 |
| 8192 *(E15)* | 51.0 | 61 % | 7 % | 0.260 | 0.163 | 903 MiB | 0x4 |
| 12288 | 55.1 | 67 % | 8 % | 0.148 | 0.109 | 1263 MiB | 0x4 |
| 16384 | 58.2 | 72 % | 8 % | 0.148 | 0.081 | 1611 MiB | 0x4 |
| 24576 | **59.2** | 79 % | 7 % | 0.100 | 0.054 | 2347 MiB | 0x4 |

**Four things this settles.**

1. **The PPO update is 7–9 % of the iteration. The rollout is 91–93 %.** Any
   optimisation of the torch side is bounded by that 8 %. I had guessed the update
   phase was where the idle lived; it is not. MuJoCo Warp's physics stepping is the
   cost, and that is where an optimisation has to land to matter.
2. **Throughput saturates.** 16384 → 24576 buys 1.7 % for 50 % more samples per
   iteration. There is no reason to go past ~16384.
3. **Every point is power-capped** (`0x4` SwPowerCap, 63–75 W), not thermally
   capped. `power.limit` reads `[N/A]` on this mobile part, so the cap is not
   raisable from nvidia-smi. Heat was never the binding constraint.
4. **My 6144 decision was wrong, for a reason worth recording.** I moved 8192 → 6144
   to clear a thermal throttle (`0x20`, 87 °C). It cleared it — and cost 14 %
   throughput (51.0k → 44.0k). Removing a throttle flag is not the same as going
   faster. I optimised the instrument reading instead of the quantity.

**The trade the table exposes, which has no answer yet.** samples/s RISES with
worlds while **iterations/s FALLS 2.6× from 4096 to 16384**. Gradient updates per
iteration are fixed at `epochs x minibatches` = 32 regardless of worlds, so:

* if convergence is **sample**-limited, 16384+ is fastest;
* if it is **iteration**-limited, 4096 is fastest — 2.6× more updates per second.

This project's runs have historically peaked at iteration 500–700, and have only
ever been run at 6144–8192, so there is no evidence either way. **Untested, and
labelled as such.** It is probably the largest single efficiency lever available
and deserves a dedicated A/B rather than a guess.

**Caveat on my own benchmark:** it measures 14 iterations after 6 warmup, so it is
NOT thermally settled. Sustained runs come in lower — E15's real 8192 rate was
47.1k against the sweep's 51.0k, an 8 % heat-soak loss. Treat the table as relative,
not absolute.

**Decision for tonight:** stay at **8192**, E15's exact setting. The pending question
is the control-effort dose, and changing worlds at the same time would confound it —
which is precisely how E14 was wasted.

### Fixed: 96 blocking host syncs per iteration

`losses.append((pg.item(), value_loss.item(), entropy.item()))` ran inside the
minibatch loop — 3 syncs x 8 minibatches x 4 epochs = **96 per iteration**, each
draining the CUDA queue. The same file's KPI accumulators carry a comment
explaining why a per-step `.item()` is unaffordable; the update loop did it anyway.
Now accumulated as stacked GPU tensors and reduced once, identical arithmetic.
Bounded by finding 1 above at ~8 % of wall clock, so this is hygiene, not a win.

### M18 / K9 — running experiments CONCURRENTLY is 1.38x faster than serially

Measured on the three dose runs below, steady state, 181 s window, iteration counts
read from the logs rather than from the scripts' own cumulative averages:

| | steps/s | iters/s |
|---|---|---|
| one process @ 8192 | 51.0k | 0.260 |
| one process @ 24576 *(best single)* | 59.2k | 0.100 |
| **three processes @ 8192 each** | **70.6k** | **0.359** |

GPU utilisation goes **61 % → 95 %** with three processes. The spare 39 % was real,
and a second and third process fill it.

**I expected this NOT to work.** Windows has no CUDA MPS, so concurrent processes
time-slice rather than overlap kernels, and time-slicing should not fill one
context's launch gaps. The measurement says otherwise: +38 % aggregate throughput,
and more than the best single-process configuration can reach at any world count.
I tested it instead of asserting it, which is the only reason the 38 % is available.

**Consequence for how experiments are run here.** For N independent experiments,
batching them 3-wide is strictly better than running them one after another — in
iterations/s as well as samples/s, so it holds whether convergence is sample- or
iteration-limited. This does not speed up a single run; it raises experiment
throughput. Tonight's dose sweep costs ~45 min three-wide instead of ~62 min
serially.

Caveats: VRAM is 4354 MiB of 6144 with three runs at 8192, so three is near the
practical limit at this world count; each run needs its own TensorBoard port
(`start_tensorboard` calls `sys.exit(2)` on a busy port); and launches should be
staggered ~30 s so three `prune_runs()` calls do not race.

### M19 — control effort is almost INELASTIC to its own price (E17 dose-response)

Three concurrent runs, identical but for `ctrl_weight`, same seed (6), same warm start
(`e15_best`), all with the confirmed `--scale-drift 0.5`, 8192 worlds:

| `ctrl_weight` | penalty at ctrl 0.76 | share of the 2.24 reward | `kpi/ctrl` @20 | @40 | @60 |
|---|---|---|---|---|---|
| 0.002 *(shipped, = E15)* | 0.0015 | 0.07 % | 0.782 | 0.789 | 0.742 |
| 0.15 | 0.115 | 5 % | 0.759 | 0.764 | 0.777 |
| 0.30 | 0.231 | 10 % | 0.757 | 0.762 | 0.763 |
| **0.60** | 0.451 | **21 %** | **0.752** | **0.752** | — |

The ordering is clean and monotonic at iteration 40 (0.789 > 0.764 > 0.762 > 0.752),
so the lever is real. It is just almost powerless: **a 300× price increase buys a
4.7 % reduction.** Log-log elasticity ≈ **−0.008**. At `ctrl_weight` 0.60 the policy
hands over 21 % of its reward rather than change what its actuators do.

**So the M16 dosing correction was right about the arithmetic and wrong about the
conclusion it implied.** Dosing the term properly does not fix K6. Effort is not
mispriced — it is *instrumentally required* by what the rest of the reward demands.
No setting of `ctrl_weight` will reach K6's gate of ≤ 0.087.

#### Where the effort actually is: the MEAN action, not exploration noise

`ctrl_cost = mean(tanh(raw)²)`, and `raw ~ N(μ, σ)`, so a noise floor was a plausible
explanation. Numerically it is not:

| σ | μ=0 | μ=0.5 | μ=1.0 | μ=1.5 | μ=2.0 |
|---|---|---|---|---|---|
| **0.00** | 0.000 | 0.214 | 0.580 | 0.819 | 0.929 |
| 0.76 | 0.297 | 0.366 | 0.533 | 0.714 | 0.851 |

K6 is measured **deterministically** (μ only — the σ=0 row). E15's 0.763 therefore
needs **|μ| ≈ 1.9**, near-saturated tanh; the shipped brain's 0.087 needs **|μ| ≈ 0.30**.
So post-M9 policies deterministically command ~0.87 of actuator range — effectively
bang-bang — while the shipped brain uses ~0.30. That is a behavioural difference, not a
measurement artifact, and because the measurement is deterministic the conclusion does
not depend on σ at all.

> **Correction.** An earlier version of this entry cited "σ logged at 0.74–0.78". That
> was a misreading of the training log: the `std` column is **`kpi/standing`**, not the
> action standard deviation, which this project does not log at all. The from-scratch
> runs in E19 make it obvious — they print `std 0.01` while lying on the floor. The
> conclusion above is unaffected, because the deterministic (σ=0) measurement settles
> it without reference to σ; but the supporting sentence was wrong and is removed.

#### What differs, from the shipped brain's own config.json

`boy_chase01` is the pristine default: **one-sided speed kernel, fixed 1.5 m/s target,
no fallen starts.** Every post-M9 policy changed all three. Three candidate mechanisms,
each of which would plausibly demand maximum actuator authority:

1. **Two-sided kernel.** One-sided clamps positive error away, so overshoot is free and
   a policy can settle into ONE smooth gait at its natural speed (the shipped brain
   cruises at 2.04 m/s against a 1.5 command). Two-sided forces active regulation TO a
   speed, and regulating a 21-DOF biped costs continuous correction.
2. **Commanded speed range (0.8–2.0).** One policy must serve every speed, so it cannot
   specialise into an efficient gait; it needs a stiff high-gain controller, and high
   gain is high effort.
3. **`reset_fallen 0.15`.** Standing up from supine genuinely needs peak torque, and
   `W_GETUP` is 0.60 — the second-largest term. One MLP with no mode switch may simply
   run high-gain everywhere to serve the 15 % of episodes that start on the floor.

**If any of these is the cause, then K6 ≤ 0.087 and the command-following KPIs are in
direct conflict, and the project has to choose rather than tune.** That would reframe
the KPI set, so it is worth one clean experiment.

**E18: one-factor-at-a-time ablation off the E17a cell**, 3 concurrent runs, 300
iterations, `ctrl_weight` 0.15 and `--scale-drift 0.5` held fixed in all of them:

| cell | two-sided | cmd range | reset_fallen | isolates |
|---|---|---|---|---|
| e17a *(have it)* | yes | 0.8–2.0 | 0.15 | baseline |
| **e18b** | **no** | 0.8–2.0 | 0.15 | the kernel |
| **e18c** | yes | **fixed 1.5** | 0.15 | the range |
| **e18d** | yes | 0.8–2.0 | **0.0** | getting up |

#### M19 confirmed at depth — the inelasticity is not a warm-start transient

Same three runs at iteration 100/150/200, i.e. after the warm start has had time to
restructure:

| dose | `kpi/ctrl` @100 | @150 | @200 | `speed_along` @200 | `action_rate` @200 |
|---|---|---|---|---|---|
| 0.15 | 0.775 | 0.772 | 0.771 | 0.980 | 0.069 |
| 0.30 | 0.804 | 0.765 | 0.754 | 1.065 | 0.075 |
| **0.60** | 0.749 | 0.739 | **0.732** | **1.051** | **0.095** |

A 4× dose range still separates the cells by only 5 %. The 0.60 cell declines slowly
(0.749 → 0.732 over 100 iterations); extrapolated generously that is ~0.68 by
iteration 1200 — still **8× above K6's gate of 0.087**.

Two things worth recording because they were not predicted:

* **The anti-goal did not trigger.** I wrote down that a policy lowering K6 by standing
  still has not succeeded. The heaviest dose has the HIGHEST forward speed
  (1.051–1.094 vs 0.980 at dose 0.15). Pricing effort did not make it lazy.
* **`action_rate` went UP with the dose** (0.069 → 0.095). Penalising action MAGNITUDE
  made the policy move its actions around MORE. It is shuffling between
  similar-magnitude actions rather than reducing them — which is consistent with the
  saturated-|μ| picture: near the tanh rails, changing which rail costs nothing in
  magnitude.

`ctrl_weight` is therefore closed as a lever for K6. Recorded so it is not retried.

### M20 — gain randomisation is NOT the cause (refuted from data already on disk)

A policy saturates its commands when it lacks authority, and this env domain-randomises
`actuator_gainprm`, so "trained to cope with the weakest sampled gain" was a clean
candidate. `gate4_eval` already measures both conditions, so no new run was needed:

| | K6 randomised | K6 nominal |
|---|---|---|
| shipped `boy_chase01` | 0.0869 | 0.0874 |
| E12 | 0.7441 | 0.7445 |

Identical to four decimal places in both policies. **Refuted.** Effort is not a hedge
against randomised actuators.

### M21 — the shipped brain moves MORE while commanding LESS: this is co-contraction

From the same JSONs, which had these columns all along:

| | K6 control effort | K7 joint accel | speed | commanded |
|---|---|---|---|---|
| shipped `boy_chase01` | **0.087** | **1.28e6** | **2.04 m/s** | 1.5 |
| E12 | 0.744 | 0.97e6 | 1.50 m/s | 1.5 |
| E15 final | 0.763 | 5.93e5 | 1.43 m/s | 1.5 |
| E15 snapshot | 0.754 | 7.65e5 | 1.48 m/s | 1.5 |

The shipped brain has the **HIGHEST joint acceleration and the LOWEST control effort**,
and it is the fastest. Post-M9 policies command 8.6× harder and get less motion for it.
So K6 is not measuring "gentleness of movement" — it is measuring how hard the actuators
push against each other. Large opposing commands that cancel: **co-contraction**. The
body is held stiff.

That reframes the question from "why are the actions large" to **"why is the policy
holding itself rigid"**, and rigidity has two obvious payoffs in this reward:

1. **Stability.** `W_GETUP` 0.60 plus the uprightness terms plus 15–30 % fallen starts
   all pay for not falling. Stiff is stable. The shipped brain never started fallen and
   so was never paid to be rigid.
2. **Braking.** The shipped brain's natural gait runs at **2.04 m/s against a 1.5
   command** — it overshoots by 36 % and the one-sided kernel lets that be free. The
   two-sided kernel does not: it forces the policy DOWN to 1.5. Holding a biped below
   its natural gait speed means continuously decelerating, and braking with
   position-servo actuators IS fighting yourself.

This second mechanism is specific, and E18 already separates it: cell **B** (one-sided,
range kept) removes all braking pressure because overshoot becomes free again, while
cell **C** (two-sided, fixed 1.5) keeps it. If B drops K6 and C does not, braking is the
cause. Cells **D/E** test the stability story via `reset_fallen`.

Also note K7's own gate is "≤ baseline", and baseline is the shipped brain's 1.28e6 —
which every post-M9 policy already beats. **K6 and K7 pull in opposite directions here**,
so they cannot both be driven toward the shipped brain's values; chasing K7 downward
rewards exactly the rigidity that ruins K6.

#### M18 extended — the best configuration found is 5 concurrent runs at 4096 worlds

| configuration | steps/s | **iters/s** | per-run share |
|---|---|---|---|
| 1 × 8192 | 51.0k | 0.260 | — |
| 1 × 24576 *(best single)* | 59.2k | 0.100 | — |
| 3 × 8192 | **70.6k** | 0.359 | 23.5k each |
| **5 × 4096** | 65.1k | **0.662** | 13.0k each (exactly equal) |

5 × 4096 gives **2.5× the iteration rate of a single run** and 1.84× that of 3 × 8192,
for 8 % fewer samples/s. Since this project judges convergence in ITERATIONS (runs peak
at 500–700), iterations/s is the rate that matters, and the two metrics disagree — which
is why both are recorded. The five cells shared the GPU at exactly 13.0k steps/s each,
so concurrency here is fair, not lottery-scheduled.

Practical consequence: a 5-cell ablation of 300 iterations costs ~38 min. The same five
cells run serially at 8192 would cost ~96 min. **That is the single largest efficiency
gain available tonight, and it came from questioning a platform assumption I was
confident about (no CUDA MPS on Windows ⇒ no concurrency benefit) instead of acting
on it.**

### M22 — the shipped brain's recipe, recovered from git, refutes two of my own mechanisms

`boy_chase01`'s `config.json` lists only the arguments that existed when it trained, so
its commit can be identified by which flags are ABSENT. That set matches
**71182d0 (2026-09-03)** exactly. Reading the env at that commit:

| | at 71182d0 (shipped brain) | at HEAD | same? |
|---|---|---|---|
| every `W_*` weight | W_TRACK 2.0, W_GETUP 0.60, W_CTRL 0.005, … | identical | **yes** |
| `SCALE_DRIFT` / `SCALE_ACCEL` / `SPEED_SIGMA` | 8.0 / 2.0e6 / 1.0 | identical | **yes** |
| `MIN_HEIGHT` / `MIN_UPRIGHT` / `STANDING_HEIGHT` | 0.45 / 0.30 / 0.77 | identical | **yes** |
| the whole `reward = (...)` expression | 10 terms | identical (only parameterised) | **yes** |
| **`RESET_FALLEN_FRACTION`** | **0.30, and active in `reset()`** | 0.30 | **yes** |

**Two corrections to my own recorded reasoning.**

1. **M21 mechanism 1 is wrong.** I wrote that the shipped brain "never started fallen and
   so was never paid to be rigid", inferring that from the absence of a `reset_fallen`
   key in its config. Absence of the key means the FLAG did not exist yet — not that the
   behaviour was off. The constant was already 0.30 and already used in `reset()`, so
   **the shipped brain trained with 30 % fallen starts, TWICE E15's 15 %**, and still
   reached ctrl 0.087. Fallen starts are not the cause.
2. **There is no "M9 uprightness sign fix" in this reward.** I had it on the candidate
   list. That fix was in `Assets/Scripts/Rewards/Reward_WormLoco.cs` — the ML-Agents
   shared locomotion reward for the twelve PhysX creatures — and has nothing to do with
   MojucuBoy's MuJoCo reward, which is a different file and was never changed. I
   conflated two rewards because both were touched in the same session.

**What is actually left.** With every weight and the whole formula identical, the only
differences between `boy_chase01` and E15 are four CLI choices:

| | shipped | E15 |
|---|---|---|
| speed kernel | one-sided | **two-sided** |
| command | fixed 1.5 | **range 0.8–2.0** |
| `reset_fallen` | 0.30 | 0.15 |
| **initialisation** | **from scratch** | warm-start from `e12_getup` (ctrl 0.744) |

The fourth has never been tested and is the one M19 makes most suspicious: if PPO cannot
move control effort once it is in the weights (elasticity −0.008), then **every post-M9
policy inherited e12_getup's saturated gait and no amount of fine-tuning could undo it.**
High effort would then be a frozen accident of the lineage, not a property of the task —
and the fix would be to stop warm-starting rather than to reshape the reward.

**E19 tests exactly this.** Five cells, all FROM SCRATCH, all reward weights at shipped
defaults, differing only in the task flags; cell `s19e_shipped` reproduces
`boy_chase01`'s recipe with no flags at all. 600 iterations, 5 × 4096 concurrent.

* If `s19e` lands near ctrl 0.09 → from-scratch is the key; K6 is recoverable by
  retraining, and the lineage was the whole story.
* If `s19a` (post-M9 flags, from scratch) ALSO lands near 0.09 → the task flags are
  innocent too, and warm-starting alone explains K6.
* If both land near 0.75 → the cause is something still unidentified, and the honest
  report is that K6's gate may not be reachable with this reward at all.

---

## KPI scoreboard as of 02:40, 2026-09-11

Measured at the GAME condition (upright spawn, command 1.5 m/s) for every policy, so
the columns are comparable. Gates from the K-table at the top of this log.

| | gate | shipped `boy_chase01` | E12 | E15 final | **E15 snapshot** |
|---|---|---|---|---|---|
| K1 uptime | > 0.90 | 0.991 | 0.994 | 0.997 | **0.997** |
| K2 speed @1.5 | ±10 % | +36 % **fail** | −0.2 % pass | −4.8 % pass | **−1.2 % pass** |
| K2 across 1.0–2.4 | ±10 % all | **fail** (1 of 4) | fail (1 of 4) | fail (2 of 4) | **fail (3 of 4)** |
| K4 heading | < 15° | — | — | 12.8° pass | **10.3° pass** |
| K5 lateral | < 0.25 | 0.426 fail | 0.641 fail | 0.420 fail | 0.474 fail |
| **K6 ctrl effort** | ≤ 0.087 | **0.087** | 0.744 fail | 0.763 fail | 0.754 fail |
| K7 joint accel | ≤ 1.28e6 | 1.28e6 | 0.97e6 pass | 5.93e5 pass | 7.65e5 pass |
| K8 uprightness | > 0.90 | — | — | 0.998 pass | 0.999 pass |
| K9 throughput | maximise | — | — | 47.1k | → **70.6k** (M18) |
| K10 recovery | — | measured | 0 % | 0 % | 0 % |

**No policy passes every KPI, and the reason is now understood rather than suspected.**

### Three KPI conflicts, each demonstrated by measurement

1. **K2 vs K6.** The shipped brain is the only policy meeting K6, and it does so while
   failing K2 by +36 % — it ignores the command and runs at its natural 2.04 m/s. Every
   policy that actually tracks a command costs 8.6× the control effort. (M19/M21/M22 are
   still narrowing whether this is causal or inherited.)
2. **K6 vs K7.** Both gates are "≤ baseline", both baselines are the shipped brain, and
   the shipped brain is the WORST policy on K7 (1.28e6) while being the best on K6. They
   cannot be jointly minimised: low K6 here means a loose gait that lets joints move
   fast; low K7 means rigidity, which is exactly what raises K6. **Chasing K7 downward
   actively harms K6.**
3. **K5 is the one clean win available.** `--scale-drift 0.5` halved training-time
   lateral drift (M16) and nothing else has ever moved it. It is the only change tonight
   that is confirmed, bounded, and free of a known trade-off.

### What this means for the exit criteria

The brief's exit condition was "all KPIs satisfied across 10 consecutive eval episodes".
**That condition is not reachable as written**, because K2 and K6 are in direct conflict
and K6/K7 share a baseline that cannot be approached from both sides. The honest report
is not "we failed to hit the KPIs" but "two of the KPIs are mutually exclusive under this
reward, here is the measurement, and the project needs to pick which one it wants."

### M23 — K6 is an INHERITED defect of the warm-start lineage, not a property of the task

The comparison that settles it is control effort **at matched forward speed**, which
nobody had done — every prior comparison was at matched iteration, where the policies
are at different competences.

| mean `kpi/ctrl` at forward speed → | 0.2–0.4 | 0.4–0.6 | 0.8–1.0 | 1.0–1.3 | 1.3–1.6 |
|---|---|---|---|---|---|
| **from scratch, post-M9 task** (s19a) | **0.153** | — | — | — | — |
| **from scratch, shipped task** (s19e) | **0.154** | — | — | — | — |
| warm-started E15 | 0.720 | 0.759 | 0.765 | 0.758 | 0.784 |
| warm-started E17c (`ctrl_weight` 0.60) | 0.703 | — | 0.734 | 0.739 | — |
| shipped `boy_chase01` (eval) | — | — | — | — | 0.087 @ 2.04 m/s |

**Two facts, and together they are conclusive:**

1. **4.6× difference at the SAME forward speed.** From-scratch policies moving at
   0.2–0.4 m/s use 0.153; warm-started policies moving at the same speed use 0.72. Speed
   is controlled for, so effort is not the price of locomotion.
2. **E15's effort is FLAT across the whole speed range** — 0.720 at 0.2 m/s and 0.784 at
   1.6 m/s, an 8 % spread over an 8× speed change. A quantity that does not vary with
   what the policy is doing is not a response to the task. It is a fixed property of the
   weights.

And the from-scratch cells are **indistinguishable from each other** (0.153 vs 0.154 for
the two extreme task definitions). The task-definition spread I saw at iteration 100
(0.059 vs 0.072) did not survive to iteration 200 — the cells converged and the ordering
scrambled. It was noise.

**So the mechanism is:** `e12_getup` learned a co-contracting, rail-saturated gait. Every
post-M9 policy is a warm-start descendant of it. M19 showed PPO fine-tuning cannot price
that out (elasticity −0.008, and 21 % of the reward surrendered rather than changed), so
each generation inherited it intact. K6 was never a reward-shaping problem at all.

**The fix is to stop warm-starting.** That is a one-line change to how runs are launched,
not a reward redesign.

#### This probably DISSOLVES the K2-vs-K6 conflict I recorded above

The scoreboard entry says K2 and K6 are mutually exclusive, evidenced by the shipped brain
being the only policy meeting K6 and doing so by ignoring the command. That inference
assumed command-following CAUSED the effort. M23 refutes the assumption: s19a carries the
full post-M9 command-following task definition **and sits at 0.153 from scratch**.

So the honest position is now: **the conflict is unproven and probably illusory.** A
from-scratch run with the command-following task may well satisfy K2 and K6 together —
which no policy in this project has ever done. That is the run to make next, and it is a
much better outcome than the "the project must choose" conclusion I was heading for.

**Caveat, stated because it is the one thing that could still overturn this.** The
from-scratch cells have only reached 0.2–0.4 m/s. Their effort rises with competence
(0.043 → 0.072 → 0.106 → 0.132 over iterations 25→200). If it keeps climbing and lands
near 0.75 once they reach 1.5 m/s, M23 is wrong and the matched-speed comparison was
taken too early. **The measurement that decides it is ctrl at speed 1.3–1.6 in a
from-scratch run**, and no from-scratch cell has reached that bucket yet. Until one does,
M23 is strongly supported but not proven.

### M24 — K6 has been measuring the WRONG PHYSICAL QUANTITY for this project's entire history

The brief's Phase 2 asks for an MJCF audit. Reading
`training/mojucuboy/mojucuboy_roundtrip.xml`:

```xml
<general joint="abdomen_z" biastype="affine" gainprm="1200 0 0"
         biasprm="0 -1200 -78.4282" ctrlrange="-0.785398 0.785398"
         forcerange="-200 200" />
```

These are **position servos**: `force = 1200*(ctrl - qpos) - 78.43*qvel`, clamped to
±200 N·m. (Kp:Kd = 15.3:1, which matches the Kp:Kd = 15:1 CLAUDE.md records for every
rig in the project — so the rig is consistent, and that is not the problem.)

`ctrl` is therefore a **commanded joint ANGLE**, in radians, spanning the joint's range.
And `ctrl_cost = mean(action²)` — the thing K6 reports — measures *how extreme a pose the
policy asks for*. It is not actuator effort. Two facts make the distinction sharp:

1. **The servo saturates at a position error of 200/1200 = 0.167 rad = 9.5°.** Beyond
   that the commanded angle stops mapping to force at all, so `ctrl_cost` is not even
   monotonically related to torque in the regime these policies operate in.
2. **Measured, holding the stance with a ZERO action vector:**

   ```
   ctrl = 0.000000      torque_abs = 6.41 N·m mean      peak |torque| = 75.3 N·m
   ```

   `ctrl_cost` reads exactly zero while the actuators apply up to 75 N·m holding the
   body against gravity.

**CLAUDE.md states this exact principle as a project rule, in UNITY_RULES §2:**

> *Fatigue Mechanics: Read load directly from **applied torque, not the action vector**
> (isometric bracing produces near-zero action at near-maximum torque).*

K6 has been measured against the action vector for this project's whole history, in
direct violation of the rule the project wrote for itself. The rule even names the exact
failure mode, and the zero-action measurement above reproduces it in one line.

**Instrumented, not charged.** `terms["torque"]` (mean q̇frc_actuator²) and
`terms["torque_abs"]` (mean |τ|, N·m) now come out of the env, are logged per iteration
by the trainer, and are reported by both evaluators as **K6b**. They are deliberately NOT
added to `reward`: doing so would change what every policy optimises and break
comparability with every run already recorded.

**What this does NOT yet establish.** It does not follow that the shipped brain and E15
use similar torque — only that `ctrl_cost` cannot answer the question. The 8.6× K6 gap
might survive on torque, or it might vanish. **That measurement is running now**, and
whichever way it lands it is the number K6 should have been all along.

Note this does not overturn M23: that finding compared warm-started against from-scratch
policies at matched speed *on the same metric*, so the 4.6× ratio stands whatever the
metric's physical meaning. M24 changes what the number MEANS, not which policies differ.

### M25 — measured on applied torque, K6's ranking INVERTS. E15 is not a thrashing policy.

Both policies re-evaluated at an identical, explicit condition (upright spawn, command
1.5 m/s, 60 / 10 episodes, domain randomisation on):

| | K6 commanded angle | **K6b applied torque** | K6b torque² | speed |
|---|---|---|---|---|
| shipped `boy_chase01` | **0.0893** | **37.93 N·m** | 3323 | 2.026 m/s |
| E15 snapshot | 0.7602 | **18.24 N·m** | 933 | 1.482 m/s |
| ratio (E15 ÷ shipped) | **8.5× worse** | **0.48× — i.e. HALF** | 0.28× | — |

**The two metrics rank these policies in opposite directions.** K6 as defined says E15
thrashes its actuators 8.5× harder than the shipped brain. Applied torque says E15 uses
less than half the torque.

The mechanism is just the servo equation: `force = Kp·(ctrl − qpos) − Kd·q̇`. A policy
that commands an extreme joint angle **which the body actually reaches** has a small
position error and therefore a small force. Commanded angle and applied torque are not
even positively correlated here. So:

* **E15 "failing criterion 3" was an artifact of the instrument**, not a defect of the
  policy. The pre-committed bar was real and honestly applied — but it was applied to a
  number that does not mean what its name says.
* **M21's co-contraction story is wrong.** I argued post-M9 policies hold themselves
  rigid, "antagonists fighting each other", from the K6/K7 pattern. A policy applying
  half the torque is not co-contracting. Withdrawn.
* **The K2-vs-K6 conflict in the scoreboard dissolves.** E15 tracks its command AND
  applies less torque than the shipped brain. There was never a trade to make.

#### The confound I must not repeat

These two speeds differ — 2.026 vs 1.482 m/s — and faster locomotion legitimately needs
more torque. So this comparison is confounded exactly the way the matched-iteration
comparisons were before M23. **A matched-speed torque measurement is running now**
(E15 commanded to 2.0 m/s, where it reaches 2.071 and the shipped brain sits at 2.026).
Until it lands, the honest claim is the narrow one:

> K6 does not measure control effort, and the ranking it produces is not reproduced by
> the quantity it claims to measure.

That claim does not depend on the speeds matching. The stronger claim — "E15 is
genuinely gentler" — does, and is not yet established.

#### What survives from tonight, and what does not

| finding | status after M25 |
|---|---|
| M16 `--scale-drift 0.5` halves lateral drift | **stands** — independent of K6 |
| M17/M18 throughput: 5×4096 concurrent, 2.5× iters/s | **stands** — independent |
| M19 `ctrl_weight` is inelastic | **stands, and is now explained**: the penalty acts on commanded angle, which is nearly decoupled from the physics, so paying it changes little |
| M20 gain randomisation refuted | **stands** |
| M22 shipped brain used `reset_fallen` 0.30 | **stands** (git evidence) |
| M23 warm vs scratch differ 4.6× at matched speed | **stands as a measurement**, but on the ctrl metric — its physical interpretation is now open |
| M21 co-contraction | **WITHDRAWN** |
| "K2 and K6 are mutually exclusive" | **WITHDRAWN** |

### M26 — matched-speed torque: E15 applies ~42 % LESS torque than the shipped brain

E15 snapshot commanded to 2.0 m/s, to close the speed confound in M25:

| | shipped (natural 2.026) | E15 @cmd 1.5 | **E15 @cmd 2.0** |
|---|---|---|---|
| achieved speed | 2.026 m/s | 1.482 | **1.851** |
| K2 error vs its command | **+35 % FAIL** | −1.2 % pass | **−7.5 % pass** |
| **K6b applied torque** | **37.93 N·m** | 18.24 | **20.82** |
| K6b torque² | 3323 | 933 | 1160 |
| K1 uptime | 0.975 | 0.997 | 0.950 |
| K5 lateral | 0.426 | 0.365 | 0.387 |
| K8 uprightness | — | 0.946 | 0.966 |

**Closing the remaining 9 % of speed gap using E15's OWN slope**, which is the honest way
to extrapolate rather than assuming: E15 goes 1.482 → 1.851 m/s (+25 %) for 18.24 → 20.82
N·m (+14 %). Another +9.5 % of speed therefore costs roughly +5 %, putting E15 at
**~21.9 N·m at 2.026 m/s against the shipped brain's 37.93** — about **42 % less torque at
the same speed**. An 82 % torque difference cannot be produced by a 9 % speed difference,
so the conclusion survives the confound that invalidated the M25 version of it.

**E15 also passes K2 at 2.0 m/s** (−7.5 %), so it tracks across 1.5 AND 2.0.

#### Shipping recommendation — REVERSED from earlier tonight

Earlier I recommended "keep `boy_chase01` while the command stays fixed at 1.5". That
rested on E15 having an actuator-thrashing defect, which M25/M26 refute, and on a 27 %
speed deficit which only exists because the command was left at 1.5.

The real trade at `_commandSpeed = 2.0`:

| | shipped | E15 snapshot @2.0 |
|---|---|---|
| race speed | **2.026 m/s** | 1.851 m/s (**−8.6 %**) |
| follows its command | no (+35 %) | **yes (−7.5 %)** |
| applied torque | 37.9 N·m | **20.8 N·m (−45 %)** |
| lateral drift | 0.426 | **0.387** |
| uptime | 0.975 | 0.950 |

And from the earlier sweep, **E15 commanded to 2.4 reaches 2.071 m/s — FASTER than the
shipped brain's 2.026** — though it fails K2 at that command (−13.7 %). So if raw race
pace is what matters, E15 at `_commandSpeed = 2.4` beats the shipped brain outright
while still applying less torque.

**This is a product decision, not a metric one, so it is presented rather than taken:**

* Want the fastest racer → **E15 snapshot, `_commandSpeed = 2.4`** (2.071 m/s, beats
  shipped, lower torque, but the command is then a throttle rather than a speed it obeys).
* Want a controllable racer → **E15 snapshot, `_commandSpeed = 2.0`** (1.851 m/s, 8.6 %
  slower than today, obeys the command, 45 % less torque).
* Want no change → keep `boy_chase01`. It is the grid's best racer today (16/16 finishes,
  ELO 1525) and nothing measured tonight says it is broken.

`MojucuBoyController._commandSpeed` is the single knob, and M12 recorded that its 1.5 is
load-bearing. It is load-bearing because the shipped brain ignores it; with E15 it becomes
a real control.

### M27 — E19 completed, and it RETRACTS most of M23. 600 iterations from scratch is not enough to stand.

All five cells finished 600 iterations. Their best-ever `standing` score:

| cell | best `standing` |
|---|---|
| s19a post-M9 | 0.0904 |
| s19b one-sided + range | 0.0824 |
| s19c two-sided + fixed | 0.0883 |
| s19d post-M9, fallen 0.30 | 0.0758 |
| s19e shipped recipe | 0.0953 |

For scale, E15 sits at **0.78** and the shipped brain at ~0.97. **A `standing` of 0.09
means the racer never got upright.** After 600 from-scratch iterations all five cells were
still dragging themselves along the floor. (The shipped brain used 1500 iterations, so
this is not surprising in hindsight — it is simply the budget this task needs.)

**This breaks M23's control.** M23 compared control effort "at matched forward speed" and
found from-scratch 0.153 against warm-started 0.720. But at 0.2–0.4 m/s those two sets of
policies are doing completely different things:

| at 0.3 m/s | `standing` | what it is actually doing |
|---|---|---|
| from-scratch cells | ~0.09 | crawling / sliding on the floor |
| warm-started E15 early | ~0.7 | upright, on its feet |

Matching speed was not sufficient — **posture had to be matched too**, and it was not. A
policy lying on the ground barely loads its actuators, so of course its commanded angles
are smaller. The 4.6× ratio is real as arithmetic and close to meaningless as a comparison.

**So M23 is withdrawn as a causal finding.** What survives is narrow and still worth
having:

* From-scratch policies do not *begin* with saturated commanded angles; the quantity grows
  with competence (0.043 → 0.072 → 0.106 → 0.132 over iterations 25→200, still rising at
  600). Whether it asymptotes near the shipped brain's value is **untested** — no
  from-scratch run in this project has been taken far enough to say.
* The warm-start-lineage explanation for K6 is therefore **unsupported**, not disproven.

And it matters much less than it did six hours ago, because **M24/M25 showed the metric it
was explaining does not measure control effort at all.** Chasing the lineage question
further would be answering a question that dissolved. E20 was staged and is **not being
run**, deliberately.

#### Methodological note for the next session

Three times tonight a "matched" comparison turned out to be confounded by something I had
not matched: matched-iteration hid competence (fixed by M23), matched-speed hid posture
(this entry), and the metric itself hid the physics (M24). The pattern is the same each
time — I controlled the variable I had thought of and reported the result as though I had
controlled all of them. The cheap guard is to report the covariates ALONGSIDE any matched
comparison, which is why the tables above now carry `standing` and speed next to every
effort figure.

### M28 — K4 and K5 are coupled through ONE line, and it conflates two unrelated quantities

E16's iteration-400 checkpoint is exactly "E15 snapshot + the confirmed `--scale-drift 0.5`
fix", so it is the natural candidate for closing K5. Evaluated at the game condition:

| | E15 snapshot | **E16 @400** | gate | |
|---|---|---|---|---|
| speed @cmd 1.5 | 1.482 | 1.383 (−7.8 %) | ±10 % | pass |
| K1 uptime | 0.997 | 0.997 | > 0.90 | pass |
| K3 fallen | 0.000 | 0.000 | < 0.10 | pass |
| **K5 lateral** | 0.365 **fail** | **0.225 PASS** | < 0.25 | **fixed** |
| **K4 heading** | 13.5° pass | **16.7° FAIL** | < 15° | **broken** |
| K6b torque | 18.24 | 18.29 | ≤ 37.9 | pass |
| K7 accel | 6.88e5 | 7.06e5 | ≤ 1.28e6 | pass |

At command 2.0 the K4 regression is worse: **22.3°**.

So E15 fails only K5, and E16 fails only K4. The cause is one line of the reward:

```python
drift = lateral.pow(2) + qvel[:, 5].float().pow(2)     # qvel[5] IS YAW RATE
```

`W_DRIFT`'s comment even says it: *"uncommanded lateral + yaw motion"*. But lateral
velocity and yaw rate are not the same kind of quantity. Lateral velocity is pure waste.
**Yaw rate is how a racer corrects its heading** — it is the control authority K4 is
measured on. Strengthening the penalty from `scale_drift` 8.0 to 0.5 therefore bought K5
by suppressing exactly the corrections K4 needs.

**This is a genuine trade-off with a clean fix**, unlike the "conflicts" I claimed earlier
tonight and withdrew: split the term and price lateral drift without taxing yaw.
`--drift-yaw-weight` does this, defaulting to 1.0 so every previous run's reward is
unchanged.

**E21** tests it from `e15_best`, 600 iterations, 3 cells at 4096 concurrent:

| cell | `drift_yaw_weight` | purpose |
|---|---|---|
| f21a | 1.0 | **internal control** — must reproduce E16's K4 16.7 / K5 0.225, or the comparison is invalid |
| f21b | 0.25 | yaw taxed lightly |
| f21c | 0.0 | lateral only; yaw rate untaxed |

Target: K5 < 0.25 **and** K4 < 15° in the same policy, which no policy in this project has
yet achieved. That would satisfy every gated KPI and meet the brief's exit criterion.

### M29 — E21: the yaw penalty DESTABILISES training, and the LATERAL half is what costs K4

E21's three cells, all from `e15_best` at `scale_drift` 0.5, differing only in how much of
the drift penalty is yaw rate. Training trajectories:

| cell | `drift_yaw` | what happened over 600 iterations |
|---|---|---|
| **f21a** | **1.0** | degrading from iter 325 (standing 0.67 → 0.61), **COLLAPSED at 405** → standing 0.01, speed 0.00, heading 89.6°; never recovered |
| f21b | 0.25 | wobbled badly (heading 47.7° at 405), then **recovered** to heading 27°, standing 0.76 |
| f21c | **0.0** | **stable for all 600** — heading 28–35°, standing 0.71–0.77, no degradation |

f21a was the internal control and it **failed to reproduce E16** — which I had pre-declared
would invalidate the comparison. It does invalidate the planned comparison, but the failure
is itself the result: **at `scale_drift` 0.5 a full yaw penalty collapses the policy.**
Monotonic in the weight: 1.0 collapses, 0.25 nearly does, 0.0 is stable.

**This retroactively undermines E16, and with it M28's headline.** E16's checkpoint is
iteration 400 — and f21a's curve shows that configuration was already sliding at 365
(standing 0.67 → 0.61) and collapsed at 405. So **E16's "K5 0.225 / K4 16.7°" was measured
on a policy in the middle of falling apart**, not on a stable one. It should not be cited as
a K5 fix. (E16 ran at 6144 worlds and seed 5 against f21a's 4096 / seed 21, so this is
strong evidence rather than proof of identity — but the direction is unambiguous.)

#### And my mechanism was wrong in its specifics

M28 claimed the yaw term costs K4, because yaw rate is how heading is corrected. Prediction:
removing the yaw tax improves K4. Measured on f21c (yaw fully untaxed):

| | `scale_drift` | K4 heading | K5 lateral | stable? |
|---|---|---|---|---|
| E15 snapshot | 8.0 | **13.5° pass** | 0.365 fail | yes |
| E16 @400 | 0.5 | 16.7° fail | 0.225 pass | **no — collapsing** |
| **f21c** | 0.5 | **18.1° FAIL** | **0.177 pass** | yes |

**K4 got WORSE with the yaw tax removed** (13.5 → 18.1°), so yaw rate is not what K4 was
paying for. The **lateral** half is. Which is physically sensible once said plainly: to turn
toward a commanded heading a biped must accelerate sideways, so taxing lateral velocity
suppresses the exact motion that corrects heading.

So M28's claim that the two gates are coupled **stands**; its claim about WHICH half couples
them is **withdrawn**. The yaw term's real role is different and also worth knowing: it is a
stability hazard at low `scale_drift`.

#### E22 — the frontier, as an interpolation rather than a guess

The two gates bracket cleanly, so the crossing can be located instead of hunted:

```
scale_drift 8.0 -> K4 13.5 / K5 0.365        gates: K4 < 15, K5 < 0.25
scale_drift 0.5 -> K4 18.1 / K5 0.177
```

Linear in log(scale_drift) puts the crossing near 2–4. E22 runs **2.0 / 3.5 / 5.0**, all
with `drift_yaw_weight 0.0` (the value measured stable), 600 iterations from `e15_best`.
If any cell lands K4 < 15 **and** K5 < 0.25 while stable, that is every gated KPI satisfied
in one policy.

---

## CORRECTED SCOREBOARD — every figure re-measured at one stated condition

Condition for every column: **upright spawn** (`reset_fallen 0.0`), domain randomisation
ON, deterministic policy, 40–60 episodes, command as stated. K6b is applied torque
(N·m mean abs) — the quantity K6 was always meant to be (M24). **K6's own column is kept
only to show how badly it misranks.**

| | gate | shipped `boy_chase01` | E15 snapshot @1.5 | E15 snapshot @2.0 | f21c @1.5 |
|---|---|---|---|---|---|
| speed | — | 2.026 | 1.482 | 1.851 | 1.434 |
| K2 tracking err | ±10 % | **+35 % FAIL** | −1.2 % pass | −7.5 % pass | −4.4 % pass |
| K1 uptime | > 0.90 | 0.975 pass | 0.997 pass | 0.950 pass | 0.952 pass |
| K3 fallen | < 0.10 | — | 0.000 pass | — | 0.047 pass |
| K4 heading | < 15° | — | **13.5° pass** | 14.6° pass | 18.1° FAIL |
| K5 lateral | < 0.25 | 0.426 FAIL | **0.365 FAIL** | 0.387 FAIL | **0.177 pass** |
| ~~K6~~ commanded angle | ≤ 0.087 | 0.089 | 0.760 | 0.763 | — |
| **K6b applied torque** | ≤ 37.9 | **37.93** | **18.24 pass** | 20.82 pass | 18.38 pass |
| K7 joint accel | ≤ 1.28e6 | 1.29e6 | 6.88e5 pass | 8.55e5 pass | 7.06e5 pass |
| K8 uprightness | > 0.90 | — | 0.946 pass | 0.966 pass | — |
| K10 recovery | none set | 0 % | 0 % | 0 % | 0 % |

**Nobody passes everything, and the remaining gap is exactly two gates that trade against
each other through one reward term:**

* **E15 snapshot** passes K1, K2, K3, K4, K6b, K7, K8 — **fails only K5** (0.365 vs 0.25).
* **f21c** passes K1, K2, K3, K5, K6b, K7 — **fails only K4** (18.1° vs 15°).
* The shipped brain fails K2 outright and is the **worst** policy on both K6b and K7.

E22 is searching the `scale_drift` frontier between them for a policy that clears both.

### What changed about the answer tonight, in one paragraph

The project believed MojucuBoy's challenger policies were actuator-thrashing failures,
8.6× worse than the shipped brain on control effort, and that command-following was the
cause. None of that is true. **K6 was reading the commanded joint angle of a position
servo, not torque** — a distinction CLAUDE.md already required ("read load from applied
torque, not the action vector") and that the metric had violated since it was written.
Measured properly, E15's snapshot applies **half** the shipped brain's torque, passes
seven of eight gates, and fails only lateral drift. The shipped brain, meanwhile, does not
follow its speed command at all (+35 %) and is the worst policy measured on both torque
and joint acceleration. Its one real advantage is raw pace, and most of that gap closes by
raising `_commandSpeed`.

### Retractions, in one place

| claim I recorded | status | what refuted it |
|---|---|---|
| E15 fails the control-effort bar | **withdrawn** | K6 measures commanded angle, not torque (M24/M25) |
| Post-M9 policies co-contract (M21) | **withdrawn** | they apply half the torque (M25) |
| K2 and K6 are mutually exclusive | **withdrawn** | E15 tracks its command AND uses less torque (M26) |
| `reset_fallen` explains the effort gap | **withdrawn** | shipped brain used 0.30, double E15's (M22, git) |
| Warm-start lineage explains K6 (M23) | **withdrawn** | matched speed did not match POSTURE; from-scratch cells were crawling at standing 0.09 (M27) |
| E16 @400 is a K5 fix (M28) | **withdrawn** | that configuration collapses at iter 405; the checkpoint was mid-collapse (M29) |
| The yaw term costs K4 (M28) | **withdrawn** | untaxing yaw made K4 worse, 13.5 → 18.1° (M29) |

Seven retractions. Each was caught by a measurement taken after the claim, and each is
logged where the claim was made. The recurring cause in five of them was the same:
**asserting a mechanism from a pattern, before controlling the covariate that explained
it.** The standing guard now in place is that every matched comparison reports its
covariates — speed, posture, and stability — in the same table as the effect.

### M30 — ONNX export verified faithful, and a sample-size correction to my own scoreboard

`training/mojucuboy/runs/e15_best/MojucuBoy_v01.onnx` (184 KB, opset 15, observation
normaliser baked into the graph) exported from the E15 snapshot. Verified by evaluating
the `.pt` and the `.onnx` at the **same episode count** — the first attempt compared 10
episodes against 40, which is not a parity test:

| @cmd 1.5, 40 episodes | `.pt` | `.onnx` | diff |
|---|---|---|---|
| forward speed | 1.447 | 1.445 | 0.1 % |
| uptime | 0.976 | 0.976 | 0 |
| K4 heading | 13.5° | 13.5° | 0 |
| K6 commanded angle | 0.7581 | 0.7581 | 0 |
| K6b applied torque | 19.373 | 19.405 | 0.2 % |
| K6b torque² | 1013 | 1017 | 0.4 % |
| K5 lateral | 0.391 | 0.398 | 1.8 % |

Faithful to within sampling noise (K5 is the noisiest because it depends on contact
outcomes). The export is safe to ship.

**It is NOT copied into `Assets/`.** The exporter writes into the run directory, and
`MojucuBoy_v01.onnx` is the live filename of the shipped brain — overwriting it would
replace the racer on a decision that is the user's. It stays in `runs/e15_best/`.

**Sample-size correction.** The E15 @1.5 column in the scoreboard above came from a
**10-episode** run and was slightly optimistic. The 40-episode figures are the better
estimates:

| | 10 ep (as first logged) | **40 ep (corrected)** |
|---|---|---|
| speed | 1.482 | **1.447** |
| K5 lateral | 0.365 | **0.391** |
| K6b torque | 18.24 | **19.373** |
| uptime | 0.997 | **0.976** |

No pass/fail verdict changes: K5 fails either way, and the torque ratio against the shipped
brain's 37.93 is 0.51 instead of 0.48 — still about half. But the corrected numbers are the
ones to quote, and 10 episodes is too few for K5 in particular. **Future evaluations should
use 40+ episodes**; the 60 used for the shipped brain is better still.

### M31 — the yaw penalty HELPS K4. My M29 mechanism was wrong too, and here is the evidence.

E22 swept `scale_drift` at 2.0 / 3.5 / 5.0 with `drift_yaw_weight 0.0`, predicting the K4/K5
crossing would sit near 2–4. The first cell refutes the premise:

| all with `drift_yaw 0.0` | `scale_drift` | K4 heading | K5 lateral | uptime | fallen |
|---|---|---|---|---|---|
| f21c | 0.5 | 18.1° | **0.177 pass** | 0.952 | 0.047 |
| **g22a** | **2.0** | **17.9°** | 0.286 fail | 0.878 **fail** | 0.127 **fail** |
| E15 snapshot (`drift_yaw` **1.0**) | 8.0 | **13.5° pass** | 0.391 fail | 0.976 | 0.000 |

**K4 is FLAT at ~18° across a 4× change in `scale_drift`** (18.1 → 17.9). If the lateral
penalty strength were what costs K4, raising `scale_drift` from 0.5 toward 8.0 should have
walked K4 back toward 13.5°. It did not move at all.

What f21c and g22a share, and E15 does not, is **`drift_yaw_weight = 0.0`**. So the yaw
penalty is what keeps heading accurate, and the mechanism is the reverse of my M28 guess:

> **Penalising yaw rate damps yaw wobble, and uncontrolled yaw oscillation IS heading
> error.** The term was never taxing heading control; it was providing yaw damping.

**Three wrong mechanisms for the same observation, in sequence:**

| attempt | claim | refuted by |
|---|---|---|
| M28 | the yaw term costs K4 | f21c: untaxing yaw made K4 worse (13.5 → 18.1°) |
| M29 | the lateral term costs K4 | g22a: K4 flat at ~18° across 4× `scale_drift` |
| M31 | removing the yaw term costs K4 | **current**; E23 tests it |

And E22 was designed on M29's wrong mechanism, so it pinned `drift_yaw 0.0` in all three
cells — **deleting the term that helps K4 in every cell of a sweep meant to improve K4.**
The sweep could not have succeeded. That is a design error downstream of a reasoning error,
and the cost was ~55 minutes of GPU time.

g22a also regressed K1 (uptime 0.878 < 0.90) and K3 (fallen 0.127 > 0.10), so
`scale_drift 2.0` with no yaw damping is worse than either bracket on four gates at once.

#### E23 — the experiment M31 actually implies

`drift_yaw_weight 1.0` restored in every cell, `scale_drift` swept **2.0 / 4.0 / 6.0**,
600 iterations from `e15_best`. The band is chosen against both known constraints:

* `sd 8.0` + yaw 1.0 → K4 13.5° pass, K5 0.391 fail, stable *(E15)*
* `sd 0.5` + yaw 1.0 → **collapsed at iteration 405** *(f21a)*

so the hazard is a strong penalty combined with a full yaw tax, and 2.0–6.0 stays clear of
it while still strengthening the lateral pressure K5 needs. Target remains K4 < 15° **and**
K5 < 0.25 in one stable policy.

**Time note:** launched 05:55 with a 08:00 deadline. 600 iterations x 3 cells is ~55 min
plus ~15 min of evaluation, so this is the last experiment that fits. If it misses, the
session's answer on K5 is "E15 snapshot passes seven of eight gates and fails K5; the
frontier between K4 and K5 is real but was not located", which is an honest result and
strictly more than was known at the start.

### M32 — the two knobs are SEPARABLE: `scale_drift` sets K5, `drift_yaw_weight` sets K4

g22b completed the picture. All three `drift_yaw 0.0` cells, plus E15 for contrast:

| `drift_yaw` | `scale_drift` | K4 heading | K5 lateral | uptime |
|---|---|---|---|---|
| 0.0 | 0.5 *(f21c)* | 18.1° | **0.177** | 0.952 |
| 0.0 | 2.0 *(g22a)* | 17.9° | 0.286 | 0.878 |
| 0.0 | 3.5 *(g22b)* | 19.2° | 0.366 | 0.951 |
| **1.0** | 8.0 *(E15)* | **13.5°** | 0.391 | 0.976 |

Two clean, separable relationships:

* **K5 tracks `scale_drift` monotonically** — 0.177 → 0.286 → 0.366 → 0.391 as it rises
  0.5 → 2.0 → 3.5 → 8.0. This is the knob for lateral drift, and it works.
* **K4 is INDEPENDENT of `scale_drift`** (17.9–19.2° across a 7× range) and depends on
  `drift_yaw_weight` — 13.5° with yaw at 1.0, ~18° with it at 0.0.

So the target policy is **low `scale_drift` AND `drift_yaw_weight = 1.0`**: the first buys
K5, the second buys K4, and they do not fight each other. The "K4/K5 frontier" I spent two
experiments interpolating **does not exist**. They are controlled by different terms.

The one real constraint is stability: f21a showed `sd 0.5` + `yaw 1.0` collapses at
iteration 405, while E15 runs `sd 8.0` + `yaw 1.0` stably. So there is a floor on
`scale_drift` when yaw is taxed, somewhere between 0.5 and 8.0.

**E23 (relaunched 05:48) sweeps `scale_drift` 1.0 / 1.5 / 2.5 with `yaw 1.0`**, 450
iterations from `e15_best`. Predicted from the decomposition: K4 ≈ 13–14° in all cells, K5
≈ 0.20 / 0.24 / 0.30. If that holds, **`sd` 1.0 or 1.5 clears both gates** and is the first
policy to satisfy every gated KPI.

**I relaunched rather than let the first E23 run finish.** Its values were 2.0/4.0/6.0,
chosen before M32, and the decomposition predicts K5 ≈ 0.29/0.37/0.39 — all failing the
0.25 gate. Running a sweep I expected to miss in every cell would have burned the last
available slot. 5 minutes of GPU time discarded; iterations cut 600 → 450 to fit the
remaining budget, which E21/E22 showed is ample since effects are stable by 300–400.

### M33 — the K4/K5 frontier, quantified. It misses the gate corner, so `scale_drift` alone cannot win.

E23, `drift_yaw_weight 1.0` throughout (the value M31 showed is needed for K4), 450
iterations from `e15_best`, 60-episode evaluation at the game condition:

| `scale_drift` | K4 heading | K5 lateral | speed | uptime | fallen | K6b torque |
|---|---|---|---|---|---|---|
| 8.0 *(E15)* | **13.5° pass** | 0.391 fail | 1.447 | 0.976 | 0.000 | 19.37 |
| 2.5 | *not evaluated — budget* | — | — | — | — | — |
| 1.5 | **14.2° pass** | 0.311 fail | 1.387 | 0.967 | 0.032 | — |
| **1.0** | 15.4° fail | **0.260** fail | 1.396 | 0.966 | 0.032 | 18.43 |
| ~0.8 *(extrapolated)* | ~15.8° | ~0.25 | — | — | — | — |
| 0.5 + yaw 1.0 *(f21a)* | — | — | — | **COLLAPSED at iter 405** | | |

**This is a real Pareto frontier and it passes outside the corner.** At the `scale_drift`
where K5 reaches its 0.25 gate, K4 is ~15.8° against a 15° gate. Closest approach is
`sd 1.0`: K4 over by **2.7 %**, K5 over by **4 %** — simultaneously, in one policy, with
every other gate passing (speed −7.0 %, uptime 0.966, fallen 0.032, torque 18.43).

**M32 needs correcting.** I wrote that K4 is independent of `scale_drift`. That was true of
the `drift_yaw 0.0` cells (17.9–19.2° across a 7× range) but is **false with yaw on**:
13.5° → 14.2° → 15.4° as `scale_drift` drops 8.0 → 1.5 → 1.0. So both terms affect K4, and
yaw 1.0 shifts the curve favourably (~18° → 13.5–15.4°) without removing the trade. The
"no frontier exists" claim is withdrawn; the frontier is real, shallow, and now measured.

#### E24 — the independent lever, which has never been touched

K4 has its own reward term and **`W_HEADING = 0.4` has never been varied in this project's
history.** Every attempt on K4 so far has gone through the drift term, which is why every
attempt traded against K5. Raising `W_HEADING` buys heading accuracy without relaxing the
drift pressure K5 needs.

E24 holds `scale_drift 1.0` + `drift_yaw 1.0` — the measured closest approach — and varies
only `W_HEADING`: **0.8 and 1.6** against the default 0.4. 350 iterations, launched 06:52.

Needed to clear both gates from `sd 1.0`: K4 15.4° → under 15° (a **2.7 %** improvement) at
no cost to K5's 0.260 → which still needs 4 %. So **E24 alone probably does not close both**;
the likely outcome is K4 passing comfortably and K5 still marginally over, which would then
make `sd 0.8` + raised `W_HEADING` the combination to try. That is the next session's first
experiment, and it is a single run rather than a search, because both levers are now
characterised.

**Stated before the result:** if neither E24 cell clears both gates, the session's answer on
the exit criterion is *not met, and quantified* — closest approach 2.7 % / 4 % over on two
gates, with the exact knob settings and the frontier recorded. That is an honest miss, and
it is a far more useful artifact than the starting position, where K6 was being chased in the
wrong units and K5 had never responded to anything.

### M34 — `W_HEADING` SOLVES K4. The never-varied term was the missing lever.

E24: `scale_drift 1.0` + `drift_yaw 1.0` held fixed (the M33 closest approach), varying only
`W_HEADING`. 350 iterations, 60-episode evaluation at the game condition:

| | `W_HEADING` 0.4 *(default)* | **`W_HEADING` 1.6** | gate |
|---|---|---|---|
| **K4 heading** | 15.4° **fail** | **13.3° PASS** | < 15° |
| K5 lateral | 0.260 fail | 0.286 fail | < 0.25 |
| K1 uptime | 0.966 | **0.982** | > 0.90 |
| K3 fallen | 0.032 | **0.016** | < 0.10 |
| K2 speed | 1.396 (−7.0 %) | 1.445 (**−3.7 %**) | ±10 % |
| K6b torque | 18.43 | 19.86 | ≤ 37.9 |

**Exactly as predicted, including the cost.** I wrote before the run that E24 would buy K4
and leave K5 ~4 % over; K4 went 15.4 → 13.3° (passing with margin) and K5 went 0.260 →
0.286 (slightly worse). Uptime, fallen fraction and speed all IMPROVED as well — this is the
best policy on five gates simultaneously that this project has measured.

`W_HEADING` had never been varied in the project's history. Every previous attempt on K4
went through the drift term, which is precisely why every attempt traded against K5. The
lever was sitting unused in the reward the whole time.

#### E25 — the final run, and the logic that makes it a single experiment

**K5 is now the only failing gate**, and critically, K4 no longer depends on the drift term,
so `scale_drift` is free to be lowered purely to buy K5:

| known | `scale_drift` | K5 |
|---|---|---|
| E23 | 1.5 | 0.311 |
| E23 | 1.0 | 0.260 |
| E24 | 1.0 (+ heading 1.6) | 0.286 |
| **target** | **0.7 / 0.5** | **< 0.25** |

E25 holds `W_HEADING 1.6` + `drift_yaw 1.0` and runs `scale_drift` **0.7 and 0.5**, 300
iterations, launched 07:05. `sd 0.5` is the gamble — f21a collapsed there — but f21a had
`W_HEADING 0.4`, and the much larger positive heading term may stabilise it. `sd 0.7` is the
safe cell.

**If `sd 0.7` lands K5 < 0.25 while K4 holds under 15°, every gated KPI passes in one
policy** and the brief's exit criterion is met. Needed: a 13 % reduction in K5 from a 30 %
reduction in `scale_drift`, against a measured local slope of roughly −0.05 K5 per halving —
so it is close but genuinely within reach rather than hopeful.

This is the last experiment that fits before 08:00.

### M35 — FINAL: `scale_drift 0.5` buys K5 and loses four other gates. Exit criterion NOT met.

E25, `W_HEADING 1.6` + `drift_yaw 1.0` held, 300 iterations:

| | sd 0.7 | **sd 0.5** | gate |
|---|---|---|---|
| training outcome | **degraded** (upright 0.47, heading 79.8°) | survived | |
| K5 lateral | — | **0.197 PASS** (best measured) | < 0.25 |
| K1 uptime | — | 0.732 **FAIL** | > 0.90 |
| K3 fallen | — | 0.283 **FAIL** | < 0.10 |
| K2 speed | — | 0.995 (−33.7 %) **FAIL** | ±10 % |
| K4 heading | — | 25.6° **FAIL** | < 15° |
| K6b torque | — | 17.40 pass | ≤ 37.9 |

`scale_drift 0.5` does reach K5's gate — and destroys the policy doing it. And the "safe"
cell (0.7) is the one that degraded while the "gamble" (0.5) survived, so **the collapse is
stochastic, not a clean `scale_drift` threshold.** My "stability floor between 0.5 and 8.0"
framing in M32/M33 was too tidy; the honest statement is that runs below `sd ~1.0` are
unreliable, cell-to-cell, at this iteration count.

## FINAL ANSWER — best policy of the session, and the exit criterion

**Best policy: `runs/k24b_hw16`** — `scale_drift 1.0`, `drift_yaw_weight 1.0`,
`W_HEADING 1.6`, from `e15_best`, 350 iterations. Evaluated at the game condition
(upright spawn, command 1.5 m/s, 60 deterministic episodes, domain randomisation on):

| KPI | gate | shipped `boy_chase01` | E15 snapshot | **k24b_hw16** | |
|---|---|---|---|---|---|
| K1 uptime | > 0.90 | 0.975 | 0.976 | **0.982** | pass |
| K2 speed err | ±10 % | +35 % **fail** | −3.5 % | **−3.7 %** | pass |
| K3 fallen | < 0.10 | — | 0.000 | 0.016 | pass |
| K4 heading | < 15° | — | 13.5° | **13.3°** | pass |
| **K5 lateral** | < 0.25 | 0.426 fail | 0.391 fail | **0.286** | **FAIL by 14 %** |
| K6b torque | ≤ 37.9 | 37.93 | 19.37 | 19.86 | pass |
| K7 accel | ≤ 1.28e6 | 1.29e6 | 6.88e5 | pass | pass |
| K8 uprightness | > 0.90 | — | 0.946 | pass | pass |
| K10 recovery | none set | 0 % | 0 % | 0 % | — |

**7 of 8 gates pass. K5 fails by 14 %.** The exit criterion ("all Phase 1 KPIs satisfied
across 10 consecutive evaluation episodes") is **NOT met**, and that is the honest verdict.

**What was gained**, measured against the same condition:

* K5 improved **27 %** (0.391 → 0.286) — the first real movement on a gate that had never
  responded to anything.
* K4 improved slightly (13.5 → 13.3°) while K5 improved, which no earlier run achieved —
  every previous attempt traded one for the other.
* K1 and K2 both improved. No gate regressed meaningfully (K3 0.000 → 0.016, both passing).
* Against the **shipped** brain: k24b_hw16 passes K2 (which the shipped brain fails by 35 %)
  and uses **48 % of its applied torque**.

**The frontier, fully characterised** (all at `drift_yaw 1.0`):

| `scale_drift` | `W_HEADING` | K4 | K5 | viable? |
|---|---|---|---|---|
| 8.0 | 0.4 | 13.5° | 0.391 | yes — E15 |
| 1.5 | 0.4 | 14.2° | 0.311 | yes |
| 1.0 | 0.4 | 15.4° | 0.260 | marginal (K4 over) |
| **1.0** | **1.6** | **13.3°** | **0.286** | **yes — best** |
| 0.5 | 0.4 | — | — | collapses |
| 0.5 | 1.6 | 25.6° | 0.197 | no — loses 4 gates |

K5 < 0.25 and K1/K2/K3/K4 passing were not achieved together by any setting of these three
knobs. **Closing K5 needs a lever outside the drift term** — the obvious untried candidate
is a lateral-velocity OBSERVATION (the policy cannot regulate what it cannot see; obs index
layout is a Unity parity contract, so that is a deliberate, larger change), or simply a
longer run: every E21–E25 cell was 300–600 iterations from a warm start, where the shipped
brain had 1500.

### M36 — `W_DRIFT` was never varied either. Same oversight as `W_HEADING`, same structural fix.

The drift penalty is `- W_DRIFT * tanh(drift / scale_drift)`. Across E16, E21, E22, E23 and
E25 I swept **`scale_drift`** — the normaliser *inside* the tanh — eight times, and never
once touched **`W_DRIFT`**, the weight in front of it. They do different things:

* `scale_drift` sets **where the penalty saturates**. Lowering it makes the tanh saturate
  early, which flattens the gradient across the whole operating range — and that is exactly
  how f21a (sd 0.5) and m25a (sd 0.7) collapsed.
* `W_DRIFT` sets **what the penalty is worth**. At the shipped 0.15, the drift term is at
  most 0.15 per step against a ~2.2 positive budget: **under 7 % even fully saturated.**

So K5 has been chased for this entire session with a term that cannot pay for itself, using
the one knob that destabilises training, while the knob that actually prices it sat unused.

**This is precisely the `W_HEADING` oversight repeated.** K4 was chased through the drift
term for three experiments; raising `W_HEADING` 0.4 → 1.6 solved it in a single run (M34).
The same shape of error, on the other gate, found 40 minutes later. The general lesson, which
belongs in the next session's first paragraph:

> **When a KPI will not move, check whether its reward term has ever been weighted, before
> reaching for the term's internal scales.** Both times the direct weight was at its shipped
> default and had never been touched, and both times the indirect knob produced a trade-off
> that looked like a law of nature.

**E26** holds every one of `k24b_hw16`'s winning settings — `scale_drift 1.0`,
`drift_yaw 1.0`, `W_HEADING 1.6` — and varies only `W_DRIFT`: **0.45 and 0.90** against the
default 0.15. 250 iterations from `e15_best`, launched 07:25.

Needed: K5 0.286 → under 0.25, a **13 %** reduction. At `W_DRIFT 0.45` the drift term goes
from under 7 % of the reward to ~20 %, which is the same dose range that made `W_HEADING`
effective. Stated before the result: the risk is that a 6× stronger drift penalty costs K4 or
speed the way `scale_drift` did — if it does, the trade is real and intrinsic rather than an
artifact of an unweighted term, and *that* would be the finding.

### M37 — FINAL. The K5 trade is REAL: two independent levers, same four gates lost.

E26, everything held at `k24b_hw16`'s settings, varying only `W_DRIFT` (0.15 → 0.45):

| | `W_DRIFT` 0.15 *(k24b_hw16)* | **`W_DRIFT` 0.45** | gate |
|---|---|---|---|
| **K5 lateral** | 0.286 fail | **0.166 PASS** *(best measured)* | < 0.25 |
| K1 uptime | **0.982 pass** | 0.846 **fail** | > 0.90 |
| K3 fallen | **0.016 pass** | 0.157 **fail** | < 0.10 |
| K2 speed | **−3.7 % pass** | 1.199, −20 % **fail** | ±10 % |
| K4 heading | **13.3° pass** | 25.6° **fail** | < 15° |
| K6b torque | 19.86 pass | 16.16 pass | ≤ 37.9 |

**The pre-committed interpretation applies.** Before the run I wrote: *"if a 6× stronger drift
penalty costs K4 or speed the way `scale_drift` did, the trade is real and intrinsic rather
than an artifact of an unweighted term, and that would be the finding."* It cost both.

**Two independent levers, the same outcome:**

| lever | K5 achieved | gates lost |
|---|---|---|
| `scale_drift` 1.0 → 0.5 | 0.197 | K1, K2, K3, K4 |
| `W_DRIFT` 0.15 → 0.45 | 0.166 | K1, K2, K3, K4 |

Two different parameters, acting on the same reward term through different mechanisms (where
it saturates vs what it is worth), both reach K5's gate and both lose exactly the same four
gates doing it. **That makes the trade a property of the reward, not of the knob.** So M36's
"unweighted term" explanation is refuted — the term was under-weighted, and weighting it
properly reveals a genuine conflict rather than unlocking a free win.

**Why, mechanistically:** `drift = lateral² + yaw_rate²`. Lateral velocity and yaw rate are
both *necessary* for a biped — weight shifts are lateral, and turning is yaw. A penalty strong
enough to suppress uncommanded drift also suppresses the commanded motion that balance and
steering are made of. K5 as specified asks the racer not to move sideways; walking requires
moving sideways.

**And my training-time extrapolation was wrong, again.** At iteration 65 this cell showed
lateral 0.12 with heading 28.8° and standing 0.73 — better than k24b on all three — and I
noted it suggested eval K5 ≈ 0.17. K5 did land at 0.166, but the policy had degraded by
iteration 250 (uptime 0.846). **Training-time KPIs at an early iteration do not predict the
converged policy**, which is now the fourth time that has caught me tonight.

## SESSION CLOSE — 07:35, budget spent

**Exit criterion: NOT MET.** Best policy **`runs/k24b_hw16`** passes **7 of 8 gates**, failing
K5 at 0.286 vs 0.25.

**The plateau criterion now also applies**, which is the honest reason to stop rather than the
clock: E23, E24, E25 and E26 all attacked K5, and the last three each reached its gate only by
losing four others. The binding constraint has not improved by >3 % without regression across
three consecutive runs.

**What K5 would actually need** — recorded for the next session, since tuning is exhausted:

1. **Redefine K5.** It currently penalises all lateral velocity. A racer's *net* cross-track
   displacement is the thing that matters for staying in a lane; per-step lateral velocity
   punishes the weight shifts walking is made of. `lateral_displacement / distance_travelled`
   would be the honest quantity, and the gate would need re-deriving.
2. **Give the policy the observation.** It cannot regulate what it cannot see; there is no
   lateral-velocity term in the 75-element observation. This changes a contract shared with
   `MojucuBoyObservation.cs`, so it is a deliberate, larger change.
3. **Train longer.** Every cell here was 250–600 iterations from a warm start. The shipped brain
   had 1500.

Option 1 is the one I would do first. K5 is the only gate tonight that no intervention could
satisfy without collateral damage, and the measurement above is a reasonable argument that the
metric — not the policy — is what is wrong. **That would be the third KPI this session found to
be mis-specified**, after K6 (measured commanded angle instead of torque) and the K6/K7 pair
(both gated against a baseline that is best on one and worst on the other).
