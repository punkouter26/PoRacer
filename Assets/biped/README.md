# Biped walk-to-target

A hand-built two-legged robot that learns to walk to a target, in the same shape as the
`spider/` and `worm/` tasks in this repo. Everything here is primitives only — the rig is
meant to have a skinned mesh bound to it later.

## The rig

`make_urdf.py` generates `assets/biped.urdf`; `convert_biped.py` turns it into
`assets/biped_usd/biped/biped.usda`.

- **11 links, 10 actuated joints, 14.8 kg, 0.66 m hip height** when standing.
- Base link is `torso`. Its origin sits **on the hip line**, and the box geometry extends
  upward from there, so the root position *is* the hip height.

```
torso
└── {L,R}_hip_yaw_link      revolute about Z   [-0.6,  0.6]   50 N*m
    └── {L,R}_hip_roll_link revolute about X   [-0.5,  0.5]  100 N*m
        └── {L,R}_thigh     revolute about Y   [-1.5,  1.0]  150 N*m   (hip_pitch)
            └── {L,R}_shank revolute about Y   [ 0.0,  2.2]  150 N*m   (knee)
                └── {L,R}_foot revolute about Y [-0.8, 0.8]   80 N*m   (ankle)
```

The two hub links are zero-collision stubs that exist only to carry the yaw and roll DOFs,
the way a real hip does. Limb segments are cylinders (0.32 m thigh, 0.32 m shank); feet are
0.20 x 0.09 x 0.04 m boxes offset 0.04 m forward of the ankle, so there is heel behind and
toe in front.

**Sign convention.** All pitch joints rotate about +Y, so a *positive* angle swings the
segment backward (-X) and forward stepping is negative hip_pitch. The knee only bends one
way, like a human knee. Nominal standing crouch is
`hip_pitch = -0.25, knee = +0.50, ankle = -0.25`; those sum to zero, which is what keeps the
sole flat on the ground.

### Notes for skinning

- The USD nests the links as a transform hierarchy under `Geometry/`:
  `Robot/Geometry/torso/L_hip_yaw_link/L_hip_roll_link/L_thigh/L_shank/L_foot`. Joints are
  separate prims under `Robot/Physics/`.
- Because the bodies are nested, Isaac Lab's `activate_contact_sensors` helper only reaches
  `torso` (it stops descending at the first rigid body). That is why this task tracks foot
  contact from foot height instead of using a `ContactSensor`. If you later flatten the
  hierarchy for skinning, a real contact sensor becomes available.
- Every link frame is at its parent joint with geometry hanging along -Z, which is the usual
  convention for binding a skinned mesh.

## The task — `Isaac-Biped-Direct-v0`

A green marker is dropped 2.0–4.5 m away at a random bearing; the robot has to walk to it.
Reaching it (within 0.4 m) scores a bonus and immediately spawns a new target, so one
episode chains many targets. The robot also starts each episode facing a random direction,
so it has to learn to turn, not just to walk straight.

- **Observation (42):** 10 joint pos (relative to the standing pose) + 10 joint vel +
  3 linear vel + 3 angular vel + 3 projected gravity + 2 unit heading to target +
  1 scaled distance + 10 previous actions.
- **Action (10):** joint position offsets around the standing pose, scaled by 0.5 rad.
- **Timing:** physics 200 Hz, policy 50 Hz, 20 s episodes.

Rewards are progress toward the target (15/m) and a heading term, plus the terms that keep a
biped honest: an alive bonus, an upright term, a penalty for dropping below hip height, and a
foot air-time term that pays for real swing phases so it steps instead of shuffling. Falling
(torso below 0.40 m or leaning past ~66 deg) ends the episode and costs -20.

## Running it

```bat
biped_train.bat                       :: headless PPO, 4096 envs
biped_train.bat --max_iterations 500  :: shorter run
biped_play.bat                        :: watch the latest checkpoint in the GUI
```

```bat
run_example.bat biped\sanity_check.py --headless        :: does the rig stand up?
run_example.bat biped\sanity_check.py --viz kit --num_envs 4 --steps 200000   :: look at it
run_example.bat biped\eval_policy.py --headless --num_envs 256    :: targets per minute
```

Checkpoints land in `logs/rsl_rl/biped_direct/`.
