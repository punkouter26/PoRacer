"""
reference_trajectory.json  ->  mujoco_reference.json  (verified, self-describing)

The raw export is already ground truth; this script re-derives every observation
slice from the recorded simulator state and refuses to write the file unless each
one reproduces to 1e-6. What comes out therefore carries proof, not just data:

  * the quaternion order is asserted, not assumed - the same 150 steps are scored
    under (w,x,y,z) and (x,y,z,w) and the loser's residual is recorded so the
    margin is visible (RIG_AUDIT section D);
  * obs[7:10] is shown to be R^T applied to an ALREADY-BODY-LOCAL angular
    velocity, the double rotation Unity has to reproduce;
  * joint velocities are carried both raw and clipped, because obs[22:34] is the
    clipped copy and only the raw one can be compared against a Unity rig;
  * the observation slice map ships alongside the data, so the C# agent and the
    tests read their layout from the same place.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "biped_sentis", "reference_trajectory.json"))
OUT = os.path.join(HERE, "mujoco_reference.json")
TOL = 1e-6

CLIP_LIN, CLIP_ANG, CLIP_JOINTVEL, MAX_DIST = 10.0, 10.0, 20.0, 10.0

SLICES = [
    dict(name="torso_height", start=0, size=1, source="qpos[2]",
         note="Height of the TORSO BODY-FRAME ORIGIN, not its centre of mass."),
    dict(name="projected_gravity", start=1, size=3, source="R^T @ (0,0,-1)"),
    dict(name="linear_velocity", start=4, size=3, source="clip(R^T @ qvel[0:3], +/-10)",
         note="qvel[0:3] is the WORLD-frame velocity of the body-frame origin."),
    dict(name="angular_velocity", start=7, size=3, source="clip(R^T @ qvel[3:6], +/-10)",
         note="qvel[3:6] is ALREADY body-local for a MuJoCo free joint, so this term "
              "is rotated into the torso frame TWICE. Reproduce it, do not fix it."),
    dict(name="joint_positions", start=10, size=12, source="qpos[7:]"),
    dict(name="joint_velocities", start=22, size=12, source="clip(qvel[6:], +/-20)"),
    dict(name="target_direction", start=34, size=2,
         source="unit(target_xy - torso_xy) rotated by -yaw"),
    dict(name="target_distance", start=36, size=1, source="min(|target_xy - torso_xy|, 10)"),
    dict(name="last_action", start=37, size=12, source="previous step's network output"),
]


def rot(q, order):
    w, x, y, z = q if order == "wxyz" else (q[3], q[0], q[1], q[2])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def score(steps, order):
    """Max residual over every rotation-dependent slice under one quaternion order."""
    worst = 0.0
    for s in steps:
        o = np.array(s["observation"])
        R = rot(s["root_pose"]["quaternion_wxyz"], order)
        worst = max(worst, np.abs(o[1:4] - R.T @ np.array([0.0, 0.0, -1.0])).max())
        worst = max(worst, np.abs(o[4:7] - np.clip(R.T @ np.array(s["root_velocity"]["linear"]),
                                                   -CLIP_LIN, CLIP_LIN)).max())
        worst = max(worst, np.abs(o[7:10] - np.clip(R.T @ np.array(s["root_velocity"]["angular"]),
                                                    -CLIP_ANG, CLIP_ANG)).max())
    return float(worst)


def omega_frame_evidence(steps, dt):
    """
    Is qvel[3:6] world-frame or body-local? Compare the AXIS of the frame-to-frame
    relative rotation against the recorded omega. Local composition is q0* (x) q1,
    world composition is q1 (x) q0*; only the matching one shares omega's axis.
    Axis angle, not magnitude, so a coarse 40 Hz sample cannot muddy the verdict.
    """
    def qm(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    loc, wld = [], []
    for i in range(len(steps) - 1):
        q0 = np.array(steps[i]["root_pose"]["quaternion_wxyz"])
        q1 = np.array(steps[i + 1]["root_pose"]["quaternion_wxyz"])
        if np.dot(q0, q1) < 0:
            q1 = -q1
        w = np.array(steps[i]["root_velocity"]["angular"])
        n = np.linalg.norm(w)
        if n < 0.5:
            continue
        qc = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        for acc, dq in ((loc, qm(qc, q1)[1:]), (wld, qm(q1, qc)[1:])):
            m = np.linalg.norm(dq)
            if m > 1e-9:
                acc.append(float(np.degrees(np.arccos(np.clip(np.dot(dq / m, w / n), -1, 1)))))
    return dict(samples=len(loc), dt=dt,
                medianAxisErrorIfBodyLocalDeg=float(np.median(loc)),
                medianAxisErrorIfWorldDeg=float(np.median(wld)),
                verdict="bodyLocal" if np.median(loc) < np.median(wld) else "world")


def main():
    raw = json.load(open(SRC))
    steps = raw["trajectory"]
    dt = raw["conventions"]["control_dt"]

    res = {o: score(steps, o) for o in ("wxyz", "xyzw")}
    if res["wxyz"] > TOL:
        sys.exit("quaternion order (w,x,y,z) does not reproduce the observations "
                 "(residual %.3e) - refusing to write a 'verified' file." % res["wxyz"])

    # Every remaining slice, re-derived from the recorded state.
    checks = {}
    prev = None
    acc = {k: 0.0 for k in ("torso_height", "joint_positions", "joint_velocities",
                            "target_direction", "target_distance", "last_action")}
    for s in steps:
        o = np.array(s["observation"])
        R = rot(s["root_pose"]["quaternion_wxyz"], "wxyz")
        acc["torso_height"] = max(acc["torso_height"], abs(o[0] - s["root_pose"]["position"][2]))
        acc["joint_positions"] = max(acc["joint_positions"],
                                     np.abs(o[10:22] - np.array(s["joint_positions"])).max())
        acc["joint_velocities"] = max(acc["joint_velocities"],
                                      np.abs(o[22:34] - np.clip(np.array(s["joint_velocities"]),
                                                                -CLIP_JOINTVEL,
                                                                CLIP_JOINTVEL)).max())
        v = np.array(s["target"]["position"])[:2] - np.array(s["root_pose"]["position"])[:2]
        dist = float(np.linalg.norm(v))
        wd = v / max(dist, 1e-6)
        h = np.arctan2(R[1, 0], R[0, 0])
        c, sn = np.cos(-h), np.sin(-h)
        ld = np.array([c * wd[0] - sn * wd[1], sn * wd[0] + c * wd[1]])
        acc["target_direction"] = max(acc["target_direction"], np.abs(o[34:36] - ld).max())
        acc["target_distance"] = max(acc["target_distance"], abs(o[36] - min(dist, MAX_DIST)))
        expect_last = np.array(prev["action"]) if prev is not None else np.zeros(12)
        acc["last_action"] = max(acc["last_action"], np.abs(o[37:49] - expect_last).max())
        prev = s
    checks.update({k: float(v) for k, v in acc.items()})
    checks["projected_gravity/linear_velocity/angular_velocity"] = res["wxyz"]

    bad = {k: v for k, v in checks.items() if v > TOL}
    if bad:
        sys.exit("slices failed to reproduce within %.0e: %s" % (TOL, bad))

    qv = np.array([s["joint_velocities"] for s in steps])
    out = dict(
        description="Verified copy of biped_sentis/reference_trajectory.json. Every "
                    "observation slice below was re-derived from the recorded simulator "
                    "state and reproduced to better than %.0e before this file was written."
                    % TOL,
        source="biped_sentis/reference_trajectory.json",
        conventions=dict(
            frame="MuJoCo: right-handed, Z-up, X-forward, metres, radians",
            quaternionOrder="wxyz",
            quaternionIndices=dict(w=0, x=1, y=2, z=3),
            quaternionOrderResidual=res,
            quaternionOrderNote="Scored both ways over all %d steps; (w,x,y,z) reproduces "
                                "the observations, (x,y,z,w) is off by %.3f."
                                % (len(steps), res["xyzw"]),
            angularVelocityFrame="bodyLocal",
            angularVelocityEvidence=omega_frame_evidence(steps, dt),
            linearVelocityFrame="world",
            linearVelocityReference="bodyFrameOrigin",
            controlDt=dt,
            jointOrder=raw["conventions"]["joint_order"],
            clip=dict(linearVelocity=CLIP_LIN, angularVelocity=CLIP_ANG,
                      jointVelocity=CLIP_JOINTVEL, targetDistance=MAX_DIST)),
        observationLayout=SLICES,
        verification=checks,
        jointVelocityStats=dict(
            note="obs[22:34] is the CLIPPED copy; jointVelocitiesRaw below is not clipped.",
            maxAbs=[float(x) for x in np.abs(qv).max(0)],
            p99Abs=[float(x) for x in np.percentile(np.abs(qv), 99, axis=0)],
            clipThreshold=CLIP_JOINTVEL,
            clippedSampleCount=int((np.abs(qv) > CLIP_JOINTVEL).sum()),
            totalSamples=int(qv.size)),
        steps=len(steps),
        trajectory=[dict(
            step=s["step"], time=s["time"],
            observation=s["observation"], action=s["action"],
            rootPosMuj=s["root_pose"]["position"],
            rootQuatMujWxyz=s["root_pose"]["quaternion_wxyz"],
            rootLinVelWorldMuj=s["root_velocity"]["linear"],
            rootAngVelBodyLocalMuj=s["root_velocity"]["angular"],
            jointPositionsRad=s["joint_positions"],
            jointVelocitiesRaw=s["joint_velocities"],
            targetPosMuj=s["target"]["position"],
            reward=s["reward"], terminated=s["terminated"],
            targetsReached=s["targets_reached"]) for s in steps])

    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote " + OUT)
    print("  %d steps, every slice verified to better than %.0e" % (len(steps), TOL))
    print("  quaternion order wxyz residual %.3e  (xyzw would be %.3f)"
          % (res["wxyz"], res["xyzw"]))
    ev = out["conventions"]["angularVelocityEvidence"]
    print("  angular velocity frame: %s  (axis error %.1f deg local vs %.1f deg world, n=%d)"
          % (ev["verdict"], ev["medianAxisErrorIfBodyLocalDeg"],
             ev["medianAxisErrorIfWorldDeg"], ev["samples"]))
    print("  joint velocity p100 %.2f rad/s, p99 %.2f rad/s, %d of %d samples clipped at %.0f"
          % (max(out["jointVelocityStats"]["maxAbs"]),
             max(out["jointVelocityStats"]["p99Abs"]),
             out["jointVelocityStats"]["clippedSampleCount"],
             out["jointVelocityStats"]["totalSamples"], CLIP_JOINTVEL))


if __name__ == "__main__":
    main()
