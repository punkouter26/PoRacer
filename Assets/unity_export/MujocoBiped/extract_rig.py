"""
biped_sentis  ->  MujocoBiped_rig.json

Everything the Unity side needs, in ONE file, still in MuJoCo coordinates
(right-handed, Z-up, X-forward, metres, radians). The frame map is applied
exactly once, in C# (MujocoBipedFrameMap), so a single test proves it.

Two things this script does that a naive port would not:

1. It DECOMPOSES MuJoCo's multi-joint bodies into a chain of single-DOF Unity
   links. MuJoCo composes a body's joints sequentially (R = R_j1 R_j2 R_j3, each
   axis in the frame left by the previous joint); PhysX's spherical articulation
   joint is a single 3-DOF quaternion joint and does NOT reproduce that, nor does
   its jointPosition map back to MuJoCo's qpos. A chain of revolute links does,
   exactly. The extra links are near-massless placeholders ("dummy") - RIG_AUDIT
   section A quantifies what they cost.

2. It solves for the ARMATURE FOLD COEFFICIENTS exactly. MuJoCo's armature adds
   to the joint-space mass-matrix diagonal, H[i][i] += A. Unity has no such
   field, so it can only be bought with link inertia - but link inertia is
   spatial and accumulates up the tree, so naively adding A*a*a^T to every
   jointed link over-counts every parallel-axis run. Placing c_k*a_k*a_k^T on
   link k contributes c_k*(a_i.a_k)^2 to H[i][i] for every ancestor i, which at
   the zero pose is a triangular system in c. Solved leaf-upward it is exact.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "biped_sentis"))
OUT = os.path.join(HERE, "MujocoBiped_rig.json")

spec = json.load(open(os.path.join(SRC, "robot_spec.json")))
BODY = {b["name"]: b for b in spec["bodies"]}
JOINT = {j["name"]: j for j in spec["joints"]}
ACT = {a["name"]: a for a in spec["actuators"]}
GEOM = {g["name"]: g for g in spec["geoms"]}

JOINT_ORDER = ["hip_z_l", "hip_x_l", "hip_y_l", "knee_l", "ankle_y_l", "ankle_x_l",
               "hip_z_r", "hip_x_r", "hip_y_r", "knee_r", "ankle_y_r", "ankle_x_r"]

# ---------------------------------------------------------------- MJCF geoms --
# Read straight off model/biped.xml. MuJoCo box "size" is HALF-extents; a capsule
# is a cylinder of length |fromto| capped by two hemispheres of the given radius.
CAP, BOX, SPH = "capsule", "box", "sphere"
GEOMS = {
    "torso": [
        dict(name="pelvis", kind=CAP, a=(0, -0.09, 0), b=(0, 0.09, 0), r=0.085),
        dict(name="chest", kind=CAP, a=(0, 0, 0.06), b=(0, 0, 0.36), r=0.09),
        dict(name="head", kind=SPH, pos=(0, 0, 0.50), r=0.09),
    ],
    "thigh_l": [dict(name="thigh_l", kind=CAP, a=(0, 0, 0), b=(0, 0.01, -0.38), r=0.06)],
    "shin_l": [dict(name="shin_l", kind=CAP, a=(0, 0, 0), b=(0, 0, -0.38), r=0.049)],
    "foot_l": [dict(name="foot_l", kind=BOX, pos=(0.045, 0, -0.035), half=(0.115, 0.05, 0.03))],
    "thigh_r": [dict(name="thigh_r", kind=CAP, a=(0, 0, 0), b=(0, -0.01, -0.38), r=0.06)],
    "shin_r": [dict(name="shin_r", kind=CAP, a=(0, 0, 0), b=(0, 0, -0.38), r=0.049)],
    "foot_r": [dict(name="foot_r", kind=BOX, pos=(0.045, 0, -0.035), half=(0.115, 0.05, 0.03))],
}

# ------------------------------------------------------- MuJoCo body topology --
# body -> (parent, joints in MuJoCo's own order). That order is what makes the
# decomposition below equal MuJoCo's sequential composition.
MJ_TREE = [
    ("torso", None, []),
    ("thigh_l", "torso", ["hip_z_l", "hip_x_l", "hip_y_l"]),
    ("shin_l", "thigh_l", ["knee_l"]),
    ("foot_l", "shin_l", ["ankle_y_l", "ankle_x_l"]),
    ("thigh_r", "torso", ["hip_z_r", "hip_x_r", "hip_y_r"]),
    ("shin_r", "thigh_r", ["knee_r"]),
    ("foot_r", "shin_r", ["ankle_y_r", "ankle_x_r"]),
]


def dummy_name(joint):
    """A dummy link is named for the joint it carries, so the hierarchy reads as
    the kinematic chain and no name can collide with a real MuJoCo body."""
    return "j_" + joint


def build_links():
    """MuJoCo bodies -> one Unity link per DOF, breadth-first from the root."""
    links = []
    unity_leaf_of_body = {}

    for body, parent, joints in MJ_TREE:
        b = BODY[body]
        if parent is None:              # torso: the free joint IS the articulation root
            # localPos is ZERO, not the MJCF <body pos="0 0 0.88">. For a free joint
            # MuJoCo folds the body's own pos into qpos[0:3] - init_qpos[0:3] is that
            # same 0.88 - so carrying it here as well would place the torso 0.88 m above
            # wherever the prefab root is put, i.e. double the spawn height. The spawn
            # position is the single owner of that number; it lives in rig["spawn"].
            links.append(dict(
                name=body, parent="", isRoot=True, isDummy=False, mjBody=body,
                localPosMuj=[0.0, 0.0, 0.0], localRotMujWxyz=b["local_quat_wxyz"],
                mass=b["mass"], comMuj=b["com_local"], inertiaDiagMuj=b["inertia_diag"],
                joint=None, geoms=GEOMS[body]))
            unity_leaf_of_body[body] = body
            continue

        # The body's own offset goes on the FIRST link of its chain; the rest sit
        # at zero offset so every joint of that body shares one anchor point,
        # exactly where MuJoCo puts them.
        cur = unity_leaf_of_body[parent]
        for k, jn in enumerate(joints):
            last = (k == len(joints) - 1)
            j = JOINT[jn]
            a = ACT[jn]
            links.append(dict(
                name=body if last else dummy_name(jn),
                parent=cur, isRoot=False, isDummy=not last, mjBody=body,
                localPosMuj=b["local_pos"] if k == 0 else [0.0, 0.0, 0.0],
                localRotMujWxyz=[1.0, 0.0, 0.0, 0.0],
                mass=b["mass"] if last else 0.0,
                comMuj=b["com_local"] if last else [0.0, 0.0, 0.0],
                inertiaDiagMuj=b["inertia_diag"] if last else [0.0, 0.0, 0.0],
                geoms=GEOMS[body] if last else [],
                joint=dict(
                    name=jn, index=JOINT_ORDER.index(jn), axisInChildMuj=j["axis"],
                    lowerRad=j["range_rad"][0], upperRad=j["range_rad"][1],
                    damping=j["damping"], armature=j["armature"],
                    gear=a["gear"], ctrlLower=a["ctrlrange"][0], ctrlUpper=a["ctrlrange"][1],
                    peakTorqueNm=a["peak_torque_Nm"]),
            ))
            cur = links[-1]["name"]
        unity_leaf_of_body[body] = cur
    return links


def solve_armature(links):
    """
    Exact fold coefficients at the zero pose.

      H[i][i] gains  sum over k in subtree(i) of  c_k * (a_i . a_k)^2

    Requiring that to equal armature_i for every i gives a triangular system in c
    (a link's coefficient appears only in its own row and its ancestors'), so one
    leaf-upward pass solves it with no iteration and no residual.
    """
    by_name = {l["name"]: l for l in links}
    children = {l["name"]: [] for l in links}
    for l in links:
        if l["parent"]:
            children[l["parent"]].append(l["name"])

    def subtree(n):
        out = [n]
        for c in children[n]:
            out += subtree(c)
        return out

    depth = {}

    def d(n):
        if n not in depth:
            depth[n] = 0 if not by_name[n]["parent"] else d(by_name[n]["parent"]) + 1
        return depth[n]

    order = sorted([l["name"] for l in links if l["joint"]], key=d, reverse=True)

    c = {n: 0.0 for n in by_name}
    for n in order:
        li = by_name[n]
        ai = li["joint"]["axisInChildMuj"]
        acc = 0.0
        for k in subtree(n):
            if k == n or c[k] == 0.0:
                continue
            ak = by_name[k]["joint"]["axisInChildMuj"]
            dot = sum(x * y for x, y in zip(ai, ak))
            acc += c[k] * dot * dot
        c[n] = li["joint"]["armature"] - acc

    for l in links:
        l["armatureFoldExact"] = c[l["name"]]
        l["armatureFoldNaive"] = l["joint"]["armature"] if l["joint"] else 0.0
    return c


def main():
    links = build_links()
    solve_armature(links)

    sim = spec["simulation"]
    rig = dict(
        source="biped_sentis (MuJoCo 3.12.0, PPO 14,499,768 steps)",
        note="All vectors are MuJoCo: right-handed, Z-up, X-forward, metres, radians. "
             "Quaternions are (w, x, y, z) - verified against the recorded observations.",
        jointOrder=JOINT_ORDER,
        obsDim=49, actDim=12,
        timing=dict(policyDt=sim["control_timestep"], mujocoPhysicsDt=sim["physics_timestep"],
                    mujocoFrameSkip=sim["frame_skip"], controlHz=sim["control_frequency_hz"]),
        spawn=dict(posMuj=spec["reset_state"]["init_qpos"][0:3],
                   quatMujWxyz=spec["reset_state"]["init_qpos"][3:7]),
        observation=dict(
            clipLinVel=10.0, clipAngVel=10.0, clipJointVel=20.0, maxTargetDistance=10.0,
            angularVelocityIsDoubleRotated=True,
            angularVelocityNote=(
                "MuJoCo's free joint stores qvel[3:6] in the BODY-LOCAL frame (proven in "
                "RIG_AUDIT section D) and env.py's _get_obs applies rot.T to it anyway, so "
                "obs[7:10] is R^T R^T w_world. Reproduce it, do not fix it - the policy was "
                "trained on this."),
            linearVelocityReference="bodyFrameOrigin"),
        task=dict(
            reachRadiusM=spec["task"]["reach_radius_m"],
            targetDistanceRangeM=spec["task"]["target_distance_range_m"],
            targetAngleRangeRad=spec["task"]["target_angle_range_rad"],
            targetHeightMuj=BODY["target"]["local_pos"][2],
            healthyZRange=spec["task"]["healthy_z_range"],
            minUprightness=spec["task"]["min_uprightness"],
            maxEpisodeSteps=spec["task"]["max_episode_steps"]),
        physics=dict(
            gravityMuj=sim["gravity"],
            floorFriction=GEOM["floor"]["friction"][0],
            footFriction=GEOM["foot_l"]["friction"][0],
            bodyFriction=GEOM["thigh_l"]["friction"][0],
            # MuJoCo takes the ELEMENTWISE MAXIMUM of two geoms' friction, so the
            # foot/floor pair ran at max(1.2, 1.0) = 1.2. See CONTRACT.md.
            effectiveFootGroundFriction=max(GEOM["foot_l"]["friction"][0],
                                            GEOM["floor"]["friction"][0]),
            mujocoFrictionCombine="maximum",
            mujocoSolver=sim["solver"], mujocoIntegrator=sim["integrator_name"],
            mujocoIterations=sim["iterations"],
            # No MuJoCo joint carries a velocity limit; the caps below are Unity-side
            # safety valves set well above the recorded p100 of 37.14 rad/s.
            maxJointVelocity=200.0, maxAngularVelocity=200.0, maxLinearVelocity=200.0,
            maxDepenetrationVelocity=1.0,
            linearDamping=0.0, angularDamping=0.0, jointFriction=0.0,
            solverPositionIterations=12, solverVelocityIterations=4,
            contactOffset=0.01, restOffset=0.0,
            # MuJoCo excludes only DIRECT parent-child geom pairs; everything else
            # (leg vs leg, thigh vs its own foot) collided during training.
            selfCollisionExcludesParentChildOnly=True,
            dummyLinkMass=0.01, inertiaFloor=1e-4),
        eval=dict(
            targetsReachedPerEpisode=4.0, episodeLengthSteps=516.0,
            meanClosingSpeed=1.15, survivedFullEpisodeFraction=0.30),
        links=links)

    json.dump(rig, open(OUT, "w"), indent=1)
    real = [l for l in links if not l["isDummy"]]
    dummies = [l for l in links if l["isDummy"]]
    print("wrote " + OUT)
    print("  %d Unity links = %d real + %d dummy, %d revolute DOF"
          % (len(links), len(real), len(dummies), len(JOINT_ORDER)))
    print("  total real mass %.4f kg (+ %.3f kg of dummies)"
          % (sum(l["mass"] for l in real), len(dummies) * rig["physics"]["dummyLinkMass"]))
    print("  armature fold coefficients (kg.m^2):")
    for l in links:
        if l["joint"]:
            print("    %-10s naive %.3f  exact %.3f"
                  % (l["joint"]["name"], l["armatureFoldNaive"], l["armatureFoldExact"]))


if __name__ == "__main__":
    main()
