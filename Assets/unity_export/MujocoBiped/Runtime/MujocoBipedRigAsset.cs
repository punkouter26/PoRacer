using System;
using UnityEngine;

namespace MujocoBiped
{
    /// <summary>
    /// One joint of the rig. Index is MuJoCo's actuator index, which is also the
    /// observation and action index - the export uses one order throughout.
    /// </summary>
    [Serializable]
    public class MujocoBipedJointDef
    {
        public string name;
        public int index;

        /// <summary>Unit rotation axis in the CHILD link frame, MuJoCo convention.</summary>
        public Vector3 axisInChildMuj;

        public float lowerRad;
        public float upperRad;

        /// <summary>MJCF joint damping, 1.0 N.m.s/rad. Passive, always on, never clipped
        /// by the actuator limit - it is not part of the actuator.</summary>
        public float damping;

        /// <summary>MJCF joint armature, 0.02 kg.m^2 of rotor inertia. See
        /// <see cref="armatureFoldExact"/>.</summary>
        public float armature;

        /// <summary>Actuator gear. Torque is action * gear; ctrlrange is [-1, 1], so this
        /// is also the peak torque in N.m.</summary>
        public float gear;

        public float ctrlLower;
        public float ctrlUpper;
    }

    /// <summary>One MJCF primitive geom, still in MuJoCo coordinates.</summary>
    [Serializable]
    public class MujocoBipedGeomDef
    {
        /// <summary>capsule | box | sphere.</summary>
        public string kind;

        /// <summary>Capsule endpoints (MJCF <c>fromto</c>).</summary>
        public Vector3 a;
        public Vector3 b;

        /// <summary>Capsule/sphere radius.</summary>
        public float radius;

        /// <summary>Box/sphere centre in the body frame.</summary>
        public Vector3 pos;

        /// <summary>Box HALF-extents, which is what MJCF <c>size</c> means for a box.</summary>
        public Vector3 half;

        /// <summary>Sliding friction. 1.2 on the feet, 0.9 everywhere else.</summary>
        public float friction;

        public string name;
    }

    /// <summary>
    /// One Unity ArticulationBody. MuJoCo bodies with several hinges become a chain of
    /// these - see <see cref="isDummy"/>.
    /// </summary>
    [Serializable]
    public class MujocoBipedLinkDef
    {
        public string name;
        public string parent;
        public bool isRoot;

        /// <summary>
        /// True for a link that exists only to carry one extra DOF of a multi-hinge
        /// MuJoCo body. It has no geometry, a token mass and an explicit inertia floor.
        /// PhysX's spherical articulation joint cannot stand in for MuJoCo's sequential
        /// hinge composition, and its jointPosition would not map back to qpos.
        /// </summary>
        public bool isDummy;

        /// <summary>The MuJoCo body this link belongs to; several links can share one.</summary>
        public string mjBody;

        public float mass;
        public Vector3 comMuj;
        public Vector3 inertiaDiagMuj;

        /// <summary>Pose relative to the parent link at the ZERO pose, MuJoCo frame.</summary>
        public Vector3 localPosMuj;

        /// <summary>Parent-relative rotation as (w, x, y, z), MuJoCo frame.</summary>
        public Vector4 localRotMujWxyz;

        public bool hasJoint;
        public MujocoBipedJointDef joint;

        public MujocoBipedGeomDef[] geoms = Array.Empty<MujocoBipedGeomDef>();

        /// <summary>
        /// Armature to add to THIS link's inertia about its own joint axis, solved so
        /// that every joint's H[i][i] gains exactly its MJCF armature and no more.
        /// Zero for most joints: a run of parallel axes (hip_y -> knee -> ankle_y) is
        /// satisfied entirely by the distal member, because spatial link inertia
        /// accumulates up the tree. RIG_AUDIT.md section A derives the solve.
        /// </summary>
        public float armatureFoldExact;

        /// <summary>The naive per-joint fold, kept so the sweep can measure the difference.</summary>
        public float armatureFoldNaive;
    }

    /// <summary>Physics values, applied per body and per collider - never project-wide.</summary>
    [Serializable]
    public class MujocoBipedPhysicsDef
    {
        public Vector3 gravityMuj = new Vector3(0f, 0f, -9.81f);

        public float floorFriction = 1.0f;
        public float footFriction = 1.2f;
        public float bodyFriction = 0.9f;

        /// <summary>
        /// MuJoCo takes the elementwise MAXIMUM of the two geoms' friction, so the
        /// foot/floor pair ran at max(1.2, 1.0) = 1.2 during training. Unity has no
        /// Maximum-equivalent that is also safe against an unknown scene ground, so the
        /// shipped material is Minimum - see CONTRACT.md and the rung-6 sweep.
        /// </summary>
        public float effectiveFootGroundFriction = 1.2f;

        // No MJCF joint carries a velocity limit, so these are Unity-side safety valves,
        // set 5x clear of the recorded p100 of 37.14 rad/s.
        public float maxJointVelocity = 200f;
        public float maxAngularVelocity = 200f;
        public float maxLinearVelocity = 200f;
        public float maxDepenetrationVelocity = 10f;

        public float linearDamping = 0f;
        public float angularDamping = 0f;
        public float jointFriction = 0f;

        public int solverPositionIterations = 12;
        public int solverVelocityIterations = 4;

        public float contactOffset = 0.01f;
        public float restOffset = 0f;

        /// <summary>MuJoCo excludes only DIRECT parent-child geom pairs. Leg-vs-leg and
        /// thigh-vs-its-own-foot contacts were live during training.</summary>
        public bool selfCollisionExcludesParentChildOnly = true;

        public float dummyLinkMass = 0.01f;
        public float inertiaFloor = 1e-4f;
    }

    /// <summary>
    /// The rig exactly as MuJoCo simulated it. Generated from MujocoBiped_rig.json by
    /// <c>MujocoBiped/Rebuild Rig Asset From JSON</c>; do not hand-edit.
    ///
    /// Every vector on this asset is in the MUJOCO frame (right-handed, Z-up, X-forward,
    /// metres, radians) and every quaternion is (w, x, y, z).
    /// <see cref="MujocoBipedFrameMap"/> converts, and it does so in exactly one place -
    /// <see cref="MujocoBipedRigBuilder"/> - so one test proves the map for the whole rig.
    /// </summary>
    [CreateAssetMenu(fileName = "MujocoBipedRig", menuName = "MujocoBiped/Rig Asset")]
    public class MujocoBipedRigAsset : ScriptableObject
    {
        public string source;
        public string[] jointOrder;

        public int obsDim = 49;
        public int actDim = 12;

        /// <summary>40 Hz. Divides the project's 0.005 s fixed step exactly 5 times.</summary>
        public float policyDt = 0.025f;
        public float mujocoPhysicsDt = 0.005f;
        public int mujocoFrameSkip = 5;

        /// <summary>init_qpos[0:3]. The TORSO BODY-FRAME ORIGIN, not its centre of mass.</summary>
        public Vector3 spawnPosMuj = new Vector3(0f, 0f, 0.88f);

        // ---- observation, from env.py's _get_obs. All part of the trained contract.
        public float clipLinVel = 10f;
        public float clipAngVel = 10f;
        public float clipJointVel = 20f;
        public float maxTargetDistance = 10f;

        /// <summary>
        /// obs[7:10] is R^T applied to qvel[3:6], which MuJoCo already stores in the BODY
        /// frame - so the angular velocity is rotated into the torso frame TWICE. Proven
        /// in RIG_AUDIT.md section D. Reproduce it; do not fix it. Exposed only so the
        /// rung-4 sweep can measure what "fixing" it would cost.
        /// </summary>
        public bool angularVelocityIsDoubleRotated = true;

        // ---- task, from env.py
        public float reachRadiusM = 0.6f;
        public Vector2 targetDistanceRangeM = new Vector2(3f, 6f);
        public float targetAngleRangeRad = 2.4f;
        public float targetHeightMuj = 0.02f;
        public Vector2 healthyZRange = new Vector2(0.55f, 1.1f);
        public float minUprightness = 0.4f;
        public int maxEpisodeSteps = 1000;

        public MujocoBipedPhysicsDef physics = new MujocoBipedPhysicsDef();
        public MujocoBipedLinkDef[] links;

        // ---- MuJoCo's own eval numbers, for the rung-6 speed-parity gate.
        public float mujocoTargetsReachedPerEpisode = 4f;
        public float mujocoEpisodeLengthSteps = 516f;
        public float mujocoMeanClosingSpeed = 1.15f;
        public float mujocoSurvivedFullEpisodeFraction = 0.30f;

        public MujocoBipedLinkDef Link(string n)
        {
            if (links == null) return null;
            for (int i = 0; i < links.Length; i++)
            {
                if (links[i].name == n) return links[i];
            }
            return null;
        }

        /// <summary>Links that carry a joint, indexed by MuJoCo joint index.</summary>
        public MujocoBipedLinkDef[] JointLinksInMujocoOrder()
        {
            var outp = new MujocoBipedLinkDef[jointOrder.Length];
            for (int i = 0; i < links.Length; i++)
            {
                if (links[i].hasJoint) outp[links[i].joint.index] = links[i];
            }
            return outp;
        }

        /// <summary>Spawn height in Unity metres - MuJoCo Z is Unity Y.</summary>
        public float SpawnHeight => spawnPosMuj.z;
    }
}
