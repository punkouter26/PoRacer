using System;
using UnityEngine;

namespace IsaacBox
{
    /// <summary>
    /// The IsaacBox rig exactly as Isaac Lab simulates it. Generated from isaacbox_rig.json by
    /// <c>IsaacBox/Rebuild Rig Asset From JSON</c>; do not hand-edit.
    ///
    /// Every vector on this asset is in the ISAAC frame (right-handed, Z-up, X-forward,
    /// metres, radians). <see cref="PoRacer.IsaacPorts.IsaacFrameMap"/> converts, and it
    /// does so in exactly one place - <c>IsaacBoxRigBuilder</c> - so the map is provable by
    /// the kinematics test.
    ///
    /// The ZERO joint pose is the authored T-pose of IsaacBox_Character.fbx; every link frame
    /// is world-aligned there. <see cref="BoyJointDef.defaultPosRad"/> is the standing
    /// pose the policy's actions are offset from.
    /// </summary>
    [Serializable]
    public class BoyJointDef
    {
        public string name;
        public int index;
        public string family;

        /// <summary>Unit rotation axis in the CHILD link frame, Isaac convention.</summary>
        public Vector3 axisInChildIsaac;

        public float lowerRad;
        public float upperRad;

        /// <summary>kp [N.m/rad], the ImplicitActuator stiffness the policy trained against.</summary>
        public float stiffness;

        /// <summary>kd [N.m.s/rad].</summary>
        public float damping;

        /// <summary>effort_limit_sim [N.m].</summary>
        public float effortLimit;

        public float defaultPosRad;

        /// <summary>PhysX joint armature. Zero on this rig by design, so Unity needs no fold.</summary>
        public float armature;
    }

    [Serializable]
    public class BoyColliderDef
    {
        /// <summary>box | sphere | capsule</summary>
        public string kind;
        public Vector3 centerIsaac;

        /// <summary>box only: full extents.</summary>
        public Vector3 sizeIsaac;

        /// <summary>sphere and capsule.</summary>
        public float radius;

        /// <summary>capsule only: length of the cylindrical part (caps excluded, USD convention).</summary>
        public float height;

        /// <summary>capsule only: X | Y | Z in the Isaac link frame.</summary>
        public string axis;
    }

    [Serializable]
    public class BoyBodyDef
    {
        public string name;
        public string parent;
        public bool isRoot;

        /// <summary>The skinned-mesh bone this link carries, or empty for intermediate links.</summary>
        public string boneName;

        public float mass;
        public Vector3 comIsaac;
        public Vector3 inertiaDiagIsaac;

        /// <summary>World position at the zero (T-) pose, Isaac frame. The builder checks the FBX against it.</summary>
        public Vector3 worldPosIsaac;

        /// <summary>Position relative to the parent link at the zero pose, Isaac frame.</summary>
        public Vector3 localPosIsaac;

        /// <summary>Parent-relative rotation as (w, x, y, z). Identity on every link of this rig.</summary>
        public Vector4 localRotIsaacWxyz;

        public bool hasJoint;
        public BoyJointDef joint;

        public BoyColliderDef[] colliders = Array.Empty<BoyColliderDef>();
    }

    [Serializable]
    public class BoyPhysicsDef
    {
        public Vector3 gravityIsaac = new Vector3(0f, 0f, -9.81f);

        public float groundStaticFriction = 1f;
        public float groundDynamicFriction = 1f;
        public float groundRestitution = 0f;

        /// <summary>The Play task's fixed robot friction; training randomised [0.6,1.0]/[0.4,0.8].</summary>
        public float robotStaticFriction = 0.8f;
        public float robotDynamicFriction = 0.6f;
        public float robotRestitution = 0f;
        public string frictionCombineMode = "multiply";

        public float maxLinearVelocity = 1000f;
        public float maxAngularVelocity = 1000f;
        public float maxDepenetrationVelocity = 1f;
        public float linearDamping = 0f;
        public float angularDamping = 0f;
        public float jointFriction = 0f;
        public int solverPositionIterations = 4;
        public int solverVelocityIterations = 4;
        public bool enabledSelfCollisions = false;
        public float contactOffset = 0.02f;
        public float restOffset = 0f;
        public string isaacSolverType = "TGS";
    }

    [Serializable]
    public class BoyChaseDef
    {
        /// <summary>target_pos_b is scaled down to this length when it is longer.</summary>
        public float targetObsClip = 5f;
        public float targetRadiusMin = 3f;
        public float targetRadiusMax = 10f;
        public float reachRadius = 0.5f;
        public float resampleSecondsMin = 8f;
        public float resampleSecondsMax = 12f;
        public float targetSpeed = 1f;
    }

    [CreateAssetMenu(fileName = "BoyRig", menuName = "IsaacBox/Rig Asset")]
    public class IsaacBoxRigAsset : ScriptableObject
    {
        public string sourceTask;
        public string trainTask;
        public string sourceModel;
        public string checkpoint;

        public string[] jointOrder;
        public string[] bodyOrder;
        public string[] skinBones;

        public int obsDim = 75;
        public int actDim = 21;

        /// <summary>joint_position_target[i] = default[i] + actionScale * action[i].</summary>
        public float actionScale = 0.5f;
        public bool useDefaultOffset = true;

        public float policyDt = 0.02f;
        public float isaacPhysicsDt = 0.005f;
        public int isaacDecimation = 4;
        public float episodeLengthS = 20f;

        public Vector3 spawnPosIsaac = new Vector3(0f, 0f, 0.764f);
        public float hipsHeightAtDefaultPoseRest = 0.744f;
        public float hipsHeightAtZeroPoseRest = 0.76f;

        public float totalMass = 45f;

        public BoyPhysicsDef physics = new BoyPhysicsDef();
        public BoyChaseDef chase = new BoyChaseDef();
        public BoyBodyDef[] bodies;

        // Isaac's own evaluation, filled by export_bundle.py. Zero until a policy exists.
        public float isaacMeanSpeed;
        public float isaacMeanSpeedTowardTarget;
        public float isaacReferenceForwardSpeed;
        public float isaacReferenceTargetDistance = 8f;
        public float isaacFallsPerRobotPerMinute;
        public float isaacTargetsReachedPerMinute;

        public BoyBodyDef Body(string n)
        {
            if (bodies == null) return null;
            for (int i = 0; i < bodies.Length; i++)
                if (bodies[i].name == n) return bodies[i];
            return null;
        }

        /// <summary>Bodies that carry a joint, ordered by Isaac's joint index.</summary>
        public BoyBodyDef[] JointBodiesInIsaacOrder()
        {
            var outp = new BoyBodyDef[jointOrder.Length];
            for (int i = 0; i < bodies.Length; i++)
                if (bodies[i].hasJoint) outp[bodies[i].joint.index] = bodies[i];
            return outp;
        }
    }
}
