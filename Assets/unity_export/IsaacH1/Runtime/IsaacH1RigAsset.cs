using System;
using UnityEngine;

namespace IsaacH1
{
    /// <summary>
    /// The rig, exactly as Isaac simulated it. Generated from IsaacH1_rig.json by
    /// <c>IsaacH1/Rebuild Rig Asset From JSON</c>; do not hand-edit.
    ///
    /// Every vector on this asset is in the ISAAC frame (right-handed, Z-up, metres,
    /// radians). <see cref="IsaacH1FrameMap"/> converts, and it does so in exactly one
    /// place - <c>IsaacH1RigBuilder</c> - so the map is provable by a single test.
    ///
    /// Source precedence, and it matters: the vendor URDF disagrees with the USD Isaac
    /// actually simulated on masses, inertias, joint limits, effort limits and the
    /// number of collision shapes. The USD/env.yaml values win everywhere; the raw URDF
    /// values are carried alongside so audits and floors stay re-appliable.
    /// </summary>
    [Serializable]
    public class IsaacH1JointDef
    {
        public string name;
        public int index;

        /// <summary>Unit rotation axis in the CHILD link frame, Isaac convention.</summary>
        public Vector3 axisInChildIsaac;

        public float lowerRad;
        public float upperRad;

        /// <summary>kp, from env.yaml actuators (NOT from the USD drive block).</summary>
        public float stiffness;

        /// <summary>kd, from env.yaml actuators.</summary>
        public float damping;

        /// <summary>effort_limit_sim: 300 legs/arms, 100 ankles. Overrides the URDF.</summary>
        public float effortLimit;

        public float defaultPosRad;

        /// <summary>
        /// PhysX articulation rotor inertia (physxJoint:armature = 0.1 kg.m2 on every
        /// joint). env.yaml leaves armature null, so this USD value was live during
        /// training. Unity's ArticulationBody exposes no equivalent field - the builder
        /// folds it into the child link's inertia about this axis. Without it the
        /// explicit-PD stability ratio is 9x worse (RIG_AUDIT.md section C).
        /// </summary>
        public float armature;

        /// <summary>URDF &lt;limit velocity&gt;. Isaac set velocity_limit_sim: null and the
        /// recording exceeds this, so it is reported, not enforced.</summary>
        public float urdfVelocityLimit;

        public float urdfEffortLimit;
    }

    [Serializable]
    public class IsaacH1ColliderDef
    {
        public Vector3 centerIsaac;
        public Vector3 sizeIsaac;
        public string sourceApproximation;
        public int sourceVertexCount;
    }

    [Serializable]
    public class IsaacH1VisualDef
    {
        public string kind;          // box | cylinder | sphere
        public Vector3 originIsaac;
        public Vector3 rpy;
        public Vector3 size;
        public float radius;
        public float length;
    }

    [Serializable]
    public class IsaacH1BodyDef
    {
        public string name;
        public string parent;
        public bool isRoot;

        public float mass;
        public Vector3 comIsaac;
        public Vector3 inertiaDiagIsaac;

        /// <summary>Raw vendor-URDF values, kept so a mass/inertia floor stays re-appliable.</summary>
        public float urdfMass;
        public Vector3 urdfInertiaDiagIsaac;

        /// <summary>Pose relative to the parent link at the ZERO-joint pose, Isaac frame.</summary>
        public Vector3 localPosIsaac;

        /// <summary>Parent-relative rotation as (w, x, y, z), Isaac frame.</summary>
        public Vector4 localRotIsaacWxyz;

        public bool hasJoint;
        public IsaacH1JointDef joint;

        /// <summary>Exactly the shapes Isaac simulated. Only torso and the two feet have any.</summary>
        public IsaacH1ColliderDef[] colliders = Array.Empty<IsaacH1ColliderDef>();

        /// <summary>Non-colliding render proxies from the URDF, for visibility only.</summary>
        public IsaacH1VisualDef[] visuals = Array.Empty<IsaacH1VisualDef>();
    }

    [Serializable]
    public class IsaacH1PhysicsDef
    {
        public Vector3 gravityIsaac = new Vector3(0f, 0f, -9.81f);

        // sim.physics_material - the GROUND plane
        public float groundStaticFriction = 1f;
        public float groundDynamicFriction = 1f;
        public float groundRestitution = 0f;

        // events.physics_material (startup) - the ROBOT's own shapes. The degenerate
        // ranges [0.8,0.8] / [0.6,0.6] make this a fixed value, not a random draw.
        // export_report.json's "ground_material" reports only the ground, so the
        // effective PAIR friction during training was 0.8 static / 0.6 dynamic.
        public float robotStaticFriction = 0.8f;
        public float robotDynamicFriction = 0.6f;
        public float robotRestitution = 0f;

        /// <summary>Isaac combined with multiply. See CONTRACT.md for why Unity ships Minimum.</summary>
        public string frictionCombineMode = "multiply";

        // scene.robot.spawn.rigid_props / articulation_props - applied PER BODY,
        // never project-wide.
        public float maxLinearVelocity = 1000f;
        public float maxAngularVelocity = 1000f;
        public float maxDepenetrationVelocity = 1f;
        public float linearDamping = 0f;
        public float angularDamping = 0f;
        public float jointFriction = 0f;
        public int solverPositionIterations = 4;
        public int solverVelocityIterations = 4;
        public bool enabledSelfCollisions = false;

        /// <summary>PhysX default Isaac left untouched. Unity's project default is 0.01,
        /// so this is applied per collider.</summary>
        public float contactOffset = 0.02f;
        public float restOffset = 0f;

        /// <summary>Isaac ran TGS. Unity's solver type is project-wide - reported, not changed.</summary>
        public string isaacSolverType = "TGS";
    }

    [CreateAssetMenu(fileName = "IsaacH1Rig", menuName = "IsaacH1/Rig Asset")]
    public class IsaacH1RigAsset : ScriptableObject
    {
        public string sourceTask;
        public string[] jointOrder;
        public string[] bodyOrder;

        public int obsDim = 69;
        public int actDim = 19;

        /// <summary>joint_position_target[i] = default[i] + actionScale * action[i].</summary>
        public float actionScale = 0.5f;
        public bool useDefaultOffset = true;

        public float policyDt = 0.02f;
        public float isaacPhysicsDt = 0.005f;
        public int isaacDecimation = 4;

        public Vector3 spawnPosIsaac = new Vector3(0f, 0f, 1.05f);

        public IsaacH1PhysicsDef physics = new IsaacH1PhysicsDef();
        public IsaacH1BodyDef[] bodies;

        // Isaac's own eval numbers, for the rung tables and the speed-parity gate.
        public float isaacMeanSpeed = 0.51f;
        public float isaacMeanLinVelTrackingError = 0.117f;
        public float isaacFallsPerRobotPerMinute = 0.125f;
        public Vector3 referenceCommand = new Vector3(1f, 0f, 0f);

        // The reference recording used a randomised torso mass; the prefab ships nominal.
        public float torsoMassNominal = 17.789f;
        public float torsoMassInReferenceRecording = 15.332636f;

        public IsaacH1BodyDef Body(string n)
        {
            for (int i = 0; i < bodies.Length; i++)
                if (bodies[i].name == n) return bodies[i];
            return null;
        }

        /// <summary>Bodies that carry a joint, ordered by Isaac's joint index.</summary>
        public IsaacH1BodyDef[] JointBodiesInIsaacOrder()
        {
            var outp = new IsaacH1BodyDef[jointOrder.Length];
            for (int i = 0; i < bodies.Length; i++)
                if (bodies[i].hasJoint) outp[bodies[i].joint.index] = bodies[i];
            return outp;
        }
    }
}
