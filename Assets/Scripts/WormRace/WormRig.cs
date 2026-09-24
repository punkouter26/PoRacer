using System;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// worm_rig.json, parsed and checked. Both worms are built from this one file, which
    /// build_worm.py writes from the same constants as worm.xml, so the MuJoCo worm and the
    /// PhysX worm are the same animal by construction.
    ///
    /// Everything is kept in MuJoCo coordinates here (x forward, y left, z up). The builders
    /// apply their own frame map; see <see cref="WormFrames"/>.
    ///
    /// The parser is strict on purpose: the builders assume a tree in which every non-root
    /// body carries exactly one hinge (the pitch link and the yaw segment), and a rig that
    /// breaks that assumption must fail loudly at load rather than build a different worm.
    /// </summary>
    internal sealed class WormRig
    {
        private const float RANGE_TOLERANCE = 1e-4f;

        internal sealed class BodyDef
        {
            public string Name;
            public int ParentIndex;
            public Vector3 PositionMuJoCo;
            public float Mass;
            public bool HasCapsule;
            public float CapsuleHalfLength;
            public float CapsuleRadius;
            /// <summary>Diagonal inertia for bodies without geometry (the pitch links).</summary>
            public float PointInertia;
            public bool HasJoint;
            public string JointName;
            public Vector3 JointAxisMuJoCo;
            public Vector3 JointAnchorMuJoCo;
            public int JointActionIndex;
            /// <summary>0..4 for capsule segments, -1 for links.</summary>
            public int SegmentIndex;
        }

        public BodyDef[] Bodies { get; private set; }
        public int[] SegmentBodies { get; private set; }
        public int[] ActionBodies { get; private set; }
        public float Kp { get; private set; }
        public float ForceLimit { get; private set; }
        public float JointDamping { get; private set; }
        public float Armature { get; private set; }
        public float Friction { get; private set; }
        public float JointRangeRad { get; private set; }
        public float SegmentRadius { get; private set; }
        public float SegmentHalfLength { get; private set; }

        /// <summary>Distance from segment 0's centre to the tip of its capsule.</summary>
        public float NoseOffset => SegmentHalfLength + SegmentRadius;

        public static bool TryParse(string json, out WormRig rig, out string error)
        {
            rig = null;
            if (string.IsNullOrEmpty(json))
            {
                error = "worm_rig.json is empty or unassigned";
                return false;
            }

            RigJson source;
            try
            {
                source = JsonUtility.FromJson<RigJson>(json);
            }
            catch (Exception exception)
            {
                error = "worm_rig.json does not parse: " + exception.Message;
                return false;
            }
            if (source == null || source.bodies == null || source.joints == null || source.actionOrder == null)
            {
                error = "worm_rig.json is missing bodies, joints or actionOrder";
                return false;
            }
            if (source.segments != WormContract.SEGMENT_COUNT)
            {
                error = $"worm_rig.json has {source.segments} segments; the contract is {WormContract.SEGMENT_COUNT}";
                return false;
            }
            if (source.actionOrder.Length != WormContract.ACTION_SIZE || source.joints.Length != WormContract.ACTION_SIZE)
            {
                error = $"worm_rig.json has {source.joints.Length} joints; the contract is {WormContract.ACTION_SIZE}";
                return false;
            }
            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                if (source.actionOrder[actionIndex] != WormContract.ActionOrder[actionIndex])
                {
                    error = $"worm_rig.json actionOrder[{actionIndex}] is '{source.actionOrder[actionIndex]}', "
                          + $"the contract says '{WormContract.ActionOrder[actionIndex]}'";
                    return false;
                }
            }
            if (Mathf.Abs(source.jointRangeRad - WormContract.JOINT_RANGE_RAD) > RANGE_TOLERANCE)
            {
                error = $"worm_rig.json jointRangeRad {source.jointRangeRad} is not the contract's 45 degrees";
                return false;
            }

            var parsed = new WormRig
            {
                Kp = source.kp,
                ForceLimit = source.forceLimit,
                JointDamping = source.jointDamping,
                Armature = source.armature,
                Friction = source.friction,
                JointRangeRad = source.jointRangeRad,
                SegmentRadius = source.radius,
                SegmentHalfLength = source.capsuleHalfLength,
                Bodies = new BodyDef[source.bodies.Length],
                SegmentBodies = new int[WormContract.SEGMENT_COUNT],
                ActionBodies = new int[WormContract.ACTION_SIZE],
            };

            int segmentCount = 0;
            for (int bodyIndex = 0; bodyIndex < source.bodies.Length; bodyIndex++)
            {
                BodyJson body = source.bodies[bodyIndex];
                int parentIndex = -1;
                if (!string.IsNullOrEmpty(body.parent))
                {
                    parentIndex = FindBody(parsed.Bodies, bodyIndex, body.parent);
                    if (parentIndex < 0)
                    {
                        error = $"body '{body.name}' names parent '{body.parent}', which is not listed before it";
                        return false;
                    }
                }
                else if (bodyIndex != 0)
                {
                    error = $"body '{body.name}' has no parent but is not the first body";
                    return false;
                }

                bool hasCapsule = body.capsule != null && body.capsule.radius > 0f;
                if (hasCapsule && body.capsule.axis != "x")
                {
                    error = $"body '{body.name}' has a capsule along '{body.capsule.axis}'; only 'x' is supported";
                    return false;
                }
                if (body.pos == null || body.pos.Length != 3)
                {
                    error = $"body '{body.name}' has no 3-element pos";
                    return false;
                }

                var def = new BodyDef
                {
                    Name = body.name,
                    ParentIndex = parentIndex,
                    PositionMuJoCo = new Vector3(body.pos[0], body.pos[1], body.pos[2]),
                    Mass = body.mass,
                    HasCapsule = hasCapsule,
                    CapsuleHalfLength = hasCapsule ? body.capsule.halfLength : 0f,
                    CapsuleRadius = hasCapsule ? body.capsule.radius : 0f,
                    PointInertia = body.inertia > 0f ? body.inertia : source.linkInertia,
                    SegmentIndex = -1,
                    JointActionIndex = -1,
                };
                if (hasCapsule)
                {
                    if (segmentCount >= WormContract.SEGMENT_COUNT)
                    {
                        error = "worm_rig.json has more capsule bodies than segments";
                        return false;
                    }
                    def.SegmentIndex = segmentCount;
                    parsed.SegmentBodies[segmentCount] = bodyIndex;
                    segmentCount++;
                }
                if (def.Mass <= 0f)
                {
                    error = $"body '{body.name}' has no mass";
                    return false;
                }
                parsed.Bodies[bodyIndex] = def;
            }
            if (segmentCount != WormContract.SEGMENT_COUNT)
            {
                error = $"worm_rig.json has {segmentCount} capsule bodies, expected {WormContract.SEGMENT_COUNT}";
                return false;
            }

            for (int jointIndex = 0; jointIndex < source.joints.Length; jointIndex++)
            {
                JointJson joint = source.joints[jointIndex];
                int bodyIndex = FindBody(parsed.Bodies, parsed.Bodies.Length, joint.body);
                if (bodyIndex < 0)
                {
                    error = $"joint '{joint.name}' names unknown body '{joint.body}'";
                    return false;
                }
                BodyDef def = parsed.Bodies[bodyIndex];
                if (def.HasJoint)
                {
                    error = $"body '{def.Name}' carries two joints; the builders support one hinge per body";
                    return false;
                }
                if (def.ParentIndex < 0)
                {
                    error = $"joint '{joint.name}' sits on the root body; the root has a free joint only";
                    return false;
                }
                if (joint.axis == null || joint.axis.Length != 3 || joint.anchor == null || joint.anchor.Length != 3)
                {
                    error = $"joint '{joint.name}' needs a 3-element axis and anchor";
                    return false;
                }
                int actionIndex = Array.IndexOf(WormContract.ActionOrder, joint.name);
                if (actionIndex < 0)
                {
                    error = $"joint '{joint.name}' is not in the contract's action order";
                    return false;
                }
                def.HasJoint = true;
                def.JointName = joint.name;
                def.JointAxisMuJoCo = new Vector3(joint.axis[0], joint.axis[1], joint.axis[2]).normalized;
                def.JointAnchorMuJoCo = new Vector3(joint.anchor[0], joint.anchor[1], joint.anchor[2]);
                def.JointActionIndex = actionIndex;
                parsed.ActionBodies[actionIndex] = bodyIndex;
            }
            for (int bodyIndex = 1; bodyIndex < parsed.Bodies.Length; bodyIndex++)
            {
                if (!parsed.Bodies[bodyIndex].HasJoint)
                {
                    error = $"body '{parsed.Bodies[bodyIndex].Name}' has no joint; welded bodies are not supported";
                    return false;
                }
            }

            rig = parsed;
            error = string.Empty;
            return true;
        }

        /// <summary>
        /// True for the pairs worm.xml excludes from contact: two segments joined through a
        /// single pitch link (segment - link - segment). They overlap at the joint by design.
        /// </summary>
        public bool IsAdjacentSegmentPair(int childSegmentBody, out int parentSegmentBody)
        {
            parentSegmentBody = -1;
            BodyDef child = Bodies[childSegmentBody];
            if (!child.HasCapsule || child.ParentIndex < 0)
            {
                return false;
            }
            BodyDef link = Bodies[child.ParentIndex];
            if (link.HasCapsule || link.ParentIndex < 0)
            {
                return false;
            }
            if (!Bodies[link.ParentIndex].HasCapsule)
            {
                return false;
            }
            parentSegmentBody = link.ParentIndex;
            return true;
        }

        /// <summary>
        /// Solid-capsule inertia about its centre, the formula MuJoCo's compiler uses for a
        /// capsule geom with an explicit mass, so PhysX and MuJoCo carry the same tensor.
        /// </summary>
        public static void CapsuleInertia(float mass, float radius, float halfLength,
                                          out float axial, out float transverse)
        {
            float height = 2f * halfLength;
            float sphereMass = mass * 4f * radius / (4f * radius + 3f * height);
            float cylinderMass = mass - sphereMass;
            transverse = cylinderMass * (3f * radius * radius + height * height) / 12f;
            axial = cylinderMass * radius * radius / 2f;
            float sphereInertia = sphereMass * 2f * radius * radius / 5f;
            transverse += sphereInertia + sphereMass * height * (3f * radius + 2f * height) / 8f;
            axial += sphereInertia;
        }

        private static int FindBody(BodyDef[] bodies, int count, string wanted)
        {
            for (int bodyIndex = 0; bodyIndex < count; bodyIndex++)
            {
                if (bodies[bodyIndex] != null && bodies[bodyIndex].Name == wanted)
                {
                    return bodyIndex;
                }
            }
            return -1;
        }

        // ---------------------------------------------------------------- JSON shape --
        // Field names match worm_rig.json exactly; JsonUtility maps by name. Only JsonUtility
        // writes them, which the compiler cannot see, hence the CS0649 suppression.
#pragma warning disable 0649

        [Serializable]
        private sealed class RigJson
        {
            public string model;
            public int segments;
            public float segmentSpacing;
            public float capsuleHalfLength;
            public float radius;
            public float segmentMass;
            public float linkMass;
            public float linkInertia;
            public float totalMass;
            public float jointRangeRad;
            public float kp;
            public float forceLimit;
            public float jointDamping;
            public float armature;
            public float friction;
            public float spawnHeight;
            public float physicsDt;
            public int decimation;
            public BodyJson[] bodies;
            public JointJson[] joints;
            public string[] actionOrder;
        }

        [Serializable]
        private sealed class BodyJson
        {
            public string name;
            public string parent;
            public float[] pos;
            public float mass;
            public float inertia;
            public CapsuleJson capsule;
        }

        [Serializable]
        private sealed class CapsuleJson
        {
            public float halfLength;
            public float radius;
            public string axis;
        }

        [Serializable]
        private sealed class JointJson
        {
            public string name;
            public string body;
            public float[] axis;
            public float[] anchor;
        }
#pragma warning restore 0649
    }
}
