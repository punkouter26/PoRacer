using System.Collections.Generic;
using PoRacer.CreatureRace;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// worm_rig.json predates the creature rig format, so the worm reaches the generic
    /// MuJoCo builder through this adapter instead of a second file: the same WormRig the
    /// PhysX worm is built from becomes a <see cref="CreatureRig"/> describing exactly the
    /// model the worm's own MuJoCo builder used to create (and worm.xml describes):
    ///
    ///   * segments: one capsule along x (a fromto of +/- halfLength), its mass on the geom;
    ///   * pitch links: an explicit inertial (mass 0.1, diagonal 0.0002), no geometry;
    ///   * one hinge per non-root body (limited +/-45 degrees, damping, armature);
    ///   * eight position servos in the contract's action order (kp, ctrl +/-45 degrees, force limit);
    ///   * excludes between segments joined through one pitch link;
    ///   * worm.xml's contact on every geom and on the floor: friction (rig, 0.005, 0.0001),
    ///     condim 3, solref 0.01 1, solimp 0.9 0.95 0.001.
    ///
    /// Bodies keep WormRig's order, so a body index means the same body to the PhysX worm.
    /// </summary>
    internal static class WormRigAdapter
    {
        private const float TORSIONAL_FRICTION = 0.005f;
        private const float ROLLING_FRICTION = 0.0001f;
        private const int CONTACT_DIMENSION = 3;
        private static readonly Vector2 ContactSolRef = new(0.01f, 1f);
        private static readonly Vector3 ContactSolImp = new(0.9f, 0.95f, 0.001f);

        /// <summary>worm.xml's &lt;default&gt;&lt;geom&gt; contact, for the worm, the floor and the stand-ins.</summary>
        public static CreatureContact SegmentContact(WormRig rig)
        {
            return new CreatureContact(new Vector3(rig.Friction, TORSIONAL_FRICTION, ROLLING_FRICTION),
                                       CONTACT_DIMENSION, ContactSolRef, ContactSolImp);
        }

        public static CreatureRig ToCreatureRig(WormRig rig)
        {
            CreatureContact contact = SegmentContact(rig);
            int bodyCount = rig.Bodies.Length;
            var bodies = new CreatureBodyDef[bodyCount];
            var geoms = new CreatureGeomDef[WormContract.SEGMENT_COUNT];
            var joints = new CreatureJointDef[bodyCount - 1];
            var jointOfBody = new int[bodyCount];
            int geomCount = 0;
            int jointCount = 0;

            for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
            {
                WormRig.BodyDef def = rig.Bodies[bodyIndex];
                bodies[bodyIndex] = new CreatureBodyDef
                {
                    Name = def.Name,
                    Parent = def.ParentIndex,
                    Position = def.PositionMuJoCo,
                    FreeJoint = def.ParentIndex < 0,
                    HasInertial = !def.HasCapsule,
                    Mass = def.HasCapsule ? 0f : def.Mass,
                    InertiaDiagonal = def.HasCapsule
                        ? Vector3.zero
                        : new Vector3(def.PointInertia, def.PointInertia, def.PointInertia),
                };
                if (def.HasCapsule)
                {
                    float halfLength = def.CapsuleHalfLength;
                    geoms[geomCount++] = new CreatureGeomDef
                    {
                        Name = def.Name + "_geom",
                        Body = bodyIndex,
                        Type = CreatureGeomType.Capsule,
                        Radius = def.CapsuleRadius,
                        HalfLength = halfLength,
                        HasFromTo = true,
                        From = new Vector3(-halfLength, 0f, 0f),
                        To = new Vector3(halfLength, 0f, 0f),
                        Mass = def.Mass,
                        Contact = contact,
                    };
                }
                jointOfBody[bodyIndex] = -1;
                if (def.HasJoint)
                {
                    jointOfBody[bodyIndex] = jointCount;
                    joints[jointCount++] = new CreatureJointDef
                    {
                        Name = def.JointName,
                        Body = bodyIndex,
                        Axis = def.JointAxisMuJoCo,
                        Anchor = def.JointAnchorMuJoCo,
                        Limited = true,
                        RangeLower = -rig.JointRangeRad,
                        RangeUpper = rig.JointRangeRad,
                        Damping = rig.JointDamping,
                        Armature = rig.Armature,
                    };
                }
            }

            var actuators = new CreatureActuatorDef[WormContract.ACTION_SIZE];
            for (int actionIndex = 0; actionIndex < actuators.Length; actionIndex++)
            {
                actuators[actionIndex] = new CreatureActuatorDef
                {
                    Name = WormContract.ActionOrder[actionIndex],
                    Joint = jointOfBody[rig.ActionBodies[actionIndex]],
                    Kp = rig.Kp,
                    Kv = 0f,
                    CtrlLimited = true,
                    CtrlRange = new Vector2(-rig.JointRangeRad, rig.JointRangeRad),
                    ForceLimited = true,
                    ForceRange = new Vector2(-rig.ForceLimit, rig.ForceLimit),
                };
            }

            var excludes = new List<Vector2Int>();
            for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
            {
                if (rig.IsAdjacentSegmentPair(bodyIndex, out int parentSegment))
                {
                    excludes.Add(new Vector2Int(parentSegment, bodyIndex));
                }
            }

            return new CreatureRig("worm5", bodies, geoms, joints, actuators, excludes.ToArray(),
                                   new float[WormContract.ACTION_SIZE], rig.JointRangeRad)
            {
                PhysicsDt = WormContract.PHYSICS_DT,
                Decimation = WormContract.DECIMATION,
                TorsoBody = rig.SegmentBodies[WormContract.REFERENCE_SEGMENT],
                FloorContact = contact,
            };
        }
    }
}
