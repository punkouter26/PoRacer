using PoRacer.CreatureRace;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Builds a PhysX worm (the Isaac-trained worm, or any racer set to PhysX) as a Unity
    /// ArticulationBody chain, from the same worm_rig.json the MuJoCo worms are built from,
    /// at race start. Separate PhysX worms collide with each other natively.
    ///
    /// The tree is worm.xml's exactly: seg0 (floating root) - link0 (pitch hinge, 0.1 kg,
    /// no geometry) - seg1 (yaw hinge) - link1 - ... - seg4. Two hinges per joint, on two
    /// bodies, because PhysX's spherical joint composes rotations differently from two
    /// MuJoCo hinges and its jointPosition would not map back to qpos (the same reasoning
    /// as MujocoBipedRigBuilder's placeholder chains).
    ///
    /// Frame map (CreatureFrames SPEC map): positions Unity = (-y, z, x) of MuJoCo; hinge axes
    /// go through the axial map, so pitch (MuJoCo y) is Unity +x and yaw (MuJoCo z) is
    /// Unity -y, and a positive jointPosition is a positive MuJoCo qpos. Nothing is scaled:
    /// every localScale stays (1, 1, 1), because PhysX cooks colliders through the
    /// transform and a scaled parent silently corrupts capsule radii.
    ///
    /// Drives (Unity docs: stiffness N*m/rad, target and limits in degrees;
    /// training/bugs/README.md): stiffness = rig kp (30), forceLimit = rig forceLimit,
    /// limits +/-45, and by default NO drive damping - WORM_SPEC detail 16 makes the joint
    /// damping a passive term outside the force limit, which PhysxWormView applies through
    /// jointForce (see WormRaceSettings.PassiveDamping for the alternatives).
    /// </summary>
    internal static class PhysxWormBuilder
    {
        public static GameObject Build(WormRig rig, WormRaceSettings settings, CreatureLayout layout,
                                       CreaturePilot pilot, string rootName, Vector3 rootOrigin,
                                       Vector3 laneForward, Material material, Mesh segmentMesh,
                                       out Transform[] segmentTransforms)
        {
            var root = new GameObject(rootName);
            // Built facing Unity +Z (MuJoCo +x under the SPEC map), then turned onto the lane.
            root.transform.SetPositionAndRotation(rootOrigin, Quaternion.LookRotation(laneForward, Vector3.up));

            int bodyCount = rig.Bodies.Length;
            var bodyObjects = new GameObject[bodyCount];
            var colliders = new CapsuleCollider[bodyCount];
            var bodies = new ArticulationBody[bodyCount];
            var joints = new ArticulationBody[WormContract.ACTION_SIZE];
            segmentTransforms = new Transform[WormContract.SEGMENT_COUNT];

            for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
            {
                WormRig.BodyDef def = rig.Bodies[bodyIndex];
                Transform parent = def.ParentIndex < 0 ? root.transform : bodyObjects[def.ParentIndex].transform;

                var bodyObject = new GameObject(def.Name);
                bodyObject.transform.SetParent(parent, false);
                bodyObject.transform.localPosition = CreatureFrames.UnityFromSpecPolar(def.PositionMuJoCo);
                bodyObject.transform.localRotation = Quaternion.identity;
                bodyObject.transform.localScale = Vector3.one;

                ArticulationBody body = bodyObject.AddComponent<ArticulationBody>();
                bodies[bodyIndex] = body;
                if (def.HasCapsule)
                {
                    // Collider before the explicit mass properties, so an automatic
                    // recompute triggered by the new collider cannot overwrite them.
                    colliders[bodyIndex] = AddCapsule(bodyObject, def, settings.WormPhysicsMaterial,
                                                      material, segmentMesh);
                    segmentTransforms[def.SegmentIndex] = bodyObject.transform;
                }
                ConfigureBody(body, def, rig, settings);
                if (def.HasJoint)
                {
                    joints[def.JointActionIndex] = body;
                }
                bodyObjects[bodyIndex] = bodyObject;
            }

            // worm.xml's <contact><exclude>: neighbouring segments overlap at their joint.
            // They are not parent and child (the pitch link sits between them), so PhysX
            // would collide them. Every other pair keeps colliding (AGENTS rule M).
            for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
            {
                if (rig.IsAdjacentSegmentPair(bodyIndex, out int parentSegment))
                {
                    Physics.IgnoreCollision(colliders[bodyIndex], colliders[parentSegment], true);
                }
            }

            PhysxWormView view = root.AddComponent<PhysxWormView>();
            view.Bind(bodies, joints, pilot, laneForward, layout, PassiveJointDamping(rig, settings));
            return root;
        }

        private static CapsuleCollider AddCapsule(GameObject bodyObject, WormRig.BodyDef def,
                                                  PhysicsMaterial physicsMaterial, Material material,
                                                  Mesh segmentMesh)
        {
            // MuJoCo x is Unity z under the SPEC map: the capsule runs along local Z.
            CapsuleCollider capsule = bodyObject.AddComponent<CapsuleCollider>();
            capsule.direction = 2;
            capsule.radius = def.CapsuleRadius;
            capsule.height = 2f * (def.CapsuleHalfLength + def.CapsuleRadius);
            capsule.center = Vector3.zero;
            capsule.sharedMaterial = physicsMaterial;

            var visual = new GameObject("visual");
            visual.transform.SetParent(bodyObject.transform, false);
            // The shared mesh runs along +Y; turn it onto +Z.
            visual.transform.localRotation = Quaternion.FromToRotation(Vector3.up, Vector3.forward);
            visual.AddComponent<MeshFilter>().sharedMesh = segmentMesh;
            visual.AddComponent<MeshRenderer>().sharedMaterial = material;
            return capsule;
        }

        private static void ConfigureBody(ArticulationBody body, WormRig.BodyDef def, WormRig rig,
                                          WormRaceSettings settings)
        {
            body.useGravity = true;
            // MuJoCo has no per-body drag and worm.xml sets no frictionloss.
            body.linearDamping = 0f;
            body.angularDamping = 0f;
            body.jointFriction = 0f;
            body.solverIterations = settings.SolverIterations;
            body.solverVelocityIterations = settings.SolverVelocityIterations;

            // Explicit mass properties, matching what MuJoCo's compiler derives, so the two
            // simulators carry the same bodies rather than two engines' own estimates.
            Vector3 inertia;
            if (def.HasCapsule)
            {
                WormRig.CapsuleInertia(def.Mass, def.CapsuleRadius, def.CapsuleHalfLength,
                                       out float axial, out float transverse);
                inertia = new Vector3(transverse, transverse, axial);
            }
            else
            {
                inertia = new Vector3(def.PointInertia, def.PointInertia, def.PointInertia);
            }

            Vector3 axisUnity = def.HasJoint
                ? CreatureFrames.UnityFromSpecAxial(def.JointAxisMuJoCo).normalized
                : Vector3.zero;
            if (def.HasJoint && settings.FoldArmatureIntoInertia)
            {
                // PhysX has no armature. Adding armature * a a^T to the child's inertia adds
                // exactly the armature to this hinge's own joint-space inertia (a is
                // axis-aligned here, so the outer product is diagonal). Upstream hinges with a
                // parallel axis see it too - a few percent of their load, see INSTALL.md.
                inertia += rig.Armature * new Vector3(axisUnity.x * axisUnity.x,
                                                      axisUnity.y * axisUnity.y,
                                                      axisUnity.z * axisUnity.z);
            }
            body.mass = def.Mass;
            body.centerOfMass = Vector3.zero;
            body.inertiaTensor = inertia;
            body.inertiaTensorRotation = Quaternion.identity;

            if (!def.HasJoint)
            {
                // The free-floating root: MuJoCo's <freejoint>.
                body.immovable = false;
                return;
            }

            body.jointType = ArticulationJointType.RevoluteJoint;
            Vector3 anchorPosition = CreatureFrames.UnityFromSpecPolar(def.JointAnchorMuJoCo);
            // PhysX twists about the anchor frame's +X; put +X on the axial-mapped axis.
            Quaternion anchorRotation = Quaternion.FromToRotation(Vector3.right, axisUnity);
            // Both anchors written explicitly rather than derived (matchAnchors): the child
            // sits at localPosition with no local rotation, so the joint point in the
            // parent's frame is localPosition + anchorPosition, with the same rotation.
            body.matchAnchors = false;
            body.anchorPosition = anchorPosition;
            body.anchorRotation = anchorRotation;
            body.parentAnchorPosition = body.transform.localPosition + anchorPosition;
            body.parentAnchorRotation = anchorRotation;
            body.twistLock = ArticulationDofLock.LimitedMotion;
            body.swingYLock = ArticulationDofLock.LockedMotion;
            body.swingZLock = ArticulationDofLock.LockedMotion;

            float limitDegrees = rig.JointRangeRad * Mathf.Rad2Deg;
            ArticulationDrive drive = body.xDrive;
            drive.driveType = ArticulationDriveType.Force;
            drive.lowerLimit = -limitDegrees;
            drive.upperLimit = limitDegrees;
            drive.stiffness = rig.Kp;
            drive.damping = DriveDamping(rig, settings);
            drive.forceLimit = rig.ForceLimit;
            drive.target = 0f;
            drive.targetVelocity = 0f;
            body.xDrive = drive;
        }

        /// <summary>The damping the drive carries: none when the view applies it passively.</summary>
        private static float DriveDamping(WormRig rig, WormRaceSettings settings)
        {
            switch (settings.JointDampingMode)
            {
                case WormRaceSettings.PassiveDamping.DriveDampingPerRadian:
                    return rig.JointDamping;
                case WormRaceSettings.PassiveDamping.DriveDampingPerDegree:
                    return rig.JointDamping * Mathf.Deg2Rad;
                default:
                    return 0f;
            }
        }

        /// <summary>The -c*qdot the view applies through jointForce each step (0 = none).</summary>
        private static float PassiveJointDamping(WormRig rig, WormRaceSettings settings)
        {
            return settings.JointDampingMode == WormRaceSettings.PassiveDamping.JointForceOutsideLimit
                ? rig.JointDamping
                : 0f;
        }
    }
}
