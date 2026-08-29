using System;
using System.Collections.Generic;
using UnityEngine;

namespace MujocoBiped
{
    /// <summary>
    /// Builds the ArticulationBody hierarchy directly from a <see cref="MujocoBipedRigAsset"/>.
    ///
    /// Lives in the RUNTIME assembly, not Editor, so the PlayMode tests can build a rig
    /// without a prefab and without an Editor-only assembly (an Editor-only test assembly
    /// is classified EditMode, where FixedUpdate never runs and none of rungs 1-6 mean
    /// anything).
    ///
    /// The frame map is applied here and ONLY here, so one kinematics test proves it for
    /// the whole rig. Every object is created unscaled - localScale stays (1,1,1) all the
    /// way down - because PhysX cooks collider geometry through the transform and a scaled
    /// parent silently corrupts capsule radii and box extents.
    /// </summary>
    public static class MujocoBipedRigBuilder
    {
        /// <summary>Root object name; also the prefab name.</summary>
        public const string RootName = "MujocoBiped";

        /// <summary>
        /// The ArticulationDrive force limit, which here caps only MuJoCo's passive joint
        /// damping. Large but finite - see the note at the assignment site.
        /// </summary>
        public const float DriveForceLimit = 1e6f;

        /// <summary>
        /// How MuJoCo's per-joint armature is folded into Unity link inertia.
        /// RIG_AUDIT.md section A derives the difference; section C measures what it buys.
        /// </summary>
        public enum ArmatureMode
        {
            /// <summary>Ignore it. The joints then carry less rotor inertia than they did
            /// in training, which shows up as jitter and, on the explicit-torque path,
            /// as a 7x worse stability ratio.</summary>
            None = 0,

            /// <summary>
            /// The exact solve: coefficients chosen so every joint's H[i][i] gains exactly
            /// its MJCF armature. Shipped default.
            /// </summary>
            Exact = 1,

            /// <summary>
            /// Add the armature to every jointed link. Over-counts every parallel-axis run
            /// - the hip sees 3x its armature because hip_y, knee and ankle_y all turn
            /// about the same axis and spatial inertia accumulates up the tree. Kept for
            /// the rung-6 sweep, not for use.
            /// </summary>
            Naive = 2,
        }

        public static GameObject Build(MujocoBipedRigAsset rig, ArmatureMode armatureMode,
                                       PhysicsMaterial material = null)
        {
            if (rig == null) throw new ArgumentNullException(nameof(rig));
            if (rig.links == null || rig.links.Length == 0)
                throw new InvalidOperationException("rig asset has no links");

            var root = new GameObject(RootName);
            var created = new Dictionary<string, GameObject>(rig.links.Length);

            // Links are stored breadth-first from the articulation root, so one forward
            // pass always sees the parent before its children.
            foreach (var def in rig.links)
            {
                GameObject parentGo;
                if (def.isRoot)
                {
                    parentGo = root;
                }
                else if (!created.TryGetValue(def.parent, out parentGo))
                {
                    UnityEngine.Object.DestroyImmediate(root);
                    throw new InvalidOperationException(
                        $"link '{def.name}' names parent '{def.parent}', which has not been " +
                        "created yet - the rig asset is not in breadth-first order.");
                }

                var go = new GameObject(def.name);
                go.transform.SetParent(parentGo.transform, false);
                go.transform.localPosition = MujocoBipedFrameMap.Pos(def.localPosMuj);
                Vector4 q = def.localRotMujWxyz;                    // stored (w, x, y, z)
                go.transform.localRotation = MujocoBipedFrameMap.RotFromWxyz(q.x, q.y, q.z, q.w);
                go.transform.localScale = Vector3.one;

                var body = go.AddComponent<ArticulationBody>();
                ConfigureBody(body, def, rig, armatureMode);
                AddColliders(go, def, rig, material);

                created[def.name] = go;
            }

            return root;
        }

        static void ConfigureBody(ArticulationBody body, MujocoBipedLinkDef def,
                                  MujocoBipedRigAsset rig, ArmatureMode armatureMode)
        {
            var p = rig.physics;

            body.linearDamping = p.linearDamping;
            body.angularDamping = p.angularDamping;
            body.jointFriction = p.jointFriction;
            body.maxLinearVelocity = p.maxLinearVelocity;
            body.maxAngularVelocity = p.maxAngularVelocity;
            body.maxDepenetrationVelocity = p.maxDepenetrationVelocity;
            body.solverIterations = p.solverPositionIterations;
            body.solverVelocityIterations = p.solverVelocityIterations;
            body.useGravity = true;

            ComposeMassAndInertia(def, rig, armatureMode,
                out float mass, out Vector3 com, out Vector3 inertia);
            body.mass = mass;
            body.centerOfMass = com;
            body.inertiaTensor = inertia;
            // Identity is justified by measurement, not assumption: the largest body-axis
            // misalignment in this rig is 14.3 deg on the thighs, and because a capsule's
            // two transverse moments are equal, treating the tensor as body-diagonal costs
            // 0.68% on one moment. RIG_AUDIT.md section E.
            body.inertiaTensorRotation = Quaternion.identity;

            if (def.isRoot)
            {
                // MuJoCo's <freejoint> - a floating articulation root, 6 unconstrained DOF.
                body.immovable = false;
                return;
            }

            var j = def.joint;
            Vector3 axisUnity = MujocoBipedFrameMap.Axis(j.axisInChildMuj).normalized;

            body.jointType = ArticulationJointType.RevoluteJoint;
            body.matchAnchors = true;               // the parent anchor follows this one

            // Every MJCF joint sits at its body's origin (no <joint pos>), and a
            // multi-hinge body's joints all share that one point - which is exactly what
            // the zero-offset dummy chain reproduces.
            body.anchorPosition = Vector3.zero;

            // Unity twists about the anchor frame's local +X. Putting +X at -M*axis makes
            // a positive Unity joint angle equal a positive MuJoCo joint angle, so the
            // agent never flips a sign reading qpos or writing a torque.
            body.anchorRotation = MujocoBipedFrameMap.AnchorRotationForAxis(axisUnity);

            body.twistLock = ArticulationDofLock.LimitedMotion;
            body.swingYLock = ArticulationDofLock.LockedMotion;
            body.swingZLock = ArticulationDofLock.LockedMotion;

            var d = body.xDrive;
            d.lowerLimit = j.lowerRad * Mathf.Rad2Deg;
            d.upperLimit = j.upperRad * Mathf.Rad2Deg;

            // The MJCF actuators are direct-torque MOTORS, not position servos: there is
            // no target, no kp and no PD loop anywhere in the policy. So the drive carries
            // ONLY MuJoCo's passive joint damping, which PhysX integrates implicitly and
            // is therefore unconditionally stable. Actuator torque arrives separately
            // through ArticulationBody.jointForce.
            //
            // Leaving Unity's default position drive alive here is the single most
            // effective way to break this port: it fights every torque the policy emits
            // and the creature folds up.
            d.stiffness = 0f;
            // Scaled by Deg2Rad: ArticulationDrive.damping is a per-DEGREE gain, measured
            // by rung D (MuJoCo's 1.0 written straight through behaves as 98 N.m.s/rad).
            // MujocoBipedAgent.gainUnits re-applies this at runtime and can undo it.
            d.damping = j.damping * Mathf.Deg2Rad;
            d.target = 0f;
            d.targetVelocity = 0f;
            d.driveType = ArticulationDriveType.Force;

            // forceLimit caps the DRIVE, which here is only the passive damping term.
            // MuJoCo never clips passive damping, so this must not either - a limit of
            // gear would silently saturate the damping exactly when the joint is moving
            // fastest and needs it most.
            //
            // Large but FINITE. float.MaxValue is not a safe stand-in for "no limit":
            // PhysX scales the limit by dt and clamps against it, and at 1/120 s that
            // arithmetic blew up (rung 3 measured 8.05e6 rad/s of joint velocity with
            // float.MaxValue, against 65 rad/s at the same amplitude with this value).
            // MuJoCo's damping torque at the fastest joint velocity it ever recorded
            // (37.14 rad/s x 1.0 N.m.s/rad) is 37 N.m, so 1e6 is unreachable by six
            // orders of magnitude and effectively unlimited.
            d.forceLimit = DriveForceLimit;
            body.xDrive = d;

            // No MJCF joint has a velocity limit. This is a safety valve, not a model
            // property - RIG_AUDIT.md section B.
            body.maxJointVelocity = p.maxJointVelocity;
        }

        /// <summary>
        /// Mass, centre of mass and inertia for one link, in Unity axes. Shared by the
        /// builder and the runtime agent so a prefab and a re-applied armature mode can
        /// never disagree about what a link weighs.
        /// </summary>
        public static void ComposeMassAndInertia(MujocoBipedLinkDef def, MujocoBipedRigAsset rig,
                                                 ArmatureMode mode, out float mass,
                                                 out Vector3 com, out Vector3 inertia)
        {
            var p = rig.physics;

            if (def.isDummy)
            {
                // A placeholder carries no geometry, so it has no physical mass or inertia
                // of its own. It gets a token mass and an explicit inertia floor: an
                // articulation is solved in reduced coordinates, where a light link
                // between two heavy ones adds a degree of freedom rather than a stiff
                // constraint, so the mass ratio is benign - but a zero or near-zero
                // INERTIA is not, and PhysX will not condition around it.
                mass = p.dummyLinkMass;
                com = Vector3.zero;
                inertia = new Vector3(p.inertiaFloor, p.inertiaFloor, p.inertiaFloor);
            }
            else
            {
                mass = def.mass;
                com = MujocoBipedFrameMap.Pos(def.comMuj);
                inertia = MujocoBipedFrameMap.InertiaDiag(def.inertiaDiagMuj);
                inertia = new Vector3(Mathf.Max(inertia.x, p.inertiaFloor),
                                      Mathf.Max(inertia.y, p.inertiaFloor),
                                      Mathf.Max(inertia.z, p.inertiaFloor));
            }

            float armature = ArmatureFor(def, mode);
            if (armature != 0f && def.hasJoint)
            {
                // Adding armature * a*a^T makes a^T I' a == a^T I a + armature, i.e. the
                // joint-space inertia about THIS axis gains exactly the armature. Every
                // axis in this rig is exactly axis-aligned in its own link frame, so the
                // outer product is diagonal and no eigen-decomposition is needed; the
                // assert below is what guarantees that stays true.
                Vector3 a = MujocoBipedFrameMap.Axis(def.joint.axisInChildMuj).normalized;
                inertia += new Vector3(armature * a.x * a.x,
                                       armature * a.y * a.y,
                                       armature * a.z * a.z);
            }
        }

        /// <summary>The armature this link folds in, under one mode.</summary>
        public static float ArmatureFor(MujocoBipedLinkDef def, ArmatureMode mode)
        {
            if (!def.hasJoint) return 0f;
            switch (mode)
            {
                case ArmatureMode.Exact: return def.armatureFoldExact;
                case ArmatureMode.Naive: return def.armatureFoldNaive;
                default: return 0f;
            }
        }

        // ------------------------------------------------------------------ geometry --
        static void AddColliders(GameObject go, MujocoBipedLinkDef def,
                                 MujocoBipedRigAsset rig, PhysicsMaterial material)
        {
            if (def.geoms == null) return;

            foreach (var g in def.geoms)
            {
                // Each collider gets its own unscaled child so a tilted capsule can be
                // ROTATED rather than approximated. thigh_l's capsule runs along
                // (0, 0.01, -0.38), which no axis-aligned CapsuleCollider can express.
                var child = new GameObject("col_" + g.name);
                child.transform.SetParent(go.transform, false);
                child.transform.localScale = Vector3.one;

                Collider col;
                switch (g.kind)
                {
                    case "capsule":
                        col = AddCapsule(child, g);
                        break;
                    case "box":
                        col = AddBox(child, g);
                        break;
                    case "sphere":
                        col = AddSphere(child, g);
                        break;
                    default:
                        UnityEngine.Object.DestroyImmediate(child);
                        continue;
                }

                col.contactOffset = rig.physics.contactOffset;
                if (material != null) col.sharedMaterial = material;
            }
        }

        static Collider AddCapsule(GameObject child, MujocoBipedGeomDef g)
        {
            Vector3 a = MujocoBipedFrameMap.Pos(g.a);
            Vector3 b = MujocoBipedFrameMap.Pos(g.b);
            Vector3 axis = b - a;
            float length = axis.magnitude;

            // Put the child's local +Y along the capsule axis and use direction = 1, so
            // the collider needs no scale and the rotation carries the tilt exactly.
            child.transform.localPosition = (a + b) * 0.5f;
            child.transform.localRotation = length > 1e-6f
                ? Quaternion.FromToRotation(Vector3.up, axis / length)
                : Quaternion.identity;

            var c = child.AddComponent<CapsuleCollider>();
            c.direction = 1;                       // local Y
            c.radius = g.radius;
            // Unity measures a capsule's height end-to-end INCLUDING both hemispherical
            // caps; MuJoCo's fromto spans only the cylindrical section.
            c.height = length + 2f * g.radius;
            c.center = Vector3.zero;
            return c;
        }

        static Collider AddBox(GameObject child, MujocoBipedGeomDef g)
        {
            child.transform.localPosition = MujocoBipedFrameMap.Pos(g.pos);
            child.transform.localRotation = Quaternion.identity;

            var c = child.AddComponent<BoxCollider>();
            // MJCF box "size" is HALF-extents, so double before mapping. The map can flip
            // a sign, and a BoxCollider size must be positive.
            Vector3 s = MujocoBipedFrameMap.Pos(g.half * 2f);
            c.size = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
            c.center = Vector3.zero;
            return c;
        }

        static Collider AddSphere(GameObject child, MujocoBipedGeomDef g)
        {
            child.transform.localPosition = MujocoBipedFrameMap.Pos(g.pos);
            child.transform.localRotation = Quaternion.identity;

            var c = child.AddComponent<SphereCollider>();
            c.radius = g.radius;
            c.center = Vector3.zero;
            return c;
        }
    }
}
