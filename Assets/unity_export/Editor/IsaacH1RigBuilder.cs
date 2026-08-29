using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace IsaacH1.EditorTools
{
    /// <summary>
    /// Builds the ArticulationBody hierarchy directly from <see cref="IsaacH1RigAsset"/>.
    ///
    /// Deliberately NOT the URDF Importer: that package is not in this project's manifest
    /// and this task must not add it. It would also be the wrong tool here - it imports
    /// the vendor URDF, whose masses, inertias, joint limits and collision shapes all
    /// disagree with the USD Isaac actually simulated, and it emits scaled convex meshes.
    /// (If the importer is ever present, ISAACPORTS_HAS_URDF_IMPORTER is defined and
    /// IsaacH1Setup offers to strip its Controller/JointControl/FKRobot components.)
    ///
    /// The frame map is applied here and ONLY here, so one test proves it for the whole
    /// rig. Every child object is created unscaled (localScale == 1) and colliders are
    /// primitives, so nothing inherits a scale that would corrupt PhysX cooking.
    /// </summary>
    public static class IsaacH1RigBuilder
    {
        /// <summary>Root object name; also the prefab name.</summary>
        public const string RootName = "IsaacH1";

        public static GameObject Build(IsaacH1RigAsset rig, IsaacH1Agent.ArmatureMode armatureMode,
                                      IsaacH1MeshLibrary meshes = null)
        {
            if (rig == null) throw new ArgumentNullException(nameof(rig));

            var root = new GameObject(RootName);
            var created = new Dictionary<string, GameObject>(rig.bodies.Length);

            // Bodies are stored in Isaac's own order, which is breadth-first from the
            // articulation root, so a single forward pass always sees the parent first.
            foreach (var def in rig.bodies)
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
                        $"body '{def.name}' names parent '{def.parent}', which has not been " +
                        "created yet - the rig asset is not in breadth-first order.");
                }

                var go = new GameObject(def.name);
                go.transform.SetParent(parentGo.transform, false);
                go.transform.localPosition = IsaacH1FrameMap.Pos(def.localPosIsaac);
                Vector4 q = def.localRotIsaacWxyz;   // stored (w, x, y, z)
                go.transform.localRotation = IsaacH1FrameMap.RotFromWxyz(q.x, q.y, q.z, q.w);
                go.transform.localScale = Vector3.one;

                var body = go.AddComponent<ArticulationBody>();
                ConfigureBody(body, def, rig, armatureMode);
                AddColliders(go, def, rig);
                AddVisuals(go, def, meshes);

                created[def.name] = go;
            }

            return root;
        }

        static void ConfigureBody(ArticulationBody body, IsaacH1BodyDef def,
                                  IsaacH1RigAsset rig, IsaacH1Agent.ArmatureMode armatureMode)
        {
            var p = rig.physics;

            body.mass = def.mass;
            body.centerOfMass = IsaacH1FrameMap.Pos(def.comIsaac);

            // Isaac's principalAxes is identity, so the tensor is diagonal in the link
            // frame. Under M the Isaac axes x,y,z become Unity z,x,y, so the diagonal
            // permutes to (Iyy, Izz, Ixx).
            // Composed by IsaacH1Inertia so the prefab and any runtime re-apply
            // (a floor, or armatureMode changed on the component) agree exactly.
            Vector3 inertia = IsaacH1Inertia.DiagIsaacToUnity(def.inertiaDiagIsaac);
            Quaternion inertiaRot = Quaternion.identity;

            body.linearDamping = p.linearDamping;
            body.angularDamping = p.angularDamping;
            body.jointFriction = p.jointFriction;
            body.maxLinearVelocity = p.maxLinearVelocity;
            body.maxAngularVelocity = p.maxAngularVelocity;
            body.maxDepenetrationVelocity = p.maxDepenetrationVelocity;
            body.solverIterations = p.solverPositionIterations;
            body.solverVelocityIterations = p.solverVelocityIterations;
            body.useGravity = true;

            if (def.isRoot)
            {
                body.immovable = false;
                body.inertiaTensor = inertia;
                body.inertiaTensorRotation = inertiaRot;
                return;
            }

            var j = def.joint;
            Vector3 axisUnity = IsaacH1FrameMap.Axis(j.axisInChildIsaac).normalized;

            // Fold PhysX's joint armature into the child link inertia. Adding
            // armature * a a^T makes a^T I' a == a^T I a + armature, i.e. the joint-space
            // inertia about THIS axis matches Isaac exactly. Ancestor joints whose axes
            // share a component with a are perturbed slightly - that is the approximation,
            // and it is worth it: without the fold the explicit-PD ratio is 9x worse and
            // the light shoulder links lose all conditioning (RIG_AUDIT.md sections A, C).
            IsaacH1Inertia.Compose(def.inertiaDiagIsaac, false, 0f,
                armatureMode != IsaacH1Agent.ArmatureMode.None,
                axisUnity, ArmatureFor(rig, def, armatureMode), out inertia, out inertiaRot);

            body.inertiaTensor = inertia;
            body.inertiaTensorRotation = inertiaRot;

            body.jointType = ArticulationJointType.RevoluteJoint;
            body.matchAnchors = true;               // parent anchor derived from this one

            // Every USD joint has localPos1 == 0, i.e. the anchor sits exactly on the
            // child link origin (extract_rig.py asserts this).
            body.anchorPosition = Vector3.zero;

            // Unity twists about the anchor frame's local +X. Placing +X at -M*axis is
            // what makes a positive Unity joint angle equal a positive Isaac joint angle,
            // so the agent never flips a sign.
            body.anchorRotation = IsaacH1FrameMap.AnchorRotationForAxis(axisUnity);

            body.twistLock = ArticulationDofLock.LimitedMotion;
            body.swingYLock = ArticulationDofLock.LockedMotion;
            body.swingZLock = ArticulationDofLock.LockedMotion;

            var d = body.xDrive;
            d.lowerLimit = j.lowerRad * Mathf.Rad2Deg;
            d.upperLimit = j.upperRad * Mathf.Rad2Deg;
            d.stiffness = j.stiffness;              // radians convention; see CONTRACT.md
            d.damping = j.damping;
            d.forceLimit = j.effortLimit;           // effort_limit_sim, NOT the URDF value
            d.target = j.defaultPosRad * Mathf.Rad2Deg;
            d.targetVelocity = 0f;
            d.driveType = ArticulationDriveType.Force;
            body.xDrive = d;

            // Isaac left velocity_limit_sim null and the recording exceeds the URDF limit,
            // so the cap is the link angular cap, not the URDF value.
            body.maxJointVelocity = p.maxAngularVelocity;
        }

        /// <summary>Mirrors IsaacH1Agent.ArmatureFor so prefab and runtime agree exactly.</summary>
        static float ArmatureFor(IsaacH1RigAsset rig, IsaacH1BodyDef def,
                                 IsaacH1Agent.ArmatureMode mode)
        {
            if (!def.hasJoint || mode == IsaacH1Agent.ArmatureMode.None) return 0f;
            if (mode == IsaacH1Agent.ArmatureMode.FoldIntoInertia) return def.joint.armature;

            Vector3 a = def.joint.axisInChildIsaac.normalized;
            foreach (var c in rig.bodies)
            {
                if (!c.hasJoint || c.parent != def.name) continue;
                Vector4 q = c.localRotIsaacWxyz;                       // (w, x, y, z)
                Quaternion rot = new Quaternion(q.y, q.z, q.w, q.x);   // -> (x, y, z, w)
                Vector3 ac = (rot * c.joint.axisInChildIsaac).normalized;
                if (Mathf.Abs(Vector3.Dot(a, ac)) > 0.999f) return 0f;
            }
            return def.joint.armature;
        }

        static void AddColliders(GameObject go, IsaacH1BodyDef def, IsaacH1RigAsset rig)
        {
            if (def.colliders == null) return;
            foreach (var c in def.colliders)
            {
                // Colliders live on the link object itself, which is unscaled, so the box
                // extents are exactly the numbers below - no cooking-time rescale.
                var box = go.AddComponent<BoxCollider>();
                box.center = IsaacH1FrameMap.Pos(c.centerIsaac);
                Vector3 s = IsaacH1FrameMap.Pos(c.sizeIsaac);
                box.size = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
                // PhysX default Isaac ran with; Unity's project default is 0.01.
                box.contactOffset = rig.physics.contactOffset;
            }
        }

        /// <summary>
        /// Visuals for one link. Prefers the ORIGINAL Isaac mesh from
        /// <see cref="IsaacH1MeshLibrary"/>; falls back to primitive proxies built from
        /// the URDF collision shapes when no library is supplied or the link has no
        /// entry. Either way the visuals are non-colliding and massless.
        /// </summary>
        static void AddVisuals(GameObject go, IsaacH1BodyDef def, IsaacH1MeshLibrary meshes)
        {
            Mesh mesh = meshes != null ? meshes.Find(def.name) : null;
            if (mesh != null)
            {
                // The blob is already link-local and in Unity coordinates, so the holder
                // sits at identity - no transform may be applied here.
                var vis = new GameObject("Visual");
                vis.transform.SetParent(go.transform, false);
                vis.transform.localPosition = Vector3.zero;
                vis.transform.localRotation = Quaternion.identity;
                vis.transform.localScale = Vector3.one;
                var mf = vis.AddComponent<MeshFilter>();
                mf.sharedMesh = mesh;
                var mr = vis.AddComponent<MeshRenderer>();
                if (meshes.material != null) mr.sharedMaterial = meshes.material;
                return;
            }
            AddVisualProxies(go, def);
        }

        static void AddVisualProxies(GameObject go, IsaacH1BodyDef def)
        {
            if (def.visuals == null || def.visuals.Length == 0) return;

            var holder = new GameObject("Visual");
            holder.transform.SetParent(go.transform, false);

            foreach (var v in def.visuals)
            {
                PrimitiveType prim;
                Vector3 scale;
                switch (v.kind)
                {
                    case "box":
                        prim = PrimitiveType.Cube;
                        Vector3 s = IsaacH1FrameMap.Pos(v.size);
                        scale = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
                        break;
                    case "cylinder":
                        // Unity's cylinder mesh is 2 units tall along local Y, radius 0.5.
                        prim = PrimitiveType.Cylinder;
                        scale = new Vector3(v.radius * 2f, v.length * 0.5f, v.radius * 2f);
                        break;
                    case "sphere":
                        prim = PrimitiveType.Sphere;
                        scale = Vector3.one * (v.radius * 2f);
                        break;
                    default:
                        continue;
                }

                var g = GameObject.CreatePrimitive(prim);
                g.name = $"vis_{v.kind}";
                var col = g.GetComponent<Collider>();
                if (col != null) UnityEngine.Object.DestroyImmediate(col);

                g.transform.SetParent(holder.transform, false);
                g.transform.localPosition = IsaacH1FrameMap.Pos(v.originIsaac);

                // The URDF rpy is expressed in the Isaac frame, so build the quaternion
                // there and map it across. A URDF cylinder's axis is its local Z whereas
                // Unity's cylinder mesh runs along local Y, hence the extra +90 about X.
                Quaternion rot = IsaacH1FrameMap.RotFromWxyz(
                    RpyW(v.rpy), RpyX(v.rpy), RpyY(v.rpy), RpyZ(v.rpy));
                g.transform.localRotation = v.kind == "cylinder"
                    ? rot * Quaternion.Euler(90f, 0f, 0f)
                    : rot;
                g.transform.localScale = scale;
            }
        }

        // URDF fixed-axis roll-pitch-yaw -> quaternion components, in the Isaac frame.
        static void Rpy(Vector3 rpy, out float w, out float x, out float y, out float z)
        {
            float cr = Mathf.Cos(rpy.x * 0.5f), sr = Mathf.Sin(rpy.x * 0.5f);
            float cp = Mathf.Cos(rpy.y * 0.5f), sp = Mathf.Sin(rpy.y * 0.5f);
            float cy = Mathf.Cos(rpy.z * 0.5f), sy = Mathf.Sin(rpy.z * 0.5f);
            w = cr * cp * cy + sr * sp * sy;
            x = sr * cp * cy - cr * sp * sy;
            y = cr * sp * cy + sr * cp * sy;
            z = cr * cp * sy - sr * sp * cy;
        }

        static float RpyW(Vector3 r) { Rpy(r, out var w, out _, out _, out _); return w; }
        static float RpyX(Vector3 r) { Rpy(r, out _, out var x, out _, out _); return x; }
        static float RpyY(Vector3 r) { Rpy(r, out _, out _, out var y, out _); return y; }
        static float RpyZ(Vector3 r) { Rpy(r, out _, out _, out _, out var z); return z; }
    }
}
