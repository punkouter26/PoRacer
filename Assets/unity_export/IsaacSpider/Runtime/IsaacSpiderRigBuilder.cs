using System;
using System.Collections.Generic;
using System.Globalization;
using System.Xml;
using UnityEngine;

namespace IsaacSpider
{
    /// <summary>
    /// Builds the spider ArticulationBody rig straight from <c>robot/spider.urdf</c> — primitives only,
    /// no scaled parents, no importer Controller/JointControl/FKRobot. Used by the editor prefab builder
    /// when the URDF Importer package is absent, and by the PlayMode tests so they never depend on a
    /// prefab having been built.
    ///
    /// Frame map (identical to the URDF Importer): ROS/Isaac (x, y, z) → Unity (-y, z, x).
    /// Quaternions: ROS (x, y, z, w) → Unity (y, -z, -x, w).
    /// Joint axis: the revolute joint rotates about the anchor's X axis; the anchor X is aligned with
    /// <b>-M·axis</b> so a positive Unity joint angle is the same physical rotation as a positive Isaac
    /// joint angle (the handedness flip would otherwise negate every joint sign).
    /// </summary>
    public static class IsaacSpiderRigBuilder
    {
        /// <summary>Everything the agent needs from the URDF, per link, in Unity conventions.</summary>
        public sealed class LinkSpec
        {
            public string Name;
            public string Parent;          // null for the root
            public string JointName;       // null for the root
            public Vector3 JointOriginUnity;
            public Quaternion JointRotationUnity;
            public Vector3 JointAxisUnity;  // already negated (-M·axis)
            public float LowerLimitRad, UpperLimitRad, EffortLimit, VelocityLimit;
            public float Mass;
            public Vector3 CenterOfMassUnity;
            public Vector3 InertiaPrincipalUnity;
            public Quaternion InertiaRotationUnity;
            public string ColliderShape;   // "sphere" | "cylinder"
            public float ColliderRadius, ColliderLength;
            public Vector3 ColliderOriginUnity;
            public Quaternion ColliderRotationUnity;
        }

        /// <summary>Parse a URDF document into link specs (root first, then in document order).</summary>
        public static List<LinkSpec> Parse(string urdfXml)
        {
            var doc = new XmlDocument();
            doc.LoadXml(urdfXml);
            var specs = new Dictionary<string, LinkSpec>();
            var order = new List<string>();
            foreach (XmlNode link in doc.SelectNodes("/robot/link"))
            {
                var spec = new LinkSpec { Name = link.Attributes["name"].Value };
                XmlNode inertial = link.SelectSingleNode("inertial");
                XmlNode inertialOrigin = inertial.SelectSingleNode("origin");
                Vector3 comRos = ReadVec(inertialOrigin, "xyz");
                Vector3 rpy = ReadVec(inertialOrigin, "rpy");
                spec.Mass = ReadFloat(inertial.SelectSingleNode("mass").Attributes["value"].Value);
                XmlNode inertia = inertial.SelectSingleNode("inertia");
                float ixx = ReadFloat(inertia.Attributes["ixx"].Value);
                float iyy = ReadFloat(inertia.Attributes["iyy"].Value);
                float izz = ReadFloat(inertia.Attributes["izz"].Value);
                spec.CenterOfMassUnity = RosToUnity(comRos);
                spec.InertiaRotationUnity = RosRpyToUnity(rpy);
                // Unity x of the inertial frame is the image of ROS -y, y of ROS z, z of ROS x.
                spec.InertiaPrincipalUnity = new Vector3(iyy, izz, ixx);
                XmlNode collision = link.SelectSingleNode("collision");
                XmlNode geometry = collision.SelectSingleNode("geometry").FirstChild;
                XmlNode collisionOrigin = collision.SelectSingleNode("origin");
                spec.ColliderOriginUnity = RosToUnity(ReadVec(collisionOrigin, "xyz"));
                spec.ColliderRotationUnity = RosRpyToUnity(ReadVec(collisionOrigin, "rpy"));
                spec.ColliderShape = geometry.Name;
                spec.ColliderRadius = ReadFloat(geometry.Attributes["radius"].Value);
                spec.ColliderLength = geometry.Attributes["length"] != null ? ReadFloat(geometry.Attributes["length"].Value) : 0f;
                specs[spec.Name] = spec;
                order.Add(spec.Name);
            }
            foreach (XmlNode joint in doc.SelectNodes("/robot/joint"))
            {
                string child = joint.SelectSingleNode("child").Attributes["link"].Value;
                LinkSpec spec = specs[child];
                spec.JointName = joint.Attributes["name"].Value;
                spec.Parent = joint.SelectSingleNode("parent").Attributes["link"].Value;
                XmlNode origin = joint.SelectSingleNode("origin");
                spec.JointOriginUnity = RosToUnity(ReadVec(origin, "xyz"));
                spec.JointRotationUnity = RosRpyToUnity(ReadVec(origin, "rpy"));
                Vector3 axisRos = ReadVec(joint.SelectSingleNode("axis"), "xyz");
                spec.JointAxisUnity = -RosToUnity(axisRos);
                XmlNode limit = joint.SelectSingleNode("limit");
                spec.LowerLimitRad = ReadFloat(limit.Attributes["lower"].Value);
                spec.UpperLimitRad = ReadFloat(limit.Attributes["upper"].Value);
                spec.EffortLimit = ReadFloat(limit.Attributes["effort"].Value);
                spec.VelocityLimit = ReadFloat(limit.Attributes["velocity"].Value);
            }
            var result = new List<LinkSpec>(order.Count);
            for (int index = 0; index < order.Count; index++)
            {
                if (specs[order[index]].Parent == null)
                {
                    result.Add(specs[order[index]]);
                }
            }
            // Breadth-first so every parent exists before its child.
            for (int cursor = 0; cursor < result.Count; cursor++)
            {
                for (int index = 0; index < order.Count; index++)
                {
                    LinkSpec candidate = specs[order[index]];
                    if (candidate.Parent == result[cursor].Name)
                    {
                        result.Add(candidate);
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Build the rig under a new root GameObject. Colliders are CapsuleColliders on unscaled child
        /// objects (height = length + radius: the rounded tip touches the ground where the flat cylinder rim
        /// did at the rest angle). Per-body physics values are left at Unity defaults here; the agent's
        /// ApplyPhysicsSettings() sets them from env.yaml.
        /// </summary>
        public static GameObject Build(string urdfXml, string rootName, PhysicsMaterial physicsMaterial)
        {
            List<LinkSpec> specs = Parse(urdfXml);
            var bodies = new Dictionary<string, ArticulationBody>();
            GameObject root = null;
            for (int index = 0; index < specs.Count; index++)
            {
                LinkSpec spec = specs[index];
                var go = new GameObject(spec.Name);
                if (spec.Parent == null)
                {
                    go.name = rootName;
                    root = go;
                }
                else
                {
                    go.transform.SetParent(bodies[spec.Parent].transform, false);
                    go.transform.localPosition = spec.JointOriginUnity;
                    go.transform.localRotation = spec.JointRotationUnity;
                }
                ArticulationBody body = go.AddComponent<ArticulationBody>();
                body.mass = spec.Mass;
                body.automaticCenterOfMass = false;
                body.automaticInertiaTensor = false;
                body.centerOfMass = spec.CenterOfMassUnity;
                body.inertiaTensor = spec.InertiaPrincipalUnity;
                body.inertiaTensorRotation = spec.InertiaRotationUnity;
                if (spec.Parent != null)
                {
                    body.jointType = ArticulationJointType.RevoluteJoint;
                    body.anchorPosition = Vector3.zero;
                    body.anchorRotation = Quaternion.FromToRotation(Vector3.right, spec.JointAxisUnity);
                    body.matchAnchors = true;
                    body.twistLock = ArticulationDofLock.LimitedMotion;
                    ArticulationDrive drive = body.xDrive;
                    drive.lowerLimit = spec.LowerLimitRad * Mathf.Rad2Deg;
                    drive.upperLimit = spec.UpperLimitRad * Mathf.Rad2Deg;
                    drive.stiffness = 0f;
                    drive.damping = 0f;
                    drive.forceLimit = spec.EffortLimit;
                    drive.driveType = ArticulationDriveType.Force;
                    drive.target = 0f;
                    drive.targetVelocity = 0f;
                    body.xDrive = drive;
                }
                AddCollider(go, spec, physicsMaterial);
                bodies[spec.Name] = body;
            }
            root.name = rootName;
            return root;
        }

        /// <summary>Remove the URDF Importer's scaled convex-mesh geometry and put primitive colliders on the unscaled link.</summary>
        public static int ReplaceCollidersOnImportedRig(GameObject importedRoot, string urdfXml, PhysicsMaterial physicsMaterial)
        {
            List<LinkSpec> specs = Parse(urdfXml);
            var byName = new Dictionary<string, LinkSpec>();
            for (int index = 0; index < specs.Count; index++)
            {
                byName[specs[index].Name] = specs[index];
            }
            int replaced = 0;
            ArticulationBody[] bodies = importedRoot.GetComponentsInChildren<ArticulationBody>(true);
            for (int index = 0; index < bodies.Length; index++)
            {
                if (!byName.TryGetValue(bodies[index].name, out LinkSpec spec))
                {
                    continue;
                }
                Collider[] existing = bodies[index].GetComponentsInChildren<Collider>(true);
                for (int colliderIndex = 0; colliderIndex < existing.Length; colliderIndex++)
                {
                    // Only this link's colliders: the importer nests them under Collisions/unnamed.
                    if (existing[colliderIndex].GetComponentInParent<ArticulationBody>() == bodies[index])
                    {
                        UnityEngine.Object.DestroyImmediate(existing[colliderIndex].gameObject);
                    }
                }
                AddCollider(bodies[index].gameObject, spec, physicsMaterial);
                replaced++;
            }
            return replaced;
        }

        private static void AddCollider(GameObject link, LinkSpec spec, PhysicsMaterial physicsMaterial)
        {
            var holder = new GameObject("col_" + spec.Name);
            holder.transform.SetParent(link.transform, false);
            holder.transform.localPosition = spec.ColliderOriginUnity;
            holder.transform.localRotation = spec.ColliderRotationUnity;
            holder.transform.localScale = Vector3.one;
            Collider collider;
            if (spec.ColliderShape == "sphere")
            {
                var sphere = holder.AddComponent<SphereCollider>();
                sphere.radius = spec.ColliderRadius;
                collider = sphere;
            }
            else
            {
                var capsule = holder.AddComponent<CapsuleCollider>();
                capsule.radius = spec.ColliderRadius;
                capsule.height = spec.ColliderLength + spec.ColliderRadius;
                capsule.direction = 1; // ROS cylinder axis z → Unity y
                collider = capsule;
            }
            collider.sharedMaterial = physicsMaterial;
            AddVisual(holder, spec);
        }

        /// <summary>Visual-only primitive under the collider holder (the rig had none: colliders do not render).
        /// Unity's sphere mesh is radius 0.5, its capsule is radius 0.5 / height 2, so scale = diameter and half-height.</summary>
        private static void AddVisual(GameObject holder, LinkSpec spec)
        {
            bool sphere = spec.ColliderShape == "sphere";
            GameObject primitive = GameObject.CreatePrimitive(sphere ? PrimitiveType.Sphere : PrimitiveType.Capsule);
            primitive.name = "vis_" + spec.Name;
            Collider primitiveCollider = primitive.GetComponent<Collider>();
            if (Application.isPlaying)
            {
                UnityEngine.Object.Destroy(primitiveCollider);
            }
            else
            {
                UnityEngine.Object.DestroyImmediate(primitiveCollider);
            }
            primitive.transform.SetParent(holder.transform, false);
            primitive.transform.localPosition = Vector3.zero;
            primitive.transform.localRotation = Quaternion.identity;
            float diameter = spec.ColliderRadius * 2f;
            primitive.transform.localScale = sphere
                ? new Vector3(diameter, diameter, diameter)
                : new Vector3(diameter, (spec.ColliderLength + spec.ColliderRadius) * 0.5f, diameter);
        }

        // ------------------------------------------------------------------ conversions
        /// <summary>ROS/Isaac (x, y, z) → Unity (-y, z, x).</summary>
        public static Vector3 RosToUnity(Vector3 ros) => new Vector3(-ros.y, ros.z, ros.x);

        /// <summary>Unity (x, y, z) → ROS/Isaac (z, -x, y).</summary>
        public static Vector3 UnityToRos(Vector3 unity) => new Vector3(unity.z, -unity.x, unity.y);

        /// <summary>ROS roll-pitch-yaw (R = Rz(yaw)·Ry(pitch)·Rx(roll)) → Unity quaternion via M R Mᵀ.</summary>
        public static Quaternion RosRpyToUnity(Vector3 rpy)
        {
            // Rotate the Unity basis vectors through the ROS matrix and rebuild the rotation.
            Vector3 forward = RosToUnity(RotateRpy(rpy, UnityToRos(Vector3.forward)));
            Vector3 up = RosToUnity(RotateRpy(rpy, UnityToRos(Vector3.up)));
            return Quaternion.LookRotation(forward, up);
        }

        private static Vector3 RotateRpy(Vector3 rpy, Vector3 v)
        {
            float cr = Mathf.Cos(rpy.x), sr = Mathf.Sin(rpy.x);
            float cp = Mathf.Cos(rpy.y), sp = Mathf.Sin(rpy.y);
            float cy = Mathf.Cos(rpy.z), sy = Mathf.Sin(rpy.z);
            // Rx
            var a = new Vector3(v.x, cr * v.y - sr * v.z, sr * v.y + cr * v.z);
            // Ry
            var b = new Vector3(cp * a.x + sp * a.z, a.y, -sp * a.x + cp * a.z);
            // Rz
            return new Vector3(cy * b.x - sy * b.y, sy * b.x + cy * b.y, b.z);
        }

        private static Vector3 ReadVec(XmlNode node, string attribute)
        {
            if (node == null || node.Attributes[attribute] == null)
            {
                return Vector3.zero;
            }
            string[] parts = node.Attributes[attribute].Value.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return new Vector3(ReadFloat(parts[0]), ReadFloat(parts[1]), ReadFloat(parts[2]));
        }

        private static float ReadFloat(string s) => float.Parse(s, CultureInfo.InvariantCulture);
    }
}
