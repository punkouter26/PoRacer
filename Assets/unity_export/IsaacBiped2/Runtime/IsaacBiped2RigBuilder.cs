using System;
using System.Collections.Generic;
using System.Globalization;
using System.Xml;
using UnityEngine;

namespace IsaacBiped2
{
    /// <summary>
    /// Builds the biped ArticulationBody rig straight from <c>robot/biped.urdf</c> — primitives only,
    /// so the rig the policy was trained against is the rig that runs, with no importer in between.
    ///
    /// Adapted from the spider builder, with two differences the biped forces:
    /// <list type="bullet">
    /// <item>the torso and feet are <b>boxes</b>, not spheres or cylinders, so BoxCollider is supported;</item>
    /// <item>the four hip hub links carry a DOF but no geometry at all, so a missing
    /// <c>collision</c> element is normal and must not throw.</item>
    /// </list>
    ///
    /// Frame map (identical to the URDF Importer): ROS/Isaac (x, y, z) to Unity (-y, z, x).
    /// Joint axis: the revolute joint rotates about the anchor's X axis, and anchor X is aligned with
    /// <b>-M*axis</b> so a positive Unity joint angle is the same physical rotation as a positive
    /// Isaac joint angle — without the negation the handedness flip inverts every joint sign.
    /// </summary>
    public static class IsaacBiped2RigBuilder
    {
        public enum ColliderShape { None, Sphere, Cylinder, Box }

        /// <summary>Everything the agent needs from the URDF, per link, in Unity conventions.</summary>
        public sealed class LinkSpec
        {
            public string Name;
            public string Parent;          // null for the root
            public string JointName;       // null for the root
            public Vector3 JointOriginUnity;
            public Quaternion JointRotationUnity;
            public Vector3 JointAxisUnity;  // already negated (-M*axis)
            public float LowerLimitRad, UpperLimitRad, EffortLimit, VelocityLimit;
            public float Mass;
            public Vector3 CenterOfMassUnity;
            public Vector3 InertiaPrincipalUnity;
            public Quaternion InertiaRotationUnity;
            public ColliderShape Shape;
            public float ColliderRadius, ColliderLength;
            public Vector3 ColliderSizeUnity;   // boxes only, already axis-mapped
            public Vector3 ColliderOriginUnity;
            public Quaternion ColliderRotationUnity;
        }

        /// <summary>Parse a URDF document into link specs, root first then breadth-first.</summary>
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
                spec.Mass = ReadFloat(inertial.SelectSingleNode("mass").Attributes["value"].Value);
                XmlNode inertia = inertial.SelectSingleNode("inertia");
                float ixx = ReadFloat(inertia.Attributes["ixx"].Value);
                float iyy = ReadFloat(inertia.Attributes["iyy"].Value);
                float izz = ReadFloat(inertia.Attributes["izz"].Value);
                spec.CenterOfMassUnity = RosToUnity(ReadVec(inertialOrigin, "xyz"));
                spec.InertiaRotationUnity = RosRpyToUnity(ReadVec(inertialOrigin, "rpy"));
                // Unity x of the inertial frame is the image of ROS -y, y of ROS z, z of ROS x.
                spec.InertiaPrincipalUnity = new Vector3(iyy, izz, ixx);
                ReadCollision(link, spec);
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
                spec.JointAxisUnity = -RosToUnity(ReadVec(joint.SelectSingleNode("axis"), "xyz"));
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
        /// A link may legitimately have no collision geometry — the biped's four hip hubs exist only
        /// to carry the yaw and roll DOFs — so absence is recorded as <see cref="ColliderShape.None"/>
        /// rather than treated as malformed URDF.
        /// </summary>
        private static void ReadCollision(XmlNode link, LinkSpec spec)
        {
            XmlNode collision = link.SelectSingleNode("collision");
            if (collision == null)
            {
                spec.Shape = ColliderShape.None;
                return;
            }
            XmlNode geometry = collision.SelectSingleNode("geometry").FirstChild;
            XmlNode collisionOrigin = collision.SelectSingleNode("origin");
            spec.ColliderOriginUnity = RosToUnity(ReadVec(collisionOrigin, "xyz"));
            spec.ColliderRotationUnity = RosRpyToUnity(ReadVec(collisionOrigin, "rpy"));
            switch (geometry.Name)
            {
                case "sphere":
                    spec.Shape = ColliderShape.Sphere;
                    spec.ColliderRadius = ReadFloat(geometry.Attributes["radius"].Value);
                    break;
                case "cylinder":
                    spec.Shape = ColliderShape.Cylinder;
                    spec.ColliderRadius = ReadFloat(geometry.Attributes["radius"].Value);
                    spec.ColliderLength = ReadFloat(geometry.Attributes["length"].Value);
                    break;
                case "box":
                    spec.Shape = ColliderShape.Box;
                    Vector3 sizeRos = ReadVec(geometry, "size");
                    // Box extents are unsigned, so the axis map applies without the sign flip.
                    spec.ColliderSizeUnity = new Vector3(sizeRos.y, sizeRos.z, sizeRos.x);
                    break;
                default:
                    throw new NotSupportedException(
                        $"link {spec.Name}: unsupported collision geometry <{geometry.Name}>");
            }
        }

        /// <summary>
        /// Build the rig under a new root GameObject. Per-body physics values are left at Unity
        /// defaults; the agent's ApplyPhysicsSettings() sets them from the export report.
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

        private static void AddCollider(GameObject link, LinkSpec spec, PhysicsMaterial physicsMaterial)
        {
            if (spec.Shape == ColliderShape.None)
            {
                return;
            }
            var holder = new GameObject("col_" + spec.Name);
            holder.transform.SetParent(link.transform, false);
            holder.transform.localPosition = spec.ColliderOriginUnity;
            holder.transform.localRotation = spec.ColliderRotationUnity;
            holder.transform.localScale = Vector3.one;
            Collider collider;
            switch (spec.Shape)
            {
                case ColliderShape.Sphere:
                {
                    var sphere = holder.AddComponent<SphereCollider>();
                    sphere.radius = spec.ColliderRadius;
                    collider = sphere;
                    break;
                }
                case ColliderShape.Box:
                {
                    var box = holder.AddComponent<BoxCollider>();
                    box.size = spec.ColliderSizeUnity;
                    collider = box;
                    break;
                }
                default:
                {
                    // Isaac turns the URDF cylinder into a capsule; height = length + radius puts the
                    // rounded tip where the flat rim was, so the standing height still matches.
                    var capsule = holder.AddComponent<CapsuleCollider>();
                    capsule.radius = spec.ColliderRadius;
                    capsule.height = spec.ColliderLength + spec.ColliderRadius;
                    capsule.direction = 1; // ROS cylinder axis z to Unity y
                    collider = capsule;
                    break;
                }
            }
            collider.sharedMaterial = physicsMaterial;
            AddVisual(holder, spec);
        }

        /// <summary>Visual-only primitive under the collider holder — colliders do not render.
        /// Unity's sphere is radius 0.5, its capsule radius 0.5 / height 2, its cube edge 1.</summary>
        private static void AddVisual(GameObject holder, LinkSpec spec)
        {
            PrimitiveType type;
            switch (spec.Shape)
            {
                case ColliderShape.Sphere: type = PrimitiveType.Sphere; break;
                case ColliderShape.Box: type = PrimitiveType.Cube; break;
                default: type = PrimitiveType.Capsule; break;
            }
            GameObject primitive = GameObject.CreatePrimitive(type);
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
            switch (spec.Shape)
            {
                case ColliderShape.Sphere:
                    primitive.transform.localScale = new Vector3(diameter, diameter, diameter);
                    break;
                case ColliderShape.Box:
                    primitive.transform.localScale = spec.ColliderSizeUnity;
                    break;
                default:
                    primitive.transform.localScale =
                        new Vector3(diameter, (spec.ColliderLength + spec.ColliderRadius) * 0.5f, diameter);
                    break;
            }
        }

        // ------------------------------------------------------------------ conversions
        /// <summary>ROS/Isaac (x, y, z) to Unity (-y, z, x).</summary>
        public static Vector3 RosToUnity(Vector3 ros) => new Vector3(-ros.y, ros.z, ros.x);

        /// <summary>Unity (x, y, z) to ROS/Isaac (z, -x, y).</summary>
        public static Vector3 UnityToRos(Vector3 unity) => new Vector3(unity.z, -unity.x, unity.y);

        /// <summary>ROS roll-pitch-yaw (R = Rz(yaw)*Ry(pitch)*Rx(roll)) to a Unity quaternion.</summary>
        public static Quaternion RosRpyToUnity(Vector3 rpy)
        {
            Vector3 forward = RosToUnity(RotateRpy(rpy, UnityToRos(Vector3.forward)));
            Vector3 up = RosToUnity(RotateRpy(rpy, UnityToRos(Vector3.up)));
            return Quaternion.LookRotation(forward, up);
        }

        private static Vector3 RotateRpy(Vector3 rpy, Vector3 v)
        {
            float cr = Mathf.Cos(rpy.x), sr = Mathf.Sin(rpy.x);
            float cp = Mathf.Cos(rpy.y), sp = Mathf.Sin(rpy.y);
            float cy = Mathf.Cos(rpy.z), sy = Mathf.Sin(rpy.z);
            var a = new Vector3(v.x, cr * v.y - sr * v.z, sr * v.y + cr * v.z);
            var b = new Vector3(cp * a.x + sp * a.z, a.y, -sp * a.x + cp * a.z);
            return new Vector3(cy * b.x - sy * b.y, sy * b.x + cy * b.y, b.z);
        }

        private static Vector3 ReadVec(XmlNode node, string attribute)
        {
            if (node == null || node.Attributes[attribute] == null)
            {
                return Vector3.zero;
            }
            string[] parts = node.Attributes[attribute].Value.Split(
                new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return new Vector3(ReadFloat(parts[0]), ReadFloat(parts[1]), ReadFloat(parts[2]));
        }

        private static float ReadFloat(string s) => float.Parse(s, CultureInfo.InvariantCulture);
    }
}
