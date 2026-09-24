using Mujoco;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Builds the MuJoCo worm as org.mujoco components, from worm_rig.json, at race start.
    ///
    /// WHY NOT THE MJCF IMPORTER. worm.xml could be pushed through MjcfImporter.ImportString
    /// (runtime) or MjImporterWithAssets (editor), but both are the wrong tool here:
    ///   * the importer's hinge-axis default is Unity +X, and MuJoCo's own writer drops
    ///     default-valued axes - the defect training/mojucuboy/make_unity_mjcf.py works
    ///     around. worm.xml authors its axes, but any round trip through MuJoCo's writer
    ///     (ImportFile does one) silently turns every yaw hinge into a pitch hinge;
    ///   * the importer attaches the built-in "Standard" shader, which renders magenta in
    ///     URP, and pulls in worm.xml's own floor plane, which would then have to be
    ///     found and removed again;
    ///   * an editor-time import would bake a prefab that must then be kept in step with
    ///     worm.xml by hand, and the MjScene singleton rules mean it could not simply sit
    ///     in the scene anyway (it has to be born in the same frame as its world).
    /// Building the handful of components directly is ~150 lines, needs no editor, cannot
    /// lose an axis, and uses the same worm_rig.json the PhysX worm is built from - so one
    /// file defines both bodies. The generated MJCF is dumped (WormRaceSettings.DumpMujocoMjcf)
    /// so it can be diffed against worm.xml; INSTALL.md lists what must match.
    ///
    /// FRAMES. The plug-in maps MuJoCo (x, y, z) to Unity (x, z, y). Every local offset and
    /// axis below goes through that swap, and only the root gets a rotation: a yaw that
    /// points MuJoCo +x (the head) down the lane. The policy sees only body-frame
    /// quantities, so this yaw is invisible to it, exactly like the +/-45 degree spawn yaw
    /// it trained with.
    /// </summary>
    internal static class MujocoWormBuilder
    {
        /// <summary>
        /// Builds one MuJoCo worm. Any number can be built in the same frame under the same
        /// MjScene: the plug-in gives every element a unique generated name and the view
        /// resolves ids and qpos/qvel/ctrl addresses per component after the compile, so the
        /// worms never share an index; each worm excludes only its own adjacent segments.
        /// </summary>
        public static GameObject Build(WormRig rig, WormPilot pilot, string rootName,
                                       Vector3 rootOrigin, Vector3 laneForward, Material material,
                                       Mesh segmentMesh, out Transform[] segmentGeoms)
        {
            var root = new GameObject(rootName);
            root.transform.SetPositionAndRotation(rootOrigin, Quaternion.identity);

            // Unity +X (= MuJoCo body +x under the swap) turned onto the lane direction.
            Quaternion facing = Quaternion.LookRotation(laneForward, Vector3.up)
                              * Quaternion.FromToRotation(Vector3.right, Vector3.forward);

            int bodyCount = rig.Bodies.Length;
            var bodyObjects = new GameObject[bodyCount];
            var bodies = new MjBody[bodyCount];
            var segments = new MjBody[WormContract.SEGMENT_COUNT];
            var hinges = new MjHingeJoint[WormContract.ACTION_SIZE];
            segmentGeoms = new Transform[WormContract.SEGMENT_COUNT];

            for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
            {
                WormRig.BodyDef def = rig.Bodies[bodyIndex];
                Transform parent = def.ParentIndex < 0 ? root.transform : bodyObjects[def.ParentIndex].transform;

                var bodyObject = new GameObject(def.Name);
                bodyObject.transform.SetParent(parent, false);
                bodyObject.transform.localPosition = WormFrames.UnityFromPlugin(def.PositionMuJoCo);
                bodyObject.transform.localRotation = def.ParentIndex < 0 ? facing : Quaternion.identity;
                MjBody body = bodyObject.AddComponent<MjBody>();

                if (def.ParentIndex < 0)
                {
                    var freeJoint = new GameObject(def.Name + "_free");
                    freeJoint.transform.SetParent(bodyObject.transform, false);
                    freeJoint.AddComponent<MjFreeJoint>();
                }

                if (def.HasCapsule)
                {
                    Transform geom = AddCapsule(bodyObject, def, rig.Friction, material, segmentMesh);
                    segmentGeoms[def.SegmentIndex] = geom;
                    segments[def.SegmentIndex] = body;
                }
                else
                {
                    AddInertial(bodyObject, def);
                }

                if (def.HasJoint)
                {
                    hinges[def.JointActionIndex] = AddHinge(bodyObject, def, rig);
                }

                bodyObjects[bodyIndex] = bodyObject;
                bodies[bodyIndex] = body;
            }

            AddExcludes(root, rig, bodies);
            MjActuator[] actuators = AddActuators(root, rig, hinges);

            MujocoWormView view = root.AddComponent<MujocoWormView>();
            view.Bind(segments, hinges, actuators, pilot, laneForward, rig.NoseOffset);
            return root;
        }

        private static Transform AddCapsule(GameObject bodyObject, WormRig.BodyDef def, float friction,
                                            Material material, Mesh segmentMesh)
        {
            var geomObject = new GameObject(def.Name + "_geom");
            geomObject.transform.SetParent(bodyObject.transform, false);
            geomObject.transform.localPosition = Vector3.zero;
            // MjCapsuleShape runs along the geom's local +Y; the rig's capsules run along
            // MuJoCo x, which the swap leaves as Unity local +X.
            geomObject.transform.localRotation = Quaternion.FromToRotation(
                Vector3.up, WormFrames.UnityFromPlugin(Vector3.right));

            MjGeom geom = geomObject.AddComponent<MjGeom>();
            geom.ShapeType = MjShapeComponent.ShapeTypes.Capsule;
            geom.Capsule.Radius = def.CapsuleRadius;
            geom.Capsule.HalfHeight = def.CapsuleHalfLength;
            geom.Mass = def.Mass;
            MujocoWorldBuilder.ApplyContact(geom, friction);

            // Render child, not an MjMeshFilter: the shared capsule mesh and the shared
            // blue material, so all five segments batch together.
            var visual = new GameObject("visual");
            visual.transform.SetParent(geomObject.transform, false);
            visual.AddComponent<MeshFilter>().sharedMesh = segmentMesh;
            visual.AddComponent<MeshRenderer>().sharedMaterial = material;
            return geomObject.transform;
        }

        private static void AddInertial(GameObject bodyObject, WormRig.BodyDef def)
        {
            var inertialObject = new GameObject(def.Name + "_inertial");
            inertialObject.transform.SetParent(bodyObject.transform, false);
            MjInertial inertial = inertialObject.AddComponent<MjInertial>();
            inertial.Mass = def.Mass;
            inertial.DiagInertia = new Vector3(def.PointInertia, def.PointInertia, def.PointInertia);
        }

        private static MjHingeJoint AddHinge(GameObject bodyObject, WormRig.BodyDef def, WormRig rig)
        {
            var jointObject = new GameObject(def.JointName);
            jointObject.transform.SetParent(bodyObject.transform, false);
            jointObject.transform.localPosition = WormFrames.UnityFromPlugin(def.JointAnchorMuJoCo);
            // MjHingeJoint's axis is its transform's local +X (MjEngineTool.PositionAxisToMjcf).
            jointObject.transform.localRotation = Quaternion.FromToRotation(
                Vector3.right, WormFrames.UnityFromPlugin(def.JointAxisMuJoCo));

            MjHingeJoint hinge = jointObject.AddComponent<MjHingeJoint>();
            float rangeDegrees = rig.JointRangeRad * Mathf.Rad2Deg;
            hinge.RangeLower = -rangeDegrees;
            hinge.RangeUpper = rangeDegrees;

            // worm.xml <default><joint>: limited, damping 1.0, armature 0.01. Everything
            // else (limit solref/solimp, no spring, no frictionloss) is MuJoCo's default,
            // which is what MjJointSettings.Default writes.
            MjJointSettings jointSettings = hinge.Settings;
            jointSettings.Armature = rig.Armature;
            jointSettings.Spring.Damping = rig.JointDamping;
            jointSettings.Solver.Limited = true;
            hinge.Settings = jointSettings;
            return hinge;
        }

        private static void AddExcludes(GameObject root, WormRig rig, MjBody[] bodies)
        {
            var excludes = new GameObject("excludes");
            excludes.transform.SetParent(root.transform, false);
            for (int bodyIndex = 0; bodyIndex < rig.Bodies.Length; bodyIndex++)
            {
                if (!rig.IsAdjacentSegmentPair(bodyIndex, out int parentSegment))
                {
                    continue;
                }
                var excludeObject = new GameObject($"exclude_{rig.Bodies[parentSegment].Name}_{rig.Bodies[bodyIndex].Name}");
                excludeObject.transform.SetParent(excludes.transform, false);
                MjExclude exclude = excludeObject.AddComponent<MjExclude>();
                exclude.Body1 = bodies[parentSegment];
                exclude.Body2 = bodies[bodyIndex];
            }
        }

        private static MjActuator[] AddActuators(GameObject root, WormRig rig, MjHingeJoint[] hinges)
        {
            var actuatorRoot = new GameObject("actuators");
            actuatorRoot.transform.SetParent(root.transform, false);
            var actuators = new MjActuator[WormContract.ACTION_SIZE];
            // Sibling order = action order: MjScene sorts actuators by sibling index.
            for (int actionIndex = 0; actionIndex < WormContract.ACTION_SIZE; actionIndex++)
            {
                var actuatorObject = new GameObject("act_" + WormContract.ActionOrder[actionIndex]);
                actuatorObject.transform.SetParent(actuatorRoot.transform, false);
                MjActuator actuator = actuatorObject.AddComponent<MjActuator>();
                actuator.Type = MjActuator.ActuatorType.Position;
                actuator.Joint = hinges[actionIndex];
                // worm.xml <default><position>: kp 30, ctrlrange +/-45 deg (radians, since a
                // position servo's ctrl is the joint's own qpos unit), forcerange +/-12.
                actuator.CommonParams.CtrlLimited = true;
                actuator.CommonParams.CtrlRange = new Vector2(-rig.JointRangeRad, rig.JointRangeRad);
                actuator.CommonParams.ForceLimited = true;
                actuator.CommonParams.ForceRange = new Vector2(-rig.ForceLimit, rig.ForceLimit);
                actuator.CustomParams.Kp = rig.Kp;
                actuator.CustomParams.Kvp = 0f;
                actuators[actionIndex] = actuator;
            }
            return actuators;
        }
    }
}
