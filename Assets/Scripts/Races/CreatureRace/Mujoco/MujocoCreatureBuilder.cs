using Mujoco;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Builds any rig as org.mujoco components at race start: one MjBody per rig body, its
    /// free joint, geoms, explicit inertial and hinges, the contact excludes, and one
    /// position servo per action in action order. This is the whole creature template's
    /// answer to "how does a MuJoCo-Warp-trained creature get into Unity".
    ///
    /// WHY NOT THE MJCF IMPORTER (found on the worm, true for every creature):
    ///   * the importer's hinge-axis default is Unity +X, and MuJoCo's own writer drops
    ///     default-valued axes, so any round trip turns (0, 0, 1) hinges into (1, 0, 0) ones
    ///     - the defect training/mojucuboy/make_unity_mjcf.py works around;
    ///   * the importer attaches the built-in Standard shader (magenta in URP) and pulls in
    ///     the MJCF's own floor, which would then have to be found and removed;
    ///   * an editor-time import bakes a prefab that has to be kept in step with the MJCF by
    ///     hand, and MjScene's compile-once rule means it could not sit in the scene anyway.
    /// Building the components directly from the trainer's rig JSON cannot lose an axis and
    /// makes one file the single source of truth for training and racing.
    ///
    /// FRAMES. The plug-in maps MuJoCo (x, y, z) to Unity (x, z, y). Every local offset,
    /// axis and quaternion goes through that map, and the root alone gets an extra yaw that
    /// points MuJoCo +x (the creature's forward) down the lane. The policy sees only
    /// body-frame quantities, so this yaw is invisible to it, like the spawn yaw it trained with.
    ///
    /// Any number of racers can be built in the same frame under the same MjScene: the
    /// plug-in names every element uniquely and the view resolves ids and qpos/qvel/ctrl
    /// addresses per component after the compile, so racers never share an index.
    /// </summary>
    public static class MujocoCreatureBuilder
    {
        public static MujocoCreatureInstance Build(CreatureLayout layout, CreaturePilot pilot, string rootName,
                                                   Vector3 rootOrigin, Vector3 laneForward, Material material,
                                                   CreatureMeshCache meshes)
        {
            CreatureRig rig = layout.Rig;
            var root = new GameObject(rootName);
            root.transform.SetPositionAndRotation(rootOrigin, Quaternion.identity);

            // Unity +X (= MuJoCo body +x under the swap) turned onto the lane direction.
            Quaternion facing = Quaternion.LookRotation(laneForward, Vector3.up)
                              * Quaternion.FromToRotation(Vector3.right, Vector3.forward);

            int bodyCount = rig.Bodies.Count;
            var bodyTransforms = new Transform[bodyCount];
            var bodies = new MjBody[bodyCount];
            var hinges = new MjHingeJoint[rig.Joints.Count];
            var geomTransforms = new Transform[rig.Geoms.Count];

            for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
            {
                CreatureBodyDef def = rig.Bodies[bodyIndex];
                bool isRoot = def.Parent < 0;
                Transform parent = isRoot ? root.transform : bodyTransforms[def.Parent];

                var bodyObject = new GameObject(def.Name);
                bodyObject.transform.SetParent(parent, false);
                Vector3 position = def.Position;
                if (isRoot && float.IsFinite(rig.SpawnRootHeight))
                {
                    position.z = rig.SpawnRootHeight;
                }
                bodyObject.transform.localPosition = CreatureFrames.UnityFromPlugin(position);
                Quaternion local = CreatureFrames.UnityRotationFromPlugin(def.Rotation);
                bodyObject.transform.localRotation = isRoot ? facing * local : local;
                bodies[bodyIndex] = bodyObject.AddComponent<MjBody>();
                bodyTransforms[bodyIndex] = bodyObject.transform;

                if (def.FreeJoint)
                {
                    var freeJoint = new GameObject(def.Name + "_free");
                    freeJoint.transform.SetParent(bodyObject.transform, false);
                    freeJoint.AddComponent<MjFreeJoint>();
                }

                float sharedGeomMass = SharedGeomMass(rig, bodyIndex);
                for (int geomIndex = 0; geomIndex < rig.Geoms.Count; geomIndex++)
                {
                    CreatureGeomDef geom = rig.Geoms[geomIndex];
                    if (geom.Body == bodyIndex)
                    {
                        geomTransforms[geomIndex] = AddGeom(bodyObject, geom, sharedGeomMass, material, meshes);
                    }
                }

                if (def.HasInertial)
                {
                    AddInertial(bodyObject, def);
                }

                for (int jointIndex = 0; jointIndex < rig.Joints.Count; jointIndex++)
                {
                    if (rig.Joints[jointIndex].Body == bodyIndex)
                    {
                        hinges[jointIndex] = AddHinge(bodyObject, rig.Joints[jointIndex]);
                    }
                }
            }

            AddExcludes(root, rig, bodies);
            MjActuator[] actuators = AddActuators(root, rig, hinges);
            var actionJoints = new MjHingeJoint[rig.ActionSize];
            for (int actionIndex = 0; actionIndex < actionJoints.Length; actionIndex++)
            {
                actionJoints[actionIndex] = hinges[rig.Actuators[actionIndex].Joint];
            }

            MujocoCreatureView view = root.AddComponent<MujocoCreatureView>();
            view.Bind(bodies, actionJoints, actuators, pilot, laneForward, layout);
            return new MujocoCreatureInstance(root, bodyTransforms, geomTransforms);
        }

        /// <summary>
        /// A body with a mass but neither an explicit inertial nor geom masses: the mass is
        /// spread over its geoms by volume, so MuJoCo derives the inertia from the shapes.
        /// Returns the kg per cubic metre to give such geoms, 0 when not needed.
        /// </summary>
        private static float SharedGeomMass(CreatureRig rig, int bodyIndex)
        {
            CreatureBodyDef body = rig.Bodies[bodyIndex];
            if (body.HasInertial || body.Mass <= 0f)
            {
                return 0f;
            }
            float volume = 0f;
            for (int geomIndex = 0; geomIndex < rig.Geoms.Count; geomIndex++)
            {
                CreatureGeomDef geom = rig.Geoms[geomIndex];
                if (geom.Body != bodyIndex)
                {
                    continue;
                }
                if (geom.Mass > 0f)
                {
                    return 0f;
                }
                volume += Volume(geom);
            }
            return volume > 0f ? body.Mass / volume : 0f;
        }

        private static float Volume(CreatureGeomDef geom)
        {
            switch (geom.Type)
            {
                case CreatureGeomType.Capsule:
                    return Mathf.PI * geom.Radius * geom.Radius * (2f * geom.HalfLength + 4f / 3f * geom.Radius);
                case CreatureGeomType.Box:
                    return 8f * geom.HalfExtents.x * geom.HalfExtents.y * geom.HalfExtents.z;
                default:
                    return 4f / 3f * Mathf.PI * geom.Radius * geom.Radius * geom.Radius;
            }
        }

        private static Transform AddGeom(GameObject bodyObject, CreatureGeomDef def, float sharedDensity,
                                         Material material, CreatureMeshCache meshes)
        {
            var geomObject = new GameObject(def.Name);
            geomObject.transform.SetParent(bodyObject.transform, false);

            MjGeom geom = geomObject.AddComponent<MjGeom>();
            Mesh mesh;
            switch (def.Type)
            {
                case CreatureGeomType.Capsule:
                    if (def.HasFromTo)
                    {
                        // MjCapsuleShape runs along the geom's local +Y.
                        geomObject.transform.localPosition = CreatureFrames.UnityFromPlugin(0.5f * (def.From + def.To));
                        geomObject.transform.localRotation = Quaternion.FromToRotation(
                            Vector3.up, CreatureFrames.UnityFromPlugin(Vector3.Normalize(def.To - def.From)));
                    }
                    else
                    {
                        // MJCF capsules run along the geom's local z = Unity local +Y under the swap.
                        PlaceByPose(geomObject.transform, def);
                    }
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Capsule;
                    geom.Capsule.Radius = def.Radius;
                    geom.Capsule.HalfHeight = def.HalfLength;
                    mesh = meshes.Capsule(def.Radius, def.HalfLength);
                    break;
                case CreatureGeomType.Box:
                    PlaceByPose(geomObject.transform, def);
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Box;
                    geom.Box.Extents = MjEngineTool.UnityExtents(def.HalfExtents);
                    mesh = meshes.Box(geom.Box.Extents);
                    break;
                default:
                    PlaceByPose(geomObject.transform, def);
                    geom.ShapeType = MjShapeComponent.ShapeTypes.Sphere;
                    geom.Sphere.Radius = def.Radius;
                    mesh = meshes.Sphere(def.Radius);
                    break;
            }

            if (def.Mass > 0f)
            {
                geom.Mass = def.Mass;
            }
            else if (sharedDensity > 0f)
            {
                geom.Mass = sharedDensity * Volume(def);
            }
            MujocoCreatureWorld.ApplyContact(geom, def.Contact);

            // Render child, not an MjMeshFilter: the shared mesh and the racer's one shared
            // material, so every part of the same size batches together.
            var visual = new GameObject("visual");
            visual.transform.SetParent(geomObject.transform, false);
            visual.AddComponent<MeshFilter>().sharedMesh = mesh;
            visual.AddComponent<MeshRenderer>().sharedMaterial = material;
            return geomObject.transform;
        }

        private static void PlaceByPose(Transform geomTransform, CreatureGeomDef def)
        {
            geomTransform.localPosition = CreatureFrames.UnityFromPlugin(def.Position);
            geomTransform.localRotation = CreatureFrames.UnityRotationFromPlugin(def.Rotation);
        }

        private static void AddInertial(GameObject bodyObject, CreatureBodyDef def)
        {
            var inertialObject = new GameObject(def.Name + "_inertial");
            inertialObject.transform.SetParent(bodyObject.transform, false);
            inertialObject.transform.localPosition = CreatureFrames.UnityFromPlugin(def.InertialPosition);
            inertialObject.transform.localRotation = CreatureFrames.UnityRotationFromPlugin(def.InertialRotation);
            MjInertial inertial = inertialObject.AddComponent<MjInertial>();
            inertial.Mass = def.Mass;
            // MjInertial writes diaginertia through MjExtents (a y/z swap); undo it so the
            // MJCF carries the rig's own (x, y, z) diagonal.
            inertial.DiagInertia = MjEngineTool.UnityExtents(def.InertiaDiagonal);
        }

        private static MjHingeJoint AddHinge(GameObject bodyObject, CreatureJointDef def)
        {
            var jointObject = new GameObject(def.Name);
            jointObject.transform.SetParent(bodyObject.transform, false);
            jointObject.transform.localPosition = CreatureFrames.UnityFromPlugin(def.Anchor);
            // MjHingeJoint's axis is its transform's local +X (MjEngineTool.PositionAxisToMjcf).
            jointObject.transform.localRotation = Quaternion.FromToRotation(
                Vector3.right, CreatureFrames.UnityFromPlugin(def.Axis.normalized));

            MjHingeJoint hinge = jointObject.AddComponent<MjHingeJoint>();
            // The plug-in's compiler reads angles in degrees (MuJoCo's default).
            hinge.RangeLower = def.Limited ? def.RangeLower * Mathf.Rad2Deg : 0f;
            hinge.RangeUpper = def.Limited ? def.RangeUpper * Mathf.Rad2Deg : 0f;

            MjJointSettings settings = hinge.Settings;
            settings.Armature = def.Armature;
            settings.Spring.Damping = def.Damping;
            settings.Spring.Stiffness = def.Stiffness;
            settings.Solver.Limited = def.Limited;
            settings.Solver.FrictionLoss = def.FrictionLoss;
            if (def.HasSolRefLimit)
            {
                settings.Solver.RefLimit.TimeConst = def.SolRefLimit.x;
                settings.Solver.RefLimit.DampRatio = def.SolRefLimit.y;
            }
            hinge.Settings = settings;
            return hinge;
        }

        private static void AddExcludes(GameObject root, CreatureRig rig, MjBody[] bodies)
        {
            var excludes = new GameObject("excludes");
            excludes.transform.SetParent(root.transform, false);
            for (int pairIndex = 0; pairIndex < rig.Excludes.Count; pairIndex++)
            {
                Vector2Int pair = rig.Excludes[pairIndex];
                var excludeObject = new GameObject($"exclude_{rig.Bodies[pair.x].Name}_{rig.Bodies[pair.y].Name}");
                excludeObject.transform.SetParent(excludes.transform, false);
                MjExclude exclude = excludeObject.AddComponent<MjExclude>();
                exclude.Body1 = bodies[pair.x];
                exclude.Body2 = bodies[pair.y];
            }
        }

        private static MjActuator[] AddActuators(GameObject root, CreatureRig rig, MjHingeJoint[] hinges)
        {
            var actuatorRoot = new GameObject("actuators");
            actuatorRoot.transform.SetParent(root.transform, false);
            var actuators = new MjActuator[rig.ActionSize];
            // Sibling order = action order: MjScene sorts actuators by sibling index.
            for (int actionIndex = 0; actionIndex < rig.ActionSize; actionIndex++)
            {
                CreatureActuatorDef def = rig.Actuators[actionIndex];
                var actuatorObject = new GameObject("act_" + def.Name);
                actuatorObject.transform.SetParent(actuatorRoot.transform, false);
                MjActuator actuator = actuatorObject.AddComponent<MjActuator>();
                actuator.Type = MjActuator.ActuatorType.Position;
                actuator.Joint = hinges[def.Joint];
                // ctrl is the joint's own qpos unit for a position servo: radians.
                actuator.CommonParams.CtrlLimited = def.CtrlLimited;
                actuator.CommonParams.CtrlRange = def.CtrlRange;
                actuator.CommonParams.ForceLimited = def.ForceLimited;
                actuator.CommonParams.ForceRange = def.ForceRange;
                actuator.CustomParams.Kp = def.Kp;
                actuator.CustomParams.Kvp = def.Kv;
                actuators[actionIndex] = actuator;
            }
            return actuators;
        }
    }
}
