using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Reads a creature rig JSON, the format the MuJoCo Warp trainers write next to their
    /// MJCF (training/quad/quad_rig.json is the reference), into a <see cref="CreatureRig"/>.
    /// Everything is MuJoCo: x forward, y left, z up, radians, SI units, quaternions (w, x, y, z).
    ///
    /// <code>
    /// {
    ///   "model": "quad",
    ///   "bodies":    [ { "name", "parent" (null = root), "pos", "quat", "freejoint",
    ///                    "mass", "inertiaDiag", "ipos", "iquat" } ],       parents first
    ///   "geoms":     [ { "name", "body" ("world" + type plane = the floor), "type" capsule|box|sphere,
    ///                    "size", "pos", "quat" | "fromto", "mass", "friction", "condim",
    ///                    "solref", "solimp" } ],
    ///   "joints":    [ { "name", "body", "type" hinge|free, "axis", "pos", "range", "damping",
    ///                    "armature", "frictionloss", "stiffness", "solreflimit" } ],
    ///   "actuators": [ { "name", "joint", "type" position, "kp", "kv", "ctrlrange", "forcerange" } ],
    ///   "excludes":  [ ["bodyA", "bodyB"] | { "body1", "body2" } ],             optional
    ///   "actionOrder": [ actuator or joint names ],                             optional
    ///   "restPose": { name: rad } | [ rad ],   "actionScaleRad": 0.785398,
    ///   "physics": { "timestep", "decimation" },   "torso": "name",
    ///   "restRootHeight", "spawnRootHeight",   "task": { "targetSpeed" }
    /// }
    /// </code>
    ///
    /// Bodies may instead carry their own "geoms" / "joints" lists (the body is then implied),
    /// and a few MJCF spellings are accepted (diaginertia, action_scale_rad, ...). Anything the
    /// builder cannot represent is an error at load, never a silently different animal.
    /// </summary>
    public static class CreatureRigParser
    {
        private const string WORLD = "world";
        private static readonly Vector2 DefaultSolRefLimit = new(0.02f, 1f);

        public static bool TryParse(string json, out CreatureRig rig, out string error)
        {
            rig = null;
            if (!CreatureJson.TryParse(json, out object root, out error))
            {
                error = "the rig JSON does not parse: " + error;
                return false;
            }
            if (!(root is Dictionary<string, object> document))
            {
                error = "the rig JSON is not an object";
                return false;
            }
            try
            {
                rig = Read(document);
            }
            catch (FormatException exception)
            {
                rig = null;
                error = exception.Message;
                return false;
            }
            if (!rig.Validate(out error))
            {
                rig = null;
                return false;
            }
            return true;
        }

        // ---------------------------------------------------------------- reading --

        private static CreatureRig Read(Dictionary<string, object> document)
        {
            List<object> bodyList = List(document, "bodies");
            if (bodyList == null || bodyList.Count == 0)
            {
                throw new FormatException("the rig has no \"bodies\" list");
            }

            // Bodies, then any nested geoms/joints they carry.
            var rawBodies = new List<Dictionary<string, object>>();
            var geomSources = new List<Dictionary<string, object>>();
            var jointSources = new List<Dictionary<string, object>>();
            for (int index = 0; index < bodyList.Count; index++)
            {
                if (!(bodyList[index] is Dictionary<string, object> body))
                {
                    throw new FormatException($"bodies[{index}] is not an object");
                }
                rawBodies.Add(body);
                CollectNested(body, "geoms", geomSources);
                CollectNested(body, "joints", jointSources);
            }
            CollectTopLevel(document, "geoms", geomSources);
            CollectTopLevel(document, "joints", jointSources);

            CreatureBodyDef[] bodies = ReadBodies(rawBodies);
            var names = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                names[bodies[bodyIndex].Name] = bodyIndex;
            }

            CreatureContact floor = CreatureContact.MujocoDefault;
            bool hasFloor = false;
            var geoms = new List<CreatureGeomDef>();
            for (int index = 0; index < geomSources.Count; index++)
            {
                Dictionary<string, object> source = geomSources[index];
                string bodyName = Text(source, string.Empty, "body");
                string type = Text(source, "sphere", "type").ToLowerInvariant();
                if (string.IsNullOrEmpty(bodyName) || bodyName == WORLD)
                {
                    if (type != "plane")
                    {
                        throw new FormatException($"geom '{Text(source, "?", "name")}' is fixed to the world; "
                                                + "only the floor plane may be (the race supplies its own track)");
                    }
                    floor = ReadContact(source);
                    hasFloor = true;
                    continue;
                }
                geoms.Add(ReadGeom(source, Resolve(names, bodyName, "geom " + Text(source, "?", "name")), type));
            }
            if (!hasFloor && geoms.Count > 0)
            {
                floor = geoms[0].Contact;
            }

            var joints = new List<CreatureJointDef>();
            for (int index = 0; index < jointSources.Count; index++)
            {
                Dictionary<string, object> source = jointSources[index];
                string jointName = Text(source, "joint" + index, "name");
                int body = Resolve(names, Text(source, string.Empty, "body"), "joint " + jointName);
                string type = Text(source, "hinge", "type").ToLowerInvariant();
                if (type == "free")
                {
                    bodies[body].FreeJoint = true;
                    continue;
                }
                if (type != "hinge")
                {
                    throw new FormatException($"joint '{jointName}' is a '{type}' joint; only hinge and free are supported");
                }
                joints.Add(ReadJoint(source, jointName, body));
            }

            CreatureActuatorDef[] actuators = ReadActuators(document, joints);
            string[] actionOrder = Strings(document, "actionOrder", "action_order", "actuator_order");
            if (actionOrder != null)
            {
                actuators = Reorder(actuators, joints, actionOrder);
            }

            float actionScale = (float)Number(document, 0.0, "actionScaleRad", "action_scale_rad", "actionScale");
            if (actionScale <= 0f && actuators.Length > 0)
            {
                // No explicit scale: the servo's own ctrl half-range is the only honest guess.
                Vector2 range = actuators[0].CtrlRange;
                actionScale = 0.5f * (range.y - range.x);
            }
            var rig = new CreatureRig(
                Text(document, "creature", "model", "name"),
                bodies,
                geoms.ToArray(),
                joints.ToArray(),
                actuators,
                ReadExcludes(document, names),
                ReadRestPose(document, actuators, joints),
                actionScale);

            Dictionary<string, object> physics = Object(document, "physics");
            rig.PhysicsDt = (float)Number(physics ?? document, 0.005, "timestep", "physicsDt", "dt");
            rig.Decimation = Mathf.RoundToInt((float)Number(physics ?? document, 4.0, "decimation"));
            string torso = Text(document, string.Empty, "torso", "referenceBody");
            rig.TorsoBody = string.IsNullOrEmpty(torso) ? 0 : Resolve(names, torso, "torso");
            rig.SpawnRootHeight = (float)Number(document, double.NaN, "spawnRootHeight", "spawnHeight");
            rig.RestRootHeight = (float)Number(document, double.NaN, "restRootHeight", "restHeight");
            Dictionary<string, object> task = Object(document, "task");
            rig.TargetSpeed = (float)Number(task ?? document, double.NaN, "targetSpeed");
            rig.FloorContact = floor;
            return rig;
        }

        private static CreatureBodyDef[] ReadBodies(List<Dictionary<string, object>> raw)
        {
            // Parents first, keeping the file's order otherwise (stable topological order).
            var placed = new List<Dictionary<string, object>>(raw.Count);
            var placedNames = new HashSet<string>(StringComparer.Ordinal);
            var pending = new List<Dictionary<string, object>>(raw);
            while (pending.Count > 0)
            {
                bool progressed = false;
                for (int index = 0; index < pending.Count; index++)
                {
                    string parent = ParentName(pending[index]);
                    if (parent.Length == 0 ? placed.Count == 0 : placedNames.Contains(parent))
                    {
                        placed.Add(pending[index]);
                        placedNames.Add(Text(pending[index], string.Empty, "name"));
                        pending.RemoveAt(index);
                        progressed = true;
                        break;
                    }
                }
                if (!progressed)
                {
                    throw new FormatException(
                        $"body '{Text(pending[0], "?", "name")}' has no parent in the rig (or there are two roots)");
                }
            }

            var bodies = new CreatureBodyDef[placed.Count];
            var index0 = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int bodyIndex = 0; bodyIndex < placed.Count; bodyIndex++)
            {
                Dictionary<string, object> source = placed[bodyIndex];
                string name = Text(source, string.Empty, "name");
                if (name.Length == 0 || index0.ContainsKey(name))
                {
                    throw new FormatException($"bodies[{bodyIndex}] has no name, or a name used twice ('{name}')");
                }
                index0[name] = bodyIndex;
                string parent = ParentName(source);
                var body = new CreatureBodyDef
                {
                    Name = name,
                    Parent = parent.Length == 0 ? -1 : index0[parent],
                    Position = Vector(source, Vector3.zero, "pos"),
                    Rotation = Quat(source, "quat"),
                    FreeJoint = Flag(source, false, "freejoint", "freeJoint"),
                    Mass = (float)Number(source, 0.0, "mass"),
                };
                if (TryInertia(source, out Vector3 inertia))
                {
                    body.HasInertial = true;
                    body.InertiaDiagonal = inertia;
                    body.InertialPosition = Vector(source, Vector3.zero, "ipos", "inertialPos");
                    body.InertialRotation = Quat(source, "iquat", "inertialQuat");
                }
                bodies[bodyIndex] = body;
            }
            return bodies;
        }

        private static CreatureGeomDef ReadGeom(Dictionary<string, object> source, int body, string type)
        {
            string name = Text(source, "geom", "name");
            float[] size = Floats(source, "size");
            var geom = new CreatureGeomDef
            {
                Name = name,
                Body = body,
                Position = Vector(source, Vector3.zero, "pos"),
                Rotation = Quat(source, "quat"),
                Mass = (float)Number(source, 0.0, "mass"),
                Contact = ReadContact(source),
            };
            switch (type)
            {
                case "capsule":
                    geom.Type = CreatureGeomType.Capsule;
                    geom.Radius = (float)Number(source, size != null && size.Length > 0 ? size[0] : 0.0, "radius");
                    geom.HalfLength = (float)Number(source, size != null && size.Length > 1 ? size[1] : 0.0,
                                                    "halfLength", "half_length");
                    if (TryFromTo(source, out Vector3 from, out Vector3 to))
                    {
                        geom.HasFromTo = true;
                        geom.From = from;
                        geom.To = to;
                        geom.HalfLength = 0.5f * (to - from).magnitude;
                        geom.Position = 0.5f * (from + to);
                    }
                    break;
                case "box":
                    geom.Type = CreatureGeomType.Box;
                    if (size == null || size.Length < 3)
                    {
                        throw new FormatException($"box geom '{name}' needs a 3-value size (half-extents)");
                    }
                    geom.HalfExtents = new Vector3(size[0], size[1], size[2]);
                    break;
                case "sphere":
                    geom.Type = CreatureGeomType.Sphere;
                    geom.Radius = (float)Number(source, size != null && size.Length > 0 ? size[0] : 0.0, "radius");
                    break;
                default:
                    throw new FormatException($"geom '{name}' is a '{type}'; the builder supports capsule, box and sphere");
            }
            return geom;
        }

        private static CreatureContact ReadContact(Dictionary<string, object> source)
        {
            CreatureContact fallback = CreatureContact.MujocoDefault;
            Vector3 friction = fallback.Friction;
            if (TryGet(source, out object value, "friction"))
            {
                if (value is double single)
                {
                    friction.x = (float)single;
                }
                else if (value is List<object> list)
                {
                    float[] parts = ToFloats(list, "friction");
                    friction.x = parts.Length > 0 ? parts[0] : friction.x;
                    friction.y = parts.Length > 1 ? parts[1] : friction.y;
                    friction.z = parts.Length > 2 ? parts[2] : friction.z;
                }
            }
            float[] solRef = Floats(source, "solref");
            float[] solImp = Floats(source, "solimp");
            return new CreatureContact(
                friction,
                Mathf.RoundToInt((float)Number(source, fallback.ConDim, "condim")),
                solRef != null && solRef.Length >= 2 ? new Vector2(solRef[0], solRef[1]) : fallback.SolRef,
                solImp != null && solImp.Length >= 3 ? new Vector3(solImp[0], solImp[1], solImp[2]) : fallback.SolImp);
        }

        private static CreatureJointDef ReadJoint(Dictionary<string, object> source, string name, int body)
        {
            float[] range = Floats(source, "range");
            bool hasRange = range != null && range.Length >= 2;
            float[] solRefLimit = Floats(source, "solreflimit");
            return new CreatureJointDef
            {
                Name = name,
                Body = body,
                Axis = Vector(source, Vector3.forward, "axis"),
                Anchor = Vector(source, Vector3.zero, "pos", "anchor"),
                Limited = Flag(source, hasRange, "limited"),
                RangeLower = hasRange ? range[0] : 0f,
                RangeUpper = hasRange ? range[1] : 0f,
                Damping = (float)Number(source, 0.0, "damping"),
                Armature = (float)Number(source, 0.0, "armature"),
                FrictionLoss = (float)Number(source, 0.0, "frictionloss", "frictionLoss"),
                Stiffness = (float)Number(source, 0.0, "stiffness"),
                HasSolRefLimit = solRefLimit != null && solRefLimit.Length >= 2,
                SolRefLimit = solRefLimit != null && solRefLimit.Length >= 2
                    ? new Vector2(solRefLimit[0], solRefLimit[1])
                    : DefaultSolRefLimit,
            };
        }

        private static CreatureActuatorDef[] ReadActuators(Dictionary<string, object> document,
                                                           List<CreatureJointDef> joints)
        {
            List<object> list = List(document, "actuators");
            if (list == null)
            {
                return Array.Empty<CreatureActuatorDef>();
            }
            var entries = new List<(int order, CreatureActuatorDef actuator)>(list.Count);
            for (int index = 0; index < list.Count; index++)
            {
                if (!(list[index] is Dictionary<string, object> source))
                {
                    throw new FormatException($"actuators[{index}] is not an object");
                }
                string name = Text(source, "actuator" + index, "name");
                string type = Text(source, "position", "type").ToLowerInvariant();
                if (type != "position")
                {
                    throw new FormatException($"actuator '{name}' is a '{type}' actuator; only position servos are supported");
                }
                string jointName = Text(source, name, "joint");
                int joint = FindJoint(joints, jointName);
                if (joint < 0)
                {
                    throw new FormatException($"actuator '{name}' drives unknown joint '{jointName}'");
                }
                float[] ctrl = Floats(source, "ctrlrange");
                float[] force = Floats(source, "forcerange");
                bool hasCtrl = ctrl != null && ctrl.Length >= 2;
                bool hasForce = force != null && force.Length >= 2;
                var actuator = new CreatureActuatorDef
                {
                    Name = name,
                    Joint = joint,
                    Kp = (float)Number(source, 1.0, "kp"),
                    Kv = (float)Number(source, 0.0, "kv"),
                    CtrlLimited = Flag(source, hasCtrl, "ctrllimited"),
                    CtrlRange = hasCtrl ? new Vector2(ctrl[0], ctrl[1]) : Vector2.zero,
                    ForceLimited = Flag(source, hasForce, "forcelimited"),
                    ForceRange = hasForce ? new Vector2(force[0], force[1]) : Vector2.zero,
                };
                entries.Add((Mathf.RoundToInt((float)Number(source, index, "index")), actuator));
            }
            entries.Sort((left, right) => left.order.CompareTo(right.order));
            var result = new CreatureActuatorDef[entries.Count];
            for (int index = 0; index < entries.Count; index++)
            {
                result[index] = entries[index].actuator;
            }
            return result;
        }

        private static CreatureActuatorDef[] Reorder(CreatureActuatorDef[] actuators, List<CreatureJointDef> joints,
                                                     string[] order)
        {
            if (order.Length != actuators.Length)
            {
                throw new FormatException($"actionOrder has {order.Length} names for {actuators.Length} actuators");
            }
            var result = new CreatureActuatorDef[order.Length];
            for (int actionIndex = 0; actionIndex < order.Length; actionIndex++)
            {
                for (int index = 0; index < actuators.Length; index++)
                {
                    if (actuators[index].Name == order[actionIndex] || joints[actuators[index].Joint].Name == order[actionIndex])
                    {
                        result[actionIndex] = actuators[index];
                        break;
                    }
                }
                if (result[actionIndex] == null)
                {
                    throw new FormatException($"actionOrder names '{order[actionIndex]}', which no actuator drives");
                }
            }
            return result;
        }

        private static Vector2Int[] ReadExcludes(Dictionary<string, object> document, Dictionary<string, int> names)
        {
            List<object> list = List(document, "excludes", "exclude");
            if (list == null)
            {
                return Array.Empty<Vector2Int>();
            }
            var result = new Vector2Int[list.Count];
            for (int index = 0; index < list.Count; index++)
            {
                string first;
                string second;
                if (list[index] is List<object> pair && pair.Count == 2)
                {
                    first = pair[0] as string;
                    second = pair[1] as string;
                }
                else if (list[index] is Dictionary<string, object> entry)
                {
                    first = Text(entry, string.Empty, "body1", "bodyA");
                    second = Text(entry, string.Empty, "body2", "bodyB");
                }
                else
                {
                    throw new FormatException($"excludes[{index}] is neither a name pair nor an object");
                }
                result[index] = new Vector2Int(Resolve(names, first, "exclude"), Resolve(names, second, "exclude"));
            }
            return result;
        }

        private static float[] ReadRestPose(Dictionary<string, object> document, CreatureActuatorDef[] actuators,
                                            List<CreatureJointDef> joints)
        {
            var rest = new float[actuators.Length];
            if (!TryGet(document, out object value, "restPose", "rest_pose", "rest"))
            {
                return rest;
            }
            if (value is List<object> list)
            {
                float[] parts = ToFloats(list, "restPose");
                if (parts.Length != actuators.Length)
                {
                    throw new FormatException($"restPose has {parts.Length} values for {actuators.Length} actions");
                }
                return parts;
            }
            if (value is Dictionary<string, object> byName)
            {
                for (int actionIndex = 0; actionIndex < actuators.Length; actionIndex++)
                {
                    string jointName = joints[actuators[actionIndex].Joint].Name;
                    rest[actionIndex] = (float)Number(byName, 0.0, actuators[actionIndex].Name, jointName);
                }
                return rest;
            }
            throw new FormatException("restPose must be a list or a name -> radians object");
        }

        // ------------------------------------------------------------- helpers --

        private static void CollectNested(Dictionary<string, object> body, string key,
                                          List<Dictionary<string, object>> into)
        {
            List<object> list = List(body, key);
            if (list == null)
            {
                return;
            }
            string bodyName = Text(body, string.Empty, "name");
            for (int index = 0; index < list.Count; index++)
            {
                if (list[index] is Dictionary<string, object> entry)
                {
                    if (!entry.ContainsKey("body"))
                    {
                        entry["body"] = bodyName;
                    }
                    into.Add(entry);
                }
            }
        }

        private static void CollectTopLevel(Dictionary<string, object> document, string key,
                                            List<Dictionary<string, object>> into)
        {
            List<object> list = List(document, key);
            if (list == null)
            {
                return;
            }
            for (int index = 0; index < list.Count; index++)
            {
                if (!(list[index] is Dictionary<string, object> entry))
                {
                    throw new FormatException($"{key}[{index}] is not an object");
                }
                into.Add(entry);
            }
        }

        private static int FindJoint(List<CreatureJointDef> joints, string name)
        {
            for (int jointIndex = 0; jointIndex < joints.Count; jointIndex++)
            {
                if (joints[jointIndex].Name == name)
                {
                    return jointIndex;
                }
            }
            return -1;
        }

        private static string ParentName(Dictionary<string, object> body)
        {
            string parent = Text(body, string.Empty, "parent");
            return parent == WORLD ? string.Empty : parent;
        }

        private static int Resolve(Dictionary<string, int> names, string name, string context)
        {
            if (name != null && names.TryGetValue(name, out int index))
            {
                return index;
            }
            throw new FormatException($"{context} names unknown body '{name}'");
        }

        private static bool TryInertia(Dictionary<string, object> source, out Vector3 inertia)
        {
            inertia = Vector3.zero;
            if (!TryGet(source, out object value, "inertiaDiag", "diaginertia", "inertia"))
            {
                return false;
            }
            if (value is double single && single > 0.0)
            {
                inertia = new Vector3((float)single, (float)single, (float)single);
                return true;
            }
            if (value is List<object> list && list.Count == 3)
            {
                float[] parts = ToFloats(list, "inertiaDiag");
                inertia = new Vector3(parts[0], parts[1], parts[2]);
                return true;
            }
            if (value is List<object> full && full.Count == 6)
            {
                throw new FormatException("a full 6-value inertia is not supported; give inertiaDiag plus iquat");
            }
            return false;
        }

        private static bool TryFromTo(Dictionary<string, object> source, out Vector3 from, out Vector3 to)
        {
            from = Vector3.zero;
            to = Vector3.zero;
            if (!TryGet(source, out object value, "fromto") || !(value is List<object> list))
            {
                return false;
            }
            if (list.Count == 6)
            {
                float[] parts = ToFloats(list, "fromto");
                from = new Vector3(parts[0], parts[1], parts[2]);
                to = new Vector3(parts[3], parts[4], parts[5]);
                return true;
            }
            if (list.Count == 2 && list[0] is List<object> start && list[1] is List<object> end)
            {
                float[] first = ToFloats(start, "fromto");
                float[] second = ToFloats(end, "fromto");
                if (first.Length == 3 && second.Length == 3)
                {
                    from = new Vector3(first[0], first[1], first[2]);
                    to = new Vector3(second[0], second[1], second[2]);
                    return true;
                }
            }
            throw new FormatException("fromto must be 6 numbers or two 3-number points");
        }

        private static bool TryGet(Dictionary<string, object> source, out object value, params string[] keys)
        {
            for (int index = 0; index < keys.Length; index++)
            {
                if (source.TryGetValue(keys[index], out value) && value != null)
                {
                    return true;
                }
            }
            value = null;
            return false;
        }

        private static Dictionary<string, object> Object(Dictionary<string, object> source, string key)
        {
            return TryGet(source, out object value, key) ? value as Dictionary<string, object> : null;
        }

        private static List<object> List(Dictionary<string, object> source, params string[] keys)
        {
            return TryGet(source, out object value, keys) ? value as List<object> : null;
        }

        private static string Text(Dictionary<string, object> source, string fallback, params string[] keys)
        {
            return TryGet(source, out object value, keys) && value is string text ? text : fallback;
        }

        private static double Number(Dictionary<string, object> source, double fallback, params string[] keys)
        {
            if (!TryGet(source, out object value, keys))
            {
                return fallback;
            }
            if (value is double number)
            {
                return number;
            }
            if (value is bool flag)
            {
                return flag ? 1.0 : 0.0;
            }
            throw new FormatException($"'{keys[0]}' is not a number");
        }

        private static bool Flag(Dictionary<string, object> source, bool fallback, params string[] keys)
        {
            if (!TryGet(source, out object value, keys))
            {
                return fallback;
            }
            if (value is bool flag)
            {
                return flag;
            }
            if (value is double number)
            {
                return number != 0.0;
            }
            if (value is string text)
            {
                return text == "true";
            }
            return fallback;
        }

        private static float[] Floats(Dictionary<string, object> source, params string[] keys)
        {
            if (!TryGet(source, out object value, keys))
            {
                return null;
            }
            if (value is double single)
            {
                return new[] { (float)single };
            }
            return value is List<object> list ? ToFloats(list, keys[0]) : null;
        }

        private static string[] Strings(Dictionary<string, object> source, params string[] keys)
        {
            if (!TryGet(source, out object value, keys) || !(value is List<object> list))
            {
                return null;
            }
            var result = new string[list.Count];
            for (int index = 0; index < list.Count; index++)
            {
                result[index] = list[index] as string ?? throw new FormatException($"'{keys[0]}' must list names");
            }
            return result;
        }

        private static float[] ToFloats(List<object> list, string context)
        {
            var result = new float[list.Count];
            for (int index = 0; index < list.Count; index++)
            {
                if (!(list[index] is double number))
                {
                    throw new FormatException($"'{context}' must contain numbers only");
                }
                result[index] = (float)number;
            }
            return result;
        }

        private static Vector3 Vector(Dictionary<string, object> source, Vector3 fallback, params string[] keys)
        {
            float[] parts = Floats(source, keys);
            if (parts == null)
            {
                return fallback;
            }
            if (parts.Length != 3)
            {
                throw new FormatException($"'{keys[0]}' must have 3 numbers");
            }
            return new Vector3(parts[0], parts[1], parts[2]);
        }

        /// <summary>MJCF order (w, x, y, z) into the MuJoCo-component Quaternion the rig keeps.</summary>
        private static Quaternion Quat(Dictionary<string, object> source, params string[] keys)
        {
            float[] parts = Floats(source, keys);
            if (parts == null)
            {
                return Quaternion.identity;
            }
            if (parts.Length != 4)
            {
                throw new FormatException($"'{keys[0]}' must have 4 numbers (w, x, y, z)");
            }
            return new Quaternion(parts[1], parts[2], parts[3], parts[0]);
        }
    }
}
