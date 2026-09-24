using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// A creature body as MuJoCo sees it: bodies (a tree, parents listed first, the root
    /// first with a free joint), geoms, hinges, position servos in action order, contact
    /// excludes, and the action contract (rest pose, action scale). Everything in MuJoCo
    /// coordinates, x forward, y left, z up.
    ///
    /// Built by <see cref="CreatureRigParser"/> from a trainer's rig JSON (quad_rig.json's
    /// format), or by an adapter for an older format (the worm's WormRig). Every MuJoCo racer
    /// of every creature is built from one of these by <see cref="MujocoCreatureBuilder"/>.
    /// </summary>
    public sealed class CreatureRig
    {
        private readonly CreatureBodyDef[] _bodies;
        private readonly CreatureGeomDef[] _geoms;
        private readonly CreatureJointDef[] _joints;
        private readonly CreatureActuatorDef[] _actuators;
        private readonly Vector2Int[] _excludes;
        private readonly float[] _restPose;

        public CreatureRig(string name, CreatureBodyDef[] bodies, CreatureGeomDef[] geoms, CreatureJointDef[] joints,
                           CreatureActuatorDef[] actuators, Vector2Int[] excludes, float[] restPose, float actionScale)
        {
            Name = name ?? string.Empty;
            _bodies = bodies ?? Array.Empty<CreatureBodyDef>();
            _geoms = geoms ?? Array.Empty<CreatureGeomDef>();
            _joints = joints ?? Array.Empty<CreatureJointDef>();
            _actuators = actuators ?? Array.Empty<CreatureActuatorDef>();
            _excludes = excludes ?? Array.Empty<Vector2Int>();
            _restPose = restPose ?? new float[_actuators.Length];
            ActionScale = actionScale;
        }

        public string Name { get; }
        public IReadOnlyList<CreatureBodyDef> Bodies => _bodies;
        public IReadOnlyList<CreatureGeomDef> Geoms => _geoms;
        public IReadOnlyList<CreatureJointDef> Joints => _joints;
        /// <summary>In action order.</summary>
        public IReadOnlyList<CreatureActuatorDef> Actuators => _actuators;
        /// <summary>Body index pairs MuJoCo must not collide (beyond parent-child, which it skips itself).</summary>
        public IReadOnlyList<Vector2Int> Excludes => _excludes;
        /// <summary>Joint target at action 0, radians, action order.</summary>
        public IReadOnlyList<float> RestPose => _restPose;
        public int ActionSize => _actuators.Length;
        /// <summary>target = rest + clip(action, -1, 1) * ActionScale, radians.</summary>
        public float ActionScale { get; }

        public float PhysicsDt { get; set; } = 0.005f;
        public int Decimation { get; set; } = 4;
        /// <summary>The rig's own "torso" (reference body by default); 0 = the root.</summary>
        public int TorsoBody { get; set; }
        /// <summary>Root height to spawn at, metres; NaN = the root body's own pos.</summary>
        public float SpawnRootHeight { get; set; } = float.NaN;
        /// <summary>Root height standing still at the rest pose; NaN when the rig does not say.</summary>
        public float RestRootHeight { get; set; } = float.NaN;
        /// <summary>The task's target speed, m/s; NaN when the rig has none.</summary>
        public float TargetSpeed { get; set; } = float.NaN;
        /// <summary>The trained floor's contact (friction etc.). The race's MuJoCo plane uses it.</summary>
        public CreatureContact FloorContact { get; set; } = CreatureContact.MujocoDefault;

        public int FindBody(string name)
        {
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                if (_bodies[bodyIndex].Name == name)
                {
                    return bodyIndex;
                }
            }
            return -1;
        }

        /// <summary>Action index of an actuator, matched by its name or by its joint's name.</summary>
        public int FindAction(string name)
        {
            for (int actionIndex = 0; actionIndex < _actuators.Length; actionIndex++)
            {
                if (_actuators[actionIndex].Name == name || _joints[_actuators[actionIndex].Joint].Name == name)
                {
                    return actionIndex;
                }
            }
            return -1;
        }

        public string ActionName(int actionIndex)
        {
            return _actuators[actionIndex].Name;
        }

        /// <summary>
        /// How far the body's geometry reaches along its own +x (MuJoCo forward) from the
        /// body origin: the "nose" a finish line is judged by. 0 for a body without geoms.
        /// </summary>
        public float ForwardExtent(int body)
        {
            float extent = 0f;
            bool any = false;
            for (int geomIndex = 0; geomIndex < _geoms.Length; geomIndex++)
            {
                CreatureGeomDef geom = _geoms[geomIndex];
                if (geom.Body != body)
                {
                    continue;
                }
                float reach = GeomForwardReach(geom);
                extent = any ? Mathf.Max(extent, reach) : reach;
                any = true;
            }
            return extent;
        }

        /// <summary>Checks the invariants the builder relies on. Returns false with a reason.</summary>
        public bool Validate(out string error)
        {
            if (_bodies.Length == 0)
            {
                error = "the rig has no bodies";
                return false;
            }
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                CreatureBodyDef body = _bodies[bodyIndex];
                if (bodyIndex == 0 && (body.Parent >= 0 || !body.FreeJoint))
                {
                    error = $"body 0 '{body.Name}' must be the root with a free joint (a racer moves on its own)";
                    return false;
                }
                if (bodyIndex > 0 && (body.Parent < 0 || body.Parent >= bodyIndex))
                {
                    error = $"body '{body.Name}' must name a parent listed before it";
                    return false;
                }
                if (bodyIndex > 0 && body.FreeJoint)
                {
                    error = $"body '{body.Name}' has a free joint; only the root may";
                    return false;
                }
                if (body.HasInertial && body.Mass <= 0f)
                {
                    error = $"body '{body.Name}' has an inertial without mass";
                    return false;
                }
            }
            for (int geomIndex = 0; geomIndex < _geoms.Length; geomIndex++)
            {
                CreatureGeomDef geom = _geoms[geomIndex];
                if (geom.Body < 0 || geom.Body >= _bodies.Length)
                {
                    error = $"geom '{geom.Name}' is on an unknown body";
                    return false;
                }
                bool sized = geom.Type == CreatureGeomType.Box
                    ? geom.HalfExtents.x > 0f && geom.HalfExtents.y > 0f && geom.HalfExtents.z > 0f
                    : geom.Radius > 0f;
                if (!sized)
                {
                    error = $"geom '{geom.Name}' has no size";
                    return false;
                }
            }
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                CreatureJointDef joint = _joints[jointIndex];
                if (joint.Body <= 0 || joint.Body >= _bodies.Length)
                {
                    error = $"joint '{joint.Name}' must sit on a non-root body (the root has its free joint)";
                    return false;
                }
                if (joint.Axis.sqrMagnitude < 1e-12f)
                {
                    error = $"joint '{joint.Name}' has no axis";
                    return false;
                }
                if (joint.Limited && joint.RangeLower > joint.RangeUpper)
                {
                    error = $"joint '{joint.Name}' has its range the wrong way round";
                    return false;
                }
            }
            if (_actuators.Length == 0)
            {
                error = "the rig has no actuators, so nothing for a policy to drive";
                return false;
            }
            for (int actionIndex = 0; actionIndex < _actuators.Length; actionIndex++)
            {
                int joint = _actuators[actionIndex].Joint;
                if (joint < 0 || joint >= _joints.Length)
                {
                    error = $"actuator '{_actuators[actionIndex].Name}' drives an unknown joint";
                    return false;
                }
                for (int earlier = 0; earlier < actionIndex; earlier++)
                {
                    if (_actuators[earlier].Joint == joint)
                    {
                        error = $"joint '{_joints[joint].Name}' is driven by two actuators";
                        return false;
                    }
                }
            }
            if (_restPose.Length != _actuators.Length)
            {
                error = $"the rest pose has {_restPose.Length} values for {_actuators.Length} actions";
                return false;
            }
            if (!(ActionScale > 0f))
            {
                error = "the action scale must be positive";
                return false;
            }
            for (int pairIndex = 0; pairIndex < _excludes.Length; pairIndex++)
            {
                Vector2Int pair = _excludes[pairIndex];
                if (pair.x < 0 || pair.x >= _bodies.Length || pair.y < 0 || pair.y >= _bodies.Length)
                {
                    error = "an exclude names an unknown body";
                    return false;
                }
            }
            if (TorsoBody < 0 || TorsoBody >= _bodies.Length)
            {
                error = "the torso is not one of the bodies";
                return false;
            }
            if (PhysicsDt <= 0f || Decimation < 1)
            {
                error = "physics dt and decimation must be positive";
                return false;
            }
            error = string.Empty;
            return true;
        }

        private static float GeomForwardReach(CreatureGeomDef geom)
        {
            switch (geom.Type)
            {
                case CreatureGeomType.Capsule:
                {
                    Vector3 start;
                    Vector3 end;
                    if (geom.HasFromTo)
                    {
                        start = geom.From;
                        end = geom.To;
                    }
                    else
                    {
                        Vector3 axis = CreatureFrames.Rotate(geom.Rotation, Vector3.forward) * geom.HalfLength;
                        start = geom.Position - axis;
                        end = geom.Position + axis;
                    }
                    return Mathf.Max(start.x, end.x) + geom.Radius;
                }
                case CreatureGeomType.Box:
                {
                    Vector3 axisX = CreatureFrames.Rotate(geom.Rotation, Vector3.right);
                    Vector3 axisY = CreatureFrames.Rotate(geom.Rotation, Vector3.up);
                    Vector3 axisZ = CreatureFrames.Rotate(geom.Rotation, Vector3.forward);
                    return geom.Position.x + Mathf.Abs(axisX.x) * geom.HalfExtents.x
                         + Mathf.Abs(axisY.x) * geom.HalfExtents.y + Mathf.Abs(axisZ.x) * geom.HalfExtents.z;
                }
                default:
                    return geom.Position.x + geom.Radius;
            }
        }
    }
}
