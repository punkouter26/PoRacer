using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace MujocoBiped
{
    /// <summary>
    /// Asset paths and the reference-file loaders. In the RUNTIME assembly so the
    /// PlayMode tests can reach them: a test assembly restricted to the Editor platform
    /// is classified as EditMode, where FixedUpdate never runs and rungs 1-6 would be
    /// meaningless.
    ///
    /// Nothing outside <see cref="Root"/> is ever read or written.
    /// </summary>
    public static class MujocoBipedPaths
    {
        public const string Root = "Assets/unity_export/MujocoBiped/";

        public const string RigJson = Root + "MujocoBiped_rig.json";
        public const string RigAsset = Root + "MujocoBipedRig.asset";
        public const string Prefab = Root + "MujocoBiped.prefab";

        /// <summary>
        /// ".physicMaterial", with one 's', is the ONLY extension Unity 6000.5 imports as
        /// a PhysicsMaterial. A file written to ".physicsMaterial" is byte-for-byte
        /// correct YAML and gets a .meta, but the importer does not claim the extension,
        /// so it lands as a DefaultAsset and every LoadAssetAtPath&lt;PhysicsMaterial&gt;
        /// returns null - silently, with the colliders falling back to PhysX defaults.
        /// (Assets/unity_export/spider/PM_IsaacSpider.physicsMaterial has this problem in
        /// this project today; fixing it belongs to that creature's owner, not here.)
        /// </summary>
        public const string Material = Root + "PM_MujocoBiped.physicMaterial";
        public const string Onnx = Root + "MujocoBiped.onnx";
        public const string Reference = Root + "mujoco_reference.json";
        public const string Kinematics = Root + "kinematics_reference.json";

        /// <summary>
        /// The recorded policy I/O. Returns false rather than throwing so a test can skip
        /// with a clear message instead of erroring out on a missing file.
        /// </summary>
        public static bool TryLoadReference(out float[][] observations, out float[][] actions,
                                            out string error)
        {
            observations = null;
            actions = null;
            error = null;

            if (!File.Exists(Reference))
            {
                error = $"{Reference} not found - run: python make_reference.py";
                return false;
            }

            try
            {
                var root = MiniJson.Parse(File.ReadAllText(Reference)) as Dictionary<string, object>;
                var traj = MiniJson.Arr(root, "trajectory");
                if (traj == null || traj.Count == 0)
                {
                    error = $"{Reference} has no 'trajectory' array";
                    return false;
                }

                observations = new float[traj.Count][];
                actions = new float[traj.Count][];
                for (int i = 0; i < traj.Count; i++)
                {
                    var step = traj[i] as Dictionary<string, object>;
                    observations[i] = MiniJson.FloatArray(step, "observation");
                    actions[i] = MiniJson.FloatArray(step, "action");
                }
                return true;
            }
            catch (Exception e)
            {
                error = $"{Reference} failed to parse: {e.Message}";
                return false;
            }
        }

        /// <summary>
        /// One recorded control step, still in MuJoCo coordinates. Enough state to put a
        /// Unity rig exactly where MuJoCo was and rebuild the observation from it.
        /// </summary>
        public class RecordedStep
        {
            public float[] observation;
            public float[] action;
            public Vector3 rootPosMuj;
            public Vector4 rootQuatMujWxyz;
            public Vector3 rootLinVelWorldMuj;
            public Vector3 rootAngVelBodyLocalMuj;
            public float[] jointPositionsRad;
            public float[] jointVelocitiesRaw;
            public Vector3 targetPosMuj;
        }

        /// <summary>The full recorded trajectory, for the observation-parity test.</summary>
        public static bool TryLoadStates(out RecordedStep[] steps, out string error)
        {
            steps = null;
            error = null;

            if (!File.Exists(Reference))
            {
                error = $"{Reference} not found - run: python make_reference.py";
                return false;
            }

            try
            {
                var root = MiniJson.Parse(File.ReadAllText(Reference)) as Dictionary<string, object>;
                var traj = MiniJson.Arr(root, "trajectory");
                if (traj == null || traj.Count == 0)
                {
                    error = $"{Reference} has no 'trajectory' array";
                    return false;
                }

                steps = new RecordedStep[traj.Count];
                for (int i = 0; i < traj.Count; i++)
                {
                    var s = traj[i] as Dictionary<string, object>;
                    steps[i] = new RecordedStep
                    {
                        observation = MiniJson.FloatArray(s, "observation"),
                        action = MiniJson.FloatArray(s, "action"),
                        rootPosMuj = MiniJson.Vec3(s, "rootPosMuj"),
                        rootQuatMujWxyz = MiniJson.Vec4(s, "rootQuatMujWxyz"),
                        rootLinVelWorldMuj = MiniJson.Vec3(s, "rootLinVelWorldMuj"),
                        rootAngVelBodyLocalMuj = MiniJson.Vec3(s, "rootAngVelBodyLocalMuj"),
                        jointPositionsRad = MiniJson.FloatArray(s, "jointPositionsRad"),
                        jointVelocitiesRaw = MiniJson.FloatArray(s, "jointVelocitiesRaw"),
                        targetPosMuj = MiniJson.Vec3(s, "targetPosMuj"),
                    };
                }
                return true;
            }
            catch (Exception e)
            {
                error = $"{Reference} failed to parse: {e.Message}";
                return false;
            }
        }

        /// <summary>One pose from kinematics_reference.json, still in MuJoCo coordinates.</summary>
        public class KinematicsPose
        {
            public string label;
            public Vector3 rootPosMuj;
            public Vector4 rootQuatMujWxyz;
            public float[] jointsRad;
            public string[] bodyNames;
            public Vector3[] bodyPosMuj;
            public Vector4[] bodyQuatMujWxyz;
        }

        /// <summary>
        /// The independent Python forward kinematics. Generated straight from the MJCF by
        /// gen_kinematics_reference.py, never from MujocoBiped_rig.json - which is what
        /// lets it catch an error the rig extraction and the rig builder would share.
        /// </summary>
        public static bool TryLoadKinematics(out KinematicsPose[] poses, out float toleranceMetres,
                                             out string error)
        {
            poses = null;
            toleranceMetres = 1e-3f;
            error = null;

            if (!File.Exists(Kinematics))
            {
                error = $"{Kinematics} not found - run: python gen_kinematics_reference.py";
                return false;
            }

            try
            {
                var root = MiniJson.Parse(File.ReadAllText(Kinematics)) as Dictionary<string, object>;
                toleranceMetres = MiniJson.Num(root, "toleranceMetres");
                var arr = MiniJson.Arr(root, "poses");
                if (arr == null || arr.Count == 0)
                {
                    error = $"{Kinematics} has no 'poses' array";
                    return false;
                }

                poses = new KinematicsPose[arr.Count];
                for (int i = 0; i < arr.Count; i++)
                {
                    var p = arr[i] as Dictionary<string, object>;
                    var bodies = MiniJson.Arr(p, "bodies");
                    var pose = new KinematicsPose
                    {
                        label = MiniJson.Str(p, "label"),
                        rootPosMuj = MiniJson.Vec3(p, "rootPosMuj"),
                        rootQuatMujWxyz = MiniJson.Vec4(p, "rootQuatMujWxyz"),
                        jointsRad = MiniJson.FloatArray(p, "jointsRad"),
                        bodyNames = new string[bodies.Count],
                        bodyPosMuj = new Vector3[bodies.Count],
                        bodyQuatMujWxyz = new Vector4[bodies.Count],
                    };
                    for (int b = 0; b < bodies.Count; b++)
                    {
                        var bd = bodies[b] as Dictionary<string, object>;
                        pose.bodyNames[b] = MiniJson.Str(bd, "name");
                        pose.bodyPosMuj[b] = MiniJson.Vec3(bd, "posMuj");
                        pose.bodyQuatMujWxyz[b] = MiniJson.Vec4(bd, "quatMujWxyz");
                    }
                    poses[i] = pose;
                }
                return true;
            }
            catch (Exception e)
            {
                error = $"{Kinematics} failed to parse: {e.Message}";
                return false;
            }
        }
    }
}
