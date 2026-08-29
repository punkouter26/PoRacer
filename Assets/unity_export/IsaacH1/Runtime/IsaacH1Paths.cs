using System.IO;
using UnityEngine;

using PoRacer.IsaacPorts;

namespace IsaacH1
{
    /// <summary>
    /// Asset paths and the reference-recording loaders.
    ///
    /// These live in the RUNTIME assembly, not the editor one, for a specific reason: a
    /// test assembly whose asmdef is Editor-only is classified by the Test Runner as
    /// EditMode, and EditMode never runs FixedUpdate - every dynamics rung would silently
    /// measure nothing. Keeping the loaders here lets IsaacH1.Tests be an all-platform
    /// (PlayMode) assembly that references only IsaacH1.Runtime.
    /// </summary>
    public static class IsaacH1Paths
    {
        public const string Root = "Assets/unity_export/IsaacH1";
        public const string RigJson = Root + "/IsaacH1_rig.json";
        public const string RigAsset = Root + "/IsaacH1Rig.asset";
        public const string Prefab = Root + "/IsaacH1.prefab";

        /// <summary>
        /// NOTE the extension. Unity 6.5 renamed the class PhysicMaterial -&gt;
        /// PhysicsMaterial but kept the legacy ASSET extension: a file named
        /// "*.physicsMaterial" is given DefaultImporter and never loads (verified in this
        /// editor: .physicsMaterial -&gt; loaded=False, .physicMaterial -&gt; loaded=True).
        /// Recorded as a deviation in README_UNITY.md.
        /// </summary>
        public const string Material = Root + "/PM_IsaacH1.physicMaterial";

        public const string Onnx = Root + "/IsaacH1.onnx";
        public const string Reference = Root + "/isaac_reference.json";
        public const string KinematicsReference = Root + "/kinematics_reference.json";

        /// <summary>Recorded observations and actions - the inference-path contract.</summary>
        public static bool TryLoadReference(out float[][] obs, out float[][] actions, out string error)
        {
            obs = null; actions = null; error = null;
            if (!File.Exists(Reference)) { error = $"{Reference} not found"; return false; }

            var root = MiniJson.Parse(File.ReadAllText(Reference)) as System.Collections.Generic.Dictionary<string, object>;
            var steps = MiniJson.Arr(root, "steps");
            if (steps == null || steps.Count == 0) { error = "reference has no 'steps' array"; return false; }

            obs = new float[steps.Count][];
            actions = new float[steps.Count][];
            for (int i = 0; i < steps.Count; i++)
            {
                var s = steps[i] as System.Collections.Generic.Dictionary<string, object>;
                obs[i] = MiniJson.FloatArray(s, "obs");
                actions[i] = MiniJson.FloatArray(s, "action");
            }
            return true;
        }

        /// <summary>Recorded root pose and joint state - the physics-path contract.</summary>
        public static bool TryLoadReferenceStates(out Vector3[] rootPosIsaac, out Vector4[] rootQuatXyzw,
                                                  out float[][] jointPos, out string error)
        {
            rootPosIsaac = null; rootQuatXyzw = null; jointPos = null; error = null;
            if (!File.Exists(Reference)) { error = $"{Reference} not found"; return false; }

            var root = MiniJson.Parse(File.ReadAllText(Reference)) as System.Collections.Generic.Dictionary<string, object>;
            var steps = MiniJson.Arr(root, "steps");
            if (steps == null) { error = "reference has no 'steps' array"; return false; }

            rootPosIsaac = new Vector3[steps.Count];
            rootQuatXyzw = new Vector4[steps.Count];
            jointPos = new float[steps.Count][];
            for (int i = 0; i < steps.Count; i++)
            {
                var s = steps[i] as System.Collections.Generic.Dictionary<string, object>;
                rootPosIsaac[i] = MiniJson.Vec3(s, "root_pos_w");
                // named _xyzw in this copy precisely because it IS xyzw - RIG_AUDIT.md D
                rootQuatXyzw[i] = MiniJson.Vec4(s, "root_quat_w_xyzw");
                jointPos[i] = MiniJson.FloatArray(s, "joint_pos");
            }
            return true;
        }
    }
}
