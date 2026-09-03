using System.Collections.Generic;
using System.IO;
using UnityEngine;

using PoRacer.IsaacPorts;

namespace IsaacBox
{
    /// <summary>
    /// Asset paths and the reference-recording loaders for the IsaacBox port.
    ///
    /// These live in the RUNTIME assembly, not the editor one, so the PlayMode test
    /// assembly can reach them: a test asmdef that is Editor-only is classified as
    /// EditMode, and EditMode never runs FixedUpdate.
    /// </summary>
    public static class IsaacBoxPaths
    {
        public const string Root = "Assets/unity_export/IsaacBox";
        public const string RigJson = Root + "/isaacbox_rig.json";
        public const string RigAsset = Root + "/IsaacBoxRig.asset";
        public const string Prefab = Root + "/IsaacBox.prefab";

        /// <summary>
        /// The legacy extension on purpose: Unity 6.5 renamed the class to PhysicsMaterial
        /// but a "*.physicsMaterial" file gets DefaultImporter and never loads (verified in
        /// the IsaacH1 port).
        /// </summary>
        public const string Material = Root + "/PM_IsaacBox.physicMaterial";

        public const string Onnx = Root + "/IsaacBox.onnx";
        public const string Reference = Root + "/isaac_reference.json";
        public const string KinematicsReference = Root + "/kinematics_reference.json";
        public const string ExportReport = Root + "/export_report.json";

        /// <summary>The authored character; the builder reads its bones and skinned meshes.</summary>
        public const string Fbx = "Assets/Art/Models/IsaacBox_Character.fbx";

        /// <summary>Recorded observations and actions - the inference-path contract.</summary>
        public static bool TryLoadReference(out float[][] obs, out float[][] actions, out string error)
        {
            obs = null; actions = null; error = null;
            if (!File.Exists(Reference)) { error = $"{Reference} not found - run ISAAC/scripts/export_bundle.py"; return false; }

            var root = MiniJson.Parse(File.ReadAllText(Reference)) as Dictionary<string, object>;
            var steps = MiniJson.Arr(root, "steps");
            if (steps == null || steps.Count == 0) { error = "reference has no 'steps' array"; return false; }

            obs = new float[steps.Count][];
            actions = new float[steps.Count][];
            for (int i = 0; i < steps.Count; i++)
            {
                var s = steps[i] as Dictionary<string, object>;
                obs[i] = MiniJson.FloatArray(s, "obs");
                actions[i] = MiniJson.FloatArray(s, "action");
            }
            return true;
        }

        /// <summary>Recorded root pose, joint state and target - the physics-path contract.</summary>
        public static bool TryLoadReferenceStates(out Vector3[] rootPosIsaac, out Vector4[] rootQuatXyzw,
                                                  out float[][] jointPos, out Vector3[] targetPosIsaac,
                                                  out string error)
        {
            rootPosIsaac = null; rootQuatXyzw = null; jointPos = null; targetPosIsaac = null; error = null;
            if (!File.Exists(Reference)) { error = $"{Reference} not found"; return false; }

            var root = MiniJson.Parse(File.ReadAllText(Reference)) as Dictionary<string, object>;
            var steps = MiniJson.Arr(root, "steps");
            if (steps == null) { error = "reference has no 'steps' array"; return false; }

            rootPosIsaac = new Vector3[steps.Count];
            rootQuatXyzw = new Vector4[steps.Count];
            jointPos = new float[steps.Count][];
            targetPosIsaac = new Vector3[steps.Count];
            for (int i = 0; i < steps.Count; i++)
            {
                var s = steps[i] as Dictionary<string, object>;
                rootPosIsaac[i] = MiniJson.Vec3(s, "root_pos_w");
                rootQuatXyzw[i] = MiniJson.Vec4(s, "root_quat_w_xyzw");
                jointPos[i] = MiniJson.FloatArray(s, "joint_pos");
                targetPosIsaac[i] = MiniJson.Vec3(s, "target_pos_w");
            }
            return true;
        }
    }
}
