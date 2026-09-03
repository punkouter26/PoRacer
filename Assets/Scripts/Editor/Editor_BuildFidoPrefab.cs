using System.IO;
using Creature;
using Mujoco;
using PoRacer.Agents;
using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Builds the racing prefab for Fido out of Assets/Creature/creature.xml.
    ///
    /// Menu: PoRacer > Creatures > Rebuild Fido Prefab
    ///
    /// The MJCF importer is Editor-only, so the rig cannot be produced at runtime. Run
    /// this whenever creature.xml changes; the catalog points at the prefab it writes.
    /// The prefab holds Fido and nothing else — the solver options and the ground plane
    /// that also come out of the MJCF belong to the world, and Systems_MujocoWorld builds
    /// those once per race instead of once per racer.
    /// </summary>
    public static class Editor_BuildFidoPrefab
    {
        private const string XML_PATH = "Assets/Creature/creature.xml";
        private const string POLICY_PATH = "Assets/Creature/policy.json";
        private const string MATERIAL_PATH = "Assets/Art/Materials/M_Creature.mat";
        private const string PREFAB_DIR = "Assets/Agents/Fido_v01";
        private const string PREFAB_PATH = PREFAB_DIR + "/Fido.prefab";

        /// <summary>
        /// Physics steps each action is held for. The project runs a 0.005 s fixed
        /// timestep and the plug-in always takes MuJoCo's timestep from there, so 4 steps
        /// give a 0.020 s control period — exactly the 50 Hz the policy trained at.
        ///
        /// The drop ships 5, which is right only at its own 0.004 s timestep; at 0.005 it
        /// would control Fido at 40 Hz and CreatureAgent.CheckTiming would say so. Moving
        /// the project to 0.004 to suit Fido is the worse trade: it breaks MujocoBiped's
        /// exact 0.025 s policy step and forces DecisionPeriod 20 -> 25 on all 13
        /// ML-Agents prefabs (see CLAUDE.md). Holding the control rate exact and letting
        /// the substep be 0.005 instead of 0.004 costs only a little solver accuracy.
        /// </summary>
        private const int ACTION_DECIMATION = 4;

        [MenuItem("PoRacer/Creatures/Rebuild Fido Prefab")]
        public static void Build()
        {
            string error = BuildInternal();
            if (error != null)
            {
                // Logged, not shown in a dialog: a modal stalls every headless
                // and bridge-driven invocation of this action, and the message
                // was identical to the log line beside it.
                Debug.LogError("Editor_BuildFidoPrefab: " + error);
                return;
            }
            Debug.Log($"Editor_BuildFidoPrefab: wrote {PREFAB_PATH}.");
        }

        /// <summary>Returns null on success, or a message describing what stopped it.</summary>
        internal static string BuildInternal()
        {
            if (!File.Exists(XML_PATH))
            {
                return $"{XML_PATH} not found.";
            }
            var policy = AssetDatabase.LoadAssetAtPath<TextAsset>(POLICY_PATH);
            if (policy == null)
            {
                return $"{POLICY_PATH} not found.";
            }
            var material = AssetDatabase.LoadAssetAtPath<Material>(MATERIAL_PATH);
            if (material == null)
            {
                return $"{MATERIAL_PATH} not found — Fido would not tint like the other racers.";
            }

            GameObject root = new MjImporterWithAssets().ImportFile(XML_PATH);
            if (root == null)
            {
                return $"the MuJoCo importer returned nothing for {XML_PATH}.";
            }

            try
            {
                StripWorldComponents(root);
                root.name = "Fido";
                // Authored facing +Z. Systems_Spawn instantiates with the prefab's own
                // rotation rather than identity, so this is what puts Fido on the grid
                // pointing down the track; Agent_Fido.RestRotation repeats it for rescues.
                root.transform.rotation = Quaternion.Euler(0f, -90f, 0f);

                ApplyMaterial(root, material);

                CreatureAgent agent = root.AddComponent<CreatureAgent>();
                agent.policyJson = policy;
                agent.actionDecimation = ACTION_DECIMATION;
                // The importer drops <keyframe>, so Fido's trained stance travels in
                // policy.json and CreatureAgent writes it on the first step.
                agent.applyHomePose = true;
                agent.logBindings = false;

                root.AddComponent<Agent_Fido>();

                Directory.CreateDirectory(PREFAB_DIR);
                PrefabUtility.SaveAsPrefabAsset(root, PREFAB_PATH);
                AssetDatabase.ImportAsset(PREFAB_PATH, ImportAssetOptions.ForceUpdate);
                return null;
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        /// <summary>
        /// Drops the two things the MJCF describes that are the world's, not Fido's: the
        /// &lt;option&gt; block the importer turns into an MjGlobalSettings, and the floor
        /// plane. Systems_MujocoWorld provides both once per race. Left in the prefab,
        /// every racer would contribute a duplicate MjGlobalSettings and a coincident
        /// ground plane to the one shared MuJoCo model.
        /// </summary>
        private static void StripWorldComponents(GameObject root)
        {
            var settings = root.GetComponentInChildren<MjGlobalSettings>(true);
            if (settings != null)
            {
                Object.DestroyImmediate(settings.gameObject);
            }

            // The floor is the only geom that is not parented under a body.
            MjGeom[] geoms = root.GetComponentsInChildren<MjGeom>(true);
            for (int geomIndex = 0; geomIndex < geoms.Length; geomIndex++)
            {
                MjGeom geom = geoms[geomIndex];
                if (geom.ShapeType == MjShapeComponent.ShapeTypes.Plane)
                {
                    Object.DestroyImmediate(geom.gameObject);
                }
            }
        }

        /// <summary>
        /// Puts every geom on the shared creature material, so Fido batches with the rest
        /// of the grid and answers the per-racer tint Systems_Spawn writes through a
        /// MaterialPropertyBlock.
        /// </summary>
        private static void ApplyMaterial(GameObject root, Material material)
        {
            MeshRenderer[] renderers = root.GetComponentsInChildren<MeshRenderer>(true);
            for (int rendererIndex = 0; rendererIndex < renderers.Length; rendererIndex++)
            {
                renderers[rendererIndex].sharedMaterial = material;
            }
        }
    }
}
