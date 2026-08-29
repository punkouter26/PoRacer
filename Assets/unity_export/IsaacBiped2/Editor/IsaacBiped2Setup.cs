using System.IO;
using UnityEditor;
using UnityEngine;
#if ISAAC_BIPED2_INFERENCE
using Unity.InferenceEngine;
#endif

namespace IsaacBiped2.Editor
{
    /// <summary>
    /// Prefab builder for the Isaac biped. Builds the rig from the URDF, attaches the agent, wires
    /// the ONNX policy and saves the prefab. Never creates a scene and never touches project
    /// settings, layers or build settings.
    /// </summary>
    public static class IsaacBiped2Setup
    {
        public const string FOLDER = "Assets/unity_export/IsaacBiped2";
        public const string URDF_PATH = FOLDER + "/robot/biped.urdf";
        public const string ONNX_PATH = FOLDER + "/IsaacBiped2.onnx";
        // NOTE: ".asset", not ".physicsMaterial". AssetDatabase.CreateAsset refuses the latter in
        // Unity 6 ("should not be used to create a file of type 'physicsMaterial'") and writes a
        // file the importer then loads as a DefaultAsset, so every collider referencing it silently
        // falls back to Unity's default 0.6/0.6 Average material instead of Isaac's 0.5/0.5.
        public const string PHYSICS_MATERIAL_PATH = FOLDER + "/PM_IsaacBiped2.asset";
        public const string PREFAB_PATH = FOLDER + "/IsaacBiped2.prefab";
        public const string PREFAB_NAME = "IsaacBiped2";
        private const string MENU = "PoRacer/Isaac Biped 2/";

        /// <summary>Torso spawn height [m]; the torso origin sits on the hip line (standing 0.66).</summary>
        private const float SPAWN_HEIGHT = 0.68f;

        [MenuItem(MENU + "Build Prefab", priority = 0)]
        public static void BuildPrefab()
        {
            GameObject prefab = BuildPrefabAsset();
            if (prefab != null)
            {
                Selection.activeObject = prefab;
                EditorGUIUtility.PingObject(prefab);
            }
        }

        /// <summary>Builds and saves the prefab; returns the saved prefab asset (null on failure).</summary>
        public static GameObject BuildPrefabAsset()
        {
            if (!File.Exists(URDF_PATH))
            {
                Debug.LogError($"[IsaacBiped2] URDF not found at {URDF_PATH}");
                return null;
            }
            string urdfXml = File.ReadAllText(URDF_PATH);
            PhysicsMaterial physicsMaterial = GetOrCreatePhysicsMaterial();
            GameObject root = IsaacBiped2RigBuilder.Build(urdfXml, PREFAB_NAME, physicsMaterial);
            if (root == null)
            {
                Debug.LogError("[IsaacBiped2] rig builder returned nothing.");
                return null;
            }
            root.transform.position = new Vector3(0f, SPAWN_HEIGHT, 0f);

            IsaacBiped2Agent agent = root.GetComponent<IsaacBiped2Agent>();
            if (agent == null)
            {
                agent = root.AddComponent<IsaacBiped2Agent>();
            }
#if ISAAC_BIPED2_INFERENCE
            var serialized = new SerializedObject(agent);
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(ONNX_PATH);
            if (model == null)
            {
                Debug.LogWarning($"[IsaacBiped2] {ONNX_PATH} did not import as a ModelAsset; " +
                                 "the prefab is saved without a policy.");
            }
            serialized.FindProperty("_model").objectReferenceValue = model;
            serialized.ApplyModifiedPropertiesWithoutUndo();
#endif
            EditorUtility.SetDirty(agent);

            GameObject saved = PrefabUtility.SaveAsPrefabAsset(root, PREFAB_PATH, out bool success);
            Object.DestroyImmediate(root);
            if (!success)
            {
                Debug.LogError($"[IsaacBiped2] failed to save {PREFAB_PATH}");
                return null;
            }
            AssetDatabase.SaveAssets();
            Debug.Log($"[IsaacBiped2] prefab saved to {PREFAB_PATH} " +
                      $"({saved.GetComponentsInChildren<ArticulationBody>(true).Length} bodies, " +
                      $"{saved.GetComponentsInChildren<Collider>(true).Length} colliders)");
            return saved;
        }

        /// <summary>
        /// Isaac's ground material: 0.5/0.5 friction, no bounce. Minimum combine so the racer's
        /// friction is what the policy trained against regardless of what the track material says.
        /// </summary>
        public static PhysicsMaterial GetOrCreatePhysicsMaterial()
        {
            var existing = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(PHYSICS_MATERIAL_PATH);
            if (existing != null)
            {
                return existing;
            }
            var material = new PhysicsMaterial("PM_IsaacBiped2")
            {
                staticFriction = 0.5f,
                dynamicFriction = 0.5f,
                bounciness = 0f,
                frictionCombine = PhysicsMaterialCombine.Minimum,
                bounceCombine = PhysicsMaterialCombine.Minimum,
            };
            AssetDatabase.CreateAsset(material, PHYSICS_MATERIAL_PATH);
            return material;
        }
    }
}
