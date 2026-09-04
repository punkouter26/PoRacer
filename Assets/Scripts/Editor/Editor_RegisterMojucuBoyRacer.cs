using System.IO;
using Creature.MojucuBoy;
using Mujoco;
using PoRacer.Agents;
using PoRacer.Models;
using UnityEditor;
using UnityEngine;
#if CREATURE_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Puts MojucuBoy on the race grid: builds
    /// <c>Assets/Prefabs/MojucuBoy_v01.prefab</c> from the imported MJCF rig plus the
    /// authored character, and registers it in the CreatureCatalog the same shape as
    /// the IsaacBox_v01 entry. Idempotent -- re-running rebuilds the prefab and
    /// updates the existing entry, which is how it picks up a freshly exported ONNX.
    ///
    /// Until the ONNX exists the entry has no model, so <c>HasBrain</c> is false and
    /// MenuView files him under "coming soon" rather than racing a brainless rig.
    ///
    /// He keeps his authored textures. CLAUDE.md reserves RED for heuristic bots and
    /// GREEN for the baseline RL policy, and reads a creature that arrives with its
    /// own art as a variation -- the same call already recorded for IsaacBox. He is
    /// deliberately NOT recoloured.
    /// </summary>
    public static class Editor_RegisterMojucuBoyRacer
    {
        private const string PREFAB_PATH = "Assets/Prefabs/MojucuBoy_v01.prefab";
        private const string MJCF_PATH = "Assets/Agents/MojucuBoy_v01/mojucuboy_unity.xml";
        private const string RIG_JSON = "Assets/Agents/MojucuBoy_v01/mojucuboy_rig.json";
        private const string ONNX_PATH = "Assets/Agents/MojucuBoy_v01/MojucuBoy_v01.onnx";
        private const string GLB_PATH = "Assets/Boy_Character_mujoco.glb";
        private const string IMPORT_ASSETS = "Assets/Local/MjImports/mojucuboy";
        private const string ENTRY_ID = "MojucuBoy_v01";
        private const string DISPLAY_NAME = "MojucuBoy";

        /// <summary>
        /// Hips height the policy trained to stand at. Systems_Spawn adds +0.05 m, and
        /// Agent_MojucuBoy cancels that in Awake -- a trained policy has no slack for a
        /// drop a ragdoll shrugs off.
        /// </summary>
        private const float TRAINED_HIPS_HEIGHT = 0.7722f;

        public static void Register() => Debug.Log(Build());

        public static string Build()
        {
            var log = new System.Text.StringBuilder();

            // MjImporterWithAssets throws if its generated materials already exist, and
            // ImportString then swallows the exception and returns null. Clear first so
            // this stays re-runnable.
            if (AssetDatabase.IsValidFolder(IMPORT_ASSETS))
            {
                AssetDatabase.DeleteAsset(IMPORT_ASSETS);
                AssetDatabase.Refresh();
            }

            // Build in a scratch scene opened SINGLE, not Additive. org.mujoco resolves
            // components by name across every loaded scene, so building alongside an
            // already-open MojucuBoy scene fails with "More than one component named
            // 'abdomen_z' was found".
            var scene = UnityEditor.SceneManagement.EditorSceneManager.NewScene(
                UnityEditor.SceneManagement.NewSceneSetup.EmptyScene,
                UnityEditor.SceneManagement.NewSceneMode.Single);

            MjEngineTool.LoadPlugins();
            var importer = new MjImporterWithAssets();
            GameObject root = importer.ImportString(
                File.ReadAllText(MJCF_PATH), "mojucuboy", MJCF_PATH);
            if (root == null)
            {
                return "ABORT: MJCF import returned null; check the console for the "
                     + "exception ImportString swallowed.";
            }
            root.name = ENTRY_ID;

            // Collision primitives are debug geometry: the authored character is what
            // should be seen, and org.mujoco assigns them the built-in Standard
            // shader, which URP renders magenta anyway.
            //
            // STRIP the renderers rather than merely disabling them. A disabled
            // MeshRenderer still REFERENCES the material org.mujoco generated into
            // Assets/Local/MjImports/<name>/Resources/, and everything under a
            // Resources folder is baked into every build unconditionally -- 40
            // materials of dead weight in the shipped bundle. With no references
            // left, the whole generated folder can be deleted below.
            int hidden = 0;
            foreach (MjGeom geom in root.GetComponentsInChildren<MjGeom>(true))
            {
                var renderer = geom.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    Object.DestroyImmediate(renderer);
                    hidden++;
                }
                var mjFilter = geom.GetComponent<MjMeshFilter>();
                if (mjFilter != null)
                {
                    Object.DestroyImmediate(mjFilter);
                }
                var filter = geom.GetComponent<MeshFilter>();
                if (filter != null)
                {
                    Object.DestroyImmediate(filter);
                }
            }

            // The importer turns <option> into an MjGlobalSettings child. The race
            // scene already has one, built by Systems_MujocoWorld, and org.mujoco
            // throws "At most one MjGlobalSettings should be present" the moment a
            // second racer carrying one is spawned. Timing and solver settings come
            // from the scene's world, not from the racer.
            foreach (MjGlobalSettings stray in root.GetComponentsInChildren<MjGlobalSettings>(true))
            {
                Object.DestroyImmediate(stray.gameObject);
            }

            var glb = AssetDatabase.LoadAssetAtPath<GameObject>(GLB_PATH);
            if (glb == null)
            {
                return $"ABORT: {GLB_PATH} not found.";
            }
            var character = (GameObject)PrefabUtility.InstantiatePrefab(glb);
            character.name = "MojucuBoyCharacter";
            character.transform.SetParent(root.transform, false);
            // The MJCF carries a 180 deg facing yaw so the rig points down Unity +Z;
            // the GLB is authored unyawed and needs the same rotation before the
            // binder captures its bind-pose offsets.
            character.transform.SetLocalPositionAndRotation(
                Vector3.zero, Quaternion.Euler(0f, 180f, 0f));

            var controller = root.AddComponent<MojucuBoyController>();
            Assign(controller, "_rigJson", AssetDatabase.LoadAssetAtPath<TextAsset>(RIG_JSON));
            Object model = null;
#if CREATURE_HAS_INFERENCE
            model = AssetDatabase.LoadAssetAtPath<ModelAsset>(ONNX_PATH);
            Assign(controller, "_modelAsset", model);
#endif

            var binder = root.AddComponent<MojucuBoyVisualBinder>();
            Assign(binder, "_skinRoot", character.transform);
            Assign(binder, "_physicsRoot", root.transform);

            root.AddComponent<Agent_MojucuBoy>();

            Directory.CreateDirectory(Path.GetDirectoryName(PREFAB_PATH));
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, PREFAB_PATH);
            Object.DestroyImmediate(root);
            if (prefab == null)
            {
                return $"ABORT: could not save {PREFAB_PATH}";
            }
            log.Append($"prefab {PREFAB_PATH} ({hidden} collision renderers stripped)\n");

            // Nothing references the generated materials any more, so the Resources
            // folder that would otherwise ship inside every build can go.
            if (AssetDatabase.IsValidFolder(IMPORT_ASSETS))
            {
                AssetDatabase.DeleteAsset(IMPORT_ASSETS);
                AssetDatabase.Refresh();
                log.Append($"deleted {IMPORT_ASSETS} (Resources build bloat)\n");
            }

            string[] guids = AssetDatabase.FindAssets("t:CreatureCatalog");
            if (guids.Length == 0)
            {
                return log + "ABORT: no CreatureCatalog asset in the project.";
            }
            string catalogPath = AssetDatabase.GUIDToAssetPath(guids[0]);
            var catalog = AssetDatabase.LoadAssetAtPath<CreatureCatalog>(catalogPath);

            var so = new SerializedObject(catalog);
            SerializedProperty entries = so.FindProperty("_entries");
            int index = -1;
            for (int i = 0; i < entries.arraySize; i++)
            {
                if (entries.GetArrayElementAtIndex(i).FindPropertyRelative("id").stringValue == ENTRY_ID)
                {
                    index = i;
                    break;
                }
            }
            if (index < 0)
            {
                index = entries.arraySize;
                entries.InsertArrayElementAtIndex(index);
            }
            SerializedProperty entry = entries.GetArrayElementAtIndex(index);
            entry.FindPropertyRelative("id").stringValue = ENTRY_ID;
            entry.FindPropertyRelative("displayName").stringValue = DISPLAY_NAME;
            entry.FindPropertyRelative("prefab").objectReferenceValue = prefab;
            entry.FindPropertyRelative("model").objectReferenceValue = model;
            // Systems_Spawn adds +0.05 m; Agent_MojucuBoy cancels it in Awake, but the
            // catalog height should still name the pose the policy trained from.
            entry.FindPropertyRelative("spawnHeight").floatValue = TRAINED_HIPS_HEIGHT - 0.05f;
            entry.FindPropertyRelative("brainInPrefab").boolValue = false;
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(catalog);
            AssetDatabase.SaveAssets();

            log.Append($"catalog {catalogPath} entry {index}: id={ENTRY_ID} "
                     + $"model={(model != null ? "assigned" : "NONE -- 'coming soon'")} "
                     + $"spawnHeight={TRAINED_HIPS_HEIGHT - 0.05f:F3}\n");
            return log.ToString();
        }

        private static void Assign(Object target, string field, Object value)
        {
            var serialised = new SerializedObject(target);
            SerializedProperty property = serialised.FindProperty(field);
            if (property == null)
            {
                Debug.LogError($"field '{field}' not found on {target.GetType().Name}");
                return;
            }
            property.objectReferenceValue = value;
            serialised.ApplyModifiedPropertiesWithoutUndo();
        }
    }
}
