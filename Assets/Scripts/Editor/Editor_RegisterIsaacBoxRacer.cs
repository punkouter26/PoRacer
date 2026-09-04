using System.IO;
using IsaacBox;
using PoRacer.Agents;
using PoRacer.Models;
using UnityEditor;
using UnityEngine;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Puts the IsaacBox on the race grid: creates the <c>Assets/Prefabs/IsaacBox_v01.prefab</c> variant
    /// (IsaacBox.prefab + <see cref="Agent_IsaacBox"/>) and registers it in the CreatureCatalog, the same
    /// shape as the IsaacH1_v01 entry. Idempotent: re-running updates the existing entry, which
    /// is how the entry picks up IsaacBox.onnx once export_bundle.py has produced it.
    ///
    /// Until the ONNX exists the entry has no model, so <c>HasBrain</c> is false and the menu
    /// files the IsaacBox under "coming soon" instead of racing a brainless rig.
    /// </summary>
    public static class Editor_RegisterIsaacBoxRacer
    {
        private const string VARIANT_PATH = "Assets/Prefabs/IsaacBox_v01.prefab";
        private const string ENTRY_ID = "IsaacBox_v01";
        private const string DISPLAY_NAME = "IsaacBox";

        public static void Register()
        {
            GameObject source = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBoxPaths.Prefab);
            if (source == null)
            {
                Debug.LogError($"[IsaacBox] {IsaacBoxPaths.Prefab} not found. Run IsaacBoxSetup.BuildPrefab() first.");
                return;
            }

            IsaacBoxRigAsset rig = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset);
            GameObject variant = AssetDatabase.LoadAssetAtPath<GameObject>(VARIANT_PATH);
            if (variant == null)
            {
                GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(source);
                instance.name = ENTRY_ID;
                if (instance.GetComponent<Agent_IsaacBox>() == null)
                {
                    instance.AddComponent<Agent_IsaacBox>();
                }
                Directory.CreateDirectory(Path.GetDirectoryName(VARIANT_PATH));
                variant = PrefabUtility.SaveAsPrefabAsset(instance, VARIANT_PATH);
                Object.DestroyImmediate(instance);
                if (variant == null)
                {
                    Debug.LogError($"[IsaacBox] could not save {VARIANT_PATH}");
                    return;
                }
            }

            string[] guids = AssetDatabase.FindAssets("t:CreatureCatalog");
            if (guids.Length == 0)
            {
                Debug.LogError("[IsaacBox] no CreatureCatalog asset in the project.");
                return;
            }
            string catalogPath = AssetDatabase.GUIDToAssetPath(guids[0]);
            CreatureCatalog catalog = AssetDatabase.LoadAssetAtPath<CreatureCatalog>(catalogPath);

            Object model = null;
#if ISAACPORTS_HAS_INFERENCE
            model = AssetDatabase.LoadAssetAtPath<ModelAsset>(IsaacBoxPaths.Onnx);
#endif
            // the spawner adds +0.05 m to spawnHeight; land exactly on Isaac's spawn height
            float spawnHeight = rig != null ? rig.spawnPosIsaac.z - 0.05f : 0.71f;

            var so = new SerializedObject(catalog);
            SerializedProperty entries = so.FindProperty("_entries");
            int index = -1;
            for (int entryIndex = 0; entryIndex < entries.arraySize; entryIndex++)
            {
                if (entries.GetArrayElementAtIndex(entryIndex).FindPropertyRelative("id").stringValue == ENTRY_ID)
                {
                    index = entryIndex;
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
            entry.FindPropertyRelative("prefab").objectReferenceValue = variant;
            entry.FindPropertyRelative("model").objectReferenceValue = model;
            entry.FindPropertyRelative("spawnHeight").floatValue = spawnHeight;
            entry.FindPropertyRelative("brainInPrefab").boolValue = false;
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(catalog);
            AssetDatabase.SaveAssets();

            Debug.Log($"[IsaacBox] registered '{ENTRY_ID}' in {catalogPath} (entry {index}): prefab {VARIANT_PATH}, " +
                      $"model {(model != null ? IsaacBoxPaths.Onnx : "NONE yet - 'coming soon' until export_bundle.py runs")}, " +
                      $"spawnHeight {spawnHeight:F3}");
        }
    }
}
