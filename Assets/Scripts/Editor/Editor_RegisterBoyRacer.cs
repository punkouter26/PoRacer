using System.IO;
using Boy;
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
    /// Puts the Boy on the race grid: creates the <c>Assets/Prefabs/Boy_v01.prefab</c> variant
    /// (Boy.prefab + <see cref="Agent_Boy"/>) and registers it in the CreatureCatalog, the same
    /// shape as the IsaacH1_v01 entry. Idempotent: re-running updates the existing entry, which
    /// is how the entry picks up Boy.onnx once export_bundle.py has produced it.
    ///
    /// Until the ONNX exists the entry has no model, so <c>HasBrain</c> is false and the menu
    /// files the Boy under "coming soon" instead of racing a brainless rig.
    /// </summary>
    public static class Editor_RegisterBoyRacer
    {
        private const string VARIANT_PATH = "Assets/Prefabs/Boy_v01.prefab";
        private const string ENTRY_ID = "Boy_v01";
        private const string DISPLAY_NAME = "Boy";

        [MenuItem("PoRacer/Creatures/Register Boy Racer")]
        public static void Register()
        {
            GameObject source = AssetDatabase.LoadAssetAtPath<GameObject>(BoyPaths.Prefab);
            if (source == null)
            {
                Debug.LogError($"[Boy] {BoyPaths.Prefab} not found. Run Boy > Build Prefab first.");
                return;
            }

            BoyRigAsset rig = AssetDatabase.LoadAssetAtPath<BoyRigAsset>(BoyPaths.RigAsset);
            GameObject variant = AssetDatabase.LoadAssetAtPath<GameObject>(VARIANT_PATH);
            if (variant == null)
            {
                GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(source);
                instance.name = ENTRY_ID;
                if (instance.GetComponent<Agent_Boy>() == null)
                {
                    instance.AddComponent<Agent_Boy>();
                }
                Directory.CreateDirectory(Path.GetDirectoryName(VARIANT_PATH));
                variant = PrefabUtility.SaveAsPrefabAsset(instance, VARIANT_PATH);
                Object.DestroyImmediate(instance);
                if (variant == null)
                {
                    Debug.LogError($"[Boy] could not save {VARIANT_PATH}");
                    return;
                }
            }

            string[] guids = AssetDatabase.FindAssets("t:CreatureCatalog");
            if (guids.Length == 0)
            {
                Debug.LogError("[Boy] no CreatureCatalog asset in the project.");
                return;
            }
            string catalogPath = AssetDatabase.GUIDToAssetPath(guids[0]);
            CreatureCatalog catalog = AssetDatabase.LoadAssetAtPath<CreatureCatalog>(catalogPath);

            Object model = null;
#if ISAACPORTS_HAS_INFERENCE
            model = AssetDatabase.LoadAssetAtPath<ModelAsset>(BoyPaths.Onnx);
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

            Debug.Log($"[Boy] registered '{ENTRY_ID}' in {catalogPath} (entry {index}): prefab {VARIANT_PATH}, " +
                      $"model {(model != null ? BoyPaths.Onnx : "NONE yet - 'coming soon' until export_bundle.py runs")}, " +
                      $"spawnHeight {spawnHeight:F3}");
        }
    }
}
