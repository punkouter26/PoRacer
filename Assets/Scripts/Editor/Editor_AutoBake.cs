using System.IO;
using PoRacer.Models;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Completes the overnight-evolution loop: when scripts/evolve.ps1 stages a new
    /// brain (Assets/Agents/&lt;Name&gt;_vNN/&lt;Name&gt;_vNN.onnx + .autobake marker), this adds
    /// a catalog entry on editor load so the new version races the old ones.
    /// The prefab and spawn height are cloned from the newest existing entry of the
    /// same creature family.
    /// </summary>
    [InitializeOnLoad]
    public static class Editor_AutoBake
    {
        static Editor_AutoBake()
        {
            EditorApplication.delayCall += BakePending;
        }

        private static void BakePending()
        {
            string agentsRoot = "Assets/Agents";
            if (!Directory.Exists(agentsRoot))
            {
                return;
            }
            var catalog = AssetDatabase.LoadAssetAtPath<CreatureCatalog>("Assets/Settings/CreatureCatalog.asset");
            if (catalog == null)
            {
                return;
            }

            foreach (string markerPath in Directory.GetFiles(agentsRoot, "*.autobake", SearchOption.AllDirectories))
            {
                string versionName = Path.GetFileNameWithoutExtension(markerPath);        // e.g. Worm_v02
                string family = versionName.Substring(0, versionName.LastIndexOf("_v")); // e.g. Worm
                string onnxPath = Path.Combine(Path.GetDirectoryName(markerPath), versionName + ".onnx").Replace('\\', '/');
                AssetDatabase.ImportAsset(onnxPath);
                var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(onnxPath);
                if (model == null)
                {
                    Debug.LogWarning($"[AutoBake] {onnxPath} did not import as a model; leaving marker for next time.");
                    continue;
                }

                var so = new SerializedObject(catalog);
                SerializedProperty entries = so.FindProperty("_entries");
                int templateIndex = -1;
                bool alreadyPresent = false;
                for (int entryIndex = 0; entryIndex < entries.arraySize; entryIndex++)
                {
                    SerializedProperty entry = entries.GetArrayElementAtIndex(entryIndex);
                    string id = entry.FindPropertyRelative("id").stringValue;
                    if (id == versionName)
                    {
                        alreadyPresent = true;
                        entry.FindPropertyRelative("model").objectReferenceValue = model;
                    }
                    if (id.StartsWith(family + "_v") && entry.FindPropertyRelative("prefab").objectReferenceValue != null)
                    {
                        templateIndex = entryIndex;
                    }
                }
                if (!alreadyPresent)
                {
                    if (templateIndex < 0)
                    {
                        Debug.LogWarning($"[AutoBake] No template entry for family '{family}'; skipping {versionName}.");
                        continue;
                    }
                    SerializedProperty template = entries.GetArrayElementAtIndex(templateIndex);
                    entries.InsertArrayElementAtIndex(entries.arraySize);
                    SerializedProperty added = entries.GetArrayElementAtIndex(entries.arraySize - 1);
                    added.FindPropertyRelative("id").stringValue = versionName;
                    added.FindPropertyRelative("displayName").stringValue = versionName.Replace("_v", " v");
                    added.FindPropertyRelative("prefab").objectReferenceValue = template.FindPropertyRelative("prefab").objectReferenceValue;
                    added.FindPropertyRelative("model").objectReferenceValue = model;
                    added.FindPropertyRelative("spawnHeight").floatValue = template.FindPropertyRelative("spawnHeight").floatValue;
                }
                so.ApplyModifiedProperties();
                AssetDatabase.SaveAssets();
                File.Delete(markerPath);
                string metaPath = markerPath + ".meta";
                if (File.Exists(metaPath))
                {
                    File.Delete(metaPath);
                }
                Debug.Log($"[AutoBake] Baked {versionName} into the creature catalog.");
            }
            AssetDatabase.Refresh();
        }
    }
}
