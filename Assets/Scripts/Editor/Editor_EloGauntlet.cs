#if UNITY_EDITOR
using PoRacer.Models;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// ELO gauntlet (project rule: evaluate policies by ELO, not mean reward).
    /// Select checkpoint .onnx assets in the Project window and run "Add Selected
    /// Brains": each becomes its own catalog entry (id "worm@Worm-4999588") racing
    /// under the base creature's prefab. Run races as usual — the endless loop and
    /// Systems_Elo then rank every checkpoint on the live leaderboard. "Remove
    /// Gauntlet Entries" restores the catalog afterwards.
    /// </summary>
    public static class Editor_EloGauntlet
    {
        public static void AddSelectedBrains()
        {
            CreatureCatalog catalog = LoadCatalog();
            ModelAsset[] models = Selection.GetFiltered<ModelAsset>(SelectionMode.Assets);
            if (catalog == null || models.Length == 0)
            {
                Debug.LogWarning("Select one or more .onnx model assets first (and ensure a CreatureCatalog exists).");
                return;
            }

            var so = new SerializedObject(catalog);
            SerializedProperty entries = so.FindProperty("_entries");
            int baseCount = entries.arraySize;
            int added = 0;
            for (int modelIndex = 0; modelIndex < models.Length; modelIndex++)
            {
                ModelAsset model = models[modelIndex];
                int baseIndex = FindBaseEntry(entries, baseCount, model.name);
                if (baseIndex < 0)
                {
                    Debug.LogWarning($"No catalog entry matches '{model.name}' by id; rename the asset " +
                        "to contain the creature id (e.g. 'worm-4999588.onnx') or add the entry manually.");
                    continue;
                }
                SerializedProperty baseEntry = entries.GetArrayElementAtIndex(baseIndex);
                string baseId = baseEntry.FindPropertyRelative("id").stringValue;
                string gauntletId = $"{baseId}@{model.name}";
                if (EntryExists(entries, gauntletId))
                {
                    continue;
                }
                entries.InsertArrayElementAtIndex(entries.arraySize);
                SerializedProperty clone = entries.GetArrayElementAtIndex(entries.arraySize - 1);
                clone.FindPropertyRelative("id").stringValue = gauntletId;
                clone.FindPropertyRelative("displayName").stringValue =
                    $"{baseEntry.FindPropertyRelative("displayName").stringValue} [{model.name}]";
                clone.FindPropertyRelative("prefab").objectReferenceValue =
                    baseEntry.FindPropertyRelative("prefab").objectReferenceValue;
                clone.FindPropertyRelative("model").objectReferenceValue = model;
                clone.FindPropertyRelative("spawnHeight").floatValue =
                    baseEntry.FindPropertyRelative("spawnHeight").floatValue;
                added++;
            }
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
            Debug.Log($"Gauntlet: added {added} checkpoint entries. Race them (menu counts per entry) " +
                "and compare ELO on the leaderboard; remove entries via the Gauntlet menu when done.");
        }

        public static void RemoveGauntletEntries()
        {
            CreatureCatalog catalog = LoadCatalog();
            if (catalog == null)
            {
                return;
            }
            var so = new SerializedObject(catalog);
            SerializedProperty entries = so.FindProperty("_entries");
            int removed = 0;
            for (int entryIndex = entries.arraySize - 1; entryIndex >= 0; entryIndex--)
            {
                string id = entries.GetArrayElementAtIndex(entryIndex).FindPropertyRelative("id").stringValue;
                if (id.Contains("@"))
                {
                    entries.DeleteArrayElementAtIndex(entryIndex);
                    removed++;
                }
            }
            so.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
            Debug.Log($"Gauntlet: removed {removed} checkpoint entries.");
        }

        private static CreatureCatalog LoadCatalog()
        {
            string[] guids = AssetDatabase.FindAssets("t:CreatureCatalog");
            if (guids.Length == 0)
            {
                Debug.LogWarning("No CreatureCatalog asset found.");
                return null;
            }
            return AssetDatabase.LoadAssetAtPath<CreatureCatalog>(AssetDatabase.GUIDToAssetPath(guids[0]));
        }

        private static int FindBaseEntry(SerializedProperty entries, int baseCount, string modelName)
        {
            string lowered = modelName.ToLowerInvariant();
            for (int entryIndex = 0; entryIndex < baseCount; entryIndex++)
            {
                string id = entries.GetArrayElementAtIndex(entryIndex).FindPropertyRelative("id").stringValue;
                if (string.IsNullOrEmpty(id) || id.Contains("@"))
                {
                    continue;
                }
                // Catalog ids are versioned ("Worm_v01") while checkpoint files are
                // "Worm-4999588.onnx": match on the creature name before "_v".
                int versionCut = id.IndexOf("_v", System.StringComparison.OrdinalIgnoreCase);
                string creature = (versionCut > 0 ? id.Substring(0, versionCut) : id).ToLowerInvariant();
                if (lowered.StartsWith(creature))
                {
                    return entryIndex;
                }
            }
            return -1;
        }

        private static bool EntryExists(SerializedProperty entries, string id)
        {
            for (int entryIndex = 0; entryIndex < entries.arraySize; entryIndex++)
            {
                if (entries.GetArrayElementAtIndex(entryIndex).FindPropertyRelative("id").stringValue == id)
                {
                    return true;
                }
            }
            return false;
        }
    }
}
#endif
