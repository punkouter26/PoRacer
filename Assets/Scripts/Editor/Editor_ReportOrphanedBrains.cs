using System.IO;
using PoRacer.Models;
using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Read-only companion to Editor_AutoBake: lists Assets/Agents/&lt;Name&gt;_vNN
    /// folders that have no CreatureCatalog entry at all (a failed or abandoned
    /// bake, not a superseded-but-still-ELO-racing version — those are kept on
    /// purpose so old and new brains can compare). Never deletes anything; the
    /// developer decides what to do with what's reported.
    /// </summary>
    public static class Editor_ReportOrphanedBrains
    {
        [MenuItem("PoRacer/Report Orphaned Creature Brains")]
        private static void Report()
        {
            const string agentsRoot = "Assets/Agents";
            if (!Directory.Exists(agentsRoot))
            {
                Debug.Log("[OrphanReport] No Assets/Agents folder found.");
                return;
            }

            var catalog = AssetDatabase.LoadAssetAtPath<CreatureCatalog>("Assets/Settings/CreatureCatalog.asset");
            if (catalog == null)
            {
                Debug.LogWarning("[OrphanReport] CreatureCatalog.asset not found; cannot cross-check.");
                return;
            }

            var catalogIds = new System.Collections.Generic.HashSet<string>();
            for (int entryIndex = 0; entryIndex < catalog.Entries.Count; entryIndex++)
            {
                catalogIds.Add(catalog.Entries[entryIndex].id);
            }

            int orphanCount = 0;
            foreach (string versionDir in Directory.GetDirectories(agentsRoot))
            {
                string versionName = Path.GetFileName(versionDir);
                if (catalogIds.Contains(versionName))
                {
                    continue;
                }
                orphanCount++;
                Debug.Log($"[OrphanReport] '{versionName}' has no CreatureCatalog entry — {versionDir}");
            }

            if (orphanCount == 0)
            {
                Debug.Log("[OrphanReport] No orphaned brain folders found. Every Assets/Agents/* folder has a catalog entry.");
            }
            else
            {
                Debug.Log($"[OrphanReport] {orphanCount} orphaned folder(s) found. These are safe to delete manually — "
                    + "they were never wired to a racer. Versions that ARE in the catalog are kept on purpose for ELO comparison.");
            }
        }
    }
}
