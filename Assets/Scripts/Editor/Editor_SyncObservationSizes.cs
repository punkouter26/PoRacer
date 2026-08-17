#if UNITY_EDITOR
using PoRacer.Agents;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Rewrites every creature prefab's VectorObservationSize to match
    /// Agent_Creature's current observation layout (N*3 + 19). Run once after any
    /// observation change, then retrain: old .onnx brains expect the old size and
    /// will be rejected at inference until replaced.
    /// </summary>
    public static class Editor_SyncObservationSizes
    {
        private const int OBSERVATIONS_PER_JOINT = 3;
        private const int FIXED_OBSERVATIONS = 19;

        [MenuItem("PoRacer/Sync Agent Observation Sizes")]
        public static void Sync()
        {
            string[] guids = AssetDatabase.FindAssets("t:Prefab", new[] { "Assets/Prefabs" });
            int updated = 0;
            for (int guidIndex = 0; guidIndex < guids.Length; guidIndex++)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[guidIndex]);
                GameObject root = PrefabUtility.LoadPrefabContents(path);
                Agent_Creature agent = root.GetComponentInChildren<Agent_Creature>(true);
                BehaviorParameters behavior = root.GetComponentInChildren<BehaviorParameters>(true);
                if (agent != null && behavior != null)
                {
                    int jointCount = new SerializedObject(agent).FindProperty("_joints").arraySize;
                    int size = jointCount * OBSERVATIONS_PER_JOINT + FIXED_OBSERVATIONS;
                    if (behavior.BrainParameters.VectorObservationSize != size)
                    {
                        behavior.BrainParameters.VectorObservationSize = size;
                        PrefabUtility.SaveAsPrefabAsset(root, path);
                        updated++;
                        Debug.Log($"{path}: VectorObservationSize -> {size} ({jointCount} joints).");
                    }
                }
                PrefabUtility.UnloadPrefabContents(root);
            }
            Debug.Log($"Observation size sync complete: {updated} prefab(s) updated.");
        }
    }
}
#endif
