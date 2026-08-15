#if UNITY_EDITOR
using Unity.MLAgents;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;
using UnityEditor;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// One-click demo capture for BC/GAIL pretraining: flips every agent in the
    /// open training scene to its coded heuristic gait and attaches a
    /// DemonstrationRecorder. Enter play mode, let the gaits walk for a few
    /// minutes, exit — .demo files land in Assets/Demonstrations/ under each
    /// behavior's name, matching the demo_path entries in Config/AllLoco03.yaml.
    /// </summary>
    internal static class Editor_RecordDemos
    {
        [MenuItem("PoRacer/Training/Enable Demo Recorders In Open Scene")]
        private static void EnableRecorders()
        {
            int count = 0;
            Agent[] agents = Object.FindObjectsByType<Agent>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            for (int agentIndex = 0; agentIndex < agents.Length; agentIndex++)
            {
                Agent agent = agents[agentIndex];
                BehaviorParameters behavior = agent.GetComponent<BehaviorParameters>();
                if (behavior == null)
                {
                    continue;
                }
                behavior.BehaviorType = BehaviorType.HeuristicOnly;
                DemonstrationRecorder recorder = agent.GetComponent<DemonstrationRecorder>();
                if (recorder == null)
                {
                    recorder = agent.gameObject.AddComponent<DemonstrationRecorder>();
                }
                recorder.DemonstrationName = behavior.BehaviorName;
                recorder.Record = true;
                EditorUtility.SetDirty(agent.gameObject);
                count++;
            }
            Debug.Log($"Demo recorders enabled on {count} agents. Enter play mode to record; " +
                ".demo files are written to Assets/Demonstrations/.");
        }

        [MenuItem("PoRacer/Training/Disable Demo Recorders In Open Scene")]
        private static void DisableRecorders()
        {
            int count = 0;
            DemonstrationRecorder[] recorders =
                Object.FindObjectsByType<DemonstrationRecorder>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            for (int recorderIndex = 0; recorderIndex < recorders.Length; recorderIndex++)
            {
                Object.DestroyImmediate(recorders[recorderIndex]);
                count++;
            }
            Debug.Log($"Removed {count} demo recorders. Restore BehaviorType manually if needed " +
                "(training uses Default, races set it at spawn).");
        }
    }
}
#endif
