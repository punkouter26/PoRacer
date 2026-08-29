using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;
#if ISAAC_BIPED2_INFERENCE
using Unity.InferenceEngine;
#endif

namespace IsaacBiped2.Editor
{
    /// <summary>
    /// Spawns a batch of bipeds on clean ground with a chosen policy and reports the distribution of
    /// distance-before-falling. Call <see cref="Spawn"/> from play mode, wait, then <see cref="Report"/>.
    ///
    /// Exists because every Unity transfer measurement in this project so far was a single run, and
    /// single runs of an unstable system are noisy enough to have flipped conclusions twice.
    /// </summary>
    public static class IsaacBiped2BatchTest
    {
        private const string GROUND = "BATCH_GROUND";
        private const string PREFIX = "BATCH_";
        private const float BASE_Z = 2400f;

        /// <summary>Spawn <paramref name="count"/> bipeds running <paramref name="onnxPath"/>.</summary>
        public static string Spawn(string onnxPath, int count = 8)
        {
            Cleanup();
            Time.fixedDeltaTime = 0.005f;

            var ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = GROUND;
            ground.transform.position = new Vector3(0f, -0.5f, BASE_Z + 20f);
            ground.transform.localScale = new Vector3(40f * count, 1f, 200f);
            var material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacBiped2Setup.PHYSICS_MATERIAL_PATH);
            if (material != null)
            {
                ground.GetComponent<Collider>().sharedMaterial = material;
            }

            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBiped2Setup.PREFAB_PATH);
            if (prefab == null)
            {
                return "prefab not found";
            }
#if ISAAC_BIPED2_INFERENCE
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(onnxPath);
            if (model == null)
            {
                return $"model not found at {onnxPath}";
            }
#endif
            var holder = new GameObject("BATCH_HOLDER");
            holder.SetActive(false);
            for (int index = 0; index < count; index++)
            {
                float x = (index - (count - 1) * 0.5f) * 6f;
                var target = new GameObject($"{PREFIX}T{index}");
                target.transform.position = new Vector3(x, 0.15f, BASE_Z + 60f);

                // Instantiate UNDER AN INACTIVE PARENT. Instantiating an active prefab runs Awake
                // immediately, and IsaacBiped2Agent.Awake builds its inference worker from whatever
                // _model the prefab carries — so assigning _model afterwards is silently ignored and
                // every instance runs the prefab's brain instead of the one under test.
                GameObject instance = Object.Instantiate(prefab, holder.transform);
                instance.name = PREFIX + index;
                // Place it while still inactive: Awake captures the spawn position and Start pins the
                // root, so a body activated at the origin and moved afterwards never finds the ground
                // and stays pinned forever.
                instance.transform.position = new Vector3(x, 0.68f, BASE_Z);
                instance.transform.rotation = Quaternion.identity;
                // The race adapter would retire the racer; this test measures the policy alone.
                // Matched by type NAME rather than by type: PoRacer.Runtime references
                // IsaacBiped2.Runtime, so referencing it back from here would be circular.
                MonoBehaviour[] attached = instance.GetComponents<MonoBehaviour>();
                for (int c = 0; c < attached.Length; c++)
                {
                    if (attached[c] != null && attached[c].GetType().Name == "Agent_IsaacBiped2")
                    {
                        Object.DestroyImmediate(attached[c]);
                    }
                }
                var agent = instance.GetComponent<IsaacBiped2Agent>();
#if ISAAC_BIPED2_INFERENCE
                var so = new SerializedObject(agent);
                so.FindProperty("_model").objectReferenceValue = model;
                so.ApplyModifiedPropertiesWithoutUndo();
#endif
                agent.target = target.transform;
                instance.AddComponent<IsaacBiped2DistanceProbe>();
                // Re-parent out of the inactive holder: this is the point Awake finally runs, now
                // that _model is set and the transform is already where it belongs. worldPositionStays
                // keeps the placement made above.
                instance.transform.SetParent(null, true);
            }
            Object.DestroyImmediate(holder);
            return $"spawned {count} bipeds with {System.IO.Path.GetFileNameWithoutExtension(onnxPath)}";
        }

        /// <summary>Distribution of distance-before-falling across the batch.</summary>
        public static string Report()
        {
            var distances = new List<float>();
            var builder = new StringBuilder();
            int fallen = 0;
            float lateral = 0f;
            float foreAft = 0f;
            for (int index = 0; index < 64; index++)
            {
                var go = GameObject.Find(PREFIX + index);
                if (go == null)
                {
                    continue;
                }
                var probe = go.GetComponent<IsaacBiped2DistanceProbe>();
                if (probe == null)
                {
                    continue;
                }
                distances.Add(probe.MaxDistance);
                if (probe.Fallen)
                {
                    fallen++;
                    lateral += probe.FallLateral;
                    foreAft += probe.FallForeAft;
                }
            }
            if (distances.Count == 0)
            {
                return "no probes found";
            }
            distances.Sort();
            float sum = 0f;
            for (int index = 0; index < distances.Count; index++)
            {
                sum += distances[index];
            }
            builder.Append("n=").Append(distances.Count)
                .Append("  median=").Append(distances[distances.Count / 2].ToString("F2"))
                .Append("  mean=").Append((sum / distances.Count).ToString("F2"))
                .Append("  min=").Append(distances[0].ToString("F2"))
                .Append("  max=").Append(distances[distances.Count - 1].ToString("F2"))
                .Append("  fallen=").Append(fallen).Append('/').Append(distances.Count);
            if (fallen > 0)
            {
                builder.Append("  meanLean lateral=").Append((lateral / fallen).ToString("F2"))
                    .Append(" foreaft=").Append((foreAft / fallen).ToString("F2"));
            }
            builder.Append("  [");
            for (int index = 0; index < distances.Count; index++)
            {
                builder.Append(distances[index].ToString("F2")).Append(' ');
            }
            builder.Append(']');
            return builder.ToString();
        }

        public static void Cleanup()
        {
            var ground = GameObject.Find(GROUND);
            if (ground != null)
            {
                Object.DestroyImmediate(ground);
            }
            for (int index = 0; index < 64; index++)
            {
                var go = GameObject.Find(PREFIX + index);
                if (go != null)
                {
                    Object.DestroyImmediate(go);
                }
                var target = GameObject.Find($"{PREFIX}T{index}");
                if (target != null)
                {
                    Object.DestroyImmediate(target);
                }
            }
        }
    }
}
