using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace PoRacer.Models
{
    [CreateAssetMenu(menuName = "PoRacer/Creature Catalog")]
    public sealed class CreatureCatalog : ScriptableObject
    {
        [Serializable]
        public sealed class CreatureEntry
        {
            public string id;
            public string displayName;
            public GameObject prefab;
            public ModelAsset model;
            public float spawnHeight = 0.15f;

            /// <summary>
            /// Set when the creature carries its own policy inside the prefab instead of
            /// racing an .onnx from <see cref="model"/> — Fido runs an MLP read from
            /// policy.json by his CreatureAgent, so he has no ModelAsset to assign and
            /// would otherwise be filed under "coming soon" forever.
            /// </summary>
            public bool brainInPrefab;

            /// <summary>
            /// True when this entry has a brain to race at all, from either source.
            /// </summary>
            public bool HasBrain => model != null || brainInPrefab;
        }

        [SerializeField] private List<CreatureEntry> _entries = new();

        public IReadOnlyList<CreatureEntry> Entries => _entries;
    }
}
