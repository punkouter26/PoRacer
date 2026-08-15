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
        }

        [SerializeField] private List<CreatureEntry> _entries = new();

        public IReadOnlyList<CreatureEntry> Entries => _entries;
    }
}
