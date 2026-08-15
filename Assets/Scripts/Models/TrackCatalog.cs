using System;
using System.Collections.Generic;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Models
{
    /// <summary>
    /// Catalog of race tracks, mirroring CreatureCatalog: fixed slots that fill in
    /// over time. Procedural entries map to a TrackKind built by Systems_TrackBuilder;
    /// Imported entries reference a prefab (GLB scenes exported from Blender via
    /// glTFast). A slot with source Imported and no prefab is "coming soon".
    /// AvailableForRace gates the race rotation: training-only tracks (e.g. Rough
    /// before brains retrain on it) stay false.
    /// </summary>
    [CreateAssetMenu(menuName = "PoRacer/Track Catalog")]
    public sealed class TrackCatalog : ScriptableObject
    {
        public enum TrackSource
        {
            Procedural = 0,
            Imported = 1
        }

        [Serializable]
        public sealed class TrackEntry
        {
            public string id;
            public string displayName;
            public TrackSource source;
            public TrackKind proceduralKind;
            public GameObject importedPrefab;
            public bool availableForRace;

            public bool IsReady => source == TrackSource.Procedural || importedPrefab != null;
        }

        [SerializeField] private List<TrackEntry> _entries = new();

        public IReadOnlyList<TrackEntry> Entries => _entries;
    }
}
