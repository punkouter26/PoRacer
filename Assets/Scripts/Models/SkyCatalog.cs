using System.Collections.Generic;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Models
{
    /// <summary>
    /// Every sky preset the race can roll, in priority order: the first preset
    /// that serves a track kind is that kind's default, shown while the map is
    /// being picked in the menu.
    /// </summary>
    [CreateAssetMenu(menuName = "PoRacer/Sky Catalog")]
    public sealed class SkyCatalog : ScriptableObject
    {
        [SerializeField] private List<SkyPreset> _presets = new();

        /// <summary>
        /// Fills <paramref name="pool"/> with the presets for <paramref name="kind"/>:
        /// those that list it, or, when none do, the fallbacks that list nothing.
        /// </summary>
        public void CollectFor(TrackKind kind, List<SkyPreset> pool)
        {
            pool.Clear();
            for (int presetIndex = 0; presetIndex < _presets.Count; presetIndex++)
            {
                SkyPreset preset = _presets[presetIndex];
                if (preset != null && System.Array.IndexOf(preset.Kinds, kind) >= 0)
                {
                    pool.Add(preset);
                }
            }
            if (pool.Count > 0)
            {
                return;
            }
            for (int presetIndex = 0; presetIndex < _presets.Count; presetIndex++)
            {
                SkyPreset preset = _presets[presetIndex];
                if (preset != null && preset.Kinds.Length == 0)
                {
                    pool.Add(preset);
                }
            }
        }
    }
}
