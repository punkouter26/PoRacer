using UnityEngine;

namespace PoRacer.Presentation
{
    /// <summary>
    /// The knobs the one-shot effects read at emit time. PostFxView writes them
    /// when the quality tier or the sky changes; the effects never decide them.
    /// Static because the effects are emitted from per-limb collision relays that
    /// have no injection scope, the same reason FxUtil is static.
    /// </summary>
    internal static class FxBudget
    {
        /// <summary>0-1 multiplier on particle counts for every one-shot effect.</summary>
        public static float Detail { get; set; } = 1f;

        /// <summary>Soft blob under each racer, for tiers whose shadow maps are weak.</summary>
        public static bool ContactShadows { get; set; }

        /// <summary>Base colour of footprints and the dust a footfall kicks up.</summary>
        public static Color GroundTone { get; set; } = new(0.62f, 0.55f, 0.45f);

        /// <summary>Scales a particle count by <see cref="Detail"/>, never below one.</summary>
        public static int Scale(int count)
        {
            return Mathf.Max(1, Mathf.RoundToInt(count * Detail));
        }
    }
}
