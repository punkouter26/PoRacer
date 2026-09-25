namespace PoRacer.Presentation
{
    /// <summary>
    /// The knobs the per-racer effects read. PostFxView writes them when the quality
    /// tier changes; the effects never decide them. Static because the effects are
    /// added to racers at spawn and have no injection scope, the same reason FxUtil
    /// is static.
    /// </summary>
    internal static class FxBudget
    {
        /// <summary>Soft blob under each racer, for tiers whose shadow maps are weak.</summary>
        public static bool ContactShadows { get; set; }
    }
}
