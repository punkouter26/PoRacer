using System;

namespace PoRacer.Models
{
    /// <summary>
    /// The rendering quality tier currently in force. Written by
    /// Systems_QualityGovernor from measured frame times; read by PostFxView, which
    /// applies it to the camera, the pipeline and the particle budget, and by the
    /// fps readout, which shows it.
    ///
    /// Tiers only ever trade looks for frame time. Physics, policies and the race
    /// itself never read this: a slow phone must race the same race, just less
    /// prettily.
    /// </summary>
    public sealed class QualityModel
    {
        public const int TIER_HIGH = 0;
        public const int TIER_MEDIUM = 1;
        public const int TIER_LOW = 2;
        public const int TIER_MINIMUM = 3;

        public int Tier { get; private set; } = TIER_HIGH;

        public event Action Changed;

        public void SetTier(int tier)
        {
            int clamped = tier < TIER_HIGH ? TIER_HIGH : tier > TIER_MINIMUM ? TIER_MINIMUM : tier;
            if (clamped == Tier)
            {
                return;
            }
            Tier = clamped;
            Changed?.Invoke();
        }

        /// <summary>Two- or three-letter tag for the fps readout.</summary>
        public static string ShortName(int tier)
        {
            switch (tier)
            {
                case TIER_HIGH:
                    return "HQ";
                case TIER_MEDIUM:
                    return "MQ";
                case TIER_LOW:
                    return "LQ";
                default:
                    return "MIN";
            }
        }
    }
}
