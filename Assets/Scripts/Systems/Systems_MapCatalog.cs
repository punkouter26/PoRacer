using System.Collections.Generic;

namespace PoRacer.Systems
{
    /// <summary>
    /// The player-selectable maps. A map is a TrackKind the builder can produce,
    /// plus a race length and optional hazard features.
    /// </summary>
    public static class Systems_MapCatalog
    {
        public readonly struct MapEntry
        {
            public readonly string DisplayName;
            public readonly TrackKind Kind;
            public readonly bool Available;
            public readonly string Blurb;
            // Start line to finish line, meters. Longer than the old fixed 22 so
            // races build drama; slower terrain gets a shorter trek.
            public readonly float LengthMeters;
            // Extra hazards layered on the kind (boost pads, gusts, mud, gates).
            public readonly TrackFeatures Features;
            // Roulette: the spawn system rolls a random kind + features per race.
            public readonly bool Randomize;

            public MapEntry(string displayName, TrackKind kind, bool available, string blurb = "",
                float lengthMeters = 32f, TrackFeatures features = TrackFeatures.None, bool randomize = false)
            {
                DisplayName = displayName;
                Kind = kind;
                Available = available;
                Blurb = blurb;
                LengthMeters = lengthMeters;
                Features = features;
                Randomize = randomize;
            }
        }

        public static readonly IReadOnlyList<MapEntry> Entries = new[]
        {
            // Lengths sized to the current brains' training distribution (goals up
            // to ~20 m): a winner should cross in roughly 1-2 minutes. The 4x
            // "marathon" lengths produced all-DNF races. Width stays 24 m so the
            // 10-wide grid still fills the lane visually.
            new MapEntry("Flat", TrackKind.Flat, available: true, "Clean open ground — a pure speed test", 34f),
            new MapEntry("Lumpy", TrackKind.Lumpy, available: true, "Rough hills with chunky rocks to dodge", 26f),
            new MapEntry("Swamp", TrackKind.Swamp, available: true, "Mud pits that slow racers, gate walls to funnel them", 28f),
            new MapEntry("Gale", TrackKind.Flat, available: true,
                "Cross-winds shove the pack; boost pads reward a brave line", 30f,
                TrackFeatures.Gusts | TrackFeatures.BoostPads),
            new MapEntry("Roulette", TrackKind.Flat, available: true,
                "The wheel spins: fresh terrain and hazards every race", 28f,
                randomize: true)
        };

        /// <summary>Clamps out-of-range or placeholder picks back to the first map.</summary>
        public static MapEntry Get(int mapIndex)
        {
            if (mapIndex < 0 || mapIndex >= Entries.Count || !Entries[mapIndex].Available)
            {
                return Entries[0];
            }
            return Entries[mapIndex];
        }
    }
}
