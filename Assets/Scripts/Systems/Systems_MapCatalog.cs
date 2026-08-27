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
            // Start line to finish line, meters.
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
            // These are finish-line placements; the raced distance is ~2 m less,
            // since the grid sits ahead of the origin.
            //
            // Sized off measured pace, not off the training goal distance: the
            // fastest brain covers about 0.25 m/s, so the old 34 m Flat needed
            // ~135 s and every race died on the 120 s clock with nobody across the
            // line. 22 m puts the winner over at roughly 80 s and leaves room for
            // a second and third to land inside the window. Slower terrain gets a
            // shorter trek. Width stays 24 m so the 10-wide grid still fills the
            // lane visually. Re-measure these whenever the brains are retrained.
            new MapEntry("Flat", TrackKind.Flat, available: true, "Clean open ground — a pure speed test", 22f),
            new MapEntry("Lumpy", TrackKind.Lumpy, available: true, "Rough hills with chunky rocks to dodge", 18f),
            new MapEntry("Swamp", TrackKind.Swamp, available: true, "Mud pits that slow racers, gate walls to funnel them", 18f),
            new MapEntry("Gale", TrackKind.Flat, available: true,
                "Cross-winds shove the pack; boost pads reward a brave line", 20f,
                TrackFeatures.Gusts | TrackFeatures.BoostPads),
            new MapEntry("Roulette", TrackKind.Flat, available: true,
                "The wheel spins: fresh terrain and hazards every race", 18f,
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
