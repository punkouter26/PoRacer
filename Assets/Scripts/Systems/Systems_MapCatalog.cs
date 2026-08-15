using System.Collections.Generic;

namespace PoRacer.Systems
{
    /// <summary>
    /// The 8 player-selectable maps. A map is a TrackKind the builder can produce;
    /// slots without one yet are placeholders shown greyed out in the menu.
    /// </summary>
    public static class Systems_MapCatalog
    {
        public readonly struct MapEntry
        {
            public readonly string DisplayName;
            public readonly TrackKind Kind;
            public readonly bool Available;

            public MapEntry(string displayName, TrackKind kind, bool available)
            {
                DisplayName = displayName;
                Kind = kind;
                Available = available;
            }
        }

        public static readonly IReadOnlyList<MapEntry> Entries = new[]
        {
            new MapEntry("Flat", TrackKind.Flat, available: true),
            new MapEntry("Lumpy", TrackKind.Lumpy, available: true),
            new MapEntry("Swamp", TrackKind.Swamp, available: true),
            new MapEntry("Map 4", TrackKind.Flat, available: false),
            new MapEntry("Map 5", TrackKind.Flat, available: false),
            new MapEntry("Map 6", TrackKind.Flat, available: false),
            new MapEntry("Map 7", TrackKind.Flat, available: false),
            new MapEntry("Map 8", TrackKind.Flat, available: false)
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
