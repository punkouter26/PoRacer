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
            // Full-time clock for this map; the race is decided by distance when it runs out.
            public readonly float TimeLimitSeconds;

            /// <summary>
            /// Lateral gap between starting grid slots, metres. Per-map because the start
            /// is where the sprawling rigs lose their races.
            ///
            /// Measured 2026-09-10: a Quadruped alone on Flat finishes in 57.9 s dead
            /// upright, and the SAME brain in the default 8-racer field averages 8.0 m
            /// and did not finish once in 20 races. At the old flat 2 m the wide,
            /// low-slung creatures tangle with their neighbours on the line and never
            /// get up. Note which racers were unaffected: MojucuBoy cannot collide with
            /// a PhysX racer at all (different solver), and IsaacBox and Isaac H1 are
            /// narrow upright bipeds. The three that "failed" are exactly the three
            /// sprawling rigs.
            ///
            /// Bounded by the track: Systems_TrackBuilder builds builder maps 24 m wide,
            /// so the whole row has to fit inside +/-12 m of the centreline.
            /// </summary>
            public readonly float GridColumnSpacing;

            public MapEntry(string displayName, TrackKind kind, bool available, string blurb = "",
                float lengthMeters = 32f, TrackFeatures features = TrackFeatures.None,
                float timeLimitSeconds = DEFAULT_TIME_LIMIT_SECONDS,
                float gridColumnSpacing = DEFAULT_GRID_COLUMN_SPACING)
            {
                GridColumnSpacing = gridColumnSpacing;
                DisplayName = displayName;
                Kind = kind;
                Available = available;
                Blurb = blurb;
                LengthMeters = lengthMeters;
                Features = features;
                TimeLimitSeconds = timeLimitSeconds;
            }
        }

        public const float DEFAULT_TIME_LIMIT_SECONDS = 120f;

        /// <summary>
        /// The grid spacing every map used before it became per-map. Left as the default
        /// so the maps that have not been measured keep exactly the behaviour they had.
        /// </summary>
        public const float DEFAULT_GRID_COLUMN_SPACING = 2f;

        /// <summary>
        /// Flat's widened grid. 3 m, not more, because the row must fit the 24 m track:
        /// eight racers centred on their own count span 7 x 3 = 21 m, i.e. +/-10.5 m,
        /// which clears the +/-12 m edge with 1.5 m to spare. 3.5 m would span 24.5 m and
        /// put the outside racers off the ground.
        /// </summary>
        public const float FLAT_GRID_COLUMN_SPACING = 3f;
        // The Acrobat course is ~212 m of climbing switchbacks at scale 1.0:
        // about ten times a builder map.
        // Its length is read off the authored centreline at race time; this is
        // only the catalogue's display figure.
        public const float ACROBAT_LENGTH_METERS = 212f;
        // The Apartment lap: 16.08 m of authored checkpoints at APARTMENT_SCALE 5.0.
        // Display figure only, same as ACROBAT_LENGTH_METERS above - the real length
        // is read off the checkpoints at race time, and the scene's baked
        // RaceCourseView measures 72.8 m, not the 80 this used to claim.
        public const float APARTMENT_LENGTH_METERS = 73f;

        /// <summary>
        /// Course clocks, cut from 600 s on 2026-09-10 because 600 was never a race
        /// length — it was a number chosen so a course "had time", and measurement
        /// says the time was never the constraint.
        ///
        /// What three smoke runs actually show on Acrobat: nobody finishes. The
        /// leaders reach ~48 m of 212 m by 92 s and then go off the road, which ends
        /// their episode; at 92 s the field was 4 DNF and 3 still crawling at ~25 m.
        /// The race therefore resolves by attrition long before the clock, and all
        /// the clock decided was how long the last straggler kept a player waiting —
        /// up to ten minutes for a podium ranked on distance with nobody across the
        /// line. 240 s bounds that wait at four minutes without touching the outcome.
        ///
        /// Apartment is the opposite case and gets a different number for a reason:
        /// at 73 m and the ~0.5 m/s the same brains manage on a road, it is genuinely
        /// finishable in ~150 s. 180 s makes it a race that can be won rather than one
        /// that always times out, which 600 s hid rather than helped.
        ///
        /// Re-measure both whenever the brains are retrained — a policy that can climb
        /// (see SCN_TRAIN_ACROBAT) changes the Acrobat figure completely.
        /// </summary>
        public const float ACROBAT_TIME_LIMIT_SECONDS = 240f;

        public const float APARTMENT_TIME_LIMIT_SECONDS = 180f;

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
            new MapEntry("Flat", TrackKind.Flat, available: true, "Clean open ground — a pure speed test", 22f,
                gridColumnSpacing: FLAT_GRID_COLUMN_SPACING),
            // REMOVED 2026-09-11: Lumpy, Swamp, Gale and Roulette; the ML-Agents training
            // scenes that still used them went on 2026-09-25. TrackKind and TrackFeatures
            // keep every value: both are serialized by number in SCN_RACE_FLAT's
            // AuthoredTrack entries, so renumbering would silently re-resolve them.
            // Authored in Blender (Assets/Art/Models/AcrobatTrack.glb): a mountain
            // road of switchbacks and a tunnel, 50 m of climb. Raced along its
            // centreline, not down +Z, so it needs its authored course entry in
            // SCN_RACE_FLAT. Brains trained on the flat builder maps mostly cannot
            // finish it.
            new MapEntry("Acrobat", TrackKind.Course, available: true,
                "Mountain switchbacks and a tunnel; 50 m of climb", ACROBAT_LENGTH_METERS,
                timeLimitSeconds: ACROBAT_TIME_LIMIT_SECONDS),
            // A photo reconstruction of a flat with a toy track built through it
            // (Assets/Art/Models/ApartmentTrack.glb), raced along its twelve
            // Checkpoint_ knots. Placed at scale 5.0, which turns a 16.1 m toy lap into a 73 m one on a 3.2 m road -
            // the scale is set by road width, since the racers on it cannot be
            // resized without breaking their brains. Like Acrobat this is a display
            // figure; the real length is read off the checkpoints at race time.
            // The lap starts at the summit rather than at the painted line, so the
            // descent comes first and the climb is the last stretch; it keeps
            // Acrobat's 600 s clock because 73 m is still three builder maps long.
            new MapEntry("Apartment", TrackKind.Apartment, available: true,
                "A toy circuit through a real flat; downhill first, climb at the end",
                APARTMENT_LENGTH_METERS, timeLimitSeconds: APARTMENT_TIME_LIMIT_SECONDS)
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
