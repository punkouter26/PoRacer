using System.Collections.Generic;

namespace PoRacer.Models
{
    public enum RacerStatus
    {
        Racing,
        Finished,
        Dnf
    }

    public sealed class RacerState
    {
        public string RacerId { get; set; }
        public string CreatureId { get; set; }
        public string DisplayName { get; set; }
        public float Progress { get; set; }
        public RacerStatus Status { get; set; }
        public int Place { get; set; }
        public float FinishTime { get; set; }
        // Meters past the finish line when the crossing was reported; breaks
        // same-frame ties (farther past the line = crossed earlier).
        public float FinishOvershoot { get; set; }
        public UnityEngine.Color Tint { get; set; }
        // Cached "RRGGBB" so commentary/HUD rich text never re-encodes per line.
        public string TintHex { get; set; }
        // Visible quirk: short tag ("TURBO"); empty for a plain racer.
        public string QuirkTag { get; set; } = string.Empty;
        // HUD badge color for the quirk; alpha 0 hides the badge.
        public UnityEngine.Color QuirkColor { get; set; }
    }

    public sealed class RaceModel
    {
        public readonly List<RacerState> Racers = new();

        private readonly Dictionary<string, RacerState> _racersById = new();

        public float ElapsedSeconds;
        public bool RaceActive;
        public int RaceNumber;
        // Pre-start countdown: 3, 2, 1 while the grid settles; 0 = none.
        public int CountdownValue;
        public string TrackName = "Flat";
        // Start line to finish line, for progress-strip percentages. Overwritten
        // per race from the map catalog; this is only a fallback.
        public float TrackLengthMeters = 30f;

        public void SetRacers(IReadOnlyList<RacerState> racers)
        {
            Racers.Clear();
            _racersById.Clear();
            for (int racerIndex = 0; racerIndex < racers.Count; racerIndex++)
            {
                Racers.Add(racers[racerIndex]);
                _racersById[racers[racerIndex].RacerId] = racers[racerIndex];
            }
        }

        public void ClearRacers()
        {
            Racers.Clear();
            _racersById.Clear();
        }

        public RacerState FindRacer(string racerId)
        {
            return _racersById.TryGetValue(racerId, out RacerState racer) ? racer : null;
        }
    }
}
