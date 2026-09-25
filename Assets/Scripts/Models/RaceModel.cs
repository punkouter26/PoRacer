using System.Collections.Generic;

namespace PoRacer.Models
{
    public enum RacerStatus
    {
        Racing,
        Finished,
        // Ranked by distance when the race clock ran out: a real placing, but the
        // racer never crossed, so nothing may print a finish time for it.
        TimedOut,
        Dnf
    }

    public sealed class RacerState
    {
        public string RacerId { get; set; }
        public string CreatureId { get; set; }
        public string DisplayName { get; set; }
        public float Progress { get; set; }
        public RacerStatus Status { get; set; }
        // Set together with Status = Dnf; None for every other status.
        public KnockoutReason Knockout { get; set; }
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
        // Team in the Sim Wars league: the catalog's trainer, or Heuristic when the
        // racer fell back to its coded gait for want of a model.
        public TrainingSource TrainedBy { get; set; }
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

        /// <summary>
        /// Where this race's finish actually is, in world space. Set per race by the
        /// spawner: the builder track's arch on builder maps, the authored course's
        /// own finish on a course.
        ///
        /// It lives on the model so presentation does not have to guess it from a
        /// scene object. WinFxView used to read the serialized finish-line transform
        /// directly, and the spawner both MOVES that transform (to
        /// map.LengthMeters - 2) and DISABLES it for a course — so on Acrobat the
        /// confetti, fireworks and winner spotlight all fired at z = 210 on the flat
        /// plane, a couple of hundred metres from the mountain road being raced.
        /// </summary>
        public UnityEngine.Vector3 FinishPoint;
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
