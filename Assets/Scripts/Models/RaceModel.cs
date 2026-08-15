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
    }

    public sealed class RaceModel
    {
        public readonly List<RacerState> Racers = new();

        private readonly Dictionary<string, RacerState> _racersById = new();

        public float ElapsedSeconds;
        public bool RaceActive;
        public int RaceNumber;
        public string TrackName = "Flat";

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
