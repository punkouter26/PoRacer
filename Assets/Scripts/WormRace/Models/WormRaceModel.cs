using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Race-wide state for SCN_WORM_RACE. Plain C#; WormRaceSystem is its only writer.
    /// The editor harness (Editor_WormRace) polls <see cref="Phase"/> and
    /// <see cref="ReportPath"/> to know when a series is done and where its results went.
    ///
    /// One <see cref="WormRacerModel"/> per lane; the count is WormRaceSettings' racer list,
    /// fixed for the play session (WormRaceLifetimeScope passes it in).
    /// </summary>
    public sealed class WormRaceModel
    {
        private readonly List<WormRacerModel> _racers;
        private readonly int[] _wins;

        public WormRaceModel(int racerCount)
        {
            int count = racerCount > 0 ? racerCount : 0;
            _racers = new List<WormRacerModel>(count);
            _wins = new int[count];
            for (int lane = 0; lane < count; lane++)
            {
                _racers.Add(new WormRacerModel(lane));
            }
            Message = string.Empty;
            ReportPath = string.Empty;
            LastEndReason = string.Empty;
            LastWinnerLane = -1;
        }

        public IReadOnlyList<WormRacerModel> Racers => _racers;
        public WormRacePhase Phase { get; internal set; }
        public WormRaceMode Mode { get; internal set; }
        /// <summary>3, 2, 1 during the countdown; 0 otherwise.</summary>
        public int CountdownValue { get; internal set; }
        public int RaceNumber { get; internal set; }
        public int PlannedRaces { get; internal set; }
        public int CompletedRaces { get; internal set; }
        public float ElapsedSeconds { get; internal set; }
        public float TrackLength { get; internal set; }
        public float TimeLimitSeconds { get; internal set; }
        /// <summary>True while worms exist in the scene; the camera frames them only then.</summary>
        public bool WormsSpawned { get; internal set; }
        /// <summary>Where the camera looks when no worm is out: the middle of the start line.</summary>
        public Vector3 StartFocus { get; internal set; }
        public int LastWinnerLane { get; internal set; }
        /// <summary>"finish", "timeLimit" or "failed" for the last race.</summary>
        public string LastEndReason { get; internal set; }
        public string ReportPath { get; internal set; }
        /// <summary>Human-readable status or error line for the HUD and the harness.</summary>
        public string Message { get; internal set; }

        public int WinsFor(int lane)
        {
            return lane >= 0 && lane < _wins.Length ? _wins[lane] : 0;
        }

        public bool IsFinished => Phase == WormRacePhase.SeriesComplete
                               || Phase == WormRacePhase.SelfTestComplete
                               || Phase == WormRacePhase.Error;

        internal void AddWin(int lane)
        {
            if (lane >= 0 && lane < _wins.Length)
            {
                _wins[lane]++;
            }
        }

        internal void ResetSeries()
        {
            for (int lane = 0; lane < _wins.Length; lane++)
            {
                _wins[lane] = 0;
                _racers[lane].ResetForRace();
            }
            CompletedRaces = 0;
            RaceNumber = 0;
            LastWinnerLane = -1;
            LastEndReason = string.Empty;
            ReportPath = string.Empty;
            Message = string.Empty;
        }
    }
}
