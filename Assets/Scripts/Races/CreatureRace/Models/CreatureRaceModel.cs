using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Race-wide state of a creature race scene. Plain C#; CreatureRaceSystem is its only
    /// writer. The editor harness polls <see cref="Phase"/> and <see cref="ReportPath"/> to
    /// know when a series or self-test is done and where its results went.
    ///
    /// One <see cref="CreatureRacerModel"/> per lane; the count is the settings' racer list,
    /// fixed for the play session (the lifetime scope passes it in).
    /// </summary>
    public sealed class CreatureRaceModel
    {
        private readonly List<CreatureRacerModel> _racers;
        private readonly int[] _wins;

        public CreatureRaceModel(int racerCount)
        {
            int count = racerCount > 0 ? racerCount : 0;
            _racers = new List<CreatureRacerModel>(count);
            _wins = new int[count];
            for (int lane = 0; lane < count; lane++)
            {
                _racers.Add(new CreatureRacerModel(lane));
            }
            Message = string.Empty;
            ReportPath = string.Empty;
            LastEndReason = string.Empty;
            ModeLabel = string.Empty;
            LastWinnerLane = -1;
        }

        public IReadOnlyList<CreatureRacerModel> Racers => _racers;
        public CreatureRacePhase Phase { get; internal set; }
        /// <summary>True for a race series, false for a self-test.</summary>
        public bool IsRaceMode { get; internal set; } = true;
        /// <summary>"Race" or the running self-test's name, e.g. "YawSignTest".</summary>
        public string ModeLabel { get; internal set; }
        /// <summary>3, 2, 1 during the countdown; 0 otherwise.</summary>
        public int CountdownValue { get; internal set; }
        public int RaceNumber { get; internal set; }
        public int PlannedRaces { get; internal set; }
        public int CompletedRaces { get; internal set; }
        public float ElapsedSeconds { get; internal set; }
        public float TrackLength { get; internal set; }
        public float TimeLimitSeconds { get; internal set; }
        /// <summary>True while racers exist in the scene; the camera frames them only then.</summary>
        public bool RacersSpawned { get; internal set; }
        /// <summary>Where the camera looks when no racer is out: the middle of the start line.</summary>
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

        public bool IsFinished => Phase == CreatureRacePhase.SeriesComplete
                               || Phase == CreatureRacePhase.SelfTestComplete
                               || Phase == CreatureRacePhase.Error;

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
