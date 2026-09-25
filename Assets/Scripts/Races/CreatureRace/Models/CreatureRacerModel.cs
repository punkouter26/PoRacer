using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One racer's live race state. Plain C#, written by CreatureRaceSystem each physics tick
    /// and read on a schedule by the HUD and every frame by the camera (DOCS/Plan-P1-Worm.md
    /// D5: plain models, no hand-rolled reactive properties).
    /// </summary>
    public sealed class CreatureRacerModel
    {
        public CreatureRacerModel(int lane)
        {
            Lane = lane;
            FinishTimeSeconds = -1f;
            BrainError = string.Empty;
            Name = string.Empty;
            Method = string.Empty;
            Physics = string.Empty;
            BrainName = string.Empty;
            Upright = 1f;
        }

        public int Lane { get; }
        public string Name { get; internal set; }
        /// <summary>Training method shown on the HUD, e.g. "MuJoCo", "Isaac Lab 3".</summary>
        public string Method { get; internal set; }
        /// <summary>The simulator stepping this racer in Unity.</summary>
        public string Physics { get; internal set; }
        public string BrainName { get; internal set; }
        public Color Color { get; internal set; }

        public CreatureRacerStatus Status { get; internal set; }
        /// <summary>Lead-point distance past the start line along the lane, metres.</summary>
        public float Distance { get; internal set; }
        /// <summary>Smoothed forward speed, m/s.</summary>
        public float Speed { get; internal set; }
        public float ElapsedSeconds { get; internal set; }
        /// <summary>-1 until the lead point crosses the finish line.</summary>
        public float FinishTimeSeconds { get; internal set; }
        public float AverageSpeed { get; internal set; }
        /// <summary>1-based placing once the race is decided, 0 before.</summary>
        public int Place { get; internal set; }
        /// <summary>Unity world position of the lead point; the camera frames these.</summary>
        public Vector3 Nose { get; internal set; }
        /// <summary>Reference body up . world up (1 upright).</summary>
        public float Upright { get; internal set; }
        /// <summary>True while it lies fallen (upright below the settings' threshold).</summary>
        public bool IsDown { get; internal set; }
        public bool BrainReady { get; internal set; }
        public string BrainError { get; internal set; }

        internal void ResetForRace()
        {
            Status = CreatureRacerStatus.Waiting;
            Distance = 0f;
            Speed = 0f;
            ElapsedSeconds = 0f;
            FinishTimeSeconds = -1f;
            AverageSpeed = 0f;
            Place = 0;
            Upright = 1f;
            IsDown = false;
        }
    }
}
