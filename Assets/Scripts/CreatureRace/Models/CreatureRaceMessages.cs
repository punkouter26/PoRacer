namespace PoRacer.CreatureRace
{
    // Grouped in one file after the project's own Assets/Scripts/Models/Messages.cs.

    /// <summary>3, 2, 1 while the racers are held; 0 is "GO".</summary>
    public readonly struct CreatureCountdownMessage
    {
        public readonly int Value;

        public CreatureCountdownMessage(int value)
        {
            Value = value;
        }
    }

    public readonly struct CreatureRaceFinishedMessage
    {
        public readonly int RaceNumber;
        /// <summary>Lane of the winner, -1 when every racer failed.</summary>
        public readonly int WinnerLane;

        public CreatureRaceFinishedMessage(int raceNumber, int winnerLane)
        {
            RaceNumber = raceNumber;
            WinnerLane = winnerLane;
        }
    }

    /// <summary>A whole series (or a self-test) is over and its report is on disk.</summary>
    public readonly struct CreatureSeriesFinishedMessage
    {
        public readonly string ReportPath;

        public CreatureSeriesFinishedMessage(string reportPath)
        {
            ReportPath = reportPath;
        }
    }
}
