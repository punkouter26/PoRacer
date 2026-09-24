namespace PoRacer.WormRace
{
    // Grouped in one file after the project's own Assets/Scripts/Models/Messages.cs.

    /// <summary>3, 2, 1 while the worms are held; 0 is "GO".</summary>
    public readonly struct WormCountdownMessage
    {
        public readonly int Value;

        public WormCountdownMessage(int value)
        {
            Value = value;
        }
    }

    public readonly struct WormRaceFinishedMessage
    {
        public readonly int RaceNumber;
        /// <summary>Lane of the winner, -1 when every racer failed.</summary>
        public readonly int WinnerLane;

        public WormRaceFinishedMessage(int raceNumber, int winnerLane)
        {
            RaceNumber = raceNumber;
            WinnerLane = winnerLane;
        }
    }

    /// <summary>A whole series (or a self-test) is over and its report is on disk.</summary>
    public readonly struct WormSeriesFinishedMessage
    {
        public readonly string ReportPath;

        public WormSeriesFinishedMessage(string reportPath)
        {
            ReportPath = reportPath;
        }
    }
}
