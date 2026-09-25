namespace PoRacer.Models
{
    /// <summary>
    /// The Sim Wars league: which training tool's racers do best, race after race. Owned
    /// by Systems_SimWars, persisted to sim-wars.json beside the ELO file.
    ///
    /// The headline is the MuJoCo vs Isaac Lab head-to-head: in every race both tools had
    /// a racer in, whichever tool's best finisher placed higher takes the point.
    /// </summary>
    public sealed class SimWarsModel
    {
        private readonly SimWarsTeamStats[] _teams;

        public SimWarsModel()
        {
            int teamCount = System.Enum.GetValues(typeof(TrainingSource)).Length;
            _teams = new SimWarsTeamStats[teamCount];
            for (int teamIndex = 0; teamIndex < teamCount; teamIndex++)
            {
                _teams[teamIndex] = new SimWarsTeamStats();
            }
        }

        public int RacesScored { get; set; }
        public int MuJoCoAhead { get; set; }
        public int IsaacLabAhead { get; set; }
        public int HeadToHeadDraws { get; set; }

        /// <summary>Bumped on every change so a view rebuilds its text only when there is news.</summary>
        public int Version { get; set; }

        public SimWarsTeamStats Team(TrainingSource team) => _teams[(int)team];

        public void Reset()
        {
            for (int teamIndex = 0; teamIndex < _teams.Length; teamIndex++)
            {
                _teams[teamIndex] = new SimWarsTeamStats();
            }
            RacesScored = 0;
            MuJoCoAhead = 0;
            IsaacLabAhead = 0;
            HeadToHeadDraws = 0;
            Version++;
        }
    }
}
