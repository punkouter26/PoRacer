namespace PoRacer.Models
{
    /// <summary>One trainer team's running record in the Sim Wars league.</summary>
    public sealed class SimWarsTeamStats
    {
        /// <summary>Races this team had at least one racer in.</summary>
        public int Races { get; set; }
        public int Wins { get; set; }
        /// <summary>Podium finishes by any of the team's racers, so one race can add two.</summary>
        public int Podiums { get; set; }
        public int Points { get; set; }
        /// <summary>Points scored in the most recent race; 0 when the team was not in it.</summary>
        public int LastRacePoints { get; set; }
    }
}
