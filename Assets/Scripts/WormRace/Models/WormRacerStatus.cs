namespace PoRacer.WormRace
{
    public enum WormRacerStatus
    {
        Waiting,
        Racing,
        Finished,
        /// <summary>The race clock ran out before the nose crossed: ranked by distance.</summary>
        TimedOut,
        /// <summary>Physics blew up (non-finite state) or the simulator is unavailable here.</summary>
        Failed,
    }
}
