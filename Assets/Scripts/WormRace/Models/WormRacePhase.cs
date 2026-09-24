namespace PoRacer.WormRace
{
    public enum WormRacePhase
    {
        Idle,
        Spawning,
        Countdown,
        Racing,
        Results,
        SeriesComplete,
        SelfTest,
        SelfTestComplete,
        /// <summary>Could not race at all (no rig, bad settings). See WormRaceModel.Message.</summary>
        Error,
    }
}
