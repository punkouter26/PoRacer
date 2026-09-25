namespace PoRacer.CreatureRace
{
    public enum CreatureRacePhase
    {
        Idle,
        Spawning,
        Countdown,
        Racing,
        Results,
        SeriesComplete,
        SelfTest,
        SelfTestComplete,
        /// <summary>Could not race at all (no rig, bad settings). See CreatureRaceModel.Message.</summary>
        Error,
    }
}
