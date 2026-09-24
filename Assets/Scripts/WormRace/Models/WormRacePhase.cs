// SUPERSEDED by Assets/Scripts/CreatureRace/Models/CreatureRacePhase.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
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
#endif
