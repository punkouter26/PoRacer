// SUPERSEDED by Assets/Scripts/CreatureRace/Models/CreatureRacerStatus.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
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
#endif
