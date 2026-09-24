// SUPERSEDED by Assets/Scripts/CreatureRace/Race/CreaturePhysicsKind.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
namespace PoRacer.WormRace
{
    /// <summary>
    /// Which simulator steps a racer inside Unity. Chosen per racer in WormRaceSettings, so
    /// a brain can race in the physics it trained in: MuJoCo (and MuJoCo-Warp under Newton)
    /// brains on the org.mujoco plug-in, PhysX-trained brains on ArticulationBody.
    /// </summary>
    internal enum WormPhysicsKind
    {
        /// <summary>org.mujoco plug-in; every MuJoCo racer shares the race's one MjScene.</summary>
        MujocoPlugin = 0,
        /// <summary>Unity's own PhysX, one ArticulationBody chain per racer.</summary>
        PhysxArticulation = 1,
    }
}
#endif
