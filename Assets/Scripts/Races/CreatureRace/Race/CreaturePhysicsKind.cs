namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Which simulator steps a racer inside Unity, chosen per racer in the settings so a brain
    /// races in the physics it trained in: MuJoCo (and MuJoCo-Warp under Newton) brains on the
    /// org.mujoco plug-in, PhysX-trained brains on ArticulationBody where a creature has a
    /// PhysX builder (the worm does; the generic MuJoCo spawner does not).
    /// </summary>
    public enum CreaturePhysicsKind
    {
        /// <summary>org.mujoco plug-in; every MuJoCo racer shares the race's one MjScene.</summary>
        MujocoPlugin = 0,
        /// <summary>Unity's own PhysX, one ArticulationBody tree per racer.</summary>
        PhysxArticulation = 1,
    }
}
