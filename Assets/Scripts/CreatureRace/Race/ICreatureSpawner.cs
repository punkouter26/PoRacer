namespace PoRacer.CreatureRace
{
    /// <summary>
    /// What the race system needs from a creature: its parsed rig and racing layout, and a
    /// way to put every racer on the grid and take them away again. One implementation per
    /// kind of creature scene: <see cref="MujocoCreatureSpawnSystem"/> for any MuJoCo-only
    /// creature, the worm's own spawner for its mixed MuJoCo/PhysX grid.
    ///
    /// Spawn builds every racer (lane = pilot index) in the frame it is called, so the MuJoCo
    /// world compiles them all together at the top of the next frame; Despawn and the next
    /// Spawn must be at least one frame apart (Destroy is deferred and MjScene is a singleton).
    /// </summary>
    public interface ICreatureSpawner
    {
        bool MujocoSupported { get; }

        /// <summary>Parses the rig once and resolves the layout; false with a reason when it cannot race.</summary>
        bool TryPrepare(out CreatureLayout layout, out string error);

        void Spawn(CreatureLayout layout, CreaturePilot[] pilots);

        void Despawn();
    }
}
