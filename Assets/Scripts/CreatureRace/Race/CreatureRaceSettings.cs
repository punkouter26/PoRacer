using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// The settings asset of a MuJoCo-only creature race (the quad, and the next creature
    /// trained on MuJoCo Warp): one <see cref="CreatureRaceConfig"/>. A creature with extra
    /// simulators (the worm's PhysX lane) keeps its own asset that embeds the same config.
    /// Written by the creature's scene builder.
    /// </summary>
    [CreateAssetMenu(menuName = "PoRacer/Creature Race Settings", fileName = "CreatureRaceSettings")]
    public sealed class CreatureRaceSettings : ScriptableObject
    {
        [SerializeField] private CreatureRaceConfig _race = new();

        public CreatureRaceConfig Race => _race ??= new CreatureRaceConfig();
    }
}
