#if UNITY_EDITOR
namespace PoRacer.QuadRace.EditorTools
{
    /// <summary>
    /// Where the quad race keeps its assets. The quad needs no runtime code of its own: it
    /// races on the creature template (CreatureRaceLifetimeScope, MujocoCreatureSpawnSystem),
    /// configured by QuadRaceSettings.asset.
    /// </summary>
    public static class QuadRacePaths
    {
        public const string SETTINGS = "Assets/Races/QuadRace/QuadRaceSettings.asset";
        public const string SCENE = "Assets/Scenes/SCN_QUAD_RACE.unity";
    }
}
#endif
