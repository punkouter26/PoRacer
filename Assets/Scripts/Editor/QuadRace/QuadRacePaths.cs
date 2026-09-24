#if UNITY_EDITOR
namespace PoRacer.QuadRace.EditorTools
{
    /// <summary>
    /// Where the quad race keeps its assets, and where it copies them from. The quad needs no
    /// runtime code of its own: it races on the creature template (CreatureRaceLifetimeScope,
    /// MujocoCreatureSpawnSystem), configured by QuadRaceSettings.asset.
    /// </summary>
    public static class QuadRacePaths
    {
        public const string ROOT = "Assets/QuadRace";
        public const string SETTINGS = ROOT + "/QuadRaceSettings.asset";
        public const string RIG_JSON = ROOT + "/quad_rig.json";
        public const string BRAINS = ROOT + "/Brains";
        public const string MATERIALS = ROOT + "/Materials";
        public const string UI = ROOT + "/UI";
        public const string SCENE = "Assets/Scenes/SCN_QUAD_RACE.unity";

        /// <summary>The trainer's rig, the single source of truth (build_quad.py writes it).</summary>
        public const string TRAINING_RIG_JSON = "training/quad/quad_rig.json";
        /// <summary>The trainers' ONNX exports (QUAD_SPEC "Export and tests").</summary>
        public const string TRAINING_EXPORT = "training/quad/export/";
    }
}
#endif
