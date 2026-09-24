namespace PoRacer.WormRace
{
    /// <summary>
    /// Where the worm race expects its assets. Shared by the runtime (for error messages
    /// that say exactly what to copy where) and by Editor_BuildWormRaceScene (which wires
    /// whatever it finds at these paths into the settings asset).
    /// </summary>
    public static class WormRacePaths
    {
        public const string ROOT = "Assets/WormRace";
        public const string SETTINGS = ROOT + "/WormRaceSettings.asset";
        public const string RIG_JSON = ROOT + "/worm_rig.json";
        public const string BRAINS = ROOT + "/Brains";
        public const string MUJOCO_BRAIN = BRAINS + "/worm_mujoco.onnx";
        public const string ISAAC_BRAIN = BRAINS + "/worm_isaac.onnx";
        public const string MATERIALS = ROOT + "/Materials";
        public const string UI = ROOT + "/UI";
        public const string SCENE = "Assets/Scenes/SCN_WORM_RACE.unity";
    }
}
