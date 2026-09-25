#if UNITY_EDITOR
using PoRacer.CreatureRace;
using PoRacer.CreatureRace.EditorTools;
using UnityEditor;

namespace PoRacer.WormRace.EditorTools
{
    /// <summary>
    /// CLI driver for SCN_WORM_RACE: the creature template's harness (CreatureRaceHarness)
    /// pointed at the worm's scene and settings. Results are in Logs/wormrace_*.json.
    ///
    /// Invoke:
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Start(5);"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"zero\");"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"yaw\");"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"pitch\");"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Status();"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Stop();"
    /// </summary>
    public static class Editor_WormRace
    {
        private const string LABEL = "worm race";

        /// <summary>Runs <paramref name="races"/> races back to back.</summary>
        public static string Start(int races = 5)
        {
            return CreatureRaceHarness.Race(WormRacePaths.SCENE, LABEL, Config(), races);
        }

        /// <summary>
        /// Every racer is tested, each in its own physics, and every one must pass (a racer
        /// without a brain too: the tests use fixed actions).
        /// "zero": every worm at zero action must lie still and straight on the floor.
        /// "yaw":  j0_yaw = +0.5 rad must swing segment 1 to Unity +x.
        /// "pitch": j0_pitch = +0.5 rad must lift segment 1 relative to the head.
        /// <paramref name="seconds"/> 0 uses the defaults (5 s zero, 2 s sign tests).
        /// </summary>
        public static string SelfTest(string mode = "zero", float seconds = 0f)
        {
            return CreatureRaceHarness.SelfTest(WormRacePaths.SCENE, LABEL, Config(), mode, seconds);
        }

        public static string Status()
        {
            return CreatureRaceHarness.Status();
        }

        /// <summary>Abandons a running job and leaves play mode.</summary>
        public static string Stop()
        {
            return CreatureRaceHarness.Stop();
        }

        private static CreatureRaceConfig Config()
        {
            var settings = AssetDatabase.LoadAssetAtPath<WormRaceSettings>(WormRacePaths.SETTINGS);
            return settings != null ? settings.Race : null;
        }
    }
}
#endif
