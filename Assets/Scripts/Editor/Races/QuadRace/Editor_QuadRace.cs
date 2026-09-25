#if UNITY_EDITOR
using PoRacer.CreatureRace;
using PoRacer.CreatureRace.EditorTools;
using UnityEditor;

namespace PoRacer.QuadRace.EditorTools
{
    /// <summary>
    /// CLI driver for SCN_QUAD_RACE: the creature template's harness pointed at the quad's
    /// scene and settings. Results are in Logs/quadrace_*.json.
    ///
    /// Invoke:
    ///   unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.Start(5);"
    ///   unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.SelfTest(\"zero\");"
    ///   unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.SelfTest(\"hip\");"
    ///   unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.SelfTest(\"knee\");"
    ///   unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.Status();"
    ///   unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_QuadRace.Stop();"
    /// </summary>
    public static class Editor_QuadRace
    {
        private const string LABEL = "quad race";

        /// <summary>Runs <paramref name="races"/> races back to back.</summary>
        public static string Start(int races = 5)
        {
            return CreatureRaceHarness.Race(QuadRacePaths.SCENE, LABEL, Config(), races);
        }

        /// <summary>
        /// "zero": every servo at rest, each quad must stand still, upright, at its rest height.
        /// "hip" / "knee": +0.5 rad on the front-left hip / knee must read positive and swing
        /// the front-left foot backward (MuJoCo -x). Brains are not used, so NO BRAIN racers
        /// are tested too. <paramref name="seconds"/> 0 uses the defaults (5 s, 2 s).
        /// </summary>
        public static string SelfTest(string mode = "zero", float seconds = 0f)
        {
            return CreatureRaceHarness.SelfTest(QuadRacePaths.SCENE, LABEL, Config(), mode, seconds);
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
            var settings = AssetDatabase.LoadAssetAtPath<CreatureRaceSettings>(QuadRacePaths.SETTINGS);
            return settings != null ? settings.Race : null;
        }
    }
}
#endif
