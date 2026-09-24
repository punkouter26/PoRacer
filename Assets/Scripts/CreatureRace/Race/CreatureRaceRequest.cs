using System;
using System.Globalization;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// How the editor harness tells a fresh play session what to do.
    ///
    /// Entering play mode reloads the scripting domain, so nothing static survives the trip
    /// from the editor harness into the running scene. A process environment variable does:
    /// the editor sets it just before entering play mode, CreatureRaceSystem reads it once at
    /// start-up and clears it. It never touches disk or the registry (unlike PlayerPrefs), and
    /// in a player build it is simply absent, so the scene falls back to its settings.
    ///
    /// Format: "&lt;mode&gt;:&lt;amount&gt;", where mode is "Race" (amount = number of races) or a
    /// self-test's name or alias (amount = seconds, 0 = its default), e.g. "Race:5", "YawSignTest:2".
    /// </summary>
    public static class CreatureRaceRequest
    {
        public const string ENVIRONMENT_VARIABLE = "PORACER_CREATURERACE_REQUEST";
        public const string RACE_MODE = "Race";

        public static string Format(string mode, float amount)
        {
            return mode + ":" + amount.ToString(CultureInfo.InvariantCulture);
        }

        public static bool IsRace(string mode)
        {
            return string.Equals(mode, RACE_MODE, StringComparison.OrdinalIgnoreCase);
        }

        internal static bool TryConsume(out string mode, out float amount)
        {
            mode = RACE_MODE;
            amount = 0f;
            string request = Environment.GetEnvironmentVariable(ENVIRONMENT_VARIABLE);
            if (string.IsNullOrEmpty(request))
            {
                return false;
            }
            Environment.SetEnvironmentVariable(ENVIRONMENT_VARIABLE, null);

            int separator = request.LastIndexOf(':');
            if (separator <= 0)
            {
                UnityEngine.Debug.LogError($"[CreatureRace] ignoring malformed request '{request}'.");
                return false;
            }
            mode = request.Substring(0, separator);
            return float.TryParse(request.Substring(separator + 1), NumberStyles.Float, CultureInfo.InvariantCulture,
                                  out amount);
        }
    }
}
