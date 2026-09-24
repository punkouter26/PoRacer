using System;
using System.Globalization;

namespace PoRacer.WormRace
{
    /// <summary>
    /// How the editor harness tells a fresh play session what to do.
    ///
    /// Entering play mode reloads the scripting domain, so nothing static survives the trip
    /// from Editor_WormRace into the running scene. A process environment variable does:
    /// the editor sets it just before entering play mode, WormRaceSystem reads it once at
    /// start-up and clears it. It never touches disk or the registry (unlike PlayerPrefs),
    /// and in a player build it is simply absent, so the scene falls back to its settings.
    ///
    /// Format: "&lt;mode&gt;:&lt;amount&gt;", e.g. "Race:5" (five races) or "YawSignTest:2" (two
    /// seconds of the j0_yaw sign test).
    /// </summary>
    public static class WormRaceRequest
    {
        public const string ENVIRONMENT_VARIABLE = "PORACER_WORMRACE_REQUEST";

        public static string Format(WormRaceMode mode, float amount)
        {
            return mode + ":" + amount.ToString(CultureInfo.InvariantCulture);
        }

        internal static bool TryConsume(out WormRaceMode mode, out float amount)
        {
            mode = WormRaceMode.Race;
            amount = 0f;
            string request = Environment.GetEnvironmentVariable(ENVIRONMENT_VARIABLE);
            if (string.IsNullOrEmpty(request))
            {
                return false;
            }
            Environment.SetEnvironmentVariable(ENVIRONMENT_VARIABLE, null);

            string[] parts = request.Split(':');
            if (parts.Length != 2 || !Enum.TryParse(parts[0], true, out mode))
            {
                UnityEngine.Debug.LogError($"[WormRace] ignoring malformed request '{request}'.");
                return false;
            }
            return float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out amount);
        }
    }
}
