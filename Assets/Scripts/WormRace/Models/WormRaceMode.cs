namespace PoRacer.WormRace
{
    /// <summary>What a play session of SCN_WORM_RACE does. See WormRaceRequest.</summary>
    public enum WormRaceMode
    {
        /// <summary>N races back to back, results to Logs/wormrace_*.json.</summary>
        Race,
        /// <summary>Both worms at zero action: must lie still, straight, on the floor.</summary>
        ZeroActionTest,
        /// <summary>j0_yaw = +0.5 rad: segment 1 must swing to the worm's right (Unity +x).</summary>
        YawSignTest,
        /// <summary>j0_pitch = +0.5 rad: segment 1 must rise relative to the head.</summary>
        PitchSignTest,
    }
}
