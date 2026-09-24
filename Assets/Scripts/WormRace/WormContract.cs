namespace PoRacer.WormRace
{
    /// <summary>
    /// The trained interface both worm brains share, copied from training/worm/WORM_SPEC.md.
    /// Change it there first; both trainers and this file must move together, or a policy
    /// is fed plausible-looking garbage.
    ///
    /// <code>
    ///   obs[ 0.. 2]  gravity direction in B (unit)                 x1
    ///   obs[ 3.. 5]  segment-2 linear velocity in B                x0.5
    ///   obs[ 6.. 8]  segment-2 angular velocity in B               x0.25
    ///   obs[ 9..16]  joint positions, action order                 / 0.785398
    ///   obs[17..24]  joint velocities, action order                x0.1
    ///   obs[25..32]  previous action                               x1
    ///   obs[33..34]  goal direction in B, horizontal (x, y), unit  x1
    ///
    ///   action order: j0_pitch j0_yaw j1_pitch j1_yaw j2_pitch j2_yaw j3_pitch j3_yaw
    ///   target_rad = clip(action, -1, 1) * 0.785398, held for 4 physics steps of 0.005 s
    /// </code>
    ///
    /// B is segment 2, the middle segment, in MuJoCo's convention: x forward (toward the
    /// head), y left, z up. The observation itself is assembled by the creature template
    /// (CreatureObservation) from the settings' observation definition, which
    /// Editor_BuildWormRaceScene writes to exactly this layout.
    /// </summary>
    internal static class WormContract
    {
        public const int ACTION_SIZE = 8;
        public const int SEGMENT_COUNT = 5;
        public const int REFERENCE_SEGMENT = 2;

        /// <summary>45 degrees. Both the joint limit and the action scale.</summary>
        public const float JOINT_RANGE_RAD = 0.785398f;

        public const int DECIMATION = 4;
        public const float PHYSICS_DT = 0.005f;

        /// <summary>Action order, verified against worm_rig.json's actionOrder at load.</summary>
        public static readonly string[] ActionOrder =
        {
            "j0_pitch", "j0_yaw", "j1_pitch", "j1_yaw",
            "j2_pitch", "j2_yaw", "j3_pitch", "j3_yaw",
        };
    }
}
