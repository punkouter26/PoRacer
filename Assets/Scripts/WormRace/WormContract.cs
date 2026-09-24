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
    /// head), y left, z up.
    /// </summary>
    internal static class WormContract
    {
        public const int OBS_SIZE = 35;
        public const int ACTION_SIZE = 8;
        public const int SEGMENT_COUNT = 5;
        public const int HEAD_SEGMENT = 0;
        public const int SECOND_SEGMENT = 1;
        public const int REFERENCE_SEGMENT = 2;

        /// <summary>45 degrees. Both the joint limit and the action scale.</summary>
        public const float JOINT_RANGE_RAD = 0.785398f;

        public const int DECIMATION = 4;
        public const float PHYSICS_DT = 0.005f;

        public const float LINEAR_VELOCITY_SCALE = 0.5f;
        public const float ANGULAR_VELOCITY_SCALE = 0.25f;
        public const float JOINT_VELOCITY_SCALE = 0.1f;

        public const int OBS_GRAVITY = 0;
        public const int OBS_LINEAR_VELOCITY = 3;
        public const int OBS_ANGULAR_VELOCITY = 6;
        public const int OBS_JOINT_POSITION = 9;
        public const int OBS_JOINT_VELOCITY = 17;
        public const int OBS_PREVIOUS_ACTION = 25;
        public const int OBS_GOAL = 33;

        public const string INPUT_NAME = "obs";
        public const string OUTPUT_NAME = "actions";

        /// <summary>Action order, verified against worm_rig.json's actionOrder at load.</summary>
        public static readonly string[] ActionOrder =
        {
            "j0_pitch", "j0_yaw", "j1_pitch", "j1_yaw",
            "j2_pitch", "j2_yaw", "j3_pitch", "j3_yaw",
        };

        /// <summary>Index of j0_yaw in the action vector; the sign test drives it.</summary>
        public const int J0_YAW_INDEX = 1;

        /// <summary>Index of j0_pitch in the action vector; the pitch sign test drives it.</summary>
        public const int J0_PITCH_INDEX = 0;
    }
}
