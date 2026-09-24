using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// The reference body B's pose and velocity plus the goal, all in ONE right-handed, z-up,
    /// MuJoCo-convention world frame W. Which W that is differs per simulator (see
    /// <see cref="CreatureFrames"/>); the observation only projects these onto B's axes, so
    /// it does not matter as long as every field uses the same one.
    /// </summary>
    public readonly struct CreatureBodyState
    {
        public CreatureBodyState(Vector3 axisX, Vector3 axisY, Vector3 axisZ,
                                 Vector3 linearVelocity, Vector3 angularVelocity, Vector3 goal)
        {
            AxisX = axisX;
            AxisY = axisY;
            AxisZ = axisZ;
            LinearVelocity = linearVelocity;
            AngularVelocity = angularVelocity;
            Goal = goal;
        }

        /// <summary>B's forward axis in W.</summary>
        public Vector3 AxisX { get; }
        /// <summary>B's left axis in W.</summary>
        public Vector3 AxisY { get; }
        /// <summary>B's up axis in W.</summary>
        public Vector3 AxisZ { get; }
        /// <summary>B's origin velocity in W, m/s.</summary>
        public Vector3 LinearVelocity { get; }
        /// <summary>B's angular velocity in W, rad/s.</summary>
        public Vector3 AngularVelocity { get; }
        /// <summary>Race direction in W. Need not be unit length or horizontal.</summary>
        public Vector3 Goal { get; }

        public bool IsFinite()
        {
            return Finite(AxisX) && Finite(AxisY) && Finite(AxisZ)
                && Finite(LinearVelocity) && Finite(AngularVelocity);
        }

        private static bool Finite(Vector3 value)
        {
            return float.IsFinite(value.x) && float.IsFinite(value.y) && float.IsFinite(value.z);
        }
    }
}
