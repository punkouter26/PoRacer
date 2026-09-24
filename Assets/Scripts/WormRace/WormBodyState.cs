// SUPERSEDED by Assets/Scripts/CreatureRace/Brain/CreatureBodyState.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// Segment 2's pose and velocity plus the goal, all in ONE right-handed, z-up,
    /// MuJoCo-convention world frame W. Which W that is differs per worm (see
    /// <see cref="WormFrames"/>); the observation only ever projects these onto B's axes,
    /// so it does not matter as long as every field here uses the same one.
    /// </summary>
    internal readonly struct WormBodyState
    {
        /// <summary>B's forward axis (toward the head) in W.</summary>
        public readonly Vector3 AxisX;
        /// <summary>B's left axis in W.</summary>
        public readonly Vector3 AxisY;
        /// <summary>B's up axis in W.</summary>
        public readonly Vector3 AxisZ;
        /// <summary>Segment-2 origin velocity in W, m/s.</summary>
        public readonly Vector3 LinearVelocity;
        /// <summary>Segment-2 angular velocity in W, rad/s.</summary>
        public readonly Vector3 AngularVelocity;
        /// <summary>Race direction in W. Need not be unit length or horizontal.</summary>
        public readonly Vector3 Goal;

        public WormBodyState(Vector3 axisX, Vector3 axisY, Vector3 axisZ,
                             Vector3 linearVelocity, Vector3 angularVelocity, Vector3 goal)
        {
            AxisX = axisX;
            AxisY = axisY;
            AxisZ = axisZ;
            LinearVelocity = linearVelocity;
            AngularVelocity = angularVelocity;
            Goal = goal;
        }

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
#endif
