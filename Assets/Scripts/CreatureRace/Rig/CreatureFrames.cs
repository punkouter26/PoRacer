using Mujoco;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Every frame map a creature racer needs, side by side, because mixing them up is the
    /// easiest mistake to make here and the hardest to see.
    ///
    /// PLUG-IN map (org.mujoco MjEngineTool): what a MuJoCo-plug-in racer uses, because the
    /// plug-in owns that conversion and it is a plain y/z swap.
    ///   MuJoCo (x, y, z) = Unity (x, z, y); quaternions (w, x, y, z) -> Unity (x, z, y, -w).
    ///
    /// SPEC map (WORM_SPEC.md "Unity mapping", training/bugs/README.md, quad_rig.json
    /// "frames"): what a creature built directly in Unity's PhysX uses.
    ///   positions (polar):  Unity (x, y, z) = MuJoCo (-y, z, x), so MuJoCo = (z, -x, y)
    ///   hinge axes (axial): Unity a -> MuJoCo (-a.z, a.x, -a.y)
    ///   Both carry det = -1, so a positive Unity joint angle is a positive MuJoCo qpos.
    ///
    /// Both MuJoCo-side frames are right-handed and z-up, and every observation is built in
    /// the reference body's own frame, so a policy sees the same numbers in either world
    /// (their worlds differ only by a yaw, like the spawn yaw the trainers randomise).
    /// </summary>
    public static class CreatureFrames
    {
        private const float IDENTITY_TOLERANCE = 1e-7f;

        public static Vector3 PluginFromUnity(Vector3 unity)
        {
            return new Vector3(unity.x, unity.z, unity.y);
        }

        /// <summary>The swap is its own inverse; named separately so call sites read right.</summary>
        public static Vector3 UnityFromPlugin(Vector3 mujoco)
        {
            return new Vector3(mujoco.x, mujoco.z, mujoco.y);
        }

        /// <summary>
        /// A MuJoCo-frame rotation (components as <see cref="CreatureBodyDef.Rotation"/> stores
        /// them) as a Unity local rotation under the plug-in map. Identity stays the exact
        /// Unity identity, so an unrotated body writes the same MJCF as one built by hand.
        /// </summary>
        public static Quaternion UnityRotationFromPlugin(Quaternion mujoco)
        {
            if (IsIdentity(mujoco))
            {
                return Quaternion.identity;
            }
            return MjEngineTool.UnityQuaternion(Normalised(mujoco));
        }

        /// <summary>Rotates a MuJoCo-frame vector by a MuJoCo-frame quaternion.</summary>
        public static Vector3 Rotate(Quaternion mujoco, Vector3 vector)
        {
            return IsIdentity(mujoco) ? vector : Normalised(mujoco) * vector;
        }

        public static Vector3 SpecFromUnityPolar(Vector3 unity)
        {
            return new Vector3(unity.z, -unity.x, unity.y);
        }

        public static Vector3 SpecFromUnityAxial(Vector3 unity)
        {
            return new Vector3(-unity.z, unity.x, -unity.y);
        }

        public static Vector3 UnityFromSpecPolar(Vector3 mujoco)
        {
            return new Vector3(-mujoco.y, mujoco.z, mujoco.x);
        }

        /// <summary>Inverse of <see cref="SpecFromUnityAxial"/>.</summary>
        public static Vector3 UnityFromSpecAxial(Vector3 mujoco)
        {
            return new Vector3(mujoco.y, -mujoco.z, -mujoco.x);
        }

        private static bool IsIdentity(Quaternion value)
        {
            return Mathf.Abs(value.x) < IDENTITY_TOLERANCE && Mathf.Abs(value.y) < IDENTITY_TOLERANCE
                && Mathf.Abs(value.z) < IDENTITY_TOLERANCE && Mathf.Abs(Mathf.Abs(value.w) - 1f) < IDENTITY_TOLERANCE;
        }

        private static Quaternion Normalised(Quaternion value)
        {
            float norm = Mathf.Sqrt(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);
            return norm > 0f
                ? new Quaternion(value.x / norm, value.y / norm, value.z / norm, value.w / norm)
                : Quaternion.identity;
        }
    }
}
