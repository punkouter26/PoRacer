// SUPERSEDED by Assets/Scripts/CreatureRace/Rig/CreatureFrames.cs (creature template, 2026-09-24).
// Kept out of compilation, content unchanged, until someone deletes this file and its .meta.
#if PORACER_WORMRACE_LEGACY
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// The two coordinate maps this scene uses, kept side by side because mixing them up is
    /// the easiest mistake to make here and the hardest to see.
    ///
    /// SPEC map (WORM_SPEC.md "Unity mapping", training/bugs/README.md): used for the PhysX
    /// (Isaac) worm, which is built directly in Unity.
    ///   positions (polar):     Unity (x, y, z) = MuJoCo (-y, z, x)   so MuJoCo = (z, -x, y)
    ///   hinge axes (axial):    Unity a -> MuJoCo (-a.z, a.x, -a.y)
    ///   Both carry det = -1, so a positive Unity joint angle equals a positive MuJoCo qpos.
    ///
    /// PLUG-IN map (org.mujoco MjEngineTool.MjVector3): used for the MuJoCo worm, because the
    /// plug-in owns that conversion and it is a plain y/z swap.
    ///   MuJoCo (x, y, z) = Unity (x, z, y)
    ///
    /// Both MuJoCo-side frames are right-handed and z-up, and the observation is built in the
    /// body frame of segment 2, so the two worms feed their policies the same numbers even
    /// though their world frames differ by a yaw. The goal direction is expressed in each
    /// worm's own world frame, which is what keeps that true.
    /// </summary>
    internal static class WormFrames
    {
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

        public static Vector3 PluginFromUnity(Vector3 unity)
        {
            return new Vector3(unity.x, unity.z, unity.y);
        }

        /// <summary>The swap is its own inverse; named separately so call sites read right.</summary>
        public static Vector3 UnityFromPlugin(Vector3 mujoco)
        {
            return new Vector3(mujoco.x, mujoco.z, mujoco.y);
        }
    }
}
#endif
