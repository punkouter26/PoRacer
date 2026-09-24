using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One MuJoCo body of a rig, in MuJoCo coordinates (x forward, y left, z up), relative
    /// to its parent. Quaternions keep MuJoCo's components in Unity's struct: x, y, z are
    /// MuJoCo's vector part and w its scalar part (CreatureFrames converts them).
    ///
    /// Mass either sits here as an explicit inertial (<see cref="HasInertial"/>: MuJoCo's
    /// &lt;inertial&gt;, which then overrides every geom's mass) or on the body's geoms.
    /// </summary>
    public sealed class CreatureBodyDef
    {
        public string Name { get; set; } = string.Empty;
        /// <summary>Index into the rig's bodies; -1 for the root (always body 0).</summary>
        public int Parent { get; set; } = -1;
        public Vector3 Position { get; set; }
        public Quaternion Rotation { get; set; } = Quaternion.identity;
        /// <summary>The root's &lt;freejoint&gt;. Only the root may have one.</summary>
        public bool FreeJoint { get; set; }
        public bool HasInertial { get; set; }
        public float Mass { get; set; }
        public Vector3 InertiaDiagonal { get; set; }
        public Vector3 InertialPosition { get; set; }
        public Quaternion InertialRotation { get; set; } = Quaternion.identity;
    }
}
