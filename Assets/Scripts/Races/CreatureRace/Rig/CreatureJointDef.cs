using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One hinge of a rig, on the child body it moves, in that body's MuJoCo frame: the
    /// axis (need not be unit length) and the anchor point. Angles in radians.
    /// </summary>
    public sealed class CreatureJointDef
    {
        public string Name { get; set; } = string.Empty;
        public int Body { get; set; }
        public Vector3 Axis { get; set; } = Vector3.forward;
        public Vector3 Anchor { get; set; }
        public bool Limited { get; set; }
        public float RangeLower { get; set; }
        public float RangeUpper { get; set; }
        /// <summary>Passive joint damping, N*m*s/rad (outside any actuator force limit).</summary>
        public float Damping { get; set; }
        public float Armature { get; set; }
        public float FrictionLoss { get; set; }
        public float Stiffness { get; set; }
        public bool HasSolRefLimit { get; set; }
        /// <summary>x time constant, y damping ratio of the joint-limit constraint.</summary>
        public Vector2 SolRefLimit { get; set; } = new(0.02f, 1f);
    }
}
