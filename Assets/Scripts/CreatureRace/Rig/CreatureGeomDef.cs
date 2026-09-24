using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// One collision geom of a rig body, in MuJoCo coordinates of that body. A capsule is
    /// placed either by a fromto segment (<see cref="HasFromTo"/>) or by pos/quat with its
    /// axis along the geom's local z, exactly as in MJCF.
    /// </summary>
    public sealed class CreatureGeomDef
    {
        public string Name { get; set; } = string.Empty;
        public int Body { get; set; }
        public CreatureGeomType Type { get; set; }
        /// <summary>Capsule and sphere radius.</summary>
        public float Radius { get; set; }
        /// <summary>Capsule half-length of the cylinder part (without the caps).</summary>
        public float HalfLength { get; set; }
        /// <summary>Box half-sizes along the geom's local x, y, z.</summary>
        public Vector3 HalfExtents { get; set; }
        public Vector3 Position { get; set; }
        public Quaternion Rotation { get; set; } = Quaternion.identity;
        public bool HasFromTo { get; set; }
        public Vector3 From { get; set; }
        public Vector3 To { get; set; }
        /// <summary>Explicit geom mass in kg; 0 = none (the body's inertial carries the mass).</summary>
        public float Mass { get; set; }
        public CreatureContact Contact { get; set; } = CreatureContact.MujocoDefault;
    }
}
