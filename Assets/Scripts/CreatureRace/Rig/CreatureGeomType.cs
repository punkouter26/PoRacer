namespace PoRacer.CreatureRace
{
    /// <summary>The collision shapes a rig may use; each maps to one org.mujoco shape.</summary>
    public enum CreatureGeomType
    {
        /// <summary>MuJoCo capsule: radius and half-length, along the geom's local z (or a fromto).</summary>
        Capsule = 0,
        /// <summary>MuJoCo box: half-extents along the geom's local x, y, z.</summary>
        Box = 1,
        /// <summary>MuJoCo sphere: radius.</summary>
        Sphere = 2,
    }
}
