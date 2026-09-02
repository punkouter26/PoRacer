using UnityEngine;

namespace PoRacer.IsaacPorts
{
    /// <summary>
    /// Isaac Lab (right-handed, Z-up, X-forward) &lt;-&gt; Unity (left-handed, Y-up, Z-forward),
    /// shared by every Isaac port. The IsaacH1 port carried this as <c>IsaacH1FrameMap</c>;
    /// that class now forwards here so the two can never disagree.
    ///
    /// The map is the single linear operator
    ///
    ///     M : (x, y, z)_isaac  ->  (-y, z, x)_unity          det(M) = -1
    ///
    /// det(M) is negative because the map reverses handedness. That one fact drives
    /// everything else here:
    ///
    ///   * TRUE VECTORS (position, linear velocity, force, a gravity direction)
    ///     transform by M.                                     -> <see cref="Pos"/>
    ///
    ///   * PSEUDOVECTORS (angular velocity, torque, rotation axes, and the vector
    ///     part of a quaternion) pick up an extra sign flip under an
    ///     orientation-reversing map, so they transform by -M.  -> <see cref="Axis"/>
    ///
    /// A rotation about axis a by angle t in Isaac is the same physical motion as a
    /// rotation about -M*a by +t in Unity. Building every revolute anchor with its
    /// X axis at -M*axis is what makes a positive Unity joint angle mean a positive
    /// Isaac joint angle, so an agent never flips a sign when filling observations
    /// or applying actions.
    /// </summary>
    public static class IsaacFrameMap
    {
        /// <summary>True vectors: M. Isaac (x,y,z) -> Unity (-y, z, x).</summary>
        public static Vector3 Pos(Vector3 i) => new Vector3(-i.y, i.z, i.x);

        /// <summary>Inverse of <see cref="Pos"/>. Unity (x,y,z) -> Isaac (z, -x, y).</summary>
        public static Vector3 PosToIsaac(Vector3 u) => new Vector3(u.z, -u.x, u.y);

        /// <summary>Pseudovectors and rotation axes: -M. Isaac (x,y,z) -> Unity (y, -z, -x).</summary>
        public static Vector3 Axis(Vector3 i) => new Vector3(i.y, -i.z, -i.x);

        /// <summary>Inverse of <see cref="Axis"/>. Unity (x,y,z) -> Isaac (-z, x, -y).</summary>
        public static Vector3 AxisToIsaac(Vector3 u) => new Vector3(-u.z, u.x, -u.y);

        /// <summary>Isaac quaternion given as (x, y, z, w) -> Unity. Vector part maps by -M.</summary>
        public static Quaternion RotFromXyzw(float x, float y, float z, float w)
            => new Quaternion(y, -z, -x, w);

        /// <summary>Isaac quaternion given as (w, x, y, z) (USD storage order) -> Unity.</summary>
        public static Quaternion RotFromWxyz(float w, float x, float y, float z)
            => new Quaternion(y, -z, -x, w);

        /// <summary>Unity rotation -> Isaac quaternion as (x, y, z, w).</summary>
        public static Vector4 RotToIsaacXyzw(Quaternion q)
            => new Vector4(-q.z, q.x, -q.y, q.w);

        /// <summary>
        /// A diagonal inertia expressed on Isaac's axes, re-labelled onto Unity's. Under M
        /// the Isaac axes x, y, z become Unity z, x, y, so (Ixx, Iyy, Izz) -> (Iyy, Izz, Ixx).
        /// Valid only when the tensor is diagonal in the link frame (principal axes identity).
        /// </summary>
        public static Vector3 InertiaDiagToUnity(Vector3 isaac) => new Vector3(isaac.y, isaac.z, isaac.x);

        /// <summary>
        /// A rotation whose local +X lies along <paramref name="unityAxis"/>. Unity's
        /// revolute ArticulationBody always twists about the anchor frame's X, so this
        /// is what <c>anchorRotation</c> must be. Built from an explicit orthonormal
        /// basis rather than Quaternion.FromToRotation, which is degenerate when the
        /// axis is antiparallel to +X (true for several joints on every rig).
        /// </summary>
        public static Quaternion AnchorRotationForAxis(Vector3 unityAxis)
        {
            Vector3 ax = unityAxis.normalized;
            Vector3 seed = Mathf.Abs(Vector3.Dot(ax, Vector3.up)) > 0.9f ? Vector3.forward : Vector3.up;
            Vector3 fwd = Vector3.Cross(ax, seed).normalized;   // local +Z
            Vector3 up = Vector3.Cross(fwd, ax).normalized;     // local +Y
            return Quaternion.LookRotation(fwd, up);
        }
    }
}
