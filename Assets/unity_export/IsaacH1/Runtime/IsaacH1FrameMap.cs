using UnityEngine;

using PoRacer.IsaacPorts;

namespace IsaacH1
{
    /// <summary>
    /// Isaac Lab (right-handed, Z-up, X-forward) &lt;-&gt; Unity (left-handed, Y-up, Z-forward).
    ///
    /// Since 2026-09-02 this is a thin forwarder to the shared <see cref="IsaacFrameMap"/>
    /// in <c>Assets/unity_export/Shared/</c>, which the Boy port also uses. The maths and
    /// its rationale (M for true vectors, -M for pseudovectors, anchors built at -M*axis)
    /// are documented there; the H1 tests and CONTRACT.md keep referring to this name.
    /// </summary>
    public static class IsaacH1FrameMap
    {
        /// <summary>True vectors: M. Isaac (x,y,z) -> Unity (-y, z, x).</summary>
        public static Vector3 Pos(Vector3 i) => IsaacFrameMap.Pos(i);

        /// <summary>Inverse of <see cref="Pos"/>. Unity (x,y,z) -> Isaac (z, -x, y).</summary>
        public static Vector3 PosToIsaac(Vector3 u) => IsaacFrameMap.PosToIsaac(u);

        /// <summary>Pseudovectors and rotation axes: -M. Isaac (x,y,z) -> Unity (y, -z, -x).</summary>
        public static Vector3 Axis(Vector3 i) => IsaacFrameMap.Axis(i);

        /// <summary>Inverse of <see cref="Axis"/>. Unity (x,y,z) -> Isaac (-z, x, -y).</summary>
        public static Vector3 AxisToIsaac(Vector3 u) => IsaacFrameMap.AxisToIsaac(u);

        /// <summary>
        /// Isaac quaternion (components given in XYZW order) -> Unity.
        /// NOTE: isaac_reference.json stores root orientation as XYZW even though the
        /// raw export names the field "..._wxyz" - proven in RIG_AUDIT.md section D.
        /// </summary>
        public static Quaternion RotFromXyzw(float x, float y, float z, float w)
            => IsaacFrameMap.RotFromXyzw(x, y, z, w);

        /// <summary>Isaac quaternion given in WXYZ order (the USD storage order) -> Unity.</summary>
        public static Quaternion RotFromWxyz(float w, float x, float y, float z)
            => IsaacFrameMap.RotFromWxyz(w, x, y, z);

        /// <summary>Unity rotation -> Isaac quaternion as (x, y, z, w).</summary>
        public static Vector4 RotToIsaacXyzw(Quaternion q) => IsaacFrameMap.RotToIsaacXyzw(q);

        /// <summary>A rotation whose local +X lies along <paramref name="unityAxis"/>.</summary>
        public static Quaternion AnchorRotationForAxis(Vector3 unityAxis)
            => IsaacFrameMap.AnchorRotationForAxis(unityAxis);
    }
}
