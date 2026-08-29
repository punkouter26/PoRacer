using UnityEngine;

namespace MujocoBiped
{
    /// <summary>
    /// MuJoCo (right-handed, Z-up, X-forward) &lt;-&gt; Unity (left-handed, Y-up, Z-forward).
    ///
    /// The map is the single linear operator
    ///
    ///     M : (x, y, z)_mujoco  ->  (-y, z, x)_unity          det(M) = -1
    ///
    /// det(M) is negative because the map reverses handedness, and that one fact drives
    /// everything else here:
    ///
    ///   * TRUE VECTORS (position, linear velocity, force, a gravity direction)
    ///     transform by M.                                      -> <see cref="Pos"/>
    ///
    ///   * PSEUDOVECTORS (angular velocity, torque, rotation axes, and the vector part
    ///     of a quaternion) pick up an extra sign flip under an orientation-reversing
    ///     map, so they transform by -M.                        -> <see cref="Axis"/>
    ///
    /// A rotation about axis a by angle t in MuJoCo is the same physical motion as a
    /// rotation about -M*a by +t in Unity. Building every revolute anchor with its local
    /// +X at -M*axis is what makes a positive Unity joint angle mean a positive MuJoCo
    /// joint angle, so the agent never flips a sign when filling observations or applying
    /// torques. See CONTRACT.md.
    ///
    /// The quaternion map below is the one the export's own README arrived at after an
    /// exhaustive search over all 24 signed axis permutations: (w,x,y,z) -> (y,-z,-x,w).
    /// The form originally requested there, (-x,z,-y,w), is not a rotation map under ANY
    /// position convention - do not reintroduce it.
    /// </summary>
    public static class MujocoBipedFrameMap
    {
        /// <summary>True vectors: M. MuJoCo (x,y,z) -> Unity (-y, z, x).</summary>
        public static Vector3 Pos(Vector3 m) => new Vector3(-m.y, m.z, m.x);

        /// <summary>Inverse of <see cref="Pos"/>. Unity (x,y,z) -> MuJoCo (z, -x, y).</summary>
        public static Vector3 PosToMujoco(Vector3 u) => new Vector3(u.z, -u.x, u.y);

        /// <summary>Pseudovectors and rotation axes: -M. MuJoCo (x,y,z) -> Unity (y, -z, -x).</summary>
        public static Vector3 Axis(Vector3 m) => new Vector3(m.y, -m.z, -m.x);

        /// <summary>Inverse of <see cref="Axis"/>. Unity (x,y,z) -> MuJoCo (-z, x, -y).</summary>
        public static Vector3 AxisToMujoco(Vector3 u) => new Vector3(-u.z, u.x, -u.y);

        /// <summary>MuJoCo quaternion, stored (w, x, y, z), -> Unity.</summary>
        public static Quaternion RotFromWxyz(float w, float x, float y, float z)
            => new Quaternion(y, -z, -x, w);

        /// <summary>Unity rotation -> MuJoCo quaternion as (w, x, y, z).</summary>
        public static Vector4 RotToMujocoWxyz(Quaternion q)
            => new Vector4(q.w, -q.z, q.x, -q.y);

        /// <summary>
        /// An inertia tensor's diagonal, MuJoCo axes -> Unity axes. Under M the MuJoCo
        /// axes x, y, z become Unity z, x, y, so the diagonal permutes to (Iyy, Izz, Ixx).
        /// Valid only while the tensor is diagonal in the link frame, which it is for
        /// every body in this rig (proven by the symmetry argument in RIG_AUDIT.md
        /// section E: every geom is symmetric about the body's x = 0 and y = 0 planes,
        /// so all three products of inertia vanish and MuJoCo's iquat is the identity).
        /// </summary>
        public static Vector3 InertiaDiag(Vector3 m) => new Vector3(m.y, m.z, m.x);

        /// <summary>
        /// A rotation whose local +X lies along <paramref name="unityAxis"/>. Unity's
        /// revolute ArticulationBody always twists about the anchor frame's X, so this is
        /// what <c>anchorRotation</c> must be. Built from an explicit orthonormal basis
        /// rather than Quaternion.FromToRotation, which is degenerate when the axis is
        /// antiparallel to +X - true here for every hip_x and ankle_x joint.
        /// </summary>
        public static Quaternion AnchorRotationForAxis(Vector3 unityAxis)
        {
            Vector3 ax = unityAxis.normalized;
            Vector3 seed = Mathf.Abs(Vector3.Dot(ax, Vector3.up)) > 0.9f ? Vector3.forward : Vector3.up;
            Vector3 fwd = Vector3.Cross(ax, seed).normalized;   // local +Z
            Vector3 up = Vector3.Cross(fwd, ax).normalized;     // local +Y
            // Unity: right = cross(up, forward) = cross(cross(fwd, ax), fwd) = ax
            return Quaternion.LookRotation(fwd, up);
        }

        /// <summary>
        /// The creature's heading, as MuJoCo measures it: atan2(R[1,0], R[0,0]) of the
        /// torso rotation matrix, i.e. the yaw of MuJoCo's +X axis about MuJoCo's +Z.
        /// MuJoCo +X is Unity +Z and MuJoCo +Y is Unity -X, so the same quantity in Unity
        /// is atan2(-forward.x, forward.z). Used for the target-direction observation,
        /// which is rotated by yaw ONLY - never by the full orientation.
        /// </summary>
        public static float HeadingRad(Quaternion unityRotation)
        {
            Vector3 fwd = unityRotation * Vector3.forward;
            return Mathf.Atan2(-fwd.x, fwd.z);
        }
    }
}
