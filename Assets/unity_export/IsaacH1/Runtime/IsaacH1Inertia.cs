using UnityEngine;

namespace IsaacH1
{
    /// <summary>
    /// Inertia maths shared by the editor rig builder and the runtime agent, so the
    /// prefab and a re-applied floor can never disagree about what a link's inertia is.
    ///
    /// Unity serialises m_InertiaTensor / m_InertiaRotation / m_CenterOfMass on an
    /// ArticulationBody, but NOT contactOffset, solverIterations, maxJointVelocity,
    /// maxAngular/LinearVelocity or maxDepenetrationVelocity - those are runtime-only and
    /// must be re-applied every Awake. IsaacH1Agent.ApplyPerBodyOverrides does that.
    /// </summary>
    public static class IsaacH1Inertia
    {
        /// <summary>
        /// Isaac's diagonal inertia -> Unity's. principalAxes is identity in this export,
        /// so the tensor is diagonal in the link frame; under M the Isaac axes x, y, z
        /// become Unity z, x, y, so the diagonal permutes to (Iyy, Izz, Ixx).
        /// </summary>
        public static Vector3 DiagIsaacToUnity(Vector3 isaac) => new Vector3(isaac.y, isaac.z, isaac.x);

        /// <summary>
        /// The full recipe, in the order the physics requires: permute, floor, then fold
        /// the armature. The floor is applied to the LINK's own inertia (which is what a
        /// floor is for); the armature is rotor inertia and is added afterwards, so a
        /// floor can never mask it or be masked by it.
        /// </summary>
        public static void Compose(Vector3 isaacDiag, bool applyFloor, float floor,
                                   bool foldArmature, Vector3 axisUnity, float armature,
                                   out Vector3 diag, out Quaternion rotation)
        {
            diag = DiagIsaacToUnity(isaacDiag);
            rotation = Quaternion.identity;

            if (applyFloor)
                diag = new Vector3(Mathf.Max(diag.x, floor),
                                   Mathf.Max(diag.y, floor),
                                   Mathf.Max(diag.z, floor));

            if (foldArmature && armature > 0f)
                FoldArmature(ref diag, ref rotation, axisUnity, armature);
        }

        /// <summary>
        /// Adds <paramref name="armature"/> along <paramref name="axis"/> to a diagonal
        /// inertia, returned as eigenvalues + eigenvector rotation (how Unity stores it).
        ///
        /// Adding armature * a*a^T makes a^T I' a == a^T I a + armature, i.e. the
        /// joint-space inertia about THIS axis matches Isaac's PhysX articulation
        /// armature exactly. Ancestor joints whose axes share a component with a are
        /// perturbed slightly - that is the approximation, and it buys a 9x better
        /// explicit-PD conditioning number (RIG_AUDIT.md section C).
        ///
        /// For this rig every joint axis is exactly axis-aligned in its child frame, so
        /// this reduces to a single diagonal add; the general path exists so a future rig
        /// cannot silently get it wrong.
        /// </summary>
        public static void FoldArmature(ref Vector3 diag, ref Quaternion rot,
                                        Vector3 axis, float armature)
        {
            Vector3 a = axis.normalized;

            const float kAligned = 1e-6f;
            for (int i = 0; i < 3; i++)
            {
                float c = i == 0 ? a.x : i == 1 ? a.y : a.z;
                if (Mathf.Abs(Mathf.Abs(c) - 1f) > kAligned) continue;
                if (i == 0) diag.x += armature;
                else if (i == 1) diag.y += armature;
                else diag.z += armature;
                return;
            }

            var m = new float[3, 3];
            m[0, 0] = diag.x; m[1, 1] = diag.y; m[2, 2] = diag.z;
            float[] av = { a.x, a.y, a.z };
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 3; k++)
                    m[i, k] += armature * av[i] * av[k];

            Jacobi(m, out diag, out rot);
        }

        /// <summary>Cyclic Jacobi eigen-decomposition of a symmetric 3x3.</summary>
        static void Jacobi(float[,] a, out Vector3 eigenvalues, out Quaternion eigenvectors)
        {
            var v = new float[3, 3] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            for (int sweep = 0; sweep < 32; sweep++)
            {
                float off = Mathf.Abs(a[0, 1]) + Mathf.Abs(a[0, 2]) + Mathf.Abs(a[1, 2]);
                if (off < 1e-12f) break;
                for (int p = 0; p < 2; p++)
                {
                    for (int q = p + 1; q < 3; q++)
                    {
                        if (Mathf.Abs(a[p, q]) < 1e-15f) continue;
                        float theta = (a[q, q] - a[p, p]) / (2f * a[p, q]);
                        float t = theta == 0f
                            ? 1f
                            : Mathf.Sign(theta) / (Mathf.Abs(theta) + Mathf.Sqrt(theta * theta + 1f));
                        float c = 1f / Mathf.Sqrt(t * t + 1f);
                        float s = t * c;
                        for (int k = 0; k < 3; k++)
                        {
                            float akp = a[k, p], akq = a[k, q];
                            a[k, p] = c * akp - s * akq;
                            a[k, q] = s * akp + c * akq;
                        }
                        for (int k = 0; k < 3; k++)
                        {
                            float apk = a[p, k], aqk = a[q, k];
                            a[p, k] = c * apk - s * aqk;
                            a[q, k] = s * apk + c * aqk;
                        }
                        for (int k = 0; k < 3; k++)
                        {
                            float vkp = v[k, p], vkq = v[k, q];
                            v[k, p] = c * vkp - s * vkq;
                            v[k, q] = s * vkp + c * vkq;
                        }
                    }
                }
            }
            eigenvalues = new Vector3(a[0, 0], a[1, 1], a[2, 2]);
            Vector3 col0 = new Vector3(v[0, 0], v[1, 0], v[2, 0]).normalized;
            Vector3 col1 = new Vector3(v[0, 1], v[1, 1], v[2, 1]).normalized;
            Vector3 col2 = Vector3.Cross(col0, col1).normalized;
            col1 = Vector3.Cross(col2, col0).normalized;
            eigenvectors = Quaternion.LookRotation(col2, col1);
        }
    }
}
