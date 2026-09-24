using System;
using Mujoco;
using UnityEngine;

namespace PoRacer.WormRace
{
    /// <summary>
    /// The one render mesh every worm segment shares: a capsule of exactly the collision
    /// size (radius 0.045 m, 0.24 m tip to tip), along local +Y. Built once per session
    /// with the MuJoCo plug-in's own generator - the same one MjMeshFilter uses, so the
    /// winding is already right for Unity - and destroyed by WormSpawnSystem.Dispose.
    ///
    /// A scaled primitive capsule would squash the end caps; this does not.
    /// </summary>
    internal static class WormCapsuleMesh
    {
        public static Mesh Create(float radius, float halfLength)
        {
            Tuple<Vector3[], int[]> data = MeshGenerators.BuildCapsule(radius, 2f * (halfLength + radius));
            var mesh = new Mesh { name = "WormSegmentCapsule" };
            mesh.vertices = data.Item1;
            mesh.triangles = data.Item2;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}
