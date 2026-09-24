using System;
using System.Collections.Generic;
using Mujoco;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Render meshes of exactly the collision shapes, built once per session with the MuJoCo
    /// plug-in's own generators (the ones MjMeshFilter uses, so the winding is already right
    /// for Unity) and shared by every racer: one mesh per distinct shape, so all segments or
    /// legs of the same size batch together. A scaled primitive would squash capsule caps.
    ///
    /// Capsules and spheres run along local +Y (the plug-in's capsule axis); boxes take
    /// Unity-axis half-extents. The owner (a spawn system) disposes the cache.
    /// </summary>
    public sealed class CreatureMeshCache : IDisposable
    {
        private readonly Dictionary<Vector3, Mesh> _capsules = new();
        private readonly Dictionary<Vector3, Mesh> _boxes = new();
        private readonly Dictionary<Vector3, Mesh> _spheres = new();

        public Mesh Capsule(float radius, float halfLength)
        {
            var key = new Vector3(radius, halfLength, 0f);
            if (!_capsules.TryGetValue(key, out Mesh mesh) || mesh == null)
            {
                mesh = ToMesh("CreatureCapsule", MeshGenerators.BuildCapsule(radius, 2f * (halfLength + radius)));
                _capsules[key] = mesh;
            }
            return mesh;
        }

        public Mesh Box(Vector3 unityHalfExtents)
        {
            if (!_boxes.TryGetValue(unityHalfExtents, out Mesh mesh) || mesh == null)
            {
                mesh = ToMesh("CreatureBox", MeshGenerators.BuildBox(unityHalfExtents));
                _boxes[unityHalfExtents] = mesh;
            }
            return mesh;
        }

        public Mesh Sphere(float radius)
        {
            var key = new Vector3(radius, 0f, 0f);
            if (!_spheres.TryGetValue(key, out Mesh mesh) || mesh == null)
            {
                mesh = ToMesh("CreatureSphere", MeshGenerators.BuildSphere(Vector3.one * radius));
                _spheres[key] = mesh;
            }
            return mesh;
        }

        public void Dispose()
        {
            DestroyAll(_capsules);
            DestroyAll(_boxes);
            DestroyAll(_spheres);
        }

        private static Mesh ToMesh(string name, Tuple<Vector3[], int[]> data)
        {
            var mesh = new Mesh { name = name };
            mesh.vertices = data.Item1;
            mesh.triangles = data.Item2;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static void DestroyAll(Dictionary<Vector3, Mesh> meshes)
        {
            foreach (KeyValuePair<Vector3, Mesh> entry in meshes)
            {
                if (entry.Value != null)
                {
                    UnityEngine.Object.Destroy(entry.Value);
                }
            }
            meshes.Clear();
        }
    }
}
