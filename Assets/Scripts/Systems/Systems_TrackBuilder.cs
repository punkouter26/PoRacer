using System;
using UnityEngine;

namespace PoRacer.Systems
{
    public enum TrackKind
    {
        Flat = 0,
        Hills = 1,
        Bumps = 2,
        Walls = 3
    }

    /// <summary>
    /// Procedural track construction shared by the race scene and training areas.
    /// Builds ground (flat plane or a sine-hill mesh) plus scattered obstacles
    /// under a single parent that the caller owns and can clear/rebuild.
    /// </summary>
    public sealed class Systems_TrackBuilder
    {
        private const float HILL_AMPLITUDE = 0.35f;
        private const float HILL_WAVELENGTH = 6f;
        private const int MESH_STEPS_PER_METER = 2;

        private readonly Material _groundMaterial;
        private readonly Material _obstacleMaterial;
        private readonly PhysicsMaterial _physicsMaterial;

        public Systems_TrackBuilder(Material groundMaterial, Material obstacleMaterial, PhysicsMaterial physicsMaterial)
        {
            _groundMaterial = groundMaterial;
            _obstacleMaterial = obstacleMaterial;
            _physicsMaterial = physicsMaterial;
        }

        public static TrackKind Roll(System.Random rng) => (TrackKind)rng.Next(0, 4);

        /// <summary>Clears previous children and builds the track under 'parent'. Origin is the start line center.</summary>
        public void Build(TrackKind kind, Transform parent, float width, float length, System.Random rng)
        {
            for (int childIndex = parent.childCount - 1; childIndex >= 0; childIndex--)
            {
                UnityEngine.Object.Destroy(parent.GetChild(childIndex).gameObject);
            }

            if (kind == TrackKind.Hills)
            {
                BuildHillMesh(parent, width, length);
            }
            else
            {
                BuildFlatGround(parent, width, length);
            }

            if (kind == TrackKind.Bumps)
            {
                ScatterBoxes(parent, width, length, rng, count: 24,
                    minSize: new Vector3(0.4f, 0.08f, 0.4f), maxSize: new Vector3(1.2f, 0.22f, 1.2f));
            }
            else if (kind == TrackKind.Walls)
            {
                ScatterBoxes(parent, width, length, rng, count: 10,
                    minSize: new Vector3(2f, 0.25f, 0.25f), maxSize: new Vector3(5f, 0.4f, 0.35f));
            }
        }

        /// <summary>Surface height at a local (x, z) for the given kind — used to place goals/spawns.</summary>
        public static float SurfaceHeight(TrackKind kind, float z)
        {
            return kind == TrackKind.Hills
                ? (Mathf.Sin(z / HILL_WAVELENGTH * 2f * Mathf.PI) * 0.5f + 0.5f) * HILL_AMPLITUDE
                : 0f;
        }

        private void BuildFlatGround(Transform parent, float width, float length)
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(parent, false);
            // A plane primitive is 10x10 at scale 1; margin behind the start line.
            ground.transform.localPosition = new Vector3(0f, 0f, length * 0.5f - 3f);
            ground.transform.localScale = new Vector3(width / 10f, 1f, (length + 8f) / 10f);
            ground.GetComponent<MeshRenderer>().sharedMaterial = _groundMaterial;
            ground.GetComponent<MeshCollider>().sharedMaterial = _physicsMaterial;
        }

        private void BuildHillMesh(Transform parent, float width, float length)
        {
            float fullLength = length + 8f;
            float zStart = -5f;
            int stepsZ = Mathf.CeilToInt(fullLength * MESH_STEPS_PER_METER);
            int stepsX = Mathf.CeilToInt(width * MESH_STEPS_PER_METER / 4f);
            var vertices = new Vector3[(stepsX + 1) * (stepsZ + 1)];
            var triangles = new int[stepsX * stepsZ * 6];

            for (int zIndex = 0; zIndex <= stepsZ; zIndex++)
            {
                float z = zStart + fullLength * zIndex / stepsZ;
                float height = SurfaceHeight(TrackKind.Hills, z);
                for (int xIndex = 0; xIndex <= stepsX; xIndex++)
                {
                    float x = -width * 0.5f + width * xIndex / stepsX;
                    vertices[zIndex * (stepsX + 1) + xIndex] = new Vector3(x, height, z);
                }
            }
            int triangleIndex = 0;
            for (int zIndex = 0; zIndex < stepsZ; zIndex++)
            {
                for (int xIndex = 0; xIndex < stepsX; xIndex++)
                {
                    int corner = zIndex * (stepsX + 1) + xIndex;
                    triangles[triangleIndex++] = corner;
                    triangles[triangleIndex++] = corner + stepsX + 1;
                    triangles[triangleIndex++] = corner + 1;
                    triangles[triangleIndex++] = corner + 1;
                    triangles[triangleIndex++] = corner + stepsX + 1;
                    triangles[triangleIndex++] = corner + stepsX + 2;
                }
            }
            var mesh = new Mesh { vertices = vertices, triangles = triangles };
            mesh.RecalculateNormals();

            var ground = new GameObject("Ground");
            ground.transform.SetParent(parent, false);
            ground.AddComponent<MeshFilter>().sharedMesh = mesh;
            ground.AddComponent<MeshRenderer>().sharedMaterial = _groundMaterial;
            var collider = ground.AddComponent<MeshCollider>();
            collider.sharedMesh = mesh;
            collider.sharedMaterial = _physicsMaterial;
        }

        private void ScatterBoxes(Transform parent, float width, float length, System.Random rng,
            int count, Vector3 minSize, Vector3 maxSize)
        {
            for (int boxIndex = 0; boxIndex < count; boxIndex++)
            {
                var box = GameObject.CreatePrimitive(PrimitiveType.Cube);
                box.name = "Obstacle";
                box.transform.SetParent(parent, false);
                var size = new Vector3(
                    Mathf.Lerp(minSize.x, maxSize.x, (float)rng.NextDouble()),
                    Mathf.Lerp(minSize.y, maxSize.y, (float)rng.NextDouble()),
                    Mathf.Lerp(minSize.z, maxSize.z, (float)rng.NextDouble()));
                box.transform.localScale = size;
                // Keep the first 3 m after the start line clear so racers are not born stuck.
                float z = 3f + (float)rng.NextDouble() * (length - 4f);
                float x = ((float)rng.NextDouble() - 0.5f) * (width - size.x);
                box.transform.localPosition = new Vector3(x, size.y * 0.5f, z);
                box.transform.localRotation = Quaternion.Euler(0f, (float)rng.NextDouble() * 40f - 20f, 0f);
                box.GetComponent<MeshRenderer>().sharedMaterial = _obstacleMaterial;
                box.GetComponent<BoxCollider>().sharedMaterial = _physicsMaterial;
            }
        }
    }
}
