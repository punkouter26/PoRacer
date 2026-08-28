using UnityEngine;

namespace IsaacH1
{
    /// <summary>
    /// The original Isaac visual meshes, one per link, already in Unity coordinates.
    ///
    /// Built by <c>IsaacH1 ▸ Import Original Meshes</c> from the `.ih1mesh` blobs that
    /// `extract_meshes.py` writes out of `robot/usd/instanceable_meshes.usd`. That USD is
    /// the ONLY place the real geometry exists - the vendor URDF points at
    /// `package://h1_description/meshes/*.STL` and those STL files ship nowhere in the
    /// export or in the Isaac Lab tree.
    ///
    /// The meshes are stored as sub-assets of this object, so the whole set is one file.
    /// If this library is absent or a link has no entry, <c>IsaacH1RigBuilder</c> falls
    /// back to the primitive proxies built from the URDF collision shapes.
    /// </summary>
    public class IsaacH1MeshLibrary : ScriptableObject
    {
        [Tooltip("Link names, parallel to meshes[].")]
        public string[] linkNames = System.Array.Empty<string>();

        public Mesh[] meshes = System.Array.Empty<Mesh>();

        [Tooltip("Material applied to every link renderer. URP Lit by default.")]
        public Material material;

        public int totalVertices;
        public int totalTriangles;

        public Mesh Find(string linkName)
        {
            if (linkNames == null || meshes == null) return null;
            int n = Mathf.Min(linkNames.Length, meshes.Length);
            for (int i = 0; i < n; i++)
                if (linkNames[i] == linkName) return meshes[i];
            return null;
        }
    }
}
