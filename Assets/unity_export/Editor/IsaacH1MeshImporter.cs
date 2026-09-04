using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace IsaacH1.EditorTools
{
    /// <summary>
    /// Turns the `.ih1mesh` blobs written by `extract_meshes.py` into a single
    /// <see cref="IsaacH1MeshLibrary"/> asset holding one Unity <c>Mesh</c> per link.
    ///
    /// The blobs are already in Unity coordinates, link-local, with winding reversed for
    /// the handedness flip - see `extract_meshes.py`. Nothing here transforms geometry;
    /// it only reads and packages.
    /// </summary>
    public static class IsaacH1MeshImporter
    {
        public const string MeshDir = IsaacH1Paths.Root + "/Meshes";
        public const string DecimatedMeshDir = MeshDir + "/decimated";
        public const string LibraryPath = MeshDir + "/IsaacH1Meshes.asset";
        public const string MaterialPath = MeshDir + "/M_IsaacH1.mat";

        const string Magic = "IH1M";
        const int Version = 1;

        public static void ImportMeshes()
        {
            var lib = Build();
            if (lib == null) return;
            Selection.activeObject = lib;
            EditorGUIUtility.PingObject(lib);
            Debug.Log($"[IsaacH1] imported {lib.meshes.Length} original Isaac meshes -> " +
                      $"{LibraryPath}\n" +
                      $"  {lib.totalVertices:N0} vertices, {lib.totalTriangles:N0} triangles\n" +
                      $"  Source: robot/usd/instanceable_meshes.usd (the URDF's *.STL files " +
                      $"are not present in the export). Re-run IsaacH1Setup.BuildPrefab() to put " +
                      $"them on the creature.");
        }

        public static void ImportDecimatedMeshes() => ReplaceInPlace(DecimatedMeshDir, "decimated");

        public static void RestoreFullDetailMeshes() => ReplaceInPlace(MeshDir, "full-detail");

        /// <summary>
        /// Swaps the geometry of the meshes ALREADY inside the library, keeping every asset
        /// GUID and sub-asset fileID. That matters: <see cref="Build"/> deletes and recreates
        /// the library, which mints new IDs and forces a `Build Prefab` to re-link - and that
        /// rebuild would drop anything added to the prefab afterwards, `Agent_IsaacH1`
        /// included. This path leaves the prefab, its components and the catalog entry alone.
        ///
        /// Reversible in both directions: `Meshes/*.ih1mesh` are the originals and are never
        /// written by the decimator, so "Restore Full-Detail Meshes" always undoes a swap.
        /// </summary>
        static void ReplaceInPlace(string meshDir, string label)
        {
            var lib = AssetDatabase.LoadAssetAtPath<IsaacH1MeshLibrary>(LibraryPath);
            if (lib == null)
            {
                Debug.LogError($"[IsaacH1] no mesh library at {LibraryPath}. " +
                               "Run IsaacH1MeshImporter.ImportMeshes() first.");
                return;
            }
            if (!Directory.Exists(meshDir))
            {
                Debug.LogError($"[IsaacH1] {meshDir} not found." +
                               (meshDir == DecimatedMeshDir
                                   ? "\n  Run:  python decimate_meshes.py"
                                   : "\n  Run:  python extract_meshes.py"));
                return;
            }

            int swapped = 0, totalV = 0, totalT = 0;
            for (int i = 0; i < lib.meshes.Length; i++)
            {
                Mesh target = lib.meshes[i];
                if (target == null) continue;
                string file = Path.Combine(meshDir, lib.linkNames[i] + ".ih1mesh");
                if (!File.Exists(file))
                {
                    Debug.LogWarning($"[IsaacH1] no {label} blob for '{lib.linkNames[i]}'; " +
                                     "leaving that link at its current detail.");
                    totalV += target.vertexCount;
                    totalT += target.triangles.Length / 3;
                    continue;
                }
                Mesh source = ReadMesh(file, lib.linkNames[i], out string err);
                if (source == null)
                {
                    Debug.LogError($"[IsaacH1] {Path.GetFileName(file)}: {err}");
                    continue;
                }

                target.Clear();
                target.indexFormat = source.indexFormat;
                target.vertices = source.vertices;
                target.normals = source.normals;
                target.triangles = source.triangles;
                target.RecalculateBounds();
                target.UploadMeshData(false);
                Object.DestroyImmediate(source);

                swapped++;
                totalV += target.vertexCount;
                totalT += target.triangles.Length / 3;
            }

            lib.totalVertices = totalV;
            lib.totalTriangles = totalT;
            EditorUtility.SetDirty(lib);
            AssetDatabase.SaveAssets();
            AssetDatabase.ImportAsset(LibraryPath, ImportAssetOptions.ForceUpdate);

            Debug.Log($"[IsaacH1] swapped {swapped} link meshes to {label} geometry in place.\n" +
                      $"  {totalV:N0} vertices, {totalT:N0} triangles per creature\n" +
                      "  Asset GUIDs and sub-asset IDs are unchanged, so the prefab needs no rebuild.");
        }

        public static IsaacH1MeshLibrary Build()
        {
            if (!Directory.Exists(MeshDir))
            {
                Debug.LogError($"[IsaacH1] {MeshDir} not found. Run:\n" +
                               "  python extract_meshes.py");
                return null;
            }

            var files = Directory.GetFiles(MeshDir, "*.ih1mesh");
            if (files.Length == 0)
            {
                Debug.LogError($"[IsaacH1] no .ih1mesh files in {MeshDir}. Run extract_meshes.py.");
                return null;
            }
            System.Array.Sort(files);

            // Rebuild from scratch: sub-assets cannot be reliably replaced in place.
            AssetDatabase.DeleteAsset(LibraryPath);
            var lib = ScriptableObject.CreateInstance<IsaacH1MeshLibrary>();
            AssetDatabase.CreateAsset(lib, LibraryPath);

            var names = new List<string>(files.Length);
            var meshes = new List<Mesh>(files.Length);
            int totalV = 0, totalT = 0;

            foreach (var file in files)
            {
                string link = Path.GetFileNameWithoutExtension(file);
                Mesh m = ReadMesh(file, link, out string err);
                if (m == null)
                {
                    Debug.LogError($"[IsaacH1] {Path.GetFileName(file)}: {err}");
                    continue;
                }
                AssetDatabase.AddObjectToAsset(m, lib);
                names.Add(link);
                meshes.Add(m);
                totalV += m.vertexCount;
                totalT += m.triangles.Length / 3;
            }

            lib.linkNames = names.ToArray();
            lib.meshes = meshes.ToArray();
            lib.totalVertices = totalV;
            lib.totalTriangles = totalT;
            lib.material = CreateOrLoadMaterial();

            EditorUtility.SetDirty(lib);
            AssetDatabase.SaveAssets();
            AssetDatabase.ImportAsset(LibraryPath, ImportAssetOptions.ForceUpdate);
            return AssetDatabase.LoadAssetAtPath<IsaacH1MeshLibrary>(LibraryPath);
        }

        static Mesh ReadMesh(string path, string name, out string error)
        {
            error = null;
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            string magic = new string(br.ReadChars(4));
            if (magic != Magic) { error = $"bad magic '{magic}'"; return null; }
            int version = br.ReadInt32();
            if (version != Version) { error = $"unsupported version {version}"; return null; }

            int nv = br.ReadInt32();
            int nt = br.ReadInt32();
            if (nv <= 0 || nt <= 0) { error = $"empty mesh ({nv} verts, {nt} tris)"; return null; }

            long expected = 16 + (long)nv * 12 * 2 + (long)nt * 12;
            if (fs.Length != expected)
            {
                error = $"size mismatch: {fs.Length} bytes, expected {expected}";
                return null;
            }

            var verts = new Vector3[nv];
            for (int i = 0; i < nv; i++)
                verts[i] = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

            var normals = new Vector3[nv];
            for (int i = 0; i < nv; i++)
                normals[i] = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

            var tris = new int[nt * 3];
            for (int i = 0; i < tris.Length; i++) tris[i] = br.ReadInt32();

            var m = new Mesh { name = name };
            // Every H1 link is well over the 16-bit limit once merged.
            m.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            m.vertices = verts;
            m.normals = normals;
            m.triangles = tris;
            m.RecalculateBounds();
            m.UploadMeshData(false);   // keep readable: the tests and gizmos want it
            return m;
        }

        /// <summary>A plain lit material. URP if the project uses it, else the built-in default.</summary>
        static Material CreateOrLoadMaterial()
        {
            var existing = AssetDatabase.LoadAssetAtPath<Material>(MaterialPath);
            if (existing != null) return existing;

            Shader sh = Shader.Find("Universal Render Pipeline/Lit")
                        ?? Shader.Find("Standard")
                        ?? Shader.Find("Diffuse");
            if (sh == null)
            {
                Debug.LogWarning("[IsaacH1] no usable shader found; meshes will render pink.");
                return null;
            }

            var mat = new Material(sh) { name = "M_IsaacH1" };
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", new Color(0.72f, 0.74f, 0.78f));
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", new Color(0.72f, 0.74f, 0.78f));
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0.35f);
            if (mat.HasProperty("_Metallic")) mat.SetFloat("_Metallic", 0.1f);

            AssetDatabase.CreateAsset(mat, MaterialPath);
            AssetDatabase.SaveAssets();
            return AssetDatabase.LoadAssetAtPath<Material>(MaterialPath);
        }
    }
}
