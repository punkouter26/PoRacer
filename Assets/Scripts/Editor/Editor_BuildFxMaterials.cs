#if UNITY_EDITOR
using PoRacer.Presentation;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoRacer.Editor
{
    /// <summary>
    /// Creates the two shared particle material assets under Assets/Resources/FX.
    /// They exist as ASSETS (not runtime Shader.Find materials) so the URP
    /// particle shader and its transparent variants ship in player builds —
    /// runtime-only materials get their shader stripped and render magenta on
    /// device. FxUtil loads these via Resources at runtime.
    /// </summary>
    public static class Editor_BuildFxMaterials
    {
        private const string FOLDER = "Assets/Resources/FX";

        [MenuItem("PoRacer/Build FX Particle Materials")]
        public static void Build()
        {
            if (!AssetDatabase.IsValidFolder("Assets/Resources"))
            {
                AssetDatabase.CreateFolder("Assets", "Resources");
            }
            if (!AssetDatabase.IsValidFolder(FOLDER))
            {
                AssetDatabase.CreateFolder("Assets/Resources", "FX");
            }
            BuildMaterial($"{FOLDER}/M_ParticleSoft.mat", additive: false);
            BuildMaterial($"{FOLDER}/M_ParticleGlow.mat", additive: true);
            AssetDatabase.SaveAssets();
            Debug.Log($"FX particle materials written to {FOLDER}.");
        }

        private static void BuildMaterial(string path, bool additive)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (shader == null)
            {
                Debug.LogError("URP Particles/Unlit shader not found; is URP installed?");
                return;
            }
            Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
            bool isNew = material == null;
            if (isNew)
            {
                material = new Material(shader);
            }
            else
            {
                material.shader = shader;
            }
            // Same setup FxUtil used to do at runtime: unlit transparent, no
            // depth write; additive variant sums overlapping particles to a glow.
            // _Blend uses URP's enum: 0 = Alpha, 2 = Additive (editor-side
            // material validation recomputes Src/Dst from it on save).
            material.SetFloat("_Surface", 1f);
            material.SetFloat("_Blend", additive ? 2f : 0f);
            material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", additive ? (int)BlendMode.One : (int)BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            material.renderQueue = (int)RenderQueue.Transparent;
            if (isNew)
            {
                AssetDatabase.CreateAsset(material, path);
            }
            else
            {
                EditorUtility.SetDirty(material);
            }
        }
    }
}
#endif
