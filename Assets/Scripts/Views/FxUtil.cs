using UnityEngine;
using UnityEngine.Rendering;

namespace PoRacer.Views
{
    /// <summary>
    /// Shared code-generated FX assets: one soft-circle sprite texture and one
    /// transparent particle material, reused by every particle system so they
    /// all share a single material state.
    /// </summary>
    internal static class FxUtil
    {
        private static Texture2D _softCircle;
        private static Material _softParticleMaterial;

        /// <summary>Radial-falloff white circle used as the universal particle sprite.</summary>
        public static Texture2D SoftCircle()
        {
            if (_softCircle != null)
            {
                return _softCircle;
            }
            const int size = 64;
            var texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float dx = (x + 0.5f) / size - 0.5f;
                    float dy = (y + 0.5f) / size - 0.5f;
                    float alpha = Mathf.Clamp01(1f - Mathf.Sqrt(dx * dx + dy * dy) * 2f);
                    texture.SetPixel(x, y, new Color(1f, 1f, 1f, alpha * alpha));
                }
            }
            texture.Apply();
            _softCircle = texture;
            return texture;
        }

        /// <summary>
        /// Shared alpha-blended unlit particle material. Null in headless builds
        /// where shaders are stripped.
        /// </summary>
        public static Material SoftParticleMaterial()
        {
            if (_softParticleMaterial != null)
            {
                return _softParticleMaterial;
            }
            Shader particleShader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (particleShader == null)
            {
                return null;
            }
            var material = new Material(particleShader);
            material.SetFloat("_Surface", 1f);
            material.SetFloat("_Blend", 0f);
            material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha);
            material.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha);
            material.SetInt("_ZWrite", 0);
            material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            material.renderQueue = (int)RenderQueue.Transparent;
            material.SetTexture("_BaseMap", SoftCircle());
            _softParticleMaterial = material;
            return material;
        }
    }
}
