using UnityEngine;
using UnityEngine.Rendering;

namespace PoRacer.Presentation
{
    /// <summary>
    /// Shared code-generated FX assets: one soft-circle sprite texture and one
    /// transparent particle material, reused by every particle system so they
    /// all share a single material state.
    /// </summary>
    internal static class FxUtil
    {
        private static readonly int BaseMapId = Shader.PropertyToID("_BaseMap");
        private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");

        private static Texture2D _softCircle;
        private static Material _softParticleMaterial;
        private static Material _glowParticleMaterial;
        private static Material _contactShadowMaterial;
        private static Mesh _flatQuad;

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
        /// Shared alpha-blended unlit particle material. Loaded from a material
        /// ASSET (Assets/Resources/FX) so the URP particle shader survives build
        /// stripping — a runtime Shader.Find material renders magenta on device.
        /// Null in headless builds where rendering is absent.
        /// </summary>
        public static Material SoftParticleMaterial()
        {
            if (_softParticleMaterial == null)
            {
                _softParticleMaterial = LoadParticleMaterial("FX/M_ParticleSoft");
            }
            return _softParticleMaterial;
        }

        /// <summary>
        /// Shared additive particle material: overlapping particles sum to a hot
        /// glow. For sparks, fireworks, boost FX. Same asset-loading rule as
        /// SoftParticleMaterial.
        /// </summary>
        public static Material GlowParticleMaterial()
        {
            if (_glowParticleMaterial == null)
            {
                _glowParticleMaterial = LoadParticleMaterial("FX/M_ParticleGlow");
            }
            return _glowParticleMaterial;
        }

        /// <summary>
        /// Shared soft black blob for the contact shadows under racers. One material
        /// for the whole field, so every blob batches together.
        /// </summary>
        public static Material ContactShadowMaterial()
        {
            if (_contactShadowMaterial == null)
            {
                Material soft = LoadParticleMaterial("FX/M_ParticleSoft");
                if (soft == null)
                {
                    return null;
                }
                soft.name = "ContactShadow";
                soft.SetColor(BaseColorId, new Color(0f, 0f, 0f, 0.55f));
                _contactShadowMaterial = soft;
            }
            return _contactShadowMaterial;
        }

        /// <summary>Unit quad lying in the XZ plane, facing up, with white vertex colour.</summary>
        public static Mesh FlatQuad()
        {
            if (_flatQuad != null)
            {
                return _flatQuad;
            }
            var mesh = new Mesh { name = "FlatQuad" };
            mesh.SetVertices(new[]
            {
                new Vector3(-0.5f, 0f, -0.5f), new Vector3(0.5f, 0f, -0.5f),
                new Vector3(-0.5f, 0f, 0.5f), new Vector3(0.5f, 0f, 0.5f)
            });
            mesh.SetUVs(0, new[] { new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0f, 1f), new Vector2(1f, 1f) });
            // The particle shader multiplies by vertex colour; without it the blob
            // takes whatever the platform defaults a missing channel to.
            mesh.SetColors(new[] { Color.white, Color.white, Color.white, Color.white });
            mesh.SetNormals(new[] { Vector3.up, Vector3.up, Vector3.up, Vector3.up });
            // Clockwise seen from above: the front face points up.
            mesh.SetTriangles(new[] { 0, 2, 1, 2, 3, 1 }, 0);
            mesh.RecalculateBounds();
            _flatQuad = mesh;
            return mesh;
        }

        private static Material LoadParticleMaterial(string resourcePath)
        {
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
            {
                return null;
            }
            Material asset = Resources.Load<Material>(resourcePath);
            if (asset != null)
            {
                // Copy so the runtime-generated sprite never dirties the asset in
                // the editor; the copy keeps the asset's shader variant.
                var material = new Material(asset);
                material.SetTexture(BaseMapId, SoftCircle());
                return material;
            }
            // Fallback if the assets are missing: Sprites/Default is in Always Included Shaders, so it
            // never strips. Alpha-blended only — glow loses its additive pop but
            // nothing renders magenta.
            Shader spriteShader = Shader.Find("Sprites/Default");
            if (spriteShader == null)
            {
                return null;
            }
            return new Material(spriteShader) { mainTexture = SoftCircle() };
        }
    }
}
