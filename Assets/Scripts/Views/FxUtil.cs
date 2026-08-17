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
        private static readonly int BaseMapId = Shader.PropertyToID("_BaseMap");

        private static Texture2D _softCircle;
        private static Material _softParticleMaterial;
        private static Material _glowParticleMaterial;
        private static ParticleSystem _knockoutPuff;

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
            // Fallback if the assets are missing (run "PoRacer/Build FX Particle
            // Materials"): Sprites/Default is in Always Included Shaders, so it
            // never strips. Alpha-blended only — glow loses its additive pop but
            // nothing renders magenta.
            Shader spriteShader = Shader.Find("Sprites/Default");
            if (spriteShader == null)
            {
                return null;
            }
            return new Material(spriteShader) { mainTexture = SoftCircle() };
        }

        /// <summary>
        /// Gray smoke puff at a knocked-out racer's position, so eliminations read
        /// on screen instead of the creature silently vanishing. One shared
        /// world-space system serves every knockout.
        /// </summary>
        public static void KnockoutPuff(Vector3 position)
        {
            if (_knockoutPuff == null)
            {
                Material material = SoftParticleMaterial();
                if (material == null)
                {
                    return;
                }
                var go = new GameObject("KnockoutPuff");
                var ps = go.AddComponent<ParticleSystem>();
                ParticleSystem.MainModule main = ps.main;
                main.playOnAwake = false;
                main.simulationSpace = ParticleSystemSimulationSpace.World;
                main.startLifetime = new ParticleSystem.MinMaxCurve(0.6f, 1f);
                main.startSpeed = new ParticleSystem.MinMaxCurve(0.6f, 1.6f);
                main.startSize = new ParticleSystem.MinMaxCurve(0.2f, 0.5f);
                main.startColor = new ParticleSystem.MinMaxGradient(
                    new Color(0.55f, 0.55f, 0.55f, 0.5f), new Color(0.35f, 0.35f, 0.35f, 0.4f));
                main.gravityModifier = -0.05f;
                main.maxParticles = 200;
                ParticleSystem.EmissionModule emission = ps.emission;
                emission.rateOverTime = 0f;
                ParticleSystem.ShapeModule shape = ps.shape;
                shape.shapeType = ParticleSystemShapeType.Sphere;
                shape.radius = 0.25f;
                var puffRenderer = ps.GetComponent<ParticleSystemRenderer>();
                puffRenderer.material = material;
                puffRenderer.shadowCastingMode = ShadowCastingMode.Off;
                puffRenderer.receiveShadows = false;
                _knockoutPuff = ps;
            }
            _knockoutPuff.transform.position = position;
            _knockoutPuff.Emit(14);
        }
    }
}
