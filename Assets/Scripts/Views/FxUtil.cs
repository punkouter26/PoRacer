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
        private static ParticleSystem _impactSparks;
        private static ParticleSystem _wipeoutDebris;

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

        /// <summary>
        /// Sparks thrown off a hard limb landing, fired along the contact normal
        /// with a spread. Stretched billboards on the additive material, so a
        /// cluster reads as one hot flash rather than a handful of dots.
        ///
        /// <paramref name="strength"/> is 0-1 and scales both the count and the
        /// throw speed, so a scuff and a crash landing look different.
        /// </summary>
        public static void ImpactSparks(Vector3 position, Vector3 normal, float strength)
        {
            if (_impactSparks == null)
            {
                Material material = GlowParticleMaterial();
                if (material == null)
                {
                    return;
                }
                var go = new GameObject("ImpactSparks");
                var ps = go.AddComponent<ParticleSystem>();
                ParticleSystem.MainModule main = ps.main;
                main.playOnAwake = false;
                main.simulationSpace = ParticleSystemSimulationSpace.World;
                main.startLifetime = new ParticleSystem.MinMaxCurve(0.12f, 0.3f);
                main.startSize = new ParticleSystem.MinMaxCurve(0.03f, 0.08f);
                main.startColor = new ParticleSystem.MinMaxGradient(
                    new Color(1f, 0.85f, 0.45f, 1f), new Color(1f, 0.55f, 0.15f, 1f));
                // Sparks are light: they arc, they do not drop like rubble.
                main.gravityModifier = 0.45f;
                main.maxParticles = 400;
                ParticleSystem.EmissionModule emission = ps.emission;
                emission.rateOverTime = 0f;

                var sparkRenderer = ps.GetComponent<ParticleSystemRenderer>();
                sparkRenderer.material = material;
                // Stretching along velocity is what separates a spark from a dot.
                sparkRenderer.renderMode = ParticleSystemRenderMode.Stretch;
                sparkRenderer.velocityScale = 0.08f;
                sparkRenderer.lengthScale = 2.5f;
                sparkRenderer.shadowCastingMode = ShadowCastingMode.Off;
                sparkRenderer.receiveShadows = false;
                _impactSparks = ps;
            }

            float clamped = Mathf.Clamp01(strength);
            int count = 3 + Mathf.RoundToInt(clamped * 9f);
            float speed = Mathf.Lerp(1.4f, 4.5f, clamped);
            var emitParams = new ParticleSystem.EmitParams { applyShapeToPosition = false };
            for (int sparkIndex = 0; sparkIndex < count; sparkIndex++)
            {
                emitParams.position = position;
                // Cone around the contact normal, widened by the insideUnitSphere
                // term so the spray is not a perfectly symmetric fan.
                emitParams.velocity = (normal + Random.insideUnitSphere * 0.75f).normalized * speed
                    * Random.Range(0.6f, 1.2f);
                _impactSparks.Emit(emitParams, 1);
            }
        }

        /// <summary>
        /// Flung dirt and grit when a racer goes down, thrown outward and falling
        /// under full gravity so the eye reads weight. Sits under the existing
        /// knockout puff, which supplies the slow smoke on top of it.
        /// </summary>
        public static void WipeoutDebris(Vector3 position)
        {
            if (_wipeoutDebris == null)
            {
                Material material = SoftParticleMaterial();
                if (material == null)
                {
                    return;
                }
                var go = new GameObject("WipeoutDebris");
                var ps = go.AddComponent<ParticleSystem>();
                ParticleSystem.MainModule main = ps.main;
                main.playOnAwake = false;
                main.simulationSpace = ParticleSystemSimulationSpace.World;
                main.startLifetime = new ParticleSystem.MinMaxCurve(0.7f, 1.4f);
                main.startSpeed = new ParticleSystem.MinMaxCurve(2.5f, 6f);
                main.startSize = new ParticleSystem.MinMaxCurve(0.05f, 0.14f);
                main.startColor = new ParticleSystem.MinMaxGradient(
                    new Color(0.42f, 0.35f, 0.26f, 0.95f), new Color(0.25f, 0.21f, 0.16f, 0.9f));
                main.gravityModifier = 1f;
                main.maxParticles = 300;
                ParticleSystem.EmissionModule emission = ps.emission;
                emission.rateOverTime = 0f;

                ParticleSystem.ShapeModule shape = ps.shape;
                // A shallow upward cone: dirt comes off the ground, not out of a
                // sphere centred in the creature.
                shape.shapeType = ParticleSystemShapeType.Cone;
                shape.angle = 55f;
                shape.radius = 0.2f;
                shape.rotation = new Vector3(-90f, 0f, 0f);

                ParticleSystem.CollisionModule collision = ps.collision;
                collision.enabled = true;
                collision.type = ParticleSystemCollisionType.World;
                collision.mode = ParticleSystemCollisionMode.Collision3D;
                collision.bounce = 0.25f;
                collision.dampen = 0.5f;
                // Only against the ground layers; bouncing grit off articulation
                // bodies would put a physics query behind every particle.
                collision.quality = ParticleSystemCollisionQuality.Medium;

                var debrisRenderer = ps.GetComponent<ParticleSystemRenderer>();
                debrisRenderer.material = material;
                debrisRenderer.shadowCastingMode = ShadowCastingMode.Off;
                debrisRenderer.receiveShadows = false;
                _wipeoutDebris = ps;
            }
            _wipeoutDebris.transform.position = position;
            _wipeoutDebris.Emit(22);
        }
    }
}
