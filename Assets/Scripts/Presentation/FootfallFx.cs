using UnityEngine;
using UnityEngine.Rendering;

namespace PoRacer.Presentation
{
    /// <summary>
    /// Footprints and landing puffs, placed where a limb actually met the ground.
    ///
    /// The prints are the point of this: they are laid by real contacts, so they
    /// show the gait the physics produced. A clean trot leaves an evenly spaced
    /// double line; a creature that skates leaves a smear of overlapping prints; a
    /// stumble leaves a cluster. That is a readout of foot skating nobody has to
    /// open a graph for.
    ///
    /// Two shared world-space particle systems serve every racer, on the shared
    /// soft particle material, so the whole field's footfalls cost two draws.
    /// Prints lie flat (horizontal billboards) and fade over a few seconds.
    /// </summary>
    internal static class FootfallFx
    {
        private const int MAX_PRINTS = 700;
        private const float PRINT_SECONDS_MIN = 4.5f;
        private const float PRINT_SECONDS_MAX = 6.5f;
        private const float PRINT_SIZE_MIN = 0.12f;
        private const float PRINT_SIZE_MAX = 0.3f;
        // Clear of the ground's own depth, or distant prints z-fight it.
        private const float PRINT_LIFT = 0.03f;
        private const int PUFF_MIN = 2;
        private const int PUFF_MAX = 7;
        // A gentle step marks the ground but does not raise dust.
        private const float PUFF_MIN_STRENGTH = 0.25f;

        private static ParticleSystem _prints;
        private static ParticleSystem _puffs;

        /// <summary>
        /// A limb landed at <paramref name="point"/>. <paramref name="strength"/> is
        /// 0-1 and sizes both the print and the puff.
        /// </summary>
        public static void Footfall(Vector3 point, Vector3 normal, float strength)
        {
            float clamped = Mathf.Clamp01(strength);
            ParticleSystem prints = Prints();
            if (prints == null)
            {
                return;
            }
            Color tone = FxBudget.GroundTone;
            var printParams = new ParticleSystem.EmitParams
            {
                applyShapeToPosition = false,
                position = point + normal * PRINT_LIFT,
                velocity = Vector3.zero,
                startSize = Mathf.Lerp(PRINT_SIZE_MIN, PRINT_SIZE_MAX, clamped),
                startLifetime = Random.Range(PRINT_SECONDS_MIN, PRINT_SECONDS_MAX),
                // Darker than the dust: a print is compressed, shaded ground.
                startColor = new Color(tone.r * 0.45f, tone.g * 0.42f, tone.b * 0.4f, 0.55f)
            };
            prints.Emit(printParams, 1);

            if (clamped < PUFF_MIN_STRENGTH)
            {
                return;
            }
            ParticleSystem puffs = Puffs();
            if (puffs == null)
            {
                return;
            }
            int count = FxBudget.Scale(Mathf.RoundToInt(Mathf.Lerp(PUFF_MIN, PUFF_MAX, clamped)));
            var puffParams = new ParticleSystem.EmitParams
            {
                applyShapeToPosition = false,
                startColor = new Color(tone.r, tone.g, tone.b, Mathf.Lerp(0.25f, 0.45f, clamped))
            };
            for (int puffIndex = 0; puffIndex < count; puffIndex++)
            {
                // Dust squirts out sideways from under the foot, not straight up.
                Vector2 spray = Random.insideUnitCircle.normalized * Random.Range(0.3f, 1.1f);
                puffParams.position = point + normal * 0.05f;
                puffParams.velocity = new Vector3(spray.x, Random.Range(0.1f, 0.45f), spray.y)
                    * Mathf.Lerp(0.6f, 1.4f, clamped);
                puffParams.startSize = Random.Range(0.1f, 0.22f) * Mathf.Lerp(0.8f, 1.5f, clamped);
                puffs.Emit(puffParams, 1);
            }
        }

        private static ParticleSystem Prints()
        {
            if (_prints != null)
            {
                return _prints;
            }
            Material material = FxUtil.SoftParticleMaterial();
            if (material == null)
            {
                return null;
            }
            ParticleSystem ps = NewSystem("Footprints", material, MAX_PRINTS);
            ParticleSystem.MainModule main = ps.main;
            main.gravityModifier = 0f;
            // Fade in over the first instant (no pop), hold, then fade out.
            ParticleSystem.ColorOverLifetimeModule fade = ps.colorOverLifetime;
            fade.enabled = true;
            fade.color = FadeGradient(0.03f, 0.55f);
            var printRenderer = ps.GetComponent<ParticleSystemRenderer>();
            printRenderer.renderMode = ParticleSystemRenderMode.HorizontalBillboard;
            // Oldest first, so a fresh print never sorts under a fading one.
            printRenderer.sortMode = ParticleSystemSortMode.OldestInFront;
            _prints = ps;
            return ps;
        }

        private static ParticleSystem Puffs()
        {
            if (_puffs != null)
            {
                return _puffs;
            }
            Material material = FxUtil.SoftParticleMaterial();
            if (material == null)
            {
                return null;
            }
            ParticleSystem ps = NewSystem("FootfallPuffs", material, 300);
            ParticleSystem.MainModule main = ps.main;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.45f, 0.9f);
            // Dust hangs: barely any gravity, and drag bleeds the sideways squirt.
            main.gravityModifier = 0.05f;
            ParticleSystem.LimitVelocityOverLifetimeModule drag = ps.limitVelocityOverLifetime;
            drag.enabled = true;
            drag.drag = 2.5f;
            ParticleSystem.SizeOverLifetimeModule grow = ps.sizeOverLifetime;
            grow.enabled = true;
            grow.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 0.7f, 1f, 1.8f));
            ParticleSystem.ColorOverLifetimeModule fade = ps.colorOverLifetime;
            fade.enabled = true;
            fade.color = FadeGradient(0.1f, 0.3f);
            _puffs = ps;
            return ps;
        }

        private static ParticleSystem NewSystem(string name, Material material, int maxParticles)
        {
            var go = new GameObject(name);
            var ps = go.AddComponent<ParticleSystem>();
            // AddComponent starts it playing with default emission; stop that first.
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = maxParticles;
            main.startSpeed = 0f;
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.enabled = false;
            ParticleSystem.ShapeModule shape = ps.shape;
            shape.enabled = false;
            var particleRenderer = ps.GetComponent<ParticleSystemRenderer>();
            particleRenderer.sharedMaterial = material;
            particleRenderer.shadowCastingMode = ShadowCastingMode.Off;
            particleRenderer.receiveShadows = false;
            ps.Play();
            return ps;
        }

        private static Gradient FadeGradient(float fadeInEnd, float fadeOutStart)
        {
            var gradient = new Gradient();
            gradient.SetKeys(
                new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                new[]
                {
                    new GradientAlphaKey(0f, 0f),
                    new GradientAlphaKey(1f, fadeInEnd),
                    new GradientAlphaKey(1f, fadeOutStart),
                    new GradientAlphaKey(0f, 1f)
                });
            return gradient;
        }
    }
}
