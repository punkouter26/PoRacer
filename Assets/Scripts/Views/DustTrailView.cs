using PoRacer.Presentation;
using UnityEngine;
using UnityEngine.Rendering;

namespace PoRacer.Views
{
    /// <summary>
    /// Speed-driven dust puffs at a racer's feet. The particle system, its soft
    /// round texture, and its material are generated in code; the material is
    /// shared between all racers so every dust system batches together.
    /// </summary>
    public sealed class DustTrailView : MonoBehaviour
    {
        private const float MAX_RATE = 22f;
        private const float FULL_RATE_SPEED = 2.5f;

        private ParticleSystem _particles;
        private ParticleSystem _sparks;
        private Transform _transform;
        private Vector3 _lastPosition;

        private void Awake()
        {
            _transform = transform;
            _lastPosition = _transform.position;
            _particles = BuildParticles();
            _sparks = BuildSparks();
        }

        private void Update()
        {
            if (_particles == null)
            {
                enabled = false;
                return;
            }
            Vector3 position = _transform.position;
            float speed = (position - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.0001f);
            _lastPosition = position;

            ParticleSystem.EmissionModule emission = _particles.emission;
            emission.rateOverTime = Mathf.Clamp01(speed / FULL_RATE_SPEED) * MAX_RATE;

            if (_sparks != null)
            {
                ParticleSystem.EmissionModule sparkEmission = _sparks.emission;
                sparkEmission.rateOverTime = speed > 2.0f ? (speed - 2.0f) * 18f : 0f;
            }
        }

        private ParticleSystem BuildParticles()
        {
            var go = new GameObject("DustTrail");
            go.transform.SetParent(_transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.6f, 1.1f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(0.2f, 0.6f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.12f, 0.35f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(0.62f, 0.55f, 0.45f, 0.35f),
                new Color(0.72f, 0.66f, 0.55f, 0.25f));
            main.gravityModifier = -0.02f;
            main.maxParticles = 60;

            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 0f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Hemisphere;
            shape.radius = 0.25f;

            ParticleSystem.ColorOverLifetimeModule colorOverLifetime = ps.colorOverLifetime;
            colorOverLifetime.enabled = true;
            var gradient = new Gradient();
            gradient.SetKeys(
                new[]
                {
                    new GradientColorKey(Color.white, 0f),
                    new GradientColorKey(Color.white, 1f)
                },
                new[]
                {
                    new GradientAlphaKey(0f, 0f),
                    new GradientAlphaKey(1f, 0.15f),
                    new GradientAlphaKey(0f, 1f)
                });
            colorOverLifetime.color = gradient;

            ParticleSystem.SizeOverLifetimeModule sizeOverLifetime = ps.sizeOverLifetime;
            sizeOverLifetime.enabled = true;
            sizeOverLifetime.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 0.6f, 1f, 1.6f));

            var particleRenderer = ps.GetComponent<ParticleSystemRenderer>();
            Material sharedMaterial = FxUtil.SoftParticleMaterial();
            if (sharedMaterial != null)
            {
                particleRenderer.material = sharedMaterial;
            }
            particleRenderer.shadowCastingMode = ShadowCastingMode.Off;
            particleRenderer.receiveShadows = false;
            return ps;
        }

        private ParticleSystem BuildSparks()
        {
            var go = new GameObject("SpeedSparks");
            go.transform.SetParent(_transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.2f, 0.45f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(1.5f, 3.5f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.06f, 0.15f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(1f, 0.8f, 0.2f, 0.9f),
                new Color(1f, 0.35f, 0.1f, 0.8f));
            main.gravityModifier = 0.5f;
            main.maxParticles = 50;

            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 0f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 25f;
            shape.rotation = new Vector3(-180f, 0f, 0f);

            var particleRenderer = ps.GetComponent<ParticleSystemRenderer>();
            Material glowMat = FxUtil.GlowParticleMaterial();
            if (glowMat != null)
            {
                particleRenderer.material = glowMat;
            }
            particleRenderer.shadowCastingMode = ShadowCastingMode.Off;
            particleRenderer.receiveShadows = false;
            return ps;
        }
    }
}
