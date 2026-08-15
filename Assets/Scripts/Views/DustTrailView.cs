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

        private ParticleSystem.EmissionModule _emission;
        private Transform _transform;
        private Vector3 _lastPosition;

        private void Awake()
        {
            _transform = transform;
            _lastPosition = _transform.position;
            ParticleSystem particles = BuildParticles();
            _emission = particles.emission;
        }

        private void Update()
        {
            Vector3 position = _transform.position;
            float speed = (position - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.0001f);
            _lastPosition = position;
            _emission.rateOverTime = Mathf.Clamp01(speed / FULL_RATE_SPEED) * MAX_RATE;
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
    }
}
