using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Hazard trigger stripe built by Systems_TrackBuilder: cross-wind gusts
    /// shove every ArticulationBody inside sideways on a slow cycle. Bodies are
    /// collected in OnTriggerStay and forced once per FixedUpdate so
    /// multi-collider limbs are never double-pushed. Streaking particles and a
    /// wind loop rise and fall with the gust. All assets are generated in code.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class GustZoneView : MonoBehaviour
    {
        // Sideways acceleration at full gust; enough to bend a line, not flip a racer.
        private const float GUST_ACCEL = 3.5f;
        private const float GUST_CYCLE_SECONDS = 6f;
        private const float MAX_WIND_VOLUME = 0.3f;
        private const float STREAK_RATE = 50f;

        private static AudioClip SharedWind;

        private readonly List<ArticulationBody> _bodiesInZone = new();
        private float _direction = 1f;
        private float _phaseOffset;
        private ParticleSystem _streaks;
        private AudioSource _windSource;
        private float _gust;

        /// <summary>Called by the track builder right after AddComponent.</summary>
        public void Initialize(float direction, float phaseOffset, float zoneWidth)
        {
            _direction = direction;
            _phaseOffset = phaseOffset;
            ParticleSystem.VelocityOverLifetimeModule velocity = _streaks.velocityOverLifetime;
            velocity.enabled = true;
            // All three axes must share one curve mode (two-constants here),
            // or Unity rejects the whole module.
            velocity.x = new ParticleSystem.MinMaxCurve(4f * direction, 7f * direction);
            velocity.y = new ParticleSystem.MinMaxCurve(0f, 0f);
            velocity.z = new ParticleSystem.MinMaxCurve(0f, 0f);
            ParticleSystem.ShapeModule shape = _streaks.shape;
            shape.scale = new Vector3(zoneWidth, 1.2f, 3f);
        }

        private void Awake()
        {
            _streaks = BuildStreaks();
            _windSource = gameObject.AddComponent<AudioSource>();
            _windSource.clip = GetSharedWind();
            _windSource.loop = true;
            _windSource.playOnAwake = false;
            _windSource.spatialBlend = 1f;
            _windSource.dopplerLevel = 0f;
            _windSource.minDistance = 4f;
            _windSource.maxDistance = 40f;
            _windSource.volume = 0f;
            _windSource.Play();
        }

        private void OnTriggerStay(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body != null && !_bodiesInZone.Contains(body))
            {
                _bodiesInZone.Add(body);
            }
        }

        private void FixedUpdate()
        {
            float gust = GustFactor(Time.time);
            if (gust > 0f)
            {
                for (int bodyIndex = 0; bodyIndex < _bodiesInZone.Count; bodyIndex++)
                {
                    ArticulationBody body = _bodiesInZone[bodyIndex];
                    if (body != null)
                    {
                        body.AddForce(Vector3.right * (_direction * GUST_ACCEL * gust * body.mass));
                    }
                }
            }
            _bodiesInZone.Clear();
        }

        private void Update()
        {
            _gust = GustFactor(Time.time);
            ParticleSystem.EmissionModule emission = _streaks.emission;
            emission.rateOverTime = _gust * STREAK_RATE;
            _windSource.volume = Mathf.MoveTowards(_windSource.volume, _gust * MAX_WIND_VOLUME, Time.deltaTime * 0.8f);
        }

        /// <summary>0 in the lull, 1 at full gust; clipped sine so gusts come in waves.</summary>
        private float GustFactor(float time)
        {
            return Mathf.Clamp01(Mathf.Sin(2f * Mathf.PI * (time + _phaseOffset) / GUST_CYCLE_SECONDS) * 1.5f - 0.35f);
        }

        private ParticleSystem BuildStreaks()
        {
            var go = new GameObject("WindStreaks");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = true;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.5f, 0.9f);
            main.startSpeed = 0f;
            main.startSize = new ParticleSystem.MinMaxCurve(0.03f, 0.08f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(0.9f, 0.93f, 1f, 0.5f),
                new Color(0.8f, 0.85f, 0.95f, 0.3f));
            main.gravityModifier = 0f;
            main.maxParticles = 300;

            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 0f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = new Vector3(10f, 1.2f, 3f);

            var streakRenderer = ps.GetComponent<ParticleSystemRenderer>();
            // Soft round sprite; stretched along velocity so gusts read as streaks.
            Material soft = FxUtil.SoftParticleMaterial();
            if (soft != null)
            {
                streakRenderer.material = soft;
            }
            streakRenderer.renderMode = ParticleSystemRenderMode.Stretch;
            streakRenderer.velocityScale = 0.12f;
            streakRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            streakRenderer.receiveShadows = false;
            return ps;
        }

        private static AudioClip GetSharedWind()
        {
            if (SharedWind != null)
            {
                return SharedWind;
            }
            // Steady filtered-noise wind bed; the source volume carries the gusting.
            const int sampleRate = 44100;
            const float seconds = 4f;
            int samples = (int)(sampleRate * seconds);
            var data = new float[samples];
            var rng = new System.Random(9090);
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                low += (white - low) * 0.06f;
                data[sampleIndex] = low * 1.4f;
            }
            var clip = AudioClip.Create("GustWind", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            SharedWind = clip;
            return clip;
        }
    }
}
