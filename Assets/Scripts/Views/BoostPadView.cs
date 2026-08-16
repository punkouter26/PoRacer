using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Hazard trigger volume built by Systems_TrackBuilder: any ArticulationBody
    /// part inside gets a forward push, so pads reward racers whose line crosses
    /// them. Bodies are collected in OnTriggerStay and forced once per
    /// FixedUpdate so multi-collider limbs are never double-boosted.
    /// Entering the pad sparks particles and plays a rising whoosh. The whoosh is a
    /// recorded CC0 clip with a synthesized fallback; the particles and their
    /// material are built in code.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class BoostPadView : MonoBehaviour
    {
        // Forward acceleration while inside; creatures crawl at ~0.5-2 m/s, so a
        // pad crossing gives a visible surge without launching anyone.
        private const float BOOST_ACCEL = 5f;
        private const int SPARK_PARTICLES = 14;
        private const float WHOOSH_COOLDOWN_SECONDS = 0.25f;
        // The recorded whoosh peaks near 0.9, so this reads as a straight mix level.
        private const float WHOOSH_VOLUME = 0.4f;

        private static readonly Color BoostColor = new(0.3f, 1f, 0.6f);

        private readonly List<ArticulationBody> _bodiesInPad = new();
        private ParticleSystem _sparks;
        private ParticleSystem.EmitParams _emitParams;
        private AudioSource _whooshSource;
        private AudioClip _whoosh;
        private float _nextWhooshTime;

        private void Awake()
        {
            _sparks = BuildSparks();
            // Resolved once here, never from the trigger callback.
            _whoosh = AudioLibrary.GetOrSynthesize("boost_whoosh", SynthesizeWhoosh);
            _whooshSource = gameObject.AddComponent<AudioSource>();
            _whooshSource.playOnAwake = false;
            _whooshSource.spatialBlend = 1f;
            _whooshSource.dopplerLevel = 0f;
            _whooshSource.minDistance = 3f;
            _whooshSource.maxDistance = 35f;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.attachedArticulationBody == null)
            {
                return;
            }
            _emitParams.position = other.bounds.center;
            _sparks.Emit(_emitParams, SPARK_PARTICLES);
            if (Time.time >= _nextWhooshTime && _whoosh != null)
            {
                _nextWhooshTime = Time.time + WHOOSH_COOLDOWN_SECONDS;
                _whooshSource.PlayOneShot(_whoosh, WHOOSH_VOLUME);
            }
        }

        private void OnTriggerStay(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body != null && !_bodiesInPad.Contains(body))
            {
                _bodiesInPad.Add(body);
            }
        }

        private void FixedUpdate()
        {
            for (int bodyIndex = 0; bodyIndex < _bodiesInPad.Count; bodyIndex++)
            {
                ArticulationBody body = _bodiesInPad[bodyIndex];
                if (body != null)
                {
                    body.AddForce(Vector3.forward * (BOOST_ACCEL * body.mass));
                }
            }
            _bodiesInPad.Clear();
        }

        private ParticleSystem BuildSparks()
        {
            var go = new GameObject("BoostSparks");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.3f, 0.6f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(2f, 4.5f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.05f, 0.14f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                BoostColor,
                new Color(0.7f, 1f, 0.85f));
            main.gravityModifier = 0.4f;
            main.maxParticles = 200;

            // A slow ambient sparkle telegraphs the pad as a bonus even when
            // nothing is crossing; entry bursts still come from Emit().
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 4f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 20f;
            shape.radius = 0.4f;
            shape.rotation = new Vector3(-90f, 0f, 0f);

            var sparkRenderer = ps.GetComponent<ParticleSystemRenderer>();
            Shader particleShader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (particleShader != null)
            {
                sparkRenderer.material = new Material(particleShader);
            }
            sparkRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            sparkRenderer.receiveShadows = false;
            return ps;
        }

        /// <summary>
        /// Fallback whoosh for when the recorded clip is absent: noise through an
        /// opening low-pass plus a sine sweep.
        /// </summary>
        private static AudioClip SynthesizeWhoosh()
        {
            const int sampleRate = 44100;
            const float seconds = 0.35f;
            int samples = (int)(sampleRate * seconds);
            var data = new float[samples];
            var rng = new System.Random(1717);
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / sampleRate;
                float progress = t / seconds;
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                low += (white - low) * Mathf.Lerp(0.05f, 0.5f, progress);
                float sweep = Mathf.Sin(2f * Mathf.PI * Mathf.Lerp(300f, 900f, progress) * t);
                float envelope = Mathf.Sin(Mathf.PI * progress);
                data[sampleIndex] = (low * 1.1f + sweep * 0.25f) * envelope * 0.6f;
            }
            var clip = AudioClip.Create("BoostWhoosh", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
