using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Hazard trigger volume built by Systems_TrackBuilder: any ArticulationBody
    /// part inside gets a viscous drag force, so creatures wade slowly through
    /// mud. Bodies are collected in OnTriggerStay and forced once per
    /// FixedUpdate so multi-collider limbs are never double-damped.
    /// Entering the pit splashes mud particles; a looping spatial squelch rises
    /// with the number of bodies wading. All assets are generated in code.
    /// </summary>
    [RequireComponent(typeof(BoxCollider))]
    public sealed class MudZoneView : MonoBehaviour
    {
        // Deceleration per m/s of speed; creatures crawl at ~0.5-2 m/s.
        private const float MUD_DRAG_PER_SPEED = 2f;
        private const float MIN_SPLASH_SPEED = 0.3f;
        private const int SPLASH_PARTICLES = 10;
        private const float SQUELCH_VOLUME_PER_BODY = 0.06f;
        private const float MAX_SQUELCH_VOLUME = 0.4f;

        private static readonly Color MudColor = new(0.4f, 0.28f, 0.13f);
        private static AudioClip SharedSquelch;

        private readonly List<ArticulationBody> _bodiesInMud = new();
        private ParticleSystem _splash;
        private ParticleSystem.EmitParams _emitParams;
        private AudioSource _squelchSource;
        private int _wadingBodies;

        private void Awake()
        {
            _splash = BuildSplash();
            _squelchSource = gameObject.AddComponent<AudioSource>();
            _squelchSource.clip = GetSharedSquelch();
            _squelchSource.loop = true;
            _squelchSource.playOnAwake = false;
            _squelchSource.spatialBlend = 1f;
            _squelchSource.dopplerLevel = 0f;
            _squelchSource.minDistance = 3f;
            _squelchSource.maxDistance = 35f;
            _squelchSource.volume = 0f;
            _squelchSource.Play();
        }

        private void OnTriggerEnter(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body == null || body.linearVelocity.magnitude < MIN_SPLASH_SPEED)
            {
                return;
            }
            _emitParams.position = other.bounds.center;
            _splash.Emit(_emitParams, SPLASH_PARTICLES);
        }

        private void OnTriggerStay(Collider other)
        {
            ArticulationBody body = other.attachedArticulationBody;
            if (body != null && !_bodiesInMud.Contains(body))
            {
                _bodiesInMud.Add(body);
            }
        }

        private void FixedUpdate()
        {
            for (int bodyIndex = 0; bodyIndex < _bodiesInMud.Count; bodyIndex++)
            {
                ArticulationBody body = _bodiesInMud[bodyIndex];
                if (body != null)
                {
                    body.AddForce(-body.linearVelocity * (MUD_DRAG_PER_SPEED * body.mass));
                }
            }
            _wadingBodies = _bodiesInMud.Count;
            _bodiesInMud.Clear();
        }

        private void Update()
        {
            float target = Mathf.Min(_wadingBodies * SQUELCH_VOLUME_PER_BODY, MAX_SQUELCH_VOLUME);
            _squelchSource.volume = Mathf.MoveTowards(_squelchSource.volume, target, Time.deltaTime * 1.2f);
        }

        private ParticleSystem BuildSplash()
        {
            var go = new GameObject("MudSplash");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();

            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.4f, 0.8f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(1.2f, 3f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.06f, 0.18f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                MudColor,
                new Color(MudColor.r * 0.7f, MudColor.g * 0.7f, MudColor.b * 0.7f));
            main.gravityModifier = 1.4f;
            main.maxParticles = 200;

            // A slow ambient blorp telegraphs the pit as a hazard even when nothing
            // is wading; entry splashes still come from Emit().
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 2.5f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 35f;
            shape.radius = 0.15f;
            shape.rotation = new Vector3(-90f, 0f, 0f);

            var splashRenderer = ps.GetComponent<ParticleSystemRenderer>();
            Shader particleShader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if (particleShader != null)
            {
                splashRenderer.material = new Material(particleShader);
            }
            splashRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            splashRenderer.receiveShadows = false;
            return ps;
        }

        private static AudioClip GetSharedSquelch()
        {
            if (SharedSquelch != null)
            {
                return SharedSquelch;
            }
            // Slow wet blorps: low-passed noise gated by an irregular LFO.
            const int sampleRate = 44100;
            const float seconds = 2f;
            int samples = (int)(sampleRate * seconds);
            var data = new float[samples];
            var rng = new System.Random(4242);
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / sampleRate;
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                low += (white - low) * 0.05f;
                float gate = Mathf.Max(0f, Mathf.Sin(2f * Mathf.PI * 2.3f * t) + 0.4f * Mathf.Sin(2f * Mathf.PI * 3.7f * t) - 0.3f);
                data[sampleIndex] = low * gate * 1.6f;
            }
            var clip = AudioClip.Create("Squelch", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            SharedSquelch = clip;
            return clip;
        }
    }
}
