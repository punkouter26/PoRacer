using MessagePipe;
using PoRacer.Models;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Finish-line celebration: confetti burst and a synthesized fanfare when a
    /// racer takes first place, small tick for later places. All effects are
    /// generated in code — no audio or particle asset files.
    /// </summary>
    public sealed class WinFxView : MonoBehaviour
    {
        private const int CONFETTI_COUNT = 120;

        [SerializeField] private Transform _finishLine;

        private ParticleSystem _confetti;
        private ParticleSystem _fireworks;
        private AudioSource _audioSource;
        private AudioClip _fanfare;
        private AudioClip _tick;
        private System.IDisposable _subscription;

        [Inject]
        public void Construct(ISubscriber<RacerFinishedMessage> racerFinished)
        {
            _subscription = racerFinished.Subscribe(OnRacerFinished);
        }

        private void Awake()
        {
            _audioSource = gameObject.AddComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _audioSource.spatialBlend = 0f;
            _fanfare = SynthesizeFanfare();
            _tick = SynthesizeTick();
            _confetti = BuildConfetti();
            _fireworks = BuildFireworks();
        }

        private void OnDestroy() => _subscription?.Dispose();

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (_finishLine != null)
            {
                _confetti.transform.position = _finishLine.position + Vector3.up * 2f;
                _fireworks.transform.position = _finishLine.position + Vector3.up * 5f;
            }
            if (message.Place == 1)
            {
                _confetti.Emit(CONFETTI_COUNT);
                _fireworks.Emit(90);
                _audioSource.PlayOneShot(_fanfare, 0.8f);
            }
            else
            {
                // Small podium puff for 2nd and 3rd, a bare tick after that.
                if (message.Place <= 3)
                {
                    _confetti.Emit(CONFETTI_COUNT / 4);
                }
                _audioSource.PlayOneShot(_tick, 0.5f);
            }
        }

        private ParticleSystem BuildConfetti()
        {
            var go = new GameObject("Confetti");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();
            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.startLifetime = 2.5f;
            main.startSpeed = new ParticleSystem.MinMaxCurve(4f, 9f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.08f, 0.2f);
            main.gravityModifier = 0.6f;
            main.startColor = new ParticleSystem.MinMaxGradient(Color.yellow, new Color(1f, 0.3f, 0.6f));
            main.maxParticles = 500;
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 0f;
            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 40f;
            shape.rotation = new Vector3(-90f, 0f, 0f);
            // A third of the confetti pieces pull short streamer tails.
            ParticleSystem.TrailModule trails = ps.trails;
            trails.enabled = true;
            trails.ratio = 0.35f;
            trails.lifetime = new ParticleSystem.MinMaxCurve(0.25f);
            trails.dieWithParticles = true;
            trails.inheritParticleColor = true;
            trails.widthOverTrail = new ParticleSystem.MinMaxCurve(0.05f);
            var renderer = ps.GetComponent<ParticleSystemRenderer>();
            // Soft round sprite instead of the default square. Null in headless
            // builds where shaders are stripped; defaults are fine there.
            Material soft = FxUtil.SoftParticleMaterial();
            if (soft != null)
            {
                renderer.material = soft;
                renderer.trailMaterial = soft;
            }
            return ps;
        }

        private ParticleSystem BuildFireworks()
        {
            // Radial spark burst above the arch: the winner's firework.
            var go = new GameObject("Fireworks");
            go.transform.SetParent(transform, false);
            var ps = go.AddComponent<ParticleSystem>();
            ParticleSystem.MainModule main = ps.main;
            main.playOnAwake = false;
            main.startLifetime = new ParticleSystem.MinMaxCurve(1.2f, 2f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(6f, 12f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.06f, 0.15f);
            main.gravityModifier = 0.9f;
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(1f, 0.85f, 0.2f), new Color(0.3f, 0.8f, 1f));
            main.maxParticles = 400;
            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 0f;
            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Sphere;
            shape.radius = 0.2f;
            // Every spark drags a bright tail: reads as a real firework burst.
            ParticleSystem.TrailModule trails = ps.trails;
            trails.enabled = true;
            trails.ratio = 1f;
            trails.lifetime = new ParticleSystem.MinMaxCurve(0.35f);
            trails.dieWithParticles = true;
            trails.inheritParticleColor = true;
            trails.widthOverTrail = new ParticleSystem.MinMaxCurve(0.04f);
            var fireworkRenderer = ps.GetComponent<ParticleSystemRenderer>();
            // Additive glow: crossing spark tails sum into a bright bloom.
            Material glow = FxUtil.GlowParticleMaterial();
            if (glow != null)
            {
                fireworkRenderer.material = glow;
                fireworkRenderer.trailMaterial = glow;
            }
            fireworkRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            fireworkRenderer.receiveShadows = false;
            return ps;
        }

        private static AudioClip SynthesizeFanfare()
        {
            // Three-note rising arpeggio (C5 E5 G5), 0.6 s total.
            const int sampleRate = 44100;
            float[] notes = { 523.25f, 659.25f, 783.99f };
            const float noteSeconds = 0.2f;
            int samplesPerNote = (int)(sampleRate * noteSeconds);
            var data = new float[samplesPerNote * notes.Length];
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                for (int sampleIndex = 0; sampleIndex < samplesPerNote; sampleIndex++)
                {
                    float t = (float)sampleIndex / sampleRate;
                    float envelope = Mathf.Exp(-3f * t / noteSeconds);
                    data[noteIndex * samplesPerNote + sampleIndex] =
                        Mathf.Sin(2f * Mathf.PI * notes[noteIndex] * t) * envelope * 0.5f;
                }
            }
            var clip = AudioClip.Create("Fanfare", data.Length, 1, sampleRate, false);
            clip.SetData(data, 0);
            return clip;
        }

        private static AudioClip SynthesizeTick()
        {
            const int sampleRate = 44100;
            int samples = (int)(sampleRate * 0.08f);
            var data = new float[samples];
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / sampleRate;
                data[sampleIndex] = Mathf.Sin(2f * Mathf.PI * 880f * t) * Mathf.Exp(-40f * t) * 0.4f;
            }
            var clip = AudioClip.Create("Tick", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
