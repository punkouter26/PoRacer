using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Per-creature 3D sound. Two layers:
    ///
    /// 1. A looping scuttle bed whose volume follows the creature's actual speed.
    ///    Still synthesized — a seamless body-drag loop has no good recorded source,
    ///    and per-instance pitch offsets keep the shared clip from phasing.
    /// 2. Contact thuds: recorded soft impacts fired when a limb lands on the
    ///    ground hard enough, positioned at the contact point and rate limited so a
    ///    scrabbling creature cannot machine-gun the mix.
    ///
    /// Both are fully spatial — pan and attenuation come from the AudioListener on
    /// the camera. LimbContactView relays the collisions; this view owns the policy.
    /// </summary>
    public sealed class CreatureAudioView : MonoBehaviour
    {
        private const float FULL_VOLUME_SPEED = 2.5f;
        private const float MAX_VOLUME = 0.45f;

        // Contact thuds. Below MIN_IMPACT_SPEED a landing is a scuff, not a step;
        // at FULL_IMPACT_SPEED it is as loud as it gets.
        private const float MIN_IMPACT_SPEED = 0.9f;
        private const float FULL_IMPACT_SPEED = 4.5f;
        private const float MIN_THUD_VOLUME = 0.08f;
        private const float MAX_THUD_VOLUME = 0.35f;
        // Four per second: enough for a gait, quiet enough for eight racers at once.
        private const float MIN_THUD_INTERVAL = 0.25f;
        private const int MAX_THUD_VARIANTS = 5;

        private static readonly AudioClip[] ThudClips = new AudioClip[MAX_THUD_VARIANTS];
        private static int ThudClipCount = -1;
        private static AudioClip SharedScuttle;

        private AudioSource _source;
        private AudioSource _thudSource;
        private Transform _transform;
        private Transform _thudTransform;
        private Vector3 _lastPosition;
        private float _nextThudTime;
        private int _thudVariant;

        private void Awake()
        {
            _transform = transform;
            _lastPosition = _transform.position;
            _source = gameObject.AddComponent<AudioSource>();
            _source.clip = GetSharedScuttle();
            _source.loop = true;
            _source.playOnAwake = false;
            _source.spatialBlend = 1f;
            // Mild doppler: audible pitch bend when a racer passes the camera,
            // without warbling from articulated-body jitter.
            _source.dopplerLevel = 0.35f;
            _source.spread = 60f;
            _source.minDistance = 2f;
            _source.maxDistance = 40f;
            int entityId = GetEntityId().GetHashCode();
            _source.volume = 0f;
            _source.pitch = 0.85f + (entityId & 15) * 0.02f;

            BuildThudSource(entityId);
            AttachLimbRelays();
        }

        private void OnEnable()
        {
            if (_source != null && _source.clip != null)
            {
                _source.time = (GetEntityId().GetHashCode() & 7) * 0.11f;
                _source.Play();
            }
        }

        private void OnDisable()
        {
            if (_source != null)
            {
                _source.Stop();
            }
        }

        private void Update()
        {
            Vector3 position = _transform.position;
            float speed = (position - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.0001f);
            _lastPosition = position;
            float target = Mathf.Clamp01(speed / FULL_VOLUME_SPEED) * MAX_VOLUME;
            _source.volume = Mathf.MoveTowards(_source.volume, target, Time.deltaTime * 1.5f);
        }

        /// <summary>
        /// A limb landed on static geometry. Called from LimbContactView during the
        /// physics step — allocation-free, and cheap enough to survive every racer
        /// scrabbling at once because the rate gate rejects before anything else.
        /// </summary>
        internal void ReportLimbImpact(float impactSpeed, Vector3 contactPoint)
        {
            if (_thudSource == null || ThudClipCount <= 0
                || impactSpeed < MIN_IMPACT_SPEED || Time.time < _nextThudTime)
            {
                return;
            }
            AudioClip clip = ThudClips[_thudVariant];
            if (clip == null)
            {
                return;
            }
            _nextThudTime = Time.time + MIN_THUD_INTERVAL;

            float loudness = Mathf.Clamp01(
                (impactSpeed - MIN_IMPACT_SPEED) / (FULL_IMPACT_SPEED - MIN_IMPACT_SPEED));
            // Move the emitter onto the foot that actually landed, so a near miss
            // pans correctly instead of coming from the middle of the creature.
            _thudTransform.position = contactPoint;
            // Heavier landings sit a touch lower, the way a real impact does.
            _thudSource.pitch = Mathf.Lerp(1.12f, 0.88f, loudness);
            // Floor the gain: a landing that just cleared the threshold should be
            // faint, not silent, or the gate would swallow every gentle step.
            _thudSource.PlayOneShot(clip, Mathf.Lerp(MIN_THUD_VOLUME, MAX_THUD_VOLUME, loudness));

            _thudVariant++;
            if (_thudVariant >= ThudClipCount)
            {
                _thudVariant = 0;
            }
        }

        private void BuildThudSource(int entityId)
        {
            if (ThudClipCount < 0)
            {
                ThudClipCount = AudioLibrary.GetSeries("footstep", ThudClips);
                if (ThudClipCount == 0)
                {
                    ThudClips[0] = AudioLibrary.GetOrSynthesize("footstep_0", SynthesizeThud);
                    ThudClipCount = ThudClips[0] != null ? 1 : 0;
                }
            }
            if (ThudClipCount <= 0)
            {
                return;
            }
            var go = new GameObject("LimbThud");
            _thudTransform = go.transform;
            _thudTransform.SetParent(_transform, false);
            _thudSource = go.AddComponent<AudioSource>();
            _thudSource.playOnAwake = false;
            _thudSource.loop = false;
            _thudSource.spatialBlend = 1f;
            // No doppler: a thud is over before a pitch bend could read as anything
            // but a glitch, and the emitter teleports between contacts.
            _thudSource.dopplerLevel = 0f;
            _thudSource.minDistance = 2f;
            _thudSource.maxDistance = 30f;
            // Stagger the starting variant so same-gait racers do not land in unison.
            _thudVariant = (entityId & 7) % ThudClipCount;
        }

        /// <summary>
        /// Adds a collision relay to every solid limb collider. Runs once per racer
        /// at spawn; the GetComponentsInChildren allocation never reaches a hot path.
        /// </summary>
        private void AttachLimbRelays()
        {
            if (_thudSource == null)
            {
                return;
            }
            Collider[] colliders = GetComponentsInChildren<Collider>(true);
            for (int colliderIndex = 0; colliderIndex < colliders.Length; colliderIndex++)
            {
                Collider limb = colliders[colliderIndex];
                if (limb == null || limb.isTrigger)
                {
                    continue;
                }
                GameObject limbObject = limb.gameObject;
                if (limbObject.TryGetComponent(out LimbContactView existing))
                {
                    existing.Bind(this);
                    continue;
                }
                limbObject.AddComponent<LimbContactView>().Bind(this);
            }
        }

        private static AudioClip GetSharedScuttle()
        {
            if (SharedScuttle != null)
            {
                return SharedScuttle;
            }
            const int sampleRate = 44100;
            const float seconds = 2f;
            int samples = (int)(sampleRate * seconds);
            var data = new float[samples];
            var rng = new System.Random(12345);
            float brown = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                brown = Mathf.Clamp(brown + white * 0.12f, -1f, 1f) * 0.98f;
                float t = (float)sampleIndex / sampleRate;
                float patter = 0.55f + 0.45f * Mathf.Sin(2f * Mathf.PI * 7f * t);
                data[sampleIndex] = brown * patter * 0.5f;
            }
            var clip = AudioClip.Create("Scuttle", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            SharedScuttle = clip;
            return clip;
        }

        /// <summary>Fallback thud: a low sine drop under a fast noise burst.</summary>
        private static AudioClip SynthesizeThud()
        {
            const int sampleRate = 44100;
            const float seconds = 0.14f;
            int samples = (int)(sampleRate * seconds);
            var data = new float[samples];
            var rng = new System.Random(9081);
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / sampleRate;
                float progress = t / seconds;
                // Body: a pitch-dropping sine, the way a soft mass settles.
                float body = Mathf.Sin(2f * Mathf.PI * Mathf.Lerp(150f, 55f, progress) * t);
                // Skin: filtered noise that dies well before the body does.
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                low += (white - low) * 0.35f;
                float envelope = Mathf.Min(1f, t * 400f) * Mathf.Exp(-7f * progress);
                data[sampleIndex] = (body * 0.8f + low * Mathf.Exp(-22f * progress) * 0.5f) * envelope;
            }
            var clip = AudioClip.Create("LimbThud", samples, 1, sampleRate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
