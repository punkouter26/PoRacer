using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Per-creature 3D sound. Three layers, all synthesized in code:
    ///
    /// 1. A looping scuttle bed whose volume follows the creature's actual speed,
    ///    with per-instance pitch offsets so the shared clip never phases.
    /// 2. Contact thuds fired when a limb lands on the ground hard enough,
    ///    positioned at the contact point and rate limited so a scrabbling creature
    ///    cannot machine-gun the mix.
    /// 3. An occasional two-note voice chirp. Its pitch scales inversely with the
    ///    creature's total articulated mass — a 135 kg worm calls low, a 38 kg
    ///    spider calls high — and a global rate cap keeps a 90-racer field from
    ///    turning into a chorus.
    ///
    /// All three are fully spatial with doppler on: pan, attenuation and pitch bend
    /// come from the AudioListener on the camera. LimbContactView relays the
    /// collisions; this view owns the policy.
    /// </summary>
    public sealed class CreatureAudioView : MonoBehaviour
    {
        private const int SAMPLE_RATE = 44100;
        private const float FULL_VOLUME_SPEED = 2.5f;
        private const float MAX_VOLUME = 0.45f;
        // Audible pitch bend as a racer passes the camera. Low enough that
        // articulated-body jitter does not warble the held loop.
        private const float DOPPLER_LEVEL = 0.6f;

        // Contact thuds. Below MIN_IMPACT_SPEED a landing is a scuff, not a step;
        // at FULL_IMPACT_SPEED it is as loud as it gets.
        private const float MIN_IMPACT_SPEED = 0.9f;
        private const float FULL_IMPACT_SPEED = 4.5f;
        private const float MIN_THUD_VOLUME = 0.08f;
        private const float MAX_THUD_VOLUME = 0.35f;
        // Four per second: enough for a gait, quiet enough for eight racers at once.
        private const float MIN_THUD_INTERVAL = 0.25f;
        private const int THUD_VARIANTS = 4;

        // Idle voice chirps.
        private const float CHIRP_VOLUME = 0.2f;
        private const float CHIRP_MIN_INTERVAL = 6f;
        private const float CHIRP_MAX_INTERVAL = 15f;
        // Global throttle: the whole field shares this budget, so a full grid
        // chirps at a conversational rate instead of all at once.
        private const int MAX_CHIRPS_PER_WINDOW = 3;
        private const float CHIRP_WINDOW_SECONDS = 1f;
        // Mass that plays the chirp at its recorded pitch; heavier creatures drop
        // below it, lighter ones rise above, on a square-root curve.
        private const float REFERENCE_MASS_KG = 60f;
        private const float MIN_CHIRP_PITCH = 0.55f;
        private const float MAX_CHIRP_PITCH = 1.6f;

        private static readonly System.Func<int, AudioClip> ThudFactory = SynthesizeThud;
        private static readonly AudioClip[] ThudClips = new AudioClip[THUD_VARIANTS];
        private static AudioClip SharedScuttle;
        private static AudioClip SharedChirp;
        private static float ChirpWindowEndTime;
        private static int ChirpsThisWindow;

        private AudioSource _source;
        private AudioSource _thudSource;
        private AudioSource _voiceSource;
        private Transform _transform;
        private Transform _thudTransform;
        private Vector3 _lastPosition;
        private float _nextThudTime;
        private float _nextChirpTime;
        private float _chirpPitch = 1f;
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
            _source.dopplerLevel = DOPPLER_LEVEL;
            _source.spread = 60f;
            _source.minDistance = 2f;
            _source.maxDistance = 40f;
            int entityId = GetEntityId().GetHashCode();
            _source.volume = 0f;
            _source.pitch = 0.85f + (entityId & 15) * 0.02f;

            BuildThudSource(entityId);
            BuildVoiceSource();
            AttachLimbRelays();
        }

        private void Start()
        {
            // Mass is read here rather than in Awake: the spawner applies quirk mass
            // scaling after it adds this component, and Start runs after that pass.
            _chirpPitch = ComputeChirpPitch();
            ScheduleNextChirp();
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

            if (Time.time >= _nextChirpTime)
            {
                TryChirp();
            }
        }

        /// <summary>
        /// A limb landed on static geometry. Called from LimbContactView during the
        /// physics step — allocation-free, and cheap enough to survive every racer
        /// scrabbling at once because the rate gate rejects before anything else.
        /// </summary>
        internal void ReportLimbImpact(float impactSpeed, Vector3 contactPoint)
        {
            if (_thudSource == null || impactSpeed < MIN_IMPACT_SPEED || Time.time < _nextThudTime)
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
            if (_thudVariant >= THUD_VARIANTS)
            {
                _thudVariant = 0;
            }
        }

        /// <summary>
        /// Fires an idle call if this creature's timer is up and the field has not
        /// already spent its chirp budget for the current window. The timer is
        /// rescheduled either way, so a throttled creature simply waits its turn.
        /// </summary>
        private void TryChirp()
        {
            ScheduleNextChirp();
            if (_voiceSource == null || _voiceSource.clip == null)
            {
                return;
            }
            float now = Time.time;
            // The second test catches a stale window left behind by entering play
            // mode without a domain reload, which would otherwise mute every chirp.
            if (now >= ChirpWindowEndTime || ChirpWindowEndTime - now > CHIRP_WINDOW_SECONDS)
            {
                ChirpWindowEndTime = now + CHIRP_WINDOW_SECONDS;
                ChirpsThisWindow = 0;
            }
            if (ChirpsThisWindow >= MAX_CHIRPS_PER_WINDOW)
            {
                return;
            }
            ChirpsThisWindow++;
            _voiceSource.pitch = _chirpPitch;
            _voiceSource.Play();
        }

        private void ScheduleNextChirp()
        {
            _nextChirpTime = Time.time + Random.Range(CHIRP_MIN_INTERVAL, CHIRP_MAX_INTERVAL);
        }

        /// <summary>
        /// Total articulated mass sets the voice: big bodies resonate low. Square
        /// root keeps the spread musical — a 3.5x mass ratio becomes a touch under
        /// two octaves of pitch, not five.
        /// </summary>
        private float ComputeChirpPitch()
        {
            // One allocation at spawn, never in a hot path (same as AttachLimbRelays).
            ArticulationBody[] bodies = GetComponentsInChildren<ArticulationBody>(true);
            float totalMass = 0f;
            for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
            {
                ArticulationBody body = bodies[bodyIndex];
                if (body != null)
                {
                    totalMass += body.mass;
                }
            }
            if (totalMass <= 0.01f)
            {
                return 1f;
            }
            return Mathf.Clamp(
                Mathf.Sqrt(REFERENCE_MASS_KG / totalMass), MIN_CHIRP_PITCH, MAX_CHIRP_PITCH);
        }

        private void BuildThudSource(int entityId)
        {
            for (int variantIndex = 0; variantIndex < THUD_VARIANTS; variantIndex++)
            {
                if (ThudClips[variantIndex] == null)
                {
                    ThudClips[variantIndex] = AudioLibrary.GetOrSynthesize(
                        "footstep_" + variantIndex, variantIndex, ThudFactory);
                }
            }
            var go = new GameObject("LimbThud");
            _thudTransform = go.transform;
            _thudTransform.SetParent(_transform, false);
            _thudSource = go.AddComponent<AudioSource>();
            _thudSource.playOnAwake = false;
            _thudSource.loop = false;
            _thudSource.spatialBlend = 1f;
            _thudSource.dopplerLevel = DOPPLER_LEVEL;
            _thudSource.minDistance = 2f;
            _thudSource.maxDistance = 30f;
            // Stagger the starting variant so same-gait racers do not land in unison.
            _thudVariant = (entityId & 7) % THUD_VARIANTS;
        }

        private void BuildVoiceSource()
        {
            _voiceSource = gameObject.AddComponent<AudioSource>();
            _voiceSource.clip = GetSharedChirp();
            _voiceSource.playOnAwake = false;
            _voiceSource.loop = false;
            _voiceSource.spatialBlend = 1f;
            _voiceSource.dopplerLevel = DOPPLER_LEVEL;
            _voiceSource.spread = 40f;
            _voiceSource.minDistance = 3f;
            _voiceSource.maxDistance = 45f;
            _voiceSource.volume = CHIRP_VOLUME;
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
            const float seconds = 2f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(12345);
            float brown = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float white = SynthUtil.White(rng);
                brown = Mathf.Clamp(brown + white * 0.12f, -1f, 1f) * 0.98f;
                float t = (float)sampleIndex / SAMPLE_RATE;
                float patter = 0.55f + 0.45f * Mathf.Sin(SynthUtil.TWO_PI * 7f * t);
                data[sampleIndex] = brown * patter * 0.5f;
            }
            var clip = AudioClip.Create("Scuttle", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            SharedScuttle = clip;
            return clip;
        }

        /// <summary>
        /// Two-note call: a rising interval with a formant-filtered noise breath
        /// under it, which is what stops a pair of sine tones sounding like a
        /// menu beep. One shared clip for the whole field — the per-creature size
        /// difference is carried by AudioSource.pitch, not by 90 separate clips.
        /// </summary>
        private static AudioClip GetSharedChirp()
        {
            if (SharedChirp != null)
            {
                return SharedChirp;
            }
            float[] notes = { 430f, 645f };
            float[] lengths = { 0.11f, 0.17f };
            const float gapSeconds = 0.04f;
            int totalSamples = 0;
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                totalSamples += (int)(SAMPLE_RATE * (lengths[noteIndex] + gapSeconds));
            }
            var data = new float[totalSamples];
            var rng = new System.Random(60613);
            int cursor = 0;
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                float length = lengths[noteIndex];
                int noteSamples = (int)(SAMPLE_RATE * length);
                int gapSamples = (int)(SAMPLE_RATE * gapSeconds);
                float phase = 0f;
                // Vowel: the call opens on a low formant and brightens as it lands.
                var throat = new SynthUtil.BandPass(700f + noteIndex * 250f, 5f, SAMPLE_RATE);
                for (int sampleIndex = 0; sampleIndex < noteSamples; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    float progress = t / length;
                    // Each note bends up a little, the way a small animal's call does.
                    float frequency = notes[noteIndex] * (1f + 0.05f * progress);
                    SynthUtil.AdvancePhase(ref phase, frequency, SAMPLE_RATE);
                    float envelope = Mathf.Min(1f, t * 200f) * Mathf.Exp(-4.5f * progress);
                    float voice = Mathf.Sin(phase)
                        + 0.3f * Mathf.Sin(2f * phase)
                        + 0.12f * Mathf.Sin(3f * phase) * Mathf.Exp(-8f * progress);
                    float breath = throat.Process(SynthUtil.White(rng)) * 0.35f * Mathf.Exp(-9f * progress);
                    data[cursor + sampleIndex] = (voice + breath) * envelope * 0.4f;
                }
                cursor += noteSamples + gapSamples;
            }
            var clip = AudioClip.Create("CreatureChirp", data.Length, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            SharedChirp = clip;
            return clip;
        }

        /// <summary>
        /// Limb thud: a low sine drop under a fast noise burst. The variant index
        /// nudges the body pitch, the noise seed and the skin decay so a gait cycles
        /// through four related-but-different landings instead of one stamp.
        /// </summary>
        private static AudioClip SynthesizeThud(int variant)
        {
            const float seconds = 0.14f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(9081 + variant * 137);
            float startFrequency = 150f + variant * 11f;
            float endFrequency = 55f - variant * 2f;
            float skinDecay = 22f + variant * 3f;
            float phase = 0f;
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float progress = t / seconds;
                // Body: a pitch-dropping sine, the way a soft mass settles.
                SynthUtil.AdvancePhase(ref phase, Mathf.Lerp(startFrequency, endFrequency, progress), SAMPLE_RATE);
                float body = Mathf.Sin(phase);
                // Skin: filtered noise that dies well before the body does.
                low += (SynthUtil.White(rng) - low) * 0.35f;
                float envelope = Mathf.Min(1f, t * 400f) * Mathf.Exp(-7f * progress);
                data[sampleIndex] = (body * 0.8f + low * Mathf.Exp(-skinDecay * progress) * 0.5f) * envelope;
            }
            var clip = AudioClip.Create("LimbThud", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
