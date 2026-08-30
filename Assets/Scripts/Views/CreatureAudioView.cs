using PoRacer.Presentation;
using System.Collections.Generic;
using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Per-creature 3D sound, and — since the music, crowd and hazard layers were
    /// removed — the only sound the game makes. Three layers, all synthesized in
    /// code:
    ///
    /// 1. A looping scuttle bed whose volume follows the creature's actual speed,
    ///    with per-instance pitch offsets so the shared clip never phases.
    /// 2. Contact thuds fired when a limb lands on the ground hard enough,
    ///    positioned at the contact point and rate limited so a scrabbling creature
    ///    cannot machine-gun the mix.
    /// 3. Clashes, fired when a limb strikes *another racer* rather than the
    ///    ground. Harder and brighter than a thud so a pile-up reads differently
    ///    from a gallop, and gated separately: two creatures grinding against each
    ///    other would otherwise spend the footstep budget and mute the gait.
    ///
    /// All three are fully spatial with doppler on: pan, attenuation and pitch bend
    /// come from the AudioListener on the camera. LimbContactView relays the
    /// collisions and tells the two apart; this view owns the policy.
    ///
    /// Fido is silent. He is simulated by MuJoCo and carries no Unity colliders, so
    /// nothing reaches the relays — there is no contact for them to hear.
    ///
    /// Voice budget. Unity is configured for 32 real voices, and each racer builds
    /// two sources — a 100-racer field asks for 200. Left alone, Unity virtualises
    /// the excess by priority, and every source here would carry the same default
    /// priority, so which 32 survive is arbitrary: the racer the camera is chasing
    /// can fall silent while one off-screen keeps playing. Instead the live views
    /// are ranked by distance to the listener a few times a second, the nearest few
    /// keep their loop, and everything beyond audible range stops and is skipped by
    /// the thud and clash gates. Priority is written from the same ranking, so the
    /// voices Unity does virtualise are the ones furthest away.
    ///
    /// The ranking is shared static state driven by whichever instance notices the
    /// timer first: one pass per interval for the whole field rather than one per
    /// racer per frame.
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

        // Racer-on-racer clashes. Harder to trigger than a footstep — a limb
        // brushing a rival is not worth a sound — and louder when it does land.
        private const float MIN_CLASH_SPEED = 1.4f;
        private const float FULL_CLASH_SPEED = 6f;
        private const float MIN_CLASH_VOLUME = 0.14f;
        private const float MAX_CLASH_VOLUME = 0.5f;
        // Its own gate, deliberately not the thud's: a scrum holds contact for
        // whole seconds, and sharing the footstep budget would silence the gait
        // of both creatures for as long as they leaned on each other.
        private const float MIN_CLASH_INTERVAL = 0.18f;

        // Voice budget. Twelve simultaneous loops leaves room under the 32-voice
        // ceiling for the thuds and the clashes.
        private const int MAX_AUDIBLE_LOOPS = 12;
        private const float AUDIBLE_RANGE = 50f;
        private const float RANK_INTERVAL_SECONDS = 0.25f;
        // Distance low-pass: air and distance eat the top end long before they eat
        // the level, which is most of what makes a far sound read as far.
        private const float LOWPASS_NEAR_HZ = 5000f;
        private const float LOWPASS_FAR_HZ = 900f;

        private static readonly System.Func<int, AudioClip> ThudFactory = SynthesizeThud;
        private static readonly AudioClip[] ThudClips = new AudioClip[THUD_VARIANTS];
        private static AudioClip SharedScuttle;
        private static AudioClip SharedClash;

        // Every enabled view in the scene, plus the scratch buffer the ranking
        // sorts. Both are reused, so a re-rank allocates nothing after the field
        // reaches its largest size.
        private static readonly List<CreatureAudioView> Live = new();
        private static readonly List<CreatureAudioView> RankBuffer = new();
        private static readonly System.Comparison<CreatureAudioView> ByDistance =
            (first, second) => first._listenerSqrDistance.CompareTo(second._listenerSqrDistance);
        private static float NextRankTime;
        private static Transform ListenerTransform;

        private AudioSource _source;
        private AudioSource _thudSource;
        private Transform _transform;
        private Transform _thudTransform;
        private Vector3 _lastPosition;
        private float _nextThudTime;
        private float _nextClashTime;
        private int _thudVariant;
        private AudioLowPassFilter _lowPass;
        private Systems_AudioMix _mix;
        private float _listenerSqrDistance;
        // Set by the shared ranking pass; gates the loop, the thuds and the clashes.
        private bool _audible = true;

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

            // A filter applies to every source on its GameObject, so this covers
            // the loop but not the thud or the clash, which live on a child. That
            // split is the one we want: sustained layers read as distant through
            // their tone, short transients through their level.
            _lowPass = gameObject.AddComponent<AudioLowPassFilter>();
            _lowPass.cutoffFrequency = LOWPASS_NEAR_HZ;

            BuildThudSource(entityId);
            AttachLimbRelays();
        }

        /// <summary>
        /// Hands this racer the mix buses. Called by the spawner right after the
        /// component is added; without it the view still works and simply plays at
        /// its design volume, which is what happens in the training scenes.
        /// </summary>
        internal void Initialize(Systems_AudioMix mix)
        {
            _mix = mix;
        }

        private void OnEnable()
        {
            Live.Add(this);
            if (_source != null && _source.clip != null)
            {
                _source.time = (GetEntityId().GetHashCode() & 7) * 0.11f;
                _source.Play();
            }
        }

        private void OnDisable()
        {
            Live.Remove(this);
            if (_source != null)
            {
                _source.Stop();
            }
        }

        private void Update()
        {
            // Whichever instance gets here first past the interval re-ranks the
            // whole field; the rest of that frame's instances read the result.
            if (Time.time >= NextRankTime)
            {
                RankField();
            }

            Vector3 position = _transform.position;
            float speed = (position - _lastPosition).magnitude / Mathf.Max(Time.deltaTime, 0.0001f);
            _lastPosition = position;

            if (!_audible)
            {
                // Out of budget or out of range: hold the loop silent rather than
                // leaving Unity to decide which 32 of 300 sources win.
                if (_source.isPlaying)
                {
                    _source.Stop();
                }
                return;
            }
            if (!_source.isPlaying && _source.clip != null)
            {
                _source.Play();
            }

            float busGain = _mix != null ? _mix.Gain(AudioBus.Sfx) : 1f;
            float target = Mathf.Clamp01(speed / FULL_VOLUME_SPEED) * MAX_VOLUME * busGain;
            _source.volume = Mathf.MoveTowards(_source.volume, target, Time.deltaTime * 1.5f);
            _source.pitch = Mathf.Lerp(0.85f, 1.28f, Mathf.Clamp01(speed / FULL_VOLUME_SPEED));

            if (_lowPass != null)
            {
                // Distance rolls the top end off; the curve is on the raw distance
                // rather than the squared one so it tracks how it sounds, not how
                // it is stored.
                float distance = Mathf.Sqrt(_listenerSqrDistance);
                _lowPass.cutoffFrequency = Mathf.Lerp(
                    LOWPASS_NEAR_HZ, LOWPASS_FAR_HZ, Mathf.Clamp01(distance / AUDIBLE_RANGE));
            }
        }

        /// <summary>
        /// Sorts every live view by distance to the listener and hands out the loop
        /// budget. Runs at most once per RANK_INTERVAL_SECONDS for the whole field.
        /// </summary>
        private static void RankField()
        {
            NextRankTime = Time.time + RANK_INTERVAL_SECONDS;

            if (ListenerTransform == null)
            {
                AudioListener listener = FindAnyObjectByType<AudioListener>();
                if (listener == null)
                {
                    return;
                }
                ListenerTransform = listener.transform;
            }
            Vector3 listenerPosition = ListenerTransform.position;

            RankBuffer.Clear();
            for (int liveIndex = 0; liveIndex < Live.Count; liveIndex++)
            {
                CreatureAudioView view = Live[liveIndex];
                if (view == null)
                {
                    continue;
                }
                view._listenerSqrDistance =
                    (view._transform.position - listenerPosition).sqrMagnitude;
                RankBuffer.Add(view);
            }
            RankBuffer.Sort(ByDistance);

            const float rangeSqr = AUDIBLE_RANGE * AUDIBLE_RANGE;
            for (int rankIndex = 0; rankIndex < RankBuffer.Count; rankIndex++)
            {
                CreatureAudioView view = RankBuffer[rankIndex];
                view._audible = rankIndex < MAX_AUDIBLE_LOOPS
                    && view._listenerSqrDistance <= rangeSqr;
                // 0 is the most important voice Unity will keep. Rank maps onto the
                // priority range so that if Unity does have to virtualise, it drops
                // the racers the camera is furthest from.
                int priority = Mathf.Clamp(rankIndex * 4, 0, 255);
                view.ApplyPriority(priority);
            }
        }

        private void ApplyPriority(int priority)
        {
            if (_source != null)
            {
                _source.priority = priority;
            }
            if (_thudSource != null)
            {
                // Impacts are transient and carry the physicality of the race, so
                // they outrank the sustained loop of the same racer.
                _thudSource.priority = Mathf.Max(0, priority - 2);
            }
        }

        /// <summary>
        /// A limb landed on static geometry. Called from LimbContactView during the
        /// physics step — allocation-free, and cheap enough to survive every racer
        /// scrabbling at once because the rate gate rejects before anything else.
        /// </summary>
        internal void ReportLimbImpact(float impactSpeed, Vector3 contactPoint)
        {
            if (_thudSource == null || !_audible || impactSpeed < MIN_IMPACT_SPEED
                || Time.time < _nextThudTime)
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
            float thudGain = _mix != null ? _mix.Gain(AudioBus.Sfx) : 1f;
            _thudSource.PlayOneShot(
                clip, Mathf.Lerp(MIN_THUD_VOLUME, MAX_THUD_VOLUME, loudness) * thudGain);

            _thudVariant++;
            if (_thudVariant >= THUD_VARIANTS)
            {
                _thudVariant = 0;
            }
        }

        /// <summary>
        /// A limb struck another racer. Relayed by LimbContactView, which has
        /// already established that the other collider belongs to a different
        /// creature — a creature's own limbs knocking together stay silent.
        ///
        /// Shares the thud emitter (it is already the child that gets moved onto the
        /// contact point) but keeps its own rate gate, so a sustained shove cannot
        /// starve the footsteps of the racers involved.
        /// </summary>
        internal void ReportRivalImpact(float impactSpeed, Vector3 contactPoint)
        {
            if (_thudSource == null || !_audible || impactSpeed < MIN_CLASH_SPEED
                || Time.time < _nextClashTime)
            {
                return;
            }
            AudioClip clip = GetSharedClash();
            if (clip == null)
            {
                return;
            }
            _nextClashTime = Time.time + MIN_CLASH_INTERVAL;

            float force = Mathf.Clamp01(
                (impactSpeed - MIN_CLASH_SPEED) / (FULL_CLASH_SPEED - MIN_CLASH_SPEED));
            _thudTransform.position = contactPoint;
            // Harder hits ring lower and longer, the way a bigger collision does.
            _thudSource.pitch = Mathf.Lerp(1.25f, 0.8f, force);
            float clashGain = _mix != null ? _mix.Gain(AudioBus.Sfx) : 1f;
            _thudSource.PlayOneShot(
                clip, Mathf.Lerp(MIN_CLASH_VOLUME, MAX_CLASH_VOLUME, force) * clashGain);
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
        /// Body-on-body clash: a broadband slap over a short mid-frequency ring.
        /// Deliberately brighter and harder-edged than the thud, because the two
        /// fire into the same emitter and a listener should be able to tell a
        /// bodycheck from a footfall without seeing it. One shared clip for the
        /// whole field; force is carried by AudioSource.pitch and volume.
        /// </summary>
        private static AudioClip GetSharedClash()
        {
            if (SharedClash != null)
            {
                return SharedClash;
            }
            const float seconds = 0.19f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(24601);
            // The ring: two detuned partials, high enough to read as a knock rather
            // than the thud's settling mass.
            float phaseLow = 0f;
            float phaseHigh = 0f;
            // The slap: filtered noise with a fast decay, which is the transient
            // the ear actually uses to place the hit.
            var body = new SynthUtil.BandPass(420f, 2.2f, SAMPLE_RATE);
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float progress = t / seconds;
                SynthUtil.AdvancePhase(ref phaseLow, 196f, SAMPLE_RATE);
                SynthUtil.AdvancePhase(ref phaseHigh, 293f, SAMPLE_RATE);
                float ring = (Mathf.Sin(phaseLow) + 0.6f * Mathf.Sin(phaseHigh))
                    * Mathf.Exp(-11f * progress);
                float slap = body.Process(SynthUtil.White(rng)) * Mathf.Exp(-34f * progress);
                float envelope = Mathf.Min(1f, t * 900f);
                data[sampleIndex] = (ring * 0.45f + slap * 1.1f) * envelope;
            }
            var clip = AudioClip.Create("CreatureClash", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            SharedClash = clip;
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
