using MessagePipe;
using PoRacer.Models;
using PoRacer.Presentation;
using PoRacer.Systems;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Global race audio: start horn, crowd reactions, a reactive four-stem music
    /// bed, a per-map nature ambience layer, and the reverb zone the whole mix sits
    /// in. Big events duck the music like a sidechain compressor; the whole mix
    /// softens while the menu is open.
    ///
    /// Everything is synthesized in code at startup — there is no audio folder, no
    /// Resources load, no imported clip anywhere in the project. AudioLibrary is a
    /// cache in front of the generators, nothing more.
    ///
    /// The music is four separate sources playing four 8-second stems (pad, bass,
    /// arpeggio, percussion). They are scheduled to the same DSP sample, share one
    /// loop length, and always carry the same pitch, so they can never drift apart;
    /// race phase only ever changes their volumes. Per-racer and win sounds live in
    /// CreatureAudioView and WinFxView.
    /// </summary>
    public sealed class AudioDirectorView : MonoBehaviour
    {
        private const int SAMPLE_RATE = 44100;
        private const float AMBIENCE_VOLUME = 0.17f;
        // One-shot levels. Every clip AudioLibrary hands back is peak-matched, so
        // these read as a straight mix balance.
        private const float HORN_VOLUME = 0.55f;
        private const float CHEER_VOLUME = 0.4f;
        private const float ROAR_VOLUME = 0.6f;
        private const float DNF_VOLUME = 0.5f;
        private const float COUNTDOWN_VOLUME = 0.4f;

        // --- Music bed ---------------------------------------------------------
        // Per-stem gain trim under the master music level. The stems are individually
        // peak-matched, so these are what actually balances them against each other.
        private const float MUSIC_VOLUME = 0.13f;
        private const int STEM_PAD = 0;
        private const int STEM_BASS = 1;
        private const int STEM_ARP = 2;
        private const int STEM_PERC = 3;
        private const int STEM_COUNT = 4;
        private const float STEM_FADE_SECONDS = 1.6f;
        // Pad level while nothing is racing: present, but clearly idling.
        private const float PAD_IDLE_LEVEL = 0.75f;
        // Race phase thresholds, as a fraction of TrackLengthMeters covered by the
        // leader: arpeggio joins at mid-race, percussion at the final stretch.
        private const float MID_RACE_FRACTION = 0.4f;
        private const float FINAL_STRETCH_FRACTION = 0.8f;
        private const float FINAL_STRETCH_PITCH = 1.07f;
        private const float PITCH_RATE = 0.25f;
        private const float PHASE_POLL_SECONDS = 0.25f;
        // Lead-in for the scheduled start so all four stems land on one DSP sample
        // even if a stem's first buffer is late.
        private const double STEM_START_LEAD_SECONDS = 0.25;

        // --- Music theory ------------------------------------------------------
        private const float CHORD_SECONDS = 2f;
        private const int ARPS_PER_CHORD = 8;
        private const int PERC_BEATS = 16;

        private static readonly float[][] Chords =
        {
            new[] { 220f, 261.63f, 329.63f },  // Am
            new[] { 174.61f, 220f, 261.63f },  // F
            new[] { 196f, 246.94f, 293.66f },  // G
            new[] { 164.81f, 220f, 261.63f }   // Em
        };

        // Which chord tone each eighth note takes: up, over, down, over.
        private static readonly int[] ArpPattern = { 0, 1, 2, 1, 0, 2, 1, 2 };

        private static readonly float[] StemLevel = { 1f, 0.8f, 0.5f, 0.45f };

        // --- Reverb ------------------------------------------------------------
        private const float REVERB_MIN_DISTANCE = 15f;
        private const float REVERB_MAX_DISTANCE = 60f;

        private readonly AudioSource[] _stems = new AudioSource[STEM_COUNT];
        private readonly float[] _stemGain = new float[STEM_COUNT];
        private readonly float[] _stemTarget = new float[STEM_COUNT];
        private AudioSource _sfxSource;
        private AudioSource _ambienceSource;
        private AudioReverbZone _reverbZone;

        private AudioClip _startHorn;
        private AudioClip _crowdCheer;
        private AudioClip _crowdRoar;
        private AudioClip _crowdGasp;
        private AudioClip _wipeoutSting;
        private AudioClip _photoFinishFanfare;
        private AudioClip _subBassDrop;
        private AudioClip _dnfBlip;
        private AudioClip _countdownBeep;
        private RaceConfigModel _config;
        private RaceModel _raceModel;
        private int _lastCountdown;
        private System.IDisposable _subscriptions;
        private readonly System.Collections.Generic.Dictionary<TrackKind, AudioClip> _ambienceByKind = new();
        private TrackKind _ambienceKind = (TrackKind)(-1);
        private Systems_AudioMix _mix;
        private float _phasePollTimer;
        private float _musicPitch = 1f;
        private bool _finalStretch;

        [Inject]
        public void Construct(
            RaceConfigModel config,
            RaceModel raceModel,
            Systems_AudioMix mix,
            ISubscriber<RaceStartedMessage> raceStarted,
            ISubscriber<LeadChangedMessage> leadChanged,
            ISubscriber<RacerFinishedMessage> racerFinished,
            ISubscriber<RacerDnfMessage> racerDnf,
            ISubscriber<RacerWipeoutMessage> racerWipeout = null,
            ISubscriber<PhotoFinishMessage> photoFinish = null)
        {
            _config = config;
            _raceModel = raceModel;
            _mix = mix;
            var bag = DisposableBag.CreateBuilder();
            raceStarted.Subscribe(OnRaceStarted).AddTo(bag);
            leadChanged.Subscribe(OnLeadChanged).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            racerDnf.Subscribe(OnRacerDnf).AddTo(bag);
            racerWipeout?.Subscribe(OnRacerWipeout).AddTo(bag);
            photoFinish?.Subscribe(OnPhotoFinish).AddTo(bag);
            _subscriptions = bag.Build();
        }

        private void Awake()
        {
            // Everything below sums into one mix of individually peak-matched
            // synthesized clips, which is exactly the case that clips.
            MasterLimiterView.EnsureOnListener();

            _sfxSource = gameObject.AddComponent<AudioSource>();
            _sfxSource.playOnAwake = false;
            _sfxSource.spatialBlend = 0f;

            BuildMusicStems();

            _ambienceSource = gameObject.AddComponent<AudioSource>();
            _ambienceSource.playOnAwake = false;
            _ambienceSource.spatialBlend = 0f;
            _ambienceSource.loop = true;
            _ambienceSource.volume = 0f;

            // Track-sized reverb. The listener rides the camera, so a single zone
            // centred on the director covers the whole course; the preset is picked
            // per map in OnConfigChanged.
            _reverbZone = gameObject.AddComponent<AudioReverbZone>();
            _reverbZone.minDistance = REVERB_MIN_DISTANCE;
            _reverbZone.maxDistance = REVERB_MAX_DISTANCE;
            _reverbZone.reverbPreset = AudioReverbPreset.Plain;

            _startHorn = AudioLibrary.GetOrSynthesize("start_horn", SynthesizeStartHorn);
            _subBassDrop = AudioLibrary.GetOrSynthesize("sub_bass_drop", SynthesizeSubBassDrop);
            _crowdCheer = AudioLibrary.GetOrSynthesize("crowd_cheer", SynthesizeCheer);
            _crowdRoar = AudioLibrary.GetOrSynthesize("crowd_roar", SynthesizeRoar);
            _crowdGasp = AudioLibrary.GetOrSynthesize("crowd_gasp", SynthesizeCrowdGasp);
            _wipeoutSting = AudioLibrary.GetOrSynthesize("wipeout_sting", SynthesizeWipeoutSting);
            _photoFinishFanfare = AudioLibrary.GetOrSynthesize("photofinish_fanfare", SynthesizePhotoFinishFanfare);
            _dnfBlip = AudioLibrary.GetOrSynthesize("dnf_blip", SynthesizeDnfBlip);
            _countdownBeep = AudioLibrary.GetOrSynthesize("countdown_beep", SynthesizeCountdownBeep);
        }

        private void Start()
        {
            if (_config != null)
            {
                _config.Changed += OnConfigChanged;
                OnConfigChanged();
            }
        }

        private void Update()
        {
            // Countdown pips: RaceModel drives the timing; one beep per tick.
            if (_raceModel != null && _raceModel.CountdownValue != _lastCountdown)
            {
                if (_raceModel.CountdownValue > 0)
                {
                    _sfxSource.PlayOneShot(_countdownBeep, COUNTDOWN_VOLUME * SfxGain());
                }
                _lastCountdown = _raceModel.CountdownValue;
            }

            float deltaTime = Time.deltaTime;

            // Walking the racer list every frame would be wasted work: phases change
            // on the scale of seconds, so poll and cache the stem targets.
            _phasePollTimer -= deltaTime;
            if (_phasePollTimer <= 0f)
            {
                _phasePollTimer = PHASE_POLL_SECONDS;
                UpdatePhaseTargets();
            }

            _musicPitch = Mathf.MoveTowards(
                _musicPitch, _finalStretch ? FINAL_STRETCH_PITCH : 1f, deltaTime * PITCH_RATE);

            // Duck release and the menu-versus-race level are both envelopes on the
            // mix model now, so the creature and hazard sources duck with the music
            // instead of staying at full level through a start horn.
            float musicMix = _mix != null ? _mix.Gain(AudioBus.Music) : 1f;
            float ambienceMix = _mix != null ? _mix.Gain(AudioBus.Ambience) : 1f;

            float fadeStep = deltaTime / STEM_FADE_SECONDS;
            for (int stemIndex = 0; stemIndex < STEM_COUNT; stemIndex++)
            {
                AudioSource stem = _stems[stemIndex];
                if (stem == null)
                {
                    continue;
                }
                _stemGain[stemIndex] = Mathf.MoveTowards(
                    _stemGain[stemIndex], _stemTarget[stemIndex], fadeStep);
                stem.volume = MUSIC_VOLUME * StemLevel[stemIndex] * _stemGain[stemIndex] * musicMix;
                // One pitch for every stem, written from one place: this is what
                // keeps four independently-playing sources sample-locked.
                stem.pitch = _musicPitch;
            }
            _ambienceSource.volume = AMBIENCE_VOLUME * ambienceMix;
        }

        /// <summary>
        /// SFX bus multiplier for a one-shot. Read at fire time rather than cached,
        /// because the duck it folds in is moving while the race runs.
        /// </summary>
        private float SfxGain() => _mix != null ? _mix.Gain(AudioBus.Sfx) : 1f;

        private void OnDestroy()
        {
            _subscriptions?.Dispose();
            if (_config != null)
            {
                _config.Changed -= OnConfigChanged;
            }
        }

        private void BuildMusicStems()
        {
            _stems[STEM_PAD] = CreateStem(AudioLibrary.GetOrSynthesize("music_pad", SynthesizePadStem));
            _stems[STEM_BASS] = CreateStem(AudioLibrary.GetOrSynthesize("music_bass", SynthesizeBassStem));
            _stems[STEM_ARP] = CreateStem(AudioLibrary.GetOrSynthesize("music_arp", SynthesizeArpStem));
            _stems[STEM_PERC] = CreateStem(AudioLibrary.GetOrSynthesize("music_perc", SynthesizePercStem));

            // Menu state until the first phase poll: pad only, everything else muted.
            _stemTarget[STEM_PAD] = PAD_IDLE_LEVEL;

            // Schedule every stem onto the same DSP sample. Play() would start each
            // source on the next mixer buffer it happens to be picked up in, which
            // is where multi-source music drifts out of phase.
            double startTime = AudioSettings.dspTime + STEM_START_LEAD_SECONDS;
            for (int stemIndex = 0; stemIndex < STEM_COUNT; stemIndex++)
            {
                if (_stems[stemIndex] != null)
                {
                    _stems[stemIndex].PlayScheduled(startTime);
                }
            }
        }

        private AudioSource CreateStem(AudioClip clip)
        {
            if (clip == null)
            {
                return null;
            }
            AudioSource source = gameObject.AddComponent<AudioSource>();
            source.playOnAwake = false;
            source.spatialBlend = 0f;
            source.loop = true;
            source.clip = clip;
            source.volume = 0f;
            // Music is a bed, not a room sound: keep it out of the track reverb.
            source.reverbZoneMix = 0f;
            return source;
        }

        /// <summary>
        /// Maps race state onto stem volumes. Menu/between races: pad. Racing: pad +
        /// bass. Leader past MID_RACE_FRACTION: + arpeggio. Final stretch: + drums,
        /// and the whole bed pitches up.
        /// </summary>
        private void UpdatePhaseTargets()
        {
            bool racing = _raceModel != null && _raceModel.RaceActive;
            float leaderFraction = racing ? LeaderFraction() : 0f;
            _finalStretch = racing && leaderFraction >= FINAL_STRETCH_FRACTION;
            bool midRace = racing && leaderFraction >= MID_RACE_FRACTION;

            _stemTarget[STEM_PAD] = racing ? 1f : PAD_IDLE_LEVEL;
            _stemTarget[STEM_BASS] = racing ? 1f : 0f;
            _stemTarget[STEM_ARP] = midRace ? 1f : 0f;
            _stemTarget[STEM_PERC] = _finalStretch ? 1f : 0f;
        }

        /// <summary>Furthest-along still-racing entrant, as a fraction of the track.</summary>
        private float LeaderFraction()
        {
            if (_raceModel.TrackLengthMeters <= 0f)
            {
                return 0f;
            }
            float best = 0f;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                if (racer.Status != RacerStatus.Racing)
                {
                    continue;
                }
                float fraction = racer.Progress / _raceModel.TrackLengthMeters;
                if (fraction > best)
                {
                    best = fraction;
                }
            }
            return best;
        }

        private void OnConfigChanged()
        {
            TrackKind kind = Systems_MapCatalog.Get(_config.SelectedMapIndex).Kind;
            if (kind == _ambienceKind)
            {
                return;
            }
            _ambienceKind = kind;
            if (!_ambienceByKind.TryGetValue(kind, out AudioClip clip))
            {
                clip = SynthesizeAmbience(kind);
                _ambienceByKind[kind] = clip;
            }
            _ambienceSource.clip = clip;
            _ambienceSource.Play();
            ApplyReverbPreset(kind);
        }

        /// <summary>
        /// Room tone per map: the swamp is a dank, wet basin, the lumpy map is open
        /// hillside with returns off the rocks, everything else is open field.
        /// </summary>
        private void ApplyReverbPreset(TrackKind kind)
        {
            if (_reverbZone == null)
            {
                return;
            }
            if (kind == TrackKind.Swamp)
            {
                _reverbZone.reverbPreset = AudioReverbPreset.Cave;
            }
            else if (kind == TrackKind.Lumpy)
            {
                _reverbZone.reverbPreset = AudioReverbPreset.Mountains;
            }
            else
            {
                _reverbZone.reverbPreset = AudioReverbPreset.Plain;
            }
        }

        private void OnRaceStarted(RaceStartedMessage message)
        {
            _sfxSource.PlayOneShot(_startHorn, HORN_VOLUME * SfxGain());
            if (_subBassDrop != null)
            {
                _sfxSource.PlayOneShot(_subBassDrop, 0.85f * SfxGain());
            }
            _mix?.Duck(1f);
            // Do not wait for the poll: the bass should arrive with the horn.
            UpdatePhaseTargets();
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            _sfxSource.PlayOneShot(_crowdCheer, CHEER_VOLUME * SfxGain());
            _mix?.Duck(0.5f);
        }

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place == 1)
            {
                _sfxSource.PlayOneShot(_crowdRoar, ROAR_VOLUME * SfxGain());
                if (_subBassDrop != null)
                {
                    _sfxSource.PlayOneShot(_subBassDrop, 0.9f * SfxGain());
                }
                _mix?.Duck(1f);
            }
        }

        private void OnRacerDnf(RacerDnfMessage message)
        {
            _sfxSource.PlayOneShot(_dnfBlip, DNF_VOLUME * SfxGain());
        }

        private void OnRacerWipeout(RacerWipeoutMessage message)
        {
            _sfxSource.PlayOneShot(_crowdGasp, 0.45f * SfxGain());
            if (message.IsFatal)
            {
                _sfxSource.PlayOneShot(_wipeoutSting, 0.6f * SfxGain());
            }
            _mix?.Duck(0.6f);
        }

        private void OnPhotoFinish(PhotoFinishMessage message)
        {
            _sfxSource.PlayOneShot(_photoFinishFanfare, 0.7f * SfxGain());
            _mix?.Duck(1f);
        }

        // ====================================================================
        //  Music stems. All four are exactly Chords.Length * CHORD_SECONDS long
        //  (8 s), which is what lets them loop against each other forever.
        // ====================================================================

        private static int SamplesPerChord => (int)(SAMPLE_RATE * CHORD_SECONDS);

        private static int LoopSamples => SamplesPerChord * Chords.Length;

        /// <summary>
        /// Pad: three chord tones, each tripled and detuned a few cents, under a
        /// sine window per chord so the harmony breathes and the joins never click.
        /// </summary>
        private static AudioClip SynthesizePadStem()
        {
            int samplesPerChord = SamplesPerChord;
            var data = new float[LoopSamples];
            for (int chordIndex = 0; chordIndex < Chords.Length; chordIndex++)
            {
                float[] chord = Chords[chordIndex];
                for (int sampleIndex = 0; sampleIndex < samplesPerChord; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    float window = Mathf.Sin(Mathf.PI * sampleIndex / samplesPerChord);
                    // One slow tremolo cycle per chord keeps the loop point seamless.
                    float shimmer = 0.5f + 0.5f * Mathf.Sin(SynthUtil.TWO_PI * t / CHORD_SECONDS);
                    float value = 0f;
                    for (int noteIndex = 0; noteIndex < chord.Length; noteIndex++)
                    {
                        float frequency = chord[noteIndex];
                        value += Mathf.Sin(SynthUtil.TWO_PI * frequency * t) * 0.16f;
                        value += Mathf.Sin(SynthUtil.TWO_PI * frequency * 1.005f * t) * 0.09f;
                        value += Mathf.Sin(SynthUtil.TWO_PI * frequency * 0.994f * t) * 0.06f;
                        value += Mathf.Sin(SynthUtil.TWO_PI * frequency * 2f * t) * 0.035f * shimmer;
                    }
                    data[chordIndex * samplesPerChord + sampleIndex] = value * window * 0.5f;
                }
            }
            return MakeClip("MusicPad", data);
        }

        /// <summary>
        /// Bass: the chord root an octave down, two plucked pulses per chord, with a
        /// second and third partial for body on small speakers.
        /// </summary>
        private static AudioClip SynthesizeBassStem()
        {
            int samplesPerChord = SamplesPerChord;
            int samplesPerPulse = samplesPerChord / 2;
            int fadeSamples = (int)(SAMPLE_RATE * 0.02f);
            var data = new float[LoopSamples];
            for (int chordIndex = 0; chordIndex < Chords.Length; chordIndex++)
            {
                float root = Chords[chordIndex][0] * 0.5f;
                for (int sampleIndex = 0; sampleIndex < samplesPerChord; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    int pulseSlot = sampleIndex / samplesPerPulse;
                    float pulseT = (sampleIndex - pulseSlot * samplesPerPulse) / (float)SAMPLE_RATE;
                    // Fast attack, long tail: a finger-plucked electric bass shape.
                    float envelope = Mathf.Min(1f, pulseT * 120f) * Mathf.Exp(-2.2f * pulseT);
                    float value = Mathf.Sin(SynthUtil.TWO_PI * root * t)
                        + 0.3f * Mathf.Sin(SynthUtil.TWO_PI * root * 2f * t) * Mathf.Exp(-4f * pulseT)
                        + 0.12f * Mathf.Sin(SynthUtil.TWO_PI * root * 3f * t) * Mathf.Exp(-8f * pulseT);
                    float edge = SynthUtil.EdgeFade(sampleIndex, samplesPerChord, fadeSamples);
                    data[chordIndex * samplesPerChord + sampleIndex] = value * envelope * edge * 0.45f;
                }
            }
            return MakeClip("MusicBass", data);
        }

        /// <summary>
        /// Arpeggio: eighth-note plucks on chord tones an octave up, following a
        /// fixed up-over-down figure so it reads as a melody rather than a scale.
        /// </summary>
        private static AudioClip SynthesizeArpStem()
        {
            int samplesPerChord = SamplesPerChord;
            int samplesPerArp = samplesPerChord / ARPS_PER_CHORD;
            var data = new float[LoopSamples];
            for (int chordIndex = 0; chordIndex < Chords.Length; chordIndex++)
            {
                float[] chord = Chords[chordIndex];
                for (int sampleIndex = 0; sampleIndex < samplesPerChord; sampleIndex++)
                {
                    int arpSlot = sampleIndex / samplesPerArp;
                    if (arpSlot >= ARPS_PER_CHORD)
                    {
                        arpSlot = ARPS_PER_CHORD - 1;
                    }
                    float frequency = chord[ArpPattern[arpSlot] % chord.Length] * 2f;
                    float arpT = (sampleIndex - arpSlot * samplesPerArp) / (float)SAMPLE_RATE;
                    float body = Mathf.Sin(SynthUtil.TWO_PI * frequency * arpT) * Mathf.Exp(-13f * arpT);
                    // A brighter partial that dies first: the "pick" of the pluck.
                    float pick = Mathf.Sin(SynthUtil.TWO_PI * frequency * 2f * arpT) * Mathf.Exp(-40f * arpT) * 0.35f;
                    float attack = Mathf.Min(1f, arpT * 500f);
                    data[chordIndex * samplesPerChord + sampleIndex] = (body + pick) * attack * 0.5f;
                }
            }
            return MakeClip("MusicArp", data);
        }

        /// <summary>
        /// Percussion: a sine kick on every other beat plus filtered-noise hats on
        /// the eighths. Everything decays well before the loop point, so the tail
        /// never has to wrap.
        /// </summary>
        private static AudioClip SynthesizePercStem()
        {
            var data = new float[LoopSamples];
            int samplesPerBeat = LoopSamples / PERC_BEATS;
            var rng = new System.Random(4242);
            for (int beatIndex = 0; beatIndex < PERC_BEATS; beatIndex++)
            {
                int beatStart = beatIndex * samplesPerBeat;
                if (beatIndex % 2 == 0)
                {
                    AddKick(data, beatStart);
                }
                // Backbeat snap on beats 2 and 6 of each bar: a bright, short hat.
                bool accent = beatIndex % 4 == 2;
                AddHat(data, beatStart, rng, accent ? 0.55f : 0.3f, accent ? 90f : 130f);
                AddHat(data, beatStart + samplesPerBeat / 2, rng, 0.18f, 160f);
            }
            return MakeClip("MusicPerc", data);
        }

        /// <summary>Sine kick: a pitch drop from 120 Hz to 45 Hz with a click on top.</summary>
        private static void AddKick(float[] data, int startSample)
        {
            const float seconds = 0.3f;
            int samples = (int)(SAMPLE_RATE * seconds);
            float phase = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                int target = startSample + sampleIndex;
                if (target >= data.Length)
                {
                    return;
                }
                float t = (float)sampleIndex / SAMPLE_RATE;
                // Integrate the swept frequency so the phase stays continuous.
                float frequency = Mathf.Lerp(120f, 45f, Mathf.Clamp01(t / 0.09f));
                SynthUtil.AdvancePhase(ref phase, frequency, SAMPLE_RATE);
                float envelope = Mathf.Min(1f, t * 900f) * Mathf.Exp(-9f * t);
                float click = Mathf.Sin(SynthUtil.TWO_PI * 1400f * t) * Mathf.Exp(-260f * t) * 0.18f;
                data[target] += (Mathf.Sin(phase) + click) * envelope * 0.9f;
            }
        }

        /// <summary>Hi-hat: high-passed noise with a very fast decay.</summary>
        private static void AddHat(float[] data, int startSample, System.Random rng, float gain, float decay)
        {
            const float seconds = 0.09f;
            int samples = (int)(SAMPLE_RATE * seconds);
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                int target = startSample + sampleIndex;
                if (target >= data.Length)
                {
                    return;
                }
                float t = (float)sampleIndex / SAMPLE_RATE;
                float white = SynthUtil.White(rng);
                // Subtracting a lag of the signal from itself is a one-pole high-pass.
                low += (white - low) * 0.55f;
                float high = white - low;
                float envelope = Mathf.Min(1f, t * 3000f) * Mathf.Exp(-decay * t);
                data[target] += high * envelope * gain;
            }
        }

        // ====================================================================
        //  One-shots
        // ====================================================================

        private static AudioClip SynthesizeCheer()
            => SynthesizeCrowd(seconds: 1.1f, gain: 0.6f, seed: 777, clapsPerSecond: 14f);

        private static AudioClip SynthesizeRoar()
            => SynthesizeCrowd(seconds: 2.4f, gain: 0.9f, seed: 1234, clapsPerSecond: 26f);

        /// <summary>
        /// Crowd reaction. Noise pushed through three resonant band-passes parked on
        /// vowel formants (~800 / 1200 / 2600 Hz) whose gains drift against each
        /// other, which is what makes it read as voices instead of wind. Whistles
        /// and scattered claps sit on top of the swell.
        /// </summary>
        private static AudioClip SynthesizeCrowd(float seconds, float gain, int seed, float clapsPerSecond)
        {
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(seed);
            var formantLow = new SynthUtil.BandPass(800f, 4f, SAMPLE_RATE);
            var formantMid = new SynthUtil.BandPass(1200f, 5f, SAMPLE_RATE);
            var formantHigh = new SynthUtil.BandPass(2600f, 6f, SAMPLE_RATE);
            float murmur = 0f;
            float clapEnvelope = 0f;
            float clapLow = 0f;
            int nextClapSample = 0;
            float meanClapSpacing = clapsPerSecond > 0.01f ? SAMPLE_RATE / clapsPerSecond : float.MaxValue;
            // 15 ms clap tail as a per-sample multiplier.
            float clapDecay = Mathf.Exp(-1f / (0.015f * SAMPLE_RATE));

            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float white = SynthUtil.White(rng);
                // Slightly smoothed source: a crowd is many soft voices, not hiss.
                murmur += (white - murmur) * 0.45f;

                // Vowel morph. Three drifting gains at unrelated rates means the
                // formant balance never repeats over the length of the clip.
                float gainLow = 0.55f + 0.45f * Mathf.Sin(SynthUtil.TWO_PI * 0.9f * t);
                float gainMid = 0.50f + 0.50f * Mathf.Sin(SynthUtil.TWO_PI * 1.3f * t + 1.7f);
                float gainHigh = 0.35f + 0.35f * Mathf.Sin(SynthUtil.TWO_PI * 0.6f * t + 3.1f);
                float voices = formantLow.Process(murmur) * gainLow
                    + formantMid.Process(murmur) * 0.7f * gainMid
                    + formantHigh.Process(murmur) * 0.4f * gainHigh;

                if (sampleIndex >= nextClapSample && meanClapSpacing < float.MaxValue)
                {
                    clapEnvelope = 1f;
                    // Poisson-ish spacing: a clean grid would sound like a drum machine.
                    nextClapSample = sampleIndex
                        + (int)(meanClapSpacing * (0.35f + (float)rng.NextDouble() * 1.3f));
                }
                clapEnvelope *= clapDecay;
                clapLow += (white - clapLow) * 0.7f;
                float claps = (white - clapLow) * clapEnvelope;

                // Swell up fast, decay slow, like a stand reacting to a pass.
                float envelope = Mathf.Sin(Mathf.PI * Mathf.Pow(t / seconds, 0.7f));
                float whistle = 0.25f * Mathf.Sin(
                    SynthUtil.TWO_PI * 1900f * t + 3f * Mathf.Sin(SynthUtil.TWO_PI * 5f * t));
                data[sampleIndex] = (voices * 0.8f + claps * 0.6f + whistle) * envelope * gain;
            }
            return MakeClip("Crowd", data);
        }

        /// <summary>
        /// Start horn: two pips and a long call. Brass character comes from a
        /// harmonic stack (1x/2x/3x/4x) whose upper partials decay faster than the
        /// fundamental, two oscillators detuned a few cents against each other, a
        /// vibrato that fades in after the attack, and a hard 4 ms front edge.
        /// </summary>
        private static AudioClip SynthesizeStartHorn()
        {
            float[] notes = { 392f, 392f, 587.33f };
            float[] lengths = { 0.14f, 0.14f, 0.5f };
            const float gapSeconds = 0.06f;
            const float detune = 0.0017f;    // ~3 cents either side
            const float vibratoHz = 5.5f;
            const float vibratoDepth = 0.005f;

            int totalSamples = 0;
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                totalSamples += (int)(SAMPLE_RATE * (lengths[noteIndex] + gapSeconds));
            }
            var data = new float[totalSamples];
            var rng = new System.Random(5150);
            int cursor = 0;
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                float length = lengths[noteIndex];
                int noteSamples = (int)(SAMPLE_RATE * length);
                int gapSamples = (int)(SAMPLE_RATE * gapSeconds);
                float phaseUp = 0f;
                float phaseDown = 0f;
                float air = 0f;
                for (int sampleIndex = 0; sampleIndex < noteSamples; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    // Vibrato only arrives once the note has settled, the way a
                    // player leans on a held note rather than the attack.
                    float vibrato = 1f + vibratoDepth
                        * Mathf.Clamp01((t - 0.08f) / 0.12f)
                        * Mathf.Sin(SynthUtil.TWO_PI * vibratoHz * t);
                    float baseFrequency = notes[noteIndex] * vibrato;
                    SynthUtil.AdvancePhase(ref phaseUp, baseFrequency * (1f + detune), SAMPLE_RATE);
                    SynthUtil.AdvancePhase(ref phaseDown, baseFrequency * (1f - detune), SAMPLE_RATE);

                    // Brightness falls off through the note: brass loses its upper
                    // partials as the breath pressure drops.
                    float bright = Mathf.Exp(-5f * t);
                    float bright2 = bright * bright;
                    float voice = Stack(phaseUp, bright, bright2) + 0.7f * Stack(phaseDown, bright, bright2);

                    // Breath: a sliver of filtered noise across the front edge only.
                    air += (SynthUtil.White(rng) - air) * 0.3f;
                    float breath = air * Mathf.Exp(-45f * t) * 0.25f;

                    float envelope = Mathf.Min(1f, t * 250f) * Mathf.Exp(-2.5f * t / length);
                    data[cursor + sampleIndex] = (voice + breath) * envelope * 0.3f;
                }
                cursor += noteSamples + gapSamples;
            }
            return MakeClip("StartHorn", data);
        }

        /// <summary>Harmonic stack for the horn: fundamental plus 2x/3x/4x partials.</summary>
        private static float Stack(float phase, float bright, float bright2)
        {
            return Mathf.Sin(phase)
                + 0.5f * bright * Mathf.Sin(2f * phase)
                + 0.28f * bright2 * Mathf.Sin(3f * phase)
                + 0.12f * bright2 * bright * Mathf.Sin(4f * phase);
        }

        /// <summary>
        /// Countdown pip: 660 Hz with an octave and a fifth-above overtone that fade
        /// out faster than the fundamental, plus a 2 ms noise tick so it cuts through
        /// the music bed. The start horn itself is the "GO".
        /// </summary>
        private static AudioClip SynthesizeCountdownBeep()
        {
            const float seconds = 0.14f;
            const float fundamental = 660f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(311);
            float tick = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float envelope = Mathf.Min(1f, t * 330f) * Mathf.Exp(-5f * t / seconds);
                float value = Mathf.Sin(SynthUtil.TWO_PI * fundamental * t)
                    + 0.3f * Mathf.Sin(SynthUtil.TWO_PI * fundamental * 2f * t) * Mathf.Exp(-30f * t)
                    + 0.12f * Mathf.Sin(SynthUtil.TWO_PI * fundamental * 3f * t) * Mathf.Exp(-60f * t);
                tick += (SynthUtil.White(rng) - tick) * 0.5f;
                value += tick * Mathf.Exp(-500f * t) * 0.5f;
                data[sampleIndex] = value * envelope * 0.35f;
            }
            return MakeClip("CountdownBeep", data);
        }

        /// <summary>
        /// DNF blip: two falling notes, each sagging a little in pitch as it dies —
        /// a soft "aw" for a knocked-out racer.
        /// </summary>
        private static AudioClip SynthesizeDnfBlip()
        {
            float[] notes = { 294f, 220f };
            const float noteSeconds = 0.18f;
            int samplesPerNote = (int)(SAMPLE_RATE * noteSeconds);
            var data = new float[samplesPerNote * notes.Length];
            var rng = new System.Random(2027);
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                float phase = 0f;
                float tick = 0f;
                for (int sampleIndex = 0; sampleIndex < samplesPerNote; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    float progress = t / noteSeconds;
                    // Droop: the note loses 6% of its pitch across its own length.
                    float frequency = notes[noteIndex] * (1f - 0.06f * progress);
                    SynthUtil.AdvancePhase(ref phase, frequency, SAMPLE_RATE);
                    float envelope = Mathf.Min(1f, t * 300f) * Mathf.Exp(-6f * progress);
                    float value = Mathf.Sin(phase)
                        + 0.2f * Mathf.Sin(2f * phase) * Mathf.Exp(-14f * progress)
                        + 0.25f * Mathf.Sin(3f * phase) * Mathf.Exp(-8f * progress);
                    tick += (SynthUtil.White(rng) - tick) * 0.4f;
                    value += tick * Mathf.Exp(-400f * t) * 0.35f;
                    data[noteIndex * samplesPerNote + sampleIndex] = value * envelope * 0.3f;
                }
            }
            return MakeClip("DnfBlip", data);
        }

        private static AudioClip SynthesizeAmbience(TrackKind kind)
        {
            const float seconds = 8f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random((int)kind * 101 + 7);
            float low = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float white = SynthUtil.White(rng);
                low += (white - low) * 0.04f;
                // Gusting wind bed: integer LFO cycles so the loop point is seamless.
                float gust = 0.55f + 0.45f * Mathf.Sin(SynthUtil.TWO_PI * 2f * t / seconds)
                    * Mathf.Sin(SynthUtil.TWO_PI * 5f * t / seconds);
                float value = low * gust * 1.2f;

                if (kind == TrackKind.Swamp)
                {
                    // Low bog drone plus cricket chirp bursts.
                    value = value * 0.5f
                        + 0.10f * Mathf.Sin(SynthUtil.TWO_PI * 55f * t)
                        + 0.06f * Mathf.Sin(SynthUtil.TWO_PI * 82.4f * t);
                    float chirpGate = Mathf.Max(0f, Mathf.Sin(SynthUtil.TWO_PI * 3f * t / seconds) - 0.3f);
                    float chirp = Mathf.Sin(SynthUtil.TWO_PI * 4200f * t)
                        * Mathf.Max(0f, Mathf.Sin(SynthUtil.TWO_PI * 26f * t) - 0.6f);
                    value += chirp * chirpGate * 0.22f;
                }
                else if (kind != TrackKind.Lumpy)
                {
                    // Open field: soft breeze plus sparse bird chirps (two integer-
                    // cycle envelopes so the loop stays seamless).
                    value *= 0.6f;
                    float birdEnvelope = Mathf.Max(0f, Mathf.Sin(SynthUtil.TWO_PI * (t / seconds + 0.13f)) - 0.93f) * 14f;
                    float birdEnvelope2 = Mathf.Max(0f, Mathf.Sin(SynthUtil.TWO_PI * (2f * t / seconds + 0.61f)) - 0.95f) * 20f;
                    float sweep = Mathf.Sin(SynthUtil.TWO_PI * (3000f - 600f * birdEnvelope) * t);
                    value += sweep * Mathf.Clamp01(birdEnvelope + birdEnvelope2) * 0.12f;
                }
                data[sampleIndex] = value;
            }
            var clip = AudioClip.Create($"Ambience_{kind}", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }

        private static AudioClip SynthesizeCrowdGasp()
        {
            const float seconds = 0.6f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(9021);
            var formantLow = new SynthUtil.BandPass(600f, 3f, SAMPLE_RATE);
            var formantMid = new SynthUtil.BandPass(1400f, 4f, SAMPLE_RATE);
            float murmur = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float white = SynthUtil.White(rng);
                murmur += (white - murmur) * 0.4f;
                // Quick rising then falling intake
                float envelope = Mathf.Sin(Mathf.PI * Mathf.Clamp01(t / seconds));
                float filterShift = 1f + 0.5f * Mathf.Sin(SynthUtil.TWO_PI * 1.5f * t);
                float sound = (formantLow.Process(murmur) + 0.6f * formantMid.Process(murmur)) * envelope * filterShift;
                data[sampleIndex] = sound * 0.5f;
            }
            return MakeClip("CrowdGasp", data);
        }

        private static AudioClip SynthesizeWipeoutSting()
        {
            const float seconds = 0.8f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(404);
            float phase1 = 0f;
            float phase2 = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                // Dissonant dramatic minor second brass chord drop
                float freq1 = Mathf.Lerp(160f, 65f, Mathf.Clamp01(t / 0.4f));
                float freq2 = Mathf.Lerp(170f, 69f, Mathf.Clamp01(t / 0.4f));
                SynthUtil.AdvancePhase(ref phase1, freq1, SAMPLE_RATE);
                SynthUtil.AdvancePhase(ref phase2, freq2, SAMPLE_RATE);
                float envelope = Mathf.Min(1f, t * 500f) * Mathf.Exp(-4.5f * t);
                float impact = SynthUtil.White(rng) * Mathf.Exp(-40f * t) * 0.4f;
                float brass = (Mathf.Sin(phase1) + 0.8f * Mathf.Sin(phase2) + 0.4f * Mathf.Sin(2f * phase1)) * envelope;
                data[sampleIndex] = (brass + impact) * 0.6f;
            }
            return MakeClip("WipeoutSting", data);
        }

        private static AudioClip SynthesizePhotoFinishFanfare()
        {
            float[] arpeggio = { 440f, 554.37f, 659.25f, 880f, 1108.73f };
            const float noteSeconds = 0.08f;
            int totalSamples = (int)(SAMPLE_RATE * noteSeconds * arpeggio.Length + SAMPLE_RATE * 0.4f);
            var data = new float[totalSamples];
            int noteSamples = (int)(SAMPLE_RATE * noteSeconds);
            for (int n = 0; n < arpeggio.Length; n++)
            {
                float phase = 0f;
                int start = n * noteSamples;
                for (int sampleIndex = 0; sampleIndex < noteSamples * 3; sampleIndex++)
                {
                    int target = start + sampleIndex;
                    if (target >= data.Length) break;
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    SynthUtil.AdvancePhase(ref phase, arpeggio[n], SAMPLE_RATE);
                    float envelope = Mathf.Min(1f, t * 800f) * Mathf.Exp(-6f * t);
                    float shimmer = Mathf.Sin(2f * phase) * 0.3f;
                    data[target] += (Mathf.Sin(phase) + shimmer) * envelope * 0.35f;
                }
            }
            return MakeClip("PhotoFinishFanfare", data);
        }

        private static AudioClip SynthesizeSubBassDrop()
        {
            const float duration = 1.25f;
            int totalSamples = (int)(SAMPLE_RATE * duration);
            var data = new float[totalSamples];
            float phase = 0f;
            for (int sampleIndex = 0; sampleIndex < totalSamples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float freq = Mathf.Lerp(80f, 32f, t / duration);
                SynthUtil.AdvancePhase(ref phase, freq, SAMPLE_RATE);
                float envelope = Mathf.Exp(-2.2f * t);
                data[sampleIndex] = Mathf.Sin(phase) * envelope * 0.75f;
            }
            return MakeClip("SubBassDrop", data);
        }

        private static AudioClip MakeClip(string name, float[] data)
        {
            var clip = AudioClip.Create(name, data.Length, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
