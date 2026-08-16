using MessagePipe;
using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Global race audio: start horn, crowd reactions, a synthesized music loop
    /// (pad + bass + arpeggio), and a per-map nature ambience bed (swamp crickets,
    /// dusty wind, open-field birds). Big events duck the music like a sidechain
    /// compressor; the whole mix softens while the menu is open. All clips are
    /// generated in code — no audio asset files. Per-racer and win sounds live in
    /// CreatureAudioView and WinFxView.
    /// </summary>
    public sealed class AudioDirectorView : MonoBehaviour
    {
        private const int SAMPLE_RATE = 44100;
        private const float MUSIC_VOLUME = 0.22f;
        private const float AMBIENCE_VOLUME = 0.17f;
        private const float MENU_MIX = 0.5f;
        private const float DUCK_DEPTH = 0.65f;
        private const float DUCK_RECOVERY_SECONDS = 1.2f;

        private AudioSource _sfxSource;
        private AudioSource _musicSource;
        private AudioSource _ambienceSource;
        private AudioClip _startHorn;
        private AudioClip _crowdCheer;
        private AudioClip _crowdRoar;
        private AudioClip _dnfBlip;
        private AudioClip _countdownBeep;
        private RaceConfigModel _config;
        private RaceModel _raceModel;
        private int _lastCountdown;
        private System.IDisposable _subscriptions;
        private readonly System.Collections.Generic.Dictionary<TrackKind, AudioClip> _ambienceByKind = new();
        private TrackKind _ambienceKind = (TrackKind)(-1);
        private float _duck;
        private float _menuMix = MENU_MIX;

        [Inject]
        public void Construct(
            RaceConfigModel config,
            RaceModel raceModel,
            ISubscriber<RaceStartedMessage> raceStarted,
            ISubscriber<LeadChangedMessage> leadChanged,
            ISubscriber<RacerFinishedMessage> racerFinished,
            ISubscriber<RacerDnfMessage> racerDnf)
        {
            _config = config;
            _raceModel = raceModel;
            var bag = DisposableBag.CreateBuilder();
            raceStarted.Subscribe(OnRaceStarted).AddTo(bag);
            leadChanged.Subscribe(OnLeadChanged).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            racerDnf.Subscribe(OnRacerDnf).AddTo(bag);
            _subscriptions = bag.Build();
        }

        private void Awake()
        {
            _sfxSource = gameObject.AddComponent<AudioSource>();
            _sfxSource.playOnAwake = false;
            _sfxSource.spatialBlend = 0f;

            _musicSource = gameObject.AddComponent<AudioSource>();
            _musicSource.playOnAwake = false;
            _musicSource.spatialBlend = 0f;
            _musicSource.loop = true;
            _musicSource.volume = MUSIC_VOLUME * MENU_MIX;
            _musicSource.clip = SynthesizeMusicLoop();
            _musicSource.Play();

            _ambienceSource = gameObject.AddComponent<AudioSource>();
            _ambienceSource.playOnAwake = false;
            _ambienceSource.spatialBlend = 0f;
            _ambienceSource.loop = true;
            _ambienceSource.volume = AMBIENCE_VOLUME * MENU_MIX;

            _startHorn = SynthesizeStartHorn();
            _crowdCheer = SynthesizeCrowd(seconds: 1.1f, gain: 0.6f, seed: 777);
            _crowdRoar = SynthesizeCrowd(seconds: 2.4f, gain: 0.9f, seed: 1234);
            _dnfBlip = SynthesizeDnfBlip();
            _countdownBeep = SynthesizeCountdownBeep();
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
                    _sfxSource.PlayOneShot(_countdownBeep, 0.55f);
                }
                _lastCountdown = _raceModel.CountdownValue;
            }
            _duck = Mathf.MoveTowards(_duck, 0f, Time.deltaTime / DUCK_RECOVERY_SECONDS);
            float menuTarget = _config != null && _config.MenuVisible ? MENU_MIX : 1f;
            _menuMix = Mathf.MoveTowards(_menuMix, menuTarget, Time.deltaTime * 1.5f);
            float mix = (1f - DUCK_DEPTH * _duck) * _menuMix;
            _musicSource.volume = MUSIC_VOLUME * mix;
            _ambienceSource.volume = AMBIENCE_VOLUME * mix;
        }

        private void OnDestroy()
        {
            _subscriptions?.Dispose();
            if (_config != null)
            {
                _config.Changed -= OnConfigChanged;
            }
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
        }

        private void OnRaceStarted(RaceStartedMessage message)
        {
            _sfxSource.PlayOneShot(_startHorn, 0.7f);
            _duck = 1f;
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            _sfxSource.PlayOneShot(_crowdCheer, 0.5f);
            _duck = Mathf.Max(_duck, 0.5f);
        }

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place == 1)
            {
                _sfxSource.PlayOneShot(_crowdRoar, 0.8f);
                _duck = 1f;
            }
        }

        private void OnRacerDnf(RacerDnfMessage message)
        {
            _sfxSource.PlayOneShot(_dnfBlip, 0.4f);
        }

        /// <summary>Band-passed noise swell with whistle overtones: a crowd reaction.</summary>
        private static AudioClip SynthesizeCrowd(float seconds, float gain, int seed)
        {
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            var rng = new System.Random(seed);
            float low = 0f;
            float band = 0f;
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                // Cheap band-pass: low-pass twice, take the difference.
                low += (white - low) * 0.25f;
                band += (low - band) * 0.06f;
                float noise = low - band;
                // Swell up fast, decay slow, like a stand reacting to a pass.
                float envelope = Mathf.Sin(Mathf.PI * Mathf.Pow(t / seconds, 0.7f));
                float whistle = 0.06f * Mathf.Sin(2f * Mathf.PI * 1900f * t + 3f * Mathf.Sin(2f * Mathf.PI * 5f * t));
                data[sampleIndex] = (noise * 1.6f + whistle) * envelope * gain;
            }
            var clip = AudioClip.Create("Crowd", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }

        private static AudioClip SynthesizeStartHorn()
        {
            // Two short pips then a long high note: "ready, set, GO".
            float[] notes = { 392f, 392f, 587.33f };
            float[] lengths = { 0.14f, 0.14f, 0.5f };
            int totalSamples = 0;
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                totalSamples += (int)(SAMPLE_RATE * (lengths[noteIndex] + 0.06f));
            }
            var data = new float[totalSamples];
            int cursor = 0;
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                int noteSamples = (int)(SAMPLE_RATE * lengths[noteIndex]);
                int gapSamples = (int)(SAMPLE_RATE * 0.06f);
                for (int sampleIndex = 0; sampleIndex < noteSamples; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    float envelope = Mathf.Min(1f, t * 60f) * Mathf.Exp(-2.5f * t / lengths[noteIndex]);
                    float value = Mathf.Sin(2f * Mathf.PI * notes[noteIndex] * t)
                        + 0.35f * Mathf.Sin(2f * Mathf.PI * notes[noteIndex] * 2f * t);
                    data[cursor + sampleIndex] = value * envelope * 0.4f;
                }
                cursor += noteSamples + gapSamples;
            }
            var clip = AudioClip.Create("StartHorn", data.Length, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }

        private static AudioClip SynthesizeCountdownBeep()
        {
            // Single short pip; the start horn itself is the "GO".
            const float seconds = 0.12f;
            int samples = (int)(SAMPLE_RATE * seconds);
            var data = new float[samples];
            for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
            {
                float t = (float)sampleIndex / SAMPLE_RATE;
                float envelope = Mathf.Min(1f, t * 90f) * Mathf.Exp(-5f * t / seconds);
                data[sampleIndex] = Mathf.Sin(2f * Mathf.PI * 660f * t) * envelope * 0.4f;
            }
            var clip = AudioClip.Create("CountdownBeep", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }

        private static AudioClip SynthesizeDnfBlip()
        {
            // Two falling notes: a soft "aw" for a knocked-out racer.
            float[] notes = { 294f, 220f };
            const float noteSeconds = 0.16f;
            int samplesPerNote = (int)(SAMPLE_RATE * noteSeconds);
            var data = new float[samplesPerNote * notes.Length];
            for (int noteIndex = 0; noteIndex < notes.Length; noteIndex++)
            {
                for (int sampleIndex = 0; sampleIndex < samplesPerNote; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    float envelope = Mathf.Min(1f, t * 80f) * Mathf.Exp(-6f * t / noteSeconds);
                    float value = Mathf.Sin(2f * Mathf.PI * notes[noteIndex] * t)
                        + 0.25f * Mathf.Sin(2f * Mathf.PI * notes[noteIndex] * 3f * t);
                    data[noteIndex * samplesPerNote + sampleIndex] = value * envelope * 0.35f;
                }
            }
            var clip = AudioClip.Create("DnfBlip", data.Length, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }

        private static AudioClip SynthesizeMusicLoop()
        {
            // Four chords, two seconds each: soft detuned pad, a bass root, and a
            // plucked eighth-note arpeggio on top. Eight-second seamless loop.
            const float chordSeconds = 2f;
            const int arpsPerChord = 8;
            float[][] chords =
            {
                new[] { 220f, 261.63f, 329.63f },
                new[] { 174.61f, 220f, 261.63f },
                new[] { 196f, 246.94f, 293.66f },
                new[] { 164.81f, 220f, 261.63f }
            };
            int samplesPerChord = (int)(SAMPLE_RATE * chordSeconds);
            int samplesPerArp = samplesPerChord / arpsPerChord;
            var data = new float[samplesPerChord * chords.Length];
            for (int chordIndex = 0; chordIndex < chords.Length; chordIndex++)
            {
                float bass = chords[chordIndex][0] * 0.5f;
                for (int sampleIndex = 0; sampleIndex < samplesPerChord; sampleIndex++)
                {
                    float t = (float)sampleIndex / SAMPLE_RATE;
                    float window = Mathf.Sin(Mathf.PI * sampleIndex / (float)samplesPerChord);
                    float value = Mathf.Sin(2f * Mathf.PI * bass * t) * 0.3f;
                    for (int noteIndex = 0; noteIndex < chords[chordIndex].Length; noteIndex++)
                    {
                        float frequency = chords[chordIndex][noteIndex];
                        value += Mathf.Sin(2f * Mathf.PI * frequency * t) * 0.16f;
                        value += Mathf.Sin(2f * Mathf.PI * frequency * 1.005f * t) * 0.08f;
                    }
                    // Arpeggio: chord tones an octave up, plucked with a fast decay.
                    int arpSlot = sampleIndex / samplesPerArp;
                    float arpFrequency = chords[chordIndex][arpSlot % chords[chordIndex].Length] * 2f;
                    float arpT = (sampleIndex - arpSlot * samplesPerArp) / (float)SAMPLE_RATE;
                    value += Mathf.Sin(2f * Mathf.PI * arpFrequency * arpT) * Mathf.Exp(-14f * arpT) * 0.12f;
                    data[chordIndex * samplesPerChord + sampleIndex] = value * window * 0.5f;
                }
            }
            var clip = AudioClip.Create("MusicLoop", data.Length, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
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
                float white = (float)(rng.NextDouble() * 2.0 - 1.0);
                low += (white - low) * 0.04f;
                // Gusting wind bed: integer LFO cycles so the loop point is seamless.
                float gust = 0.55f + 0.45f * Mathf.Sin(2f * Mathf.PI * 2f * t / seconds)
                    * Mathf.Sin(2f * Mathf.PI * 5f * t / seconds);
                float value = low * gust * 1.2f;

                if (kind == TrackKind.Swamp)
                {
                    // Low bog drone plus cricket chirp bursts.
                    value = value * 0.5f
                        + 0.10f * Mathf.Sin(2f * Mathf.PI * 55f * t)
                        + 0.06f * Mathf.Sin(2f * Mathf.PI * 82.4f * t);
                    float chirpGate = Mathf.Max(0f, Mathf.Sin(2f * Mathf.PI * 3f * t / seconds) - 0.3f);
                    float chirp = Mathf.Sin(2f * Mathf.PI * 4200f * t)
                        * Mathf.Max(0f, Mathf.Sin(2f * Mathf.PI * 26f * t) - 0.6f);
                    value += chirp * chirpGate * 0.22f;
                }
                else if (kind != TrackKind.Lumpy)
                {
                    // Open field: soft breeze plus sparse bird chirps (two integer-
                    // cycle envelopes so the loop stays seamless).
                    value *= 0.6f;
                    float birdEnvelope = Mathf.Max(0f, Mathf.Sin(2f * Mathf.PI * (t / seconds + 0.13f)) - 0.93f) * 14f;
                    float birdEnvelope2 = Mathf.Max(0f, Mathf.Sin(2f * Mathf.PI * (2f * t / seconds + 0.61f)) - 0.95f) * 20f;
                    float sweep = Mathf.Sin(2f * Mathf.PI * (3000f - 600f * birdEnvelope) * t);
                    value += sweep * Mathf.Clamp01(birdEnvelope + birdEnvelope2) * 0.12f;
                }
                data[sampleIndex] = value;
            }
            var clip = AudioClip.Create($"Ambience_{kind}", samples, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
