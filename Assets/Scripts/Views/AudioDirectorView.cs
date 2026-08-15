using MessagePipe;
using PoRacer.Models;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Global race audio: a start horn when a race begins and a soft synthesized
    /// music pad looping underneath everything. All clips are generated in code —
    /// no audio asset files. Per-racer and win sounds live in CreatureAudioView
    /// and WinFxView.
    /// </summary>
    public sealed class AudioDirectorView : MonoBehaviour
    {
        private const int SAMPLE_RATE = 44100;
        private const float MUSIC_VOLUME = 0.22f;

        private AudioSource _sfxSource;
        private AudioSource _musicSource;
        private AudioClip _startHorn;
        private System.IDisposable _subscription;

        [Inject]
        public void Construct(ISubscriber<RaceStartedMessage> raceStarted)
        {
            _subscription = raceStarted.Subscribe(OnRaceStarted);
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
            _musicSource.volume = MUSIC_VOLUME;
            _musicSource.clip = SynthesizeMusicLoop();
            _musicSource.Play();

            _startHorn = SynthesizeStartHorn();
        }

        private void OnDestroy() => _subscription?.Dispose();

        private void OnRaceStarted(RaceStartedMessage message)
        {
            _sfxSource.PlayOneShot(_startHorn, 0.7f);
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

        private static AudioClip SynthesizeMusicLoop()
        {
            // Four soft pad chords, two seconds each, eight-second seamless loop.
            const float chordSeconds = 2f;
            float[][] chords =
            {
                new[] { 220f, 261.63f, 329.63f },
                new[] { 174.61f, 220f, 261.63f },
                new[] { 196f, 246.94f, 293.66f },
                new[] { 164.81f, 220f, 261.63f }
            };
            int samplesPerChord = (int)(SAMPLE_RATE * chordSeconds);
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
                    data[chordIndex * samplesPerChord + sampleIndex] = value * window * 0.5f;
                }
            }
            var clip = AudioClip.Create("MusicLoop", data.Length, 1, SAMPLE_RATE, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
