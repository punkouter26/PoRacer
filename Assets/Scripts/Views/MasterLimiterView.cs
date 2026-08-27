using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Brickwall limiter on the final mix, sitting on the AudioListener so it sees
    /// everything after spatialisation.
    ///
    /// It exists because every clip in this game is synthesized and peak-matched
    /// individually: a start horn, a crowd roar, a sub-bass drop and forty limb
    /// thuds can all land on the same sample, and summed peaks clip hard where
    /// recorded material would merely be loud. An AudioMixer would carry a
    /// compressor for this; there is no mixer asset, so the DSP lives here.
    ///
    /// Design: peak-following gain reduction with a fast attack and a slow release,
    /// no lookahead. Without lookahead a single sample can still overshoot the
    /// threshold before the envelope catches it, so the output is hard-clipped at
    /// the ceiling as a last resort — inaudible on a transient, and it guarantees
    /// nothing leaves here above 1.0.
    ///
    /// OnAudioFilterRead runs on the audio thread: no allocation, no Unity API and
    /// no logging in this file. The one value published back to the main thread is
    /// a float, written by one thread and read by another for display only.
    /// </summary>
    [RequireComponent(typeof(AudioListener))]
    [DisallowMultipleComponent]
    internal sealed class MasterLimiterView : MonoBehaviour
    {
        // Level the limiter starts working at, and the hard ceiling it guarantees.
        private const float THRESHOLD = 0.82f;
        private const float CEILING = 0.99f;
        // Attack and release as per-sample smoothing coefficients at 48 kHz. Fast
        // enough to catch a horn, slow enough that the release does not pump on
        // the percussion stem.
        private const float ATTACK_COEFFICIENT = 0.35f;
        private const float RELEASE_COEFFICIENT = 0.0006f;
        // A little makeup, since the whole mix now sits under the threshold.
        private const float MAKEUP_GAIN = 1.12f;

        // Written on the audio thread, read on the main thread by the diagnostic
        // overlay. A torn read of a float would only mis-draw one frame of a
        // debug readout, so this deliberately takes no lock.
        private static float _gainReductionDb;

        /// <summary>Current gain reduction in decibels, for the telemetry overlay.</summary>
        internal static float GainReductionDb => _gainReductionDb;

        private float _envelope = 1f;

        /// <summary>Adds the limiter to the active AudioListener if it has none.</summary>
        internal static void EnsureOnListener()
        {
            AudioListener listener = FindAnyObjectByType<AudioListener>();
            if (listener == null)
            {
                return;
            }
            if (!listener.TryGetComponent(out MasterLimiterView _))
            {
                listener.gameObject.AddComponent<MasterLimiterView>();
            }
        }

        private void OnAudioFilterRead(float[] data, int channels)
        {
            float envelope = _envelope;

            for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex += channels)
            {
                // Peak across the frame's channels: the loudest channel decides the
                // reduction, so the stereo image does not shift under a hard hit.
                float peak = 0f;
                for (int channelIndex = 0; channelIndex < channels; channelIndex++)
                {
                    float sample = data[sampleIndex + channelIndex];
                    float magnitude = sample < 0f ? -sample : sample;
                    if (magnitude > peak)
                    {
                        peak = magnitude;
                    }
                }

                // Gain that would put this peak exactly on the threshold. Above the
                // threshold that is less than one; below it the limiter is open.
                float targetGain = peak > THRESHOLD ? THRESHOLD / peak : 1f;

                // Attack is a fast move down, release a slow crawl back up.
                float coefficient = targetGain < envelope ? ATTACK_COEFFICIENT : RELEASE_COEFFICIENT;
                envelope += (targetGain - envelope) * coefficient;

                float gain = envelope * MAKEUP_GAIN;
                for (int channelIndex = 0; channelIndex < channels; channelIndex++)
                {
                    float value = data[sampleIndex + channelIndex] * gain;
                    // Backstop for the overshoot a lookahead-free design allows.
                    data[sampleIndex + channelIndex] = value > CEILING ? CEILING
                        : value < -CEILING ? -CEILING : value;
                }
            }

            _envelope = envelope;
            // 20*log10(envelope), as a negative number; 0 dB means fully open.
            _gainReductionDb = envelope >= 1f ? 0f : 20f * Mathf.Log10(Mathf.Max(envelope, 0.0001f));
        }

        private void OnDisable()
        {
            _envelope = 1f;
            _gainReductionDb = 0f;
        }
    }
}
