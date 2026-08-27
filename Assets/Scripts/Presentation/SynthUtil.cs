using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Shared DSP primitives for the game's runtime synthesis. Nothing here touches
    /// assets or the scene: these are the building blocks the audio views use to
    /// generate their clips at startup (oscillator phase, loop-edge fades, a
    /// resonant band-pass for formant work, white noise).
    ///
    /// All of it runs during clip generation only — never per frame — so the code
    /// favours clarity over micro-optimisation.
    /// </summary>
    internal static class SynthUtil
    {
        internal const float TWO_PI = 6.2831853f;

        /// <summary>
        /// Advances an oscillator phase by one sample and returns it wrapped into
        /// [0, 2PI). Wrapping keeps float precision usable over long clips, and
        /// because the wrap is a whole turn, integer harmonics of the returned phase
        /// (2x, 3x, ...) stay exactly in step with the fundamental.
        /// </summary>
        internal static float AdvancePhase(ref float phase, float frequency, int sampleRate)
        {
            phase += TWO_PI * frequency / sampleRate;
            if (phase >= TWO_PI)
            {
                phase -= TWO_PI;
            }
            return phase;
        }

        /// <summary>
        /// Linear fade in/out over the first and last <paramref name="fadeSamples"/>
        /// of a span. Used to guarantee a seamless loop point (and click-free chord
        /// joins) without the deep amplitude scoop a full sine window would impose.
        /// </summary>
        internal static float EdgeFade(int sampleIndex, int spanSamples, int fadeSamples)
        {
            if (fadeSamples <= 0)
            {
                return 1f;
            }
            int fromStart = sampleIndex;
            int toEnd = spanSamples - 1 - sampleIndex;
            int nearest = fromStart < toEnd ? fromStart : toEnd;
            if (nearest >= fadeSamples)
            {
                return 1f;
            }
            return nearest <= 0 ? 0f : nearest / (float)fadeSamples;
        }

        /// <summary>White noise in [-1, 1] from a seeded generator (deterministic clips).</summary>
        internal static float White(System.Random rng)
        {
            return (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        /// <summary>
        /// Chamberlin state-variable filter, band-pass tap. Two poles, cheap, and
        /// stable well past the frequencies used here. Held as a struct so a caller
        /// can keep several of them as plain locals — one per formant.
        /// </summary>
        internal struct BandPass
        {
            private float _low;
            private float _band;
            private readonly float _f;
            private readonly float _damping;

            internal BandPass(float centerHz, float q, int sampleRate)
            {
                _low = 0f;
                _band = 0f;
                // Clamp well below Nyquist: the SVF goes unstable as f approaches 2.
                float safeCenter = Mathf.Clamp(centerHz, 20f, sampleRate * 0.18f);
                _f = 2f * Mathf.Sin(Mathf.PI * safeCenter / sampleRate);
                _damping = 1f / Mathf.Max(q, 0.5f);
            }

            internal float Process(float input)
            {
                _low += _f * _band;
                float high = input - _low - _damping * _band;
                _band += _f * high;
                return _band;
            }
        }
    }
}
