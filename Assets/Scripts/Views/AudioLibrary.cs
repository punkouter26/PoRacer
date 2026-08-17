using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Memo table for PoRacer's code-synthesized clips. Every sound in the game is
    /// generated at runtime — there are no audio assets to import, load, or ship —
    /// so this is a name-keyed cache in front of the synth functions rather than an
    /// asset loader.
    ///
    /// A name is synthesized at most once per session and peak-matched to
    /// TARGET_PEAK, so a caller's mix volume means the same thing for every clip
    /// regardless of how hot its generator happened to run.
    /// </summary>
    internal static class AudioLibrary
    {
        /// <summary>Peak every synthesized clip is scaled to before it is cached.</summary>
        private const float TARGET_PEAK = 0.9f;

        private static readonly Dictionary<string, AudioClip> Cache = new();

        /// <summary>
        /// The clip for <paramref name="clipName"/>, building it via
        /// <paramref name="synthesize"/> the first time it is asked for.
        /// </summary>
        internal static AudioClip GetOrSynthesize(string clipName, Func<AudioClip> synthesize)
        {
            AudioClip cached = Lookup(clipName);
            if (cached != null)
            {
                return cached;
            }
            return Store(clipName, synthesize());
        }

        /// <summary>
        /// Variant-aware overload: one generator produces a numbered family (thud_0,
        /// thud_1, ...). Lets callers cache a static method group instead of
        /// allocating a closure per variant.
        /// </summary>
        internal static AudioClip GetOrSynthesize(string clipName, int variant, Func<int, AudioClip> synthesize)
        {
            AudioClip cached = Lookup(clipName);
            if (cached != null)
            {
                return cached;
            }
            return Store(clipName, synthesize(variant));
        }

        /// <summary>
        /// Cache read that respects Unity's destroyed-object semantics: a domain
        /// reload can leave a dead clip in the dictionary that a plain hit would
        /// happily hand back as live.
        /// </summary>
        private static AudioClip Lookup(string clipName)
        {
            if (Cache.TryGetValue(clipName, out AudioClip cached) && cached != null)
            {
                return cached;
            }
            return null;
        }

        private static AudioClip Store(string clipName, AudioClip built)
        {
            if (built == null)
            {
                return null;
            }
            NormalizePeak(built);
            Cache[clipName] = built;
            return built;
        }

        /// <summary>
        /// Scales a clip so its loudest sample sits at TARGET_PEAK. Only ever runs
        /// once per name, at the moment the clip is generated.
        /// </summary>
        private static void NormalizePeak(AudioClip clip)
        {
            int sampleCount = clip.samples * clip.channels;
            if (sampleCount <= 0)
            {
                return;
            }
            var data = new float[sampleCount];
            if (!clip.GetData(data, 0))
            {
                return;
            }
            float peak = 0f;
            for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
            {
                float magnitude = Mathf.Abs(data[sampleIndex]);
                if (magnitude > peak)
                {
                    peak = magnitude;
                }
            }
            if (peak <= 0.0001f)
            {
                return;
            }
            float scale = TARGET_PEAK / peak;
            for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
            {
                data[sampleIndex] *= scale;
            }
            clip.SetData(data, 0);
        }
    }
}
