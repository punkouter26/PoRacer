using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Central lookup for the recorded one-shots under Assets/Resources/Audio.
    /// Every caller supplies a code-synthesized fallback, so the game stays fully
    /// playable — and compiles — with an empty or missing Audio folder.
    ///
    /// Names map straight onto file names: Get("crowd_cheer") loads
    /// Resources/Audio/crowd_cheer. Hits, misses and synthesized fallbacks are all
    /// cached, so a given name costs at most one Resources.Load per session and
    /// lookups never touch the disk from a hot path.
    /// </summary>
    internal static class AudioLibrary
    {
        private const string ROOT = "Audio/";
        /// <summary>Peak that fallbacks are scaled to so mix volumes hold either way.</summary>
        private const float FALLBACK_PEAK = 0.9f;

        private static readonly Dictionary<string, AudioClip> Recorded = new();
        private static readonly Dictionary<string, AudioClip> Fallbacks = new();
        private static readonly HashSet<string> Missing = new();

        /// <summary>
        /// The recorded clip for <paramref name="clipName"/>, or null when the asset
        /// is absent. Callers that need a guaranteed clip use GetOrSynthesize.
        /// </summary>
        internal static AudioClip Get(string clipName)
        {
            // Unity's == catches entries destroyed by a domain reload, which a
            // plain dictionary hit would happily hand back as a live clip.
            if (Recorded.TryGetValue(clipName, out AudioClip cached) && cached != null)
            {
                return cached;
            }
            if (Missing.Contains(clipName))
            {
                return null;
            }
            AudioClip loaded = Resources.Load<AudioClip>(ROOT + clipName);
            if (loaded == null)
            {
                Missing.Add(clipName);
                return null;
            }
            Recorded[clipName] = loaded;
            return loaded;
        }

        /// <summary>
        /// The recorded clip for <paramref name="clipName"/>, falling back to
        /// <paramref name="synthesize"/> when the asset is missing. The fallback runs
        /// at most once per name and is peak-matched to the recorded set so the
        /// caller's mix volume means the same thing on both paths.
        /// </summary>
        internal static AudioClip GetOrSynthesize(string clipName, Func<AudioClip> synthesize)
        {
            AudioClip recorded = Get(clipName);
            if (recorded != null)
            {
                return recorded;
            }
            if (Fallbacks.TryGetValue(clipName, out AudioClip cached) && cached != null)
            {
                return cached;
            }
            AudioClip built = synthesize();
            if (built != null)
            {
                NormalizePeak(built);
                Fallbacks[clipName] = built;
            }
            return built;
        }

        /// <summary>
        /// Fills <paramref name="destination"/> with the numbered variants of a
        /// series — "footstep" loads footstep_0, footstep_1, ... — and returns how
        /// many were found. Stops at the first gap. Nothing is allocated: the caller
        /// owns the array.
        /// </summary>
        internal static int GetSeries(string prefix, AudioClip[] destination)
        {
            if (destination == null)
            {
                return 0;
            }
            int found = 0;
            for (int variantIndex = 0; variantIndex < destination.Length; variantIndex++)
            {
                AudioClip clip = Get(prefix + "_" + variantIndex);
                if (clip == null)
                {
                    break;
                }
                destination[found] = clip;
                found++;
            }
            return found;
        }

        /// <summary>
        /// Scales a synthesized clip so its loudest sample sits at FALLBACK_PEAK.
        /// Only ever runs on code-built clips at startup — recorded assets are left
        /// exactly as imported.
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
            float scale = FALLBACK_PEAK / peak;
            for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
            {
                data[sampleIndex] *= scale;
            }
            clip.SetData(data, 0);
        }
    }
}
