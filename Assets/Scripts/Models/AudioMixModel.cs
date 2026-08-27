using System;

namespace PoRacer.Models
{
    /// <summary>The mix buses every sound in the game is routed through.</summary>
    public enum AudioBus
    {
        /// <summary>Scales every other bus. The player's overall volume.</summary>
        Master = 0,
        /// <summary>The four-stem race score.</summary>
        Music = 1,
        /// <summary>One-shots: horn, crowd, impacts, boost pads, wind.</summary>
        Sfx = 2,
        /// <summary>Per-map nature bed.</summary>
        Ambience = 3,
        /// <summary>Creature calls, which need their own trim against the crowd.</summary>
        Voice = 4
    }

    /// <summary>
    /// Bus gains for the whole mix, plus the shared duck envelope.
    ///
    /// The project synthesizes every sound at runtime and ships no AudioMixer
    /// asset, so the bus structure an AudioMixer would provide lives here instead:
    /// per-bus player gains, a duck that big moments push and time releases, and a
    /// single place that answers "how loud should this actually be".
    ///
    /// Pure C#: no Unity types, so the mix rules are testable without a scene.
    /// Duck depth is per bus on purpose — a start horn should push the music well
    /// out of the way and barely touch the crowd.
    /// </summary>
    public sealed class AudioMixModel
    {
        private const int BUS_COUNT = 5;

        /// <summary>How far a full duck pulls each bus down, as a fraction.</summary>
        private static readonly float[] DuckDepth = { 0f, 0.65f, 0.15f, 0.45f, 0.3f };

        private readonly float[] _userGains = { 1f, 1f, 1f, 1f, 1f };

        /// <summary>Current duck amount, 0 (open) to 1 (fully pushed down).</summary>
        public float Duck { get; private set; }

        /// <summary>
        /// Global attenuation applied on top of the buses, used for the softened
        /// mix while the menu is open. Smoothed by the owning system, not here.
        /// </summary>
        public float GlobalMix { get; set; } = 1f;

        public float GetUserGain(AudioBus bus) => _userGains[(int)bus];

        /// <summary>Sets a bus gain, clamped to the 0..1 range a slider produces.</summary>
        public void SetUserGain(AudioBus bus, float gain)
        {
            _userGains[(int)bus] = gain < 0f ? 0f : gain > 1f ? 1f : gain;
        }

        /// <summary>
        /// Pushes the duck to at least <paramref name="amount"/>. Raising only:
        /// a small event landing during a big one must not undo the big one.
        /// </summary>
        public void PushDuck(float amount)
        {
            if (amount > Duck)
            {
                Duck = amount > 1f ? 1f : amount;
            }
        }

        /// <summary>Releases the duck toward open over <paramref name="seconds"/>.</summary>
        public void ReleaseDuck(float deltaSeconds, float recoverySeconds)
        {
            if (Duck <= 0f)
            {
                return;
            }
            float step = deltaSeconds / Math.Max(recoverySeconds, 0.0001f);
            Duck = Duck - step < 0f ? 0f : Duck - step;
        }

        /// <summary>
        /// The multiplier a source on <paramref name="bus"/> should apply to its
        /// own design volume. Master and the global mix are folded in, so callers
        /// never have to remember to apply them twice.
        /// </summary>
        public float Gain(AudioBus bus)
        {
            int index = (int)bus;
            float duckFactor = 1f - DuckDepth[index] * Duck;
            float master = bus == AudioBus.Master ? 1f : _userGains[(int)AudioBus.Master];
            return _userGains[index] * master * duckFactor * GlobalMix;
        }

        /// <summary>Resets every bus to unity. Used by tests and by a settings reset.</summary>
        public void ResetGains()
        {
            for (int busIndex = 0; busIndex < BUS_COUNT; busIndex++)
            {
                _userGains[busIndex] = 1f;
            }
            Duck = 0f;
            GlobalMix = 1f;
        }
    }
}
