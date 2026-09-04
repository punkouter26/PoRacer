namespace PoRacer.Models
{
    /// <summary>
    /// The mix buses every sound in the game is routed through. Music, Ambience and
    /// Voice were removed along with the score, the nature bed and the creature
    /// calls: the only sounds left are the creatures' footfalls and collisions, and
    /// a bus nothing can route to is a bus that lies about what the game plays.
    /// </summary>
    public enum AudioBus
    {
        /// <summary>Scales every other bus. The player's overall volume.</summary>
        Master = 0,
        /// <summary>Creature footfalls and body-on-body collisions.</summary>
        Sfx = 1
    }

    /// <summary>
    /// Bus gains for the whole mix.
    ///
    /// The project synthesizes every sound at runtime and ships no AudioMixer
    /// asset, so the bus structure an AudioMixer would provide lives here instead:
    /// per-bus player gains and a single place that answers "how loud should this
    /// actually be".
    ///
    /// The duck envelope was removed with the music it existed to push out of the
    /// way — the start horn and crowd reactions were its only callers.
    ///
    /// Pure C#: no Unity types, so the mix rules are testable without a scene.
    /// </summary>
    public sealed class AudioMixModel
    {
        private const int BUS_COUNT = 2;

        private readonly float[] _userGains = { 1f, 1f };

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
        /// The multiplier a source on <paramref name="bus"/> should apply to its
        /// own design volume. Master and the global mix are folded in, so callers
        /// never have to remember to apply them twice.
        /// </summary>
        public float Gain(AudioBus bus)
        {
            int index = (int)bus;
            float master = bus == AudioBus.Master ? 1f : _userGains[(int)AudioBus.Master];
            return _userGains[index] * master * GlobalMix;
        }

    }
}
