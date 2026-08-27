using System;
using PoRacer.Models;
using VContainer;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Owns <see cref="AudioMixModel"/> and runs its envelopes: the shared duck
    /// releases on a timer, and the global mix slides between the softened menu
    /// level and the full race level instead of stepping.
    ///
    /// Every audio view multiplies its own design volume by <see cref="Gain"/>,
    /// which is what an AudioMixer group would do for sources routed into it. The
    /// difference is that this runs against code-synthesized clips with no asset
    /// in the project, and stays unit-testable.
    ///
    /// Ticked on unscaled time: the winner slow-motion must not stretch the
    /// duck release into a hole in the mix.
    /// </summary>
    public sealed class Systems_AudioMix : ITickable, IDisposable
    {
        private const float DUCK_RECOVERY_SECONDS = 1.2f;
        // Level the whole mix drops to while the menu is open.
        private const float MENU_MIX = 0.5f;
        private const float MENU_MIX_RATE = 1.5f;

        private readonly AudioMixModel _model;
        private readonly RaceConfigModel _config;

        [Inject]
        public Systems_AudioMix(AudioMixModel model, RaceConfigModel config)
        {
            _model = model;
            _config = config;
        }

        public void Tick() => Advance(UnityEngine.Time.unscaledDeltaTime);

        /// <summary>Separated from Tick so the envelopes can be driven in tests.</summary>
        public void Advance(float deltaSeconds)
        {
            _model.ReleaseDuck(deltaSeconds, DUCK_RECOVERY_SECONDS);

            float target = _config != null && _config.MenuVisible ? MENU_MIX : 1f;
            float current = _model.GlobalMix;
            float step = deltaSeconds * MENU_MIX_RATE;
            if (current < target)
            {
                current = current + step > target ? target : current + step;
            }
            else if (current > target)
            {
                current = current - step < target ? target : current - step;
            }
            _model.GlobalMix = current;
        }

        /// <summary>The multiplier a source on <paramref name="bus"/> should apply.</summary>
        public float Gain(AudioBus bus) => _model.Gain(bus);

        /// <summary>
        /// Ducks the mix for a moment worth hearing over. Callers pass how big the
        /// moment is; the model keeps the largest push currently in flight.
        /// </summary>
        public void Duck(float amount) => _model.PushDuck(amount);

        public void SetUserGain(AudioBus bus, float gain) => _model.SetUserGain(bus, gain);

        public float GetUserGain(AudioBus bus) => _model.GetUserGain(bus);

        public void Dispose() { }
    }
}
