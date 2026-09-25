using System;
using PoRacer.Models;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Loads the vibration setting at startup and saves it whenever it changes.
    /// A single on/off preference, so PlayerPrefs rather than a JSON file.
    /// </summary>
    public sealed class Systems_Haptics : IStartable, IDisposable
    {
        private const string PREF_KEY = "haptics_enabled";

        private readonly HapticsModel _model;

        public Systems_Haptics(HapticsModel model)
        {
            _model = model;
        }

        public void Start()
        {
            _model.SetEnabled(PlayerPrefs.GetInt(PREF_KEY, 1) != 0);
            _model.Changed += Save;
        }

        public void Dispose()
        {
            _model.Changed -= Save;
        }

        /// <summary>Menu toggle.</summary>
        public void Toggle()
        {
            _model.SetEnabled(!_model.Enabled);
        }

        private void Save()
        {
            PlayerPrefs.SetInt(PREF_KEY, _model.Enabled ? 1 : 0);
            PlayerPrefs.Save();
        }
    }
}
