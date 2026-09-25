using System;
using System.Collections.Generic;
using MessagePipe;
using PoRacer.Models;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Chooses the sky. In the menu it shows the selected map's default preset, so
    /// the backdrop previews the map being picked. Once a track is built it rolls a
    /// preset for the kind actually built, never repeating the one just raced when
    /// there is another to choose from, so back-to-back races on one map still
    /// look different.
    /// </summary>
    public sealed class Systems_Sky : IStartable, IDisposable
    {
        private readonly SkyCatalog _catalog;
        private readonly SkyModel _model;
        private readonly RaceConfigModel _config;
        private readonly IDisposable _subscription;
        private readonly List<SkyPreset> _pool = new();
        private readonly Random _rng = new();

        public Systems_Sky(SkyCatalog catalog, SkyModel model, RaceConfigModel config,
            ISubscriber<TrackBuiltMessage> trackBuilt)
        {
            _catalog = catalog;
            _model = model;
            _config = config;
            _subscription = trackBuilt.Subscribe(OnTrackBuilt);
        }

        public void Start()
        {
            _config.Changed += OnConfigChanged;
            OnConfigChanged();
        }

        public void Dispose()
        {
            _config.Changed -= OnConfigChanged;
            _subscription.Dispose();
        }

        /// <summary>
        /// Index of the next preset in a pool of <paramref name="poolCount"/>,
        /// avoiding <paramref name="currentIndex"/> whenever the pool has another.
        /// Public for the EditMode tests.
        /// </summary>
        public static int PickIndex(int poolCount, int currentIndex, Random rng)
        {
            if (poolCount <= 1)
            {
                return 0;
            }
            if (currentIndex < 0 || currentIndex >= poolCount)
            {
                return rng.Next(poolCount);
            }
            // Draw from the others only: shift past the current slot.
            int pick = rng.Next(poolCount - 1);
            return pick >= currentIndex ? pick + 1 : pick;
        }

        private void OnConfigChanged()
        {
            if (!_config.MenuVisible || _catalog == null)
            {
                return;
            }
            _catalog.CollectFor(Systems_MapCatalog.Get(_config.SelectedMapIndex).Kind, _pool);
            if (_pool.Count > 0)
            {
                _model.Set(_pool[0]);
            }
        }

        private void OnTrackBuilt(TrackBuiltMessage message)
        {
            if (_catalog == null)
            {
                return;
            }
            _catalog.CollectFor(message.Kind, _pool);
            if (_pool.Count == 0)
            {
                return;
            }
            int current = _pool.IndexOf(_model.Current);
            _model.Set(_pool[PickIndex(_pool.Count, current, _rng)]);
        }
    }
}
