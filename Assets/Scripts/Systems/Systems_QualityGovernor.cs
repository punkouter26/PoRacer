using PoRacer.Models;
using Unity.MLAgents;
using UnityEngine;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Steps the rendering quality down when the frame rate cannot hold, and back
    /// up once it has room again. Decides the tier only; PostFxView applies it.
    ///
    /// The measure is the SHARE of slow frames in a two-second window, not the
    /// average. A spawn or a track build costs one enormous frame, and an average
    /// would read that single stall as a slow phone and throw the quality away for
    /// the rest of the session. A share counts it as one frame out of a hundred.
    /// Frames past <see cref="STALL_SECONDS"/> are left out altogether for the same
    /// reason: they are loading, not rendering.
    ///
    /// Dropping is quick (two bad windows, four seconds) because a stuttering race
    /// is the thing being prevented. Rising is slow (ten clean windows) and is
    /// abandoned for the session if the tier it rose to fails again within
    /// <see cref="RETRY_GRACE_WINDOWS"/>: bouncing between two tiers every few
    /// seconds looks worse than either of them.
    ///
    /// Only race frames are measured. The menu renders a nearly empty scene, and
    /// clean menu windows would climb the tier straight back up before every race.
    /// </summary>
    public sealed class Systems_QualityGovernor : IStartable, ITickable
    {
        private const float WINDOW_SECONDS = 2f;
        // 45 fps: the point below which a 60 Hz race visibly judders.
        private const float SLOW_FRAME_SECONDS = 1f / 45f;
        private const float STALL_SECONDS = 0.25f;
        private const float DROP_SLOW_SHARE = 0.5f;
        private const int DROP_WINDOWS = 2;
        private const float RISE_SLOW_SHARE = 0.05f;
        private const int RISE_WINDOWS = 10;
        private const int RETRY_GRACE_WINDOWS = 15;

        private readonly QualityModel _model;
        private readonly RaceConfigModel _config;
        private bool _active;
        private float _windowSeconds;
        private int _windowFrames;
        private int _windowSlowFrames;
        private int _badWindows;
        private int _goodWindows;
        // Best tier the governor may still climb to this session.
        private int _ceiling = QualityModel.TIER_HIGH;
        private int _windowsSinceRise = int.MaxValue;

        public Systems_QualityGovernor(QualityModel model, RaceConfigModel config)
        {
            _model = model;
            _config = config;
        }

        public void Start()
        {
            // A trainer-driven build runs at whatever rate the trainer sets; that
            // is not a slow phone, and nothing it renders is being watched.
            _active = !(Academy.IsInitialized && Academy.Instance.IsCommunicatorOn);
        }

        public void Tick()
        {
            if (!_active || _config.MenuVisible)
            {
                return;
            }
            Sample(Time.unscaledDeltaTime);
        }

        /// <summary>Feeds one frame's duration. Public for the EditMode tests.</summary>
        public void Sample(float frameSeconds)
        {
            if (frameSeconds <= 0f || frameSeconds > STALL_SECONDS)
            {
                return;
            }
            _windowSeconds += frameSeconds;
            _windowFrames++;
            if (frameSeconds > SLOW_FRAME_SECONDS)
            {
                _windowSlowFrames++;
            }
            if (_windowSeconds < WINDOW_SECONDS)
            {
                return;
            }
            CloseWindow((float)_windowSlowFrames / _windowFrames);
            _windowSeconds = 0f;
            _windowFrames = 0;
            _windowSlowFrames = 0;
        }

        private void CloseWindow(float slowShare)
        {
            if (_windowsSinceRise < int.MaxValue)
            {
                _windowsSinceRise++;
            }
            if (slowShare >= DROP_SLOW_SHARE)
            {
                _goodWindows = 0;
                _badWindows++;
                if (_badWindows >= DROP_WINDOWS)
                {
                    _badWindows = 0;
                    StepDown();
                }
                return;
            }
            _badWindows = 0;
            if (slowShare > RISE_SLOW_SHARE)
            {
                _goodWindows = 0;
                return;
            }
            _goodWindows++;
            if (_goodWindows >= RISE_WINDOWS)
            {
                _goodWindows = 0;
                StepUp();
            }
        }

        private void StepDown()
        {
            if (_model.Tier >= QualityModel.TIER_MINIMUM)
            {
                return;
            }
            if (_windowsSinceRise <= RETRY_GRACE_WINDOWS)
            {
                // The tier just climbed to could not hold. Stop trying it.
                _ceiling = _model.Tier + 1;
            }
            // Only the drop straight after a rise says anything about that rise; a
            // later one is just a heavier race and must not lower the ceiling.
            _windowsSinceRise = int.MaxValue;
            _model.SetTier(_model.Tier + 1);
        }

        private void StepUp()
        {
            if (_model.Tier <= _ceiling)
            {
                return;
            }
            _model.SetTier(_model.Tier - 1);
            _windowsSinceRise = 0;
        }
    }
}
