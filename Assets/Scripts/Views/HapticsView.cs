using Lofelt.NiceVibrations;
using MessagePipe;
using PoRacer.Models;
using UnityEngine;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Phone vibration on the moments that matter: the start, a new leader, the
    /// racer on screen going down, a photo finish and the win. Touch instead of
    /// sound, which keeps the audio mix to footfalls and collisions as decided.
    ///
    /// Falls only buzz for the racer the broadcast director is showing. A field of
    /// a hundred has somebody falling every second, and a buzz for something the
    /// viewer cannot see is noise. Lead changes carry a cooldown for the same
    /// reason: a tight pack can swap the lead several times a second.
    ///
    /// Android and the editor only; Nice Vibrations' playback call is a no-op on a
    /// device without haptics, so there is nothing to guard here.
    /// </summary>
    public sealed class HapticsView : MonoBehaviour
    {
        private const float LEAD_COOLDOWN_SECONDS = 1.5f;

        private HapticsModel _model;
        private System.IDisposable _subscriptions;
        private string _subjectId;
        private float _nextLeadBuzzTime;

        [Inject]
        public void Construct(
            HapticsModel model,
            ISubscriber<RaceStartedMessage> raceStarted,
            ISubscriber<LeadChangedMessage> leadChanged,
            ISubscriber<RacerFinishedMessage> racerFinished,
            ISubscriber<RacerWipeoutMessage> racerWipeout,
            ISubscriber<PhotoFinishMessage> photoFinish,
            ISubscriber<CameraShotChangedMessage> shotChanged)
        {
            _model = model;
            var bag = DisposableBag.CreateBuilder();
            raceStarted.Subscribe(_ => Buzz(HapticPatterns.PresetType.MediumImpact)).AddTo(bag);
            leadChanged.Subscribe(OnLeadChanged).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            racerWipeout.Subscribe(OnRacerWipeout).AddTo(bag);
            photoFinish.Subscribe(_ => Buzz(HapticPatterns.PresetType.RigidImpact)).AddTo(bag);
            shotChanged.Subscribe(OnShotChanged).AddTo(bag);
            _subscriptions = bag.Build();
        }

        private void OnDestroy()
        {
            _subscriptions?.Dispose();
        }

        private void OnShotChanged(CameraShotChangedMessage message)
        {
            _subjectId = message.SubjectId;
        }

        private void OnLeadChanged(LeadChangedMessage message)
        {
            if (Time.unscaledTime < _nextLeadBuzzTime)
            {
                return;
            }
            _nextLeadBuzzTime = Time.unscaledTime + LEAD_COOLDOWN_SECONDS;
            Buzz(HapticPatterns.PresetType.LightImpact);
        }

        private void OnRacerWipeout(RacerWipeoutMessage message)
        {
            if (string.IsNullOrEmpty(_subjectId) || message.RacerId != _subjectId)
            {
                return;
            }
            Buzz(message.IsFatal
                ? HapticPatterns.PresetType.HeavyImpact
                : HapticPatterns.PresetType.MediumImpact);
        }

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place == 1)
            {
                Buzz(HapticPatterns.PresetType.Success);
            }
        }

        private void Buzz(HapticPatterns.PresetType preset)
        {
            if (_model == null || !_model.Enabled)
            {
                return;
            }
            HapticPatterns.PlayPreset(preset);
        }
    }
}
