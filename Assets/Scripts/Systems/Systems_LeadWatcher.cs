using System;
using MessagePipe;
using PoRacer.Models;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Watches RaceModel progress and publishes <see cref="LeadChangedMessage"/>
    /// when the front runner actually changes. Systems_CameraDirector is the only
    /// consumer: it re-aims the hero shot at whoever is leading.
    ///
    /// This was Systems_Commentary, which also wrote announcer lines into a
    /// CommentaryModel for a HUD ticker. The ticker is gone, and with it every
    /// string; what is left is the lead detection the camera actually needs.
    ///
    /// The cooldown is the point of the class. Two racers a few centimetres apart
    /// swap the lead many times a second, and re-aiming on each swap makes the
    /// camera unwatchable — so a change only counts once every
    /// LEAD_CHANGE_COOLDOWN_SECONDS.
    /// </summary>
    public sealed class Systems_LeadWatcher : ITickable, IDisposable
    {
        private const float LEAD_CHANGE_COOLDOWN_SECONDS = 3f;
        // Ignores the shuffle on the start line, where everyone is near zero and
        // the "leader" is whoever twitched first.
        private const float MIN_LEAD_PROGRESS_METERS = 1f;

        private readonly RaceModel _model;
        private readonly IPublisher<LeadChangedMessage> _leadChangedPublisher;

        private string _leaderId;
        private float _lastLeadChangeSeconds;

        public Systems_LeadWatcher(RaceModel model, IPublisher<LeadChangedMessage> leadChangedPublisher)
        {
            _model = model;
            _leadChangedPublisher = leadChangedPublisher;
        }

        public void Tick()
        {
            if (!_model.RaceActive)
            {
                // A finished race must not hold the previous leader: the next race
                // reuses the model, and a stale id would suppress its first change.
                _leaderId = null;
                return;
            }

            RacerState leader = null;
            for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
            {
                RacerState racer = _model.Racers[racerIndex];
                if (racer.Status == RacerStatus.Racing && (leader == null || racer.Progress > leader.Progress))
                {
                    leader = racer;
                }
            }
            if (leader == null || leader.Progress < MIN_LEAD_PROGRESS_METERS)
            {
                return;
            }
            if (leader.RacerId == _leaderId)
            {
                return;
            }

            // The first leader of a race is published immediately. The version of
            // this that lived in Systems_Commentary only published inside the
            // cooldown branch, so the opening lead never reached the camera and the
            // hero shot stayed on the overview until the first overtake.
            bool firstLead = _leaderId == null;
            if (!firstLead && _model.ElapsedSeconds - _lastLeadChangeSeconds < LEAD_CHANGE_COOLDOWN_SECONDS)
            {
                // Inside the cooldown: track who is in front, but do not move the
                // camera for it.
                _leaderId = leader.RacerId;
                return;
            }

            _leaderId = leader.RacerId;
            _lastLeadChangeSeconds = _model.ElapsedSeconds;
            _leadChangedPublisher.Publish(new LeadChangedMessage(leader.RacerId));
        }

        public void Dispose()
        {
        }
    }
}
