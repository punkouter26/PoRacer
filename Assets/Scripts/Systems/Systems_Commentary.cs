using System;
using MessagePipe;
using PoRacer.Models;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Race announcer: turns race events (start, lead changes, finishes, DNFs)
    /// into CommentaryModel lines the HUD ticker displays. Lead changes are
    /// detected by polling RaceModel progress each Tick with a cooldown so the
    /// ticker never spams during close racing.
    /// </summary>
    public sealed class Systems_Commentary : ITickable, IDisposable
    {
        private const float LEAD_CHANGE_COOLDOWN_SECONDS = 3f;
        private const float MIN_LEAD_PROGRESS_METERS = 1f;
        private const int MAX_COMMENTED_PLACE = 3;
        // Big fields (50-100 racers) would drown the ticker in DNF lines.
        private const int MAX_RACERS_FOR_DNF_COMMENTS = 12;

        private readonly RaceModel _model;
        private readonly CommentaryModel _commentary;
        private readonly IPublisher<LeadChangedMessage> _leadChangedPublisher;
        private readonly IDisposable _subscriptions;

        private string _leaderId;
        private float _lastLeadChangeSeconds;

        public Systems_Commentary(
            RaceModel model,
            CommentaryModel commentary,
            IPublisher<LeadChangedMessage> leadChangedPublisher,
            ISubscriber<RaceStartedMessage> started,
            ISubscriber<RacerFinishedMessage> racerFinished,
            ISubscriber<RacerDnfMessage> dnf,
            ISubscriber<RaceFinishedMessage> raceFinished)
        {
            _model = model;
            _commentary = commentary;
            _leadChangedPublisher = leadChangedPublisher;
            DisposableBagBuilder bag = DisposableBag.CreateBuilder();
            started.Subscribe(OnRaceStarted).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            dnf.Subscribe(OnRacerDnf).AddTo(bag);
            raceFinished.Subscribe(OnRaceFinished).AddTo(bag);
            _subscriptions = bag.Build();
        }

        public void Tick()
        {
            if (!_model.RaceActive)
            {
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
            if (leader == null || leader.Progress < MIN_LEAD_PROGRESS_METERS || leader.RacerId == _leaderId)
            {
                return;
            }

            if (_leaderId == null)
            {
                _commentary.Add($"{ColoredName(leader)} grabs the early lead!");
            }
            else if (_model.ElapsedSeconds - _lastLeadChangeSeconds >= LEAD_CHANGE_COOLDOWN_SECONDS)
            {
                _commentary.Add($"{ColoredName(leader)} takes the lead!");
                _leadChangedPublisher.Publish(new LeadChangedMessage(leader.RacerId));
            }
            else
            {
                // Too soon after the last call: swallow the line but still track
                // the leader so the next call names the right racer.
                _leaderId = leader.RacerId;
                return;
            }
            _leaderId = leader.RacerId;
            _lastLeadChangeSeconds = _model.ElapsedSeconds;
        }

        public void Dispose() => _subscriptions.Dispose();

        private void OnRaceStarted(RaceStartedMessage message)
        {
            _leaderId = null;
            _lastLeadChangeSeconds = 0f;
            _commentary.Clear();
            _commentary.Add($"They're off! {message.RacerCount} racers charge for the finish!");
        }

        private void OnRacerFinished(RacerFinishedMessage message)
        {
            if (message.Place > MAX_COMMENTED_PLACE)
            {
                return;
            }
            RacerState racer = _model.FindRacer(message.RacerId);
            string name = racer != null ? ColoredName(racer) : message.RacerId;
            string line = message.Place switch
            {
                1 => $"{name} WINS in {message.Time:0.0}s!",
                2 => $"{name} takes second place!",
                _ => $"{name} takes third place!"
            };
            _commentary.Add(line);
        }

        private void OnRacerDnf(RacerDnfMessage message)
        {
            if (_model.Racers.Count > MAX_RACERS_FOR_DNF_COMMENTS)
            {
                return;
            }
            RacerState racer = _model.FindRacer(message.RacerId);
            _commentary.Add($"{(racer != null ? ColoredName(racer) : message.RacerId)} is out of the race!");
        }

        private void OnRaceFinished(RaceFinishedMessage message)
        {
            _commentary.Add("Race complete!");
        }

        private static string ColoredName(RacerState racer)
        {
            return string.IsNullOrEmpty(racer.TintHex)
                ? racer.DisplayName
                : $"<color=#{racer.TintHex}>{racer.DisplayName}</color>";
        }
    }
}
