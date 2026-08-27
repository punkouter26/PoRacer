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

        private readonly Random _rng = new();
        private bool _commentedFinalStretch;

        public Systems_Commentary(
            RaceModel model,
            CommentaryModel commentary,
            IPublisher<LeadChangedMessage> leadChangedPublisher,
            ISubscriber<RaceStartedMessage> started,
            ISubscriber<RacerFinishedMessage> racerFinished,
            ISubscriber<RacerDnfMessage> dnf,
            ISubscriber<RaceFinishedMessage> raceFinished,
            ISubscriber<RacerWipeoutMessage> wipeout = null,
            ISubscriber<PhotoFinishMessage> photoFinish = null)
        {
            _model = model;
            _commentary = commentary;
            _leadChangedPublisher = leadChangedPublisher;
            DisposableBagBuilder bag = DisposableBag.CreateBuilder();
            started.Subscribe(OnRaceStarted).AddTo(bag);
            racerFinished.Subscribe(OnRacerFinished).AddTo(bag);
            dnf.Subscribe(OnRacerDnf).AddTo(bag);
            raceFinished.Subscribe(OnRaceFinished).AddTo(bag);
            wipeout?.Subscribe(OnRacerWipeout).AddTo(bag);
            photoFinish?.Subscribe(OnPhotoFinish).AddTo(bag);
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
            if (leader == null || leader.Progress < MIN_LEAD_PROGRESS_METERS)
            {
                return;
            }

            // Final stretch announcement once per race
            if (!_commentedFinalStretch && _model.TrackLengthMeters > 0f && leader.Progress >= _model.TrackLengthMeters * 0.8f)
            {
                _commentedFinalStretch = true;
                _commentary.Add($"🏁 {ColoredName(leader)} hits the final stretch! The podium is in sight!");
            }

            if (leader.RacerId == _leaderId)
            {
                return;
            }

            if (_leaderId == null)
            {
                _commentary.Add($"⚡ {ColoredName(leader)} grabs the early lead!");
            }
            else if (_model.ElapsedSeconds - _lastLeadChangeSeconds >= LEAD_CHANGE_COOLDOWN_SECONDS)
            {
                string[] leadLines =
                {
                    $"⚡ {ColoredName(leader)} surges into the lead!",
                    $"🔥 {ColoredName(leader)} makes a bold move and takes 1st!",
                    $"🚀 {ColoredName(leader)} hits the front with blistering pace!"
                };
                _commentary.Add(leadLines[_rng.Next(leadLines.Length)]);
                _leadChangedPublisher.Publish(new LeadChangedMessage(leader.RacerId));
            }
            else
            {
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
            _commentedFinalStretch = false;
            _commentary.Clear();
            _commentary.Add($"🟢 They're off! {message.RacerCount} racers charge for the finish!");
        }

        private void OnRacerWipeout(RacerWipeoutMessage message)
        {
            if (_model.Racers.Count > MAX_RACERS_FOR_DNF_COMMENTS)
            {
                return;
            }
            RacerState racer = _model.FindRacer(message.RacerId);
            string name = racer != null ? ColoredName(racer) : message.RacerId;
            if (message.IsFatal)
            {
                _commentary.Add($"🚨 Catastrophic wipeout! {name} is knocked out!");
            }
            else
            {
                _commentary.Add($"💥 Big tumble for {name}! Marshal scrambling to assist!");
            }
        }

        private void OnPhotoFinish(PhotoFinishMessage message)
        {
            RacerState winner = _model.FindRacer(message.WinnerId);
            RacerState runnerUp = _model.FindRacer(message.RunnerUpId);
            string winName = winner != null ? ColoredName(winner) : message.WinnerId;
            string runName = runnerUp != null ? ColoredName(runnerUp) : message.RunnerUpId;
            _commentary.Add($"📸 INCREDIBLE PHOTO FINISH! {winName} edges {runName} by just {message.MarginSeconds:0.00}s!");
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
                1 => $"🏆 {name} WINS in {message.Time:0.0}s!",
                2 => $"🥈 {name} takes second place!",
                _ => $"🥉 {name} takes third place!"
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
            _commentary.Add("🏁 Race complete! What a battle!");
        }

        private static string ColoredName(RacerState racer)
        {
            return string.IsNullOrEmpty(racer.TintHex)
                ? racer.DisplayName
                : $"<color=#{racer.TintHex}>{racer.DisplayName}</color>";
        }
    }
}
