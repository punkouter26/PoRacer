using System;
using System.Collections.Generic;
using MessagePipe;
using PoRacer.Models;
using VContainer.Unity;

namespace PoRacer.Systems
{
    /// <summary>
    /// Referee: owns RaceModel, tracks progress, finish order, DNF timeouts.
    /// Advance(dt) is separated from Tick so tests can drive time explicitly.
    /// </summary>
    public sealed class Systems_Race : ITickable, IDisposable
    {
        public const float NO_PROGRESS_TIMEOUT_SECONDS = 20f;
        public const float RACE_TIMEOUT_SECONDS = 120f;
        private const float MIN_PROGRESS_DELTA = 0.05f;

        private readonly RaceModel _model;
        private readonly IPublisher<RaceStartedMessage> _startedPublisher;
        private readonly IPublisher<RacerFinishedMessage> _racerFinishedPublisher;
        private readonly IPublisher<RacerDnfMessage> _dnfPublisher;
        private readonly IPublisher<RaceFinishedMessage> _raceFinishedPublisher;

        private readonly Dictionary<string, float> _lastProgress = new();
        private readonly Dictionary<string, float> _lastProgressTime = new();
        private int _nextPlace;

        public Systems_Race(
            RaceModel model,
            IPublisher<RaceStartedMessage> startedPublisher,
            IPublisher<RacerFinishedMessage> racerFinishedPublisher,
            IPublisher<RacerDnfMessage> dnfPublisher,
            IPublisher<RaceFinishedMessage> raceFinishedPublisher)
        {
            _model = model;
            _startedPublisher = startedPublisher;
            _racerFinishedPublisher = racerFinishedPublisher;
            _dnfPublisher = dnfPublisher;
            _raceFinishedPublisher = raceFinishedPublisher;
        }

        public void Tick() => Advance(UnityEngine.Time.deltaTime);

        public void StartRace(IReadOnlyList<RacerState> racers)
        {
            _model.SetRacers(racers);
            _lastProgress.Clear();
            _lastProgressTime.Clear();
            _nextPlace = 1;
            for (int racerIndex = 0; racerIndex < racers.Count; racerIndex++)
            {
                _lastProgress[racers[racerIndex].RacerId] = 0f;
                _lastProgressTime[racers[racerIndex].RacerId] = 0f;
            }
            _model.ElapsedSeconds = 0f;
            _model.RaceActive = true;
            _model.RaceNumber++;            _startedPublisher.Publish(new RaceStartedMessage(racers.Count));
        }

        public void ReportProgress(string racerId, float progressMeters)
        {
            RacerState racer = _model.FindRacer(racerId);
            if (racer == null || racer.Status != RacerStatus.Racing)
            {
                return;
            }
            racer.Progress = progressMeters;
            if (progressMeters > _lastProgress[racerId] + MIN_PROGRESS_DELTA)
            {
                _lastProgress[racerId] = progressMeters;
                _lastProgressTime[racerId] = _model.ElapsedSeconds;
            }
        }

        public void NotifyFinish(string racerId)
        {
            RacerState racer = _model.FindRacer(racerId);
            if (racer == null || racer.Status != RacerStatus.Racing)
            {
                return;
            }
            racer.Status = RacerStatus.Finished;
            racer.Place = _nextPlace++;
            racer.FinishTime = _model.ElapsedSeconds;            _racerFinishedPublisher.Publish(new RacerFinishedMessage(racerId, racer.Place, racer.FinishTime));
            CheckRaceEnd();
        }

        public void NotifyFailure(string racerId)
        {
            MarkDnf(_model.FindRacer(racerId));
            CheckRaceEnd();
        }

        /// <summary>Stops the race silently (no RaceFinishedMessage) when returning to the menu.</summary>
        public void AbortRace()
        {
            _model.RaceActive = false;
            _model.ClearRacers();
            _lastProgress.Clear();
            _lastProgressTime.Clear();        }

        public void Advance(float deltaSeconds)
        {
            if (!_model.RaceActive)
            {
                return;
            }
            _model.ElapsedSeconds += deltaSeconds;

            bool timedOut = _model.ElapsedSeconds >= RACE_TIMEOUT_SECONDS;
            for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
            {
                RacerState racer = _model.Racers[racerIndex];
                if (racer.Status != RacerStatus.Racing)
                {
                    continue;
                }
                if (timedOut || _model.ElapsedSeconds - _lastProgressTime[racer.RacerId] >= NO_PROGRESS_TIMEOUT_SECONDS)
                {
                    MarkDnf(racer);
                }
            }
            CheckRaceEnd();
        }

        public void Dispose() { }

        private void MarkDnf(RacerState racer)
        {
            if (racer == null || racer.Status != RacerStatus.Racing)
            {
                return;
            }
            racer.Status = RacerStatus.Dnf;
            racer.Place = -1;            _dnfPublisher.Publish(new RacerDnfMessage(racer.RacerId));
        }

        private void CheckRaceEnd()
        {
            if (!_model.RaceActive)
            {
                return;
            }
            for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
            {
                if (_model.Racers[racerIndex].Status == RacerStatus.Racing)
                {
                    return;
                }
            }
            _model.RaceActive = false;
            var results = new List<RaceResultEntry>(_model.Racers.Count);
            for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
            {
                RacerState racer = _model.Racers[racerIndex];
                results.Add(new RaceResultEntry(
                    racer.RacerId,
                    racer.CreatureId,
                    racer.Place,
                    racer.FinishTime,
                    racer.Status == RacerStatus.Dnf));
            }
            _raceFinishedPublisher.Publish(new RaceFinishedMessage(results));
        }
    }
}
