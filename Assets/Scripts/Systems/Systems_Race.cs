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
        public const float NO_PROGRESS_TIMEOUT_SECONDS = 30f;
        public const float RACE_TIMEOUT_SECONDS = 120f;
        // The race is decided at the podium: once this many racers finish, the
        // rest are scored as non-finishers and the race ends immediately.
        public const int PODIUM_FINISHERS = 3;
        private const float MIN_PROGRESS_DELTA = 0.05f;

        private readonly RaceModel _model;
        private readonly IPublisher<RaceStartedMessage> _startedPublisher;
        private readonly IPublisher<RacerFinishedMessage> _racerFinishedPublisher;
        private readonly IPublisher<RacerDnfMessage> _dnfPublisher;
        private readonly IPublisher<RaceFinishedMessage> _raceFinishedPublisher;

        private readonly Dictionary<string, float> _lastProgress = new();
        private readonly Dictionary<string, float> _lastProgressTime = new();
        // Finishers sharing one ElapsedSeconds stamp (same frame): their places are
        // re-ranked by overshoot, since trigger callback order is arbitrary.
        private readonly List<RacerState> _sameTimeFinishers = new();
        private float _sameTimeStamp = -1f;
        private int _nextPlace;

        private readonly IPublisher<RacerWipeoutMessage> _wipeoutPublisher;
        private readonly IPublisher<PhotoFinishMessage> _photoFinishPublisher;
        private float _firstPlaceTime = -1f;
        private string _firstPlaceId;

        public Systems_Race(
            RaceModel model,
            IPublisher<RaceStartedMessage> startedPublisher,
            IPublisher<RacerFinishedMessage> racerFinishedPublisher,
            IPublisher<RacerDnfMessage> dnfPublisher,
            IPublisher<RaceFinishedMessage> raceFinishedPublisher,
            IPublisher<RacerWipeoutMessage> wipeoutPublisher = null,
            IPublisher<PhotoFinishMessage> photoFinishPublisher = null)
        {
            _model = model;
            _startedPublisher = startedPublisher;
            _racerFinishedPublisher = racerFinishedPublisher;
            _dnfPublisher = dnfPublisher;
            _raceFinishedPublisher = raceFinishedPublisher;
            _wipeoutPublisher = wipeoutPublisher;
            _photoFinishPublisher = photoFinishPublisher;
        }

        // Unscaled: the winner slow-mo (CameraFxView) must not stretch the race
        // clock or the DNF timers.
        public void Tick() => Advance(UnityEngine.Time.unscaledDeltaTime);

        public void StartRace(IReadOnlyList<RacerState> racers)
        {
            _model.SetRacers(racers);
            _lastProgress.Clear();
            _lastProgressTime.Clear();
            _sameTimeFinishers.Clear();
            _sameTimeStamp = -1f;
            _nextPlace = 1;
            for (int racerIndex = 0; racerIndex < racers.Count; racerIndex++)
            {
                // Negative infinity: progress is measured from the common start
                // line, so back-row racers begin below zero. The first report
                // sets the real baseline for the stall timer.
                _lastProgress[racers[racerIndex].RacerId] = float.NegativeInfinity;
                _lastProgressTime[racers[racerIndex].RacerId] = 0f;
            }
            _model.ElapsedSeconds = 0f;
            _model.RaceActive = true;
            _model.RaceNumber++;
            _firstPlaceTime = -1f;
            _firstPlaceId = null;
            _startedPublisher.Publish(new RaceStartedMessage(racers.Count));
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

        public void NotifyWipeout(string racerId, UnityEngine.Vector3 position, bool isFatal)
        {
            _wipeoutPublisher?.Publish(new RacerWipeoutMessage(racerId, position, isFatal));
        }

        public void NotifyFinish(string racerId, float overshootMeters = 0f)
        {
            RacerState racer = _model.FindRacer(racerId);
            if (racer == null || racer.Status != RacerStatus.Racing)
            {
                return;
            }
            racer.Status = RacerStatus.Finished;
            racer.Place = _nextPlace++;
            racer.FinishTime = _model.ElapsedSeconds;
            racer.FinishOvershoot = overshootMeters;
            ResolveSameFrameTie(racer);
            _racerFinishedPublisher.Publish(new RacerFinishedMessage(racerId, racer.Place, racer.FinishTime));

            if (racer.Place == 1)
            {
                _firstPlaceTime = racer.FinishTime;
                _firstPlaceId = racer.RacerId;
            }
            else if (racer.Place == 2 && _firstPlaceId != null && _firstPlaceTime >= 0f)
            {
                float margin = racer.FinishTime - _firstPlaceTime;
                if (margin <= 0.35f)
                {
                    _photoFinishPublisher?.Publish(new PhotoFinishMessage(_firstPlaceId, racer.RacerId, margin));
                }
            }

            // Podium cutoff: with the top places decided there is nothing left to
            // win. Everyone still racing is scored as a non-finisher — quietly, no
            // per-racer DNF fanfare for a whole field at once — and ELO then
            // scores the finishers above them all.
            int podiumLimit = Math.Min(PODIUM_FINISHERS, _model.Racers.Count);
            if (_nextPlace - 1 >= podiumLimit)
            {
                for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
                {
                    RacerState remaining = _model.Racers[racerIndex];
                    if (remaining.Status == RacerStatus.Racing)
                    {
                        remaining.Status = RacerStatus.Dnf;
                        remaining.Place = -1;
                    }
                }
            }
            CheckRaceEnd();
        }

        /// <summary>
        /// Finishers reported within one frame share a FinishTime; callback order is
        /// engine-arbitrary. Re-rank that group by overshoot: whoever is farther
        /// past the line crossed it earlier in the frame.
        /// </summary>
        private void ResolveSameFrameTie(RacerState racer)
        {
            if (racer.FinishTime != _sameTimeStamp)
            {
                _sameTimeFinishers.Clear();
                _sameTimeStamp = racer.FinishTime;
            }
            _sameTimeFinishers.Add(racer);
            if (_sameTimeFinishers.Count < 2)
            {
                return;
            }
            // The group's places are consecutive by construction (DNFs take no place).
            _sameTimeFinishers.Sort(
                (first, second) => second.FinishOvershoot.CompareTo(first.FinishOvershoot));
            int firstPlace = _nextPlace - _sameTimeFinishers.Count;
            for (int tieIndex = 0; tieIndex < _sameTimeFinishers.Count; tieIndex++)
            {
                _sameTimeFinishers[tieIndex].Place = firstPlace + tieIndex;
            }
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
            _lastProgressTime.Clear();
            _sameTimeFinishers.Clear();
            _sameTimeStamp = -1f;
        }

        public void Advance(float deltaSeconds)
        {
            if (!_model.RaceActive)
            {
                return;
            }
            _model.ElapsedSeconds += deltaSeconds;

            if (_model.ElapsedSeconds >= RACE_TIMEOUT_SECONDS)
            {
                // Full time: the race is decided by distance instead of ending
                // with an empty podium. Everyone still racing is ranked by how far
                // they got, but as TimedOut rather than Finished - none of them
                // crossed, so none of them owns a finish time.
                RankByDistanceAtTimeout();
            }
            else
            {
                for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
                {
                    RacerState racer = _model.Racers[racerIndex];
                    if (racer.Status != RacerStatus.Racing)
                    {
                        continue;
                    }
                    if (_model.ElapsedSeconds - _lastProgressTime[racer.RacerId] >= NO_PROGRESS_TIMEOUT_SECONDS)
                    {
                        MarkDnf(racer);
                    }
                }
            }
            CheckRaceEnd();
        }

        /// <summary>
        /// Timeout referee: ranks every still-racing racer by how far it got and
        /// hands out the remaining places in that order.
        ///
        /// These racers are TimedOut, not Finished: the placing is real but the
        /// crossing never happened, so FinishTime stays unset and the HUD prints
        /// distance instead of a fictional lap time. Only the podium publishes a
        /// RacerFinishedMessage - firing one per racer would set off the whole
        /// win fanfare (flash, slow-mo, confetti, stinger) once per entrant in a
        /// single frame.
        /// </summary>
        private void RankByDistanceAtTimeout()
        {
            while (true)
            {
                RacerState farthest = null;
                for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
                {
                    RacerState racer = _model.Racers[racerIndex];
                    if (racer.Status == RacerStatus.Racing
                        && (farthest == null || racer.Progress > farthest.Progress))
                    {
                        farthest = racer;
                    }
                }
                if (farthest == null)
                {
                    return;
                }
                farthest.Status = RacerStatus.TimedOut;
                farthest.Place = _nextPlace++;
                farthest.FinishOvershoot = farthest.Progress;
                if (farthest.Place <= PODIUM_FINISHERS)
                {
                    _racerFinishedPublisher.Publish(
                        new RacerFinishedMessage(farthest.RacerId, farthest.Place, farthest.FinishTime));
                }
            }
        }

        public void Dispose() { }

        private void MarkDnf(RacerState racer)
        {
            if (racer == null || racer.Status != RacerStatus.Racing)
            {
                return;
            }
            racer.Status = RacerStatus.Dnf;
            racer.Place = -1;
            _dnfPublisher.Publish(new RacerDnfMessage(racer.RacerId));
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
            RankLeftoverDnfs();
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

        /// <summary>
        /// Fills any podium places still empty at race end (fewer than 3 crossed,
        /// or nobody did) with the knocked-out racers who got farthest. They keep
        /// DNF status, so the HUD marks them as such and ELO still scores them as
        /// non-finishers.
        /// </summary>
        private void RankLeftoverDnfs()
        {
            int podiumLimit = Math.Min(PODIUM_FINISHERS, _model.Racers.Count);
            for (int place = _nextPlace; place <= podiumLimit; place++)
            {
                RacerState farthest = null;
                for (int racerIndex = 0; racerIndex < _model.Racers.Count; racerIndex++)
                {
                    RacerState racer = _model.Racers[racerIndex];
                    if (racer.Status == RacerStatus.Dnf && racer.Place <= 0
                        && (farthest == null || racer.Progress > farthest.Progress))
                    {
                        farthest = racer;
                    }
                }
                if (farthest == null)
                {
                    return;
                }
                farthest.Place = place;
            }
        }
    }
}
