using System.Collections.Generic;
using NUnit.Framework;
using PoRacer.Models;
using PoRacer.Systems;

namespace PoRacer.Tests
{
    public sealed class Systems_RaceTests
    {
        private RaceModel _model;
        private FakePublisher<RaceFinishedMessage> _raceFinished;
        private Systems_Race _sut;

        [SetUp]
        public void SetUp()
        {
            _model = new RaceModel();
            _raceFinished = new FakePublisher<RaceFinishedMessage>();
            _sut = new Systems_Race(
                _model,
                new FakePublisher<RaceStartedMessage>(),
                new FakePublisher<RacerFinishedMessage>(),
                new FakePublisher<RacerDnfMessage>(),
                _raceFinished);
            _sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "worm#1", CreatureId = "worm", Status = RacerStatus.Racing },
                new() { RacerId = "worm#2", CreatureId = "worm", Status = RacerStatus.Racing }
            });
        }

        [Test]
        public void NotifyFinish_AssignsPlacesInOrder()
        {
            _sut.NotifyFinish("worm#2");
            _sut.NotifyFinish("worm#1");

            Assert.That(_model.FindRacer("worm#2").Place, Is.EqualTo(1));
            Assert.That(_model.FindRacer("worm#1").Place, Is.EqualTo(2));
            Assert.That(_raceFinished.Published, Has.Count.EqualTo(1));
        }

        [Test]
        public void NoProgress_MarksDnfAfterTimeout()
        {
            _sut.ReportProgress("worm#1", 5f);
            _sut.Advance(Systems_Race.NO_PROGRESS_TIMEOUT_SECONDS + 1f);

            Assert.That(_model.FindRacer("worm#1").Status, Is.EqualTo(RacerStatus.Dnf));
            Assert.That(_model.FindRacer("worm#2").Status, Is.EqualTo(RacerStatus.Dnf));
            Assert.That(_model.RaceActive, Is.False);
        }

        [Test]
        public void ProgressKeepsRacerAlive()
        {
            float half = Systems_Race.NO_PROGRESS_TIMEOUT_SECONDS * 0.6f;
            _sut.Advance(half);
            _sut.ReportProgress("worm#1", 5f);
            _sut.Advance(half);

            Assert.That(_model.FindRacer("worm#1").Status, Is.EqualTo(RacerStatus.Racing));
            Assert.That(_model.FindRacer("worm#2").Status, Is.EqualTo(RacerStatus.Dnf));
        }

        [Test]
        public void Timeout_DecidesRaceByDistance()
        {
            _sut.ReportProgress("worm#2", 8f);
            _sut.ReportProgress("worm#1", 3f);
            _sut.Advance(Systems_Race.RACE_TIMEOUT_SECONDS + 1f);

            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_raceFinished.Published, Has.Count.EqualTo(1));
            Assert.That(_model.FindRacer("worm#2").Place, Is.EqualTo(1));
            // Ranked on distance, not a crossing: the status says so and no
            // fictional finish time is stamped.
            Assert.That(_model.FindRacer("worm#2").Status, Is.EqualTo(RacerStatus.TimedOut));
            Assert.That(_model.FindRacer("worm#2").FinishTime, Is.EqualTo(0f));
            Assert.That(_model.FindRacer("worm#1").Place, Is.EqualTo(2));
        }

        [Test]
        public void AllKnockedOut_PodiumStillRanksByDistance()
        {
            _sut.ReportProgress("worm#2", 8f);
            _sut.ReportProgress("worm#1", 3f);
            _sut.NotifyFailure("worm#1");
            _sut.NotifyFailure("worm#2");

            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_model.FindRacer("worm#2").Place, Is.EqualTo(1));
            Assert.That(_model.FindRacer("worm#2").Status, Is.EqualTo(RacerStatus.Dnf));
            Assert.That(_model.FindRacer("worm#1").Place, Is.EqualTo(2));
            IReadOnlyList<RaceResultEntry> results = _raceFinished.Published[0].Results;
            for (int resultIndex = 0; resultIndex < results.Count; resultIndex++)
            {
                Assert.That(results[resultIndex].Dnf, Is.True);
            }
        }

        [Test]
        public void ThirdFinisher_EndsRaceAndScoresTheRestAsDnf()
        {
            _sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "a", CreatureId = "worm", Status = RacerStatus.Racing },
                new() { RacerId = "b", CreatureId = "spider", Status = RacerStatus.Racing },
                new() { RacerId = "c", CreatureId = "crab", Status = RacerStatus.Racing },
                new() { RacerId = "d", CreatureId = "blob", Status = RacerStatus.Racing },
                new() { RacerId = "e", CreatureId = "crab", Status = RacerStatus.Racing }
            });

            _sut.NotifyFinish("a");
            _sut.NotifyFinish("b");
            Assert.That(_model.RaceActive, Is.True);
            _sut.NotifyFinish("c");

            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_model.FindRacer("d").Status, Is.EqualTo(RacerStatus.Dnf));
            Assert.That(_model.FindRacer("e").Status, Is.EqualTo(RacerStatus.Dnf));
            IReadOnlyList<RaceResultEntry> results = _raceFinished.Published[^1].Results;
            int finishers = 0;
            for (int resultIndex = 0; resultIndex < results.Count; resultIndex++)
            {
                if (!results[resultIndex].Dnf)
                {
                    finishers++;
                }
            }
            Assert.That(finishers, Is.EqualTo(3));
        }

        [Test]
        public void OneFinisher_RemainingPodiumFilledByDistance()
        {
            _sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "a", CreatureId = "worm", Status = RacerStatus.Racing },
                new() { RacerId = "b", CreatureId = "spider", Status = RacerStatus.Racing },
                new() { RacerId = "c", CreatureId = "crab", Status = RacerStatus.Racing }
            });
            _sut.ReportProgress("b", 2f);
            _sut.ReportProgress("c", 6f);
            _sut.NotifyFinish("a");
            _sut.NotifyFailure("b");
            _sut.NotifyFailure("c");

            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_model.FindRacer("a").Place, Is.EqualTo(1));
            Assert.That(_model.FindRacer("c").Place, Is.EqualTo(2));
            Assert.That(_model.FindRacer("c").Status, Is.EqualTo(RacerStatus.Dnf));
            Assert.That(_model.FindRacer("b").Place, Is.EqualTo(3));
        }

        [Test]
        public void FinishAfterDnf_IsIgnored()
        {
            _sut.Advance(Systems_Race.NO_PROGRESS_TIMEOUT_SECONDS + 1f);
            _sut.NotifyFinish("worm#1");

            Assert.That(_model.FindRacer("worm#1").Status, Is.EqualTo(RacerStatus.Dnf));
        }

        [Test]
        public void PhotoFinish_PublishesMessageWhenMarginUnderThreshold()
        {
            var photoFinishPublisher = new FakePublisher<PhotoFinishMessage>();
            var sut = new Systems_Race(
                _model,
                new FakePublisher<RaceStartedMessage>(),
                new FakePublisher<RacerFinishedMessage>(),
                new FakePublisher<RacerDnfMessage>(),
                _raceFinished,
                new FakePublisher<RacerWipeoutMessage>(),
                photoFinishPublisher);

            sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "a", CreatureId = "worm", Status = RacerStatus.Racing },
                new() { RacerId = "b", CreatureId = "spider", Status = RacerStatus.Racing }
            });

            sut.NotifyFinish("a");
            sut.Advance(0.15f); // 0.15s margin < 0.35s threshold
            sut.NotifyFinish("b");

            Assert.That(photoFinishPublisher.Published, Has.Count.EqualTo(1));
            Assert.That(photoFinishPublisher.Published[0].WinnerId, Is.EqualTo("a"));
            Assert.That(photoFinishPublisher.Published[0].RunnerUpId, Is.EqualTo("b"));
            Assert.That(photoFinishPublisher.Published[0].MarginSeconds, Is.EqualTo(0.15f).Within(0.01f));
        }

        [Test]
        public void NotifyWipeout_PublishesWipeoutMessage()
        {
            var wipeoutPublisher = new FakePublisher<RacerWipeoutMessage>();
            var sut = new Systems_Race(
                _model,
                new FakePublisher<RaceStartedMessage>(),
                new FakePublisher<RacerFinishedMessage>(),
                new FakePublisher<RacerDnfMessage>(),
                _raceFinished,
                wipeoutPublisher);

            sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "a", CreatureId = "worm", Status = RacerStatus.Racing }
            });

            sut.NotifyWipeout("a", UnityEngine.Vector3.up * 2f, isFatal: true);

            Assert.That(wipeoutPublisher.Published, Has.Count.EqualTo(1));
            Assert.That(wipeoutPublisher.Published[0].RacerId, Is.EqualTo("a"));
            Assert.That(wipeoutPublisher.Published[0].IsFatal, Is.True);
        }
    }
}
