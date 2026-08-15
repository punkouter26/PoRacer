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
        public void AllDnf_EndsRaceWithNoWinner()
        {
            _sut.Advance(Systems_Race.RACE_TIMEOUT_SECONDS + 1f);

            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_raceFinished.Published, Has.Count.EqualTo(1));
            IReadOnlyList<RaceResultEntry> results = _raceFinished.Published[0].Results;
            for (int resultIndex = 0; resultIndex < results.Count; resultIndex++)
            {
                Assert.That(results[resultIndex].Dnf, Is.True);
            }
        }

        [Test]
        public void FinishAfterDnf_IsIgnored()
        {
            _sut.Advance(Systems_Race.RACE_TIMEOUT_SECONDS + 1f);
            _sut.NotifyFinish("worm#1");

            Assert.That(_model.FindRacer("worm#1").Status, Is.EqualTo(RacerStatus.Dnf));
        }
    }
}
