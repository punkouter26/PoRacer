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
        private FakePublisher<RacerFinishedMessage> _racerFinished;
        private Systems_Race _sut;

        [SetUp]
        public void SetUp()
        {
            _model = new RaceModel();
            _raceFinished = new FakePublisher<RaceFinishedMessage>();
            _racerFinished = new FakePublisher<RacerFinishedMessage>();
            _sut = new Systems_Race(
                _model,
                new FakePublisher<RaceStartedMessage>(),
                _racerFinished,
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
        public void NoProgress_RecordsStalledAsTheReason()
        {
            _sut.Advance(Systems_Race.NO_PROGRESS_TIMEOUT_SECONDS + 1f);

            Assert.That(_model.FindRacer("worm#1").Knockout, Is.EqualTo(KnockoutReason.Stalled));
        }

        [Test]
        public void NotifyFailure_RecordsTheReasonGiven()
        {
            _sut.NotifyFailure("worm#1", KnockoutReason.LeftTrack);

            Assert.That(_model.FindRacer("worm#1").Knockout, Is.EqualTo(KnockoutReason.LeftTrack));
            Assert.That(_model.FindRacer("worm#2").Knockout, Is.EqualTo(KnockoutReason.None));
        }

        [Test]
        public void PodiumCutoff_RecordsCutoffNotFailure()
        {
            _sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "a", CreatureId = "worm", Status = RacerStatus.Racing },
                new() { RacerId = "b", CreatureId = "spider", Status = RacerStatus.Racing },
                new() { RacerId = "c", CreatureId = "crab", Status = RacerStatus.Racing },
                new() { RacerId = "d", CreatureId = "blob", Status = RacerStatus.Racing }
            });

            _sut.NotifyFinish("a");
            _sut.NotifyFinish("b");
            _sut.NotifyFinish("c");

            Assert.That(_model.FindRacer("d").Knockout, Is.EqualTo(KnockoutReason.PodiumCutoff));
            Assert.That(_model.FindRacer("a").Knockout, Is.EqualTo(KnockoutReason.None));
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

        /// <summary>
        /// Two racers reported in the same frame share a FinishTime and the trigger
        /// callback order is engine-arbitrary, so the one farther past the line has
        /// to take the higher place regardless of which was reported first.
        /// </summary>
        [Test]
        public void SameFrameFinishers_AreRankedByOvershoot()
        {
            _sut.NotifyFinish("worm#1", overshootMeters: 0.05f);
            _sut.NotifyFinish("worm#2", overshootMeters: 0.4f);

            Assert.That(_model.FindRacer("worm#2").Place, Is.EqualTo(1));
            Assert.That(_model.FindRacer("worm#1").Place, Is.EqualTo(2));
        }

        /// <summary>
        /// The timeout referee ranks the whole field but must announce only the
        /// podium: a RacerFinishedMessage per entrant would set the win fanfare -
        /// flash, slow-mo, confetti, stinger - off once per racer in one frame.
        /// </summary>
        [Test]
        public void Timeout_AnnouncesThePodiumOnly_NotTheWholeField()
        {
            var field = new List<RacerState>();
            for (int racerIndex = 0; racerIndex < 20; racerIndex++)
            {
                field.Add(new RacerState
                {
                    RacerId = "r" + racerIndex, CreatureId = "crab", Status = RacerStatus.Racing
                });
            }
            _sut.StartRace(field);
            for (int racerIndex = 0; racerIndex < field.Count; racerIndex++)
            {
                _sut.ReportProgress(field[racerIndex].RacerId, racerIndex);
            }
            _racerFinished.Published.Clear();

            _sut.Advance(Systems_Race.RACE_TIMEOUT_SECONDS + 1f);

            Assert.That(_racerFinished.Published, Has.Count.EqualTo(Systems_Race.PODIUM_FINISHERS));
            // Farthest first: r19 led on distance, so it takes the win.
            Assert.That(_model.FindRacer("r19").Place, Is.EqualTo(1));
            Assert.That(_model.FindRacer("r0").Place, Is.EqualTo(field.Count));
        }

        /// <summary>A hundred-racer field still ends at the podium, not at the last straggler.</summary>
        [Test]
        public void LargeField_EndsAtThePodium()
        {
            var field = new List<RacerState>();
            for (int racerIndex = 0; racerIndex < 100; racerIndex++)
            {
                field.Add(new RacerState
                {
                    RacerId = "r" + racerIndex, CreatureId = "crab", Status = RacerStatus.Racing
                });
            }
            _sut.StartRace(field);
            _raceFinished.Published.Clear();

            _sut.NotifyFinish("r7");
            _sut.NotifyFinish("r3");
            _sut.NotifyFinish("r55");

            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_raceFinished.Published, Has.Count.EqualTo(1));
            Assert.That(_raceFinished.Published[0].Results, Has.Count.EqualTo(100));
        }

        /// <summary>
        /// The menu path. AbortRace has to be silent - a RaceFinishedMessage here
        /// would pour the produce, run the ELO update and raise the results panel
        /// over a menu the player just asked for.
        /// </summary>
        [Test]
        public void AbortRace_IsSilentAndClearsTheField()
        {
            _sut.ReportProgress("worm#1", 4f);
            _raceFinished.Published.Clear();

            _sut.AbortRace();

            Assert.That(_raceFinished.Published, Is.Empty);
            Assert.That(_model.RaceActive, Is.False);
            Assert.That(_model.Racers, Is.Empty);
            // And the dead clock must not resurrect it.
            _sut.Advance(Systems_Race.RACE_TIMEOUT_SECONDS + 1f);
            Assert.That(_raceFinished.Published, Is.Empty);
        }

        /// <summary>
        /// RACE AGAIN: the same referee runs the next race, so every per-race
        /// carry-over - places, statuses, the clock, the stall timers - has to be
        /// cleared by StartRace rather than by anything the caller remembers to do.
        /// </summary>
        [Test]
        public void StartRace_AfterAFinishedRace_ResetsEveryCarryOver()
        {
            _sut.NotifyFinish("worm#1");
            _sut.NotifyFinish("worm#2");
            Assert.That(_model.RaceActive, Is.False);
            int raceNumber = _model.RaceNumber;

            _sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "worm#1", CreatureId = "worm", Status = RacerStatus.Racing },
                new() { RacerId = "worm#2", CreatureId = "worm", Status = RacerStatus.Racing }
            });

            Assert.That(_model.RaceActive, Is.True);
            Assert.That(_model.ElapsedSeconds, Is.EqualTo(0f));
            Assert.That(_model.RaceNumber, Is.EqualTo(raceNumber + 1));
            Assert.That(_model.FindRacer("worm#1").Place, Is.EqualTo(0));
            Assert.That(_model.FindRacer("worm#1").Status, Is.EqualTo(RacerStatus.Racing));
            // Places restart at 1 rather than continuing the previous race's count.
            _sut.NotifyFinish("worm#2");
            Assert.That(_model.FindRacer("worm#2").Place, Is.EqualTo(1));
        }

        /// <summary>
        /// Courses carry their own clock (600 s against the builder maps' 120 s).
        /// A course race must not be guillotined at the default.
        /// </summary>
        [Test]
        public void StartRace_HonoursThePerMapTimeLimit()
        {
            _sut.StartRace(new List<RacerState>
            {
                new() { RacerId = "a", CreatureId = "crab", Status = RacerStatus.Racing }
            }, timeLimitSeconds: 600f);

            // Well past the default limit, nowhere near this map's own.
            for (int step = 0; step < 20; step++)
            {
                _sut.ReportProgress("a", step + 1);
                _sut.Advance(10f);
            }

            Assert.That(_model.ElapsedSeconds, Is.GreaterThan(Systems_Race.RACE_TIMEOUT_SECONDS));
            Assert.That(_model.RaceActive, Is.True);
            Assert.That(_model.FindRacer("a").Status, Is.EqualTo(RacerStatus.Racing));
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
