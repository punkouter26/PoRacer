using System.Collections.Generic;
using NUnit.Framework;
using PoRacer.Models;
using PoRacer.Systems;

namespace PoRacer.Tests
{
    public sealed class Systems_SimWarsTests
    {
        private SimWarsModel _model;
        private Systems_SimWars _sut;
        private Dictionary<string, TrainingSource> _teams;

        [SetUp]
        public void SetUp()
        {
            _model = new SimWarsModel();
            _sut = new Systems_SimWars(_model);
            _teams = new Dictionary<string, TrainingSource>
            {
                { "boy#1", TrainingSource.MuJoCo },
                { "h1#1", TrainingSource.IsaacLab },
                { "box#1", TrainingSource.IsaacLab },
                { "quad#1", TrainingSource.MlAgents }
            };
        }

        private TrainingSource TeamOf(string racerId) => _teams[racerId];

        private static RaceResultEntry Entry(string racerId, int place, bool dnf = false)
        {
            return new RaceResultEntry(racerId, racerId, place, 10f, dnf);
        }

        [Test]
        public void ApplyResults_AwardsPodiumPointsAndWinToTheWinnersTeam()
        {
            _sut.ApplyResults(new List<RaceResultEntry>
            {
                Entry("h1#1", 1), Entry("boy#1", 2), Entry("quad#1", 3), Entry("box#1", -1, dnf: true)
            }, TeamOf);

            Assert.That(_model.Team(TrainingSource.IsaacLab).Points, Is.EqualTo(5));
            Assert.That(_model.Team(TrainingSource.IsaacLab).Wins, Is.EqualTo(1));
            Assert.That(_model.Team(TrainingSource.MuJoCo).Points, Is.EqualTo(3));
            Assert.That(_model.Team(TrainingSource.MlAgents).Points, Is.EqualTo(1));
            Assert.That(_model.Team(TrainingSource.IsaacLab).Races, Is.EqualTo(1));
            Assert.That(_model.IsaacLabAhead, Is.EqualTo(1));
            Assert.That(_model.MuJoCoAhead, Is.EqualTo(0));
        }

        [Test]
        public void ApplyResults_DnfHandedAPodiumSlotScoresNothing()
        {
            _sut.ApplyResults(new List<RaceResultEntry>
            {
                Entry("boy#1", 1, dnf: true), Entry("h1#1", 2, dnf: true)
            }, TeamOf);

            Assert.That(_model.Team(TrainingSource.MuJoCo).Points, Is.EqualTo(0));
            Assert.That(_model.Team(TrainingSource.MuJoCo).Wins, Is.EqualTo(0));
            Assert.That(_model.HeadToHeadDraws, Is.EqualTo(1));
        }

        [Test]
        public void ApplyResults_HeadToHeadOnlyCountsRacesBothToolsEntered()
        {
            _sut.ApplyResults(new List<RaceResultEntry> { Entry("boy#1", 1), Entry("quad#1", 2) }, TeamOf);

            Assert.That(_model.MuJoCoAhead + _model.IsaacLabAhead + _model.HeadToHeadDraws, Is.EqualTo(0));
            Assert.That(_model.RacesScored, Is.EqualTo(1));
        }

        [Test]
        public void ApplyResults_LastRacePointsResetEachRace()
        {
            _sut.ApplyResults(new List<RaceResultEntry> { Entry("boy#1", 1), Entry("h1#1", 2) }, TeamOf);
            _sut.ApplyResults(new List<RaceResultEntry> { Entry("h1#1", 1), Entry("boy#1", 2) }, TeamOf);

            Assert.That(_model.Team(TrainingSource.MuJoCo).LastRacePoints, Is.EqualTo(3));
            Assert.That(_model.Team(TrainingSource.MuJoCo).Points, Is.EqualTo(8));
            Assert.That(_model.MuJoCoAhead, Is.EqualTo(1));
            Assert.That(_model.IsaacLabAhead, Is.EqualTo(1));
        }
    }
}
