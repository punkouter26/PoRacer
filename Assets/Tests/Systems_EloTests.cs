using System.Collections.Generic;
using NUnit.Framework;
using PoRacer.Models;
using PoRacer.Systems;

namespace PoRacer.Tests
{
    public sealed class Systems_EloTests
    {
        private static RaceResultEntry Entry(string creatureId, int place, bool dnf = false)
        {
            return new RaceResultEntry(creatureId + "#1", creatureId, place, 10f, dnf);
        }

        [Test]
        public void ApplyResults_WinnerGainsLoserLoses()
        {
            var model = new EloModel();
            var sut = new Systems_Elo(model);

            sut.ApplyResults(new List<RaceResultEntry> { Entry("a", 1), Entry("b", 2) });

            Assert.That(model.GetRating("a"), Is.GreaterThan(EloModel.DEFAULT_RATING));
            Assert.That(model.GetRating("b"), Is.LessThan(EloModel.DEFAULT_RATING));
        }

        [Test]
        public void ApplyResults_ZeroSum()
        {
            var model = new EloModel();
            var sut = new Systems_Elo(model);

            sut.ApplyResults(new List<RaceResultEntry> { Entry("a", 1), Entry("b", 2), Entry("c", 3) });

            float total = model.GetRating("a") + model.GetRating("b") + model.GetRating("c");
            Assert.That(total, Is.EqualTo(EloModel.DEFAULT_RATING * 3).Within(0.001f));
        }

        [Test]
        public void ApplyResults_DnfLosesToFinisher()
        {
            var model = new EloModel();
            var sut = new Systems_Elo(model);

            sut.ApplyResults(new List<RaceResultEntry> { Entry("a", 2), Entry("b", -1, dnf: true) });

            Assert.That(model.GetRating("a"), Is.GreaterThan(model.GetRating("b")));
        }

        [Test]
        public void ApplyResults_AllDnf_ChangesNothing()
        {
            var model = new EloModel();
            var sut = new Systems_Elo(model);

            sut.ApplyResults(new List<RaceResultEntry> { Entry("a", -1, dnf: true), Entry("b", -1, dnf: true) });

            Assert.That(model.GetRating("a"), Is.EqualTo(EloModel.DEFAULT_RATING));
            Assert.That(model.GetRating("b"), Is.EqualTo(EloModel.DEFAULT_RATING));
        }

        [Test]
        public void ApplyResults_SameCreaturePairs_AreSkipped()
        {
            var model = new EloModel();
            var sut = new Systems_Elo(model);

            sut.ApplyResults(new List<RaceResultEntry>
            {
                new("worm#1", "worm", 1, 10f, false),
                new("worm#2", "worm", 2, 11f, false)
            });

            Assert.That(model.GetRating("worm"), Is.EqualTo(EloModel.DEFAULT_RATING));
        }
    }
}
