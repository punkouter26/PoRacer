using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using PoRacer.Models;
using PoRacer.Systems;

namespace PoRacer.Tests
{
    public sealed class Systems_PersistenceTests
    {
        private string _directory;

        [SetUp]
        public void SetUp()
        {
            _directory = Path.Combine(Path.GetTempPath(), "PoRacerTests_" + Path.GetRandomFileName());
            Directory.CreateDirectory(_directory);
        }

        [TearDown]
        public void TearDown()
        {
            if (Directory.Exists(_directory))
            {
                Directory.Delete(_directory, recursive: true);
            }
        }

        [Test]
        public void SaveRatings_ThenReload_RestoresRatings()
        {
            var eloModel = new EloModel();
            eloModel.SetRating("worm", 1300f);
            var sut = new Systems_Persistence(eloModel, new RaceModel(), _directory);
            sut.SaveRatings();

            var reloadedModel = new EloModel();
            _ = new Systems_Persistence(reloadedModel, new RaceModel(), _directory);

            Assert.That(reloadedModel.GetRating("worm"), Is.EqualTo(1300f));
        }

        [Test]
        public void CorruptRatingsFile_IsBackedUpAndRecreated()
        {
            string path = Path.Combine(_directory, "elo.json");
            File.WriteAllText(path, "{ not valid json !!!");

            var model = new EloModel();
            Assert.DoesNotThrow(() => _ = new Systems_Persistence(model, new RaceModel(), _directory));

            Assert.That(model.GetRating("worm"), Is.EqualTo(EloModel.DEFAULT_RATING));
            Assert.That(File.Exists(path + ".bak"), Is.True);
        }

        [Test]
        public void AppendHistory_CapsEntries()
        {
            var sut = new Systems_Persistence(new EloModel(), new RaceModel(), _directory);
            var results = new List<RaceResultEntry> { new("worm#1", "worm", 1, 10f, false) };

            for (int raceIndex = 0; raceIndex < Systems_Persistence.MAX_HISTORY_ENTRIES + 25; raceIndex++)
            {
                sut.AppendHistory(results);
            }

            string json = File.ReadAllText(Path.Combine(_directory, "race-history.json"));
            int count = System.Text.RegularExpressions.Regex.Matches(json, "\"raceNumber\"").Count;
            Assert.That(count, Is.EqualTo(Systems_Persistence.MAX_HISTORY_ENTRIES));
        }
    }
}
