using System;
using System.Collections.Generic;
using System.IO;
using PoRacer.Models;
using UnityEngine;
using VContainer;

namespace PoRacer.Systems
{
    /// <summary>
    /// JSON persistence for ratings and race history. Writes are atomic
    /// (tmp file + replace); a corrupt file is backed up and recreated.
    /// History is capped so the self-running app never grows a huge file.
    /// </summary>
    public sealed class Systems_Persistence : IDisposable
    {
        public const int MAX_HISTORY_ENTRIES = 200;

        [Serializable]
        private sealed class RatingsFile
        {
            public List<string> creatureIds = new();
            public List<float> ratings = new();
        }

        [Serializable]
        private sealed class HistoryEntry
        {
            public int raceNumber;
            public string timestamp;
            public List<string> racerIds = new();
            public List<string> creatureIds = new();
            public List<int> places = new();
            public List<float> times = new();
        }

        [Serializable]
        private sealed class HistoryFile
        {
            public List<HistoryEntry> races = new();
        }

        private readonly EloModel _eloModel;
        private readonly RaceModel _raceModel;
        private readonly string _directory;
        private readonly HistoryFile _history;

        [Inject]
        public Systems_Persistence(EloModel eloModel, RaceModel raceModel)
            : this(eloModel, raceModel, Application.persistentDataPath) { }

        public Systems_Persistence(EloModel eloModel, RaceModel raceModel, string directory)
        {
            _eloModel = eloModel;
            _raceModel = raceModel;
            _directory = directory;
            RatingsFile ratings = LoadOrRecreate<RatingsFile>(RatingsPath);
            for (int ratingIndex = 0; ratingIndex < ratings.creatureIds.Count && ratingIndex < ratings.ratings.Count; ratingIndex++)
            {
                _eloModel.SetRating(ratings.creatureIds[ratingIndex], ratings.ratings[ratingIndex]);
            }
            _history = LoadOrRecreate<HistoryFile>(HistoryPath);
        }

        public string RatingsPath => Path.Combine(_directory, "elo.json");
        public string HistoryPath => Path.Combine(_directory, "race-history.json");

        public void SaveRatings()
        {
            var file = new RatingsFile();
            foreach (KeyValuePair<string, float> pair in _eloModel.Ratings)
            {
                file.creatureIds.Add(pair.Key);
                file.ratings.Add(pair.Value);
            }
            WriteAtomic(RatingsPath, JsonUtility.ToJson(file, true));
        }

        public void AppendHistory(IReadOnlyList<RaceResultEntry> results)
        {
            var entry = new HistoryEntry
            {
                raceNumber = _raceModel.RaceNumber,
                timestamp = DateTime.UtcNow.ToString("o")
            };
            for (int resultIndex = 0; resultIndex < results.Count; resultIndex++)
            {
                entry.racerIds.Add(results[resultIndex].RacerId);
                entry.creatureIds.Add(results[resultIndex].CreatureId);
                entry.places.Add(results[resultIndex].Place);
                entry.times.Add(results[resultIndex].Time);
            }
            _history.races.Add(entry);
            while (_history.races.Count > MAX_HISTORY_ENTRIES)
            {
                _history.races.RemoveAt(0);
            }
            WriteAtomic(HistoryPath, JsonUtility.ToJson(_history, true));
        }

        public void Dispose() { }

        private T LoadOrRecreate<T>(string path) where T : class, new()
        {
            if (!File.Exists(path))
            {
                return new T();
            }
            try
            {
                T parsed = JsonUtility.FromJson<T>(File.ReadAllText(path));
                return parsed ?? new T();
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"Corrupt save file '{path}' ({exception.Message}); backing up and recreating.");
                try
                {
                    File.Copy(path, path + ".bak", overwrite: true);
                    File.Delete(path);
                }
                catch (IOException)
                {
                    // Backup is best-effort; a locked file must not stop the app.
                }
                return new T();
            }
        }

        private static void WriteAtomic(string path, string json)
        {
            string tmpPath = path + ".tmp";
            File.WriteAllText(tmpPath, json);
            if (File.Exists(path))
            {
                File.Replace(tmpPath, path, null);
            }
            else
            {
                File.Move(tmpPath, path);
            }
        }
    }
}
