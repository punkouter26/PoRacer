using System.Collections.Generic;

namespace PoRacer.Models
{
    public sealed class EloModel
    {
        public const float DEFAULT_RATING = 1200f;

        private readonly Dictionary<string, float> _ratings = new();
        // Net rating swing per creature from the most recent scored race, so the
        // podium can show it without keeping its own before/after snapshot.
        private readonly Dictionary<string, float> _lastRaceDeltas = new();

        public IReadOnlyDictionary<string, float> Ratings => _ratings;

        public float GetRating(string creatureId)
        {
            return _ratings.TryGetValue(creatureId, out float rating) ? rating : DEFAULT_RATING;
        }

        public void SetRating(string creatureId, float rating)
        {
            _ratings[creatureId] = rating;
        }

        public float GetLastRaceDelta(string creatureId)
        {
            return _lastRaceDeltas.TryGetValue(creatureId, out float delta) ? delta : 0f;
        }

        public void AccumulateDelta(string creatureId, float delta)
        {
            _lastRaceDeltas[creatureId] = GetLastRaceDelta(creatureId) + delta;
        }

        public void ClearDeltas()
        {
            _lastRaceDeltas.Clear();
        }
    }
}
