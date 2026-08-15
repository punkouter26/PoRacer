using System.Collections.Generic;

namespace PoRacer.Models
{
    public sealed class EloModel
    {
        public const float DEFAULT_RATING = 1200f;

        private readonly Dictionary<string, float> _ratings = new();

        public IReadOnlyDictionary<string, float> Ratings => _ratings;

        public float GetRating(string creatureId)
        {
            return _ratings.TryGetValue(creatureId, out float rating) ? rating : DEFAULT_RATING;
        }

        public void SetRating(string creatureId, float rating)
        {
            _ratings[creatureId] = rating;
        }
    }
}
