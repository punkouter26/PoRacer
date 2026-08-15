using System;
using MessagePipe;
using PoRacer.Models;
using VContainer;

namespace PoRacer.Systems
{
    /// <summary>
    /// Pairwise ELO for N-racer results: every distinct-creature pair is one 2-player
    /// update with K scaled by 1/(N-1). DNF loses to every finisher; DNF vs DNF is a
    /// draw; an all-DNF race changes nothing. Same-creature pairs are skipped.
    /// </summary>
    public sealed class Systems_Elo : IDisposable
    {
        private const float BASE_K = 32f;

        private readonly EloModel _model;
        private readonly Systems_Persistence _persistence;
        private readonly IDisposable _subscription;

        [Inject]
        public Systems_Elo(EloModel model, Systems_Persistence persistence, ISubscriber<RaceFinishedMessage> raceFinished)
        {
            _model = model;
            _persistence = persistence;
            _subscription = raceFinished.Subscribe(OnRaceFinished);
        }

        // Test constructor: rating math only, no messaging or persistence.
        public Systems_Elo(EloModel model)
        {
            _model = model;
        }

        public void Dispose() => _subscription?.Dispose();

        public void ApplyResults(System.Collections.Generic.IReadOnlyList<RaceResultEntry> results)
        {
            bool anyFinisher = false;
            for (int entryIndex = 0; entryIndex < results.Count; entryIndex++)
            {
                if (!results[entryIndex].Dnf)
                {
                    anyFinisher = true;
                    break;
                }
            }
            if (!anyFinisher || results.Count < 2)
            {
                return;
            }

            float pairK = BASE_K / (results.Count - 1);
            for (int firstIndex = 0; firstIndex < results.Count; firstIndex++)
            {
                for (int secondIndex = firstIndex + 1; secondIndex < results.Count; secondIndex++)
                {
                    RaceResultEntry first = results[firstIndex];
                    RaceResultEntry second = results[secondIndex];
                    if (first.CreatureId == second.CreatureId)
                    {
                        continue;
                    }
                    float firstScore = ScoreAgainst(first, second);
                    float firstRating = _model.GetRating(first.CreatureId);
                    float secondRating = _model.GetRating(second.CreatureId);
                    float expected = 1f / (1f + (float)Math.Pow(10.0, (secondRating - firstRating) / 400.0));
                    float delta = pairK * (firstScore - expected);
                    _model.SetRating(first.CreatureId, firstRating + delta);
                    _model.SetRating(second.CreatureId, secondRating - delta);
                }
            }        }

        private static float ScoreAgainst(RaceResultEntry first, RaceResultEntry second)
        {
            if (first.Dnf && second.Dnf)
            {
                return 0.5f;
            }
            if (first.Dnf)
            {
                return 0f;
            }
            if (second.Dnf)
            {
                return 1f;
            }
            return first.Place < second.Place ? 1f : 0f;
        }

        private void OnRaceFinished(RaceFinishedMessage message)
        {
            ApplyResults(message.Results);
            _persistence.SaveRatings();
            _persistence.AppendHistory(message.Results);
        }
    }
}
