using System.Collections.Generic;

namespace PoRacer.Models
{
    public readonly struct RaceStartedMessage
    {
        public readonly int RacerCount;

        public RaceStartedMessage(int racerCount)
        {
            RacerCount = racerCount;
        }
    }

    public readonly struct RacerFinishedMessage
    {
        public readonly string RacerId;
        public readonly int Place;
        public readonly float Time;

        public RacerFinishedMessage(string racerId, int place, float time)
        {
            RacerId = racerId;
            Place = place;
            Time = time;
        }
    }

    public readonly struct LeadChangedMessage
    {
        public readonly string RacerId;

        public LeadChangedMessage(string racerId)
        {
            RacerId = racerId;
        }
    }

    public readonly struct RacerDnfMessage
    {
        public readonly string RacerId;

        public RacerDnfMessage(string racerId)
        {
            RacerId = racerId;
        }
    }

    public readonly struct RaceFinishedMessage
    {
        public readonly IReadOnlyList<RaceResultEntry> Results;

        public RaceFinishedMessage(IReadOnlyList<RaceResultEntry> results)
        {
            Results = results;
        }
    }

    public readonly struct RaceResultEntry
    {
        public readonly string RacerId;
        public readonly string CreatureId;
        public readonly int Place;
        public readonly float Time;
        public readonly bool Dnf;

        public RaceResultEntry(string racerId, string creatureId, int place, float time, bool dnf)
        {
            RacerId = racerId;
            CreatureId = creatureId;
            Place = place;
            Time = time;
            Dnf = dnf;
        }
    }

    public readonly struct RacerWipeoutMessage
    {
        public readonly string RacerId;
        public readonly UnityEngine.Vector3 Position;
        public readonly bool IsFatal;

        public RacerWipeoutMessage(string racerId, UnityEngine.Vector3 position, bool isFatal)
        {
            RacerId = racerId;
            Position = position;
            IsFatal = isFatal;
        }
    }

    public readonly struct RacerOvertakeMessage
    {
        public readonly string OvertakerId;
        public readonly string OvertakenId;
        public readonly int NewPlace;

        public RacerOvertakeMessage(string overtakerId, string overtakenId, int newPlace)
        {
            OvertakerId = overtakerId;
            OvertakenId = overtakenId;
            NewPlace = newPlace;
        }
    }

    /// <summary>
    /// The broadcast director cut to a new shot. RivalId is set only for a duel.
    /// </summary>
    public readonly struct CameraShotChangedMessage
    {
        public readonly ShotReason Reason;
        public readonly string SubjectId;
        public readonly string RivalId;

        public CameraShotChangedMessage(ShotReason reason, string subjectId, string rivalId)
        {
            Reason = reason;
            SubjectId = subjectId;
            RivalId = rivalId;
        }
    }

    /// <summary>
    /// The ground for the next race exists: generated or authored, colliders in
    /// place, racers not yet spawned. Kind is the kind actually built, which is
    /// Flat when a course map had to fall back for want of its scene entry.
    /// </summary>
    public readonly struct TrackBuiltMessage
    {
        public readonly PoRacer.Systems.TrackKind Kind;

        public TrackBuiltMessage(PoRacer.Systems.TrackKind kind)
        {
            Kind = kind;
        }
    }

    public readonly struct PhotoFinishMessage
    {
        public readonly string WinnerId;
        public readonly string RunnerUpId;
        public readonly float MarginSeconds;

        public PhotoFinishMessage(string winnerId, string runnerUpId, float marginSeconds)
        {
            WinnerId = winnerId;
            RunnerUpId = runnerUpId;
            MarginSeconds = marginSeconds;
        }
    }
}
