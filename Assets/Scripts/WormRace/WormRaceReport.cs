using System;
using System.Collections.Generic;

namespace PoRacer.WormRace
{
    /// <summary>
    /// The JSON written to Logs/wormrace_&lt;stamp&gt;.json (series) and
    /// Logs/wormrace_selftest_&lt;mode&gt;_&lt;stamp&gt;.json (self-tests), so an agent can read the
    /// results without the editor. JsonUtility shapes: public lowerCamelCase fields, no
    /// dictionaries, -1 where a number does not apply (JsonUtility cannot write NaN).
    /// </summary>
    internal static class WormRaceReport
    {
        [Serializable]
        internal sealed class Series
        {
            public string kind = "wormrace_series";
            public string scene;
            public string startedAt;
            public string finishedAt;
            public int plannedRaces;
            public int completedRaces;
            public float trackLengthMeters;
            public float timeLimitSeconds;
            public float physicsDtSeconds;
            public string finishRule = "first segment-0 nose past the finish line; time limit ranks by distance";
            public string error = string.Empty;
            public List<Race> races = new();
            public List<RacerSummary> summary = new();
        }

        [Serializable]
        internal sealed class Race
        {
            public int raceNumber;
            public string winner;
            public string winnerMethod;
            /// <summary>"finish", "timeLimit" or "failed".</summary>
            public string endReason;
            public List<RacerResult> racers = new();
        }

        [Serializable]
        internal sealed class RacerResult
        {
            public int lane;
            public string name;
            public string method;
            public string physics;
            public string brain;
            public bool brainLoaded;
            public string brainError;
            public string status;
            public int place;
            public float finishTimeSeconds = -1f;
            public float distanceMeters;
            public float averageSpeedMps;
            public string failReason;
        }

        [Serializable]
        internal sealed class RacerSummary
        {
            public int lane;
            public string name;
            public string method;
            public string physics;
            public string brain;
            public bool brainLoaded;
            public string brainError;
            public int wins;
            public int finishes;
            public float meanFinishTimeSeconds = -1f;
            public float bestFinishTimeSeconds = -1f;
            public float meanDistanceMeters;
            public float meanAverageSpeedMps;
        }

        [Serializable]
        internal sealed class SelfTest
        {
            public string kind = "wormrace_selftest";
            public string mode;
            public string startedAt;
            public string finishedAt;
            public float seconds;
            public string commandedJoint;
            public float commandedRad;
            public string expectation;
            public bool allPassed;
            public string error = string.Empty;
            public List<SelfTestRacer> racers = new();
        }

        [Serializable]
        internal sealed class SelfTestRacer
        {
            public int lane;
            public string name;
            public string method;
            public string physics;
            public string brain;
            public bool brainLoaded;
            public string brainError;
            public string status;
            public bool passed;
            public string verdict;
            /// <summary>Final angle of the commanded joint (sign tests), radians.</summary>
            public float commandedJointRad;
            public float maxAbsJointRad;
            /// <summary>Segment 1 centre minus segment 0 centre, along the head's right.</summary>
            public float secondSegmentLateralMeters;
            /// <summary>Segment 1 centre minus segment 0 centre, along the head's up.</summary>
            public float secondSegmentVerticalMeters;
            public float noseDisplacementMeters;
            public float referenceSegmentHeightMeters;
            public float finalSpeedMps;
            public float[] finalJointRad;
        }
    }
}
