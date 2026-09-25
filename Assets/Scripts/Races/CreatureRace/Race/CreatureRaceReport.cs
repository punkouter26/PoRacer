using System;
using System.Collections.Generic;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// The JSON written to Logs/&lt;prefix&gt;_&lt;stamp&gt;.json (series) and
    /// Logs/&lt;prefix&gt;_selftest_&lt;test&gt;_&lt;stamp&gt;.json (self-tests), so an agent can read the
    /// results without the editor. JsonUtility shapes: public lowerCamelCase fields, no
    /// dictionaries, -1 where a number does not apply (JsonUtility cannot write NaN).
    /// </summary>
    internal static class CreatureRaceReport
    {
        [Serializable]
        internal sealed class Series
        {
            public string kind;
            public string creature;
            public string scene;
            public string startedAt;
            public string finishedAt;
            public int plannedRaces;
            public int completedRaces;
            public float trackLengthMeters;
            public float timeLimitSeconds;
            public float physicsDtSeconds;
            public string finishRule;
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
            /// <summary>Lowest reference up . world up after GO (below the threshold = it fell).</summary>
            public float minUpright;
            public bool fellOver;
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
            public int falls;
            public float meanFinishTimeSeconds = -1f;
            public float bestFinishTimeSeconds = -1f;
            public float meanDistanceMeters;
            public float meanAverageSpeedMps;
        }

        [Serializable]
        internal sealed class SelfTest
        {
            public string kind;
            public string creature;
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
            /// <summary>Largest |joint - rest| since GO, radians.</summary>
            public float maxAbsJointRad;
            /// <summary>Probe point on body B minus body A's origin, in A's frame (MuJoCo x fwd, y left, z up).</summary>
            public float[] probeOffsetMeters;
            /// <summary>The probe along the test's axis (change since GO when the test says so).</summary>
            public float probeAlongAxisMeters;
            public float leadDisplacementMeters;
            public float referenceHeightMeters;
            public float referenceUpright;
            public float finalSpeedMps;
            public float[] finalJointRad;
        }
    }
}
