using System;
using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// Everything a creature race needs that is not the body itself (AGENTS.md section 5:
    /// tunables are serialised, never buried in code): the rig file, the racers, the
    /// observation definition, the track, the race timing, the self-tests and the labels.
    /// The body - masses, gains, limits, friction - comes from the rig JSON, the one file the
    /// trainers used, so an Inspector edit cannot race a different animal.
    ///
    /// Plain serialisable data, embedded in a settings asset (CreatureRaceSettings for a
    /// MuJoCo-only creature, WormRaceSettings for the worm, which adds its PhysX knobs) and
    /// registered in VContainer by that scene's lifetime scope. The scene builders write every
    /// field; the defaults below only matter for a hand-made asset.
    /// </summary>
    [Serializable]
    public sealed class CreatureRaceConfig
    {
        [Header("Creature")]
        [Tooltip("The trainer's rig JSON, copied under Assets/. Every racer is built from it.")]
        [SerializeField] private TextAsset _rigJson;
        [Tooltip("The body whose front crosses the finish line (worm: seg0, quad: the torso). Empty = the rig's torso.")]
        [SerializeField] private string _leadBody = string.Empty;
        [SerializeField] private CreatureObservationSpec _observation = new();
        [Tooltip("Feed back the clipped action as 'previous action' (the trainers clip to [-1, 1]).")]
        [SerializeField] private bool _previousActionClipped = true;

        [Header("Racers, one per lane (AGENTS rule D: red and green are reserved)")]
        [SerializeField] private CreatureRacerDefinition[] _racers = Array.Empty<CreatureRacerDefinition>();

        [Header("Track")]
        [SerializeField] private float _trackLength = 20f;
        [SerializeField] private float _laneSpacing = 2f;
        [SerializeField] private float _startLineZ;
        [Tooltip("The lead point starts this far behind the start line.")]
        [SerializeField] private float _startGap = 0.01f;

        [Header("Race")]
        [SerializeField] private int _countdownSeconds = 3;
        [SerializeField] private float _timeLimitSeconds = 60f;
        [SerializeField] private float _resultsHoldSeconds = 4f;
        [SerializeField] private int _seriesLength = 5;
        [Tooltip("Start a series on play. The editor harness overrides count and mode.")]
        [SerializeField] private bool _autoStartSeries = true;
        [SerializeField] private float _selfTestSettleSeconds = 1f;
        [Tooltip("Reference up . world up below this counts as fallen (HUD and report only; nobody is stood up, "
               + "rule H). -1 = never (a worm rolls; it has no 'up').")]
        [SerializeField] private float _fallenUprightThreshold = 0.5f;

        [Header("Self-tests")]
        [SerializeField] private CreatureSelfTestDefinition[] _selfTests = Array.Empty<CreatureSelfTestDefinition>();

        [Header("MuJoCo (every MuJoCo racer shares one MjScene)")]
        [Tooltip("The trainers' <option iterations>. ls_iterations has no plug-in field; MuJoCo uses 50.")]
        [SerializeField] private int _mujocoSolverIterations = 10;
        [Tooltip("Write the MJCF the plug-in generates to Application.temporaryCachePath, to diff against the trainer's.")]
        [SerializeField] private bool _dumpMujocoMjcf = true;

        [Header("Labels")]
        [SerializeField] private string _title = "CREATURE RACE";
        [Tooltip("Results go to Logs/<prefix>_<stamp>.json and Logs/<prefix>_selftest_<test>_<stamp>.json.")]
        [SerializeField] private string _reportPrefix = "creaturerace";
        [SerializeField] private string _finishRule = "first lead point past the finish line; time limit ranks by distance";
        [Tooltip("What a racer without a brain does, for the HUD: \"holds straight\", \"stands still\".")]
        [SerializeField] private string _holdLabel = "holds its rest pose";
        [Tooltip("Where brains are expected, for the missing-brain message.")]
        [SerializeField] private string _brainsFolder = "Assets";

        public TextAsset RigJson => _rigJson;
        public string LeadBody => _leadBody ?? string.Empty;
        public CreatureObservationSpec Observation => _observation ??= new CreatureObservationSpec();
        public bool PreviousActionClipped => _previousActionClipped;
        public IReadOnlyList<CreatureRacerDefinition> Racers => _racers ?? Array.Empty<CreatureRacerDefinition>();
        public int LaneCount => _racers != null ? _racers.Length : 0;
        public float TrackLength => Mathf.Max(1f, _trackLength);
        public float LaneSpacing => Mathf.Max(0.5f, _laneSpacing);
        public float StartLineZ => _startLineZ;
        public float StartGap => Mathf.Max(0f, _startGap);
        public int CountdownSeconds => Mathf.Max(0, _countdownSeconds);
        public float TimeLimitSeconds => Mathf.Max(1f, _timeLimitSeconds);
        public float ResultsHoldSeconds => Mathf.Max(0f, _resultsHoldSeconds);
        public int SeriesLength => Mathf.Max(1, _seriesLength);
        public bool AutoStartSeries => _autoStartSeries;
        public float SelfTestSettleSeconds => Mathf.Max(0f, _selfTestSettleSeconds);
        public float FallenUprightThreshold => _fallenUprightThreshold;

        /// <summary>False when falling is not a thing for this creature (threshold -1 or below).</summary>
        public bool TracksFalls => _fallenUprightThreshold > -1f;

        public bool IsFallen(float upright)
        {
            return TracksFalls && upright < _fallenUprightThreshold;
        }
        public IReadOnlyList<CreatureSelfTestDefinition> SelfTests =>
            _selfTests ?? Array.Empty<CreatureSelfTestDefinition>();
        public int MujocoSolverIterations => Mathf.Max(1, _mujocoSolverIterations);
        public bool DumpMujocoMjcf => _dumpMujocoMjcf;
        public string Title => string.IsNullOrEmpty(_title) ? "CREATURE RACE" : _title;
        public string ReportPrefix => string.IsNullOrEmpty(_reportPrefix) ? "creaturerace" : _reportPrefix;
        public string FinishRule => _finishRule ?? string.Empty;
        public string HoldLabel => _holdLabel ?? string.Empty;
        public string BrainsFolder => _brainsFolder ?? string.Empty;
        /// <summary>File name under Application.temporaryCachePath for the generated MJCF, or empty.</summary>
        public string MjcfDumpFile => _dumpMujocoMjcf ? ReportPrefix + "_mujoco_scene.xml" : string.Empty;

        /// <summary>
        /// Lane centre, x offset from the track centre: lanes <see cref="LaneSpacing"/> apart and
        /// centred on x = 0 whatever their number. The scene builders paint the lines from this.
        /// </summary>
        public float LaneX(int lane)
        {
            return (lane - (Mathf.Max(1, LaneCount) - 1) * 0.5f) * LaneSpacing;
        }

        /// <summary>The start-line point of a lane, where the lead point starts just behind.</summary>
        public Vector3 LaneOrigin(int lane)
        {
            return new Vector3(LaneX(lane), 0f, StartLineZ);
        }

        public string ExpectedBrainPath(CreatureRacerDefinition racer)
        {
            return BrainsFolder + "/" + (string.IsNullOrEmpty(racer.BrainFile) ? "<brain>.onnx" : racer.BrainFile);
        }

        public CreatureSelfTestDefinition FindSelfTest(string key)
        {
            IReadOnlyList<CreatureSelfTestDefinition> tests = SelfTests;
            for (int index = 0; index < tests.Count; index++)
            {
                if (tests[index] != null && tests[index].Matches(key))
                {
                    return tests[index];
                }
            }
            return null;
        }
    }
}
