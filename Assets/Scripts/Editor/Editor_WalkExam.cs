#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using PoRacer.Models;
using PoRacer.Systems;
using PoRacer.Views;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using VContainer;
using VContainer.Unity;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// The walking exam from DOCS/Plan-TrainingMethodComparison.md: every brain, alone
    /// on the track, several times per map, scored against one standard. It exists so the
    /// three training methods (ML-Agents, Isaac Lab, MuJoCo) are judged by the same test
    /// in the same place rather than by their own reward curves.
    ///
    /// Why alone: the map catalogue records a Quadruped that finishes Flat solo in 58 s
    /// and never once in a full field, because wide rigs tangle on the grid. A mixed race
    /// measures the grid, not the gait. Why no quirks: a +12 % drive is a handicap the
    /// brain did not earn. Why several trials: in the 2026-09-24 smoke run the winner
    /// changed on every map.
    ///
    /// Speed is distance over PHYSICS time (Time.fixedTime), not the race clock. The race
    /// clock is unscaled wall time, and the editor dropped to 2-5 fps in that same run; a
    /// hitch the physics clamps away would otherwise be charged to the racer.
    ///
    /// Measured here: W1 speed, W2 uptime, W3 time fallen, W7 get-up, plus completion and
    /// the knockout reason. W4 heading, W5 foot skating, W6 smoothness and W8 torque
    /// limits need per-agent internals and are listed as not measured yet.
    ///
    /// Invoke: unity cmd eval --code "return PoRacer.EditorTools.Editor_WalkExam.Start(\"\", \"0,1,2\", 3);"
    ///         unity cmd eval --code "return PoRacer.EditorTools.Editor_WalkExam.Status();"
    /// An empty creature list means every raceable creature in the catalogue.
    /// </summary>
    public static class Editor_WalkExam
    {
        private const string JOB_KEY = "PoRacer.WalkExam.Job";
        private const string REPORT_KEY = "PoRacer.WalkExam.LastReport";
        private const string RACE_SCENE = "Assets/Scenes/SCN_RACE_FLAT.unity";
        private const float SCOPE_TIMEOUT_SECONDS = 30f;
        private const float START_TIMEOUT_SECONDS = 40f;
        private const float COOLDOWN_SECONDS = 2f;
        // Past the map's own clock, for the referee to score a timeout.
        private const float RACE_OVERRUN_MARGIN_SECONDS = 30f;

        // The standard. Fr 0.25 is a relaxed walk for any legged animal; speed-for-size
        // is what lets a crab and a human share one target.
        private const float FROUDE_NUMBER = 0.25f;
        private const float GRAVITY = 9.81f;
        private const float SPEED_TOLERANCE = 0.10f;
        private const float MIN_UPTIME = 0.90f;
        private const float MAX_FALLEN_FRACTION = 0.10f;
        private const float MIN_RECOVERY_RATE = 0.5f;

        // Orientation bands, as the y of the rest-relative up axis. Upright is within
        // 45 degrees; fallen is past 60, the same line the Isaac ports' own fall check uses.
        private const float UPRIGHT_COS = 0.707f;
        private const float FALLEN_COS = 0.5f;
        // A tip has to last this long to count as a fall, and standing has to last this
        // long to count as having got up, so a stumble is neither.
        private const float FALL_CONFIRM_SECONDS = 0.5f;
        private const float RECOVERY_CONFIRM_SECONDS = 1f;

        private static readonly string[] NotYetMeasured =
        {
            "W4 heading", "W5 foot skating", "W6 smoothness", "W8 torque/speed limits", "W9 ELO"
        };

        [Serializable]
        private sealed class Job
        {
            public string[] creatures;
            public int[] maps;
            public int trials;
            public int step;
            public string startedAt;
        }

        [Serializable]
        private sealed class TrialResult
        {
            public string creatureId;
            public string map;
            public int trial;
            public string outcome;
            public string knockout;
            public float distanceMeters;
            public float trackLengthMeters;
            public float physicsSeconds;
            public float raceClockSeconds;
            public float speedMps;
            public float uprightFraction;
            public float fallenFraction;
            public int falls;
            public int recoveries;
            public float fpsMin;
        }

        [Serializable]
        private sealed class ReportCard
        {
            public string creatureId;
            public string map;
            public int trials;
            public int finished;
            public string knockouts;
            public float legLengthMeters;
            public float targetSpeedMps;
            public float meanSpeedMps;
            public float speedRatio;
            public float meanUprightFraction;
            public float meanFallenFraction;
            public int falls;
            public int recoveries;
            public string w1Speed;
            public string w2Uptime;
            public string w3Falls;
            public string w7GetUp;
            public bool passesMeasured;
        }

        [Serializable]
        private sealed class Report
        {
            public string startedAt;
            public string finishedAt;
            public float froudeNumber = FROUDE_NUMBER;
            public string legLengthSource = "CreatureCatalog spawnHeight (root height at rest)";
            public string[] notYetMeasured = NotYetMeasured;
            public List<ReportCard> cards = new();
            public List<TrialResult> trials = new();
            public List<string> errors = new();
        }

        private enum Phase
        {
            WaitScope,
            Arm,
            WaitRaceStart,
            Racing,
            Cooldown,
        }

        private static Job _job;
        private static Report _report;
        private static Phase _phase;
        private static double _phaseStart;
        private static Systems_Spawn _spawn;
        private static RaceConfigModel _config;
        private static RaceModel _raceModel;
        private static Systems_Warmup _warmup;
        private static CreatureCatalog _catalog;

        // Per-trial sampling state.
        private static TrialResult _trial;
        private static Transform _body;
        private static Quaternion _restInverse;
        private static float _startFixedTime;
        private static float _endFixedTime;
        private static float _sampledSeconds;
        private static float _uprightSeconds;
        private static float _fallenSeconds;
        private static bool _isDown;
        private static float _tippedFor;
        private static float _standingFor;
        private static int _lastFrame;

        public static string Start(string creaturesCsv = "", string mapsCsv = "0,1,2", int trials = 3)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "already in play mode - stop it first";
            }
            if (EditorSceneManager.GetActiveScene().isDirty)
            {
                return "the open scene has unsaved changes - save or discard them first";
            }
            var maps = new List<int>();
            string[] mapParts = mapsCsv.Split(',');
            for (int partIndex = 0; partIndex < mapParts.Length; partIndex++)
            {
                if (int.TryParse(mapParts[partIndex].Trim(), out int mapIndex))
                {
                    maps.Add(mapIndex);
                }
            }
            if (maps.Count == 0 || trials < 1)
            {
                return "need at least one map index and one trial";
            }
            var creatures = new List<string>();
            string[] creatureParts = creaturesCsv.Split(',');
            for (int partIndex = 0; partIndex < creatureParts.Length; partIndex++)
            {
                string id = creatureParts[partIndex].Trim();
                if (id.Length > 0)
                {
                    creatures.Add(id);
                }
            }
            EditorSceneManager.OpenScene(RACE_SCENE, OpenSceneMode.Single);
            var job = new Job
            {
                creatures = creatures.ToArray(),
                maps = maps.ToArray(),
                trials = trials,
                step = 0,
                startedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            };
            SessionState.SetString(JOB_KEY, JsonUtility.ToJson(job));
            SessionState.EraseString(REPORT_KEY);
            EditorApplication.isPlaying = true;
            string who = creatures.Count > 0 ? creatures.Count + " creature(s)" : "every raceable creature";
            return $"queued | {who} x {maps.Count} map(s) x {trials} trial(s) - poll Status()";
        }

        public static string Status()
        {
            string pending = SessionState.GetString(JOB_KEY, string.Empty);
            if (!string.IsNullOrEmpty(pending))
            {
                if (_job != null && _job.creatures != null && _job.creatures.Length > 0)
                {
                    return $"running | step {_job.step + 1}/{StepCount()} ({DescribeStep()}) phase {_phase} "
                        + $"| started {_job.startedAt}";
                }
                return "running | starting up";
            }
            string last = SessionState.GetString(REPORT_KEY, string.Empty);
            if (string.IsNullOrEmpty(last) || !File.Exists(last))
            {
                return "idle | no exam this session";
            }
            string summary = Path.ChangeExtension(last, ".md");
            return "done | " + last + "\n" + (File.Exists(summary) ? File.ReadAllText(summary) : string.Empty);
        }

        [InitializeOnLoadMethod]
        private static void Resume()
        {
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return;
            }
            string pending = SessionState.GetString(JOB_KEY, string.Empty);
            if (string.IsNullOrEmpty(pending))
            {
                return;
            }
            _job = JsonUtility.FromJson<Job>(pending);
            _report = new Report { startedAt = _job.startedAt };
            _phase = Phase.WaitScope;
            _phaseStart = EditorApplication.timeSinceStartup;
            EditorApplication.update += Tick;
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
        }

        private static void OnPlayModeChanged(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.ExitingPlayMode)
            {
                Finish("play mode exited early");
            }
        }

        private static void Tick()
        {
            if (!EditorApplication.isPlaying || _job == null)
            {
                return;
            }
            double elapsed = EditorApplication.timeSinceStartup - _phaseStart;
            switch (_phase)
            {
                case Phase.WaitScope:
                    if (TryResolve())
                    {
                        EnterPhase(Phase.Arm);
                    }
                    else if (elapsed > SCOPE_TIMEOUT_SECONDS)
                    {
                        _report.errors.Add("no built LifetimeScope after " + SCOPE_TIMEOUT_SECONDS + " s");
                        Finish("no LifetimeScope");
                    }
                    break;
                case Phase.Arm:
                    ArmTrial();
                    EnterPhase(Phase.WaitRaceStart);
                    break;
                case Phase.WaitRaceStart:
                    if (_raceModel.RaceActive && _raceModel.Racers.Count > 0)
                    {
                        BeginSampling();
                        EnterPhase(Phase.Racing);
                    }
                    else if (elapsed > START_TIMEOUT_SECONDS)
                    {
                        _trial.outcome = "NeverStarted";
                        _report.errors.Add($"{DescribeStep()}: race never became active");
                        EndTrial();
                    }
                    break;
                case Phase.Racing:
                    Sample();
                    if (!_raceModel.RaceActive)
                    {
                        EndTrial();
                    }
                    else if (elapsed > Systems_MapCatalog.Get(CurrentMap()).TimeLimitSeconds + RACE_OVERRUN_MARGIN_SECONDS)
                    {
                        _report.errors.Add($"{DescribeStep()}: harness cut the trial after the map clock");
                        EndTrial();
                    }
                    break;
                case Phase.Cooldown:
                    if (elapsed > COOLDOWN_SECONDS)
                    {
                        _job.step++;
                        SessionState.SetString(JOB_KEY, JsonUtility.ToJson(_job));
                        if (_job.step >= StepCount())
                        {
                            Finish("all trials run");
                        }
                        else
                        {
                            EnterPhase(Phase.Arm);
                        }
                    }
                    break;
            }
        }

        private static void EnterPhase(Phase phase)
        {
            _phase = phase;
            _phaseStart = EditorApplication.timeSinceStartup;
        }

        private static bool TryResolve()
        {
            LifetimeScope scope = UnityEngine.Object.FindFirstObjectByType<LifetimeScope>();
            if (scope == null || scope.Container == null)
            {
                return false;
            }
            _spawn = scope.Container.Resolve<Systems_Spawn>();
            _config = scope.Container.Resolve<RaceConfigModel>();
            _raceModel = scope.Container.Resolve<RaceModel>();
            _warmup = scope.Container.Resolve<Systems_Warmup>();
            _catalog = scope.Container.Resolve<CreatureCatalog>();
            if (_spawn == null || _config == null || _raceModel == null || _warmup == null || _catalog == null)
            {
                return false;
            }
            if (!_warmup.IsComplete)
            {
                return false;
            }
            if (_job.creatures == null || _job.creatures.Length == 0)
            {
                // The menu's default roster: Systems_Spawn.Start enters exactly the
                // creatures that can race on this platform, one each.
                var raceable = new List<string>();
                for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
                {
                    CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                    if (entry.HasBrain && _config.GetCount(entry.id) > 0)
                    {
                        raceable.Add(entry.id);
                    }
                }
                _job.creatures = raceable.ToArray();
                SessionState.SetString(JOB_KEY, JsonUtility.ToJson(_job));
            }
            return _job.creatures.Length > 0;
        }

        // Trial-major order, so a run stopped part-way has still examined everyone.
        private static int StepCount() => _job.creatures.Length * _job.maps.Length * _job.trials;
        private static string CurrentCreature() => _job.creatures[_job.step % _job.creatures.Length];
        private static int CurrentMap() => _job.maps[_job.step / _job.creatures.Length % _job.maps.Length];
        private static int CurrentTrial() => _job.step / (_job.creatures.Length * _job.maps.Length) + 1;

        private static string DescribeStep()
        {
            return $"{CurrentCreature()} on {Systems_MapCatalog.Get(CurrentMap()).DisplayName} trial {CurrentTrial()}";
        }

        private static void ArmTrial()
        {
            string creatureId = CurrentCreature();
            int mapIndex = CurrentMap();
            _trial = new TrialResult
            {
                creatureId = creatureId,
                map = Systems_MapCatalog.Get(mapIndex).DisplayName,
                trial = CurrentTrial(),
                fpsMin = float.MaxValue,
            };
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                string id = _catalog.Entries[entryIndex].id;
                _config.SetCount(id, id == creatureId ? 1 : 0);
            }
            _config.QuirksEnabled = false;
            _config.SetMap(mapIndex);
            _spawn.BeginRacing();
        }

        private static void BeginSampling()
        {
            _body = null;
            string racerId = _raceModel.Racers[0].RacerId;
            RacerView[] views = UnityEngine.Object.FindObjectsByType<RacerView>(FindObjectsSortMode.None);
            for (int viewIndex = 0; viewIndex < views.Length; viewIndex++)
            {
                if (views[viewIndex].RacerId == racerId)
                {
                    _body = views[viewIndex].transform;
                    break;
                }
            }
            if (_body == null)
            {
                _report.errors.Add($"{DescribeStep()}: no RacerView for {racerId}; orientation not sampled");
            }
            else
            {
                // Rest-relative, so a creature authored lying along its body axis still
                // reads 1 when it stands the way it was built to.
                _restInverse = Quaternion.Inverse(_body.rotation);
            }
            _startFixedTime = Time.fixedTime;
            _endFixedTime = -1f;
            _sampledSeconds = 0f;
            _uprightSeconds = 0f;
            _fallenSeconds = 0f;
            _isDown = false;
            _tippedFor = 0f;
            _standingFor = 0f;
            _lastFrame = Time.frameCount;
        }

        private static void Sample()
        {
            if (Time.frameCount == _lastFrame)
            {
                return;
            }
            _lastFrame = Time.frameCount;
            RacerState racer = _raceModel.Racers[0];
            if (racer.Status != RacerStatus.Racing)
            {
                if (_endFixedTime < 0f)
                {
                    _endFixedTime = Time.fixedTime;
                }
                return;
            }
            float unscaled = Time.unscaledDeltaTime;
            if (unscaled > 0f && 1f / unscaled < _trial.fpsMin)
            {
                _trial.fpsMin = 1f / unscaled;
            }
            // A knocked-out body is deactivated; Unity's == catches the destroyed case.
            if (_body == null || !_body.gameObject.activeInHierarchy)
            {
                return;
            }
            float step = Time.deltaTime;
            float uprightness = (_body.rotation * _restInverse * Vector3.up).y;
            _sampledSeconds += step;
            if (uprightness >= UPRIGHT_COS)
            {
                _uprightSeconds += step;
            }
            if (uprightness < FALLEN_COS)
            {
                _fallenSeconds += step;
            }
            if (!_isDown)
            {
                _tippedFor = uprightness < FALLEN_COS ? _tippedFor + step : 0f;
                if (_tippedFor >= FALL_CONFIRM_SECONDS)
                {
                    _isDown = true;
                    _trial.falls++;
                    _standingFor = 0f;
                }
            }
            else
            {
                _standingFor = uprightness >= UPRIGHT_COS ? _standingFor + step : 0f;
                if (_standingFor >= RECOVERY_CONFIRM_SECONDS)
                {
                    _isDown = false;
                    _trial.recoveries++;
                    _tippedFor = 0f;
                }
            }
        }

        private static void EndTrial()
        {
            if (_raceModel.Racers.Count > 0)
            {
                RacerState racer = _raceModel.Racers[0];
                if (_trial.outcome == null)
                {
                    _trial.outcome = racer.Status.ToString();
                }
                _trial.knockout = racer.Status == RacerStatus.Dnf ? racer.Knockout.ToString() : string.Empty;
                _trial.distanceMeters = Mathf.Max(0f, racer.Progress);
                _trial.trackLengthMeters = _raceModel.TrackLengthMeters;
                _trial.raceClockSeconds = _raceModel.ElapsedSeconds;
                float end = _endFixedTime >= 0f ? _endFixedTime : Time.fixedTime;
                _trial.physicsSeconds = end - _startFixedTime;
                _trial.speedMps = _trial.physicsSeconds > 0f ? _trial.distanceMeters / _trial.physicsSeconds : 0f;
                if (_sampledSeconds > 0f)
                {
                    _trial.uprightFraction = _uprightSeconds / _sampledSeconds;
                    _trial.fallenFraction = _fallenSeconds / _sampledSeconds;
                }
            }
            if (_trial.fpsMin == float.MaxValue)
            {
                _trial.fpsMin = 0f;
            }
            _report.trials.Add(_trial);
            Debug.Log($"[WalkExam] {DescribeStep()}: {_trial.outcome} {_trial.knockout} "
                + $"{_trial.distanceMeters:0.0} m at {_trial.speedMps:0.00} m/s, upright {_trial.uprightFraction:P0}, "
                + $"falls {_trial.falls}, got up {_trial.recoveries}");
            _spawn.RequestMenu();
            EnterPhase(Phase.Cooldown);
        }

        private static void Finish(string reason)
        {
            if (_report == null)
            {
                return;
            }
            EditorApplication.update -= Tick;
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            if (_config != null)
            {
                _config.QuirksEnabled = true;
            }
            _report.finishedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " (" + reason + ")";
            BuildCards();

            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string logDir = Path.Combine(projectRoot, "Logs");
            Directory.CreateDirectory(logDir);
            string path = Path.Combine(logDir, "walkexam_" + DateTime.Now.ToString("yyyyMMdd_HHmm") + ".json");
            File.WriteAllText(path, JsonUtility.ToJson(_report, true));
            File.WriteAllText(Path.ChangeExtension(path, ".md"), BuildSummary());
            Debug.Log($"[WalkExam] {reason} - {_report.trials.Count} trial(s) -> {path}");

            SessionState.EraseString(JOB_KEY);
            SessionState.SetString(REPORT_KEY, path);
            _report = null;
            _job = null;
            if (EditorApplication.isPlaying)
            {
                EditorApplication.isPlaying = false;
            }
        }

        /// <summary>One card per creature and map, averaged over its trials.</summary>
        private static void BuildCards()
        {
            var cardsByKey = new Dictionary<string, ReportCard>();
            var knockoutsByKey = new Dictionary<string, Dictionary<string, int>>();
            for (int trialIndex = 0; trialIndex < _report.trials.Count; trialIndex++)
            {
                TrialResult trial = _report.trials[trialIndex];
                string key = trial.creatureId + "|" + trial.map;
                if (!cardsByKey.TryGetValue(key, out ReportCard card))
                {
                    float legLength = LegLengthOf(trial.creatureId);
                    card = new ReportCard
                    {
                        creatureId = trial.creatureId,
                        map = trial.map,
                        legLengthMeters = legLength,
                        targetSpeedMps = Mathf.Sqrt(FROUDE_NUMBER * GRAVITY * legLength),
                    };
                    cardsByKey[key] = card;
                    knockoutsByKey[key] = new Dictionary<string, int>();
                    _report.cards.Add(card);
                }
                card.trials++;
                if (trial.outcome == nameof(RacerStatus.Finished))
                {
                    card.finished++;
                }
                if (!string.IsNullOrEmpty(trial.knockout))
                {
                    Dictionary<string, int> counts = knockoutsByKey[key];
                    counts.TryGetValue(trial.knockout, out int count);
                    counts[trial.knockout] = count + 1;
                }
                card.meanSpeedMps += trial.speedMps;
                card.meanUprightFraction += trial.uprightFraction;
                card.meanFallenFraction += trial.fallenFraction;
                card.falls += trial.falls;
                card.recoveries += trial.recoveries;
            }
            for (int cardIndex = 0; cardIndex < _report.cards.Count; cardIndex++)
            {
                ReportCard card = _report.cards[cardIndex];
                card.meanSpeedMps /= card.trials;
                card.meanUprightFraction /= card.trials;
                card.meanFallenFraction /= card.trials;
                card.speedRatio = card.targetSpeedMps > 0f ? card.meanSpeedMps / card.targetSpeedMps : 0f;
                bool speedOk = Mathf.Abs(card.speedRatio - 1f) <= SPEED_TOLERANCE;
                bool uptimeOk = card.meanUprightFraction > MIN_UPTIME;
                bool fallsOk = card.meanFallenFraction < MAX_FALLEN_FRACTION;
                card.w1Speed = Verdict(speedOk);
                card.w2Uptime = Verdict(uptimeOk);
                card.w3Falls = Verdict(fallsOk);
                bool getUpOk = true;
                if (card.falls == 0)
                {
                    card.w7GetUp = "n/a (never fell)";
                }
                else
                {
                    getUpOk = (float)card.recoveries / card.falls >= MIN_RECOVERY_RATE;
                    card.w7GetUp = Verdict(getUpOk);
                }
                card.passesMeasured = speedOk && uptimeOk && fallsOk && getUpOk;
                card.knockouts = DescribeCounts(knockoutsByKey[card.creatureId + "|" + card.map]);
            }
        }

        private static float LegLengthOf(string creatureId)
        {
            if (_catalog != null)
            {
                for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
                {
                    if (_catalog.Entries[entryIndex].id == creatureId)
                    {
                        return _catalog.Entries[entryIndex].spawnHeight;
                    }
                }
            }
            return 0f;
        }

        private static string Verdict(bool passed) => passed ? "PASS" : "FAIL";

        private static string DescribeCounts(Dictionary<string, int> counts)
        {
            if (counts.Count == 0)
            {
                return "none";
            }
            var text = new StringBuilder();
            foreach (KeyValuePair<string, int> pair in counts)
            {
                if (text.Length > 0)
                {
                    text.Append(", ");
                }
                text.Append(pair.Key).Append(" x").Append(pair.Value);
            }
            return text.ToString();
        }

        private static string BuildSummary()
        {
            var text = new StringBuilder();
            text.Append("# Walking exam — ").Append(_report.startedAt).Append('\n').Append('\n');
            text.Append("Finished ").Append(_report.finishedAt).Append(". Speed target is Froude ")
                .Append(FROUDE_NUMBER.ToString("0.00")).Append(" on the catalogue's rest height. ")
                .Append("Not measured yet: ").Append(string.Join(", ", NotYetMeasured)).Append(".\n\n");
            text.Append("| Creature | Map | Finished | Speed (target) | W1 speed | W2 uptime | W3 fallen | W7 get-up | Knockouts |\n");
            text.Append("|---|---|---|---|---|---|---|---|---|\n");
            for (int cardIndex = 0; cardIndex < _report.cards.Count; cardIndex++)
            {
                ReportCard card = _report.cards[cardIndex];
                text.Append("| ").Append(card.creatureId)
                    .Append(" | ").Append(card.map)
                    .Append(" | ").Append(card.finished).Append('/').Append(card.trials)
                    .Append(" | ").Append(card.meanSpeedMps.ToString("0.00")).Append(" (")
                    .Append(card.targetSpeedMps.ToString("0.00")).Append(") m/s")
                    .Append(" | ").Append(card.w1Speed).Append(' ').Append(card.speedRatio.ToString("P0"))
                    .Append(" | ").Append(card.w2Uptime).Append(' ').Append(card.meanUprightFraction.ToString("P0"))
                    .Append(" | ").Append(card.w3Falls).Append(' ').Append(card.meanFallenFraction.ToString("P0"))
                    .Append(" | ").Append(card.w7GetUp).Append(" (").Append(card.recoveries).Append('/')
                    .Append(card.falls).Append(')')
                    .Append(" | ").Append(card.knockouts).Append(" |\n");
            }
            if (_report.errors.Count > 0)
            {
                text.Append("\n**Harness errors:**\n");
                for (int errorIndex = 0; errorIndex < _report.errors.Count; errorIndex++)
                {
                    text.Append("- ").Append(_report.errors[errorIndex]).Append('\n');
                }
            }
            return text.ToString();
        }
    }
}
#endif
