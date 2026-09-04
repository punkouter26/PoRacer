#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using PoRacer.Models;
using PoRacer.Systems;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using VContainer;
using VContainer.Unity;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Play-mode smoke run driven from the CLI. <see cref="Start"/> races every
    /// selected map in SCN_RACE_FLAT with the default roster; <see cref="StartScene"/>
    /// just plays one scene for a while. Both sample frame time, collect every
    /// error and exception the log sees, and write Logs/smoke_&lt;stamp&gt;.json.
    ///
    /// Entering play mode reloads the domain, so the job lives in SessionState and
    /// <see cref="Resume"/> re-arms the driver on the other side of the reload.
    ///
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_SmokeRace.Start(\"0,1,2,3,4\", 60f);"
    ///         unity command eval --code "return PoRacer.EditorTools.Editor_SmokeRace.Status();"
    /// </summary>
    public static class Editor_SmokeRace
    {
        private const string JOB_KEY = "PoRacer.SmokeRace.Job";
        private const string REPORT_KEY = "PoRacer.SmokeRace.LastReport";
        private const string RACE_SCENE = "Assets/Scenes/SCN_RACE_FLAT.unity";
        // URP re-warns this once per shader array per frame after a build-target
        // switch until the editor restarts; it is editor noise, not a game error.
        private const string URP_ARRAY_NOISE = "exceeds previous array size";
        private const float SCOPE_TIMEOUT_SECONDS = 20f;
        private const float START_TIMEOUT_SECONDS = 30f;
        private const float COOLDOWN_SECONDS = 2f;

        [Serializable]
        private sealed class Job
        {
            public string scenePath;
            public int[] maps;
            public float secondsPerStep;
            public int step;
            public string startedAt;
        }

        [Serializable]
        private sealed class StepReport
        {
            public string name;
            public float seconds;
            public int frames;
            public float fpsAvg;
            public float fpsMin;
            public bool raceStarted;
            public bool raceEnded;
            public int racers;
            public int finished;
            public int timedOut;
            public int dnf;
            public string[] placings;
        }

        [Serializable]
        private sealed class LogEntry
        {
            public string type;
            public int count;
            public string message;
            public string stackTop;
        }

        [Serializable]
        private sealed class Report
        {
            public string scene;
            public string startedAt;
            public string finishedAt;
            public int errorCount;
            public int warningCount;
            public List<StepReport> steps = new();
            public List<LogEntry> errors = new();
            public List<LogEntry> warnings = new();
        }

        private enum Phase
        {
            WaitScope,
            WaitRaceStart,
            Racing,
            Cooldown,
            Playing,
        }

        private static Job _job;
        private static Report _report;
        private static StepReport _step;
        private static Phase _phase;
        private static double _phaseStart;
        private static int _lastFrame;
        private static Systems_Spawn _spawn;
        private static RaceConfigModel _config;
        private static RaceModel _raceModel;
        private static readonly Dictionary<string, LogEntry> LogIndex = new();

        public static string Start(string mapsCsv = "0,1,2,3,4", float secondsPerRace = 60f)
        {
            string[] parts = mapsCsv.Split(',');
            var maps = new List<int>();
            for (int partIndex = 0; partIndex < parts.Length; partIndex++)
            {
                if (int.TryParse(parts[partIndex].Trim(), out int mapIndex))
                {
                    maps.Add(mapIndex);
                }
            }
            if (maps.Count == 0)
            {
                return "no map indices in \"" + mapsCsv + "\"";
            }
            return Launch(RACE_SCENE, maps.ToArray(), secondsPerRace);
        }

        public static string StartScene(string scenePath, float seconds = 30f)
        {
            return Launch(scenePath, Array.Empty<int>(), seconds);
        }

        public static string Status()
        {
            string pending = SessionState.GetString(JOB_KEY, string.Empty);
            if (!string.IsNullOrEmpty(pending))
            {
                Job job = JsonUtility.FromJson<Job>(pending);
                return $"running | {job.scenePath} step {job.step + 1}/{Math.Max(1, job.maps.Length)} " +
                    $"phase {_phase} | started {job.startedAt}";
            }
            string last = SessionState.GetString(REPORT_KEY, string.Empty);
            if (string.IsNullOrEmpty(last) || !File.Exists(last))
            {
                return "idle | no report this session";
            }
            return "done | " + last + "\n" + File.ReadAllText(last);
        }

        private static string Launch(string scenePath, int[] maps, float seconds)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "already in play mode - stop it first";
            }
            if (!File.Exists(scenePath))
            {
                return "no scene at " + scenePath;
            }
            if (EditorSceneManager.GetActiveScene().isDirty)
            {
                return "the open scene has unsaved changes - save or discard them first " +
                    "(opening another scene would raise a modal prompt)";
            }
            EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            var job = new Job
            {
                scenePath = scenePath,
                maps = maps,
                secondsPerStep = seconds,
                step = 0,
                startedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
            };
            SessionState.SetString(JOB_KEY, JsonUtility.ToJson(job));
            SessionState.EraseString(REPORT_KEY);
            EditorApplication.isPlaying = true;
            return $"queued | {scenePath} | {maps.Length} race step(s), {seconds:0}s each - poll Status()";
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
            _report = new Report { scene = _job.scenePath, startedAt = _job.startedAt };
            LogIndex.Clear();
            _phase = _job.maps.Length > 0 ? Phase.WaitScope : Phase.Playing;
            _phaseStart = EditorApplication.timeSinceStartup;
            _lastFrame = Time.frameCount;
            if (_phase == Phase.Playing)
            {
                BeginStep(Path.GetFileNameWithoutExtension(_job.scenePath));
            }
            Application.logMessageReceived += OnLog;
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
            if (!EditorApplication.isPlaying)
            {
                return;
            }
            SampleFrame();
            double elapsed = EditorApplication.timeSinceStartup - _phaseStart;
            switch (_phase)
            {
                case Phase.WaitScope:
                    if (TryResolve())
                    {
                        StartRaceStep();
                    }
                    else if (elapsed > SCOPE_TIMEOUT_SECONDS)
                    {
                        Record("error", "smoke: no built LifetimeScope after " + SCOPE_TIMEOUT_SECONDS + " s", "");
                        Finish("no LifetimeScope");
                    }
                    break;
                case Phase.WaitRaceStart:
                    if (_raceModel.RaceActive)
                    {
                        _step.raceStarted = true;
                        _phase = Phase.Racing;
                        _phaseStart = EditorApplication.timeSinceStartup;
                    }
                    else if (elapsed > START_TIMEOUT_SECONDS)
                    {
                        Record("error", $"smoke: race on {_step.name} never became active", "");
                        EndRaceStep();
                    }
                    break;
                case Phase.Racing:
                    if (!_raceModel.RaceActive)
                    {
                        _step.raceEnded = true;
                        EndRaceStep();
                    }
                    else if (elapsed > _job.secondsPerStep)
                    {
                        EndRaceStep();
                    }
                    break;
                case Phase.Cooldown:
                    if (elapsed > COOLDOWN_SECONDS)
                    {
                        _job.step++;
                        SessionState.SetString(JOB_KEY, JsonUtility.ToJson(_job));
                        if (_job.step >= _job.maps.Length)
                        {
                            Finish("all maps raced");
                        }
                        else
                        {
                            StartRaceStep();
                        }
                    }
                    break;
                case Phase.Playing:
                    if (elapsed > _job.secondsPerStep)
                    {
                        CloseStep();
                        Finish("scene played");
                    }
                    break;
            }
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
            return _spawn != null && _config != null && _raceModel != null;
        }

        private static void StartRaceStep()
        {
            int mapIndex = _job.maps[_job.step];
            Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Get(mapIndex);
            BeginStep($"{mapIndex}:{map.DisplayName}");
            _config.SetMap(mapIndex);
            _spawn.BeginRacing();
            _phase = Phase.WaitRaceStart;
            _phaseStart = EditorApplication.timeSinceStartup;
        }

        private static void EndRaceStep()
        {
            _step.racers = _raceModel.Racers.Count;
            var placings = new List<string>();
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                switch (racer.Status)
                {
                    case RacerStatus.Finished: _step.finished++; break;
                    case RacerStatus.TimedOut: _step.timedOut++; break;
                    case RacerStatus.Dnf: _step.dnf++; break;
                }
                placings.Add($"{racer.DisplayName} {racer.Status} {racer.Progress:0.0}m" +
                    (racer.Place > 0 ? $" P{racer.Place}" : string.Empty));
            }
            _step.placings = placings.ToArray();
            CloseStep();
            _spawn.RequestMenu();
            _phase = Phase.Cooldown;
            _phaseStart = EditorApplication.timeSinceStartup;
        }

        private static void BeginStep(string name)
        {
            _step = new StepReport { name = name, fpsMin = float.MaxValue };
            _report.steps.Add(_step);
        }

        private static void CloseStep()
        {
            if (_step == null)
            {
                return;
            }
            _step.seconds = (float)(EditorApplication.timeSinceStartup - _phaseStart);
            if (_step.frames > 0)
            {
                _step.fpsAvg = _step.frames / Mathf.Max(_step.seconds, 0.001f);
            }
            if (_step.fpsMin == float.MaxValue)
            {
                _step.fpsMin = 0f;
            }
            _step = null;
        }

        private static void SampleFrame()
        {
            if (_step == null || Time.frameCount == _lastFrame)
            {
                return;
            }
            _lastFrame = Time.frameCount;
            _step.frames++;
            float dt = Time.unscaledDeltaTime;
            if (dt > 0f)
            {
                float fps = 1f / dt;
                if (fps < _step.fpsMin)
                {
                    _step.fpsMin = fps;
                }
            }
        }

        private static void OnLog(string message, string stackTrace, LogType type)
        {
            if (type == LogType.Log)
            {
                return;
            }
            if (message.Contains(URP_ARRAY_NOISE))
            {
                return;
            }
            string top = string.Empty;
            if (!string.IsNullOrEmpty(stackTrace))
            {
                int newline = stackTrace.IndexOf('\n');
                top = newline > 0 ? stackTrace.Substring(0, newline) : stackTrace;
            }
            Record(type == LogType.Warning ? "warning" : type.ToString().ToLowerInvariant(), message, top);
        }

        private static void Record(string type, string message, string stackTop)
        {
            string key = type + "|" + message;
            if (LogIndex.TryGetValue(key, out LogEntry existing))
            {
                existing.count++;
            }
            else
            {
                existing = new LogEntry { type = type, count = 1, message = message, stackTop = stackTop };
                LogIndex[key] = existing;
                if (type == "warning")
                {
                    _report.warnings.Add(existing);
                }
                else
                {
                    _report.errors.Add(existing);
                }
            }
            if (type == "warning")
            {
                _report.warningCount++;
            }
            else
            {
                _report.errorCount++;
            }
        }

        private static void Finish(string reason)
        {
            if (_report == null)
            {
                return;
            }
            CloseStep();
            EditorApplication.update -= Tick;
            Application.logMessageReceived -= OnLog;
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            _report.finishedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " (" + reason + ")";

            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string logDir = Path.Combine(projectRoot, "Logs");
            Directory.CreateDirectory(logDir);
            string path = Path.Combine(logDir, "smoke_" + DateTime.Now.ToString("yyyyMMdd_HHmm") + ".json");
            File.WriteAllText(path, JsonUtility.ToJson(_report, true));

            var summary = new StringBuilder();
            summary.Append("[SmokeRace] ").Append(reason).Append(" - ").Append(_report.errorCount)
                .Append(" errors, ").Append(_report.warningCount).Append(" warnings -> ").Append(path);
            Debug.Log(summary.ToString());

            SessionState.EraseString(JOB_KEY);
            SessionState.SetString(REPORT_KEY, path);
            _report = null;
            _job = null;
            if (EditorApplication.isPlaying)
            {
                EditorApplication.isPlaying = false;
            }
        }
    }
}
#endif
