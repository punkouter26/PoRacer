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
using UnityEngine.UIElements;
using PoRacer.Presentation;
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
        // Results stay up this long before the menu is requested, so what happens
        // at race end (the produce shower) is exercised and counted.
        private const float RESULTS_HOLD_SECONDS = 8f;
        private const string FRUIT_ROOT = "FruitPour";
        // Portrait readability gate, in panel units (dp at the 420 dp reference):
        // Android's body-text and touch-target minimums, and how far into a corner
        // each piece of screen furniture must sit as a fraction of the panel.
        private const float MIN_BODY_DP = 14f;
        private const float MIN_TOUCH_DP = 48f;
        private const float CORNER_FRACTION = 0.34f;
        private const float RACE_AUDIT_DELAY_SECONDS = 4f;

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
            public int fruitPieces;
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
            Results,
            Cooldown,
            Playing,
        }

        private static Job _job;
        private static Report _report;
        private static StepReport _step;
        private static Phase _phase;
        private static double _phaseStart;
        private static double _stepStart;
        private static int _lastFrame;
        private static bool _raceAudited;
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
                        AuditUi("menu");
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
                    if (!_raceAudited && elapsed > RACE_AUDIT_DELAY_SECONDS)
                    {
                        _raceAudited = true;
                        AuditUi("race");
                    }
                    if (!_raceModel.RaceActive)
                    {
                        _step.raceEnded = true;
                        _phase = Phase.Results;
                        _phaseStart = EditorApplication.timeSinceStartup;
                    }
                    else if (elapsed > _job.secondsPerStep)
                    {
                        EndRaceStep();
                    }
                    break;
                case Phase.Results:
                    if (elapsed > RESULTS_HOLD_SECONDS)
                    {
                        GameObject fruitRoot = GameObject.Find(FRUIT_ROOT);
                        _step.fruitPieces = fruitRoot != null ? fruitRoot.transform.childCount : 0;
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

        /// <summary>
        /// Walks every visible label and button in every UI document on screen and
        /// records a violation for text under MIN_BODY_DP, controls under
        /// MIN_TOUCH_DP, and any of the five furniture anchors out of its corner.
        /// Violations land in the report as errors, so a build that breaks the
        /// HUD layout fails the smoke run the same way an exception would.
        /// </summary>
        private static void AuditUi(string when)
        {
            _raceAudited = when == "race";
            UIDocument[] documents = UnityEngine.Object.FindObjectsByType<UIDocument>(FindObjectsSortMode.None);
            int labels = 0;
            int buttons = 0;
            var found = new Dictionary<string, Rect>();
            Rect panel = default;
            for (int documentIndex = 0; documentIndex < documents.Length; documentIndex++)
            {
                VisualElement root = documents[documentIndex].rootVisualElement;
                if (root == null || root.resolvedStyle.display == DisplayStyle.None)
                {
                    continue;
                }
                if (panel.width <= 0f)
                {
                    panel = root.worldBound;
                }
                root.Query<Label>().ForEach(label =>
                {
                    if (!IsShown(label) || string.IsNullOrEmpty(label.text))
                    {
                        return;
                    }
                    labels++;
                    float size = label.resolvedStyle.fontSize;
                    if (size < MIN_BODY_DP - 0.01f)
                    {
                        Record("error", $"ui-audit[{when}]: label '{Trim(label.text)}' is {size:0.0} dp, under {MIN_BODY_DP}", string.Empty);
                    }
                    if (!string.IsNullOrEmpty(label.name) && label.name.StartsWith("Furniture."))
                    {
                        found[label.name] = label.worldBound;
                    }
                });
                root.Query<Button>().ForEach(button =>
                {
                    if (!IsShown(button))
                    {
                        return;
                    }
                    buttons++;
                    Rect bound = button.worldBound;
                    if (bound.height < MIN_TOUCH_DP - 0.5f)
                    {
                        Record("error", $"ui-audit[{when}]: button '{Trim(button.text)}' is {bound.height:0} dp tall, under {MIN_TOUCH_DP}", string.Empty);
                    }
                    if (!string.IsNullOrEmpty(button.name) && button.name.StartsWith("Furniture."))
                    {
                        found[button.name] = bound;
                    }
                });
            }
            if (panel.width <= 0f)
            {
                Record("error", "ui-audit[" + when + "]: no UI document on screen", string.Empty);
                return;
            }
            // The five anchors. The menu screen has no MENU button or FPS readout of
            // its own by design (DebugOverlay owns FPS on every screen).
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_TITLE, left: true, top: true);
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_VERSION, left: false, top: false);
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_DBG, left: true, top: false);
            ExpectCentreTop(when, found, panel, UiTheme.FURNITURE_FPS);
            if (when == "race")
            {
                ExpectCorner(when, found, panel, UiTheme.FURNITURE_MENU, left: false, top: true);
            }
            Debug.Log($"[SmokeRace] ui-audit[{when}]: {labels} labels, {buttons} buttons, {found.Count} furniture anchors checked");
        }

        private static bool IsShown(VisualElement element)
        {
            if (element.resolvedStyle.display == DisplayStyle.None || element.resolvedStyle.visibility == Visibility.Hidden)
            {
                return false;
            }
            Rect bound = element.worldBound;
            if (bound.width <= 0f || bound.height <= 0f || float.IsNaN(bound.x))
            {
                return false;
            }
            for (VisualElement parent = element.parent; parent != null; parent = parent.parent)
            {
                if (parent.resolvedStyle.display == DisplayStyle.None)
                {
                    return false;
                }
            }
            return true;
        }

        private static string Trim(string text)
        {
            text = (text ?? string.Empty).Replace('\n', ' ');
            return text.Length > 32 ? text.Substring(0, 32) + "..." : text;
        }

        private static void ExpectCorner(string when, Dictionary<string, Rect> found, Rect panel, string name, bool left, bool top)
        {
            if (!found.TryGetValue(name, out Rect bound))
            {
                Record("error", $"ui-audit[{when}]: {name} is not on screen", string.Empty);
                return;
            }
            float cx = (bound.center.x - panel.x) / panel.width;
            float cy = (bound.center.y - panel.y) / panel.height;
            bool okX = left ? cx < CORNER_FRACTION : cx > 1f - CORNER_FRACTION;
            bool okY = top ? cy < CORNER_FRACTION : cy > 1f - CORNER_FRACTION;
            if (!okX || !okY)
            {
                Record("error", $"ui-audit[{when}]: {name} sits at ({cx:0.00}, {cy:0.00}) of the panel, expected {(top ? "top" : "bottom")}-{(left ? "left" : "right")}", string.Empty);
            }
        }

        private static void ExpectCentreTop(string when, Dictionary<string, Rect> found, Rect panel, string name)
        {
            if (!found.TryGetValue(name, out Rect bound))
            {
                Record("error", $"ui-audit[{when}]: {name} is not on screen", string.Empty);
                return;
            }
            float cx = (bound.center.x - panel.x) / panel.width;
            float cy = (bound.center.y - panel.y) / panel.height;
            if (cx < CORNER_FRACTION || cx > 1f - CORNER_FRACTION || cy > CORNER_FRACTION)
            {
                Record("error", $"ui-audit[{when}]: {name} sits at ({cx:0.00}, {cy:0.00}) of the panel, expected top-centre", string.Empty);
            }
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
            _stepStart = EditorApplication.timeSinceStartup;
            _report.steps.Add(_step);
        }

        private static void CloseStep()
        {
            if (_step == null)
            {
                return;
            }
            _step.seconds = (float)(EditorApplication.timeSinceStartup - _stepStart);
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
