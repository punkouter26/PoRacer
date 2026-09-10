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
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_SmokeRace.Start(\"0,1,2,3,4,5,6\", 140f);"
    ///         unity command eval --code "return PoRacer.EditorTools.Editor_SmokeRace.Status();"
    ///
    /// The defaults cover all SEVEN maps, and 140 s per step is not generous — it is
    /// the minimum that lets a builder map reach its own 120 s clock. They used to be
    /// five maps at 60 s, which silently meant Acrobat and Apartment were never raced
    /// at all and Lumpy was cut off before it could resolve. A step the harness
    /// abandons reports raceEnded false and proves nothing about race end, the produce
    /// shower or the podium.
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
        //
        // Longer than RacerView.KNOCKDOWN_SECONDS (12), and that is the whole reason
        // for the number. At 8 s this window stopped four seconds short of the
        // knockdown referee's timer, which used to keep counting after the flag — so
        // a finisher that flopped over the line was puffed out of existence and
        // deactivated while the podium camera held on it, and the harness went home
        // before it could see. Keep this above that constant.
        private const float RESULTS_HOLD_SECONDS = 14f;
        // How far into the results hold the podium UI is audited: late enough for the
        // panel's entrance animation to have landed, early enough to be well inside
        // the hold.
        private const float RESULTS_AUDIT_DELAY_SECONDS = 2f;
        // Settling time after the menu's NEXT button is driven, so UI Toolkit has run
        // a layout pass before the racer screen is measured.
        private const float MENU_SETTLE_SECONDS = 0.5f;
        private const string FRUIT_ROOT = "FruitPour";
        // Portrait readability gate, in panel units (dp at the 420 dp reference):
        // Android's body-text and touch-target minimums, and how far into a corner
        // each piece of screen furniture must sit as a fraction of the panel.
        private const float MIN_BODY_DP = 14f;
        private const float MIN_TOUCH_DP = 48f;
        private const float CORNER_FRACTION = 0.34f;
        private const float RACE_AUDIT_DELAY_SECONDS = 4f;

        /// <summary>
        /// Panel height, in panel units, of the phone this layout actually has to fit.
        ///
        /// Every other assertion here is a fraction of the panel and so is aspect-proof.
        /// The vertical budget is not, and the editor hides the problem rather than
        /// showing it: a game view measured at 960 x 2658 resolves to a 420 x 1163
        /// panel, while a 1080 x 2400 handset at ~400 dpi is 432 x 960 dp and, matched
        /// on width to the 420 reference, resolves to 420 x 933. So the editor validates
        /// the menu against a screen 230 units TALLER than the target — which is 25%
        /// more room than the roster will ever have, on the one screen whose whole
        /// design constraint is fitting without scrolling.
        /// </summary>
        private const float HANDSET_PANEL_DP = 933f;

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
            // Racers still active at the end of the results hold. Should equal
            // `racers` minus whoever was knocked out DURING the race; anything the
            // results screen itself removes is a bug (see RESULTS_HOLD_SECONDS).
            public int racersAliveAtResults;
            // Panel height the layout was measured at, and the handset height it is
            // judged against, so a clean run still records which screen it proved.
            public float panelHeightDp;
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
            // The menu's second screen, reached by driving NEXT. It has to be audited
            // separately because it is the one with the roster on it: the map screen
            // is what WaitScope catches, and for months it was all that ever got
            // checked — which is how eight rows of 42 dp count buttons stayed
            // invisible to a touch-target audit.
            MenuRacers,
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
        private static bool _resultsAudited;
        private static Systems_Spawn _spawn;
        private static RaceConfigModel _config;
        private static RaceModel _raceModel;
        private static Systems_Warmup _warmup;
        private static readonly Dictionary<string, LogEntry> LogIndex = new();

        public static string Start(string mapsCsv = "0,1,2,3,4,5,6", float secondsPerRace = 140f)
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
                        AuditUi("menu-map");
                        // Onto the roster screen, through the button a player would
                        // press rather than by poking MenuView's private _mapStep.
                        if (!DriveButton("NEXT"))
                        {
                            Record("error", "ui-audit[menu-map]: no NEXT button to reach the roster screen",
                                string.Empty);
                        }
                        _phase = Phase.MenuRacers;
                        _phaseStart = EditorApplication.timeSinceStartup;
                    }
                    else if (elapsed > SCOPE_TIMEOUT_SECONDS)
                    {
                        Record("error", "smoke: no built LifetimeScope after " + SCOPE_TIMEOUT_SECONDS + " s", "");
                        Finish("no LifetimeScope");
                    }
                    break;
                case Phase.MenuRacers:
                    if (elapsed > MENU_SETTLE_SECONDS)
                    {
                        AuditUi("menu-racers");
                        StartRaceStep();
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
                    // The podium is a screen like any other and was the one never
                    // audited: its RESULTS panel, its three rows and its RACE AGAIN /
                    // MENU buttons had no layout assertion at all.
                    if (!_resultsAudited && elapsed > RESULTS_AUDIT_DELAY_SECONDS)
                    {
                        _resultsAudited = true;
                        AuditUi("results");
                    }
                    if (elapsed > RESULTS_HOLD_SECONDS)
                    {
                        GameObject fruitRoot = GameObject.Find(FRUIT_ROOT);
                        _step.fruitPieces = fruitRoot != null ? fruitRoot.transform.childCount : 0;
                        // Anything the referee scored must still be on the grid. The
                        // knockdown referee used to keep running through this hold and
                        // deactivate a finisher that was lying down, so count what
                        // survived the results screen rather than trusting it did.
                        _step.racersAliveAtResults = CountActiveRacers();
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
            _warmup = scope.Container.Resolve<Systems_Warmup>();
            if (_spawn == null || _config == null || _raceModel == null || _warmup == null)
            {
                return false;
            }
            // Not resolved until the brains are warm. The warm-up is menu-time work by
            // design (the spawner waits on it too), so measuring it as part of step 0
            // would report the cold start this is here to remove.
            return _warmup.IsComplete;
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
            UIDocument[] documents = UnityEngine.Object.FindObjectsByType<UIDocument>(FindObjectsSortMode.None);
            int labels = 0;
            int buttons = 0;
            var found = new Dictionary<string, Rect>();
            Rect panel = default;
            // Split at the same fraction the corner checks use: anything whose centre
            // sits below it is bottom-anchored furniture that moves with the panel,
            // anything above is flow content that does not. The vertical-budget check
            // below needs to tell the two apart.
            float contentBottom = 0f;
            float anchoredTop = float.PositiveInfinity;
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
                    Rect labelBound = label.worldBound;
                    if (IsFlowContent(labelBound, panel) && labelBound.yMax > contentBottom)
                    {
                        contentBottom = labelBound.yMax;
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
                    // Width as well as height, because height alone let the real defect
                    // through: the roster's count cells were 42 x 60 dp and passed this
                    // audit for months. In a horizontal segmented control width is the
                    // axis a finger misses on, and it is the axis nothing was checking.
                    if (bound.width < MIN_TOUCH_DP - 0.5f)
                    {
                        Record("error", $"ui-audit[{when}]: button '{Trim(button.text)}' is {bound.width:0} dp wide, under {MIN_TOUCH_DP}", string.Empty);
                    }
                    if (!string.IsNullOrEmpty(button.name) && button.name.StartsWith("Furniture."))
                    {
                        found[button.name] = bound;
                    }
                    if (IsFlowContent(bound, panel))
                    {
                        if (bound.yMax > contentBottom)
                        {
                            contentBottom = bound.yMax;
                        }
                    }
                    else if (bound.y < anchoredTop)
                    {
                        anchoredTop = bound.y;
                    }
                });
            }
            if (panel.width <= 0f)
            {
                Record("error", "ui-audit[" + when + "]: no UI document on screen", string.Empty);
                return;
            }
            if (_step != null)
            {
                _step.panelHeightDp = panel.height;
            }
            // The five anchors. The menu screens have no MENU button of their own by
            // design; DebugOverlay owns the FPS readout on every screen.
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_TITLE, left: true, top: true);
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_VERSION, left: false, top: false);
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_DBG, left: true, top: false);
            ExpectCentreTop(when, found, panel, UiTheme.FURNITURE_FPS);
            if (when == "race" || when == "results")
            {
                ExpectCorner(when, found, panel, UiTheme.FURNITURE_MENU, left: false, top: true);
            }
            ExpectHandsetFit(when, panel, contentBottom, anchoredTop);
            Debug.Log($"[SmokeRace] ui-audit[{when}]: {labels} labels, {buttons} buttons, "
                + $"{found.Count} furniture anchors, panel {panel.height:0} dp");
        }

        /// <summary>
        /// True for an element that scrolls or flows with the layout, false for the
        /// bottom-anchored furniture. Split on the same fraction the corner checks use.
        /// </summary>
        private static bool IsFlowContent(Rect bound, Rect panel)
        {
            if (panel.height <= 0f || bound.height <= 0f || float.IsNaN(bound.y))
            {
                return false;
            }
            float centreFraction = (bound.center.y - panel.y) / panel.height;
            return centreFraction <= 1f - CORNER_FRACTION;
        }

        /// <summary>
        /// The vertical-budget check the editor cannot make on its own.
        ///
        /// Bottom-anchored controls ride with the panel, so they are always clear here;
        /// flow content does not move, so on a shorter screen the gap between the two
        /// closes by exactly the difference in panel height. A game view 230 units
        /// taller than a handset therefore shows 230 units of clearance that will not
        /// exist on the device — which is how the menu's "whole roster visible without
        /// scrolling" budget could be 44 units from overflowing into the START button
        /// and look comfortable in the editor.
        ///
        /// So the assertion is made against <see cref="HANDSET_PANEL_DP"/>, not against
        /// the panel being rendered: content must clear the furniture with the screen's
        /// surplus height subtracted. A game view already at or below handset height
        /// needs no correction and is checked as it stands.
        /// </summary>
        private static void ExpectHandsetFit(string when, Rect panel, float contentBottom, float anchoredTop)
        {
            if (contentBottom <= 0f || float.IsPositiveInfinity(anchoredTop))
            {
                return;
            }
            float surplus = Mathf.Max(0f, panel.height - HANDSET_PANEL_DP);
            float budget = anchoredTop - surplus;
            if (contentBottom > budget)
            {
                Record("error",
                    $"ui-audit[{when}]: content reaches {contentBottom:0} dp but on a {HANDSET_PANEL_DP:0} dp "
                    + $"handset the bottom controls start at {budget:0} dp "
                    + $"(panel is {panel.height:0} dp, {surplus:0} dp taller than the target)",
                    string.Empty);
                return;
            }
            Debug.Log($"[SmokeRace] ui-audit[{when}]: vertical budget OK — {budget - contentBottom:0} dp "
                + $"spare against a {HANDSET_PANEL_DP:0} dp handset");
        }

        /// <summary>
        /// Presses the first visible button whose text starts with
        /// <paramref name="textPrefix"/>, through the same submit event a real press
        /// raises — so the audit drives the menu the way a player does rather than
        /// reaching into MenuView's private screen state.
        /// </summary>
        private static bool DriveButton(string textPrefix)
        {
            UIDocument[] documents = UnityEngine.Object.FindObjectsByType<UIDocument>(FindObjectsSortMode.None);
            for (int documentIndex = 0; documentIndex < documents.Length; documentIndex++)
            {
                VisualElement root = documents[documentIndex].rootVisualElement;
                if (root == null || root.resolvedStyle.display == DisplayStyle.None)
                {
                    continue;
                }
                Button match = null;
                root.Query<Button>().ForEach(button =>
                {
                    if (match == null && IsShown(button) && !string.IsNullOrEmpty(button.text)
                        && button.text.StartsWith(textPrefix, StringComparison.Ordinal))
                    {
                        match = button;
                    }
                });
                if (match != null)
                {
                    using (NavigationSubmitEvent submit = NavigationSubmitEvent.GetPooled())
                    {
                        submit.target = match;
                        match.SendEvent(submit);
                    }
                    return true;
                }
            }
            return false;
        }

        /// <summary>Racers still on the grid, by active RacerView count.</summary>
        private static int CountActiveRacers()
        {
            RacerView[] views = UnityEngine.Object.FindObjectsByType<RacerView>(FindObjectsSortMode.None);
            int active = 0;
            for (int viewIndex = 0; viewIndex < views.Length; viewIndex++)
            {
                if (views[viewIndex].gameObject.activeInHierarchy)
                {
                    active++;
                }
            }
            return active;
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
            // Per step, so every map audits its own race and results screens rather
            // than the first map's audit standing in for all of them.
            _raceAudited = false;
            _resultsAudited = false;
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
