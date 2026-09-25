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
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_SmokeRace.Start(\"0,1,2\", 140f);"
    ///         unity command eval --code "return PoRacer.EditorTools.Editor_SmokeRace.Status();"
    ///
    /// The defaults cover all THREE maps — Flat, Acrobat, Apartment — since Lumpy,
    /// Swamp, Gale and Roulette were removed on 2026-09-11. The CSV is map INDICES, so
    /// it must be re-checked whenever Systems_MapCatalog.Entries changes: an index past
    /// the end does not fail, it clamps back to Flat, so a stale CSV silently races the
    /// first map several extra times instead of erroring.
    ///
    /// `secondsPerRace` is a FLOOR, not the budget — see RaceBudgetSeconds(). A step
    /// always gets at least its own map's clock, because a harness that stops a race
    /// before its map can resolve proves nothing about race end, the produce shower or
    /// the podium. At 140 s flat that is exactly what happened to both courses.
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
        // Headroom past a map's own clock for the referee to classify the finish.
        private const float RACE_OVERRUN_MARGIN_SECONDS = 20f;
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
        private const string FRUIT_ROOT = "FruitPour";
        // Portrait readability gate, in panel units (dp at the 420 dp reference):
        // Android's body-text and touch-target minimums, and how far into a corner
        // each piece of screen furniture must sit as a fraction of the panel.
        private const float MIN_BODY_DP = 14f;
        private const float MIN_TOUCH_DP = 48f;
        private const float CORNER_FRACTION = 0.34f;
        private const float RACE_AUDIT_DELAY_SECONDS = 4f;

        /// <summary>
        /// The phones this layout has to fit, as dp screen sizes. The panel matches on
        /// width to a 420-unit reference, so each resolves to 420 x (420 * h / w) panel
        /// units, and that height is what the vertical checks are made against.
        ///
        /// Every other assertion here is a fraction of the panel and so is aspect-proof.
        /// The vertical budget is not, and the editor hides the problem rather than
        /// showing it: a game view measured at 960 x 2658 resolves to a 420 x 1163
        /// panel, while a 1080 x 2400 handset at ~400 dpi is 432 x 960 dp and resolves
        /// to 420 x 933. So the editor validates the menu against a screen 230 units
        /// TALLER than the target. One size used to be checked (the 432 x 960 phone);
        /// the zero-scroll rule has to hold on the short end of the range too, where a
        /// 360 x 640 dp handset leaves only 747 units.
        ///
        /// 411 x 960 is the 21:9 end (a 1080 x 2520 handset at ~420 dpi), inside the
        /// Android build's 2.4 maximum aspect ratio.
        /// </summary>
        private static readonly Vector2[] HandsetsDp =
        {
            new(360f, 640f),
            new(390f, 844f),
            new(411f, 960f),
            new(432f, 960f),
        };

        /// <summary>Panel reference width; must match RaceHudPanelSettings and UiTheme.</summary>
        private const float PANEL_REFERENCE_WIDTH = 420f;

        // Slack for sub-pixel layout rounding before an edge counts as crossed.
        private const float EDGE_TOLERANCE_DP = 1f;

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
            // Worst frame while actually RACING, as opposed to fpsMin which covers
            // the whole step including the menu. They differ a lot and the difference
            // was misread: on 2026-09-11 step 0 reported fpsMin 0.83 (a 1.2 s frame)
            // and it was taken for a race stall, when the spawn stages showed
            // `instantiate grid: 0.062 s`. That long frame belongs to Systems_Warmup
            // instantiating a heavy prefab on the MENU -- which is the warm-up's whole
            // purpose, since it is a hitch paid before START instead of mid-race.
            // Attribute it, so a menu cost cannot read as a gameplay regression.
            public float fpsMinRace;
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
            // Race clock when the referee stopped, and the map's own limit, so a run
            // says WHY each race ended rather than only that it did. A race that stops
            // far short of its limit with exactly PODIUM_FINISHERS finishers ended on
            // the podium cutoff, which DNFs everyone still upright and running.
            public float raceEndedAtSeconds;
            public float timeLimitSeconds;
            public string endReason;
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

        public static string Start(string mapsCsv = "0,1,2", float secondsPerRace = 140f)
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
                        // One menu screen now: the map tabs and the roster together.
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
                        // Captured at the transition: Advance() stops once RaceActive
                        // clears, so this is the clock reading the referee stopped on.
                        _step.raceEndedAtSeconds = _raceModel.ElapsedSeconds;
                        _phase = Phase.Results;
                        _phaseStart = EditorApplication.timeSinceStartup;
                    }
                    else if (elapsed > RaceBudgetSeconds())
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

        /// <summary>How an element takes part in the vertical budget.</summary>
        private enum Role
        {
            /// <summary>Judged by where it sits: top two thirds flow, bottom third anchored.</summary>
            ByPosition,
            /// <summary>Inside a bottom-anchored sheet that grows upward: moves with the panel.</summary>
            Anchored,
            /// <summary>Moves with the race (rail badges): no fixed place to budget for.</summary>
            Exempt,
        }

        /// <summary>One shown label or button, as the audit measured it.</summary>
        private readonly struct Measured
        {
            public readonly string Name;
            public readonly string Text;
            public readonly Rect Bound;
            public readonly bool IsButton;
            public readonly bool InScroll;
            public readonly bool Settled;
            public readonly Role Role;

            public Measured(string name, string text, Rect bound, bool isButton, bool inScroll, bool settled, Role role)
            {
                Name = name;
                Text = text;
                Bound = bound;
                IsButton = isButton;
                InScroll = inScroll;
                Settled = settled;
                Role = role;
            }
        }

        /// <summary>
        /// Walks every visible label and button in every UI document on screen and
        /// records a violation for:
        ///   * text under MIN_BODY_DP and controls under MIN_TOUCH_DP;
        ///   * any of the five furniture anchors out of its corner;
        ///   * anything drawn past the edge of the screen;
        ///   * any other element overlapping a furniture anchor;
        ///   * a ScrollView whose content does not fit its viewport, on each handset;
        ///   * flow content running into the bottom controls, on each handset.
        /// Violations land in the report as errors, so a build that breaks the HUD
        /// layout fails the smoke run the same way an exception would.
        /// </summary>
        private static void AuditUi(string when)
        {
            UIDocument[] documents = UnityEngine.Object.FindObjectsByType<UIDocument>(FindObjectsSortMode.None);
            var measured = new List<Measured>();
            var scrolls = new List<ScrollView>();
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
                    if (IsShown(label) && !string.IsNullOrEmpty(label.text))
                    {
                        measured.Add(Measure(label, label.text, isButton: false));
                    }
                });
                root.Query<Button>().ForEach(button =>
                {
                    if (IsShown(button))
                    {
                        measured.Add(Measure(button, button.text, isButton: true));
                    }
                });
                root.Query<ScrollView>().ForEach(scroll =>
                {
                    if (IsShown(scroll))
                    {
                        scrolls.Add(scroll);
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

            // Split at the same fraction the corner checks use: anything whose centre
            // sits below it is bottom-anchored furniture that moves with the panel,
            // anything above is flow content that does not. The vertical-budget check
            // needs to tell the two apart.
            var found = new Dictionary<string, Rect>();
            float contentBottom = 0f;
            float anchoredTop = float.PositiveInfinity;
            int labels = 0;
            int buttons = 0;
            for (int index = 0; index < measured.Count; index++)
            {
                Measured item = measured[index];
                if (IsFurniture(item.Name))
                {
                    found[item.Name] = item.Bound;
                }
                if (!item.IsButton)
                {
                    labels++;
                    if (IsFlowContent(item, panel) && item.Bound.yMax > contentBottom)
                    {
                        contentBottom = item.Bound.yMax;
                    }
                    continue;
                }
                buttons++;
                Rect bound = item.Bound;
                if (bound.height < MIN_TOUCH_DP - 0.5f)
                {
                    Record("error", $"ui-audit[{when}]: button '{Trim(item.Text)}' is {bound.height:0} dp tall, under {MIN_TOUCH_DP}", string.Empty);
                }
                // Width as well as height, because height alone let the real defect
                // through: the roster's count cells were 42 x 60 dp and passed this
                // audit for months. In a horizontal segmented control width is the
                // axis a finger misses on, and it is the axis nothing was checking.
                if (bound.width < MIN_TOUCH_DP - 0.5f)
                {
                    Record("error", $"ui-audit[{when}]: button '{Trim(item.Text)}' is {bound.width:0} dp wide, under {MIN_TOUCH_DP}", string.Empty);
                }
                if (IsFlowContent(item, panel))
                {
                    if (bound.yMax > contentBottom)
                    {
                        contentBottom = bound.yMax;
                    }
                }
                else if (item.Role != Role.Exempt && bound.y < anchoredTop)
                {
                    anchoredTop = bound.y;
                }
            }
            ExpectReadableText(when, documents);

            // The five anchors. The menu screen has no MENU button of its own on a
            // device that cannot vibrate, but wherever one is drawn it must sit in its
            // corner; DebugOverlay owns the FPS readout everywhere.
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_TITLE, left: true, top: true);
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_VERSION, left: false, top: false);
            ExpectCorner(when, found, panel, UiTheme.FURNITURE_DBG, left: true, top: false);
            ExpectCentreTop(when, found, panel, UiTheme.FURNITURE_FPS);
            if (when == "race" || when == "results" || found.ContainsKey(UiTheme.FURNITURE_MENU))
            {
                ExpectCorner(when, found, panel, UiTheme.FURNITURE_MENU, left: false, top: true);
            }
            ExpectOnScreen(when, measured, panel);
            ExpectClearOfFurniture(when, measured, found);
            float liveControlScale = LiveControlScale();
            for (int handsetIndex = 0; handsetIndex < HandsetsDp.Length; handsetIndex++)
            {
                Vector2 handset = HandsetsDp[handsetIndex];
                float handsetPanel = PANEL_REFERENCE_WIDTH * handset.y / handset.x;
                string label = $"{handset.x:0}x{handset.y:0}";
                float sizeRatio = UiTheme.ControlScaleForDeviceWidth(handset.x) / liveControlScale;
                ExpectHandsetFit(when, label, handsetPanel, panel, contentBottom, anchoredTop, sizeRatio);
                ExpectNoScroll(when, label, handsetPanel, panel, scrolls, sizeRatio);
            }
            Debug.Log($"[SmokeRace] ui-audit[{when}]: {labels} labels, {buttons} buttons, {scrolls.Count} scroll views, "
                + $"{found.Count} furniture anchors, panel {panel.height:0} dp");
        }

        private static Measured Measure(VisualElement element, string text, bool isButton)
        {
            bool inScroll = false;
            bool settled = true;
            Role role = Role.ByPosition;
            for (VisualElement node = element; node != null; node = node.parent)
            {
                if (node is ScrollView)
                {
                    inScroll = true;
                }
                // By container, not by where it happens to be on screen: a rail badge
                // crosses the one-third line as the race runs, and a results sheet
                // grown upward from the bottom band straddles it.
                if (node.name == UiTheme.PROGRESS_RAIL)
                {
                    role = Role.Exempt;
                }
                else if (node.name == UiTheme.RESULTS_PANEL && role == Role.ByPosition)
                {
                    role = Role.Anchored;
                }
                // Mid-animation (fading or sliding in) an element is not where it will
                // rest, so edge and overlap checks skip it rather than report a frame.
                if (node.resolvedStyle.opacity < 0.99f)
                {
                    settled = false;
                }
                Translate translate = node.resolvedStyle.translate;
                if (Mathf.Abs(translate.x.value) > 0.5f || Mathf.Abs(translate.y.value) > 0.5f)
                {
                    settled = false;
                }
                // The banner pops in from 1.6x scale.
                Vector3 scale = node.resolvedStyle.scale.value;
                if (Mathf.Abs(scale.x - 1f) > 0.01f || Mathf.Abs(scale.y - 1f) > 0.01f)
                {
                    settled = false;
                }
            }
            return new Measured(element.name, text, element.worldBound, isButton, inScroll, settled, role);
        }

        private static bool IsFurniture(string name)
        {
            return !string.IsNullOrEmpty(name) && name.StartsWith("Furniture.", StringComparison.Ordinal);
        }

        /// <summary>Body-text floor: no shown label renders under Android's 14 sp.</summary>
        private static void ExpectReadableText(string when, UIDocument[] documents)
        {
            for (int documentIndex = 0; documentIndex < documents.Length; documentIndex++)
            {
                VisualElement root = documents[documentIndex].rootVisualElement;
                if (root == null || root.resolvedStyle.display == DisplayStyle.None)
                {
                    continue;
                }
                root.Query<Label>().ForEach(label =>
                {
                    if (!IsShown(label) || string.IsNullOrEmpty(label.text))
                    {
                        return;
                    }
                    float size = label.resolvedStyle.fontSize;
                    if (size < MIN_BODY_DP - 0.01f)
                    {
                        Record("error", $"ui-audit[{when}]: label '{Trim(label.text)}' is {size:0.0} dp, under {MIN_BODY_DP}", string.Empty);
                    }
                });
            }
        }

        /// <summary>
        /// Nothing may be drawn past the screen edge. Scroll content is left to
        /// <see cref="ExpectNoScroll"/>, which reports the cause rather than each row.
        /// </summary>
        private static void ExpectOnScreen(string when, List<Measured> measured, Rect panel)
        {
            for (int index = 0; index < measured.Count; index++)
            {
                Measured item = measured[index];
                if (item.InScroll || !item.Settled)
                {
                    continue;
                }
                Rect bound = item.Bound;
                bool off = bound.xMin < panel.xMin - EDGE_TOLERANCE_DP
                    || bound.xMax > panel.xMax + EDGE_TOLERANCE_DP
                    || bound.yMin < panel.yMin - EDGE_TOLERANCE_DP
                    || bound.yMax > panel.yMax + EDGE_TOLERANCE_DP;
                if (off)
                {
                    Record("error", $"ui-audit[{when}]: '{Trim(item.Text)}' runs off screen "
                        + $"({bound.xMin:0},{bound.yMin:0})-({bound.xMax:0},{bound.yMax:0}) "
                        + $"in a {panel.width:0}x{panel.height:0} panel", string.Empty);
                }
            }
        }

        /// <summary>
        /// The five anchors own their corners: no other label or button may be drawn
        /// over one of them.
        /// </summary>
        private static void ExpectClearOfFurniture(string when, List<Measured> measured, Dictionary<string, Rect> found)
        {
            foreach (KeyValuePair<string, Rect> anchor in found)
            {
                for (int index = 0; index < measured.Count; index++)
                {
                    Measured item = measured[index];
                    if (IsFurniture(item.Name) || !item.Settled)
                    {
                        continue;
                    }
                    Rect overlap = Rect.MinMaxRect(
                        Mathf.Max(item.Bound.xMin, anchor.Value.xMin),
                        Mathf.Max(item.Bound.yMin, anchor.Value.yMin),
                        Mathf.Min(item.Bound.xMax, anchor.Value.xMax),
                        Mathf.Min(item.Bound.yMax, anchor.Value.yMax));
                    if (overlap.width > EDGE_TOLERANCE_DP && overlap.height > EDGE_TOLERANCE_DP)
                    {
                        Record("error", $"ui-audit[{when}]: '{Trim(item.Text)}' overlaps {anchor.Key}", string.Empty);
                    }
                }
            }
        }

        /// <summary>
        /// True for an element that scrolls or flows with the layout, false for the
        /// bottom-anchored furniture. Elements whose container settles it are classed
        /// by that (see <see cref="Measure"/>); the rest split on the same fraction the
        /// corner checks use.
        /// </summary>
        private static bool IsFlowContent(Measured item, Rect panel)
        {
            if (item.Role != Role.ByPosition)
            {
                return false;
            }
            Rect bound = item.Bound;
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
        /// So the assertion is made against each handset's panel height, not against
        /// the panel being rendered: content must clear the furniture with the screen's
        /// surplus height subtracted. A game view already at or below handset height
        /// needs no correction and is checked as it stands.
        ///
        /// <paramref name="sizeRatio"/> resizes what was measured to what the handset
        /// would draw. UiTheme grows controls on narrow phones to keep 48 dp touch
        /// targets, keyed on the LIVE screen's width, so a layout measured on a 320 dp
        /// simulator carried 25% taller controls than a 360 dp phone ever draws, and
        /// the 360 x 640 check failed a menu that fits it. Both the flow content and
        /// the bottom-anchored block scale with it.
        /// </summary>
        private static void ExpectHandsetFit(string when, string handset, float handsetPanel, Rect panel,
            float contentBottom, float anchoredTop, float sizeRatio)
        {
            if (contentBottom <= 0f || float.IsPositiveInfinity(anchoredTop))
            {
                return;
            }
            float surplus = Mathf.Max(0f, panel.height - handsetPanel);
            float anchoredBlock = (panel.yMax - anchoredTop) * sizeRatio;
            float budget = panel.yMax - surplus - anchoredBlock;
            contentBottom = panel.y + (contentBottom - panel.y) * sizeRatio;
            if (contentBottom > budget)
            {
                Record("error",
                    $"ui-audit[{when}] {handset}: content reaches {contentBottom:0} dp but the bottom controls "
                    + $"start at {budget:0} dp (panel is {panel.height:0} dp, {surplus:0} dp taller than this handset)",
                    string.Empty);
                return;
            }
            Debug.Log($"[SmokeRace] ui-audit[{when}] {handset}: vertical budget OK — {budget - contentBottom:0} dp spare");
        }

        /// <summary>
        /// The zero-scroll rule. A ScrollView here is a safety net, never the reading
        /// mode, so its content must fit its viewport on every handset. Scroll views in
        /// this UI flex to fill the space left over, so on a handset shorter than the
        /// panel the viewport loses exactly the surplus height, and a taller one gains it.
        ///
        /// The exception is a list inside a sheet sized to its content and capped at a
        /// percentage of the screen (the results sheet's 78%). Its viewport is only as
        /// tall as its content, so "lose the surplus" drove it negative; its real limit
        /// is the cap, taken of the handset's height, less the sheet's own chrome.
        /// </summary>
        private static void ExpectNoScroll(string when, string handset, float handsetPanel, Rect panel,
            List<ScrollView> scrolls, float sizeRatio)
        {
            for (int scrollIndex = 0; scrollIndex < scrolls.Count; scrollIndex++)
            {
                ScrollView scroll = scrolls[scrollIndex];
                // Content and the chrome around the viewport both resize with the
                // handset's controls (see ExpectHandsetFit); the panel height does not.
                float content = scroll.contentContainer.layout.height * sizeRatio;
                float viewport;
                VisualElement cappedSheet = FindPercentCappedAncestor(scroll, out float capPercent);
                if (cappedSheet != null && cappedSheet.parent != null)
                {
                    float handsetParent = cappedSheet.parent.layout.height - (panel.height - handsetPanel);
                    float sheetChrome = cappedSheet.layout.height - scroll.contentViewport.layout.height;
                    viewport = handsetParent * capPercent / 100f - sheetChrome * sizeRatio;
                }
                else
                {
                    float chrome = panel.height - scroll.contentViewport.layout.height;
                    viewport = handsetPanel - chrome * sizeRatio;
                }
                if (content > viewport + EDGE_TOLERANCE_DP)
                {
                    Record("error",
                        $"ui-audit[{when}] {handset}: a list scrolls — {content:0} dp of content in a "
                        + $"{viewport:0} dp viewport ({content - viewport:0} dp hidden)",
                        string.Empty);
                }
            }
        }

        /// <summary>The nearest ancestor whose inline max-height is a percentage, or null.</summary>
        private static VisualElement FindPercentCappedAncestor(VisualElement element, out float capPercent)
        {
            for (VisualElement node = element.parent; node != null; node = node.parent)
            {
                StyleLength maxHeight = node.style.maxHeight;
                if (maxHeight.keyword == StyleKeyword.Undefined && maxHeight.value.unit == LengthUnit.Percent)
                {
                    capPercent = maxHeight.value.value;
                    return node;
                }
            }
            capPercent = 0f;
            return null;
        }

        /// <summary>The control scale the live screen resolved to (1 where the editor reports no DPI).</summary>
        private static float LiveControlScale()
        {
            if (Screen.width <= 0 || Screen.dpi <= 0f || float.IsNaN(Screen.dpi))
            {
                return 1f;
            }
            return UiTheme.ControlScaleForDeviceWidth(Screen.width / (Screen.dpi / 160f));
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

        /// <summary>
        /// Seconds this step may spend RACING before the harness gives up on it.
        ///
        /// It is the LARGER of the caller's per-step budget and the map's own clock,
        /// because a harness that stops a race before its map can resolve tests nothing.
        /// At the shipped default of 140 s that is exactly what happened to both
        /// courses on 2026-09-11: Acrobat (240 s) and Apartment (180 s) were cut off
        /// mid-race and reported `racersAliveAtResults: 0`, `raceEndedAtSeconds: 0`,
        /// zero produce and racers still in `Racing` — two of seven steps silently
        /// measuring only their first 140 s. A caller can therefore lengthen a step but
        /// never shorten it below the thing under test.
        ///
        /// The margin is for the referee: the race ends ON the map clock, and the
        /// podium/DNF classification lands a frame or two later.
        /// </summary>
        private static float RaceBudgetSeconds()
        {
            float mapLimit = _step != null ? _step.timeLimitSeconds : 0f;
            return Mathf.Max(_job.secondsPerStep, mapLimit + RACE_OVERRUN_MARGIN_SECONDS);
        }

        private static void StartRaceStep()
        {
            int mapIndex = _job.maps[_job.step];
            Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Get(mapIndex);
            BeginStep($"{mapIndex}:{map.DisplayName}");
            // Read BEFORE the race rather than in EndRaceStep, because the Racing
            // cutoff below needs it. Without it the harness cannot know that this
            // map wants longer than the caller's per-step budget.
            _step.timeLimitSeconds = map.TimeLimitSeconds;
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
                    (racer.Status == RacerStatus.Dnf ? $" ({racer.Knockout})" : string.Empty) +
                    (racer.Place > 0 ? $" P{racer.Place}" : string.Empty));
            }
            _step.placings = placings.ToArray();
            _step.timeLimitSeconds = Systems_MapCatalog.Get(_job.maps[_job.step]).TimeLimitSeconds;
            _step.endReason = ClassifyEnd(_step);
            CloseStep();
            _spawn.RequestMenu();
            _phase = Phase.Cooldown;
            _phaseStart = EditorApplication.timeSinceStartup;
        }

        /// <summary>
        /// Why this race stopped. The distinction that matters is the podium cutoff:
        /// <see cref="Systems_Race.NotifyFinish"/> ends the race the instant
        /// PODIUM_FINISHERS racers cross and marks everyone still going as Dnf, however
        /// well they were running. On Flat that fires around 21 s of a 120 s limit and
        /// takes out racers at 14-17 m of 20, upright and mid-stride -- which reads in
        /// the results as a roster that cannot finish, and feeds ELO a loss for each of
        /// them, when nothing about them failed.
        /// </summary>
        private static string ClassifyEnd(StepReport step)
        {
            if (!step.raceEnded)
            {
                return "harness cut the step short before the race resolved";
            }
            bool atFullTime = step.timeLimitSeconds > 0f
                && step.raceEndedAtSeconds >= step.timeLimitSeconds - 1f;
            if (atFullTime)
            {
                return $"full time ({step.raceEndedAtSeconds:0}s); ranked on distance";
            }
            if (step.finished >= Systems_Race.PODIUM_FINISHERS)
            {
                int cutOff = step.racers - step.finished;
                return $"PODIUM CUTOFF at {step.raceEndedAtSeconds:0}s of {step.timeLimitSeconds:0}s"
                     + $" - {cutOff} racer(s) DNF'd by rule, not by failing";
            }
            return $"every racer out by attrition at {step.raceEndedAtSeconds:0}s"
                 + $" of {step.timeLimitSeconds:0}s";
        }

        private static void BeginStep(string name)
        {
            _step = new StepReport { name = name, fpsMin = float.MaxValue, fpsMinRace = float.MaxValue };
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
            if (_step.fpsMinRace == float.MaxValue)
            {
                // No frame sampled while racing (a step cut off before START).
                _step.fpsMinRace = 0f;
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
                if (_phase == Phase.Racing && fps < _step.fpsMinRace)
                {
                    _step.fpsMinRace = fps;
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
