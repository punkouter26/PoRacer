using PoRacer.Presentation;
using System.Text;
using PoRacer.Models;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Toggleable diagnostic overlay (DBG button, bottom-left), grouped into
    /// colored sections: PERF (fps, memory, GC), RENDER (draws, batches, tris),
    /// PHYSICS (fixed-loop cost, step count, body and contact counts), SCENE
    /// (live object counts, sampled every 2 s), and RACE (state, leader, field).
    /// Text refresh runs on a 250 ms schedule, not per frame; Update only counts
    /// frames. Rich-text colors flag values that blow their budget.
    ///
    /// A compact always-on strip (fps / frame ms / draws) sits top-center in every
    /// build. The panel below it, the frame graph and the CSV recorder are
    /// development-build only.
    ///
    /// The frame graph plots two series: total frame time against the 60 FPS
    /// budget, and the fixed loop's own cost against the 20 ms step budget. For a
    /// field of articulation bodies the second line is usually the one that
    /// explains the first.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class DebugOverlayView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const int FRAME_SAMPLES = 120;
        private const float FRAME_BUDGET_MS = 1000f / 60f;
        // Object counting walks the scene; do it on a slower beat than the text.
        private const int SCENE_SAMPLE_EVERY_N_REFRESHES = 8;
        private const float PANEL_WIDTH = 310f;
        private const float GRAPH_HEIGHT = 48f;
        // The fixed loop gets one Time.fixedDeltaTime of wall clock before it
        // starts stealing from the frame; 0.02 s is locked by the project rules.
        private const float FIXED_BUDGET_MS = 20f;
        // Rows are buffered and flushed together so a recording session does not
        // put a file write in the middle of every refresh.

        private const string HEADER_COLOR = "#E8C55A";
        private const string DIM_COLOR = "#9A9A9A";
        private const string GOOD_COLOR = "#7CE87C";
        private const string WARN_COLOR = "#E8D45A";
        private const string BAD_COLOR = "#E86A5A";

        private static readonly Color FpsGood = new(0.49f, 0.91f, 0.49f);
        private static readonly Color FpsWarn = new(0.91f, 0.83f, 0.35f);
        private static readonly Color FpsBad = new(0.91f, 0.42f, 0.35f);

        private RaceModel _raceModel;
        private VisualElement _panel;
        private bool _diagnosticsActive;
        private Label _fpsLabel;
        private int _lastShownFps = -1;
        // Second half of the always-on strip: frame time and draw count, which is
        // enough to tell a CPU stall from a draw-call flood without opening the panel.
        private Label _stripLabel;
        private string _lastStripText = string.Empty;
        private Label _text;
        private VisualElement _graph;
        private readonly StringBuilder _builder = new();
        private readonly float[] _frameMs = new float[FRAME_SAMPLES];
        // Fixed-loop cost per frame, same cursor and length as _frameMs so the two
        // series line up sample-for-sample on the graph.
        private readonly float[] _fixedMs = new float[FRAME_SAMPLES];
        private int _frameCursor;
        private int _frameCount;
        private float _frameSeconds;
        private float _fps;
        private bool _visible;
        private ProfilerRecorder _drawCalls;
        private ProfilerRecorder _batches;
        private ProfilerRecorder _setPasses;
        private ProfilerRecorder _triangles;
        private ProfilerRecorder _gcPerFrame;
        // Physics counter names differ between Unity versions and backends; each
        // recorder is used only when it reports Valid, so a missing one degrades
        // to a dash instead of throwing.
        private ProfilerRecorder _activeBodies;
        private ProfilerRecorder _physicsQueries;
        private ProfilerRecorder _constraints;
        private int _refreshCounter;
        // Brain roster, rebuilt on the slow scene beat. One entry per distinct
        // controller, not per racer, so a 100-strong field is still a few lines.
        /// <summary>Field separator for the grouping key; never appears in a name.</summary>
        private const char KEY_SEP = '|';

        private readonly System.Collections.Generic.List<string> _brainLines = new();
        private int _brainsWithModel;
        private int _brainsWithoutModel;

        private int _bodyCount;
        private int _particleSystemCount;
        private int _audioSourceCount;
        private int _limbContactCount;
        private string _lastLeaderId;
        private float _lastLeaderProgress;
        private float _lastLeaderSampleTime;
        private float _leaderSpeed;

        [Inject]
        public void Construct(RaceModel raceModel)
        {
            _raceModel = raceModel;
        }

        private void Start()
        {
            // This used to build nothing at all in release, on the argument that
            // "nothing diagnostic belongs in a player's build". The objection was
            // really about cost - probes, ProfilerRecorders and a 250 ms schedule
            // running forever for a readout nobody asked for. So pay for them on
            // demand instead of refusing to ship them: the DBG button and the fps
            // readout cost nothing beyond a frame counter, and the recorders and the
            // physics probe pair only start the first time someone opens the panel
            // (see ActivateDiagnostics). A release player now gets the button, and
            // an unopened panel still costs what it did before: nothing.

            var document = GetComponent<UIDocument>();
            // RaceHud and DebugOverlay both shipped at sortingOrder 0, which is a tie:
            // the draw order between them was arbitrary, and on device RaceHud won.
            // That put the top-centre FPS readout BEHIND the top-3 chips and buried
            // the DBG button under the commentary ticker. A diagnostic layer belongs
            // above the HUD it is diagnosing, so claim a band of its own. (Menu is 10.)
            document.sortingOrder = 20;

            VisualElement root = document.rootVisualElement;
            root.pickingMode = PickingMode.Ignore;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            // Telemetry strip, top center.
            var stripRow = new VisualElement { pickingMode = PickingMode.Ignore };
            stripRow.style.position = Position.Absolute;
            stripRow.style.top = UiTheme.SPACE_XS;
            stripRow.style.left = 0;
            stripRow.style.right = 0;
            stripRow.style.flexDirection = FlexDirection.Row;
            stripRow.style.justifyContent = Justify.Center;
            stripRow.style.alignItems = Align.Center;
            _fpsLabel = new Label { pickingMode = PickingMode.Ignore };
            _fpsLabel.style.fontSize = UiTheme.FONT_SM;
            _fpsLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _fpsLabel.style.color = FpsGood;
            // The fps readout shares the furniture band with the title and has no
            // plate behind it either; on a bright sky it needs the same shadow.
            UiTheme.AddTextShadow(_fpsLabel);
            stripRow.Add(_fpsLabel);
            // Frame time and draw count ride alongside in one pre-formatted
            // string, so a refresh rewrites at most two labels.
            _stripLabel = new Label { pickingMode = PickingMode.Ignore };
            _stripLabel.style.fontSize = UiTheme.FONT_XS;
            _stripLabel.style.color = UiTheme.TextDim;
            _stripLabel.style.marginLeft = UiTheme.SPACE_SM;
            // Frame time, fixed-loop cost and draw counts need the recorders and the
            // probe pair, so this half stays dark until the panel is first opened.
            // The fps label beside it does not - it reads a plain frame counter.
            _stripLabel.style.display = DisplayStyle.None;
            stripRow.Add(_stripLabel);
            safeRoot.Add(stripRow);
            root.schedule.Execute(RefreshStrip).Every(REFRESH_INTERVAL_MS);

            var toggle = new Button(TogglePanel) { text = "DBG" };
            toggle.style.position = Position.Absolute;
            toggle.style.bottom = UiTheme.SPACE_SM;
            toggle.style.left = UiTheme.SPACE_SM;
            // CONTROL_SM is Android's 48 dp minimum touch target: the panel now
            // references a 420 dp-wide screen, so one UI unit is ~1 dp on a phone
            // and the token is the dp figure directly. This used to hard-code 62 to
            // work around the old 540 dp reference, which made every other button
            // in the app 30 dp.
            toggle.style.width = UiTheme.CONTROL_SM;
            toggle.style.height = UiTheme.CONTROL_SM;
            toggle.style.fontSize = UiTheme.FONT_XS;
            toggle.style.opacity = 0.75f;
            UiTheme.StyleButton(toggle);
            UiTheme.AddHover(toggle);
            safeRoot.Add(toggle);


            _panel = new VisualElement { pickingMode = PickingMode.Ignore };
            _panel.style.position = Position.Absolute;
            _panel.style.bottom = UiTheme.SPACE_SM + UiTheme.CONTROL_SM + UiTheme.SPACE_SM;
            _panel.style.left = UiTheme.SPACE_SM;
            _panel.style.width = PANEL_WIDTH;
            _panel.style.display = DisplayStyle.None;
            UiTheme.StylePanel(_panel);
            safeRoot.Add(_panel);

            _text = new Label { pickingMode = PickingMode.Ignore };
            _text.style.color = UiTheme.Text;
            _text.style.fontSize = UiTheme.FONT_SM;
            _text.style.whiteSpace = WhiteSpace.Pre;
            _panel.Add(_text);

            // Frame-time history strip: one polyline, redrawn on the refresh
            // schedule; the dim line is the 60 FPS budget.
            _graph = new VisualElement { pickingMode = PickingMode.Ignore };
            _graph.style.height = GRAPH_HEIGHT;
            _graph.style.marginTop = UiTheme.SPACE_SM;
            _graph.style.backgroundColor = UiTheme.TrackBg;
            UiTheme.SetRadius(_graph, UiTheme.RADIUS_SM);
            _graph.generateVisualContent += DrawFrameGraph;
            _panel.Add(_graph);

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
        }

        /// <summary>
        /// Starts the machinery the panel needs, once, the first time it is opened.
        /// Everything in here has a standing per-frame cost, which is why none of it
        /// runs for a player who never presses DBG.
        /// </summary>
        private void ActivateDiagnostics()
        {
            if (_diagnosticsActive)
            {
                return;
            }
            _diagnosticsActive = true;

            // The probe pair owns the fixed-loop timing the strip and panel read.
            PhysicsProbeView.EnsureOn(gameObject);
            if (_stripLabel != null)
            {
                _stripLabel.style.display = DisplayStyle.Flex;
            }

            _drawCalls = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
            _batches = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Batches Count");
            _setPasses = ProfilerRecorder.StartNew(ProfilerCategory.Render, "SetPass Calls Count");
            _triangles = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
            _gcPerFrame = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Allocated In Frame");
            _activeBodies = ProfilerRecorder.StartNew(ProfilerCategory.Physics, "Active Dynamic Bodies");
            _physicsQueries = ProfilerRecorder.StartNew(ProfilerCategory.Physics, "Physics Queries");
            _constraints = ProfilerRecorder.StartNew(ProfilerCategory.Physics, "Active Constraints");
        }

        private void OnDestroy()
        {
            if (!_diagnosticsActive)
            {
                return;
            }
            _drawCalls.Dispose();
            _batches.Dispose();
            _setPasses.Dispose();
            _triangles.Dispose();
            _gcPerFrame.Dispose();
            _activeBodies.Dispose();
            _physicsQueries.Dispose();
            _constraints.Dispose();
        }

        private void Update()
        {
            _frameCount++;
            _frameSeconds += Time.unscaledDeltaTime;
            if (_frameSeconds >= 0.5f)
            {
                _fps = _frameCount / _frameSeconds;
                _frameCount = 0;
                _frameSeconds = 0f;
            }
            _frameMs[_frameCursor] = Time.unscaledDeltaTime * 1000f;
            // Scripts plus PhysX where PhysX was attributable; the probe returns
            // -1 for single-step frames, where the two cannot be separated.
            float physxMs = PhysicsProbeView.PhysxMsPerFrame;
            _fixedMs[_frameCursor] = PhysicsProbeView.ScriptMsPerFrame + Mathf.Max(0f, physxMs);
            _frameCursor = (_frameCursor + 1) % FRAME_SAMPLES;
        }

        /// <summary>
        /// Top-center telemetry strip. Both labels are compared against their last
        /// value before being written, so a steady frame rate costs no text
        /// regeneration and no layout pass.
        /// </summary>
        private void RefreshStrip()
        {
            int fps = Mathf.RoundToInt(_fps);
            if (fps != _lastShownFps)
            {
                _lastShownFps = fps;
                _fpsLabel.text = $"{fps} FPS";
                _fpsLabel.style.color = fps >= 55 ? FpsGood : fps >= 30 ? FpsWarn : FpsBad;
            }
            if (_stripLabel == null)
            {
                return;
            }
            float frameMs = 1000f / Mathf.Max(_fps, 0.01f);
            long draws = _drawCalls.Valid ? _drawCalls.LastValue : 0L;
            float fixedMs = PhysicsProbeView.ScriptMsPerFrame
                + Mathf.Max(0f, PhysicsProbeView.PhysxMsPerFrame);
            string stripText = draws > 0L
                ? $"{frameMs:0.0} ms  |  fix {fixedMs:0.0}  |  {draws} draws"
                : $"{frameMs:0.0} ms  |  fix {fixedMs:0.0}";
            if (stripText == _lastStripText)
            {
                return;
            }
            _lastStripText = stripText;
            _stripLabel.text = stripText;
        }

        private void TogglePanel()
        {
            ActivateDiagnostics();
            _visible = !_visible;
            _panel.style.display = _visible ? DisplayStyle.Flex : DisplayStyle.None;
        }

        private void Refresh()
        {
            // With the panel closed nothing here is
            // worth doing when it is closed and idle - the scene walk below is a
            // FindObjectsByType over a field that can be a hundred racers deep.
            if (!_visible)
            {
                return;
            }

            _refreshCounter++;
            if (_refreshCounter % SCENE_SAMPLE_EVERY_N_REFRESHES == 1)
            {
                SampleSceneCounts();
            }

            if (!_visible)
            {
                return;
            }

            _builder.Clear();
            AppendPerf();
            AppendRender();
            AppendPhysics();
            AppendAudio();
            AppendBiomechanics();
            AppendScene();
            AppendBrains();
            AppendRace();
            _text.text = _builder.ToString();
            _graph.MarkDirtyRepaint();
        }

        private void AppendHeader(string title)
        {
            if (_builder.Length > 0)
            {
                _builder.Append('\n');
            }
            _builder.Append("<color=").Append(HEADER_COLOR).Append("><b>")
                .Append(title).Append("</b></color>\n");
        }

        private void AppendPerf()
        {
            float frameMs = 1000f / Mathf.Max(_fps, 0.01f);
            float worstMs = 0f;
            for (int sampleIndex = 0; sampleIndex < FRAME_SAMPLES; sampleIndex++)
            {
                if (_frameMs[sampleIndex] > worstMs)
                {
                    worstMs = _frameMs[sampleIndex];
                }
            }
            string fpsColor = _fps >= 55f ? GOOD_COLOR : _fps >= 30f ? WARN_COLOR : BAD_COLOR;
            string worstColor = worstMs <= FRAME_BUDGET_MS * 1.5f ? GOOD_COLOR
                : worstMs <= FRAME_BUDGET_MS * 3f ? WARN_COLOR : BAD_COLOR;
            float gcKb = _gcPerFrame.Valid ? _gcPerFrame.LastValue / 1024f : 0f;
            string gcColor = gcKb < 1f ? GOOD_COLOR : gcKb < 16f ? WARN_COLOR : BAD_COLOR;

            AppendHeader("PERF");
            _builder.Append("FPS     <color=").Append(fpsColor).Append('>')
                .Append(_fps.ToString("0")).Append("</color> (")
                .Append(frameMs.ToString("0.0")).Append(" ms)  worst <color=").Append(worstColor).Append('>')
                .Append(worstMs.ToString("0")).Append(" ms</color>\n");
            _builder.Append("GC      <color=").Append(gcColor).Append('>')
                .Append(gcKb.ToString("0.0")).Append(" KB/frame</color>  runs ")
                .Append(System.GC.CollectionCount(0)).Append('\n');
            _builder.Append("Memory  ")
                .Append((UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() / (1024f * 1024f)).ToString("0"))
                .Append(" MB total  ")
                .Append((UnityEngine.Profiling.Profiler.GetMonoUsedSizeLong() / (1024f * 1024f)).ToString("0"))
                .Append(" MB mono\n");
        }

        private void AppendRender()
        {
            if (!_drawCalls.Valid)
            {
                return;
            }
            AppendHeader("RENDER");
            _builder.Append("Draws   ").Append(_drawCalls.LastValue)
                .Append("  batches ").Append(_batches.Valid ? _batches.LastValue : 0)
                .Append("  setpass ").Append(_setPasses.Valid ? _setPasses.LastValue : 0)
                .Append('\n');
            if (_triangles.Valid)
            {
                _builder.Append("Tris    ").Append((_triangles.LastValue / 1000f).ToString("0")).Append("k  ");
            }
            _builder.Append("screen ").Append(Screen.width).Append('x').Append(Screen.height)
                .Append(" @").Append(Application.targetFrameRate).Append('\n');
        }

        /// <summary>
        /// The fixed loop, which for a field of articulation bodies is where the
        /// frame actually goes. Script time is always measurable; PhysX time is
        /// only separable on frames that ran more than one step, and prints as a
        /// dash otherwise rather than being folded in silently.
        /// </summary>
        private void AppendPhysics()
        {
            float scriptMs = PhysicsProbeView.ScriptMsPerFrame;
            float physxMs = PhysicsProbeView.PhysxMsPerFrame;
            float totalMs = scriptMs + Mathf.Max(0f, physxMs);
            string totalColor = totalMs <= FIXED_BUDGET_MS * 0.5f ? GOOD_COLOR
                : totalMs <= FIXED_BUDGET_MS ? WARN_COLOR : BAD_COLOR;

            AppendHeader("PHYSICS");
            _builder.Append("Fixed   <color=").Append(totalColor).Append('>')
                .Append(totalMs.ToString("0.00")).Append(" ms</color>/frame  budget ")
                .Append(FIXED_BUDGET_MS.ToString("0")).Append(" ms\n");
            _builder.Append("        scripts ").Append(scriptMs.ToString("0.00")).Append(" ms  physx ");
            if (physxMs >= 0f)
            {
                _builder.Append(physxMs.ToString("0.00")).Append(" ms\n");
            }
            else
            {
                // Single-step frames leave rendering inside the same gap, so the
                // honest answer is that it was not measured.
                _builder.Append("<color=").Append(DIM_COLOR).Append(">- (1 step)</color>\n");
            }
            _builder.Append("Steps   ").Append(PhysicsProbeView.StepsPerFrame).Append("/frame");
            if (_activeBodies.Valid)
            {
                _builder.Append("  active bodies ").Append(_activeBodies.LastValue);
            }
            _builder.Append('\n');
            if (_constraints.Valid || _physicsQueries.Valid)
            {
                _builder.Append("        ");
                if (_constraints.Valid)
                {
                    _builder.Append("constraints ").Append(_constraints.LastValue).Append("  ");
                }
                if (_physicsQueries.Valid)
                {
                    _builder.Append("queries ").Append(_physicsQueries.LastValue);
                }
                _builder.Append('\n');
            }
        }

        private void AppendBiomechanics()
        {
            AppendHeader("BIOMECHANICS & SOLVER");
            _builder.Append("Solver  PhysX Iterations: ").Append(Physics.defaultSolverIterations)
                .Append("  Fixed Δt: ").Append(Time.fixedDeltaTime.ToString("0.000")).Append("s\n");
            _builder.Append("Gravity [").Append(Physics.gravity.x.ToString("0.0")).Append(", ")
                .Append(Physics.gravity.y.ToString("0.0")).Append(", ")
                .Append(Physics.gravity.z.ToString("0.0")).Append("] m/s² (1G)\n");
            _builder.Append("Joints  Articulations: ").Append(_bodyCount)
                .Append("  Ground Contacts: ").Append(_limbContactCount).Append('\n');
        }

        /// <summary>
        /// Mix health. Gain reduction is the limiter telling you the synthesized
        /// layers are summing past the ceiling; a number pinned well below zero
        /// means the design volumes need trimming, not that the limiter is broken.
        /// </summary>
        private void AppendAudio()
        {
            float reductionDb = MasterLimiterView.GainReductionDb;
            string reductionColor = reductionDb > -1f ? DIM_COLOR
                : reductionDb > -6f ? GOOD_COLOR
                : reductionDb > -12f ? WARN_COLOR : BAD_COLOR;

            AppendHeader("AUDIO");
            _builder.Append("Sources ").Append(_audioSourceCount)
                .Append("  real-voice cap ").Append(AudioSettings.GetConfiguration().numRealVoices)
                .Append('\n');
            _builder.Append("Limiter <color=").Append(reductionColor).Append('>')
                .Append(reductionDb.ToString("0.0")).Append(" dB</color> reduction\n");
        }

        private void AppendScene()
        {
            AppendHeader("SCENE");
            _builder.Append("Bodies  ").Append(_bodyCount)
                .Append("  particles ").Append(_particleSystemCount)
                .Append("  audio ").Append(_audioSourceCount).Append('\n');
            string scaleColor = Mathf.Approximately(Time.timeScale, 1f) ? DIM_COLOR : WARN_COLOR;
            _builder.Append("Time    dt ").Append(Time.fixedDeltaTime.ToString("0.000"))
                .Append("  <color=").Append(scaleColor).Append(">scale ")
                .Append(Time.timeScale.ToString("0.00")).Append("</color>\n");
        }

        private void AppendRace()
        {
            if (_raceModel == null)
            {
                return;
            }
            AppendHeader("RACE");
            _builder.Append('#').Append(_raceModel.RaceNumber)
                .Append("  ").Append(_raceModel.TrackName)
                .Append(_raceModel.RaceActive
                    ? "  <color=" + GOOD_COLOR + ">RUNNING</color>  "
                    : "  <color=" + DIM_COLOR + ">IDLE</color>  ")
                .Append(_raceModel.ElapsedSeconds.ToString("0.0")).Append("s\n");

            int racing = 0;
            int finished = 0;
            int dnf = 0;
            RacerState leader = null;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                switch (racer.Status)
                {
                    case RacerStatus.Finished:
                    case RacerStatus.TimedOut:
                        finished++;
                        break;
                    case RacerStatus.Dnf:
                        dnf++;
                        break;
                    default:
                        racing++;
                        if (leader == null || racer.Progress > leader.Progress)
                        {
                            leader = racer;
                        }
                        break;
                }
            }
            _builder.Append("Field   <color=").Append(GOOD_COLOR).Append('>').Append(racing)
                .Append(" racing</color>  ").Append(finished).Append(" fin  <color=")
                .Append(dnf > 0 ? WARN_COLOR : DIM_COLOR).Append('>').Append(dnf).Append(" dnf</color>\n");

            UpdateLeaderSpeed(leader);
            if (leader != null)
            {
                float trackLength = Mathf.Max(1f, _raceModel.TrackLengthMeters);
                _builder.Append("Leader  ").Append(leader.DisplayName).Append('\n');
                _builder.Append("        ").Append(leader.Progress.ToString("0.0")).Append("m (")
                    .Append((Mathf.Clamp01(leader.Progress / trackLength) * 100f).ToString("0")).Append("%)  ")
                    .Append(_leaderSpeed.ToString("0.0")).Append(" m/s");
            }
        }

        /// <summary>
        /// What is actually driving the racers. The single question this answers
        /// is "did the brain load?" - a policy that silently failed to bind looks
        /// exactly like one that is thinking, and the only other symptom is a
        /// creature that stands still. Reads off the slow scene beat.
        /// </summary>
        private void AppendBrains()
        {
            if (_brainLines.Count == 0 && _brainsWithModel == 0 && _brainsWithoutModel == 0)
            {
                return;
            }
            AppendHeader("BRAINS");
            _builder.Append("Loaded  <color=").Append(_brainsWithModel > 0 ? GOOD_COLOR : DIM_COLOR)
                .Append('>').Append(_brainsWithModel).Append(" with a policy</color>");
            if (_brainsWithoutModel > 0)
            {
                _builder.Append("  <color=").Append(BAD_COLOR).Append('>')
                    .Append(_brainsWithoutModel).Append(" WITHOUT</color>");
            }
            _builder.AppendLine();
            for (int lineIndex = 0; lineIndex < _brainLines.Count; lineIndex++)
            {
                _builder.AppendLine(_brainLines[lineIndex]);
            }
        }

        /// <summary>
        /// Groups the live agents by controller and records one line each:
        /// creature, where the policy comes from, its observation and action
        /// widths, and how many are running.
        /// </summary>
        private void SampleBrains()
        {
            _brainLines.Clear();
            _brainsWithModel = 0;
            _brainsWithoutModel = 0;

            var counts = new System.Collections.Generic.Dictionary<string, int>();
            var order = new System.Collections.Generic.List<string>();

            var behaviours = FindObjectsByType<Unity.MLAgents.Policies.BehaviorParameters>(
                FindObjectsInactive.Exclude);
            for (int index = 0; index < behaviours.Length; index++)
            {
                var behaviour = behaviours[index];
                bool hasModel = behaviour.Model != null;
                string source = hasModel
                    ? "ONNX/" + behaviour.InferenceDevice
                    : "<color=" + BAD_COLOR + ">no model</color>";
                if (behaviour.BehaviorType == Unity.MLAgents.Policies.BehaviorType.HeuristicOnly)
                {
                    source = "coded gait";
                }
                var brain = behaviour.BrainParameters;
                string key = behaviour.BehaviorName + KEY_SEP + source + KEY_SEP
                    + brain.VectorObservationSize + KEY_SEP + brain.ActionSpec.NumContinuousActions;
                Tally(counts, order, key);
                if (hasModel) { _brainsWithModel++; } else { _brainsWithoutModel++; }
            }

            // The Isaac and MuJoCo racers are not ML-Agents agents: they run their
            // own inference and carry no BehaviorParameters, so they have to be
            // counted separately or they read as missing.
            var natives = FindObjectsByType<PoRacer.Agents.NativeBrainProbe>(FindObjectsInactive.Exclude);
            for (int index = 0; index < natives.Length; index++)
            {
                var probe = natives[index];
                string key = probe.CreatureName + KEY_SEP + probe.PolicySource + KEY_SEP
                    + probe.ObservationCount + KEY_SEP + probe.ActionCount;
                Tally(counts, order, key);
                if (probe.HasPolicy) { _brainsWithModel++; } else { _brainsWithoutModel++; }
            }

            for (int index = 0; index < order.Count; index++)
            {
                string[] parts = order[index].Split(KEY_SEP);
                _brainLines.Add(string.Format("{0,-9} {1}  obs {2} act {3}  x{4}",
                    Trim(parts[0], 9), parts[1], parts[2], parts[3], counts[order[index]]));
            }
        }

        private static void Tally(System.Collections.Generic.Dictionary<string, int> counts,
            System.Collections.Generic.List<string> order, string key)
        {
            if (counts.TryGetValue(key, out int existing))
            {
                counts[key] = existing + 1;
                return;
            }
            counts[key] = 1;
            order.Add(key);
        }

        private static string Trim(string value, int width)
        {
            if (string.IsNullOrEmpty(value)) { return string.Empty; }
            return value.Length <= width ? value : value.Substring(0, width);
        }

        /// <summary>
        /// Live object counts via a scene walk — too heavy for every refresh, so
        /// it runs on the slower SCENE_SAMPLE beat (debug overlay only).
        /// </summary>
        private void SampleSceneCounts()
        {
            _bodyCount = FindObjectsByType<ArticulationBody>().Length;
            _particleSystemCount = FindObjectsByType<ParticleSystem>().Length;
            _audioSourceCount = FindObjectsByType<AudioSource>().Length;
            _limbContactCount = FindObjectsByType<LimbContactView>().Length;
            SampleBrains();
        }

        private void UpdateLeaderSpeed(RacerState leader)
        {
            if (leader == null)
            {
                _lastLeaderId = null;
                _leaderSpeed = 0f;
                return;
            }
            float now = Time.unscaledTime;
            if (leader.RacerId == _lastLeaderId && now > _lastLeaderSampleTime)
            {
                float instant = (leader.Progress - _lastLeaderProgress) / (now - _lastLeaderSampleTime);
                _leaderSpeed = Mathf.Lerp(_leaderSpeed, Mathf.Max(0f, instant), 0.5f);
            }
            else
            {
                _leaderSpeed = 0f;
            }
            _lastLeaderId = leader.RacerId;
            _lastLeaderProgress = leader.Progress;
            _lastLeaderSampleTime = now;
        }

        /// <summary>
        /// Two series on one vertical scale: total frame time in accent, the fixed
        /// loop's own cost in cyan. Sharing the scale is the point — it shows at a
        /// glance how much of a missed frame was physics. Two dim rules mark the
        /// 60 FPS frame budget and the 20 ms fixed-step budget.
        /// </summary>
        private void DrawFrameGraph(MeshGenerationContext context)
        {
            float width = _graph.resolvedStyle.width;
            float height = _graph.resolvedStyle.height;
            if (width <= 4f || float.IsNaN(width))
            {
                return;
            }
            float worst = FRAME_BUDGET_MS * 2f;
            for (int sampleIndex = 0; sampleIndex < FRAME_SAMPLES; sampleIndex++)
            {
                if (_frameMs[sampleIndex] > worst)
                {
                    worst = _frameMs[sampleIndex];
                }
                if (_fixedMs[sampleIndex] > worst)
                {
                    worst = _fixedMs[sampleIndex];
                }
            }

            Painter2D painter = context.painter2D;
            DrawBudgetRule(painter, width, height, worst, FRAME_BUDGET_MS, 0.25f);
            DrawBudgetRule(painter, width, height, worst, FIXED_BUDGET_MS, 0.14f);

            DrawSeries(painter, _fixedMs, width, height, worst, UiTheme.NeonCyan, 1.2f);
            DrawSeries(painter, _frameMs, width, height, worst, UiTheme.AccentSoft, 1.5f);
        }

        private static void DrawBudgetRule(Painter2D painter, float width, float height,
            float worst, float budgetMs, float alpha)
        {
            painter.lineWidth = 1f;
            painter.strokeColor = new Color(1f, 1f, 1f, alpha);
            float budgetY = height - budgetMs / worst * (height - 2f);
            painter.BeginPath();
            painter.MoveTo(new Vector2(0f, budgetY));
            painter.LineTo(new Vector2(width, budgetY));
            painter.Stroke();
        }

        private void DrawSeries(Painter2D painter, float[] samples, float width, float height,
            float worst, Color color, float lineWidth)
        {
            painter.lineWidth = lineWidth;
            painter.strokeColor = color;
            painter.BeginPath();
            for (int sampleIndex = 0; sampleIndex < FRAME_SAMPLES; sampleIndex++)
            {
                // Oldest sample first: the cursor points at the next overwrite slot.
                float ms = samples[(_frameCursor + sampleIndex) % FRAME_SAMPLES];
                float x = width * sampleIndex / (FRAME_SAMPLES - 1);
                float y = height - Mathf.Clamp01(ms / worst) * (height - 2f);
                if (sampleIndex == 0)
                {
                    painter.MoveTo(new Vector2(x, y));
                }
                else
                {
                    painter.LineTo(new Vector2(x, y));
                }
            }
            painter.Stroke();
        }
    }
}
