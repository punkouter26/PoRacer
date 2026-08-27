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
    /// SCENE (live object counts, sampled every 2 s), and RACE (state, leader,
    /// field). Text refresh runs on a 250 ms schedule, not per frame; Update
    /// only counts frames. Rich-text colors flag values that blow their budget.
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
        private const float GRAPH_HEIGHT = 42f;

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
        private Label _fpsLabel;
        private int _lastShownFps = -1;
        private Label _text;
        private VisualElement _graph;
        private readonly StringBuilder _builder = new();
        private readonly float[] _frameMs = new float[FRAME_SAMPLES];
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
        private int _refreshCounter;
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
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            root.pickingMode = PickingMode.Ignore;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            // Always-on FPS readout, top center — present in release builds too.
            var fpsRow = new VisualElement { pickingMode = PickingMode.Ignore };
            fpsRow.style.position = Position.Absolute;
            fpsRow.style.top = UiTheme.SPACE_XS;
            fpsRow.style.left = 0;
            fpsRow.style.right = 0;
            fpsRow.style.alignItems = Align.Center;
            _fpsLabel = new Label { pickingMode = PickingMode.Ignore };
            _fpsLabel.style.fontSize = UiTheme.FONT_SM;
            _fpsLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _fpsLabel.style.color = FpsGood;
            fpsRow.Add(_fpsLabel);
            safeRoot.Add(fpsRow);
            root.schedule.Execute(RefreshFps).Every(REFRESH_INTERVAL_MS);

            // The diagnostic panel below is editor/development-only; the component
            // itself stays enabled so Update keeps feeding the FPS counter.
            if (!Debug.isDebugBuild)
            {
                return;
            }

            var toggle = new Button(TogglePanel) { text = "DBG" };
            toggle.style.position = Position.Absolute;
            toggle.style.bottom = UiTheme.SPACE_SM;
            toggle.style.left = UiTheme.SPACE_SM;
            toggle.style.width = 44;
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

            _drawCalls = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
            _batches = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Batches Count");
            _setPasses = ProfilerRecorder.StartNew(ProfilerCategory.Render, "SetPass Calls Count");
            _triangles = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
            _gcPerFrame = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Allocated In Frame");
        }

        private void OnDestroy()
        {
            _drawCalls.Dispose();
            _batches.Dispose();
            _setPasses.Dispose();
            _triangles.Dispose();
            _gcPerFrame.Dispose();
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
            _frameCursor = (_frameCursor + 1) % FRAME_SAMPLES;
        }

        /// <summary>Top-center FPS number; text/color only touched when it changes.</summary>
        private void RefreshFps()
        {
            int fps = Mathf.RoundToInt(_fps);
            if (fps == _lastShownFps)
            {
                return;
            }
            _lastShownFps = fps;
            _fpsLabel.text = $"{fps} FPS";
            _fpsLabel.style.color = fps >= 55 ? FpsGood : fps >= 30 ? FpsWarn : FpsBad;
        }

        private void TogglePanel()
        {
            _visible = !_visible;
            _panel.style.display = _visible ? DisplayStyle.Flex : DisplayStyle.None;
        }

        private void Refresh()
        {
            if (!_visible)
            {
                return;
            }
            _refreshCounter++;
            if (_refreshCounter % SCENE_SAMPLE_EVERY_N_REFRESHES == 1)
            {
                SampleSceneCounts();
            }

            _builder.Clear();
            AppendPerf();
            AppendRender();
            AppendBiomechanics();
            AppendScene();
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
        /// Live object counts via a scene walk — too heavy for every refresh, so
        /// it runs on the slower SCENE_SAMPLE beat (debug overlay only).
        /// </summary>
        private void SampleSceneCounts()
        {
            _bodyCount = FindObjectsByType<ArticulationBody>(FindObjectsSortMode.None).Length;
            _particleSystemCount = FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None).Length;
            _audioSourceCount = FindObjectsByType<AudioSource>(FindObjectsSortMode.None).Length;
            _limbContactCount = FindObjectsByType<LimbContactView>(FindObjectsSortMode.None).Length;
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
            }

            Painter2D painter = context.painter2D;
            painter.lineWidth = 1f;
            painter.strokeColor = new Color(1f, 1f, 1f, 0.25f);
            float budgetY = height - FRAME_BUDGET_MS / worst * (height - 2f);
            painter.BeginPath();
            painter.MoveTo(new Vector2(0f, budgetY));
            painter.LineTo(new Vector2(width, budgetY));
            painter.Stroke();

            painter.lineWidth = 1.5f;
            painter.strokeColor = UiTheme.AccentSoft;
            painter.BeginPath();
            for (int sampleIndex = 0; sampleIndex < FRAME_SAMPLES; sampleIndex++)
            {
                // Oldest sample first: the cursor points at the next overwrite slot.
                float ms = _frameMs[(_frameCursor + sampleIndex) % FRAME_SAMPLES];
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
