using System.Text;
using PoRacer.Models;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Toggleable diagnostic overlay (DBG button, bottom-left): FPS, frame time,
    /// memory, screen mode, physics timing, and live race telemetry. Text refresh
    /// runs on a 250 ms schedule, not per frame; Update only counts frames.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class DebugOverlayView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const int FRAME_SAMPLES = 120;
        private const float FRAME_BUDGET_MS = 1000f / 60f;

        private RaceModel _raceModel;
        private VisualElement _panel;
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
        private ProfilerRecorder _gcPerFrame;

        [Inject]
        public void Construct(RaceModel raceModel)
        {
            _raceModel = raceModel;
        }

        private void Start()
        {
            // Diagnostic surface: editor and development builds only, never release.
            if (!Debug.isDebugBuild)
            {
                enabled = false;
                return;
            }
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            root.pickingMode = PickingMode.Ignore;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            var toggle = new Button(TogglePanel) { text = "DBG" };
            toggle.style.position = Position.Absolute;
            toggle.style.bottom = 6;
            toggle.style.left = 6;
            toggle.style.width = 44;
            toggle.style.height = 24;
            toggle.style.fontSize = 11;
            toggle.style.opacity = 0.75f;
            UiTheme.StyleButton(toggle);
            UiTheme.AddHover(toggle);
            safeRoot.Add(toggle);

            _panel = new VisualElement { pickingMode = PickingMode.Ignore };
            _panel.style.position = Position.Absolute;
            _panel.style.bottom = 36;
            _panel.style.left = 6;
            _panel.style.width = 300;
            _panel.style.display = DisplayStyle.None;
            UiTheme.StylePanel(_panel);
            safeRoot.Add(_panel);

            _text = new Label { pickingMode = PickingMode.Ignore };
            _text.style.color = UiTheme.Text;
            _text.style.fontSize = 11;
            _text.style.whiteSpace = WhiteSpace.Pre;
            _panel.Add(_text);

            // Frame-time history strip: one polyline, redrawn on the refresh
            // schedule; the dim line is the 60 FPS budget.
            _graph = new VisualElement { pickingMode = PickingMode.Ignore };
            _graph.style.height = 42;
            _graph.style.marginTop = 6;
            _graph.style.backgroundColor = new Color(0f, 0f, 0f, 0.35f);
            UiTheme.SetRadius(_graph, 4f);
            _graph.generateVisualContent += DrawFrameGraph;
            _panel.Add(_graph);

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);

            _drawCalls = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
            _batches = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Batches Count");
            _setPasses = ProfilerRecorder.StartNew(ProfilerCategory.Render, "SetPass Calls Count");
            _gcPerFrame = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Allocated In Frame");
        }

        private void OnDestroy()
        {
            _drawCalls.Dispose();
            _batches.Dispose();
            _setPasses.Dispose();
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
            _builder.Clear();
            float frameMs = 1000f / Mathf.Max(_fps, 0.01f);
            float worstMs = 0f;
            for (int sampleIndex = 0; sampleIndex < FRAME_SAMPLES; sampleIndex++)
            {
                if (_frameMs[sampleIndex] > worstMs)
                {
                    worstMs = _frameMs[sampleIndex];
                }
            }
            _builder.Append("FPS      ").Append(_fps.ToString("0.0"))
                .Append("  (").Append(frameMs.ToString("0.0")).Append(" ms)  worst ")
                .Append(worstMs.ToString("0")).Append(" ms\n");
            _builder.Append("Alloc    ")
                .Append((UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() / (1024f * 1024f)).ToString("0"))
                .Append(" MB\n");
            _builder.Append("Mono     ")
                .Append((UnityEngine.Profiling.Profiler.GetMonoUsedSizeLong() / (1024f * 1024f)).ToString("0"))
                .Append(" MB\n");
            _builder.Append("Screen   ").Append(Screen.width).Append('x').Append(Screen.height)
                .Append("  target ").Append(Application.targetFrameRate).Append(" fps\n");
            _builder.Append("Physics  dt ").Append(Time.fixedDeltaTime.ToString("0.000"))
                .Append("  scale ").Append(Time.timeScale.ToString("0.00")).Append('\n');
            if (_drawCalls.Valid)
            {
                _builder.Append("Render   ").Append(_drawCalls.LastValue).Append(" draws  ")
                    .Append(_batches.Valid ? _batches.LastValue : 0).Append(" batches  ")
                    .Append(_setPasses.Valid ? _setPasses.LastValue : 0).Append(" setpass\n");
            }
            if (_gcPerFrame.Valid)
            {
                _builder.Append("GC/frame ").Append((_gcPerFrame.LastValue / 1024f).ToString("0.0")).Append(" KB\n");
            }

            if (_raceModel != null)
            {
                int racing = 0;
                int finished = 0;
                int dnf = 0;
                for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
                {
                    switch (_raceModel.Racers[racerIndex].Status)
                    {
                        case RacerStatus.Finished:
                            finished++;
                            break;
                        case RacerStatus.Dnf:
                            dnf++;
                            break;
                        default:
                            racing++;
                            break;
                    }
                }
                _builder.Append("Race     #").Append(_raceModel.RaceNumber).Append("  ").Append(_raceModel.TrackName)
                    .Append(_raceModel.RaceActive ? "  RUNNING  " : "  IDLE  ")
                    .Append(_raceModel.ElapsedSeconds.ToString("0.0")).Append("s\n");
                _builder.Append("Racers   ").Append(racing).Append(" racing / ")
                    .Append(finished).Append(" fin / ").Append(dnf).Append(" dnf");
            }
            _text.text = _builder.ToString();
            _graph.MarkDirtyRepaint();
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
