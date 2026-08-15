using System.Text;
using PoRacer.Models;
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

        private RaceModel _raceModel;
        private VisualElement _panel;
        private Label _text;
        private readonly StringBuilder _builder = new();
        private int _frameCount;
        private float _frameSeconds;
        private float _fps;
        private bool _visible;

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

            var toggle = new Button(TogglePanel) { text = "DBG" };
            toggle.style.position = Position.Absolute;
            toggle.style.bottom = 6;
            toggle.style.left = 6;
            toggle.style.width = 44;
            toggle.style.height = 24;
            toggle.style.fontSize = 11;
            toggle.style.opacity = 0.75f;
            UiTheme.StyleButton(toggle);
            root.Add(toggle);

            _panel = new VisualElement { pickingMode = PickingMode.Ignore };
            _panel.style.position = Position.Absolute;
            _panel.style.bottom = 36;
            _panel.style.left = 6;
            _panel.style.maxWidth = 300;
            _panel.style.display = DisplayStyle.None;
            UiTheme.StylePanel(_panel);
            root.Add(_panel);

            _text = new Label { pickingMode = PickingMode.Ignore };
            _text.style.color = UiTheme.Text;
            _text.style.fontSize = 11;
            _text.style.whiteSpace = WhiteSpace.Pre;
            _panel.Add(_text);

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
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
            _builder.Append("FPS      ").Append(_fps.ToString("0.0"))
                .Append("  (").Append(frameMs.ToString("0.0")).Append(" ms)\n");
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
        }
    }
}
