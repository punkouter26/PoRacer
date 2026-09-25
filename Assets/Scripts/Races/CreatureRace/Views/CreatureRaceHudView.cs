using System;
using System.Globalization;
using System.Text;
using MessagePipe;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.CreatureRace
{
    /// <summary>
    /// The creature race HUD, UI Toolkit, hierarchy built in C# like the project's RaceHudView:
    /// a status card with one row per racer lane (name, training method, physics, distance,
    /// speed, state), a centre banner for 3-2-1-GO, and a results panel with times and the
    /// series tally. Refreshed on a 100 ms schedule by reading the model (DOCS/Plan-P1-Worm.md
    /// D5), and text is only rebuilt when the value it shows actually changed.
    ///
    /// Layout follows the game's portrait corner rules (UiTheme in PoRacer.Runtime, which this
    /// standalone assembly does not reference): title top-left, fps top-centre, version
    /// bottom-right, inside the device safe area. The status card spans the width under the
    /// top band and the results sit above the bottom band, both capped at CARD_MAX_WIDTH so a
    /// landscape window keeps the old card size. The harness has no menu to return to and no
    /// debug sheet, so the MENU and DBG corners stay empty rather than hold a dead button.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    [DisallowMultipleComponent]
    public sealed class CreatureRaceHudView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 100;
        private const long GO_BANNER_MS = 1000;
        private const float BASE_FONT_SIZE = 18f;
        private const float SMALL_FONT_SIZE = 14f;
        private const float BANNER_FONT_SIZE = 120f;
        private const float EDGE_MARGIN = 8f;
        private const float CARD_PADDING = 8f;
        private const float CARD_RADIUS = 8f;
        private const float SWATCH_SIZE = 16f;
        private const float CARD_MAX_WIDTH = 560f;
        // One furniture band, matching the game's 48 dp touch-height bands.
        private const float BAND_HEIGHT = 48f;
        private const float BANNER_TOP_PERCENT = 30f;
        private const float FPS_WINDOW_SECONDS = 0.5f;
        private const string GO_TEXT = "GO!";
        // Same element names as UiTheme's furniture, so one audit can find both HUDs' anchors.
        private const string FURNITURE_TITLE = "Furniture.Title";
        private const string FURNITURE_FPS = "Furniture.Fps";
        private const string FURNITURE_VERSION = "Furniture.Version";

        private static readonly Color PanelColor = new(0.05f, 0.06f, 0.08f, 0.78f);
        private static readonly Color TextColor = new(0.96f, 0.96f, 0.96f, 1f);
        private static readonly Color DimTextColor = new(0.72f, 0.75f, 0.80f, 1f);
        private static readonly Color BannerColor = new(1f, 0.93f, 0.55f, 1f);

        private readonly StringBuilder _text = new(512);

        private UIDocument _document;
        private CreatureRaceModel _model;
        private CreatureRaceConfig _config;
        private ISubscriber<CreatureCountdownMessage> _countdownSubscriber;
        private ISubscriber<CreatureRaceFinishedMessage> _raceFinishedSubscriber;
        private IDisposable _countdownSubscription;
        private IDisposable _raceFinishedSubscription;
        private IVisualElementScheduledItem _refresh;

        private Label _headerLabel;
        private Label _messageLabel;
        private Label _fpsLabel;
        private int _fpsFrames;
        private float _fpsSeconds;
        private int _shownFps = -1;
        private Label _bannerLabel;
        private VisualElement _resultsPanel;
        private Label _resultsLabel;
        private RacerRow[] _rows;

        private int _shownRaceNumber = -1;
        private int _shownPlannedRaces = -1;
        private int _shownTenthsOfSecond = -1;
        private CreatureRacePhase _shownPhase = CreatureRacePhase.Idle;
        private bool _headerShownOnce;
        private string _shownMessage = string.Empty;
        private int _shownResultsKey = -1;

        [Inject]
        public void Construct(CreatureRaceModel model, CreatureRaceConfig config,
                              ISubscriber<CreatureCountdownMessage> countdownSubscriber,
                              ISubscriber<CreatureRaceFinishedMessage> raceFinishedSubscriber)
        {
            _model = model;
            _config = config;
            _countdownSubscriber = countdownSubscriber;
            _raceFinishedSubscriber = raceFinishedSubscriber;
        }

        private void Awake()
        {
            _document = GetComponent<UIDocument>();
        }

        private void Start()
        {
            if (_model == null)
            {
                Debug.LogError("[CreatureRace] HUD was not injected; is it in the scene of a creature race lifetime scope?", this);
                return;
            }
            VisualElement root = _document.rootVisualElement;
            if (root == null)
            {
                Debug.LogError("[CreatureRace] the HUD's UIDocument has no panel; assign the race's panel settings (re-run the scene builder).", this);
                return;
            }
            Build(root);
            _countdownSubscription = _countdownSubscriber.Subscribe(OnCountdown);
            _raceFinishedSubscription = _raceFinishedSubscriber.Subscribe(OnRaceFinished);
            _refresh = root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
            Refresh();
        }

        private void Update()
        {
            _fpsFrames++;
            _fpsSeconds += Time.unscaledDeltaTime;
        }

        private void OnDestroy()
        {
            _countdownSubscription?.Dispose();
            _raceFinishedSubscription?.Dispose();
            _refresh?.Pause();
        }

        // ---------------------------------------------------------------- build --

        private void Build(VisualElement root)
        {
            root.pickingMode = PickingMode.Ignore;
            VisualElement safe = BuildSafeRoot(root);
            BuildFurniture(safe);

            VisualElement card = Panel();
            card.style.position = Position.Absolute;
            card.style.left = EDGE_MARGIN;
            card.style.right = EDGE_MARGIN;
            card.style.top = BAND_HEIGHT + EDGE_MARGIN;
            card.style.maxWidth = CARD_MAX_WIDTH;
            safe.Add(card);

            _headerLabel = Text(string.Empty, BASE_FONT_SIZE, TextColor, true);
            card.Add(_headerLabel);
            _messageLabel = Text(string.Empty, SMALL_FONT_SIZE, DimTextColor, false);
            card.Add(_messageLabel);

            _rows = new RacerRow[_model.Racers.Count];
            for (int lane = 0; lane < _model.Racers.Count; lane++)
            {
                _rows[lane] = new RacerRow(card, _config.HoldLabel);
            }

            _bannerLabel = Text(string.Empty, BANNER_FONT_SIZE, BannerColor, true);
            _bannerLabel.style.position = Position.Absolute;
            _bannerLabel.style.left = 0f;
            _bannerLabel.style.right = 0f;
            _bannerLabel.style.top = Length.Percent(BANNER_TOP_PERCENT);
            _bannerLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _bannerLabel.style.display = DisplayStyle.None;
            safe.Add(_bannerLabel);

            _resultsPanel = Panel();
            _resultsPanel.style.position = Position.Absolute;
            _resultsPanel.style.left = EDGE_MARGIN;
            _resultsPanel.style.right = EDGE_MARGIN;
            _resultsPanel.style.bottom = BAND_HEIGHT + EDGE_MARGIN;
            _resultsPanel.style.maxWidth = CARD_MAX_WIDTH;
            _resultsPanel.style.display = DisplayStyle.None;
            _resultsLabel = Text(string.Empty, SMALL_FONT_SIZE, TextColor, false);
            _resultsLabel.style.whiteSpace = WhiteSpace.Normal;
            _resultsPanel.Add(_resultsLabel);
            safe.Add(_resultsPanel);
        }

        /// <summary>Title top-left, fps top-centre, version bottom-right, one band tall each.</summary>
        private void BuildFurniture(VisualElement safe)
        {
            Label title = Text(_config.Title, BASE_FONT_SIZE, BannerColor, true);
            title.name = FURNITURE_TITLE;
            Pin(title, left: EDGE_MARGIN, right: -1f, top: true);
            title.style.unityTextAlign = TextAnchor.MiddleLeft;
            safe.Add(title);

            _fpsLabel = Text(string.Empty, SMALL_FONT_SIZE, TextColor, true);
            _fpsLabel.name = FURNITURE_FPS;
            Pin(_fpsLabel, left: 0f, right: 0f, top: true);
            _fpsLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            safe.Add(_fpsLabel);

            Label version = Text($"v{Application.version}", SMALL_FONT_SIZE, DimTextColor, false);
            version.name = FURNITURE_VERSION;
            Pin(version, left: -1f, right: EDGE_MARGIN, top: false);
            version.style.unityTextAlign = TextAnchor.MiddleRight;
            safe.Add(version);
        }

        /// <summary>Pins a label into a furniture band; a negative inset leaves that side free.</summary>
        private static void Pin(VisualElement element, float left, float right, bool top)
        {
            element.style.position = Position.Absolute;
            element.style.height = BAND_HEIGHT;
            if (left >= 0f)
            {
                element.style.left = left;
            }
            if (right >= 0f)
            {
                element.style.right = right;
            }
            if (top)
            {
                element.style.top = 0f;
            }
            else
            {
                element.style.bottom = 0f;
            }
        }

        /// <summary>
        /// Full-screen layer inset by the device safe area (notches, rounded corners, home
        /// bars), the same rule as UiTheme.BuildSafeRoot in the game assembly.
        /// </summary>
        private static VisualElement BuildSafeRoot(VisualElement root)
        {
            var safe = new VisualElement { pickingMode = PickingMode.Ignore };
            safe.style.position = Position.Absolute;
            safe.style.left = 0f;
            safe.style.top = 0f;
            safe.style.right = 0f;
            safe.style.bottom = 0f;
            root.Add(safe);
            safe.RegisterCallback<GeometryChangedEvent>(_ => ApplySafeInsets(root, safe));
            return safe;
        }

        private static void ApplySafeInsets(VisualElement root, VisualElement safe)
        {
            float rootWidth = root.resolvedStyle.width;
            if (Screen.width <= 0 || rootWidth <= 0f || float.IsNaN(rootWidth))
            {
                return;
            }
            Rect area = Screen.safeArea;
            float scale = rootWidth / Screen.width;
            safe.style.left = Mathf.Max(0f, area.xMin * scale);
            safe.style.right = Mathf.Max(0f, (Screen.width - area.xMax) * scale);
            safe.style.top = Mathf.Max(0f, (Screen.height - area.yMax) * scale);
            safe.style.bottom = Mathf.Max(0f, area.yMin * scale);
        }

        private static VisualElement Panel()
        {
            var panel = new VisualElement { pickingMode = PickingMode.Ignore };
            panel.style.backgroundColor = PanelColor;
            panel.style.paddingLeft = CARD_PADDING;
            panel.style.paddingRight = CARD_PADDING;
            panel.style.paddingTop = CARD_PADDING;
            panel.style.paddingBottom = CARD_PADDING;
            panel.style.borderTopLeftRadius = CARD_RADIUS;
            panel.style.borderTopRightRadius = CARD_RADIUS;
            panel.style.borderBottomLeftRadius = CARD_RADIUS;
            panel.style.borderBottomRightRadius = CARD_RADIUS;
            return panel;
        }

        private static Label Text(string content, float size, Color color, bool bold)
        {
            var label = new Label(content) { pickingMode = PickingMode.Ignore };
            label.style.fontSize = size;
            label.style.color = color;
            label.style.unityFontStyleAndWeight = bold ? FontStyle.Bold : FontStyle.Normal;
            return label;
        }

        // -------------------------------------------------------------- refresh --

        private void Refresh()
        {
            RefreshFps();
            RefreshHeader();
            for (int lane = 0; lane < _rows.Length; lane++)
            {
                _rows[lane].Refresh(_model.Racers[lane], _text);
            }
            RefreshResults();
        }

        private void RefreshFps()
        {
            if (_fpsSeconds < FPS_WINDOW_SECONDS)
            {
                return;
            }
            int fps = Mathf.RoundToInt(_fpsFrames / _fpsSeconds);
            _fpsFrames = 0;
            _fpsSeconds = 0f;
            if (fps == _shownFps)
            {
                return;
            }
            _shownFps = fps;
            _fpsLabel.text = fps + " FPS";
        }

        private void RefreshHeader()
        {
            int tenths = Mathf.FloorToInt(_model.ElapsedSeconds * 10f);
            if (_headerShownOnce && _model.RaceNumber == _shownRaceNumber && _model.PlannedRaces == _shownPlannedRaces
                && tenths == _shownTenthsOfSecond && _model.Phase == _shownPhase)
            {
                RefreshMessage();
                return;
            }
            _headerShownOnce = true;
            _shownRaceNumber = _model.RaceNumber;
            _shownPlannedRaces = _model.PlannedRaces;
            _shownTenthsOfSecond = tenths;
            _shownPhase = _model.Phase;

            _text.Clear();
            if (_model.IsRaceMode)
            {
                _text.Append("race ").Append(Mathf.Max(1, _model.RaceNumber)).Append(" / ")
                     .Append(Mathf.Max(1, _model.PlannedRaces));
            }
            else
            {
                _text.Append(_model.ModeLabel);
            }
            _text.Append("   ").Append((tenths / 10f).ToString("0.0", CultureInfo.InvariantCulture)).Append(" s   ")
                 .Append(_model.Phase.ToString().ToUpperInvariant());
            _headerLabel.text = _text.ToString();
            RefreshMessage();
        }

        private void RefreshMessage()
        {
            if (_model.Message == _shownMessage)
            {
                return;
            }
            _shownMessage = _model.Message;
            _messageLabel.text = _shownMessage;
        }

        private void RefreshResults()
        {
            bool visible = _model.Phase == CreatureRacePhase.Results || _model.Phase == CreatureRacePhase.SeriesComplete
                        || _model.Phase == CreatureRacePhase.SelfTestComplete || _model.Phase == CreatureRacePhase.Error;
            _resultsPanel.style.display = visible ? DisplayStyle.Flex : DisplayStyle.None;
            if (!visible)
            {
                return;
            }
            int key = _model.CompletedRaces * 16 + (int)_model.Phase;
            if (key == _shownResultsKey)
            {
                return;
            }
            _shownResultsKey = key;
            _resultsLabel.text = BuildResultsText();
        }

        private string BuildResultsText()
        {
            _text.Clear();
            if (_model.Phase == CreatureRacePhase.Error || _model.Phase == CreatureRacePhase.SelfTestComplete)
            {
                _text.Append(_model.Message);
            }
            else
            {
                _text.Append("RACE ").Append(_model.RaceNumber).Append(" RESULT");
                if (_model.LastWinnerLane >= 0)
                {
                    _text.Append(": ").Append(_model.Racers[_model.LastWinnerLane].Name).Append(" wins");
                }
                _text.Append(" (").Append(_model.LastEndReason).Append(")\n");
                for (int place = 1; place <= _model.Racers.Count; place++)
                {
                    for (int lane = 0; lane < _model.Racers.Count; lane++)
                    {
                        CreatureRacerModel racer = _model.Racers[lane];
                        if (racer.Place != place)
                        {
                            continue;
                        }
                        _text.Append(place).Append(". ").Append(racer.Name).Append(" (").Append(racer.Method).Append(")   ");
                        if (racer.Status == CreatureRacerStatus.Finished)
                        {
                            _text.Append(racer.FinishTimeSeconds.ToString("0.00", CultureInfo.InvariantCulture)).Append(" s");
                        }
                        else if (racer.Status == CreatureRacerStatus.TimedOut)
                        {
                            _text.Append("time limit");
                        }
                        else
                        {
                            _text.Append("out");
                        }
                        _text.Append("   ").Append(racer.Distance.ToString("0.00", CultureInfo.InvariantCulture)).Append(" m   avg ")
                             .Append(racer.AverageSpeed.ToString("0.000", CultureInfo.InvariantCulture)).Append(" m/s\n");
                    }
                }
                _text.Append("Series: ");
                for (int lane = 0; lane < _model.Racers.Count; lane++)
                {
                    if (lane > 0)
                    {
                        _text.Append("  -  ");
                    }
                    _text.Append(_model.Racers[lane].Method).Append(' ').Append(_model.WinsFor(lane));
                }
                _text.Append("   after ").Append(_model.CompletedRaces).Append(" of ").Append(_model.PlannedRaces);
            }
            if (!string.IsNullOrEmpty(_model.ReportPath))
            {
                _text.Append("\n").Append(_model.ReportPath);
            }
            return _text.ToString();
        }

        // ------------------------------------------------------------- messages --

        private void OnCountdown(CreatureCountdownMessage message)
        {
            if (_bannerLabel == null)
            {
                return;
            }
            _bannerLabel.style.display = DisplayStyle.Flex;
            if (message.Value > 0)
            {
                _bannerLabel.text = message.Value.ToString(CultureInfo.InvariantCulture);
                return;
            }
            _bannerLabel.text = GO_TEXT;
            _bannerLabel.schedule.Execute(HideBanner).StartingIn(GO_BANNER_MS);
        }

        private void OnRaceFinished(CreatureRaceFinishedMessage message)
        {
            // Force the results panel to rebuild on the next tick even if the key matches.
            _shownResultsKey = -1;
        }

        private void HideBanner()
        {
            if (_bannerLabel.text == GO_TEXT)
            {
                _bannerLabel.style.display = DisplayStyle.None;
            }
        }

        /// <summary>One racer's line on the status card.</summary>
        private sealed class RacerRow
        {
            private readonly VisualElement _swatch;
            private readonly Label _name;
            private readonly Label _detail;
            private readonly Label _numbers;
            private readonly string _holdLabel;
            private int _shownCentimetres = int.MinValue;
            private int _shownCentiSpeed = int.MinValue;
            private int _shownCentiSeconds = int.MinValue;
            private CreatureRacerStatus _shownStatus = (CreatureRacerStatus)(-1);
            private Color _shownColor = Color.clear;
            private string _shownName = string.Empty;
            private int _shownBrainState = -1;
            private bool _shownDown;

            public RacerRow(VisualElement parent, string holdLabel)
            {
                _holdLabel = holdLabel;
                var row = new VisualElement { pickingMode = PickingMode.Ignore };
                row.style.flexDirection = FlexDirection.Row;
                row.style.alignItems = Align.Center;
                row.style.marginTop = CARD_PADDING * 0.5f;
                parent.Add(row);

                _swatch = new VisualElement { pickingMode = PickingMode.Ignore };
                _swatch.style.width = SWATCH_SIZE;
                _swatch.style.height = SWATCH_SIZE;
                _swatch.style.marginRight = CARD_PADDING * 0.75f;
                _swatch.style.borderTopLeftRadius = SWATCH_SIZE * 0.5f;
                _swatch.style.borderTopRightRadius = SWATCH_SIZE * 0.5f;
                _swatch.style.borderBottomLeftRadius = SWATCH_SIZE * 0.5f;
                _swatch.style.borderBottomRightRadius = SWATCH_SIZE * 0.5f;
                row.Add(_swatch);

                var column = new VisualElement { pickingMode = PickingMode.Ignore };
                column.style.flexGrow = 1f;
                column.style.flexShrink = 1f;
                column.style.minWidth = 0f;
                row.Add(column);
                // Name and numbers share the first line; the name gives way first.
                var line = new VisualElement { pickingMode = PickingMode.Ignore };
                line.style.flexDirection = FlexDirection.Row;
                line.style.justifyContent = Justify.SpaceBetween;
                column.Add(line);
                _name = Text(string.Empty, BASE_FONT_SIZE, TextColor, true);
                _name.style.flexShrink = 1f;
                _name.style.minWidth = 0f;
                _name.style.overflow = Overflow.Hidden;
                _name.style.textOverflow = TextOverflow.Ellipsis;
                _name.style.marginRight = CARD_PADDING;
                line.Add(_name);
                _numbers = Text(string.Empty, SMALL_FONT_SIZE, TextColor, false);
                _numbers.style.flexShrink = 0f;
                line.Add(_numbers);
                _detail = Text(string.Empty, SMALL_FONT_SIZE, DimTextColor, false);
                // Method | physics | NO BRAIN can outgrow the card; wrap rather than clip.
                _detail.style.whiteSpace = WhiteSpace.Normal;
                column.Add(_detail);
            }

            public void Refresh(CreatureRacerModel racer, StringBuilder text)
            {
                int brainState = racer.BrainReady ? 1 : 0;
                if (racer.Name != _shownName || racer.Color != _shownColor || brainState != _shownBrainState)
                {
                    _shownName = racer.Name;
                    _shownColor = racer.Color;
                    _shownBrainState = brainState;
                    _swatch.style.backgroundColor = racer.Color;
                    _name.text = racer.Name;
                    text.Clear();
                    text.Append(racer.Method).Append(" brain  |  ").Append(racer.Physics);
                    if (!racer.BrainReady)
                    {
                        text.Append("  |  NO BRAIN (").Append(_holdLabel).Append(')');
                    }
                    _detail.text = text.ToString();
                }

                int centimetres = Mathf.RoundToInt(racer.Distance * 100f);
                int centiSpeed = Mathf.RoundToInt(racer.Speed * 100f);
                int centiSeconds = Mathf.RoundToInt(racer.FinishTimeSeconds * 100f);
                if (centimetres == _shownCentimetres && centiSpeed == _shownCentiSpeed
                    && centiSeconds == _shownCentiSeconds && racer.Status == _shownStatus
                    && racer.IsDown == _shownDown)
                {
                    return;
                }
                _shownCentimetres = centimetres;
                _shownCentiSpeed = centiSpeed;
                _shownCentiSeconds = centiSeconds;
                _shownStatus = racer.Status;
                _shownDown = racer.IsDown;

                text.Clear();
                text.Append((centimetres / 100f).ToString("0.00", CultureInfo.InvariantCulture)).Append(" m  ")
                    .Append((centiSpeed / 100f).ToString("0.00", CultureInfo.InvariantCulture)).Append(" m/s  ");
                switch (racer.Status)
                {
                    case CreatureRacerStatus.Finished:
                        text.Append("FIN ").Append((centiSeconds / 100f).ToString("0.00", CultureInfo.InvariantCulture)).Append(" s");
                        break;
                    case CreatureRacerStatus.TimedOut:
                        text.Append("TIME LIMIT");
                        break;
                    case CreatureRacerStatus.Failed:
                        text.Append("OUT");
                        break;
                    case CreatureRacerStatus.Racing:
                        // Fallen: it lies until its own policy gets it up (rule H).
                        text.Append(racer.IsDown ? "DOWN" : "RACING");
                        break;
                    default:
                        text.Append("READY");
                        break;
                }
                _numbers.text = text.ToString();
            }
        }
    }
}
