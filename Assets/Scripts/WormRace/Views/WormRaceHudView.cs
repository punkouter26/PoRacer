using System;
using System.Globalization;
using System.Text;
using MessagePipe;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.WormRace
{
    /// <summary>
    /// The worm race HUD, UI Toolkit, hierarchy built in C# like the project's RaceHudView:
    /// a status card with one row per racer lane (name, training method, physics, distance,
    /// speed, state), a centre banner for 3-2-1-GO, and a results panel with times and the
    /// series tally. Refreshed on a 100 ms schedule by reading the model (DOCS/Plan-P1-Worm.md
    /// D5), and text is only rebuilt when the value it shows actually changed.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    [DisallowMultipleComponent]
    public sealed class WormRaceHudView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 100;
        private const long GO_BANNER_MS = 1000;
        private const float BASE_FONT_SIZE = 18f;
        private const float SMALL_FONT_SIZE = 14f;
        private const float BANNER_FONT_SIZE = 120f;
        private const float EDGE_MARGIN = 16f;
        private const float CARD_PADDING = 12f;
        private const float CARD_RADIUS = 8f;
        private const float SWATCH_SIZE = 16f;
        private const float CARD_WIDTH = 560f;
        private const float BANNER_TOP_PERCENT = 30f;
        private const string GO_TEXT = "GO!";

        private static readonly Color PanelColor = new(0.05f, 0.06f, 0.08f, 0.78f);
        private static readonly Color TextColor = new(0.96f, 0.96f, 0.96f, 1f);
        private static readonly Color DimTextColor = new(0.72f, 0.75f, 0.80f, 1f);
        private static readonly Color BannerColor = new(1f, 0.93f, 0.55f, 1f);

        private readonly StringBuilder _text = new(512);

        private UIDocument _document;
        private WormRaceModel _model;
        private ISubscriber<WormCountdownMessage> _countdownSubscriber;
        private ISubscriber<WormRaceFinishedMessage> _raceFinishedSubscriber;
        private IDisposable _countdownSubscription;
        private IDisposable _raceFinishedSubscription;
        private IVisualElementScheduledItem _refresh;

        private Label _headerLabel;
        private Label _messageLabel;
        private Label _bannerLabel;
        private VisualElement _resultsPanel;
        private Label _resultsLabel;
        private RacerRow[] _rows;

        private int _shownRaceNumber = -1;
        private int _shownPlannedRaces = -1;
        private int _shownTenthsOfSecond = -1;
        private WormRacePhase _shownPhase = WormRacePhase.Idle;
        private bool _headerShownOnce;
        private string _shownMessage = string.Empty;
        private int _shownResultsKey = -1;

        [Inject]
        public void Construct(WormRaceModel model,
                              ISubscriber<WormCountdownMessage> countdownSubscriber,
                              ISubscriber<WormRaceFinishedMessage> raceFinishedSubscriber)
        {
            _model = model;
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
                Debug.LogError("[WormRace] HUD was not injected; is it under the WormRaceLifetimeScope's scene?", this);
                return;
            }
            VisualElement root = _document.rootVisualElement;
            if (root == null)
            {
                Debug.LogError("[WormRace] the HUD's UIDocument has no panel; assign WormRacePanelSettings.", this);
                return;
            }
            Build(root);
            _countdownSubscription = _countdownSubscriber.Subscribe(OnCountdown);
            _raceFinishedSubscription = _raceFinishedSubscriber.Subscribe(OnRaceFinished);
            _refresh = root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
            Refresh();
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

            VisualElement card = Panel();
            card.style.position = Position.Absolute;
            card.style.left = EDGE_MARGIN;
            card.style.top = EDGE_MARGIN;
            card.style.width = CARD_WIDTH;
            root.Add(card);

            _headerLabel = Text(string.Empty, BASE_FONT_SIZE, TextColor, true);
            card.Add(_headerLabel);
            _messageLabel = Text(string.Empty, SMALL_FONT_SIZE, DimTextColor, false);
            card.Add(_messageLabel);

            _rows = new RacerRow[_model.Racers.Count];
            for (int lane = 0; lane < _model.Racers.Count; lane++)
            {
                _rows[lane] = new RacerRow(card);
            }

            _bannerLabel = Text(string.Empty, BANNER_FONT_SIZE, BannerColor, true);
            _bannerLabel.style.position = Position.Absolute;
            _bannerLabel.style.left = 0f;
            _bannerLabel.style.right = 0f;
            _bannerLabel.style.top = Length.Percent(BANNER_TOP_PERCENT);
            _bannerLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _bannerLabel.style.display = DisplayStyle.None;
            root.Add(_bannerLabel);

            _resultsPanel = Panel();
            _resultsPanel.style.position = Position.Absolute;
            _resultsPanel.style.right = EDGE_MARGIN;
            _resultsPanel.style.bottom = EDGE_MARGIN;
            _resultsPanel.style.display = DisplayStyle.None;
            _resultsLabel = Text(string.Empty, BASE_FONT_SIZE, TextColor, false);
            _resultsLabel.style.whiteSpace = WhiteSpace.Normal;
            _resultsPanel.Add(_resultsLabel);
            root.Add(_resultsPanel);
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
            RefreshHeader();
            for (int lane = 0; lane < _rows.Length; lane++)
            {
                _rows[lane].Refresh(_model.Racers[lane], _text);
            }
            RefreshResults();
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
            _text.Append("WORM RACE  ");
            if (_model.Mode == WormRaceMode.Race)
            {
                _text.Append("race ").Append(Mathf.Max(1, _model.RaceNumber)).Append(" / ")
                     .Append(Mathf.Max(1, _model.PlannedRaces));
            }
            else
            {
                _text.Append(_model.Mode.ToString());
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
            bool visible = _model.Phase == WormRacePhase.Results || _model.Phase == WormRacePhase.SeriesComplete
                        || _model.Phase == WormRacePhase.SelfTestComplete || _model.Phase == WormRacePhase.Error;
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
            if (_model.Phase == WormRacePhase.Error || _model.Phase == WormRacePhase.SelfTestComplete)
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
                        WormRacerModel racer = _model.Racers[lane];
                        if (racer.Place != place)
                        {
                            continue;
                        }
                        _text.Append(place).Append(". ").Append(racer.Name).Append(" (").Append(racer.Method).Append(")   ");
                        if (racer.Status == WormRacerStatus.Finished)
                        {
                            _text.Append(racer.FinishTimeSeconds.ToString("0.00", CultureInfo.InvariantCulture)).Append(" s");
                        }
                        else if (racer.Status == WormRacerStatus.TimedOut)
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

        private void OnCountdown(WormCountdownMessage message)
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

        private void OnRaceFinished(WormRaceFinishedMessage message)
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

        /// <summary>One worm's line on the status card.</summary>
        private sealed class RacerRow
        {
            private readonly VisualElement _swatch;
            private readonly Label _name;
            private readonly Label _detail;
            private readonly Label _numbers;
            private int _shownCentimetres = int.MinValue;
            private int _shownCentiSpeed = int.MinValue;
            private int _shownCentiSeconds = int.MinValue;
            private WormRacerStatus _shownStatus = (WormRacerStatus)(-1);
            private Color _shownColor = Color.clear;
            private string _shownName = string.Empty;
            private int _shownBrainState = -1;

            public RacerRow(VisualElement parent)
            {
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
                row.Add(column);
                _name = Text(string.Empty, BASE_FONT_SIZE, TextColor, true);
                column.Add(_name);
                _detail = Text(string.Empty, SMALL_FONT_SIZE, DimTextColor, false);
                // Method | physics | NO BRAIN can outgrow the card; wrap rather than clip.
                _detail.style.whiteSpace = WhiteSpace.Normal;
                column.Add(_detail);
                _numbers = Text(string.Empty, BASE_FONT_SIZE, TextColor, false);
                column.Add(_numbers);
            }

            public void Refresh(WormRacerModel racer, StringBuilder text)
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
                        text.Append("  |  NO BRAIN (holds straight)");
                    }
                    _detail.text = text.ToString();
                }

                int centimetres = Mathf.RoundToInt(racer.Distance * 100f);
                int centiSpeed = Mathf.RoundToInt(racer.Speed * 100f);
                int centiSeconds = Mathf.RoundToInt(racer.FinishTimeSeconds * 100f);
                if (centimetres == _shownCentimetres && centiSpeed == _shownCentiSpeed
                    && centiSeconds == _shownCentiSeconds && racer.Status == _shownStatus)
                {
                    return;
                }
                _shownCentimetres = centimetres;
                _shownCentiSpeed = centiSpeed;
                _shownCentiSeconds = centiSeconds;
                _shownStatus = racer.Status;

                text.Clear();
                text.Append((centimetres / 100f).ToString("0.00", CultureInfo.InvariantCulture)).Append(" m    ")
                    .Append((centiSpeed / 100f).ToString("0.00", CultureInfo.InvariantCulture)).Append(" m/s    ");
                switch (racer.Status)
                {
                    case WormRacerStatus.Finished:
                        text.Append("FINISHED ").Append((centiSeconds / 100f).ToString("0.00", CultureInfo.InvariantCulture)).Append(" s");
                        break;
                    case WormRacerStatus.TimedOut:
                        text.Append("TIME LIMIT");
                        break;
                    case WormRacerStatus.Failed:
                        text.Append("OUT");
                        break;
                    case WormRacerStatus.Racing:
                        text.Append("RACING");
                        break;
                    default:
                        text.Append("ON THE LINE");
                        break;
                }
                _numbers.text = text.ToString();
            }
        }
    }
}
