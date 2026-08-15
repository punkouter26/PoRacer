using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Runtime UI Toolkit HUD, hierarchy built entirely in C# (no .uxml/.uss).
    /// Version stamp top-left (non-pickable); leaderboard refreshed on a schedule
    /// by reading the Models — no per-frame polling in Update.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class RaceHudView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const int MAX_LEADERBOARD_ROWS = 10;
        private const int MAX_PROGRESS_DOTS = 50;
        private const int TICKER_ANIMATION_MS = 300;
        private const int ROW_POP_MS = 250;
        private const float GO_BANNER_SECONDS = 1.5f;

        private RaceModel _raceModel;
        private EloModel _eloModel;
        private RaceConfigModel _configModel;
        private CommentaryModel _commentaryModel;
        private Systems_Spawn _spawn;
        private VisualElement _hudRoot;
        private Label _statusLabel;
        private Label _bannerLabel;
        private VisualElement _leaderboard;
        private VisualElement _progressStrip;
        private readonly System.Collections.Generic.List<VisualElement> _rows = new();
        private readonly System.Collections.Generic.List<VisualElement> _rowChips = new();
        private readonly System.Collections.Generic.List<Label> _rowLabels = new();
        private readonly System.Collections.Generic.List<Label> _tickerLabels = new();
        private readonly System.Collections.Generic.List<VisualElement> _progressDots = new();
        private readonly System.Collections.Generic.List<RacerState> _sortBuffer = new();
        private readonly System.Collections.Generic.Dictionary<string, int> _lastRowByRacer = new();
        private int _commentaryVersion = -1;
        private bool _wasRaceActive;
        private float _goBannerUntil;
        private string _lastLeaderId;
        private float _lastLeaderProgress;
        private float _lastLeaderSampleTime;
        private float _leaderSpeed;

        [Inject]
        public void Construct(
            RaceModel raceModel,
            EloModel eloModel,
            RaceConfigModel configModel,
            CommentaryModel commentaryModel,
            Systems_Spawn spawn)
        {
            _raceModel = raceModel;
            _eloModel = eloModel;
            _configModel = configModel;
            _commentaryModel = commentaryModel;
            _spawn = spawn;
        }

        private void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            root.pickingMode = PickingMode.Ignore;
            _hudRoot = root;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            var versionLabel = new Label($"v{Application.version}")
            {
                pickingMode = PickingMode.Ignore
            };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.top = 4;
            versionLabel.style.left = 6;
            versionLabel.style.color = UiTheme.TextDim;
            versionLabel.style.fontSize = 12;
            safeRoot.Add(versionLabel);

            var panel = new VisualElement();
            panel.style.position = Position.Absolute;
            panel.style.top = 24;
            panel.style.left = 6;
            panel.style.right = 6;
            UiTheme.StylePanel(panel);
            panel.pickingMode = PickingMode.Ignore;
            safeRoot.Add(panel);

            _statusLabel = new Label { pickingMode = PickingMode.Ignore };
            _statusLabel.style.color = UiTheme.Text;
            _statusLabel.style.fontSize = 14;
            _statusLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            panel.Add(_statusLabel);

            // Commentary ticker: newest line bright, older lines dimmed.
            for (int tickerIndex = 0; tickerIndex < CommentaryModel.MAX_LINES; tickerIndex++)
            {
                var tickerLabel = new Label { pickingMode = PickingMode.Ignore };
                tickerLabel.style.color = tickerIndex == 0 ? UiTheme.Gold : UiTheme.TextDim;
                tickerLabel.style.fontSize = 12;
                tickerLabel.style.marginTop = tickerIndex == 0 ? 4 : 1;
                if (tickerIndex == 0)
                {
                    tickerLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
                }
                _tickerLabels.Add(tickerLabel);
                panel.Add(tickerLabel);
            }

            // Progress strip: every racer is a colored dot moving start -> finish.
            _progressStrip = new VisualElement { pickingMode = PickingMode.Ignore };
            _progressStrip.style.height = 10;
            _progressStrip.style.marginTop = 6;
            _progressStrip.style.backgroundColor = new Color(0f, 0f, 0f, 0.35f);
            _progressStrip.style.borderRightColor = UiTheme.Gold;
            _progressStrip.style.borderRightWidth = 2;
            UiTheme.SetRadius(_progressStrip, 5f);
            panel.Add(_progressStrip);

            _leaderboard = new VisualElement { pickingMode = PickingMode.Ignore };
            _leaderboard.style.marginTop = 4;
            panel.Add(_leaderboard);

            _bannerLabel = new Label { pickingMode = PickingMode.Ignore };
            _bannerLabel.style.position = Position.Absolute;
            _bannerLabel.style.top = new Length(40f, LengthUnit.Percent);
            _bannerLabel.style.left = 0;
            _bannerLabel.style.right = 0;
            _bannerLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _bannerLabel.style.fontSize = 30;
            _bannerLabel.style.color = UiTheme.Gold;
            _bannerLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _bannerLabel.style.display = DisplayStyle.None;
            safeRoot.Add(_bannerLabel);

            var menuButton = new Button(() => _spawn.RequestMenu()) { text = "MENU" };
            menuButton.style.position = Position.Absolute;
            menuButton.style.top = 4;
            menuButton.style.right = 6;
            menuButton.style.width = 64;
            menuButton.style.height = 26;
            UiTheme.StyleButton(menuButton);
            UiTheme.AddHover(menuButton);
            safeRoot.Add(menuButton);

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
        }

        private void Refresh()
        {
            if (_raceModel == null)
            {
                return;
            }
            if (_configModel != null && _configModel.MenuVisible)
            {
                _hudRoot.style.display = DisplayStyle.None;
                return;
            }
            _hudRoot.style.display = DisplayStyle.Flex;

            RacerState leader = null;
            RacerState winner = null;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                if (racer.Place == 1)
                {
                    winner = racer;
                }
                if (racer.Status == RacerStatus.Racing && (leader == null || racer.Progress > leader.Progress))
                {
                    leader = racer;
                }
            }

            if (_raceModel.RaceActive && !_wasRaceActive)
            {
                _goBannerUntil = Time.unscaledTime + GO_BANNER_SECONDS;
            }
            _wasRaceActive = _raceModel.RaceActive;

            UpdateLeaderSpeed(leader);
            string speed = leader != null && _leaderSpeed > 0.05f ? $"  {_leaderSpeed:0.0} m/s" : string.Empty;
            string ticker = leader != null ? $"   Leader: {leader.DisplayName}  {leader.Progress:0.0}m{speed}" : string.Empty;
            _statusLabel.text = _raceModel.RaceActive
                ? $"Race {_raceModel.RaceNumber}  {_raceModel.ElapsedSeconds:0}s{ticker}"
                : $"Race {_raceModel.RaceNumber} finished — next starting soon";

            if (_commentaryModel != null && _commentaryModel.Version != _commentaryVersion)
            {
                _commentaryVersion = _commentaryModel.Version;
                for (int tickerIndex = 0; tickerIndex < _tickerLabels.Count; tickerIndex++)
                {
                    _tickerLabels[tickerIndex].text = tickerIndex < _commentaryModel.Lines.Count
                        ? _commentaryModel.Lines[tickerIndex]
                        : string.Empty;
                }
                // Newest line slides in from the left and fades up.
                Label newest = _tickerLabels[0];
                newest.style.opacity = 0f;
                newest.experimental.animation.Start(0f, 1f, TICKER_ANIMATION_MS, (element, value) =>
                {
                    element.style.opacity = value;
                    element.style.translate = new Translate(-16f * (1f - value), 0f);
                });
            }

            RefreshProgressStrip();

            if (winner != null)
            {
                _bannerLabel.text = $"WINNER  {winner.DisplayName}  {winner.FinishTime:0.0}s";
                _bannerLabel.style.fontSize = 30;
                _bannerLabel.style.display = DisplayStyle.Flex;
            }
            else if (Time.unscaledTime < _goBannerUntil)
            {
                _bannerLabel.text = "GO!";
                _bannerLabel.style.fontSize = 52;
                _bannerLabel.style.display = DisplayStyle.Flex;
            }
            else
            {
                _bannerLabel.style.display = DisplayStyle.None;
            }

            _sortBuffer.Clear();
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                _sortBuffer.Add(_raceModel.Racers[racerIndex]);
            }
            _sortBuffer.Sort(CompareRacers);

            int visibleRows = _sortBuffer.Count < MAX_LEADERBOARD_ROWS ? _sortBuffer.Count : MAX_LEADERBOARD_ROWS;
            while (_rows.Count < visibleRows)
            {
                var row = new VisualElement { pickingMode = PickingMode.Ignore };
                row.style.flexDirection = FlexDirection.Row;
                row.style.alignItems = Align.Center;
                row.style.marginTop = 2;
                row.style.paddingLeft = 4;
                row.style.paddingTop = 1;
                row.style.paddingBottom = 1;
                UiTheme.SetRadius(row, 3f);

                var chip = new VisualElement { pickingMode = PickingMode.Ignore };
                chip.style.width = 8;
                chip.style.height = 8;
                chip.style.marginRight = 5;
                UiTheme.SetRadius(chip, 2f);
                row.Add(chip);

                var label = new Label { pickingMode = PickingMode.Ignore };
                label.style.color = UiTheme.Text;
                label.style.fontSize = 12;
                row.Add(label);

                _rows.Add(row);
                _rowChips.Add(chip);
                _rowLabels.Add(label);
                _leaderboard.Add(row);
            }
            for (int rowIndex = 0; rowIndex < _rows.Count; rowIndex++)
            {
                if (rowIndex >= visibleRows)
                {
                    _rows[rowIndex].style.display = DisplayStyle.None;
                    continue;
                }
                _rows[rowIndex].style.display = DisplayStyle.Flex;
                RacerState racer = _sortBuffer[rowIndex];
                // Pop the row when its racer climbed the board since last refresh.
                if (_lastRowByRacer.TryGetValue(racer.RacerId, out int previousRow) && rowIndex < previousRow)
                {
                    _rows[rowIndex].experimental.animation.Start(0f, 1f, ROW_POP_MS, (element, value) =>
                    {
                        element.style.translate = new Translate(10f * (1f - value), 0f);
                    });
                }
                string status = racer.Status switch
                {
                    RacerStatus.Finished => $"#{racer.Place} {racer.FinishTime:0.0}s",
                    RacerStatus.Dnf => "DNF",
                    _ => $"{racer.Progress:0.0}m"
                };
                float rating = _eloModel.GetRating(racer.CreatureId);
                _rowChips[rowIndex].style.backgroundColor = racer.Tint;
                _rowLabels[rowIndex].text = $"{rowIndex + 1}. {racer.DisplayName}  {status}  ELO {rating:0}";
                _rows[rowIndex].style.backgroundColor = racer.Place switch
                {
                    1 => new Color(UiTheme.Gold.r, UiTheme.Gold.g, UiTheme.Gold.b, 0.18f),
                    2 => new Color(UiTheme.Silver.r, UiTheme.Silver.g, UiTheme.Silver.b, 0.14f),
                    3 => new Color(UiTheme.Bronze.r, UiTheme.Bronze.g, UiTheme.Bronze.b, 0.14f),
                    _ => StyleKeyword.Null
                };
            }
            _lastRowByRacer.Clear();
            for (int rowIndex = 0; rowIndex < visibleRows; rowIndex++)
            {
                _lastRowByRacer[_sortBuffer[rowIndex].RacerId] = rowIndex;
            }
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

        private void RefreshProgressStrip()
        {
            int dotCount = _raceModel.Racers.Count < MAX_PROGRESS_DOTS ? _raceModel.Racers.Count : MAX_PROGRESS_DOTS;
            _progressStrip.style.display = dotCount > 0 ? DisplayStyle.Flex : DisplayStyle.None;
            while (_progressDots.Count < dotCount)
            {
                var dot = new VisualElement { pickingMode = PickingMode.Ignore };
                dot.style.position = Position.Absolute;
                dot.style.width = 6;
                dot.style.height = 6;
                dot.style.top = 2;
                UiTheme.SetRadius(dot, 3f);
                _progressDots.Add(dot);
                _progressStrip.Add(dot);
            }
            float trackLength = _raceModel.TrackLengthMeters;
            for (int dotIndex = 0; dotIndex < _progressDots.Count; dotIndex++)
            {
                if (dotIndex >= dotCount)
                {
                    _progressDots[dotIndex].style.display = DisplayStyle.None;
                    continue;
                }
                RacerState racer = _raceModel.Racers[dotIndex];
                float percent = racer.Status == RacerStatus.Finished
                    ? 1f
                    : Mathf.Clamp01(racer.Progress / trackLength);
                VisualElement dot = _progressDots[dotIndex];
                dot.style.display = DisplayStyle.Flex;
                // 97% keeps the dot inside the strip's right edge.
                dot.style.left = new Length(percent * 97f, LengthUnit.Percent);
                dot.style.backgroundColor = racer.Status == RacerStatus.Dnf
                    ? new Color(0.4f, 0.4f, 0.4f, 0.5f)
                    : racer.Tint;
            }
        }

        private static int CompareRacers(RacerState first, RacerState second)
        {
            bool firstFinished = first.Status == RacerStatus.Finished;
            bool secondFinished = second.Status == RacerStatus.Finished;
            if (firstFinished && secondFinished)
            {
                return first.Place.CompareTo(second.Place);
            }
            if (firstFinished != secondFinished)
            {
                return firstFinished ? -1 : 1;
            }
            bool firstDnf = first.Status == RacerStatus.Dnf;
            bool secondDnf = second.Status == RacerStatus.Dnf;
            if (firstDnf != secondDnf)
            {
                return firstDnf ? 1 : -1;
            }
            return second.Progress.CompareTo(first.Progress);
        }
    }
}
