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

        private RaceModel _raceModel;
        private EloModel _eloModel;
        private RaceConfigModel _configModel;
        private Systems_Spawn _spawn;
        private VisualElement _hudRoot;
        private Label _statusLabel;
        private Label _bannerLabel;
        private VisualElement _leaderboard;
        private readonly System.Collections.Generic.List<Label> _rowLabels = new();
        private readonly System.Collections.Generic.List<RacerState> _sortBuffer = new();

        [Inject]
        public void Construct(RaceModel raceModel, EloModel eloModel, RaceConfigModel configModel, Systems_Spawn spawn)
        {
            _raceModel = raceModel;
            _eloModel = eloModel;
            _configModel = configModel;
            _spawn = spawn;
        }

        private void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            root.pickingMode = PickingMode.Ignore;
            _hudRoot = root;

            var versionLabel = new Label($"v{Application.version}")
            {
                pickingMode = PickingMode.Ignore
            };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.top = 4;
            versionLabel.style.left = 6;
            versionLabel.style.color = new Color(1f, 1f, 1f, 0.7f);
            versionLabel.style.fontSize = 12;
            root.Add(versionLabel);

            var panel = new VisualElement();
            panel.style.position = Position.Absolute;
            panel.style.top = 24;
            panel.style.left = 6;
            panel.style.right = 6;
            UiTheme.StylePanel(panel);
            panel.pickingMode = PickingMode.Ignore;
            root.Add(panel);

            _statusLabel = new Label { pickingMode = PickingMode.Ignore };
            _statusLabel.style.color = UiTheme.Text;
            _statusLabel.style.fontSize = 14;
            _statusLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            panel.Add(_statusLabel);

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
            root.Add(_bannerLabel);

            var menuButton = new Button(() => _spawn.RequestMenu()) { text = "MENU" };
            menuButton.style.position = Position.Absolute;
            menuButton.style.top = 4;
            menuButton.style.right = 6;
            menuButton.style.width = 64;
            menuButton.style.height = 26;
            UiTheme.StyleButton(menuButton);
            root.Add(menuButton);

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

            string ticker = leader != null ? $"  |  Leader: {leader.DisplayName} {leader.Progress:0.0}m" : string.Empty;
            _statusLabel.text = _raceModel.RaceActive
                ? $"Race {_raceModel.RaceNumber} [{_raceModel.TrackName}]  {_raceModel.ElapsedSeconds:0.0}s  ({_raceModel.Racers.Count} racers){ticker}"
                : $"Race {_raceModel.RaceNumber} finished — next starting soon";

            if (winner != null)
            {
                _bannerLabel.text = $"WINNER  {winner.DisplayName}  {winner.FinishTime:0.0}s";
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
            while (_rowLabels.Count < visibleRows)
            {
                var row = new Label { pickingMode = PickingMode.Ignore };
                row.style.color = UiTheme.Text;
                row.style.fontSize = 12;
                row.style.marginTop = 2;
                row.style.paddingLeft = 4;
                row.style.paddingTop = 1;
                row.style.paddingBottom = 1;
                UiTheme.SetRadius(row, 3f);
                _rowLabels.Add(row);
                _leaderboard.Add(row);
            }
            for (int rowIndex = 0; rowIndex < _rowLabels.Count; rowIndex++)
            {
                if (rowIndex >= visibleRows)
                {
                    _rowLabels[rowIndex].text = string.Empty;
                    _rowLabels[rowIndex].style.backgroundColor = StyleKeyword.Null;
                    continue;
                }
                RacerState racer = _sortBuffer[rowIndex];
                string status = racer.Status switch
                {
                    RacerStatus.Finished => $"#{racer.Place} {racer.FinishTime:0.0}s",
                    RacerStatus.Dnf => "DNF",
                    _ => $"{racer.Progress:0.0}m"
                };
                float rating = _eloModel.GetRating(racer.CreatureId);
                Label rowLabel = _rowLabels[rowIndex];
                rowLabel.text = $"{rowIndex + 1}. {racer.DisplayName}  {status}  ELO {rating:0}";
                rowLabel.style.backgroundColor = racer.Place switch
                {
                    1 => new Color(UiTheme.Gold.r, UiTheme.Gold.g, UiTheme.Gold.b, 0.18f),
                    2 => new Color(UiTheme.Silver.r, UiTheme.Silver.g, UiTheme.Silver.b, 0.14f),
                    3 => new Color(UiTheme.Bronze.r, UiTheme.Bronze.g, UiTheme.Bronze.b, 0.14f),
                    _ => StyleKeyword.Null
                };
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
