using PoRacer.Models;
using PoRacer.Presentation;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// The Sim Wars scoreboard on the HUD's document. During the countdown a slim pill
    /// shows the MuJoCo vs Isaac Lab head-to-head; between races the league joins the
    /// results panel, under the podium and above its buttons: the head-to-head, what each
    /// team scored in the race just run, and the season table.
    ///
    /// It lives inside the results panel rather than beside it because a second card
    /// anchored on its own cannot know how tall the podium grew, and on a short screen
    /// the two overlapped the RACE AGAIN button. The block carries its own caveat,
    /// because the numbers invite one: the teams race different bodies, so this is a
    /// league, not the comparison. Text is rebuilt only when the league changes.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class SimWarsView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const float PILL_TOP_PERCENT = 22f;
        private const float TEAM_SWATCH = 8f;
        private const float STAT_COLUMN_WIDTH = 60f;
        // The results panel ends with a divider and the button row; the league goes above both.
        private const int RESULTS_TRAILING_CHILDREN = 2;

        private static readonly string MuJoCoHex = ColorUtility.ToHtmlStringRGB(TrainerTeams.MuJoCo);
        private static readonly string IsaacLabHex = ColorUtility.ToHtmlStringRGB(TrainerTeams.IsaacLab);

        private SimWarsModel _league;
        private RaceModel _raceModel;
        private RaceConfigModel _configModel;

        private VisualElement _root;
        private VisualElement _pill;
        private Label _pillLabel;
        private VisualElement _block;
        private Label _headToHeadLabel;
        private Label _headToHeadCaption;
        private readonly VisualElement[] _teamRows = new VisualElement[TrainerTeams.LeagueOrder.Length];
        private readonly Label[] _pointsLabels = new Label[TrainerTeams.LeagueOrder.Length];
        private readonly Label[] _winsLabels = new Label[TrainerTeams.LeagueOrder.Length];
        private readonly Label[] _racesLabels = new Label[TrainerTeams.LeagueOrder.Length];
        private readonly Label[] _lastRaceLabels = new Label[TrainerTeams.LeagueOrder.Length];
        private int _shownVersion = -1;
        private bool _pillVisible;
        private bool _blockVisible;

        [Inject]
        public void Construct(SimWarsModel league, RaceModel raceModel, RaceConfigModel configModel)
        {
            _league = league;
            _raceModel = raceModel;
            _configModel = configModel;
        }

        private void Start()
        {
            _root = GetComponent<UIDocument>().rootVisualElement;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(_root);
            BuildPill(safeRoot);
            BuildBlock();
            _root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
        }

        private void BuildPill(VisualElement safeRoot)
        {
            _pill = new VisualElement { pickingMode = PickingMode.Ignore };
            _pill.style.position = Position.Absolute;
            _pill.style.top = new Length(PILL_TOP_PERCENT, LengthUnit.Percent);
            _pill.style.left = 0f;
            _pill.style.right = 0f;
            _pill.style.alignItems = Align.Center;
            _pill.style.display = DisplayStyle.None;
            safeRoot.Add(_pill);

            _pillLabel = MakeLabel(UiTheme.FONT_SM, UiTheme.Text, bold: true);
            _pillLabel.enableRichText = true;
            UiTheme.StyleChip(_pillLabel);
            _pill.Add(_pillLabel);
        }

        private void BuildBlock()
        {
            _block = new VisualElement { pickingMode = PickingMode.Ignore };
            _block.style.display = DisplayStyle.None;
            _block.Add(UiTheme.MakeDivider());

            Label title = UiTheme.MakeSectionHeader("SIM WARS  TRAINER LEAGUE");
            title.pickingMode = PickingMode.Ignore;
            title.style.unityTextAlign = TextAnchor.MiddleCenter;
            _block.Add(title);

            _headToHeadLabel = MakeLabel(UiTheme.FONT_MD, UiTheme.Text, bold: true);
            _headToHeadLabel.enableRichText = true;
            _headToHeadLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _block.Add(_headToHeadLabel);

            _headToHeadCaption = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false);
            _headToHeadCaption.style.unityTextAlign = TextAnchor.MiddleCenter;
            _headToHeadCaption.style.whiteSpace = WhiteSpace.Normal;
            _block.Add(_headToHeadCaption);

            VisualElement header = TableRow();
            header.style.marginTop = UiTheme.SPACE_XS;
            header.Add(Spacer());
            header.Add(StatCell("PTS", UiTheme.TextDim, bold: true));
            header.Add(StatCell("WON", UiTheme.TextDim, bold: true));
            header.Add(StatCell("RAN", UiTheme.TextDim, bold: true));
            header.Add(StatCell(string.Empty, UiTheme.TextDim, bold: true));
            _block.Add(header);

            for (int rowIndex = 0; rowIndex < TrainerTeams.LeagueOrder.Length; rowIndex++)
            {
                TrainingSource team = TrainerTeams.LeagueOrder[rowIndex];
                VisualElement row = TableRow();
                row.style.marginTop = UiTheme.SPACE_XXS;
                VisualElement swatch = UiTheme.MakeSwatch(TrainerTeams.ColorOf(team), TEAM_SWATCH);
                swatch.style.marginRight = UiTheme.SPACE_SM;
                row.Add(swatch);
                Label name = MakeLabel(UiTheme.FONT_SM, UiTheme.Text, bold: false, TrainerTeams.DisplayName(team));
                name.style.flexGrow = 1f;
                row.Add(name);
                _pointsLabels[rowIndex] = StatCell("0", UiTheme.Text, bold: true);
                row.Add(_pointsLabels[rowIndex]);
                _winsLabels[rowIndex] = StatCell("0", UiTheme.Text, bold: false);
                row.Add(_winsLabels[rowIndex]);
                _racesLabels[rowIndex] = StatCell("0", UiTheme.TextDim, bold: false);
                row.Add(_racesLabels[rowIndex]);
                _lastRaceLabels[rowIndex] = StatCell(string.Empty, UiTheme.Gold, bold: true);
                row.Add(_lastRaceLabels[rowIndex]);
                _teamRows[rowIndex] = row;
                _block.Add(row);
            }

            Label footnote = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false,
                "Different bodies: for fun. The fair test is the Walking Standard.");
            footnote.style.whiteSpace = WhiteSpace.Normal;
            footnote.style.marginTop = UiTheme.SPACE_XS;
            footnote.style.unityTextAlign = TextAnchor.MiddleCenter;
            _block.Add(footnote);
        }

        private void Refresh()
        {
            if (_league == null)
            {
                return;
            }
            AttachToResults();
            bool menu = _configModel != null && _configModel.MenuVisible;
            bool anyHeadToHead = _league.MuJoCoAhead + _league.IsaacLabAhead + _league.HeadToHeadDraws > 0;
            bool showPill = !menu && _raceModel.CountdownValue > 0 && anyHeadToHead;
            bool showBlock = _league.RacesScored > 0;

            if (_league.Version != _shownVersion)
            {
                _shownVersion = _league.Version;
                Rebuild();
            }
            if (showPill != _pillVisible)
            {
                _pillVisible = showPill;
                _pill.style.display = showPill ? DisplayStyle.Flex : DisplayStyle.None;
            }
            if (showBlock != _blockVisible)
            {
                _blockVisible = showBlock;
                _block.style.display = showBlock ? DisplayStyle.Flex : DisplayStyle.None;
            }
        }

        /// <summary>
        /// Moves the league block into RaceHudView's results panel once that panel exists.
        /// The two views start in no guaranteed order, so this is retried on each refresh
        /// rather than done once in Start; after it lands it is a single parent check.
        /// </summary>
        private void AttachToResults()
        {
            if (_block.parent != null)
            {
                return;
            }
            VisualElement results = _root.Q(UiTheme.RESULTS_PANEL);
            if (results == null)
            {
                return;
            }
            int index = Mathf.Max(0, results.childCount - RESULTS_TRAILING_CHILDREN);
            results.Insert(index, _block);
        }

        private void Rebuild()
        {
            string headToHead = $"<color=#{MuJoCoHex}>MuJoCo {_league.MuJoCoAhead}</color>"
                + $"  :  <color=#{IsaacLabHex}>{_league.IsaacLabAhead} Isaac Lab</color>";
            _headToHeadLabel.text = headToHead;
            _pillLabel.text = "SIM WARS   " + headToHead;
            _headToHeadCaption.text = _league.HeadToHeadDraws > 0
                ? $"best finisher, races both ran; {_league.HeadToHeadDraws} with neither home"
                : "best finisher, races both ran";

            for (int rowIndex = 0; rowIndex < TrainerTeams.LeagueOrder.Length; rowIndex++)
            {
                SimWarsTeamStats stats = _league.Team(TrainerTeams.LeagueOrder[rowIndex]);
                bool entered = stats.Races > 0;
                _teamRows[rowIndex].style.display = entered ? DisplayStyle.Flex : DisplayStyle.None;
                if (!entered)
                {
                    continue;
                }
                _pointsLabels[rowIndex].text = stats.Points.ToString();
                _winsLabels[rowIndex].text = stats.Wins.ToString();
                _racesLabels[rowIndex].text = stats.Races.ToString();
                _lastRaceLabels[rowIndex].text = stats.LastRacePoints > 0 ? "+" + stats.LastRacePoints : string.Empty;
            }
        }

        private static VisualElement TableRow()
        {
            var row = new VisualElement { pickingMode = PickingMode.Ignore };
            row.style.flexDirection = FlexDirection.Row;
            row.style.alignItems = Align.Center;
            return row;
        }

        private static VisualElement Spacer()
        {
            var spacer = new VisualElement { pickingMode = PickingMode.Ignore };
            spacer.style.flexGrow = 1f;
            return spacer;
        }

        private static Label StatCell(string text, Color color, bool bold)
        {
            Label cell = MakeLabel(UiTheme.FONT_XS, color, bold, text);
            cell.style.width = STAT_COLUMN_WIDTH;
            cell.style.flexShrink = 0f;
            cell.style.unityTextAlign = TextAnchor.MiddleRight;
            return cell;
        }

        private static Label MakeLabel(float fontSize, Color color, bool bold, string text = "")
        {
            var label = new Label(text) { pickingMode = PickingMode.Ignore };
            label.style.fontSize = fontSize;
            label.style.color = color;
            if (bold)
            {
                label.style.unityFontStyleAndWeight = FontStyle.Bold;
            }
            UiTheme.ApplyFont(label);
            return label;
        }
    }
}
