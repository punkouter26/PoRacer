using PoRacer.Models;
using PoRacer.Presentation;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// The telemetry card: a bottom sheet for the racer the viewer picked (tap it, swipe
    /// to it, or cycle with the keys). It shows what the body is doing — ground speed,
    /// how upright it is, joint effort, and mechanical cost of transport — with a 15 s
    /// graph, and what the brain is doing: one bar per policy output and a calm-to-twitchy
    /// meter of how much those outputs jump between decisions.
    ///
    /// Shares the HUD's UIDocument and follows its rules: hierarchy built in C#, refreshed
    /// on a schedule from the Models, strings rebuilt only when the shown value changes.
    /// Closing the card hands the camera back to the director.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class TelemetryCardView : MonoBehaviour
    {
        private const long BARS_REFRESH_MS = 100;
        private const int NUMBERS_EVERY_N_REFRESHES = 3;
        private const float GRAPH_HEIGHT = 56f;
        private const float BARS_HEIGHT = 40f;
        private const float METER_HEIGHT = 6f;
        // Mean change per decision at which the meter reads fully twitchy. A walking
        // gait sits well under 0.1; 0.5 means outputs swinging a quarter of their range
        // every decision.
        private const float JITTER_FULL_SCALE = 0.5f;
        private const float SIDE_MARGIN_PERCENT = 3f;
        // Clear of the DBG button and version stamp that share the bottom edge.
        private const float CARD_BOTTOM = 64f;

        private RaceTelemetryModel _telemetryModel;
        private RaceModel _raceModel;
        private EloModel _eloModel;
        private RaceConfigModel _configModel;
        private Systems_CameraDirector _cameraDirector;

        private VisualElement _card;
        private VisualElement _swatch;
        private Label _nameLabel;
        private Label _teamLabel;
        private Label _eloLabel;
        private Label _energyLabel;
        private Label _speedValue;
        private Label _uprightValue;
        private Label _effortValue;
        private Label _costValue;
        private Label _brainHeader;
        private TelemetryGraph _graph;
        private PolicyBarsGraph _bars;
        private VisualElement _jitterFill;

        private string _shownRacerId;
        private bool _visible;
        private int _refreshCount;
        // Last value each label was built from, at the precision it shows.
        private int _shownSpeed = int.MinValue;
        private int _shownUpright = int.MinValue;
        private int _shownEffort = int.MinValue;
        private int _shownCost = int.MinValue;
        private int _shownEnergy = int.MinValue;
        private int _shownActionCount = -1;

        [Inject]
        public void Construct(RaceTelemetryModel telemetryModel, RaceModel raceModel, EloModel eloModel,
            RaceConfigModel configModel, Systems_CameraDirector cameraDirector)
        {
            _telemetryModel = telemetryModel;
            _raceModel = raceModel;
            _eloModel = eloModel;
            _configModel = configModel;
            _cameraDirector = cameraDirector;
        }

        private void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);
            BuildCard(safeRoot);
            root.schedule.Execute(Refresh).Every(BARS_REFRESH_MS);
        }

        private void BuildCard(VisualElement safeRoot)
        {
            _card = new VisualElement();
            _card.style.position = Position.Absolute;
            _card.style.left = new Length(SIDE_MARGIN_PERCENT, LengthUnit.Percent);
            _card.style.right = new Length(SIDE_MARGIN_PERCENT, LengthUnit.Percent);
            _card.style.bottom = CARD_BOTTOM;
            UiTheme.StyleGlassPanel(_card);
            // It carries a button, so it takes its own clicks; InputView reads that as
            // "not a camera tap".
            _card.pickingMode = PickingMode.Position;
            _card.style.display = DisplayStyle.None;
            safeRoot.Add(_card);

            var header = Row();
            _swatch = UiTheme.MakeSwatch(UiTheme.TextDim, UiTheme.SPACE_MD);
            _swatch.style.marginRight = UiTheme.SPACE_SM;
            header.Add(_swatch);
            _nameLabel = MakeLabel(UiTheme.FONT_MD, UiTheme.Text, bold: true);
            _nameLabel.style.flexGrow = 1f;
            _nameLabel.style.flexShrink = 1f;
            header.Add(_nameLabel);
            var closeButton = new Button(OnClose) { text = "X" };
            closeButton.style.width = UiTheme.CONTROL_SM;
            closeButton.style.height = UiTheme.CONTROL_SM;
            closeButton.style.fontSize = UiTheme.FONT_SM;
            UiTheme.SetMargin(closeButton, 0f, 0f);
            UiTheme.StyleButton(closeButton);
            UiTheme.AddHover(closeButton);
            header.Add(closeButton);
            _card.Add(header);

            // Who trained it, its rating and what it has spent, on a line of their own so a
            // long racer name never runs under them.
            var subheader = Row();
            subheader.style.marginTop = UiTheme.SPACE_XXS;
            _teamLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.Text, bold: true);
            UiTheme.StyleChip(_teamLabel);
            _teamLabel.style.marginRight = UiTheme.SPACE_SM;
            subheader.Add(_teamLabel);
            _eloLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false);
            _eloLabel.style.marginRight = UiTheme.SPACE_SM;
            subheader.Add(_eloLabel);
            _energyLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false);
            subheader.Add(_energyLabel);
            _card.Add(subheader);

            var stats = Row();
            stats.style.marginTop = UiTheme.SPACE_SM;
            stats.Add(MakeStat("M/S", out _speedValue));
            stats.Add(MakeStat("UPRIGHT", out _uprightValue));
            stats.Add(MakeStat("EFFORT", out _effortValue));
            stats.Add(MakeStat("COST/M", out _costValue));
            _card.Add(stats);

            _graph = new TelemetryGraph();
            _graph.style.height = GRAPH_HEIGHT;
            _graph.style.marginTop = UiTheme.SPACE_SM;
            _card.Add(_graph);
            var legend = Row();
            legend.style.justifyContent = Justify.Center;
            legend.Add(MakeLegend("speed", TelemetryGraph.SpeedColor));
            legend.Add(MakeLegend("upright", TelemetryGraph.UprightColor));
            legend.Add(MakeLegend("effort", TelemetryGraph.EffortColor));
            _card.Add(legend);

            _card.Add(UiTheme.MakeDivider());
            _brainHeader = UiTheme.MakeSectionHeader("BRAIN");
            _brainHeader.pickingMode = PickingMode.Ignore;
            _card.Add(_brainHeader);
            _bars = new PolicyBarsGraph();
            _bars.style.height = BARS_HEIGHT;
            _card.Add(_bars);

            var jitterRow = Row();
            jitterRow.style.marginTop = UiTheme.SPACE_XS;
            jitterRow.Add(MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: true, "CALM"));
            var track = new VisualElement { pickingMode = PickingMode.Ignore };
            track.style.flexGrow = 1f;
            track.style.height = METER_HEIGHT;
            track.style.backgroundColor = UiTheme.TrackBg;
            UiTheme.SetRadius(track, METER_HEIGHT * 0.5f);
            UiTheme.SetMargin(track, 0f, UiTheme.SPACE_SM);
            _jitterFill = new VisualElement { pickingMode = PickingMode.Ignore };
            _jitterFill.style.height = METER_HEIGHT;
            _jitterFill.style.width = new Length(0f, LengthUnit.Percent);
            _jitterFill.style.backgroundColor = UiTheme.AccentSoft;
            UiTheme.SetRadius(_jitterFill, METER_HEIGHT * 0.5f);
            track.Add(_jitterFill);
            jitterRow.Add(track);
            jitterRow.Add(MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: true, "TWITCHY"));
            _card.Add(jitterRow);
        }

        private void Refresh()
        {
            if (_telemetryModel == null)
            {
                return;
            }
            RacerTelemetry telemetry = _telemetryModel.Find(_telemetryModel.FocusedRacerId);
            RacerState racer = telemetry != null ? _raceModel.FindRacer(telemetry.RacerId) : null;
            bool show = racer != null && _raceModel.RaceActive
                && (_configModel == null || !_configModel.MenuVisible);
            if (show != _visible)
            {
                _visible = show;
                _card.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
                if (!show)
                {
                    _shownRacerId = null;
                }
            }
            if (!show)
            {
                return;
            }
            if (_shownRacerId != telemetry.RacerId)
            {
                Bind(telemetry, racer);
                UiTheme.PlayEnter(_card, 0, UiTheme.PANEL_SLIDE_PX);
            }

            _bars.MarkDirtyRepaint();
            _graph.RefreshIfChanged();
            float jitter = Mathf.Clamp01(telemetry.Jitter / JITTER_FULL_SCALE);
            _jitterFill.style.width = new Length(jitter * 100f, LengthUnit.Percent);

            _refreshCount++;
            if (_refreshCount % NUMBERS_EVERY_N_REFRESHES == 0)
            {
                RefreshNumbers(telemetry);
            }
        }

        private void Bind(RacerTelemetry telemetry, RacerState racer)
        {
            _shownRacerId = telemetry.RacerId;
            _swatch.style.backgroundColor = racer.Tint;
            _nameLabel.text = racer.DisplayName;
            _teamLabel.text = TrainerTeams.DisplayName(racer.TrainedBy);
            _teamLabel.style.color = TrainerTeams.ColorOf(racer.TrainedBy);
            _eloLabel.text = $"ELO {_eloModel.GetRating(racer.CreatureId):0}";
            _graph.Bind(telemetry);
            _bars.Bind(telemetry);
            _shownSpeed = int.MinValue;
            _shownUpright = int.MinValue;
            _shownEffort = int.MinValue;
            _shownCost = int.MinValue;
            _shownEnergy = int.MinValue;
            _shownActionCount = -1;
            RefreshNumbers(telemetry);
        }

        private void RefreshNumbers(RacerTelemetry telemetry)
        {
            int speed = Mathf.RoundToInt(telemetry.SpeedMps * 100f);
            if (speed != _shownSpeed)
            {
                _shownSpeed = speed;
                _speedValue.text = (speed / 100f).ToString("0.00");
            }
            int upright = telemetry.IsDown ? -1 : Mathf.RoundToInt(Mathf.Clamp01(telemetry.Upright) * 100f);
            if (upright != _shownUpright)
            {
                _shownUpright = upright;
                _uprightValue.text = upright < 0 ? "DOWN" : upright + "%";
            }
            int effort = telemetry.EffortFraction < 0f ? -1 : Mathf.RoundToInt(telemetry.EffortFraction * 100f);
            if (effort != _shownEffort)
            {
                _shownEffort = effort;
                _effortValue.text = effort < 0 ? "—" : effort + "%";
            }
            float costOfTransport = telemetry.CostOfTransport;
            int cost = costOfTransport < 0f ? -1 : Mathf.RoundToInt(costOfTransport * 100f);
            if (cost != _shownCost)
            {
                _shownCost = cost;
                _costValue.text = cost < 0 ? "—" : (cost / 100f).ToString("0.00");
            }
            int energy = Mathf.RoundToInt(telemetry.EnergyJoules / 100f);
            if (energy != _shownEnergy)
            {
                _shownEnergy = energy;
                _energyLabel.text = telemetry.HasPower ? $"{energy / 10f:0.0} kJ spent" : string.Empty;
            }
            if (telemetry.ActionCount != _shownActionCount)
            {
                _shownActionCount = telemetry.ActionCount;
                _brainHeader.text = telemetry.ActionCount > 0
                    ? $"BRAIN  {telemetry.ActionCount} OUTPUTS"
                    : "BRAIN  NOT REPORTED";
            }
        }

        private void OnClose()
        {
            _cameraDirector.ResumeDirecting();
        }

        private static VisualElement Row()
        {
            var row = new VisualElement { pickingMode = PickingMode.Ignore };
            row.style.flexDirection = FlexDirection.Row;
            row.style.alignItems = Align.Center;
            return row;
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

        private static VisualElement MakeStat(string caption, out Label value)
        {
            var tile = new VisualElement { pickingMode = PickingMode.Ignore };
            tile.style.flexGrow = 1f;
            tile.style.flexBasis = 0f;
            tile.style.alignItems = Align.Center;
            value = MakeLabel(UiTheme.FONT_LG, UiTheme.Text, bold: true, "—");
            tile.Add(value);
            tile.Add(MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: true, caption));
            return tile;
        }

        private static VisualElement MakeLegend(string text, Color color)
        {
            var item = Row();
            UiTheme.SetMargin(item, 0f, UiTheme.SPACE_SM);
            VisualElement swatch = UiTheme.MakeSwatch(color, UiTheme.SPACE_SM);
            swatch.style.marginRight = UiTheme.SPACE_XS;
            item.Add(swatch);
            item.Add(MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false, text));
            return item;
        }
    }
}
