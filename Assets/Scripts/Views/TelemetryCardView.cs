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
        // Graph and brain column side by side, one shared height: about half the card
        // the stacked layout needed (graph, legend, divider, header, bars, meter).
        private const float GRAPH_HEIGHT = 64f;
        private const float BARS_HEIGHT = 30f;
        private const float METER_HEIGHT = 6f;
        private const float SIDE_MARGIN_PERCENT = 3f;

        private RaceTelemetryModel _telemetryModel;
        private RaceModel _raceModel;
        private RaceConfigModel _configModel;
        private Systems_CameraDirector _cameraDirector;

        private VisualElement _card;
        private VisualElement _swatch;
        private Label _nameLabel;
        private Label _teamLabel;
        private Label _energyValue;
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
        public void Construct(RaceTelemetryModel telemetryModel, RaceModel raceModel,
            RaceConfigModel configModel, Systems_CameraDirector cameraDirector)
        {
            _telemetryModel = telemetryModel;
            _raceModel = raceModel;
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

        /// <summary>
        /// Three rows: who (name, trainer, close), the numbers as one wrapping line, then
        /// the 15 s graph beside the brain. The numbers are tinted in their graph line's
        /// colour, so they double as its legend and the legend row is gone.
        /// </summary>
        private void BuildCard(VisualElement safeRoot)
        {
            _card = new VisualElement();
            _card.style.position = Position.Absolute;
            _card.style.left = new Length(SIDE_MARGIN_PERCENT, LengthUnit.Percent);
            _card.style.right = new Length(SIDE_MARGIN_PERCENT, LengthUnit.Percent);
            // Clear of the DBG button and version stamp that share the bottom band.
            _card.style.bottom = UiTheme.BottomBand;
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
            _nameLabel.style.minWidth = 0f;
            _nameLabel.style.overflow = Overflow.Hidden;
            _nameLabel.style.textOverflow = TextOverflow.Ellipsis;
            header.Add(_nameLabel);
            _teamLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.Text, bold: true);
            UiTheme.StyleChip(_teamLabel);
            _teamLabel.style.flexShrink = 0f;
            _teamLabel.style.marginRight = UiTheme.SPACE_SM;
            header.Add(_teamLabel);
            var closeButton = new Button(OnClose) { text = "\u00D7" };
            closeButton.style.width = UiTheme.CONTROL_SM;
            closeButton.style.height = UiTheme.CONTROL_SM;
            closeButton.style.fontSize = UiTheme.FONT_LG;
            closeButton.style.flexShrink = 0f;
            UiTheme.SetMargin(closeButton, 0f, 0f);
            UiTheme.StyleButton(closeButton);
            UiTheme.AddHover(closeButton);
            header.Add(closeButton);
            _card.Add(header);

            var stats = Row();
            stats.style.flexWrap = Wrap.Wrap;
            stats.style.marginTop = UiTheme.SPACE_XXS;
            stats.Add(MakeStat("m/s", TelemetryGraph.SpeedColor, out _speedValue));
            stats.Add(MakeStat("upright", TelemetryGraph.UprightColor, out _uprightValue));
            stats.Add(MakeStat("effort", TelemetryGraph.EffortColor, out _effortValue));
            stats.Add(MakeStat("cost/m", UiTheme.Text, out _costValue));
            stats.Add(MakeStat("kJ", UiTheme.Text, out _energyValue));
            _card.Add(stats);

            var body = Row();
            body.style.alignItems = Align.Stretch;
            body.style.marginTop = UiTheme.SPACE_XS;
            _card.Add(body);

            _graph = new TelemetryGraph();
            _graph.style.height = GRAPH_HEIGHT;
            _graph.style.flexGrow = 3f;
            _graph.style.flexBasis = 0f;
            _graph.style.marginRight = UiTheme.SPACE_SM;
            body.Add(_graph);

            var brain = new VisualElement { pickingMode = PickingMode.Ignore };
            brain.style.flexGrow = 2f;
            brain.style.flexBasis = 0f;
            brain.style.justifyContent = Justify.SpaceBetween;
            body.Add(brain);
            _brainHeader = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: true);
            _brainHeader.style.letterSpacing = 1f;
            brain.Add(_brainHeader);
            _bars = new PolicyBarsGraph();
            _bars.style.height = BARS_HEIGHT;
            brain.Add(_bars);

            // Calm on the left, twitchy on the right; the header names the scale.
            var track = new VisualElement { pickingMode = PickingMode.Ignore };
            track.style.height = METER_HEIGHT;
            track.style.backgroundColor = UiTheme.TrackBg;
            UiTheme.SetRadius(track, METER_HEIGHT * 0.5f);
            _jitterFill = new VisualElement { pickingMode = PickingMode.Ignore };
            _jitterFill.style.height = METER_HEIGHT;
            _jitterFill.style.width = new Length(0f, LengthUnit.Percent);
            _jitterFill.style.backgroundColor = UiTheme.AccentSoft;
            UiTheme.SetRadius(_jitterFill, METER_HEIGHT * 0.5f);
            track.Add(_jitterFill);
            brain.Add(track);
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
            _jitterFill.style.width = new Length(telemetry.Twitchiness * 100f, LengthUnit.Percent);

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
                _energyValue.text = telemetry.HasPower ? $"{energy / 10f:0.0}" : "—";
            }
            if (telemetry.ActionCount != _shownActionCount)
            {
                _shownActionCount = telemetry.ActionCount;
                _brainHeader.text = telemetry.ActionCount > 0
                    ? $"BRAIN · {telemetry.ActionCount} · TWITCH"
                    : "BRAIN · NOT REPORTED";
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

        /// <summary>One inline reading: the value in its graph colour, then its unit.</summary>
        private static VisualElement MakeStat(string unit, Color color, out Label value)
        {
            var stat = Row();
            stat.style.alignItems = Align.FlexEnd;
            stat.style.marginRight = UiTheme.SPACE_SM;
            value = MakeLabel(UiTheme.FONT_SM, color, bold: true, "—");
            stat.Add(value);
            Label unitLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false, unit);
            unitLabel.style.marginLeft = UiTheme.SPACE_XXS;
            stat.Add(unitLabel);
            return stat;
        }
    }
}
