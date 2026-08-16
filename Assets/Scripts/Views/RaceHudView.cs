using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Runtime UI Toolkit HUD, hierarchy built entirely in C# (no .uxml/.uss).
    /// Deliberately minimal so the race itself owns the screen: version stamp
    /// top-left, MENU button top-right, top-3 chips under the stamp, a thin
    /// progress rail hugging the right edge, center banners (countdown / GO /
    /// winner), and the between-races podium. Refreshed on a schedule by reading
    /// the Models — no per-frame polling in Update, and no allocation in the
    /// refresh past the elements pooled during the first build.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class RaceHudView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const float GO_BANNER_SECONDS = 1.5f;
        private const float WINNER_BANNER_SECONDS = 3f;
        private const int PODIUM_ROWS = 3;
        private const int TOP_CHIP_COUNT = 3;
        // Rail dots are pooled: a 100+ racer field shows only the leading pack.
        private const int MAX_RAIL_DOTS = 32;
        private const float RAIL_WIDTH = 7f;
        private const float RAIL_DOT_SIZE = 9f;
        // Percent of the rail a dot's top may reach, leaving room for its height.
        private const float RAIL_SPAN_PERCENT = 96f;
        private const float CHIP_SWATCH_SIZE = 8f;

        private static readonly Color[] MedalColors = { UiTheme.Gold, UiTheme.Silver, UiTheme.Bronze };

        private RaceModel _raceModel;
        private EloModel _eloModel;
        private RaceConfigModel _configModel;
        private Systems_Spawn _spawn;
        private VisualElement _hudRoot;
        private Label _bannerLabel;
        private bool _wasRaceActive;
        private float _goBannerUntil;
        private float _winnerBannerUntil;
        private VisualElement _podiumPanel;
        private readonly System.Collections.Generic.List<Label> _podiumLabels = new();
        private string _lastBannerText;

        // --- Right-edge progress rail ---
        private VisualElement _rail;
        private readonly VisualElement[] _railDots = new VisualElement[MAX_RAIL_DOTS];
        private readonly Color[] _railDotTints = new Color[MAX_RAIL_DOTS];
        private int _railDotsShown = -1;

        // --- Top-3 chips ---
        private VisualElement _chipRow;
        private readonly VisualElement[] _chips = new VisualElement[TOP_CHIP_COUNT];
        private readonly VisualElement[] _chipSwatches = new VisualElement[TOP_CHIP_COUNT];
        private readonly Label[] _chipNames = new Label[TOP_CHIP_COUNT];
        private readonly string[] _chipSourceNames = new string[TOP_CHIP_COUNT];
        private readonly Color[] _chipTints = new Color[TOP_CHIP_COUNT];
        private int _chipsShown = -1;

        // Leader ordering scratch buffer, filled in place every refresh.
        private readonly RacerState[] _leaders = new RacerState[MAX_RAIL_DOTS];
        private bool _widgetsVisible;

        [Inject]
        public void Construct(
            RaceModel raceModel,
            EloModel eloModel,
            RaceConfigModel configModel,
            Systems_Spawn spawn)
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
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            var versionLabel = new Label($"v{Application.version}")
            {
                pickingMode = PickingMode.Ignore
            };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.top = UiTheme.SPACE_XS;
            versionLabel.style.left = UiTheme.SPACE_SM;
            versionLabel.style.color = UiTheme.TextDim;
            versionLabel.style.fontSize = UiTheme.FONT_SM;
            safeRoot.Add(versionLabel);

            BuildTopChips(safeRoot);
            BuildProgressRail(safeRoot);

            _bannerLabel = new Label { pickingMode = PickingMode.Ignore };
            _bannerLabel.style.position = Position.Absolute;
            _bannerLabel.style.top = new Length(40f, LengthUnit.Percent);
            _bannerLabel.style.left = 0;
            _bannerLabel.style.right = 0;
            _bannerLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _bannerLabel.style.fontSize = UiTheme.FONT_TITLE;
            _bannerLabel.style.color = UiTheme.Gold;
            _bannerLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _bannerLabel.style.display = DisplayStyle.None;
            safeRoot.Add(_bannerLabel);

            // Podium: shown in the pause between races (top 3 with medal tints).
            _podiumPanel = new VisualElement { pickingMode = PickingMode.Ignore };
            _podiumPanel.style.position = Position.Absolute;
            _podiumPanel.style.top = new Length(38f, LengthUnit.Percent);
            _podiumPanel.style.left = new Length(12f, LengthUnit.Percent);
            _podiumPanel.style.right = new Length(12f, LengthUnit.Percent);
            UiTheme.StylePanel(_podiumPanel);
            _podiumPanel.style.display = DisplayStyle.None;
            var podiumTitle = new Label("RESULTS") { pickingMode = PickingMode.Ignore };
            podiumTitle.style.color = UiTheme.Gold;
            podiumTitle.style.fontSize = UiTheme.FONT_LG;
            podiumTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
            podiumTitle.style.unityTextAlign = TextAnchor.MiddleCenter;
            podiumTitle.style.marginBottom = UiTheme.SPACE_SM;
            _podiumPanel.Add(podiumTitle);
            for (int podiumIndex = 0; podiumIndex < PODIUM_ROWS; podiumIndex++)
            {
                var row = new VisualElement { pickingMode = PickingMode.Ignore };
                row.style.flexDirection = FlexDirection.Row;
                row.style.alignItems = Align.Center;
                row.style.marginTop = UiTheme.SPACE_XS;
                VisualElement medal = UiTheme.MakeSwatch(MedalColors[podiumIndex], UiTheme.SPACE_MD);
                medal.style.marginRight = UiTheme.SPACE_SM;
                row.Add(medal);
                var label = new Label { pickingMode = PickingMode.Ignore };
                label.style.color = UiTheme.Text;
                label.style.fontSize = UiTheme.FONT_MD;
                row.Add(label);
                _podiumLabels.Add(label);
                _podiumPanel.Add(row);
            }
            safeRoot.Add(_podiumPanel);

            var menuButton = new Button(() => _spawn.RequestMenu()) { text = "MENU" };
            menuButton.style.position = Position.Absolute;
            menuButton.style.top = UiTheme.SPACE_XS;
            menuButton.style.right = UiTheme.SPACE_SM;
            menuButton.style.width = 64;
            menuButton.style.height = UiTheme.CONTROL_SM;
            menuButton.style.fontSize = UiTheme.FONT_SM;
            UiTheme.StyleButton(menuButton);
            UiTheme.AddHover(menuButton);
            safeRoot.Add(menuButton);

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
        }

        /// <summary>
        /// Up to three tiny pills under the version stamp: place, racer tint and
        /// the first word of the racer's name. Elements are built once and only
        /// their text/tint change, and only when the occupant changes.
        /// </summary>
        private void BuildTopChips(VisualElement safeRoot)
        {
            _chipRow = new VisualElement { pickingMode = PickingMode.Ignore };
            _chipRow.style.position = Position.Absolute;
            _chipRow.style.top = UiTheme.SPACE_LG + UiTheme.SPACE_SM;
            _chipRow.style.left = 0;
            _chipRow.style.right = 0;
            _chipRow.style.flexDirection = FlexDirection.Row;
            _chipRow.style.justifyContent = Justify.Center;
            _chipRow.style.display = DisplayStyle.None;
            safeRoot.Add(_chipRow);

            for (int chipIndex = 0; chipIndex < TOP_CHIP_COUNT; chipIndex++)
            {
                var chip = new VisualElement { pickingMode = PickingMode.Ignore };
                chip.style.flexDirection = FlexDirection.Row;
                chip.style.alignItems = Align.Center;
                UiTheme.SetMargin(chip, 0f, UiTheme.SPACE_XS * 0.5f);
                UiTheme.StyleChip(chip);
                chip.style.display = DisplayStyle.None;

                var place = new Label((chipIndex + 1).ToString()) { pickingMode = PickingMode.Ignore };
                place.style.color = MedalColors[chipIndex];
                place.style.fontSize = UiTheme.FONT_XS;
                place.style.unityFontStyleAndWeight = FontStyle.Bold;
                place.style.marginRight = UiTheme.SPACE_XS;
                chip.Add(place);

                VisualElement swatch = UiTheme.MakeSwatch(UiTheme.TextDim, CHIP_SWATCH_SIZE);
                swatch.style.marginRight = UiTheme.SPACE_XS;
                chip.Add(swatch);

                var name = new Label { pickingMode = PickingMode.Ignore };
                name.style.color = UiTheme.Text;
                name.style.fontSize = UiTheme.FONT_XS;
                chip.Add(name);

                _chips[chipIndex] = chip;
                _chipSwatches[chipIndex] = swatch;
                _chipNames[chipIndex] = name;
                _chipRow.Add(chip);
            }
        }

        /// <summary>
        /// Thin vertical rail on the right edge: one pooled dot per leading racer,
        /// bottom (start line) to top (finish line).
        /// </summary>
        private void BuildProgressRail(VisualElement safeRoot)
        {
            _rail = new VisualElement { pickingMode = PickingMode.Ignore };
            _rail.style.position = Position.Absolute;
            _rail.style.right = UiTheme.SPACE_SM;
            _rail.style.top = new Length(15f, LengthUnit.Percent);
            _rail.style.bottom = new Length(15f, LengthUnit.Percent);
            _rail.style.width = RAIL_WIDTH;
            _rail.style.backgroundColor = UiTheme.TrackBg;
            UiTheme.SetRadius(_rail, RAIL_WIDTH * 0.5f);
            _rail.style.display = DisplayStyle.None;
            safeRoot.Add(_rail);

            // Gold cap marks the finish end of the rail.
            var finishCap = new VisualElement { pickingMode = PickingMode.Ignore };
            finishCap.style.position = Position.Absolute;
            finishCap.style.top = 0;
            finishCap.style.left = 0;
            finishCap.style.right = 0;
            finishCap.style.height = 2;
            finishCap.style.backgroundColor = UiTheme.Gold;
            _rail.Add(finishCap);

            for (int dotIndex = 0; dotIndex < MAX_RAIL_DOTS; dotIndex++)
            {
                var dot = new VisualElement { pickingMode = PickingMode.Ignore };
                dot.style.position = Position.Absolute;
                // Centers the dot over the narrower rail.
                dot.style.left = (RAIL_WIDTH - RAIL_DOT_SIZE) * 0.5f;
                dot.style.width = RAIL_DOT_SIZE;
                dot.style.height = RAIL_DOT_SIZE;
                dot.style.backgroundColor = UiTheme.TextDim;
                UiTheme.SetRadius(dot, RAIL_DOT_SIZE * 0.5f);
                dot.style.display = DisplayStyle.None;
                _railDots[dotIndex] = dot;
                _railDotTints[dotIndex] = UiTheme.TextDim;
                _rail.Add(dot);
            }
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

            RacerState winner = null;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                if (_raceModel.Racers[racerIndex].Place == 1)
                {
                    winner = _raceModel.Racers[racerIndex];
                    break;
                }
            }

            if (_raceModel.RaceActive && !_wasRaceActive)
            {
                _goBannerUntil = Time.unscaledTime + GO_BANNER_SECONDS;
                _winnerBannerUntil = 0f;
            }
            _wasRaceActive = _raceModel.RaceActive;

            if (winner != null && _winnerBannerUntil == 0f)
            {
                _winnerBannerUntil = Time.unscaledTime + WINNER_BANNER_SECONDS;
            }
            if (_raceModel.CountdownValue > 0)
            {
                _bannerLabel.text = _raceModel.CountdownValue.ToString();
                _bannerLabel.style.fontSize = UiTheme.FONT_COUNTDOWN;
                _bannerLabel.style.display = DisplayStyle.Flex;
                PopBanner();
            }
            else if (winner != null && Time.unscaledTime < _winnerBannerUntil)
            {
                // Brief celebration only: after a few seconds the banner clears so
                // the rest of the field stays watchable.
                _bannerLabel.text = $"WINNER  {winner.DisplayName}  {winner.FinishTime:0.0}s";
                _bannerLabel.style.fontSize = UiTheme.FONT_TITLE;
                _bannerLabel.style.display = DisplayStyle.Flex;
                PopBanner();
            }
            else if (Time.unscaledTime < _goBannerUntil)
            {
                _bannerLabel.text = "GO!";
                _bannerLabel.style.fontSize = UiTheme.FONT_BANNER;
                _bannerLabel.style.display = DisplayStyle.Flex;
                PopBanner();
            }
            else
            {
                _bannerLabel.style.display = DisplayStyle.None;
                _lastBannerText = null;
            }

            RefreshPodium();
            RefreshFieldWidgets();
        }

        /// <summary>Scale-pop the banner once each time its text changes.</summary>
        private void PopBanner()
        {
            if (_bannerLabel.text == _lastBannerText)
            {
                return;
            }
            _lastBannerText = _bannerLabel.text;
            _bannerLabel.experimental.animation.Start(0f, 1f, UiTheme.POP_MS, (element, value) =>
            {
                float scale = 1.6f - 0.6f * value;
                element.style.scale = new Scale(new Vector2(scale, scale));
            });
        }

        private void RefreshPodium()
        {
            // Only during the pause between races, once results exist.
            bool showPodium = !_raceModel.RaceActive && _raceModel.Racers.Count > 0
                && _raceModel.CountdownValue == 0;
            _podiumPanel.style.display = showPodium ? DisplayStyle.Flex : DisplayStyle.None;
            if (!showPodium)
            {
                return;
            }
            for (int podiumIndex = 0; podiumIndex < PODIUM_ROWS; podiumIndex++)
            {
                RacerState medalist = null;
                for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
                {
                    if (_raceModel.Racers[racerIndex].Place == podiumIndex + 1)
                    {
                        medalist = _raceModel.Racers[racerIndex];
                        break;
                    }
                }
                if (medalist != null)
                {
                    float rating = _eloModel.GetRating(medalist.CreatureId);
                    _podiumLabels[podiumIndex].text =
                        $"{medalist.DisplayName}  {medalist.FinishTime:0.0}s  ELO {rating:0}";
                }
                else
                {
                    _podiumLabels[podiumIndex].text = "—";
                }
            }
        }

        /// <summary>Drives both the right-edge rail and the top-3 chips.</summary>
        private void RefreshFieldWidgets()
        {
            int leaderCount = SelectLeaders();
            bool show = leaderCount > 0;
            if (show != _widgetsVisible)
            {
                _widgetsVisible = show;
                DisplayStyle display = show ? DisplayStyle.Flex : DisplayStyle.None;
                _rail.style.display = display;
                _chipRow.style.display = display;
            }
            if (!show)
            {
                return;
            }
            RefreshRail(leaderCount);
            RefreshChips(leaderCount);
        }

        /// <summary>
        /// Fills <see cref="_leaders"/> with the best racers, best first, by an
        /// insertion pass over the field. No sorting of the Model's own list and
        /// no allocation — the buffer is reused every refresh.
        /// </summary>
        private int SelectLeaders()
        {
            int count = 0;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                float key = RankKey(racer);
                int insertAt = count;
                while (insertAt > 0 && RankKey(_leaders[insertAt - 1]) < key)
                {
                    insertAt--;
                }
                if (insertAt >= MAX_RAIL_DOTS)
                {
                    continue;
                }
                int shiftFrom = count < MAX_RAIL_DOTS ? count : MAX_RAIL_DOTS - 1;
                for (int shiftIndex = shiftFrom; shiftIndex > insertAt; shiftIndex--)
                {
                    _leaders[shiftIndex] = _leaders[shiftIndex - 1];
                }
                _leaders[insertAt] = racer;
                if (count < MAX_RAIL_DOTS)
                {
                    count++;
                }
            }
            return count;
        }

        /// <summary>Finished racers outrank anything still on track, by place.</summary>
        private static float RankKey(RacerState racer)
        {
            if (racer.Status == RacerStatus.Finished)
            {
                return 1000000f - racer.Place;
            }
            return racer.Progress;
        }

        private void RefreshRail(int leaderCount)
        {
            if (_railDotsShown != leaderCount)
            {
                for (int dotIndex = 0; dotIndex < MAX_RAIL_DOTS; dotIndex++)
                {
                    _railDots[dotIndex].style.display =
                        dotIndex < leaderCount ? DisplayStyle.Flex : DisplayStyle.None;
                }
                _railDotsShown = leaderCount;
            }
            float trackLength = Mathf.Max(1f, _raceModel.TrackLengthMeters);
            for (int dotIndex = 0; dotIndex < leaderCount; dotIndex++)
            {
                RacerState racer = _leaders[dotIndex];
                float fraction = racer.Status == RacerStatus.Finished
                    ? 1f
                    : Mathf.Clamp01(racer.Progress / trackLength);
                // Bottom of the rail is the start line, top is the finish.
                _railDots[dotIndex].style.top =
                    new Length((1f - fraction) * RAIL_SPAN_PERCENT, LengthUnit.Percent);
                Color tint = racer.Status == RacerStatus.Dnf ? UiTheme.Dnf : racer.Tint;
                if (_railDotTints[dotIndex] != tint)
                {
                    _railDotTints[dotIndex] = tint;
                    _railDots[dotIndex].style.backgroundColor = tint;
                }
            }
        }

        private void RefreshChips(int leaderCount)
        {
            int shown = leaderCount < TOP_CHIP_COUNT ? leaderCount : TOP_CHIP_COUNT;
            if (_chipsShown != shown)
            {
                for (int chipIndex = 0; chipIndex < TOP_CHIP_COUNT; chipIndex++)
                {
                    _chips[chipIndex].style.display =
                        chipIndex < shown ? DisplayStyle.Flex : DisplayStyle.None;
                }
                _chipsShown = shown;
            }
            for (int chipIndex = 0; chipIndex < shown; chipIndex++)
            {
                RacerState racer = _leaders[chipIndex];
                // Name only changes when the chip's occupant does; skipping the
                // rebuild keeps the substring allocation out of the steady state.
                if (!string.Equals(_chipSourceNames[chipIndex], racer.DisplayName))
                {
                    _chipSourceNames[chipIndex] = racer.DisplayName;
                    _chipNames[chipIndex].text = FirstWord(racer.DisplayName);
                }
                Color tint = racer.Status == RacerStatus.Dnf ? UiTheme.Dnf : racer.Tint;
                if (_chipTints[chipIndex] != tint)
                {
                    _chipTints[chipIndex] = tint;
                    _chipSwatches[chipIndex].style.backgroundColor = tint;
                }
            }
        }

        private static string FirstWord(string displayName)
        {
            if (string.IsNullOrEmpty(displayName))
            {
                return string.Empty;
            }
            int space = displayName.IndexOf(' ');
            return space > 0 ? displayName.Substring(0, space) : displayName;
        }
    }
}
