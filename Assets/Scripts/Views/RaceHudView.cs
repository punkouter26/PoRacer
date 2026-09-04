using PoRacer.Models;
using PoRacer.Presentation;
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
    /// winner), a race intro card that wipes through on the start, and the
    /// between-races podium with per-creature ELO swing. Refreshed on a schedule by reading
    /// the Models — no per-frame polling in Update, and no allocation in the
    /// refresh past the elements pooled during the first build.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class RaceHudView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const float GO_BANNER_SECONDS = 1.5f;
        private const float WINNER_BANNER_SECONDS = 3f;
        /// <summary>
        /// Height of the reserved top strip: the MENU button is CONTROL_SM tall
        /// and sits SPACE_XS from the top, and the game name and fps readout share
        /// that line. Nothing else may be drawn above this.
        /// </summary>
        // A property, not a static readonly field: CONTROL_SM now reads Screen.dpi to
        // hold the touch target at 48 dp, and Unity forbids that from a static field
        // initializer on a MonoBehaviour ("get_dpi is not allowed to be called from a
        // MonoBehaviour constructor"). Evaluating on access also picks up a resolution
        // change, which a field initializer never would.
        private static float TopFurniture => UiTheme.SPACE_XS + UiTheme.CONTROL_SM;

        private const int PODIUM_ROWS = 3;
        private const int TOP_CHIP_COUNT = 3;
        // Rail dots are pooled: a 100+ racer field shows only the leading pack.
        private const int MAX_RAIL_DOTS = 32;
        private const float RAIL_WIDTH = 8f;
        private const float RAIL_DOT_SIZE = 10f;
        // Percent of the rail a dot's top may reach, leaving room for its height.
        private const float RAIL_SPAN_PERCENT = 96f;
        private const float CHIP_SWATCH_SIZE = 8f;

        // --- Race intro card ---
        // Sits above the countdown/GO banner's 40% line so the two never collide.
        private const float INTRO_TOP_PERCENT = 28f;
        private const int INTRO_TOTAL_MS = 2500;
        private const int INTRO_IN_MS = 320;
        private const int INTRO_OUT_MS = 320;
        private const float INTRO_SLIDE_PX = 280f;
        // Safety net for the schedule-driven hide, past the animation's own end.
        private const float INTRO_HIDE_GRACE_SECONDS = 0.25f;

        // --- ELO delta rich-text tints ---
        private const string DELTA_UP_HEX = "#7CE87C";
        private const string DELTA_DOWN_HEX = "#E86A5A";

        private static readonly Color[] MedalColors = { UiTheme.Gold, UiTheme.Silver, UiTheme.Bronze };

        /// <summary>
        /// Slide-in, hold, slide-out of the intro card baked into a single
        /// animation. Static so the delegate is allocated once for the process
        /// rather than per race start.
        /// </summary>
        private static readonly System.Action<VisualElement, float> IntroCardTick = (element, value) =>
        {
            float elapsedMs = value * INTRO_TOTAL_MS;
            float appear;
            if (elapsedMs < INTRO_IN_MS)
            {
                float remaining = 1f - elapsedMs / INTRO_IN_MS;
                appear = 1f - remaining * remaining * remaining;
            }
            else if (elapsedMs < INTRO_TOTAL_MS - INTRO_OUT_MS)
            {
                appear = 1f;
            }
            else
            {
                float exit = (elapsedMs - (INTRO_TOTAL_MS - INTRO_OUT_MS)) / INTRO_OUT_MS;
                appear = 1f - exit * exit;
            }
            element.style.opacity = appear;
            element.style.translate = new Translate(-INTRO_SLIDE_PX * (1f - appear), 0f);
            if (value >= 1f)
            {
                element.style.display = DisplayStyle.None;
            }
        };

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
        private readonly System.Collections.Generic.List<VisualElement> _podiumRows = new();
        private string _lastBannerText;

        // --- Race intro card ---
        private VisualElement _introCard;
        private Label _introRaceLabel;
        private Label _introTrackLabel;
        private Label _introFieldLabel;
        private float _introCardHideAt;

        // --- Results show ---
        private bool _podiumWasVisible;
        // Row change guards: text is only rebuilt when the occupant or its ELO
        // delta actually changes, keeping the shown podium allocation-free.
        private readonly string[] _podiumSourceIds = new string[PODIUM_ROWS];
        private readonly int[] _podiumSourceDeltas = new int[PODIUM_ROWS];

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

            // Corner layout: game name top-left, fps top-centre, MENU top-right,
            // DBG bottom-left (debug builds), version bottom-right.
            var titleLabel = new Label("PoRacer") { pickingMode = PickingMode.Ignore };
            titleLabel.style.position = Position.Absolute;
            titleLabel.style.top = UiTheme.SPACE_XS;
            titleLabel.style.left = UiTheme.SPACE_SM;
            titleLabel.style.color = UiTheme.Accent;
            titleLabel.style.fontSize = UiTheme.FONT_MD;
            titleLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            safeRoot.Add(titleLabel);

            var versionLabel = new Label($"v{Application.version}")
            {
                pickingMode = PickingMode.Ignore
            };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.bottom = UiTheme.SPACE_SM;
            versionLabel.style.right = UiTheme.SPACE_SM;
            versionLabel.style.color = UiTheme.TextDim;
            versionLabel.style.fontSize = UiTheme.FONT_SM;
            safeRoot.Add(versionLabel);

            // The fps readout lives on DebugOverlayView's strip, which now ships in
            // release builds too, so the HUD no longer needs a fallback copy of it -
            // two labels at the same top-centre anchor drew over each other.

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

            BuildIntroCard(safeRoot);

            // Podium: shown in the pause between races (top 3 with medal tints).
            _podiumPanel = new VisualElement { pickingMode = PickingMode.Ignore };
            _podiumPanel.style.position = Position.Absolute;
            _podiumPanel.style.top = new Length(38f, LengthUnit.Percent);
            // 6% side margins, not 12: at a 420 dp reference width a full row
            // ("Mighty Rocket the Isaac H1  18.5s  ELO 1216  +16") needs the room,
            // and the rows below are allowed to wrap rather than overflow the card.
            _podiumPanel.style.left = new Length(6f, LengthUnit.Percent);
            _podiumPanel.style.right = new Length(6f, LengthUnit.Percent);
            // Opaque, not glass: the race carries on behind this panel, and a
            // creature showing through the results made both hard to read.
            UiTheme.StyleModal(_podiumPanel);
            // The panel is a modal, so it takes its own clicks rather than letting
            // them fall through to whatever is behind it.
            _podiumPanel.pickingMode = PickingMode.Position;
            _podiumPanel.style.display = DisplayStyle.None;
            var podiumTitle = new Label("RESULTS") { pickingMode = PickingMode.Ignore };
            podiumTitle.style.color = UiTheme.TextDim;
            podiumTitle.style.fontSize = UiTheme.FONT_XS;
            podiumTitle.style.unityFontStyleAndWeight = FontStyle.Bold;
            podiumTitle.style.letterSpacing = 2f;
            podiumTitle.style.unityTextAlign = TextAnchor.MiddleCenter;
            _podiumPanel.Add(podiumTitle);
            _podiumPanel.Add(UiTheme.MakeDivider());
            for (int podiumIndex = 0; podiumIndex < PODIUM_ROWS; podiumIndex++)
            {
                var row = new VisualElement { pickingMode = PickingMode.Ignore };
                row.style.flexDirection = FlexDirection.Row;
                row.style.alignItems = Align.Center;
                row.style.marginTop = UiTheme.SPACE_XS;
                row.style.minHeight = UiTheme.SPACE_XL;
                VisualElement medal = UiTheme.MakeSwatch(MedalColors[podiumIndex], UiTheme.SPACE_MD);
                medal.style.marginRight = UiTheme.SPACE_SM;
                row.Add(medal);
                var label = new Label { pickingMode = PickingMode.Ignore };
                label.style.color = UiTheme.Text;
                label.style.fontSize = UiTheme.FONT_SM;
                // A flex-row child defaults to flex-shrink 0 and no wrapping, so a
                // long name pushed the ELO delta clean off the card's right edge.
                // Let the label take the remaining width and wrap inside it.
                label.style.flexGrow = 1f;
                label.style.flexShrink = 1f;
                label.style.whiteSpace = WhiteSpace.Normal;
                // The ELO delta is injected as a <color> tag.
                label.enableRichText = true;
                row.Add(label);
                _podiumLabels.Add(label);
                _podiumRows.Add(row);
                _podiumPanel.Add(row);
            }

            // The results panel is a stop, not a pause: the player decides what
            // happens next instead of the game silently looping forever.
            var podiumButtons = new VisualElement();
            podiumButtons.style.flexDirection = FlexDirection.Row;
            podiumButtons.style.justifyContent = Justify.Center;
            podiumButtons.style.marginTop = UiTheme.SPACE_SM;
            var raceAgainButton = new Button(() => _spawn.RaceAgain()) { text = "RACE AGAIN" };
            raceAgainButton.style.height = UiTheme.CONTROL_MD;
            raceAgainButton.style.fontSize = UiTheme.FONT_SM;
            raceAgainButton.style.flexGrow = 2f;
            raceAgainButton.style.flexBasis = 0f;
            UiTheme.SetMargin(raceAgainButton, 0f, 0f);
            // Primary action carries the accent; MENU is the quiet way out.
            UiTheme.StyleButton(raceAgainButton, accent: true);
            UiTheme.AddHover(raceAgainButton, accent: true);
            podiumButtons.Add(raceAgainButton);
            var backToMenuButton = new Button(() => _spawn.RequestMenu()) { text = "MENU" };
            backToMenuButton.style.height = UiTheme.CONTROL_MD;
            backToMenuButton.style.fontSize = UiTheme.FONT_SM;
            backToMenuButton.style.flexGrow = 1f;
            backToMenuButton.style.flexBasis = 0f;
            UiTheme.SetMargin(backToMenuButton, 0f, 0f);
            backToMenuButton.style.marginLeft = UiTheme.SPACE_SM;
            UiTheme.StyleButton(backToMenuButton);
            UiTheme.AddHover(backToMenuButton);
            podiumButtons.Add(backToMenuButton);
            _podiumPanel.Add(UiTheme.MakeDivider());
            _podiumPanel.Add(podiumButtons);

            safeRoot.Add(_podiumPanel);

            var menuButton = new Button(() => _spawn.RequestMenu()) { text = "MENU" };
            menuButton.style.position = Position.Absolute;
            menuButton.style.top = UiTheme.SPACE_XS;
            menuButton.style.right = UiTheme.SPACE_SM;
            menuButton.style.width = 76;
            menuButton.style.height = UiTheme.CONTROL_SM;
            menuButton.style.fontSize = UiTheme.FONT_SM;
            UiTheme.StyleButton(menuButton);
            UiTheme.AddHover(menuButton);
            safeRoot.Add(menuButton);

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
        }

        /// <summary>
        /// Broadcast-style bug that wipes in from the left edge as the race goes
        /// green: race number, track and field size. Built once and toggled — a
        /// race start only sets three strings and restarts one animation.
        /// </summary>
        private void BuildIntroCard(VisualElement safeRoot)
        {
            _introCard = new VisualElement { pickingMode = PickingMode.Ignore };
            _introCard.style.position = Position.Absolute;
            _introCard.style.top = new Length(INTRO_TOP_PERCENT, LengthUnit.Percent);
            _introCard.style.left = UiTheme.SPACE_MD;
            _introCard.style.flexDirection = FlexDirection.Row;
            _introCard.style.alignItems = Align.Center;
            UiTheme.StyleGlassPanel(_introCard);
            _introCard.style.display = DisplayStyle.None;
            safeRoot.Add(_introCard);

            _introRaceLabel = new Label { pickingMode = PickingMode.Ignore };
            _introRaceLabel.style.color = UiTheme.Text;
            _introRaceLabel.style.fontSize = UiTheme.FONT_TITLE;
            _introRaceLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _introCard.Add(_introRaceLabel);

            var divider = new VisualElement { pickingMode = PickingMode.Ignore };
            divider.style.width = 2f;
            divider.style.height = UiTheme.CONTROL_MD;
            divider.style.flexShrink = 0f;
            divider.style.backgroundColor = UiTheme.Gold;
            UiTheme.SetMargin(divider, 0f, UiTheme.SPACE_MD);
            _introCard.Add(divider);

            var details = new VisualElement { pickingMode = PickingMode.Ignore };
            _introCard.Add(details);

            _introTrackLabel = new Label { pickingMode = PickingMode.Ignore };
            _introTrackLabel.style.color = UiTheme.Gold;
            _introTrackLabel.style.fontSize = UiTheme.FONT_LG;
            _introTrackLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            details.Add(_introTrackLabel);

            _introFieldLabel = new Label { pickingMode = PickingMode.Ignore };
            _introFieldLabel.style.color = UiTheme.TextDim;
            _introFieldLabel.style.fontSize = UiTheme.FONT_SM;
            _introFieldLabel.style.letterSpacing = 1.5f;
            details.Add(_introFieldLabel);
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
            // Below the whole top furniture band, not just below the fps text.
            // The band is as tall as the MENU button (CONTROL_SM at SPACE_XS), and
            // the chips used to start at SPACE_XXL + SPACE_XS = 36, which put them
            // under the button and through the fps readout at the same time.
            _chipRow.style.top = TopFurniture + UiTheme.SPACE_XS;
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

            // Only a racer who actually crossed, or led on distance when the clock
            // ran out, counts for the celebration banner: an all-DNF race gets no
            // "WINNER" fanfare.
            RacerState winner = null;
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                if (racer.Place == 1
                    && (racer.Status == RacerStatus.Finished || racer.Status == RacerStatus.TimedOut))
                {
                    winner = racer;
                    break;
                }
            }

            if (_raceModel.RaceActive && !_wasRaceActive)
            {
                _goBannerUntil = Time.unscaledTime + GO_BANNER_SECONDS;
                _winnerBannerUntil = 0f;
                ShowIntroCard();
            }
            _wasRaceActive = _raceModel.RaceActive;

            // The card retires itself at the end of its animation; this is the
            // backstop for a run interrupted by the menu or a domain event.
            if (_introCardHideAt > 0f && Time.unscaledTime >= _introCardHideAt)
            {
                _introCardHideAt = 0f;
                _introCard.style.display = DisplayStyle.None;
            }

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

        /// <summary>
        /// Fills and plays the intro card. Independent of the banner elements, so
        /// the countdown and GO logic are untouched by it.
        /// </summary>
        private void ShowIntroCard()
        {
            _introRaceLabel.text = $"RACE {_raceModel.RaceNumber}";
            _introTrackLabel.text = _raceModel.TrackName;
            _introFieldLabel.text = $"{_raceModel.Racers.Count} RACERS";
            _introCard.style.opacity = 0f;
            _introCard.style.translate = new Translate(-INTRO_SLIDE_PX, 0f);
            _introCard.style.display = DisplayStyle.Flex;
            _introCardHideAt = Time.unscaledTime
                + INTRO_TOTAL_MS * 0.001f + INTRO_HIDE_GRACE_SECONDS;
            _introCard.experimental.animation.Start(0f, 1f, INTRO_TOTAL_MS, IntroCardTick);
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
            if (showPodium != _podiumWasVisible)
            {
                _podiumWasVisible = showPodium;
                _podiumPanel.style.display = showPodium ? DisplayStyle.Flex : DisplayStyle.None;
                if (showPodium)
                {
                    // Force one rebuild of every row: racer ids can repeat across
                    // races, so the change guards below must not carry over.
                    for (int rowIndex = 0; rowIndex < PODIUM_ROWS; rowIndex++)
                    {
                        _podiumSourceIds[rowIndex] = null;
                    }
                    PlayPodiumEntrance();
                }
            }
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

                string sourceId = medalist == null ? string.Empty : medalist.RacerId;
                float rating = 0f;
                int delta = 0;
                if (medalist != null)
                {
                    rating = _eloModel.GetRating(medalist.CreatureId);
                    delta = Mathf.RoundToInt(_eloModel.GetLastRaceDelta(medalist.CreatureId));
                }
                // Once the ELO update has landed nothing here changes again, so
                // the steady-state podium builds no strings at all.
                if (string.Equals(_podiumSourceIds[podiumIndex], sourceId)
                    && _podiumSourceDeltas[podiumIndex] == delta)
                {
                    continue;
                }
                _podiumSourceIds[podiumIndex] = sourceId;
                _podiumSourceDeltas[podiumIndex] = delta;

                if (medalist == null)
                {
                    _podiumLabels[podiumIndex].text = "—";
                    continue;
                }
                // Only a racer that crossed owns a finish time. One that ran out
                // of clock is ranked on distance, so show the distance; one that
                // was knocked out shows DNF.
                string timeText;
                if (medalist.Status == RacerStatus.Finished)
                {
                    timeText = $"{medalist.FinishTime:0.0}s";
                }
                else if (medalist.Status == RacerStatus.TimedOut)
                {
                    timeText = $"{medalist.Progress:0.0}m";
                }
                else
                {
                    timeText = "DNF";
                }
                if (delta == 0)
                {
                    _podiumLabels[podiumIndex].text =
                        $"{medalist.DisplayName}  {timeText}  ELO {rating:0}";
                }
                else
                {
                    string deltaHex = delta > 0 ? DELTA_UP_HEX : DELTA_DOWN_HEX;
                    string sign = delta > 0 ? "+" : "-";
                    int magnitude = delta > 0 ? delta : -delta;
                    _podiumLabels[podiumIndex].text =
                        $"{medalist.DisplayName}  {timeText}  ELO {rating:0}  "
                        + $"<color={deltaHex}>{sign}{magnitude}</color>";
                }
            }
        }

        /// <summary>
        /// Results reveal: the panel rises into place, then the three rows land
        /// top to bottom.
        /// </summary>
        private void PlayPodiumEntrance()
        {
            UiTheme.PlayEnter(_podiumPanel, 0, UiTheme.PANEL_SLIDE_PX);
            for (int podiumIndex = 0; podiumIndex < _podiumRows.Count; podiumIndex++)
            {
                UiTheme.PlayEnter(_podiumRows[podiumIndex], podiumIndex * UiTheme.STAGGER_MS);
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

        /// <summary>Placed racers outrank anything still on track, by place.</summary>
        private static float RankKey(RacerState racer)
        {
            if (racer.Status == RacerStatus.Finished || racer.Status == RacerStatus.TimedOut)
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
                    _chipNames[chipIndex].text = racer.DisplayName;
                }
                Color tint = racer.Status == RacerStatus.Dnf ? UiTheme.Dnf : racer.Tint;
                if (_chipTints[chipIndex] != tint)
                {
                    _chipTints[chipIndex] = tint;
                    _chipSwatches[chipIndex].style.backgroundColor = tint;
                }
            }
        }
    }
}
