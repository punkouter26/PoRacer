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
    /// Deliberately minimal so the race itself owns the screen: the shared corner
    /// furniture (title top-left, MENU top-right, version bottom-right), a thin
    /// progress rail hugging the right edge whose three leading dots carry place
    /// badges, one announcement lane under the top band (countdown / GO / winner and
    /// the race intro card, which the Sim Wars pill and the director's caption join),
    /// and the between-races results sheet with PODIUM / LEAGUE / STATS tabs.
    /// Refreshed on a schedule by reading the Models — no per-frame polling in
    /// Update, and no allocation in the refresh past the elements pooled during the
    /// first build.
    ///
    /// The top-3 chips that sat under the top band were retired: they restated the
    /// rail's leading dots, so the leaders' places now ride on the dots themselves.
    /// One status pill heads the lane instead: "getting racers ready" while the grid
    /// spawns, then the leader's name and the race clock. MENU mid-race raises a
    /// leave sheet (skip to results / leave / keep watching) rather than quitting.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class RaceHudView : MonoBehaviour
    {
        private const long REFRESH_INTERVAL_MS = 250;
        private const float GO_BANNER_SECONDS = 1.5f;
        private const float WINNER_BANNER_SECONDS = 3f;

        private const int PODIUM_ROWS = 3;
        private const int BADGE_COUNT = 3;
        // Rail dots are pooled: a 100+ racer field shows only the leading pack.
        private const int MAX_RAIL_DOTS = 32;
        private const float RAIL_WIDTH = 8f;
        private const float RAIL_DOT_SIZE = 10f;
        // Place badge on a leading dot: big enough to hold a FONT_XS digit.
        private const float RAIL_BADGE_SIZE = 22f;
        private const float RAIL_BADGE_BORDER = 2f;
        // Percent of the rail a dot's top may reach, leaving room for its height.
        private const float RAIL_SPAN_PERCENT = 96f;

        // --- Results sheet ---
        private const int TAB_PODIUM = 0;
        private const int TAB_LEAGUE = 1;
        private const int TAB_STATS = 2;
        // Wide enough for the "TWITCH" header at FONT_XS bold, the widest cell, at the
        // base type scale; read through UiTheme.ScaleWithFont.
        private const float STAT_COLUMN_WIDTH = 58f;
        // Racers listed by name under the podium before the rest collapse to "+N more".
        private const int ALSO_RAN_MAX = 12;

        // --- Status line (loading / leader + clock) ---
        private const float STATUS_SWATCH = 10f;
        private const int SECONDS_PER_MINUTE = 60;

        // --- Race intro card ---
        private const int INTRO_TOTAL_MS = 2500;
        private const int INTRO_IN_MS = 320;
        private const int INTRO_OUT_MS = 320;
        private const float INTRO_SLIDE_PX = 280f;
        // Safety net for the schedule-driven hide, past the animation's own end.
        private const float INTRO_HIDE_GRACE_SECONDS = 0.25f;

        // ELO swing tints. Deliberately not green and red: those two colours are the
        // racer legend (RL baseline / heuristic bot), so the HUD never spends them on
        // anything else.
        private static readonly string DeltaUpHex = "#" + ColorUtility.ToHtmlStringRGB(UiTheme.AccentSoft);
        private static readonly string DeltaDownHex = "#" + ColorUtility.ToHtmlStringRGB(UiTheme.TextDim);

        private static readonly Color[] MedalColors = { UiTheme.Gold, UiTheme.Silver, UiTheme.Bronze };

        // Rest-of-field order: placed racers by place, then the unplaced by distance.
        private static readonly System.Comparison<RacerState> AlsoRanOrder = (first, second) =>
        {
            bool firstPlaced = first.Place > 0;
            bool secondPlaced = second.Place > 0;
            if (firstPlaced != secondPlaced)
            {
                return firstPlaced ? -1 : 1;
            }
            return firstPlaced
                ? first.Place.CompareTo(second.Place)
                : second.Progress.CompareTo(first.Progress);
        };

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
        private Systems_Race _race;
        private SkyModel _skyModel;
        private SimWarsModel _league;
        private RaceTelemetryModel _telemetryModel;
        private VisualElement _hudRoot;
        private VisualElement _announceSlot;
        private Label _bannerLabel;
        private bool _wasRaceActive;
        private float _goBannerUntil;
        private float _winnerBannerUntil;
        private string _lastBannerText;

        // --- Race intro card ---
        private VisualElement _introCard;
        private Label _introRaceLabel;
        private Label _introTrackLabel;
        private Label _introFieldLabel;
        private float _introCardHideAt;

        // --- Results sheet ---
        private VisualElement _podiumPanel;
        private Button[] _resultsTabs;
        private readonly VisualElement[] _resultsPages = new VisualElement[3];
        private int _resultsTab;
        private readonly System.Collections.Generic.List<Label> _podiumLabels = new();
        private readonly System.Collections.Generic.List<VisualElement> _podiumRows = new();
        private readonly Label[] _statsNames = new Label[PODIUM_ROWS];
        private readonly Label[] _statsSpeed = new Label[PODIUM_ROWS];
        private readonly Label[] _statsCost = new Label[PODIUM_ROWS];
        private readonly Label[] _statsEnergy = new Label[PODIUM_ROWS];
        private readonly Label[] _statsTwitch = new Label[PODIUM_ROWS];
        private readonly VisualElement[] _statsRows = new VisualElement[PODIUM_ROWS];
        private bool _podiumWasVisible;
        private bool _leagueTabShown = true;
        // Row change guards: text is only rebuilt when the occupant or its ELO
        // delta actually changes, keeping the shown podium allocation-free.
        private readonly string[] _podiumSourceIds = new string[PODIUM_ROWS];
        private readonly int[] _podiumSourceDeltas = new int[PODIUM_ROWS];
        private readonly RacerState[] _medalists = new RacerState[PODIUM_ROWS];
        private readonly VisualElement[] _podiumMedals = new VisualElement[PODIUM_ROWS];
        private readonly VisualElement[] _statsMedals = new VisualElement[PODIUM_ROWS];
        private Label _winnerHeadline;
        private Label _alsoRanLabel;
        private readonly System.Collections.Generic.List<RacerState> _alsoRan = new();
        private readonly System.Text.StringBuilder _alsoRanText = new();

        // --- Status line: "getting racers ready" before the grid, leader + clock during ---
        private VisualElement _statusChip;
        private VisualElement _statusSwatch;
        private Label _statusLabel;
        private string _statusLeaderId;
        private int _statusWholeSeconds = -1;
        private bool _statusShowsLoading;

        // --- Leave-race sheet ---
        private VisualElement _leaveSheet;
        private Button _skipButton;
        private bool _leaveSheetOpen;

        // --- Right-edge progress rail ---
        private VisualElement _rail;
        private readonly VisualElement[] _railDots = new VisualElement[MAX_RAIL_DOTS];
        private readonly Color[] _railDotTints = new Color[MAX_RAIL_DOTS];
        private int _railDotsShown = -1;
        private readonly VisualElement[] _railBadges = new VisualElement[BADGE_COUNT];
        private readonly Color[] _railBadgeTints = new Color[BADGE_COUNT];

        // Leader ordering scratch buffer, filled in place every refresh.
        private readonly RacerState[] _leaders = new RacerState[MAX_RAIL_DOTS];
        private bool _widgetsVisible;

        [Inject]
        public void Construct(
            RaceModel raceModel,
            EloModel eloModel,
            RaceConfigModel configModel,
            Systems_Spawn spawn,
            Systems_Race race,
            SkyModel skyModel,
            SimWarsModel league,
            RaceTelemetryModel telemetryModel)
        {
            _raceModel = raceModel;
            _eloModel = eloModel;
            _configModel = configModel;
            _spawn = spawn;
            _race = race;
            _skyModel = skyModel;
            _league = league;
            _telemetryModel = telemetryModel;
        }

        private void Start()
        {
            VisualElement root = GetComponent<UIDocument>().rootVisualElement;
            root.pickingMode = PickingMode.Ignore;
            _hudRoot = root;
            VisualElement safeRoot = UiTheme.BuildSafeRoot(root);

            // Corner layout shared with every screen: game name top-left, fps
            // top-centre (DebugOverlayView), MENU top-right, DBG bottom-left
            // (DebugOverlayView), version bottom-right.
            safeRoot.Add(UiTheme.MakeTitleFurniture());
            safeRoot.Add(UiTheme.MakeVersionFurniture());

            BuildProgressRail(safeRoot);
            BuildAnnounceSlot(safeRoot);
            BuildResultsSheet(safeRoot);
            BuildLeaveSheet(safeRoot);

            safeRoot.Add(UiTheme.MakeMenuFurniture(OnMenuPressed));

            root.schedule.Execute(Refresh).Every(REFRESH_INTERVAL_MS);
        }

        /// <summary>
        /// The single lane for transient race messages, directly under the top band
        /// and clear of the rail. Its children are in flow, so whatever is showing at
        /// once stacks instead of overlapping — the four of them used to be pinned at
        /// four different heights (18, 22, 28 and 40%) and collided on short screens.
        /// </summary>
        private void BuildAnnounceSlot(VisualElement safeRoot)
        {
            _announceSlot = new VisualElement { name = UiTheme.ANNOUNCE_SLOT, pickingMode = PickingMode.Ignore };
            _announceSlot.style.position = Position.Absolute;
            _announceSlot.style.top = UiTheme.TopBand;
            _announceSlot.style.left = UiTheme.SPACE_MD;
            _announceSlot.style.right = RailClearance();
            _announceSlot.style.alignItems = Align.Center;
            safeRoot.Add(_announceSlot);

            BuildStatusChip(_announceSlot);

            _bannerLabel = new Label { pickingMode = PickingMode.Ignore };
            // Wraps inside the lane: the winner line at title size is wider than a
            // narrow handset, and unwrapped it ran off both edges.
            _bannerLabel.style.alignSelf = Align.Stretch;
            _bannerLabel.style.whiteSpace = WhiteSpace.Normal;
            _bannerLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _bannerLabel.style.fontSize = UiTheme.FONT_TITLE;
            _bannerLabel.style.color = UiTheme.Gold;
            _bannerLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            UiTheme.AddTextShadow(_bannerLabel);
            _bannerLabel.style.display = DisplayStyle.None;
            _announceSlot.Add(_bannerLabel);

            BuildIntroCard(_announceSlot);
        }

        /// <summary>Right-hand inset that keeps the announcement lane off the rail and its badges.</summary>
        private static float RailClearance()
        {
            return UiTheme.RAIL_CLEARANCE;
        }

        /// <summary>
        /// One quiet pill at the head of the lane. Before the grid exists it says the
        /// racers are on their way (the spawn can hold the screen for seconds, and it
        /// used to hold it blank); during the race it names the leader, in their own
        /// legend tint, beside the race clock. The rail's dots carry positions but no
        /// names, and the clock was only ever shown after the flag.
        /// </summary>
        private void BuildStatusChip(VisualElement slot)
        {
            _statusChip = new VisualElement { pickingMode = PickingMode.Ignore };
            _statusChip.style.flexDirection = FlexDirection.Row;
            _statusChip.style.alignItems = Align.Center;
            _statusChip.style.maxWidth = new Length(100f, LengthUnit.Percent);
            _statusChip.style.minHeight = UiTheme.SPACE_XL;
            UiTheme.StyleChip(_statusChip);
            _statusChip.style.display = DisplayStyle.None;
            slot.Add(_statusChip);

            _statusSwatch = UiTheme.MakeSwatch(UiTheme.TextDim, STATUS_SWATCH);
            _statusSwatch.style.marginRight = UiTheme.SPACE_XS;
            _statusSwatch.style.flexShrink = 0f;
            _statusChip.Add(_statusSwatch);

            _statusLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.Text, bold: true);
            Ellipsize(_statusLabel);
            _statusChip.Add(_statusLabel);
        }

        /// <summary>
        /// Asked before MENU throws a race away. It used to abort on the spot, which
        /// also meant nobody's rating moved; skipping calls full time instead, so the
        /// race is ranked on distance and scored like any other.
        /// </summary>
        private void BuildLeaveSheet(VisualElement safeRoot)
        {
            _leaveSheet = new VisualElement { name = UiTheme.LEAVE_SHEET };
            _leaveSheet.style.position = Position.Absolute;
            _leaveSheet.style.top = new Length(30f, LengthUnit.Percent);
            _leaveSheet.style.left = UiTheme.SPACE_LG;
            _leaveSheet.style.right = UiTheme.SPACE_LG;
            UiTheme.StyleModal(_leaveSheet);
            _leaveSheet.pickingMode = PickingMode.Position;
            _leaveSheet.style.display = DisplayStyle.None;
            safeRoot.Add(_leaveSheet);

            Label title = MakeLabel(UiTheme.FONT_LG, UiTheme.Text, bold: true, "Leave this race?");
            title.style.unityTextAlign = TextAnchor.MiddleCenter;
            _leaveSheet.Add(title);
            Label body = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false,
                "Skip to results ranks everyone by distance now and counts the race. Leave throws it away.");
            body.style.whiteSpace = WhiteSpace.Normal;
            body.style.unityTextAlign = TextAnchor.MiddleCenter;
            body.style.marginTop = UiTheme.SPACE_XS;
            body.style.marginBottom = UiTheme.SPACE_SM;
            _leaveSheet.Add(body);

            _skipButton = SheetButton("SKIP TO RESULTS", accent: true, OnSkipPressed);
            _leaveSheet.Add(_skipButton);
            _leaveSheet.Add(SheetButton("LEAVE RACE", accent: false, OnLeavePressed));
            _leaveSheet.Add(SheetButton("KEEP WATCHING", accent: false, CloseLeaveSheet));
        }

        private static Button SheetButton(string text, bool accent, System.Action onClick)
        {
            var button = new Button(onClick) { text = text };
            button.style.height = UiTheme.CONTROL_SM;
            button.style.fontSize = UiTheme.FONT_SM;
            UiTheme.SetMargin(button, 0f, 0f);
            button.style.marginTop = UiTheme.SPACE_XS;
            UiTheme.StyleButton(button, accent);
            UiTheme.AddHover(button, accent);
            return button;
        }

        /// <summary>
        /// MENU is a straight exit only when there is nothing to lose: on the results
        /// sheet, or while the grid is still loading. Mid-race or mid-countdown it asks.
        /// </summary>
        private void OnMenuPressed()
        {
            bool raceUnderway = _raceModel.RaceActive || _raceModel.CountdownValue > 0;
            if (!raceUnderway)
            {
                _spawn.RequestMenu();
                return;
            }
            if (_leaveSheetOpen)
            {
                CloseLeaveSheet();
                return;
            }
            _leaveSheetOpen = true;
            // Skipping needs a running clock; during the countdown there is nothing to rank.
            _skipButton.style.display = _raceModel.RaceActive ? DisplayStyle.Flex : DisplayStyle.None;
            _leaveSheet.style.display = DisplayStyle.Flex;
            UiTheme.PlayEnter(_leaveSheet, 0, UiTheme.PANEL_SLIDE_PX);
        }

        private void OnSkipPressed()
        {
            CloseLeaveSheet();
            _race.FinishEarly();
        }

        private void OnLeavePressed()
        {
            CloseLeaveSheet();
            _spawn.RequestMenu();
        }

        private void CloseLeaveSheet()
        {
            _leaveSheetOpen = false;
            _leaveSheet.style.display = DisplayStyle.None;
        }

        /// <summary>
        /// Results-sheet buttons. The sheet only hides on the next scheduled refresh,
        /// so for up to REFRESH_INTERVAL_MS a second tap landed on a live button and
        /// restarted the spawn it had just started. The sheet goes inert on the first
        /// tap and is re-enabled the next time it is shown.
        /// </summary>
        private void OnResultsAction(System.Action action)
        {
            if (!_podiumPanel.enabledSelf)
            {
                return;
            }
            _podiumPanel.SetEnabled(false);
            action();
        }

        /// <summary>
        /// Broadcast-style bug that wipes in from the left as the race goes green:
        /// race number, track and field size. Built once and toggled — a race start
        /// only sets three strings and restarts one animation.
        /// </summary>
        private void BuildIntroCard(VisualElement slot)
        {
            _introCard = new VisualElement { pickingMode = PickingMode.Ignore };
            _introCard.style.flexDirection = FlexDirection.Row;
            _introCard.style.alignItems = Align.Center;
            _introCard.style.maxWidth = new Length(100f, LengthUnit.Percent);
            _introCard.style.marginTop = UiTheme.SPACE_XS;
            UiTheme.StyleGlassPanel(_introCard);
            _introCard.style.display = DisplayStyle.None;
            slot.Add(_introCard);

            _introRaceLabel = new Label { pickingMode = PickingMode.Ignore };
            _introRaceLabel.style.color = UiTheme.Text;
            _introRaceLabel.style.fontSize = UiTheme.FONT_TITLE;
            _introRaceLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            _introRaceLabel.style.flexShrink = 0f;
            _introCard.Add(_introRaceLabel);

            var divider = new VisualElement { pickingMode = PickingMode.Ignore };
            divider.style.width = 2f;
            divider.style.alignSelf = Align.Stretch;
            divider.style.flexShrink = 0f;
            divider.style.backgroundColor = UiTheme.Gold;
            UiTheme.SetMargin(divider, 0f, UiTheme.SPACE_SM);
            _introCard.Add(divider);

            // The details give way on a narrow screen, never the race number.
            var details = new VisualElement { pickingMode = PickingMode.Ignore };
            details.style.flexShrink = 1f;
            details.style.minWidth = 0f;
            _introCard.Add(details);

            _introTrackLabel = new Label { pickingMode = PickingMode.Ignore };
            _introTrackLabel.style.color = UiTheme.Gold;
            _introTrackLabel.style.fontSize = UiTheme.FONT_LG;
            _introTrackLabel.style.unityFontStyleAndWeight = FontStyle.Bold;
            Ellipsize(_introTrackLabel);
            details.Add(_introTrackLabel);

            _introFieldLabel = new Label { pickingMode = PickingMode.Ignore };
            _introFieldLabel.style.color = UiTheme.TextDim;
            _introFieldLabel.style.fontSize = UiTheme.FONT_XS;
            _introFieldLabel.style.letterSpacing = 1f;
            Ellipsize(_introFieldLabel);
            details.Add(_introFieldLabel);
        }

        /// <summary>
        /// The between-races sheet. One page at a time behind a tab row, so the podium,
        /// the Sim Wars league and the finishers' efficiency numbers share one card's
        /// height instead of stacking into a panel taller than the screen.
        /// </summary>
        private void BuildResultsSheet(VisualElement safeRoot)
        {
            _podiumPanel = new VisualElement { pickingMode = PickingMode.Ignore };
            _podiumPanel.style.position = Position.Absolute;
            // Anchored above the bottom furniture band and grown upward: the space
            // above it is empty sky on every shot, so that is where it grows.
            _podiumPanel.style.bottom = UiTheme.BottomBand;
            _podiumPanel.style.maxHeight = new Length(78f, LengthUnit.Percent);
            _podiumPanel.style.left = new Length(4f, LengthUnit.Percent);
            _podiumPanel.style.right = new Length(4f, LengthUnit.Percent);
            // Opaque, not glass: the race carries on behind this panel, and a
            // creature showing through the results made both hard to read.
            UiTheme.StyleModal(_podiumPanel);
            // The panel is a modal, so it takes its own clicks rather than letting
            // them fall through to whatever is behind it.
            _podiumPanel.pickingMode = PickingMode.Position;
            _podiumPanel.style.display = DisplayStyle.None;
            _podiumPanel.name = UiTheme.RESULTS_PANEL;

            _resultsTabs = UiTheme.BuildTabs(_podiumPanel, new[] { "PODIUM", "LEAGUE", "STATS" }, SelectResultsTab);

            // The pages scroll between the fixed tab row and the fixed buttons: the
            // sheet is capped at 78% of the screen, and a long league table or a big
            // field's rest-of-field list used to spill past it instead.
            var pageScroll = new ScrollView(ScrollViewMode.Vertical);
            pageScroll.style.flexShrink = 1f;
            UiTheme.StyleScrollView(pageScroll);
            _podiumPanel.Add(pageScroll);

            _resultsPages[TAB_PODIUM] = BuildPodiumPage();
            _resultsPages[TAB_LEAGUE] = new VisualElement { name = UiTheme.RESULTS_LEAGUE_PAGE, pickingMode = PickingMode.Ignore };
            _resultsPages[TAB_STATS] = BuildStatsPage();
            for (int pageIndex = 0; pageIndex < _resultsPages.Length; pageIndex++)
            {
                _resultsPages[pageIndex].style.marginTop = UiTheme.SPACE_XS;
                pageScroll.Add(_resultsPages[pageIndex]);
            }

            // The results panel is a stop, not a pause: the player decides what
            // happens next instead of the game silently looping forever.
            var podiumButtons = new VisualElement();
            podiumButtons.style.flexDirection = FlexDirection.Row;
            podiumButtons.style.justifyContent = Justify.Center;
            podiumButtons.style.marginTop = UiTheme.SPACE_SM;
            podiumButtons.style.flexShrink = 0f;
            var raceAgainButton = new Button(() => OnResultsAction(_spawn.RaceAgain)) { text = "RACE AGAIN" };
            raceAgainButton.style.height = UiTheme.CONTROL_SM;
            raceAgainButton.style.fontSize = UiTheme.FONT_SM;
            raceAgainButton.style.flexGrow = 2f;
            raceAgainButton.style.flexBasis = 0f;
            UiTheme.SetMargin(raceAgainButton, 0f, 0f);
            // Primary action carries the accent; MENU is the quiet way out.
            UiTheme.StyleButton(raceAgainButton, accent: true);
            UiTheme.AddHover(raceAgainButton, accent: true);
            podiumButtons.Add(raceAgainButton);
            var backToMenuButton = new Button(() => OnResultsAction(_spawn.RequestMenu)) { text = "MENU" };
            backToMenuButton.style.height = UiTheme.CONTROL_SM;
            backToMenuButton.style.fontSize = UiTheme.FONT_SM;
            backToMenuButton.style.flexGrow = 1f;
            backToMenuButton.style.flexBasis = 0f;
            UiTheme.SetMargin(backToMenuButton, 0f, 0f);
            backToMenuButton.style.marginLeft = UiTheme.SPACE_SM;
            UiTheme.StyleButton(backToMenuButton);
            UiTheme.AddHover(backToMenuButton);
            podiumButtons.Add(backToMenuButton);
            _podiumPanel.Add(podiumButtons);

            safeRoot.Add(_podiumPanel);
            SelectResultsTab(TAB_PODIUM);
        }

        private VisualElement BuildPodiumPage()
        {
            var page = new VisualElement { pickingMode = PickingMode.Ignore };

            // Who won, always. The WINNER banner only shows while the race is still
            // running, so a race that ends on its first crossing (one racer) or on the
            // clock went straight to the sheet and never announced anyone.
            _winnerHeadline = MakeLabel(UiTheme.FONT_LG, UiTheme.Gold, bold: true);
            _winnerHeadline.style.whiteSpace = WhiteSpace.Normal;
            _winnerHeadline.style.marginBottom = UiTheme.SPACE_XXS;
            page.Add(_winnerHeadline);

            for (int podiumIndex = 0; podiumIndex < PODIUM_ROWS; podiumIndex++)
            {
                var row = new VisualElement { pickingMode = PickingMode.Ignore };
                row.style.flexDirection = FlexDirection.Row;
                row.style.alignItems = Align.Center;
                row.style.marginTop = UiTheme.SPACE_XXS;
                row.style.minHeight = UiTheme.SPACE_XL;
                VisualElement medal = UiTheme.MakeSwatch(MedalColors[podiumIndex], UiTheme.SPACE_MD);
                medal.style.marginRight = UiTheme.SPACE_SM;
                _podiumMedals[podiumIndex] = medal;
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
                page.Add(row);
            }

            // Everyone else. Racers still running when the podium filled used to
            // vanish from the results entirely (and read "DNF" in the logs); they are
            // unplaced, not failed, and a viewer who picked one wants to see it.
            _alsoRanLabel = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false);
            _alsoRanLabel.style.whiteSpace = WhiteSpace.Normal;
            _alsoRanLabel.style.marginTop = UiTheme.SPACE_SM;
            _alsoRanLabel.enableRichText = true;
            page.Add(_alsoRanLabel);
            return page;
        }

        /// <summary>
        /// How the medallists got there: average speed, mechanical cost of transport,
        /// joint work spent and how twitchy the brain was — the telemetry card's
        /// numbers, kept after the flag for the three racers worth comparing.
        /// </summary>
        private VisualElement BuildStatsPage()
        {
            var page = new VisualElement { pickingMode = PickingMode.Ignore };
            VisualElement header = StatsRow();
            header.Add(StatsName(string.Empty, UiTheme.TextDim));
            header.Add(StatCell("M/S", UiTheme.TextDim, bold: true));
            header.Add(StatCell("COST", UiTheme.TextDim, bold: true));
            header.Add(StatCell("KJ", UiTheme.TextDim, bold: true));
            header.Add(StatCell("TWITCH", UiTheme.TextDim, bold: true));
            page.Add(header);
            for (int rowIndex = 0; rowIndex < PODIUM_ROWS; rowIndex++)
            {
                VisualElement row = StatsRow();
                row.style.marginTop = UiTheme.SPACE_XXS;
                VisualElement medal = UiTheme.MakeSwatch(MedalColors[rowIndex], UiTheme.SPACE_SM);
                medal.style.marginRight = UiTheme.SPACE_XS;
                _statsMedals[rowIndex] = medal;
                row.Add(medal);
                _statsNames[rowIndex] = StatsName(string.Empty, UiTheme.Text);
                row.Add(_statsNames[rowIndex]);
                _statsSpeed[rowIndex] = StatCell("—", UiTheme.Text, bold: true);
                row.Add(_statsSpeed[rowIndex]);
                _statsCost[rowIndex] = StatCell("—", UiTheme.Text, bold: false);
                row.Add(_statsCost[rowIndex]);
                _statsEnergy[rowIndex] = StatCell("—", UiTheme.Text, bold: false);
                row.Add(_statsEnergy[rowIndex]);
                _statsTwitch[rowIndex] = StatCell("—", UiTheme.Text, bold: false);
                row.Add(_statsTwitch[rowIndex]);
                _statsRows[rowIndex] = row;
                page.Add(row);
            }
            Label footnote = MakeLabel(UiTheme.FONT_XS, UiTheme.TextDim, bold: false,
                "Lower cost and twitch = smoother, cheaper gait.");
            footnote.style.whiteSpace = WhiteSpace.Normal;
            footnote.style.marginTop = UiTheme.SPACE_XS;
            page.Add(footnote);
            return page;
        }

        private void SelectResultsTab(int tab)
        {
            _resultsTab = tab;
            UiTheme.SelectTab(_resultsTabs, tab);
            for (int pageIndex = 0; pageIndex < _resultsPages.Length; pageIndex++)
            {
                _resultsPages[pageIndex].style.display = pageIndex == tab ? DisplayStyle.Flex : DisplayStyle.None;
            }
        }

        /// <summary>
        /// Thin vertical rail on the right edge: one pooled dot per leading racer,
        /// bottom (start line) to top (finish line). The three leaders get a place
        /// badge instead of a dot, drawn over the pack so first place is never hidden.
        /// </summary>
        private void BuildProgressRail(VisualElement safeRoot)
        {
            _rail = new VisualElement { name = UiTheme.PROGRESS_RAIL, pickingMode = PickingMode.Ignore };
            _rail.style.position = Position.Absolute;
            _rail.style.right = UiTheme.SPACE_MD;
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

            // Added last and in reverse, so later siblings draw on top: when the
            // leaders are neck and neck, 1 sits over 2 sits over 3.
            for (int badgeIndex = BADGE_COUNT - 1; badgeIndex >= 0; badgeIndex--)
            {
                var badge = new VisualElement { pickingMode = PickingMode.Ignore };
                badge.style.position = Position.Absolute;
                badge.style.left = (RAIL_WIDTH - RAIL_BADGE_SIZE) * 0.5f;
                badge.style.width = RAIL_BADGE_SIZE;
                badge.style.height = RAIL_BADGE_SIZE;
                badge.style.justifyContent = Justify.Center;
                badge.style.alignItems = Align.Center;
                // Dark plate ringed in the racer's own tint: the tint keeps who it is,
                // the plate keeps the digit readable on any tint.
                badge.style.backgroundColor = UiTheme.ModalBg;
                UiTheme.SetRadius(badge, RAIL_BADGE_SIZE * 0.5f);
                UiTheme.SetBorder(badge, UiTheme.TextDim, RAIL_BADGE_BORDER);
                badge.style.display = DisplayStyle.None;
                var place = new Label((badgeIndex + 1).ToString()) { pickingMode = PickingMode.Ignore };
                place.style.color = MedalColors[badgeIndex];
                place.style.fontSize = UiTheme.FONT_XS;
                place.style.unityFontStyleAndWeight = FontStyle.Bold;
                UiTheme.SetPadding(place, 0f, 0f);
                UiTheme.SetMargin(place, 0f, 0f);
                badge.Add(place);
                _railBadges[badgeIndex] = badge;
                _railBadgeTints[badgeIndex] = UiTheme.TextDim;
                _rail.Add(badge);
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
                if (_leaveSheetOpen)
                {
                    CloseLeaveSheet();
                }
                return;
            }
            _hudRoot.style.display = DisplayStyle.Flex;
            // The question is moot once the race has ended on its own.
            if (_leaveSheetOpen && !_raceModel.RaceActive && _raceModel.CountdownValue == 0)
            {
                CloseLeaveSheet();
            }

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
            else if (winner != null && _raceModel.RaceActive && Time.unscaledTime < _winnerBannerUntil)
            {
                // Brief celebration while the rest of the field is still running.
                // Once the race is over the podium says the same thing, so the
                // banner stands down rather than repeat it over the results sheet.
                //
                // Only a racer that crossed owns a finish time. A winner on the
                // clock is ranked on distance and its FinishTime was never set, so
                // printing it read "WINNER Crab #1 0.0s" on every timed race - which
                // is the normal outcome on the 600 s courses. Same split as
                // RefreshPodium below; keep the two in step.
                string winnerMetric = winner.Status == RacerStatus.Finished
                    ? $"{winner.FinishTime:0.0}s"
                    : $"{winner.Progress:0.0}m";
                // WINNER on its own line, so a long name wraps at a sensible point.
                _bannerLabel.text = $"WINNER\n{winner.DisplayName}  {winnerMetric}";
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
            int leaderCount = RefreshFieldWidgets();
            RefreshStatusChip(leaderCount);
        }

        /// <summary>
        /// Loading line before the grid exists, leader and clock while racing, nothing
        /// otherwise. Text is only rebuilt when the leader changes or the clock ticks
        /// over a whole second, so a race costs about one string a second.
        /// </summary>
        private void RefreshStatusChip(int leaderCount)
        {
            bool loading = !_raceModel.RaceActive && _raceModel.CountdownValue == 0
                && _raceModel.Racers.Count == 0;
            bool racing = _raceModel.RaceActive && leaderCount > 0;
            if (!loading && !racing)
            {
                _statusChip.style.display = DisplayStyle.None;
                _statusShowsLoading = false;
                _statusLeaderId = null;
                _statusWholeSeconds = -1;
                return;
            }
            _statusChip.style.display = DisplayStyle.Flex;
            if (loading)
            {
                if (!_statusShowsLoading)
                {
                    _statusShowsLoading = true;
                    _statusLeaderId = null;
                    _statusWholeSeconds = -1;
                    _statusSwatch.style.display = DisplayStyle.None;
                    _statusLabel.text = "Getting racers ready...";
                }
                return;
            }
            _statusShowsLoading = false;
            RacerState leader = _leaders[0];
            int wholeSeconds = Mathf.FloorToInt(_raceModel.ElapsedSeconds);
            if (leader.RacerId == _statusLeaderId && wholeSeconds == _statusWholeSeconds)
            {
                return;
            }
            if (leader.RacerId != _statusLeaderId)
            {
                _statusLeaderId = leader.RacerId;
                _statusSwatch.style.display = DisplayStyle.Flex;
                _statusSwatch.style.backgroundColor = leader.Tint;
            }
            _statusWholeSeconds = wholeSeconds;
            int limit = Mathf.CeilToInt(_raceModel.TimeLimitSeconds);
            _statusLabel.text = limit > 0
                ? $"LEAD  {leader.DisplayName}   {wholeSeconds / SECONDS_PER_MINUTE}:{wholeSeconds % SECONDS_PER_MINUTE:00} / {limit / SECONDS_PER_MINUTE}:{limit % SECONDS_PER_MINUTE:00}"
                : $"LEAD  {leader.DisplayName}   {wholeSeconds / SECONDS_PER_MINUTE}:{wholeSeconds % SECONDS_PER_MINUTE:00}";
        }

        /// <summary>
        /// Fills and plays the intro card. Independent of the banner elements, so
        /// the countdown and GO logic are untouched by it.
        /// </summary>
        private void ShowIntroCard()
        {
            _introRaceLabel.text = $"RACE {_raceModel.RaceNumber}";
            _introTrackLabel.text = _raceModel.TrackName;
            // The sky is rolled per race, so the card names the one this race got.
            SkyPreset sky = _skyModel != null ? _skyModel.Current : null;
            _introFieldLabel.text = sky != null
                ? $"{_raceModel.Racers.Count} RACERS  ·  {sky.DisplayName.ToUpperInvariant()}"
                : $"{_raceModel.Racers.Count} RACERS";
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
                // The results sheet owns the screen between races; the lane's
                // transient messages would only compete with it.
                _announceSlot.style.display = showPodium ? DisplayStyle.None : DisplayStyle.Flex;
                if (showPodium)
                {
                    // Live again: the last RACE AGAIN / MENU tap left it inert.
                    _podiumPanel.SetEnabled(true);
                    // Force one rebuild of every row: racer ids can repeat across
                    // races, so the change guards below must not carry over.
                    for (int rowIndex = 0; rowIndex < PODIUM_ROWS; rowIndex++)
                    {
                        _podiumSourceIds[rowIndex] = null;
                    }
                    SelectResultsTab(TAB_PODIUM);
                    PlayPodiumEntrance();
                }
            }
            if (!showPodium)
            {
                return;
            }

            // The league tab only exists once a race has been scored.
            bool leagueReady = _league != null && _league.RacesScored > 0;
            if (leagueReady != _leagueTabShown)
            {
                _leagueTabShown = leagueReady;
                _resultsTabs[TAB_LEAGUE].style.display = leagueReady ? DisplayStyle.Flex : DisplayStyle.None;
                if (!leagueReady && _resultsTab == TAB_LEAGUE)
                {
                    SelectResultsTab(TAB_PODIUM);
                }
            }

            bool anyRowChanged = false;
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
                _medalists[podiumIndex] = medalist;

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
                anyRowChanged = true;

                // A place the referee filled with a knocked-out racer (fewer than three
                // crossed) wears no medal: grey, not gold, beside a "DNF" metric.
                bool medalEarned = medalist != null && medalist.Status != RacerStatus.Dnf;
                Color medalColor = medalEarned ? MedalColors[podiumIndex] : UiTheme.Dnf;
                _podiumMedals[podiumIndex].style.backgroundColor = medalColor;
                _statsMedals[podiumIndex].style.backgroundColor = medalColor;

                if (medalist == null)
                {
                    _podiumLabels[podiumIndex].text = "—";
                    continue;
                }
                string timeText = MetricText(medalist);
                // Bare rating with a triangle for the swing, not "ELO 1216 +16": the
                // prefix restated what every roster row already teaches.
                if (delta == 0)
                {
                    _podiumLabels[podiumIndex].text =
                        $"{medalist.DisplayName}  {timeText}  {rating:0}";
                }
                else
                {
                    string deltaHex = delta > 0 ? DeltaUpHex : DeltaDownHex;
                    string arrow = delta > 0 ? "▲" : "▼";
                    int magnitude = delta > 0 ? delta : -delta;
                    _podiumLabels[podiumIndex].text =
                        $"{medalist.DisplayName}  {timeText}  {rating:0} "
                        + $"<color={deltaHex}>{arrow}{magnitude}</color>";
                }
            }
            if (anyRowChanged)
            {
                RefreshStats();
                RefreshWinnerHeadline();
                RefreshAlsoRan();
            }
        }

        /// <summary>Gold line over the podium naming the winner, or saying nobody made it.</summary>
        private void RefreshWinnerHeadline()
        {
            RacerState first = _medalists[0];
            if (first == null || first.Status == RacerStatus.Dnf)
            {
                _winnerHeadline.text = "NO FINISHERS";
                _winnerHeadline.style.color = UiTheme.TextDim;
                return;
            }
            _winnerHeadline.style.color = UiTheme.Gold;
            _winnerHeadline.text = first.Status == RacerStatus.Finished
                ? $"{first.DisplayName} WINS  ·  {first.FinishTime:0.0}s"
                // Ranked on distance: the clock ran out, or the race was skipped.
                : $"{first.DisplayName} WINS  ·  led on distance";
        }

        /// <summary>
        /// The rest of the field under the podium: anyone placed below third by
        /// place, then the unplaced by distance. Rebuilt once per results screen.
        /// </summary>
        private void RefreshAlsoRan()
        {
            _alsoRan.Clear();
            for (int racerIndex = 0; racerIndex < _raceModel.Racers.Count; racerIndex++)
            {
                RacerState racer = _raceModel.Racers[racerIndex];
                if (racer.Place < 1 || racer.Place > PODIUM_ROWS)
                {
                    _alsoRan.Add(racer);
                }
            }
            if (_alsoRan.Count == 0)
            {
                _alsoRanLabel.style.display = DisplayStyle.None;
                return;
            }
            _alsoRan.Sort(AlsoRanOrder);
            _alsoRanText.Clear();
            _alsoRanText.Append("<b>REST OF FIELD</b>\n");
            int listed = Mathf.Min(_alsoRan.Count, ALSO_RAN_MAX);
            for (int listIndex = 0; listIndex < listed; listIndex++)
            {
                RacerState racer = _alsoRan[listIndex];
                if (listIndex > 0)
                {
                    _alsoRanText.Append("  ·  ");
                }
                if (racer.Place > 0)
                {
                    _alsoRanText.Append(racer.Place).Append(". ");
                }
                _alsoRanText.Append(racer.DisplayName).Append(' ');
                _alsoRanText.Append(racer.Progress.ToString("0.0")).Append(" m");
                if (racer.Status == RacerStatus.Dnf)
                {
                    // Cut off by the podium is not a failure; anything else is.
                    _alsoRanText.Append(racer.Knockout == KnockoutReason.PodiumCutoff ? " (unplaced)" : " (out)");
                }
            }
            if (_alsoRan.Count > listed)
            {
                _alsoRanText.Append("  ·  +").Append(_alsoRan.Count - listed).Append(" more");
            }
            _alsoRanLabel.text = _alsoRanText.ToString();
            _alsoRanLabel.style.display = DisplayStyle.Flex;
        }

        /// <summary>
        /// Only a racer that crossed owns a finish time. One that ran out of clock is
        /// ranked on distance, so show the distance; one that was knocked out shows DNF.
        /// </summary>
        private static string MetricText(RacerState racer)
        {
            if (racer.Status == RacerStatus.Finished)
            {
                return $"{racer.FinishTime:0.0}s";
            }
            if (racer.Status == RacerStatus.TimedOut)
            {
                return $"{racer.Progress:0.0}m";
            }
            // Distance too: it is what ranked a knocked-out racer onto the podium.
            return $"DNF {racer.Progress:0.0}m";
        }

        /// <summary>
        /// Fills the STATS page from the telemetry the race left behind. Runs only when
        /// a podium row changed, so a results screen left open rebuilds nothing.
        /// </summary>
        private void RefreshStats()
        {
            for (int rowIndex = 0; rowIndex < PODIUM_ROWS; rowIndex++)
            {
                RacerState medalist = _medalists[rowIndex];
                RacerTelemetry telemetry = medalist != null && _telemetryModel != null
                    ? _telemetryModel.Find(medalist.RacerId)
                    : null;
                _statsRows[rowIndex].style.display = medalist != null ? DisplayStyle.Flex : DisplayStyle.None;
                if (medalist == null)
                {
                    continue;
                }
                _statsNames[rowIndex].text = medalist.DisplayName;
                float seconds = medalist.Status == RacerStatus.Finished ? medalist.FinishTime : _raceModel.ElapsedSeconds;
                _statsSpeed[rowIndex].text = seconds > 0.01f ? (medalist.Progress / seconds).ToString("0.00") : "—";
                if (telemetry == null)
                {
                    _statsCost[rowIndex].text = "—";
                    _statsEnergy[rowIndex].text = "—";
                    _statsTwitch[rowIndex].text = "—";
                    continue;
                }
                float cost = telemetry.CostOfTransport;
                _statsCost[rowIndex].text = cost < 0f ? "—" : cost.ToString("0.00");
                _statsEnergy[rowIndex].text = telemetry.HasPower ? (telemetry.EnergyJoules / 1000f).ToString("0.0") : "—";
                _statsTwitch[rowIndex].text = telemetry.ActionCount > 0
                    ? Mathf.RoundToInt(telemetry.Twitchiness * 100f) + "%"
                    : "—";
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

        /// <summary>Drives the right-edge rail and its leader badges; returns how many leaders it ranked.</summary>
        private int RefreshFieldWidgets()
        {
            int leaderCount = SelectLeaders();
            bool show = leaderCount > 0;
            if (show != _widgetsVisible)
            {
                _widgetsVisible = show;
                _rail.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
            }
            if (show)
            {
                RefreshRail(leaderCount);
            }
            return leaderCount;
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
                    // The leaders are drawn as badges, so their plain dots stay hidden.
                    bool dotShown = dotIndex >= BADGE_COUNT && dotIndex < leaderCount;
                    _railDots[dotIndex].style.display = dotShown ? DisplayStyle.Flex : DisplayStyle.None;
                }
                for (int badgeIndex = 0; badgeIndex < BADGE_COUNT; badgeIndex++)
                {
                    _railBadges[badgeIndex].style.display =
                        badgeIndex < leaderCount ? DisplayStyle.Flex : DisplayStyle.None;
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
                var top = new Length((1f - fraction) * RAIL_SPAN_PERCENT, LengthUnit.Percent);
                Color tint = racer.Status == RacerStatus.Dnf ? UiTheme.Dnf : racer.Tint;
                if (dotIndex < BADGE_COUNT)
                {
                    _railBadges[dotIndex].style.top = top;
                    if (_railBadgeTints[dotIndex] != tint)
                    {
                        _railBadgeTints[dotIndex] = tint;
                        UiTheme.SetBorder(_railBadges[dotIndex], tint, RAIL_BADGE_BORDER);
                    }
                    continue;
                }
                _railDots[dotIndex].style.top = top;
                if (_railDotTints[dotIndex] != tint)
                {
                    _railDotTints[dotIndex] = tint;
                    _railDots[dotIndex].style.backgroundColor = tint;
                }
            }
        }

        private static void Ellipsize(Label label)
        {
            label.style.whiteSpace = WhiteSpace.NoWrap;
            label.style.overflow = Overflow.Hidden;
            label.style.textOverflow = TextOverflow.Ellipsis;
            label.style.flexShrink = 1f;
            label.style.minWidth = 0f;
        }

        private static VisualElement StatsRow()
        {
            var row = new VisualElement { pickingMode = PickingMode.Ignore };
            row.style.flexDirection = FlexDirection.Row;
            row.style.alignItems = Align.Center;
            return row;
        }

        private static Label StatsName(string text, Color color)
        {
            Label name = MakeLabel(UiTheme.FONT_XS, color, bold: false, text);
            name.style.flexGrow = 1f;
            Ellipsize(name);
            return name;
        }

        private static Label StatCell(string text, Color color, bool bold)
        {
            Label cell = MakeLabel(UiTheme.FONT_XS, color, bold, text);
            cell.style.width = UiTheme.ScaleWithFont(STAT_COLUMN_WIDTH);
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
