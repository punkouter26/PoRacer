using PoRacer.Presentation;
using System.Collections.Generic;
using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Start menu, UI Toolkit hierarchy built in C#. One portrait screen, no second
    /// step: the maps as a row of tabs, the count presets, then one line per catalog
    /// slot with the counts (0/1/10/50/100) as a segmented control, and RACE pinned
    /// above the bottom furniture. Slots without a trained brain are omitted and
    /// counted in the footer as "soon". Start hands off to Systems_Spawn.
    ///
    /// The map picker used to be a screen of its own with a NEXT button, and the
    /// roster screen carried a back button to it. With three maps a tab row says the
    /// same thing in one control height, and removes a page, a button and a round trip.
    ///
    /// Density. The whole roster is meant to be visible at once on a phone - the
    /// list keeps its ScrollView as a safety net for short screens and future
    /// creatures, but on a normal handset nothing should need scrolling (the smoke
    /// run's ui-audit flags it if anything does). That budget is what every fixed
    /// number below is protecting, so before adding a block here, check what it
    /// costs against <see cref="UiTheme.CONTROL_SM"/> times the roster size. The
    /// standings table that used to sit above the list was removed on 2026-09-03 for
    /// exactly this reason; the ranking it conveyed is carried by sorting the rows by
    /// rating instead.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class MenuView : MonoBehaviour
    {
        private const float AVATAR_SIZE = 24f;
        // Past this many racers the frame rate on a phone starts to give.
        private const int LARGE_FIELD = 100;
        // Avatar hues skip the two bands the racer legend owns (AGENTS.md rule D):
        // red for heuristic bots, green for the baseline RL policy. What is left is
        // orange-yellow [0.06, 0.20] and cyan-through-magenta [0.47, 0.92].
        private const float HUE_WARM_START = 0.06f;
        private const float HUE_WARM_SPAN = 0.14f;
        private const float HUE_COOL_START = 0.47f;
        private const float HUE_COOL_SPAN = 0.45f;

        /// <summary>Horizontal room a bottom corner anchor (DBG, version) takes from the band.</summary>
        private static float CornerClearance => UiTheme.CONTROL_LG + UiTheme.SPACE_LG;

        private CreatureCatalog _catalog;
        private RaceConfigModel _config;
        private EloModel _eloModel;
        private Systems_Spawn _spawn;
        private HapticsModel _haptics;
        private Systems_Haptics _hapticsSystem;
        private Button _hapticsButton;
        private VisualElement _optionsSheet;
        private bool _optionsOpen;
        private VisualElement _root;
        private VisualElement _content;
        private Label _totalLabel;
        private Button _startButton;
        private Button[] _mapTabs;
        private Label[] _mapTabNames;
        private Label[] _mapTabLengths;
        private VisualElement _creatureList;
        private int _comingSoonCount;
        // Per-row ELO labels, refreshed when the menu re-shows after a race.
        private readonly List<Label> _ratingLabels = new();
        private readonly List<string> _ratingCreatureIds = new();
        private readonly List<VisualElement> _rowElements = new();
        private readonly List<CreatureCatalog.CreatureEntry> _ranked = new();
        private bool _wasVisible;
        // Entrance stagger plays once; menu rebuilds (preset) must not re-animate
        // the whole screen under the user's finger.
        private bool _playEntrance = true;

        [Inject]
        public void Construct(CreatureCatalog catalog, RaceConfigModel config, EloModel eloModel, Systems_Spawn spawn,
            HapticsModel haptics, Systems_Haptics hapticsSystem)
        {
            _catalog = catalog;
            _config = config;
            _eloModel = eloModel;
            _spawn = spawn;
            _haptics = haptics;
            _hapticsSystem = hapticsSystem;
        }

        private void Start()
        {
            _root = GetComponent<UIDocument>().rootVisualElement;
            BuildMenu();
            _config.Changed += OnConfigChanged;
            _haptics.Changed += RefreshHapticsButton;
            OnConfigChanged();
        }

        private void OnDestroy()
        {
            if (_config != null)
            {
                _config.Changed -= OnConfigChanged;
            }
            if (_haptics != null)
            {
                _haptics.Changed -= RefreshHapticsButton;
            }
        }

        private void OnConfigChanged()
        {
            _root.style.display = _config.MenuVisible ? DisplayStyle.Flex : DisplayStyle.None;
            if (_config.MenuVisible && !_wasVisible)
            {
                // Menu fades up instead of snapping in.
                _root.style.opacity = 0f;
                _root.experimental.animation.Start(0f, 1f, UiTheme.FADE_MS, (element, value) =>
                {
                    element.style.opacity = value;
                });
                // Ratings moved while the race ran; re-read them on the way back in
                // instead of polling them every frame.
                RefreshRatings();
            }
            _wasVisible = _config.MenuVisible;
            RefreshTotals();
            if (_startButton != null)
            {
                // A disabled start button explains itself; a silent no-op does not.
                bool hasRacers = _config.TotalCount() > 0;
                _startButton.SetEnabled(hasRacers);
                _startButton.text = hasRacers ? "RACE" : "PICK RACERS";
            }
        }

        /// <summary>
        /// Footer counter, on the furniture line. Carries the "soon" tally and the
        /// large-field warning as short clauses rather than lines of their own.
        /// </summary>
        private void RefreshTotals()
        {
            if (_totalLabel == null)
            {
                return;
            }
            int total = _config.TotalCount();
            string text = $"{total} racers";
            if (_comingSoonCount > 0)
            {
                text += $" · +{_comingSoonCount} soon";
            }
            if (total > LARGE_FIELD)
            {
                text += " · may lag";
            }
            _totalLabel.text = text;
        }

        private void BuildMenu()
        {
            // The whole hierarchy is rebuilt on a preset, so the element caches must
            // not keep pointing at discarded elements.
            _ratingLabels.Clear();
            _ratingCreatureIds.Clear();
            _rowElements.Clear();
            _root.style.backgroundColor = UiTheme.ScreenBg;
            VisualElement safe = UiTheme.BuildSafeRoot(_root);

            // Screen content lives in its own padded layer between the two furniture
            // bands; the furniture is pinned to the unpadded safe root beside it, so
            // no anchor depends on how absolute children treat a parent's padding.
            _content = new VisualElement { pickingMode = PickingMode.Position };
            _content.style.position = Position.Absolute;
            _content.style.left = 0f;
            _content.style.top = 0f;
            _content.style.right = 0f;
            _content.style.bottom = 0f;
            _content.style.paddingTop = UiTheme.TopBand;
            _content.style.paddingLeft = UiTheme.SPACE_MD;
            _content.style.paddingRight = UiTheme.SPACE_MD;
            _content.style.paddingBottom = UiTheme.BottomBand;
            safe.Add(_content);

            BuildFurniture(safe);
            BuildMapTabs();
            BuildPresetRow();

            // The roster is sized to fit without scrolling on a phone; the
            // ScrollView is the safety net for short screens and a growing
            // catalog, not the expected reading mode.
            var creatureScroll = new ScrollView(ScrollViewMode.Vertical);
            creatureScroll.style.flexGrow = 1f;
            creatureScroll.style.flexShrink = 1f;
            UiTheme.StyleScrollView(creatureScroll);
            _content.Add(creatureScroll);
            _creatureList = creatureScroll.contentContainer;

            // Rows are ordered by rating, strongest first. That ordering is what
            // carries the standings information now that the separate ELO table
            // is gone, so it has to survive every rebuild.
            _ranked.Clear();
            _comingSoonCount = 0;
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (Systems_MujocoWorld.CanRace(entry))
                {
                    _ranked.Add(entry);
                }
                else
                {
                    _comingSoonCount++;
                }
            }
            _ranked.Sort((first, second) =>
                _eloModel.GetRating(second.id).CompareTo(_eloModel.GetRating(first.id)));
            for (int rankIndex = 0; rankIndex < _ranked.Count; rankIndex++)
            {
                _creatureList.Add(BuildRow(_ranked[rankIndex]));
            }

            BuildFooter();
            RefreshTotals();

            if (_playEntrance)
            {
                PlayEntrance();
                _playEntrance = false;
            }
        }

        /// <summary>
        /// The corner anchors every screen shares, plus the racer count centred on the
        /// bottom band. That strip is reserved for DBG and the version anyway, so a
        /// line of text between them costs the roster nothing.
        /// </summary>
        private void BuildFurniture(VisualElement safe)
        {
            safe.Add(UiTheme.MakeTitleFurniture());
            safe.Add(UiTheme.MakeVersionFurniture());

            _totalLabel = new Label { pickingMode = PickingMode.Ignore };
            _totalLabel.style.position = Position.Absolute;
            // Clear of DBG on the left and the version on the right. Symmetric, so the
            // text stays centred; sized for the wider of the two (the version text).
            _totalLabel.style.left = CornerClearance;
            _totalLabel.style.right = CornerClearance;
            _totalLabel.style.bottom = UiTheme.SPACE_SM;
            _totalLabel.style.height = UiTheme.CONTROL_SM;
            _totalLabel.style.color = UiTheme.AccentSoft;
            _totalLabel.style.fontSize = UiTheme.FONT_XS;
            _totalLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _totalLabel.style.overflow = Overflow.Hidden;
            _totalLabel.style.textOverflow = TextOverflow.Ellipsis;
            safe.Add(_totalLabel);

            // MENU opens the options sheet. The only option is vibration, so the
            // button exists only where the device can vibrate.
            if (Application.platform == RuntimePlatform.Android || Application.isEditor)
            {
                safe.Add(UiTheme.MakeMenuFurniture(ToggleOptions));
                BuildOptionsSheet(safe);
            }
        }

        /// <summary>Small sheet under MENU holding the settings that used to crowd the title row.</summary>
        private void BuildOptionsSheet(VisualElement safe)
        {
            _optionsSheet = new VisualElement();
            _optionsSheet.style.position = Position.Absolute;
            _optionsSheet.style.top = UiTheme.TopBand;
            _optionsSheet.style.right = UiTheme.SPACE_SM;
            _optionsSheet.style.flexDirection = FlexDirection.Row;
            _optionsSheet.style.alignItems = Align.Center;
            UiTheme.StyleModal(_optionsSheet);
            _optionsSheet.style.display = DisplayStyle.None;
            _optionsOpen = false;
            safe.Add(_optionsSheet);

            var caption = new Label("Vibration") { pickingMode = PickingMode.Ignore };
            caption.style.color = UiTheme.Text;
            caption.style.fontSize = UiTheme.FONT_SM;
            caption.style.marginRight = UiTheme.SPACE_SM;
            UiTheme.ApplyFont(caption);
            _optionsSheet.Add(caption);

            _hapticsButton = new Button(_hapticsSystem.Toggle);
            _hapticsButton.style.height = UiTheme.CONTROL_SM;
            _hapticsButton.style.minWidth = UiTheme.CONTROL_SM;
            _hapticsButton.style.fontSize = UiTheme.FONT_SM;
            UiTheme.SetMargin(_hapticsButton, 0f, 0f);
            UiTheme.StyleButton(_hapticsButton);
            UiTheme.AddHover(_hapticsButton);
            _optionsSheet.Add(_hapticsButton);
            RefreshHapticsButton();
        }

        private void ToggleOptions()
        {
            _optionsOpen = !_optionsOpen;
            _optionsSheet.style.display = _optionsOpen ? DisplayStyle.Flex : DisplayStyle.None;
            if (_optionsOpen)
            {
                UiTheme.PlayEnter(_optionsSheet, 0);
            }
        }

        private void RefreshHapticsButton()
        {
            if (_hapticsButton == null)
            {
                return;
            }
            _hapticsButton.text = _haptics.Enabled ? "ON" : "OFF";
            _hapticsButton.style.color = _haptics.Enabled ? UiTheme.Text : UiTheme.TextDim;
        }

        /// <summary>
        /// Every playable map as one tab: name over length. Tapping selects; there is
        /// no separate confirm, RACE uses whichever tab is lit.
        /// </summary>
        private void BuildMapTabs()
        {
            int mapCount = Systems_MapCatalog.Entries.Count;
            _mapTabs = new Button[mapCount];
            _mapTabNames = new Label[mapCount];
            _mapTabLengths = new Label[mapCount];

            var strip = new VisualElement();
            UiTheme.StyleSegmentGroup(strip);
            strip.style.flexShrink = 0f;
            strip.style.marginBottom = UiTheme.SPACE_XS;
            _content.Add(strip);

            for (int mapIndex = 0; mapIndex < mapCount; mapIndex++)
            {
                Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Entries[mapIndex];
                if (!map.Available)
                {
                    continue;
                }
                int capturedIndex = mapIndex;
                var tab = new Button(() =>
                {
                    _config.SetMap(capturedIndex);
                    RefreshMapTabs();
                });
                tab.style.flexDirection = FlexDirection.Column;
                tab.style.justifyContent = Justify.Center;
                tab.style.alignItems = Align.Center;

                var name = new Label(map.DisplayName) { pickingMode = PickingMode.Ignore };
                name.style.fontSize = UiTheme.FONT_SM;
                name.style.unityFontStyleAndWeight = FontStyle.Bold;
                name.style.overflow = Overflow.Hidden;
                name.style.textOverflow = TextOverflow.Ellipsis;
                // Two lines in one CONTROL_SM cell leave no room for the labels'
                // default padding: with it the pair stood 3 dp taller than the tab and
                // "Flat" poked out over the title (smoke run ui-audit, 2026-09-25).
                UiTheme.SetPadding(name, 0f, 0f);
                UiTheme.SetMargin(name, 0f, 0f);
                UiTheme.ApplyFont(name);
                tab.Add(name);

                var length = new Label($"{map.LengthMeters:0} m") { pickingMode = PickingMode.Ignore };
                length.style.fontSize = UiTheme.FONT_XS;
                UiTheme.SetPadding(length, 0f, 0f);
                UiTheme.SetMargin(length, 0f, 0f);
                UiTheme.ApplyFont(length);
                tab.Add(length);

                _mapTabs[mapIndex] = tab;
                _mapTabNames[mapIndex] = name;
                _mapTabLengths[mapIndex] = length;
                strip.Add(tab);
            }

            RefreshMapTabs();
        }

        private void RefreshMapTabs()
        {
            for (int mapIndex = 0; mapIndex < _mapTabs.Length; mapIndex++)
            {
                if (_mapTabs[mapIndex] == null)
                {
                    continue;
                }
                bool isSelected = mapIndex == _config.SelectedMapIndex;
                UiTheme.StyleSegment(_mapTabs[mapIndex], isSelected);
                _mapTabNames[mapIndex].style.color = isSelected ? Color.white : UiTheme.Text;
                _mapTabLengths[mapIndex].style.color = isSelected ? Color.white : UiTheme.TextDim;
            }
        }

        /// <summary>
        /// Re-reads ELO for every roster row, then restores the rating order the
        /// rows carry now that there is no separate standings table.
        /// </summary>
        private void RefreshRatings()
        {
            for (int labelIndex = 0; labelIndex < _ratingLabels.Count; labelIndex++)
            {
                // Same bare-number format BuildRow uses; re-adding the "ELO " prefix
                // here would put the truncation back the first time a race finished.
                _ratingLabels[labelIndex].text = $"{_eloModel.GetRating(_ratingCreatureIds[labelIndex]):0}";
            }
            ReorderRowsByRating();
        }

        /// <summary>
        /// Sorts the existing row elements by current rating and re-seats them in
        /// that order. Re-adding a child that already has this parent moves it to
        /// the end, so walking the sorted order rebuilds the list in place - no
        /// element is destroyed, so the count buttons keep their handlers.
        /// </summary>
        private void ReorderRowsByRating()
        {
            if (_creatureList == null || _rowElements.Count == 0)
            {
                return;
            }
            var order = new List<int>(_rowElements.Count);
            for (int rowIndex = 0; rowIndex < _rowElements.Count; rowIndex++)
            {
                order.Add(rowIndex);
            }
            order.Sort((first, second) => _eloModel.GetRating(_ratingCreatureIds[second])
                .CompareTo(_eloModel.GetRating(_ratingCreatureIds[first])));
            for (int position = 0; position < order.Count; position++)
            {
                _creatureList.Add(_rowElements[order[position]]);
            }
        }

        /// <summary>One-tap field setup instead of nine separate count rows.</summary>
        private void BuildPresetRow()
        {
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.flexShrink = 0f;
            row.style.marginBottom = UiTheme.SPACE_XS;
            string[] labels = { "All x1", "All x10", "Clear" };
            int[] counts = { 1, 10, 0 };
            for (int presetIndex = 0; presetIndex < labels.Length; presetIndex++)
            {
                int count = counts[presetIndex];
                var button = new Button(() => ApplyPreset(count)) { text = labels[presetIndex] };
                button.style.height = UiTheme.CONTROL_SM;
                button.style.fontSize = UiTheme.FONT_SM;
                button.style.flexGrow = 1f;
                button.style.flexBasis = 0f;
                UiTheme.SetMargin(button, 0f, UiTheme.SPACE_XS * 0.5f);
                UiTheme.StyleButton(button);
                UiTheme.AddHover(button);
                row.Add(button);
            }
            _content.Add(row);
        }

        private void BuildFooter()
        {
            var footer = new VisualElement();
            footer.style.flexShrink = 0f;
            footer.style.marginTop = UiTheme.SPACE_XXS;
            UiTheme.StyleGlassPanel(footer, glowing: true);
            // The padding baked into the glass helper is sized for a content panel;
            // the footer holds a single button and does not need that inset.
            UiTheme.SetPadding(footer, UiTheme.SPACE_XS, UiTheme.SPACE_SM);
            _content.Add(footer);

            _startButton = new Button(() => _spawn.BeginRacing()) { text = "RACE" };
            // The touch minimum, not CONTROL_MD: the 8 dp difference is a roster row's
            // worth of breathing room over ten rows, and the accent fill already makes
            // this the loudest thing on the screen.
            _startButton.style.height = UiTheme.CONTROL_SM;
            _startButton.style.fontSize = UiTheme.FONT_LG;
            UiTheme.SetMargin(_startButton, 0f, 0f);
            UiTheme.StyleButton(_startButton, accent: true);
            UiTheme.SetRadius(_startButton, UiTheme.RADIUS_LG);
            UiTheme.AddHover(_startButton, accent: true);
            footer.Add(_startButton);
        }

        /// <summary>Staggered fade-and-rise for the top-level blocks.</summary>
        private void PlayEntrance()
        {
            int childCount = _content.childCount;
            int delay = 0;
            for (int childIndex = 0; childIndex < childCount; childIndex++)
            {
                UiTheme.PlayEnter(_content[childIndex], delay);
                delay += 40;
            }
        }

        private void ApplyPreset(int count)
        {
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (Systems_MujocoWorld.CanRace(entry))
                {
                    _config.SetCount(entry.id, count);
                }
            }
            // Count buttons highlight from config state; rebuild to reflect it.
            _root.Clear();
            BuildMenu();
            _config.NotifyChanged();
        }

        private VisualElement BuildRow(CreatureCatalog.CreatureEntry entry)
        {
            // One line per creature: avatar | name + ELO | count segments. The row
            // is pared down to the touch target it contains - the count buttons are
            // CONTROL_SM tall and everything else fits inside that, so a row costs
            // the Android minimum and not a pixel more.
            var card = new VisualElement();
            card.style.marginBottom = 1f;
            UiTheme.StyleCard(card, selected: false);
            // The 1 px card border cost 2 px a row and reads as noise at this
            // density; the fill on the row already separates it from the screen.
            UiTheme.SetBorder(card, Color.clear, 0f);
            UiTheme.SetPadding(card, 1f, UiTheme.SPACE_XS);
            card.style.flexDirection = FlexDirection.Row;
            card.style.alignItems = Align.Center;

            var titleRow = new VisualElement();
            titleRow.style.flexDirection = FlexDirection.Row;
            titleRow.style.alignItems = Align.Center;
            titleRow.style.flexGrow = 1f;
            titleRow.style.flexShrink = 1f;
            titleRow.style.minWidth = 0f;
            card.Add(titleRow);

            // Round avatar chip: creature initial on a per-creature hue, so rows
            // read as cards even without portrait art.
            var avatar = new VisualElement { pickingMode = PickingMode.Ignore };
            avatar.style.width = AVATAR_SIZE;
            avatar.style.height = AVATAR_SIZE;
            avatar.style.marginRight = UiTheme.SPACE_XS;
            avatar.style.justifyContent = Justify.Center;
            avatar.style.alignItems = Align.Center;
            avatar.style.flexShrink = 0f;
            avatar.style.backgroundColor = Color.HSVToRGB(AvatarHue(entry.id), 0.55f, 0.75f);
            UiTheme.SetRadius(avatar, AVATAR_SIZE * 0.5f);
            var initial = new Label(entry.displayName.Substring(0, 1));
            initial.style.color = Color.white;
            initial.style.fontSize = UiTheme.FONT_XS;
            initial.style.unityFontStyleAndWeight = FontStyle.Bold;
            avatar.Add(initial);
            titleRow.Add(avatar);

            // Name and ELO on one line: a second text line per row is what pushed the
            // eighth creature under the START panel on 9:20 phones.
            var nameColumn = new VisualElement();
            nameColumn.style.flexDirection = FlexDirection.Row;
            nameColumn.style.alignItems = Align.Center;
            nameColumn.style.flexGrow = 1f;
            nameColumn.style.flexShrink = 1f;
            nameColumn.style.minWidth = 0f;
            // Clip rather than spill: without this, an over-long row pushes its ELO
            // out over the count buttons instead of being cut at the card edge.
            nameColumn.style.overflow = Overflow.Hidden;
            titleRow.Add(nameColumn);

            var name = new Label(entry.displayName);
            name.style.color = UiTheme.Text;
            name.style.fontSize = UiTheme.FONT_SM;
            name.style.overflow = Overflow.Hidden;
            name.style.textOverflow = TextOverflow.Ellipsis;
            // UI Toolkit defaults flex-shrink to 0, NOT to 1 as web CSS does. Without
            // these two lines the ellipsis never triggers: the name keeps its full
            // width, the row overflows, and it is the ELO label beside it that gets
            // cut off ("Quadruped ELO 121"). Verified on device.
            name.style.flexShrink = 1f;
            name.style.minWidth = 0f;
            nameColumn.Add(name);

            // Bare number, not "ELO 1201". The four characters the prefix costs are
            // what pushed "Quadruped" and "Isaac Spider" into an ellipsis on a
            // 427 dp phone, and the rows are ordered by this number anyway.
            var eloLabel = new Label($"{_eloModel.GetRating(entry.id):0}");
            eloLabel.style.color = UiTheme.TextDim;
            eloLabel.style.fontSize = UiTheme.FONT_XS;
            eloLabel.style.marginLeft = UiTheme.SPACE_XS;
            eloLabel.style.flexShrink = 0f;
            nameColumn.Add(eloLabel);
            _ratingLabels.Add(eloLabel);
            _ratingCreatureIds.Add(entry.id);
            _rowElements.Add(card);

            int[] options = RaceConfigModel.COUNT_OPTIONS;
            var segments = new VisualElement();
            UiTheme.StyleSegmentGroup(segments);
            segments.style.width = Length.Percent(56f);
            // 56% only holds four 48 dp cells at the reference width. CONTROL_SM
            // grows on narrower handsets to keep the touch target, and without this
            // floor the last cell ("100") was pushed clean off the right edge -
            // measured on the simulator device. The name column gives way instead;
            // it already ellipsizes.
            segments.style.minWidth = options.Length * (UiTheme.CONTROL_SM + 2f);
            segments.style.flexShrink = 0f;
            card.Add(segments);

            var countButtons = new Button[options.Length];
            for (int optionIndex = 0; optionIndex < options.Length; optionIndex++)
            {
                int count = options[optionIndex];
                var button = new Button(() =>
                {
                    _config.SetCount(entry.id, count);
                    RefreshRowButtons(countButtons, options, entry.id);
                }) { text = count.ToString() };
                UiTheme.StyleSegment(button, selected: false);
                countButtons[optionIndex] = button;
                segments.Add(button);
            }
            RefreshRowButtons(countButtons, options, entry.id);
            return card;
        }

        /// <summary>
        /// A stable per-creature hue that is never red or green. The hash spreads over
        /// the allowed bands only, so a chip can no longer land on a legend colour and
        /// read as "this one is a heuristic bot" or "this is the baseline policy".
        /// </summary>
        private static float AvatarHue(string creatureId)
        {
            float fraction = (creatureId.GetHashCode() & 255) / 255f;
            float along = fraction * (HUE_WARM_SPAN + HUE_COOL_SPAN);
            return along < HUE_WARM_SPAN
                ? HUE_WARM_START + along
                : HUE_COOL_START + (along - HUE_WARM_SPAN);
        }

        private void RefreshRowButtons(Button[] buttons, int[] options, string creatureId)
        {
            int selected = _config.GetCount(creatureId);
            for (int buttonIndex = 0; buttonIndex < buttons.Length; buttonIndex++)
            {
                UiTheme.StyleSegment(buttons[buttonIndex], options[buttonIndex] == selected);
            }
        }
    }
}
