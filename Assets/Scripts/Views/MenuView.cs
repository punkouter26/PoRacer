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
    /// Start menu, UI Toolkit hierarchy built in C#. Portrait-first layout: a title
    /// row carrying the brain-source toggle, a swipeable row of map cards, the
    /// count presets, then one line per catalog slot with the counts
    /// (0/1/10/50/100) as a segmented control. Slots without a trained brain are
    /// omitted and counted in the footer as "coming soon". Start hands off to
    /// Systems_Spawn.
    ///
    /// Density. The whole roster is meant to be visible at once on a phone - the
    /// list keeps its ScrollView as a safety net for short screens and future
    /// creatures, but on a normal handset nothing should need scrolling. That
    /// budget is what every fixed number below is protecting, so before adding a
    /// block here, check what it costs against <see cref="UiTheme.CONTROL_SM"/>
    /// times the roster size. The standings table that used to sit above the list
    /// was removed on 2026-09-03 for exactly this reason: it cost 150 px to
    /// restate ELO numbers that every roster row already carries. The ranking it
    /// conveyed is preserved by sorting the rows by rating instead.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class MenuView : MonoBehaviour
    {
        // Height of the reserved bottom strip: the DBG button is CONTROL_SM tall
        // and sits SPACE_SM off the bottom edge, and the version stamp shares that
        // line. Screen content stops above it - a START button drawn into this
        // band puts two live touch targets on the same pixels.
        private const float BOTTOM_FURNITURE = UiTheme.CONTROL_SM + UiTheme.SPACE_SM;

        private const float MAP_CARD_WIDTH = 138f;
        private const float MAP_STRIP_HEIGHT = 54f;
        private const float AVATAR_SIZE = 24f;

        private CreatureCatalog _catalog;
        private RaceConfigModel _config;
        private EloModel _eloModel;
        private Systems_Spawn _spawn;
        private VisualElement _root;
        private VisualElement _content;
        private Label _totalLabel;
        private Button _startButton;
        private Label _mapBlurbLabel;
        private VisualElement[] _mapCards;
        private Label[] _mapCardNames;
        private VisualElement _creatureList;
        private int _comingSoonCount;
        // Per-row ELO labels, refreshed when the menu re-shows after a race.
        private readonly List<Label> _ratingLabels = new();
        private readonly List<string> _ratingCreatureIds = new();
        private readonly List<VisualElement> _rowElements = new();
        private readonly List<CreatureCatalog.CreatureEntry> _ranked = new();
        private bool _wasVisible;
        // Entrance stagger plays once; menu rebuilds (preset / brain toggle) must
        // not re-animate the whole screen under the user's finger.
        private bool _playEntrance = true;

        [Inject]
        public void Construct(CreatureCatalog catalog, RaceConfigModel config, EloModel eloModel, Systems_Spawn spawn)
        {
            _catalog = catalog;
            _config = config;
            _eloModel = eloModel;
            _spawn = spawn;
        }

        private void Start()
        {
            _root = GetComponent<UIDocument>().rootVisualElement;
            BuildMenu();
            _config.Changed += OnConfigChanged;
            OnConfigChanged();
        }

        private void OnDestroy()
        {
            if (_config != null)
            {
                _config.Changed -= OnConfigChanged;
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
                _startButton.text = hasRacers ? "START RACING" : "PICK A CREATURE TO RACE";
            }
        }

        /// <summary>
        /// Footer counter. Carries the "coming soon" tally too: it is one short
        /// clause here, against a whole line of its own under the roster.
        /// </summary>
        private void RefreshTotals()
        {
            if (_totalLabel == null)
            {
                return;
            }
            int total = _config.TotalCount();
            string text = $"Total racers: {total}";
            if (_comingSoonCount > 0)
            {
                text += $"  -  +{_comingSoonCount} coming soon";
            }
            if (total > 100)
            {
                text += "  -  large field, frame rate may drop";
            }
            _totalLabel.text = text;
        }

        private void BuildMenu()
        {
            // The whole hierarchy is rebuilt on preset / brain-toggle, so the
            // element caches must not keep pointing at discarded elements.
            _ratingLabels.Clear();
            _ratingCreatureIds.Clear();
            _rowElements.Clear();
            _root.style.backgroundColor = UiTheme.ScreenBg;
            _content = UiTheme.BuildSafeRoot(_root);
            _content.pickingMode = PickingMode.Position;
            _content.style.paddingTop = UiTheme.SPACE_SM;
            _content.style.paddingLeft = UiTheme.SPACE_MD;
            _content.style.paddingRight = UiTheme.SPACE_MD;
            _content.style.paddingBottom = UiTheme.SPACE_XS;

            // Bottom-right, matching the race HUD. This sat top-left until 2026-08-29,
            // which is where the game name belongs - the two were stacked on the same
            // corner and the version won the top line.
            var versionLabel = new Label($"v{Application.version}") { pickingMode = PickingMode.Ignore };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.bottom = UiTheme.SPACE_XL;
            versionLabel.style.right = UiTheme.SPACE_XS;
            versionLabel.style.fontSize = UiTheme.FONT_XS;
            versionLabel.style.color = UiTheme.TextDim;
            _content.Add(versionLabel);

            BuildHeader();
            BuildMapPicker();
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
                if (entry.prefab != null && entry.HasBrain)
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
        /// Title block with the brain-source toggle on the same line. The toggle
        /// used to sit on its own row under the wordmark, which cost a full
        /// 48 px control height for one button.
        /// </summary>
        private void BuildHeader()
        {
            var header = new VisualElement();
            header.style.marginBottom = UiTheme.SPACE_XXS;
            _content.Add(header);

            var titleRow = new VisualElement();
            titleRow.style.flexDirection = FlexDirection.Row;
            titleRow.style.alignItems = Align.Center;
            header.Add(titleRow);

            var title = new Label("PoRacer");
            title.style.fontSize = UiTheme.FONT_TITLE;
            title.style.color = UiTheme.Accent;
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            title.style.flexShrink = 0f;
            titleRow.Add(title);

            // Nothing else sits on this line. Top-centre is a reserved anchor - the
            // FPS readout owns it on every screen - and the brain-source toggle
            // that used to sit top-right was removed: it only reached the six
            // ML-Agents creatures, because the other four run their own inference
            // and carry no BehaviorParameters for it to switch.
            // Accent underline gives the title a logo feel.
            var titleBar = new VisualElement { pickingMode = PickingMode.Ignore };
            titleBar.style.height = 3;
            titleBar.style.width = 60;
            titleBar.style.backgroundColor = UiTheme.Accent;
            UiTheme.SetRadius(titleBar, 2f);
            header.Add(titleBar);
        }

        private void BuildMapPicker()
        {
            // Section label and the blurb for the selected map share one line; the
            // blurb had a full row to itself and used a fifth of it.
            var mapHeader = new VisualElement();
            mapHeader.style.flexDirection = FlexDirection.Row;
            mapHeader.style.alignItems = Align.Center;
            _content.Add(mapHeader);

            Label mapLabel = UiTheme.MakeSectionHeader("MAP");
            mapLabel.style.flexShrink = 0f;
            mapHeader.Add(mapLabel);

            _mapBlurbLabel = new Label { pickingMode = PickingMode.Ignore };
            _mapBlurbLabel.style.color = UiTheme.TextDim;
            _mapBlurbLabel.style.fontSize = UiTheme.FONT_XS;
            _mapBlurbLabel.style.marginLeft = UiTheme.SPACE_SM;
            _mapBlurbLabel.style.flexShrink = 1f;
            _mapBlurbLabel.style.minWidth = 0f;
            _mapBlurbLabel.style.whiteSpace = WhiteSpace.NoWrap;
            _mapBlurbLabel.style.overflow = Overflow.Hidden;
            _mapBlurbLabel.style.textOverflow = TextOverflow.Ellipsis;
            mapHeader.Add(_mapBlurbLabel);

            // Only playable maps get cards; unfinished slots stay out of the UI
            // entirely instead of filling the strip with dead "Soon" placeholders.
            int mapCount = Systems_MapCatalog.Entries.Count;
            _mapCards = new VisualElement[mapCount];
            _mapCardNames = new Label[mapCount];

            var strip = new ScrollView(ScrollViewMode.Horizontal);
            strip.style.height = MAP_STRIP_HEIGHT;
            strip.style.flexShrink = 0f;
            strip.contentContainer.style.flexDirection = FlexDirection.Row;
            // A swipe strip: the partly-visible next card is the affordance, so
            // the bar is dead chrome and its reserved height was eating into the
            // cards and clipping their blurbs.
            UiTheme.StyleScrollView(strip, hideBar: true);
            _content.Add(strip);

            for (int mapIndex = 0; mapIndex < mapCount; mapIndex++)
            {
                Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Entries[mapIndex];
                if (!map.Available)
                {
                    continue;
                }
                int capturedIndex = mapIndex;
                var card = new Button(() =>
                {
                    _config.SetMap(capturedIndex);
                    RefreshMapButtons();
                });
                card.style.width = MAP_CARD_WIDTH;
                card.style.flexShrink = 0f;
                card.style.flexDirection = FlexDirection.Row;
                card.style.alignItems = Align.Center;
                card.style.justifyContent = Justify.SpaceBetween;
                UiTheme.SetMargin(card, 0f, UiTheme.SPACE_XXS);
                UiTheme.StyleCard(card, mapIndex == _config.SelectedMapIndex);
                UiTheme.SetPadding(card, UiTheme.SPACE_XXS, UiTheme.SPACE_SM);

                // Name and length sit side by side rather than stacked: the two-line
                // card forced a 72 px strip for two short strings.
                var name = new Label(map.DisplayName) { pickingMode = PickingMode.Ignore };
                name.style.fontSize = UiTheme.FONT_SM;
                name.style.unityFontStyleAndWeight = FontStyle.Bold;
                name.style.unityTextAlign = TextAnchor.MiddleLeft;
                name.style.color = UiTheme.Text;
                name.style.flexShrink = 1f;
                name.style.minWidth = 0f;
                name.style.overflow = Overflow.Hidden;
                name.style.textOverflow = TextOverflow.Ellipsis;
                card.Add(name);

                var lengthLabel = new Label($"{map.LengthMeters:0} m") { pickingMode = PickingMode.Ignore };
                lengthLabel.style.fontSize = UiTheme.FONT_XS;
                lengthLabel.style.color = UiTheme.AccentSoft;
                lengthLabel.style.unityTextAlign = TextAnchor.MiddleRight;
                lengthLabel.style.marginLeft = UiTheme.SPACE_XS;
                lengthLabel.style.flexShrink = 0f;
                card.Add(lengthLabel);

                _mapCards[mapIndex] = card;
                _mapCardNames[mapIndex] = name;
                strip.Add(card);
            }

            RefreshMapButtons();
        }

        private void RefreshMapButtons()
        {
            for (int mapIndex = 0; mapIndex < _mapCards.Length; mapIndex++)
            {
                if (_mapCards[mapIndex] == null)
                {
                    continue;
                }
                bool isSelected = mapIndex == _config.SelectedMapIndex;
                UiTheme.StyleCard(_mapCards[mapIndex], isSelected);
                UiTheme.SetPadding(_mapCards[mapIndex], UiTheme.SPACE_XXS, UiTheme.SPACE_SM);
                _mapCardNames[mapIndex].style.color = isSelected ? UiTheme.AccentSoft : UiTheme.Text;
            }
            Systems_MapCatalog.MapEntry selected = Systems_MapCatalog.Get(_config.SelectedMapIndex);
            _mapBlurbLabel.text = selected.Blurb;
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
            row.style.marginTop = UiTheme.SPACE_XS;
            row.style.marginBottom = UiTheme.SPACE_XXS;
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
            footer.style.marginBottom = BOTTOM_FURNITURE;
            _content.Add(footer);

            // The counter lives on the furniture line, centred between the DBG
            // button and the version stamp. That strip is reserved anyway, so a
            // line of text there is free - inside the footer it cost 30 px of the
            // budget the roster needs.
            _totalLabel = new Label { pickingMode = PickingMode.Ignore };
            _totalLabel.style.position = Position.Absolute;
            _totalLabel.style.left = 0f;
            _totalLabel.style.right = 0f;
            _totalLabel.style.bottom = UiTheme.SPACE_XL;
            _totalLabel.style.color = UiTheme.AccentSoft;
            _totalLabel.style.fontSize = UiTheme.FONT_XS;
            _totalLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _content.Add(_totalLabel);

            _startButton = new Button(() => _spawn.BeginRacing()) { text = "START RACING" };
            _startButton.style.height = UiTheme.CONTROL_MD;
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
                VisualElement child = _content[childIndex];
                if (child.style.position.value == Position.Absolute)
                {
                    continue;
                }
                UiTheme.PlayEnter(child, delay);
                delay += 40;
            }
        }

        private void ApplyPreset(int count)
        {
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (entry.prefab != null && entry.HasBrain)
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
            avatar.style.backgroundColor = Color.HSVToRGB((entry.id.GetHashCode() & 255) / 255f, 0.55f, 0.75f);
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

            var segments = new VisualElement();
            UiTheme.StyleSegmentGroup(segments);
            segments.style.width = Length.Percent(56f);
            segments.style.flexShrink = 0f;
            card.Add(segments);

            int[] options = RaceConfigModel.COUNT_OPTIONS;
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
