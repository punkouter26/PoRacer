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
    /// Start menu, UI Toolkit hierarchy built in C#. Portrait-first layout: title
    /// header, a swipeable row of map cards, the ELO standings block, then one
    /// card per catalog slot with the counts (0/1/10/50/100) as a segmented
    /// control. Slots without a trained brain are omitted as "coming soon".
    /// Start hands off to Systems_Spawn.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class MenuView : MonoBehaviour
    {
        private const float MAP_CARD_WIDTH = 148f;
        private const float MAP_STRIP_HEIGHT = 72f;
        private const float AVATAR_SIZE = 26f;
        private const int STANDINGS_ROWS = 5;

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
        private VisualElement _standingsPanel;
        // Per-row ELO labels, refreshed when the menu re-shows after a race.
        private readonly List<Label> _ratingLabels = new();
        private readonly List<string> _ratingCreatureIds = new();
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
            if (_totalLabel != null)
            {
                int total = _config.TotalCount();
                _totalLabel.text = $"Total racers: {total}" + (total > 100 ? "  (large field - frame rate may drop)" : string.Empty);
            }
            if (_startButton != null)
            {
                // A disabled start button explains itself; a silent no-op does not.
                bool hasRacers = _config.TotalCount() > 0;
                _startButton.SetEnabled(hasRacers);
                _startButton.text = hasRacers ? "START RACING" : "PICK A CREATURE TO RACE";
            }
        }

        private void BuildMenu()
        {
            // The whole hierarchy is rebuilt on preset / brain-toggle, so the
            // element caches must not keep pointing at discarded elements.
            _ratingLabels.Clear();
            _ratingCreatureIds.Clear();
            _root.style.backgroundColor = UiTheme.ScreenBg;
            _content = UiTheme.BuildSafeRoot(_root);
            _content.pickingMode = PickingMode.Position;
            _content.style.paddingTop = UiTheme.SPACE_LG;
            _content.style.paddingLeft = UiTheme.SPACE_LG;
            _content.style.paddingRight = UiTheme.SPACE_LG;
            _content.style.paddingBottom = UiTheme.SPACE_SM;

            var versionLabel = new Label($"v{Application.version}") { pickingMode = PickingMode.Ignore };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.top = UiTheme.SPACE_XS;
            versionLabel.style.left = UiTheme.SPACE_SM;
            versionLabel.style.fontSize = UiTheme.FONT_SM;
            versionLabel.style.color = UiTheme.TextDim;
            _content.Add(versionLabel);

            BuildHeader();
            BuildMapPicker();
            BuildStandings();

            _content.Add(UiTheme.MakeSectionHeader("CREATURES"));
            BuildPresetRow();

            // Creature list scrolls so the START button never leaves the screen
            // on portrait, no matter how many creatures the catalog grows to.
            var creatureScroll = new ScrollView(ScrollViewMode.Vertical);
            creatureScroll.style.flexGrow = 1f;
            creatureScroll.style.flexShrink = 1f;
            UiTheme.StyleScrollView(creatureScroll);
            _content.Add(creatureScroll);

            int comingSoonCount = 0;
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                // Scripted mode races the coded gait, so a missing brain is fine.
                if (entry.prefab != null && (entry.model != null || _config.UseScriptedBrains))
                {
                    creatureScroll.Add(BuildRow(entry));
                }
                else
                {
                    comingSoonCount++;
                }
            }
            if (comingSoonCount > 0)
            {
                var comingSoon = new Label($"+{comingSoonCount} more creatures coming soon");
                comingSoon.style.color = UiTheme.TextDim;
                comingSoon.style.fontSize = UiTheme.FONT_SM;
                comingSoon.style.unityTextAlign = TextAnchor.MiddleCenter;
                comingSoon.style.marginTop = UiTheme.SPACE_XS;
                creatureScroll.Add(comingSoon);
            }

            BuildFooter();

            if (_playEntrance)
            {
                PlayEntrance();
                _playEntrance = false;
            }
        }

        /// <summary>Title block plus the brain-source toggle.</summary>
        private void BuildHeader()
        {
            var header = new VisualElement();
            header.style.marginBottom = UiTheme.SPACE_SM;
            _content.Add(header);

            var titleRow = new VisualElement();
            titleRow.style.flexDirection = FlexDirection.Row;
            titleRow.style.alignItems = Align.FlexEnd;
            header.Add(titleRow);

            var title = new Label("PoRacer");
            title.style.fontSize = UiTheme.FONT_TITLE;
            title.style.color = UiTheme.Accent;
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            titleRow.Add(title);

            var subtitle = new Label("Race Setup");
            subtitle.style.fontSize = UiTheme.FONT_SM;
            subtitle.style.color = UiTheme.TextDim;
            subtitle.style.marginLeft = UiTheme.SPACE_SM;
            subtitle.style.marginBottom = UiTheme.SPACE_SM;
            titleRow.Add(subtitle);

            // Accent underline gives the title a logo feel.
            var titleBar = new VisualElement();
            titleBar.style.height = 3;
            titleBar.style.width = 72;
            titleBar.style.backgroundColor = UiTheme.Accent;
            titleBar.style.marginBottom = UiTheme.SPACE_SM;
            UiTheme.SetRadius(titleBar, 2f);
            header.Add(titleBar);

            var brainToggle = new Button(ToggleBrains)
            {
                text = _config.UseScriptedBrains ? "Brains: Scripted gaits" : "Brains: RL (trained)"
            };
            brainToggle.style.height = UiTheme.CONTROL_SM;
            brainToggle.style.fontSize = UiTheme.FONT_SM;
            UiTheme.SetMargin(brainToggle, 0f, 0f);
            UiTheme.StyleButton(brainToggle);
            UiTheme.AddHover(brainToggle);
            header.Add(brainToggle);
        }

        private void BuildMapPicker()
        {
            _content.Add(UiTheme.MakeSectionHeader("MAP"));

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
                card.style.flexDirection = FlexDirection.Column;
                // Stretch so the blurb label gets the card width to wrap inside.
                card.style.alignItems = Align.Stretch;
                UiTheme.SetMargin(card, 0f, UiTheme.SPACE_XS);
                UiTheme.StyleCard(card, mapIndex == _config.SelectedMapIndex);

                // Button children inherit the button's centered text alignment;
                // card copy reads as a card only when it is left-aligned.
                var name = new Label(map.DisplayName) { pickingMode = PickingMode.Ignore };
                name.style.fontSize = UiTheme.FONT_MD;
                name.style.unityFontStyleAndWeight = FontStyle.Bold;
                name.style.unityTextAlign = TextAnchor.MiddleLeft;
                name.style.color = UiTheme.Text;
                name.style.marginBottom = UiTheme.SPACE_XS;
                card.Add(name);

                var lengthLabel = new Label($"{map.LengthMeters:0} m") { pickingMode = PickingMode.Ignore };
                lengthLabel.style.fontSize = UiTheme.FONT_XS;
                lengthLabel.style.color = UiTheme.AccentSoft;
                lengthLabel.style.unityTextAlign = TextAnchor.MiddleLeft;
                card.Add(lengthLabel);

                _mapCards[mapIndex] = card;
                _mapCardNames[mapIndex] = name;
                strip.Add(card);
            }

            _mapBlurbLabel = new Label();
            _mapBlurbLabel.style.color = UiTheme.TextDim;
            _mapBlurbLabel.style.fontSize = UiTheme.FONT_XS;
            _mapBlurbLabel.style.whiteSpace = WhiteSpace.Normal;
            _mapBlurbLabel.style.whiteSpace = WhiteSpace.NoWrap;
            _mapBlurbLabel.style.overflow = Overflow.Hidden;
            _mapBlurbLabel.style.textOverflow = TextOverflow.Ellipsis;
            _content.Add(_mapBlurbLabel);

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
                _mapCardNames[mapIndex].style.color = isSelected ? UiTheme.AccentSoft : UiTheme.Text;
            }
            Systems_MapCatalog.MapEntry selected = Systems_MapCatalog.Get(_config.SelectedMapIndex);
            _mapBlurbLabel.text = $"{selected.DisplayName} — {selected.Blurb}";
        }

        /// <summary>Top-5 ELO table so race history means something in the menu.</summary>
        private void BuildStandings()
        {
            _content.Add(UiTheme.MakeSectionHeader("STANDINGS (ELO)"));
            _standingsPanel = new VisualElement();
            UiTheme.StyleGlassPanel(_standingsPanel);
            _content.Add(_standingsPanel);
            RefreshStandings();
        }

        /// <summary>
        /// Rebuilds the standings rows from current ratings. Cheap and rare — it
        /// runs on menu build and each time the menu comes back from a race.
        /// </summary>
        private void RefreshStandings()
        {
            if (_standingsPanel == null)
            {
                return;
            }
            _standingsPanel.Clear();
            _ranked.Clear();
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                if (_catalog.Entries[entryIndex].prefab != null)
                {
                    _ranked.Add(_catalog.Entries[entryIndex]);
                }
            }
            _ranked.Sort((first, second) =>
                _eloModel.GetRating(second.id).CompareTo(_eloModel.GetRating(first.id)));

            int rows = _ranked.Count < STANDINGS_ROWS ? _ranked.Count : STANDINGS_ROWS;
            // One wrapping row of "rank name rating" chips: the five-line table was the
            // single biggest reason the creature list needed scrolling on a phone.
            var chips = new VisualElement();
            chips.style.flexDirection = FlexDirection.Row;
            chips.style.flexWrap = Wrap.Wrap;
            chips.style.justifyContent = Justify.SpaceBetween;
            _standingsPanel.Add(chips);
            for (int rankIndex = 0; rankIndex < rows; rankIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _ranked[rankIndex];
                Color tone = rankIndex == 0 ? UiTheme.Gold : UiTheme.Text;
                var chip = new Label($"{rankIndex + 1}  {entry.displayName}  {_eloModel.GetRating(entry.id):0}");
                chip.style.fontSize = UiTheme.FONT_XS;
                chip.style.color = tone;
                chip.style.unityFontStyleAndWeight = rankIndex == 0 ? FontStyle.Bold : FontStyle.Normal;
                chip.style.marginRight = UiTheme.SPACE_SM;
                chips.Add(chip);
            }
            if (rows == 0)
            {
                var empty = new Label("No races yet — standings appear after the first race.");
                empty.style.color = UiTheme.TextDim;
                empty.style.fontSize = UiTheme.FONT_SM;
                empty.style.whiteSpace = WhiteSpace.Normal;
                _standingsPanel.Add(empty);
            }
        }

        /// <summary>Re-reads ELO for every roster row and the standings table.</summary>
        private void RefreshRatings()
        {
            for (int labelIndex = 0; labelIndex < _ratingLabels.Count; labelIndex++)
            {
                _ratingLabels[labelIndex].text = $"ELO {_eloModel.GetRating(_ratingCreatureIds[labelIndex]):0}";
            }
            RefreshStandings();
        }

        /// <summary>One-tap field setup instead of nine separate count rows.</summary>
        private void BuildPresetRow()
        {
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
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
            footer.style.marginTop = UiTheme.SPACE_XS;
            UiTheme.StyleGlassPanel(footer, glowing: true);
            _content.Add(footer);

            _totalLabel = new Label();
            _totalLabel.style.color = UiTheme.AccentSoft;
            _totalLabel.style.fontSize = UiTheme.FONT_SM;
            _totalLabel.style.unityTextAlign = TextAnchor.MiddleCenter;
            _totalLabel.style.marginBottom = UiTheme.SPACE_XS;
            footer.Add(_totalLabel);

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
            bool scripted = _config.UseScriptedBrains;
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                if (entry.prefab != null && (entry.model != null || scripted))
                {
                    _config.SetCount(entry.id, count);
                }
            }
            // Count buttons highlight from config state; rebuild to reflect it.
            _root.Clear();
            BuildMenu();
            _config.NotifyChanged();
        }

        private void ToggleBrains()
        {
            _config.UseScriptedBrains = !_config.UseScriptedBrains;
            // Availability changes with the mode, so rebuild the whole menu.
            _root.Clear();
            BuildMenu();
            _config.NotifyChanged();
        }

        private VisualElement BuildRow(CreatureCatalog.CreatureEntry entry)
        {
            // One line per creature: avatar | name over ELO | count segments. The
            // previous two-line card fit only three creatures above the START button.
            var card = new VisualElement();
            card.style.marginBottom = UiTheme.SPACE_XXS;
            UiTheme.StyleCard(card, selected: false);
            card.style.flexDirection = FlexDirection.Row;
            card.style.alignItems = Align.Center;
            card.style.paddingTop = UiTheme.SPACE_XXS;
            card.style.paddingBottom = UiTheme.SPACE_XXS;

            var titleRow = new VisualElement();
            titleRow.style.flexDirection = FlexDirection.Row;
            titleRow.style.alignItems = Align.Center;
            titleRow.style.flexGrow = 1f;
            titleRow.style.flexShrink = 1f;
            titleRow.style.minWidth = 0f;
            card.Add(titleRow);

            // Round avatar chip: creature initial on a per-creature hue, so rows
            // read as cards even without portrait art.
            var avatar = new VisualElement();
            avatar.style.width = AVATAR_SIZE;
            avatar.style.height = AVATAR_SIZE;
            avatar.style.marginRight = UiTheme.SPACE_SM;
            avatar.style.justifyContent = Justify.Center;
            avatar.style.alignItems = Align.Center;
            avatar.style.flexShrink = 0f;
            avatar.style.backgroundColor = Color.HSVToRGB((entry.id.GetHashCode() & 255) / 255f, 0.55f, 0.75f);
            UiTheme.SetRadius(avatar, AVATAR_SIZE * 0.5f);
            var initial = new Label(entry.displayName.Substring(0, 1));
            initial.style.color = Color.white;
            initial.style.fontSize = UiTheme.FONT_MD;
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
            titleRow.Add(nameColumn);

            var name = new Label(entry.displayName);
            name.style.color = UiTheme.Text;
            name.style.fontSize = UiTheme.FONT_SM;
            name.style.overflow = Overflow.Hidden;
            name.style.textOverflow = TextOverflow.Ellipsis;
            nameColumn.Add(name);

            var eloLabel = new Label($"ELO {_eloModel.GetRating(entry.id):0}");
            eloLabel.style.color = UiTheme.TextDim;
            eloLabel.style.fontSize = UiTheme.FONT_XS;
            eloLabel.style.marginLeft = UiTheme.SPACE_SM;
            eloLabel.style.flexShrink = 0f;
            nameColumn.Add(eloLabel);
            _ratingLabels.Add(eloLabel);
            _ratingCreatureIds.Add(entry.id);

            var segments = new VisualElement();
            UiTheme.StyleSegmentGroup(segments);
            segments.style.width = Length.Percent(58f);
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
