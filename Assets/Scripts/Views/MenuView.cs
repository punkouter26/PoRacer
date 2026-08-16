using System.Collections.Generic;
using PoRacer.Models;
using PoRacer.Systems;
using UnityEngine;
using UnityEngine.UIElements;
using VContainer;

namespace PoRacer.Views
{
    /// <summary>
    /// Start menu, UI Toolkit hierarchy built in C#. One row per catalog slot:
    /// available creatures get count buttons (0/1/10/50/100); slots without a
    /// trained brain are greyed out as "coming soon". Start hands off to Systems_Spawn.
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class MenuView : MonoBehaviour
    {
        private const int MENU_FADE_MS = 250;

        private CreatureCatalog _catalog;
        private RaceConfigModel _config;
        private EloModel _eloModel;
        private Systems_Spawn _spawn;
        private VisualElement _root;
        private VisualElement _content;
        private Label _totalLabel;
        private Label _mapBlurbLabel;
        private Button[] _mapButtons;
        private bool _wasVisible;

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
                _root.experimental.animation.Start(0f, 1f, MENU_FADE_MS, (element, value) =>
                {
                    element.style.opacity = value;
                });
            }
            _wasVisible = _config.MenuVisible;
            if (_totalLabel != null)
            {
                int total = _config.TotalCount();
                _totalLabel.text = $"Total racers: {total}" + (total > 100 ? "  (large field - frame rate may drop)" : string.Empty);
            }
        }

        private void BuildMenu()
        {
            _root.style.backgroundColor = UiTheme.ScreenBg;
            _content = UiTheme.BuildSafeRoot(_root);
            _content.pickingMode = PickingMode.Position;
            _content.style.paddingTop = 24;
            _content.style.paddingLeft = 16;
            _content.style.paddingRight = 16;
            _content.style.paddingBottom = 16;

            var versionLabel = new Label($"v{Application.version}") { pickingMode = PickingMode.Ignore };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.top = 4;
            versionLabel.style.left = 6;
            versionLabel.style.fontSize = 12;
            versionLabel.style.color = UiTheme.TextDim;
            _content.Add(versionLabel);

            var title = new Label("PoRacer");
            title.style.fontSize = 30;
            title.style.color = UiTheme.Accent;
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            title.style.marginBottom = 2;
            _content.Add(title);

            // Accent underline gives the title a logo feel.
            var titleBar = new VisualElement();
            titleBar.style.height = 3;
            titleBar.style.width = 72;
            titleBar.style.backgroundColor = UiTheme.Accent;
            titleBar.style.marginBottom = 6;
            UiTheme.SetRadius(titleBar, 2f);
            _content.Add(titleBar);

            var subtitle = new Label("Race Setup");
            subtitle.style.fontSize = 14;
            subtitle.style.color = UiTheme.TextDim;
            subtitle.style.marginBottom = 8;
            _content.Add(subtitle);

            var brainToggle = new Button(ToggleBrains)
            {
                text = _config.UseScriptedBrains ? "Brains: Scripted gaits" : "Brains: RL (trained)"
            };
            brainToggle.style.height = 28;
            brainToggle.style.fontSize = 13;
            brainToggle.style.marginBottom = 12;
            UiTheme.StyleButton(brainToggle);
            UiTheme.AddHover(brainToggle);
            _content.Add(brainToggle);

            BuildMapPicker();
            BuildStandings();

            var creatureHeader = new Label("Creatures");
            creatureHeader.style.fontSize = 14;
            creatureHeader.style.color = UiTheme.TextDim;
            creatureHeader.style.marginBottom = 4;
            _content.Add(creatureHeader);

            BuildPresetRow();

            // Creature list scrolls so the START button never leaves the screen
            // on portrait, no matter how many creatures the catalog grows to.
            var creatureScroll = new ScrollView(ScrollViewMode.Vertical);
            creatureScroll.style.flexGrow = 1f;
            creatureScroll.style.flexShrink = 1f;
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
                comingSoon.style.fontSize = 12;
                comingSoon.style.marginTop = 4;
                creatureScroll.Add(comingSoon);
            }

            _totalLabel = new Label();
            _totalLabel.style.color = UiTheme.AccentSoft;
            _totalLabel.style.fontSize = 14;
            _totalLabel.style.marginTop = 10;
            _content.Add(_totalLabel);

            var startButton = new Button(() => _spawn.BeginRacing()) { text = "START RACING" };
            startButton.style.marginTop = 16;
            startButton.style.height = 44;
            startButton.style.fontSize = 18;
            UiTheme.StyleButton(startButton, accent: true);
            UiTheme.AddHover(startButton, accent: true);
            _content.Add(startButton);
        }

        private void BuildMapPicker()
        {
            var mapLabel = new Label("Map");
            mapLabel.style.fontSize = 14;
            mapLabel.style.color = UiTheme.TextDim;
            mapLabel.style.marginBottom = 4;
            _content.Add(mapLabel);

            // Only playable maps get buttons; unfinished slots stay out of the UI
            // entirely instead of filling rows with dead "Soon" placeholders.
            int mapCount = Systems_MapCatalog.Entries.Count;
            _mapButtons = new Button[mapCount];
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.marginBottom = 4;
            _content.Add(row);
            for (int mapIndex = 0; mapIndex < mapCount; mapIndex++)
            {
                Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Entries[mapIndex];
                if (!map.Available)
                {
                    continue;
                }
                int capturedIndex = mapIndex;
                var button = new Button(() =>
                {
                    _config.SetMap(capturedIndex);
                    RefreshMapButtons();
                })
                {
                    text = map.DisplayName
                };
                button.style.height = 30;
                button.style.fontSize = 12;
                button.style.marginRight = 4;
                button.style.flexGrow = 1f;
                button.style.flexBasis = 0f;
                UiTheme.StyleButton(button);
                _mapButtons[mapIndex] = button;
                row.Add(button);
            }

            _mapBlurbLabel = new Label();
            _mapBlurbLabel.style.color = UiTheme.TextDim;
            _mapBlurbLabel.style.fontSize = 12;
            _mapBlurbLabel.style.marginBottom = 10;
            _content.Add(_mapBlurbLabel);

            RefreshMapButtons();
        }

        private void RefreshMapButtons()
        {
            for (int mapIndex = 0; mapIndex < _mapButtons.Length; mapIndex++)
            {
                if (_mapButtons[mapIndex] == null)
                {
                    continue;
                }
                bool isSelected = mapIndex == _config.SelectedMapIndex;
                _mapButtons[mapIndex].style.backgroundColor = isSelected ? UiTheme.Accent : UiTheme.ButtonBg;
            }
            Systems_MapCatalog.MapEntry selected = Systems_MapCatalog.Get(_config.SelectedMapIndex);
            _mapBlurbLabel.text = $"{selected.DisplayName} — {selected.Blurb}";
        }

        /// <summary>Top-5 ELO table so race history means something in the menu.</summary>
        private void BuildStandings()
        {
            var header = new Label("Standings (ELO)");
            header.style.fontSize = 14;
            header.style.color = UiTheme.TextDim;
            header.style.marginBottom = 4;
            _content.Add(header);

            var ranked = new List<CreatureCatalog.CreatureEntry>();
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                if (_catalog.Entries[entryIndex].prefab != null)
                {
                    ranked.Add(_catalog.Entries[entryIndex]);
                }
            }
            ranked.Sort((first, second) =>
                _eloModel.GetRating(second.id).CompareTo(_eloModel.GetRating(first.id)));

            var panel = new VisualElement();
            UiTheme.StyleRow(panel);
            panel.style.marginBottom = 10;
            int rows = ranked.Count < 5 ? ranked.Count : 5;
            for (int rankIndex = 0; rankIndex < rows; rankIndex++)
            {
                CreatureCatalog.CreatureEntry entry = ranked[rankIndex];
                var line = new Label(
                    $"{rankIndex + 1}. {entry.displayName}   {_eloModel.GetRating(entry.id):0}");
                line.style.color = rankIndex == 0 ? UiTheme.Gold : UiTheme.Text;
                line.style.fontSize = 12;
                panel.Add(line);
            }
            if (rows == 0)
            {
                var empty = new Label("No races yet — standings appear after the first race.");
                empty.style.color = UiTheme.TextDim;
                empty.style.fontSize = 12;
                panel.Add(empty);
            }
            _content.Add(panel);
        }

        /// <summary>One-tap field setup instead of nine separate count rows.</summary>
        private void BuildPresetRow()
        {
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.marginBottom = 6;
            string[] labels = { "All x1", "All x10", "Clear" };
            int[] counts = { 1, 10, 0 };
            for (int presetIndex = 0; presetIndex < labels.Length; presetIndex++)
            {
                int count = counts[presetIndex];
                var button = new Button(() => ApplyPreset(count)) { text = labels[presetIndex] };
                button.style.height = 26;
                button.style.fontSize = 12;
                button.style.marginRight = 4;
                button.style.flexGrow = 1f;
                button.style.flexBasis = 0f;
                UiTheme.StyleButton(button);
                UiTheme.AddHover(button);
                row.Add(button);
            }
            _content.Add(row);
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
            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.alignItems = Align.Center;
            row.style.marginBottom = 6;
            UiTheme.StyleRow(row);

            // Round avatar chip: creature initial on a per-creature hue, so rows
            // read as cards even without portrait art.
            var avatar = new VisualElement();
            avatar.style.width = 24;
            avatar.style.height = 24;
            avatar.style.marginRight = 6;
            avatar.style.justifyContent = Justify.Center;
            avatar.style.alignItems = Align.Center;
            avatar.style.flexShrink = 0f;
            avatar.style.backgroundColor = Color.HSVToRGB((entry.id.GetHashCode() & 255) / 255f, 0.55f, 0.75f);
            UiTheme.SetRadius(avatar, 12f);
            var initial = new Label(entry.displayName.Substring(0, 1));
            initial.style.color = Color.white;
            initial.style.fontSize = 13;
            initial.style.unityFontStyleAndWeight = FontStyle.Bold;
            avatar.Add(initial);
            row.Add(avatar);

            var name = new Label(entry.displayName);
            name.style.color = UiTheme.Text;
            name.style.fontSize = 14;
            name.style.flexGrow = 1f;
            row.Add(name);

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
                button.style.width = 40;
                button.style.height = 28;
                button.style.marginLeft = 4;
                UiTheme.StyleButton(button);
                countButtons[optionIndex] = button;
                row.Add(button);
            }
            RefreshRowButtons(countButtons, options, entry.id);
            return row;
        }

        private void RefreshRowButtons(Button[] buttons, int[] options, string creatureId)
        {
            int selected = _config.GetCount(creatureId);
            for (int buttonIndex = 0; buttonIndex < buttons.Length; buttonIndex++)
            {
                bool isSelected = options[buttonIndex] == selected;
                buttons[buttonIndex].style.backgroundColor = isSelected ? UiTheme.Accent : UiTheme.ButtonBg;
            }
        }
    }
}
