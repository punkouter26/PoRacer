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
        private CreatureCatalog _catalog;
        private RaceConfigModel _config;
        private Systems_Spawn _spawn;
        private VisualElement _root;
        private Label _totalLabel;
        private Button[] _mapButtons;

        [Inject]
        public void Construct(CreatureCatalog catalog, RaceConfigModel config, Systems_Spawn spawn)
        {
            _catalog = catalog;
            _config = config;
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
            if (_totalLabel != null)
            {
                int total = _config.TotalCount();
                _totalLabel.text = $"Total racers: {total}" + (total > 100 ? "  (large field - frame rate may drop)" : string.Empty);
            }
        }

        private void BuildMenu()
        {
            _root.style.backgroundColor = UiTheme.ScreenBg;
            _root.style.paddingTop = 24;
            _root.style.paddingLeft = 16;
            _root.style.paddingRight = 16;

            var versionLabel = new Label($"v{Application.version}") { pickingMode = PickingMode.Ignore };
            versionLabel.style.position = Position.Absolute;
            versionLabel.style.top = 4;
            versionLabel.style.left = 6;
            versionLabel.style.fontSize = 12;
            versionLabel.style.color = UiTheme.TextDim;
            _root.Add(versionLabel);

            var title = new Label("PoRacer");
            title.style.fontSize = 30;
            title.style.color = UiTheme.Accent;
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            title.style.marginBottom = 2;
            _root.Add(title);

            var subtitle = new Label("Race Setup");
            subtitle.style.fontSize = 14;
            subtitle.style.color = UiTheme.TextDim;
            subtitle.style.marginBottom = 8;
            _root.Add(subtitle);

            var brainToggle = new Button(ToggleBrains)
            {
                text = _config.UseScriptedBrains ? "Brains: Scripted gaits" : "Brains: RL (trained)"
            };
            brainToggle.style.height = 28;
            brainToggle.style.fontSize = 13;
            brainToggle.style.marginBottom = 12;
            UiTheme.StyleButton(brainToggle);
            _root.Add(brainToggle);

            BuildMapPicker();

            int comingSoonCount = 0;
            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                CreatureCatalog.CreatureEntry entry = _catalog.Entries[entryIndex];
                // Scripted mode races the coded gait, so a missing brain is fine.
                if (entry.prefab != null && (entry.model != null || _config.UseScriptedBrains))
                {
                    _root.Add(BuildRow(entry));
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
                _root.Add(comingSoon);
            }

            _totalLabel = new Label();
            _totalLabel.style.color = UiTheme.AccentSoft;
            _totalLabel.style.fontSize = 14;
            _totalLabel.style.marginTop = 10;
            _root.Add(_totalLabel);

            var startButton = new Button(() => _spawn.BeginRacing()) { text = "START RACING" };
            startButton.style.marginTop = 16;
            startButton.style.height = 44;
            startButton.style.fontSize = 18;
            UiTheme.StyleButton(startButton, accent: true);
            _root.Add(startButton);
        }

        private void BuildMapPicker()
        {
            var mapLabel = new Label("Map");
            mapLabel.style.fontSize = 14;
            mapLabel.style.color = UiTheme.TextDim;
            mapLabel.style.marginBottom = 4;
            _root.Add(mapLabel);

            int mapCount = Systems_MapCatalog.Entries.Count;
            _mapButtons = new Button[mapCount];
            const int mapsPerRow = 4;
            VisualElement row = null;
            for (int mapIndex = 0; mapIndex < mapCount; mapIndex++)
            {
                if (mapIndex % mapsPerRow == 0)
                {
                    row = new VisualElement();
                    row.style.flexDirection = FlexDirection.Row;
                    row.style.marginBottom = 4;
                    _root.Add(row);
                }
                Systems_MapCatalog.MapEntry map = Systems_MapCatalog.Entries[mapIndex];
                int capturedIndex = mapIndex;
                var button = new Button(() =>
                {
                    _config.SetMap(capturedIndex);
                    RefreshMapButtons();
                })
                {
                    text = map.Available ? map.DisplayName : "Soon"
                };
                button.style.height = 30;
                button.style.fontSize = 12;
                button.style.marginRight = 4;
                button.style.flexGrow = 1f;
                button.style.flexBasis = 0f;
                UiTheme.StyleButton(button);
                if (!map.Available)
                {
                    button.SetEnabled(false);
                    button.style.color = UiTheme.TextDim;
                }
                _mapButtons[mapIndex] = button;
                row.Add(button);
            }
            RefreshMapButtons();
        }

        private void RefreshMapButtons()
        {
            for (int mapIndex = 0; mapIndex < _mapButtons.Length; mapIndex++)
            {
                bool isSelected = mapIndex == _config.SelectedMapIndex;
                _mapButtons[mapIndex].style.backgroundColor = isSelected ? UiTheme.Accent : UiTheme.ButtonBg;
            }
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

            var name = new Label(entry.displayName);
            name.style.color = UiTheme.Text;
            name.style.fontSize = 14;
            name.style.width = new Length(38f, LengthUnit.Percent);
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
