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

            var title = new Label("PoRacer");
            title.style.fontSize = 30;
            title.style.color = UiTheme.Accent;
            title.style.unityFontStyleAndWeight = FontStyle.Bold;
            title.style.marginBottom = 2;
            _root.Add(title);

            var subtitle = new Label("Race Setup");
            subtitle.style.fontSize = 14;
            subtitle.style.color = UiTheme.TextDim;
            subtitle.style.marginBottom = 14;
            _root.Add(subtitle);

            for (int entryIndex = 0; entryIndex < _catalog.Entries.Count; entryIndex++)
            {
                _root.Add(BuildRow(_catalog.Entries[entryIndex]));
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

        private VisualElement BuildRow(CreatureCatalog.CreatureEntry entry)
        {
            bool available = entry.prefab != null && entry.model != null;

            var row = new VisualElement();
            row.style.flexDirection = FlexDirection.Row;
            row.style.alignItems = Align.Center;
            row.style.marginBottom = 6;
            row.style.opacity = available ? 1f : 0.4f;
            UiTheme.StyleRow(row);

            var name = new Label(available ? entry.displayName : $"{entry.displayName} (coming soon)");
            name.style.color = UiTheme.Text;
            name.style.fontSize = 14;
            name.style.width = new Length(38f, LengthUnit.Percent);
            row.Add(name);

            if (available)
            {
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
            }
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
