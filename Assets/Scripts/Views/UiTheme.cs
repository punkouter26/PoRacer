using UnityEngine;
using UnityEngine.UIElements;

namespace PoRacer.Views
{
    /// <summary>
    /// Single source of truth for the runtime UI: palette, spacing scale, corner
    /// radius standard, font-size scale and the element styling helpers every
    /// screen (menu, HUD, debug overlay) builds from. Screens never invent their
    /// own numbers — if a value is missing here, add it here.
    /// </summary>
    internal static class UiTheme
    {
        // ---- Spacing scale (the only allowed gaps) ----
        public const float SPACE_XS = 4f;
        public const float SPACE_SM = 8f;
        public const float SPACE_MD = 12f;
        public const float SPACE_LG = 16f;

        // ---- Corner radius standard ----
        public const float RADIUS_SM = 6f;
        public const float RADIUS_MD = 10f;
        public const float RADIUS_LG = 14f;

        // ---- Font size scale ----
        public const float FONT_XS = 11f;
        public const float FONT_SM = 12f;
        public const float FONT_MD = 15f;
        public const float FONT_LG = 18f;
        public const float FONT_TITLE = 30f;
        public const float FONT_BANNER = 52f;
        public const float FONT_COUNTDOWN = 64f;

        // ---- Control heights ----
        public const float CONTROL_SM = 26f;
        public const float CONTROL_MD = 34f;
        public const float CONTROL_LG = 52f;

        // ---- Motion ----
        public const int FADE_MS = 250;
        public const int POP_MS = 250;
        public const int ENTER_MS = 220;

        // ---- Palette ----
        // Accent is the single interactive/selection color; Gold is reserved for
        // podium and celebration moments.
        public static readonly Color Accent = new(0.95f, 0.5f, 0.15f);
        public static readonly Color AccentSoft = new(1f, 0.85f, 0.4f);
        public static readonly Color AccentFill = new(0.95f, 0.5f, 0.15f, 0.18f);
        public static readonly Color ScreenBg = new(0.06f, 0.07f, 0.09f, 0.97f);
        public static readonly Color PanelBg = new(0.07f, 0.08f, 0.1f, 0.82f);
        public static readonly Color PanelBorder = new(1f, 1f, 1f, 0.08f);
        public static readonly Color RowBg = new(0.12f, 0.13f, 0.16f, 0.9f);
        public static readonly Color TrackBg = new(0f, 0f, 0f, 0.35f);
        public static readonly Color ChipBg = new(0.07f, 0.08f, 0.1f, 0.66f);
        public static readonly Color ButtonBg = new(0.2f, 0.21f, 0.24f);
        public static readonly Color Text = new(0.92f, 0.93f, 0.95f);
        public static readonly Color TextDim = new(0.6f, 0.63f, 0.68f);
        public static readonly Color Gold = new(1f, 0.84f, 0.3f);
        public static readonly Color Silver = new(0.8f, 0.83f, 0.88f);
        public static readonly Color Bronze = new(0.85f, 0.58f, 0.35f);
        // Retired / did-not-finish markers.
        public static readonly Color Dnf = new(0.42f, 0.44f, 0.48f);

        public static void StylePanel(VisualElement panel)
        {
            panel.style.backgroundColor = PanelBg;
            SetRadius(panel, RADIUS_MD);
            SetBorder(panel, PanelBorder, 1f);
            SetPadding(panel, SPACE_SM, SPACE_MD);
        }

        public static void StyleRow(VisualElement row)
        {
            row.style.backgroundColor = RowBg;
            SetRadius(row, RADIUS_SM);
            SetPadding(row, SPACE_SM, SPACE_SM);
        }

        /// <summary>
        /// Rounded selectable tile (map picker, creature entry). Selected tiles
        /// carry the accent border and a translucent accent wash.
        /// </summary>
        public static void StyleCard(VisualElement card, bool selected)
        {
            card.style.backgroundColor = selected ? AccentFill : RowBg;
            SetRadius(card, RADIUS_MD);
            SetBorder(card, selected ? Accent : PanelBorder, selected ? 2f : 1f);
            SetPadding(card, SPACE_SM, SPACE_SM);
        }

        /// <summary>Small translucent pill used for HUD status badges.</summary>
        public static void StyleChip(VisualElement chip)
        {
            chip.style.backgroundColor = ChipBg;
            SetRadius(chip, RADIUS_MD);
            SetBorder(chip, PanelBorder, 1f);
            SetPadding(chip, 2f, SPACE_SM);
        }

        public static void StyleButton(Button button, bool accent = false)
        {
            button.style.backgroundColor = accent ? Accent : ButtonBg;
            button.style.color = Text;
            button.style.unityFontStyleAndWeight = FontStyle.Bold;
            SetRadius(button, RADIUS_SM);
            SetBorder(button, PanelBorder, 1f);
        }

        /// <summary>Recessed track that hosts <see cref="StyleSegment"/> buttons.</summary>
        public static void StyleSegmentGroup(VisualElement group)
        {
            group.style.flexDirection = FlexDirection.Row;
            group.style.backgroundColor = TrackBg;
            SetRadius(group, RADIUS_SM + 2f);
            SetBorder(group, PanelBorder, 1f);
            SetPadding(group, 2f, 2f);
        }

        /// <summary>
        /// One cell of a segmented control: borderless, equal share of the track,
        /// accent-filled when it is the active choice.
        /// </summary>
        public static void StyleSegment(Button button, bool selected)
        {
            button.style.backgroundColor = selected ? Accent : Color.clear;
            button.style.color = selected ? Color.white : TextDim;
            button.style.unityFontStyleAndWeight = selected ? FontStyle.Bold : FontStyle.Normal;
            button.style.fontSize = FONT_SM;
            button.style.height = CONTROL_SM;
            button.style.flexGrow = 1f;
            button.style.flexBasis = 0f;
            SetRadius(button, RADIUS_SM);
            SetBorder(button, Color.clear, 0f);
            SetMargin(button, 0f, 1f);
            SetPadding(button, 0f, 0f);
        }

        public static void SetRadius(VisualElement element, float radius)
        {
            element.style.borderTopLeftRadius = radius;
            element.style.borderTopRightRadius = radius;
            element.style.borderBottomLeftRadius = radius;
            element.style.borderBottomRightRadius = radius;
        }

        public static void SetBorder(VisualElement element, Color color, float width)
        {
            element.style.borderTopColor = color;
            element.style.borderBottomColor = color;
            element.style.borderLeftColor = color;
            element.style.borderRightColor = color;
            element.style.borderTopWidth = width;
            element.style.borderBottomWidth = width;
            element.style.borderLeftWidth = width;
            element.style.borderRightWidth = width;
        }

        public static void SetPadding(VisualElement element, float vertical, float horizontal)
        {
            element.style.paddingTop = vertical;
            element.style.paddingBottom = vertical;
            element.style.paddingLeft = horizontal;
            element.style.paddingRight = horizontal;
        }

        public static void SetMargin(VisualElement element, float vertical, float horizontal)
        {
            element.style.marginTop = vertical;
            element.style.marginBottom = vertical;
            element.style.marginLeft = horizontal;
            element.style.marginRight = horizontal;
        }

        /// <summary>Uppercase, letter-spaced section label used above each block.</summary>
        public static Label MakeSectionHeader(string text)
        {
            var label = new Label(text);
            label.style.color = TextDim;
            label.style.fontSize = FONT_XS;
            label.style.unityFontStyleAndWeight = FontStyle.Bold;
            label.style.letterSpacing = 1.5f;
            label.style.marginTop = SPACE_SM;
            label.style.marginBottom = SPACE_XS;
            return label;
        }

        /// <summary>Small circular color dot (racer tint, medal, avatar bullet).</summary>
        public static VisualElement MakeSwatch(Color color, float size)
        {
            var swatch = new VisualElement { pickingMode = PickingMode.Ignore };
            swatch.style.width = size;
            swatch.style.height = size;
            swatch.style.flexShrink = 0f;
            swatch.style.backgroundColor = color;
            SetRadius(swatch, size * 0.5f);
            return swatch;
        }

        /// <summary>
        /// Full-screen child container inset by the device safe area (notches,
        /// rounded corners, home bars). Screens add their content here instead of
        /// the raw document root. A no-op rectangle on screens without cutouts.
        /// </summary>
        public static VisualElement BuildSafeRoot(VisualElement root)
        {
            var safe = new VisualElement { pickingMode = PickingMode.Ignore };
            safe.style.position = Position.Absolute;
            safe.style.left = 0f;
            safe.style.top = 0f;
            safe.style.right = 0f;
            safe.style.bottom = 0f;
            root.Add(safe);
            safe.RegisterCallback<GeometryChangedEvent>(_ => ApplySafeInsets(root, safe));
            return safe;
        }

        private static void ApplySafeInsets(VisualElement root, VisualElement safe)
        {
            float rootWidth = root.resolvedStyle.width;
            if (Screen.width <= 0 || rootWidth <= 0f || float.IsNaN(rootWidth))
            {
                return;
            }
            Rect area = Screen.safeArea;
            // Panel units per screen pixel; safeArea is reported in screen pixels
            // with a bottom-left origin.
            float scale = rootWidth / Screen.width;
            safe.style.left = area.xMin * scale;
            safe.style.right = (Screen.width - area.xMax) * scale;
            safe.style.top = (Screen.height - area.yMax) * scale;
            safe.style.bottom = area.yMin * scale;
        }

        /// <summary>
        /// Pointer-over brightening for buttons whose background color is static.
        /// Do not use on selection-tinted buttons — it would clobber their color.
        /// </summary>
        public static void AddHover(Button button, bool accent = false)
        {
            Color baseColor = accent ? Accent : ButtonBg;
            Color hoverColor = Color.Lerp(baseColor, Color.white, 0.18f);
            button.RegisterCallback<PointerEnterEvent>(_ => button.style.backgroundColor = hoverColor);
            button.RegisterCallback<PointerLeaveEvent>(_ => button.style.backgroundColor = baseColor);
        }

        /// <summary>
        /// Fade + rise entrance. The stagger delay is baked into one animation
        /// instead of a scheduled callback, so an element can never be stranded
        /// at zero opacity if its delayed callback is dropped.
        /// </summary>
        public static void PlayEnter(VisualElement element, int delayMs)
        {
            const float SLIDE_PIXELS = 16f;
            float delay = delayMs < 0 ? 0f : delayMs;
            float total = delay + ENTER_MS;
            element.style.opacity = 0f;
            element.style.translate = new Translate(0f, SLIDE_PIXELS);
            element.experimental.animation.Start(0f, 1f, (int)total, (target, value) =>
            {
                float linear = Mathf.Clamp01((value * total - delay) / ENTER_MS);
                float remaining = 1f - linear;
                float eased = 1f - remaining * remaining * remaining;
                target.style.opacity = eased;
                target.style.translate = new Translate(0f, SLIDE_PIXELS * (1f - eased));
            });
        }
    }
}
