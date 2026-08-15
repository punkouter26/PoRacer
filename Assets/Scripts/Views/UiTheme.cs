using UnityEngine;
using UnityEngine.UIElements;

namespace PoRacer.Views
{
    /// <summary>
    /// Shared UI palette and element styling. Single source of truth so every
    /// screen (menu, HUD, debug overlay) looks like one application.
    /// </summary>
    internal static class UiTheme
    {
        public static readonly Color Accent = new(0.95f, 0.5f, 0.15f);
        public static readonly Color AccentSoft = new(1f, 0.85f, 0.4f);
        public static readonly Color ScreenBg = new(0.06f, 0.07f, 0.09f, 0.97f);
        public static readonly Color PanelBg = new(0.07f, 0.08f, 0.1f, 0.82f);
        public static readonly Color PanelBorder = new(1f, 1f, 1f, 0.08f);
        public static readonly Color RowBg = new(0.12f, 0.13f, 0.16f, 0.9f);
        public static readonly Color ButtonBg = new(0.2f, 0.21f, 0.24f);
        public static readonly Color Text = new(0.92f, 0.93f, 0.95f);
        public static readonly Color TextDim = new(0.6f, 0.63f, 0.68f);
        public static readonly Color Gold = new(1f, 0.84f, 0.3f);
        public static readonly Color Silver = new(0.8f, 0.83f, 0.88f);
        public static readonly Color Bronze = new(0.85f, 0.58f, 0.35f);

        public static void StylePanel(VisualElement panel)
        {
            panel.style.backgroundColor = PanelBg;
            SetRadius(panel, 8f);
            SetBorder(panel, PanelBorder, 1f);
            SetPadding(panel, 6f, 10f);
        }

        public static void StyleRow(VisualElement row)
        {
            row.style.backgroundColor = RowBg;
            SetRadius(row, 6f);
            SetPadding(row, 6f, 8f);
        }

        public static void StyleButton(Button button, bool accent = false)
        {
            button.style.backgroundColor = accent ? Accent : ButtonBg;
            button.style.color = Text;
            button.style.unityFontStyleAndWeight = FontStyle.Bold;
            SetRadius(button, 6f);
            SetBorder(button, PanelBorder, 1f);
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
    }
}
