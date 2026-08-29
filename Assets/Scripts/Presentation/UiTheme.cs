using UnityEngine;
using UnityEngine.UIElements;

namespace PoRacer.Presentation
{
    /// <summary>
    /// Single source of truth for the runtime UI: palette, spacing scale, corner
    /// radius standard, font-size scale, elevation levels and the element styling
    /// helpers every screen (menu, HUD, debug overlay) builds from. Screens never
    /// invent their own numbers — if a value is missing here, add it here.
    ///
    /// Elevation. UI Toolkit has no box-shadow, so depth is built from an offset
    /// rounded panel sitting behind the element plus the specular top edge the
    /// glass style already draws. That pairing is what separates the three
    /// background tiers (screen / panel / glass) into layers the eye can order,
    /// instead of three flat rectangles of slightly different grey.
    ///
    /// Density. The panel is set to scale with screen width against a 540x960
    /// reference, so a fixed value here means the same fraction of the screen on
    /// any handset. <see cref="DensityFor"/> exists for the cases where scaling is
    /// not enough and a layout genuinely needs to drop a column.
    /// </summary>
    internal static class UiTheme
    {
        // ---- Spacing scale (the only allowed gaps) ----
        public const float SPACE_XXS = 2f;
        public const float SPACE_XS = 4f;
        public const float SPACE_SM = 8f;
        public const float SPACE_MD = 12f;
        public const float SPACE_LG = 16f;
        public const float SPACE_XL = 24f;
        public const float SPACE_XXL = 32f;

        // ---- Corner radius standard ----
        public const float RADIUS_XS = 4f;
        public const float RADIUS_SM = 6f;
        public const float RADIUS_MD = 10f;
        public const float RADIUS_LG = 14f;
        public const float RADIUS_XL = 18f;

        // ---- Font size scale ----
        // Raised ~25% over the original scale for phone legibility: the panel is
        // scaled on width, so on a 1080-wide handset held at arm's length the old
        // 11-12 px steps were below comfortable reading size. Control heights move
        // with them so text keeps its padding instead of crowding the button edge.
        //
        // These are REFERENCE pixels against the 540 x 960 PanelSettings reference,
        // matched on width. To read them as Android sp:
        //     sp = ref * (screenWidthPx / 540) / (densityDpi / 160)
        // Measured on the Pixel 9 Pro this was verified against (960 x 2142 at
        // density 360, i.e. scale 1.778 / density 2.25), that factor is 0.79. XS and
        // SM were 14 and 15, which land at 11.1 sp and 11.9 sp - both under Android's
        // 12 sp floor for body text, and they carry the version stamp and the debug
        // strip. 16 and 18 put them at 12.6 sp and 14.2 sp.
        public const float FONT_XS = 16f;
        public const float FONT_SM = 18f;
        public const float FONT_MD = 19f;
        public const float FONT_LG = 23f;
        public const float FONT_TITLE = 37f;
        public const float FONT_BANNER = 62f;
        public const float FONT_COUNTDOWN = 78f;

        // ---- Control heights (touch-ergonomic standards) ----
        // Android's minimum touch target is 48 dp. With the panel's 420 dp reference
        // width, one UI unit is ~1 dp on a phone, so these numbers are dp directly.
        // CONTROL_SM was 38 (30 dp on a Pixel 9 Pro) and CONTROL_MD was 52 (41 dp) -
        // both under spec - back when the panel referenced a 540 dp-wide screen that
        // no phone actually has.
        public const float CONTROL_SM = 48f;
        public const float CONTROL_MD = 56f;
        public const float CONTROL_LG = 64f;

        // ---- Elevation ----
        // Vertical offset and opacity of the shadow plate behind a raised element.
        // Higher levels sit further from their backdrop, so the shadow both drops
        // further and softens (spreads wider, at lower opacity).
        public const float ELEVATION_LOW = 2f;
        public const float ELEVATION_MID = 5f;
        public const float ELEVATION_HIGH = 10f;

        // ---- Scrollbar ----
        public const float SCROLLBAR_THICKNESS = 4f;

        // ---- Motion ----
        public const int FADE_MS = 250;
        public const int POP_MS = 250;
        public const int ENTER_MS = 220;
        // Gap between consecutive items of a staggered list entrance.
        public const int STAGGER_MS = 120;
        // Default rise distance of an entrance; panels use the taller travel.
        public const float ENTER_SLIDE_PX = 16f;
        public const float PANEL_SLIDE_PX = 24f;

        // ---- Palette ----
        // Accent is the single interactive/selection color; Gold is reserved for
        // podium and celebration moments.
        public static readonly Color Accent = new(0.95f, 0.5f, 0.15f);
        public static readonly Color AccentSoft = new(1f, 0.85f, 0.4f);
        public static readonly Color AccentGlow = new(1f, 0.6f, 0.2f, 0.35f);
        public static readonly Color AccentFill = new(0.95f, 0.5f, 0.15f, 0.18f);
        public static readonly Color ScreenBg = new(0.06f, 0.07f, 0.09f, 0.97f);
        public static readonly Color PanelBg = new(0.07f, 0.08f, 0.1f, 0.82f);
        public static readonly Color GlassBg = new(0.08f, 0.09f, 0.13f, 0.72f);
        public static readonly Color GlassBorder = new(1f, 1f, 1f, 0.15f);
        public static readonly Color GlassHighlight = new(1f, 1f, 1f, 0.28f);
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
        public static readonly Color NeonCyan = new(0.2f, 0.85f, 1f);
        public static readonly Color NeonGreen = new(0.3f, 0.95f, 0.45f);
        // Retired / did-not-finish markers.
        public static readonly Color Dnf = new(0.42f, 0.44f, 0.48f);
        public static readonly Color ScrollThumb = new(1f, 1f, 1f, 0.22f);
        // Shadow plate. Nearly black rather than a tinted grey: on a dark UI a
        // tinted shadow reads as a coloured outline, not as depth.
        public static readonly Color Shadow = new(0f, 0f, 0f, 0.45f);

        /// <summary>Layout density, for the few places where scaling is not enough.</summary>
        public enum Density
        {
            /// <summary>Narrow portrait: one column, fewer optional elements.</summary>
            Compact = 0,
            /// <summary>The reference 540x960 portrait the layout is designed for.</summary>
            Regular = 1,
            /// <summary>Tablet or a resized desktop window: room for more per row.</summary>
            Wide = 2
        }

        /// <summary>
        /// Density bucket for a resolved panel width, in reference units rather
        /// than device pixels — the panel has already scaled by the time a layout
        /// asks. The thresholds bracket the 540-unit reference width.
        /// </summary>
        public static Density DensityFor(float resolvedWidth)
        {
            if (float.IsNaN(resolvedWidth) || resolvedWidth <= 0f)
            {
                return Density.Regular;
            }
            if (resolvedWidth < 420f)
            {
                return Density.Compact;
            }
            return resolvedWidth > 760f ? Density.Wide : Density.Regular;
        }

        public static void StylePanel(VisualElement panel)
        {
            panel.style.backgroundColor = PanelBg;
            SetRadius(panel, RADIUS_MD);
            SetBorder(panel, PanelBorder, 1f);
            SetPadding(panel, SPACE_SM, SPACE_MD);
            AddElevation(panel, ELEVATION_MID);
        }

        /// <summary>Frosted glassmorphism container with translucent background and specular top border.</summary>
        public static void StyleGlassPanel(VisualElement panel, bool glowing = false)
        {
            panel.style.backgroundColor = GlassBg;
            SetRadius(panel, RADIUS_LG);
            SetBorder(panel, glowing ? AccentGlow : GlassBorder, 1f);
            panel.style.borderTopColor = GlassHighlight;
            panel.style.borderTopWidth = 1.5f;
            SetPadding(panel, SPACE_MD, SPACE_LG);
            // Glass sits highest: it is the layer the player is meant to act on.
            AddElevation(panel, ELEVATION_HIGH);
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
            ApplyFont(button);
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

        /// <summary>
        /// Touch-first scroller chrome. UI Toolkit ships desktop scrollbars —
        /// a light track with stepper arrow buttons — which read as OS chrome on
        /// a portrait phone layout and eat horizontal room. This strips the
        /// steppers, slims the bar and repaints it in the theme; pass
        /// <paramref name="hideBar"/> for swipe strips that need no bar at all.
        /// </summary>
        public static void StyleScrollView(ScrollView scroll, bool hideBar = false)
        {
            scroll.horizontalScrollerVisibility = ScrollerVisibility.Hidden;
            scroll.verticalScrollerVisibility = hideBar
                ? ScrollerVisibility.Hidden
                : ScrollerVisibility.Auto;
            scroll.style.backgroundColor = Color.clear;
            StyleScroller(scroll.verticalScroller);
            StyleScroller(scroll.horizontalScroller);
        }

        private static void StyleScroller(Scroller scroller)
        {
            if (scroller == null)
            {
                return;
            }
            // The steppers are pure desktop affordance — no touch user presses a
            // 10 px arrow. Removing them also reclaims their reserved height.
            scroller.lowButton.style.display = DisplayStyle.None;
            scroller.highButton.style.display = DisplayStyle.None;
            scroller.style.width = SCROLLBAR_THICKNESS;
            scroller.style.backgroundColor = Color.clear;
            SetBorder(scroller, Color.clear, 0f);

            Slider slider = scroller.slider;
            if (slider == null)
            {
                return;
            }
            slider.style.marginTop = 0f;
            slider.style.marginBottom = 0f;
            slider.style.marginLeft = 0f;
            slider.style.marginRight = 0f;

            VisualElement tracker = slider.Q("unity-tracker");
            if (tracker != null)
            {
                tracker.style.backgroundColor = TrackBg;
                SetBorder(tracker, Color.clear, 0f);
                SetRadius(tracker, SCROLLBAR_THICKNESS * 0.5f);
            }

            VisualElement dragger = slider.Q("unity-dragger");
            if (dragger != null)
            {
                dragger.style.backgroundColor = ScrollThumb;
                SetBorder(dragger, Color.clear, 0f);
                SetRadius(dragger, SCROLLBAR_THICKNESS * 0.5f);
                dragger.style.width = SCROLLBAR_THICKNESS;
                dragger.style.marginLeft = 0f;
                dragger.style.marginRight = 0f;
            }
        }

        /// <summary>
        /// Raises <paramref name="element"/> off its backdrop by drawing a shadow
        /// plate behind it.
        ///
        /// The plate is a sibling rather than a child, because a child would be
        /// clipped to the element's own rounded bounds and draw on top of its
        /// background. It is inserted once the element is attached to a panel, so
        /// this can be called during construction before there is a parent, and it
        /// tracks the element's geometry so a reflow cannot leave it behind.
        ///
        /// The plate is never pickable, so it cannot intercept a touch aimed at
        /// whatever sits under the raised element.
        /// </summary>
        public static void AddElevation(VisualElement element, float level)
        {
            element.RegisterCallback<AttachToPanelEvent>(_ => AttachShadow(element, level));
        }

        private static void AttachShadow(VisualElement element, float level)
        {
            VisualElement parent = element.parent;
            if (parent == null || element.userData is VisualElement)
            {
                // No parent to draw into, or this element already owns a plate.
                return;
            }

            var shadow = new VisualElement { pickingMode = PickingMode.Ignore };
            shadow.style.position = Position.Absolute;
            shadow.style.backgroundColor = new Color(
                Shadow.r, Shadow.g, Shadow.b, Shadow.a * Mathf.Clamp01(ELEVATION_HIGH / (level + ELEVATION_HIGH)));
            parent.Insert(parent.IndexOf(element), shadow);
            element.userData = shadow;

            // The plate mirrors the element's box, pushed down and spread out. Both
            // grow with the level, which is what reads as "further from the page".
            element.RegisterCallback<GeometryChangedEvent>(evt =>
            {
                float spread = level * 0.5f;
                Rect box = evt.newRect;
                shadow.style.left = element.layout.x - spread;
                shadow.style.top = element.layout.y + level - spread;
                shadow.style.width = box.width + spread * 2f;
                shadow.style.height = box.height + spread * 2f;
                SetRadius(shadow, RADIUS_MD + spread);
            });
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

        /// <summary>
        /// Optional project font, loaded once from Resources.
        ///
        /// The project ships no font asset, so every screen currently renders in
        /// Unity's built-in default. Dropping a .ttf at
        /// Assets/Resources/UI/PoRacerFont.ttf is enough to re-face the whole UI:
        /// every helper here routes through <see cref="ApplyFont"/>, so no screen
        /// needs to know whether a custom face is present.
        /// </summary>
        private static Font _uiFont;
        private static bool _uiFontLoaded;

        private static Font UiFont()
        {
            if (!_uiFontLoaded)
            {
                _uiFontLoaded = true;
                _uiFont = Resources.Load<Font>("UI/PoRacerFont");
            }
            return _uiFont;
        }

        /// <summary>
        /// Applies the project font to an element, if one has been supplied. A
        /// no-op otherwise, which is why callers never have to check first.
        /// </summary>
        public static void ApplyFont(VisualElement element)
        {
            Font font = UiFont();
            if (font != null)
            {
                element.style.unityFont = font;
            }
        }

        /// <summary>
        /// Label for a value that changes every frame — a clock, a distance, an ELO
        /// score.
        ///
        /// Unity's default face has proportional digits, so "11.1" is visibly
        /// narrower than "88.8" and a running counter jitters, dragging whatever
        /// sits beside it back and forth. Until a font with tabular figures is
        /// supplied, the fix is to stop the label from resizing at all: reserve the
        /// width the widest value needs and align inside it.
        /// </summary>
        public static Label MakeNumericLabel(float reservedWidth, float fontSize, Color color)
        {
            var label = new Label { pickingMode = PickingMode.Ignore };
            label.style.width = reservedWidth;
            label.style.flexShrink = 0f;
            label.style.fontSize = fontSize;
            label.style.color = color;
            label.style.unityTextAlign = TextAnchor.MiddleRight;
            ApplyFont(label);
            return label;
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
            ApplyFont(label);
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
            float scale = rootWidth / Screen.width;
            float leftInset = Mathf.Max(0f, area.xMin * scale);
            float rightInset = Mathf.Max(0f, (Screen.width - area.xMax) * scale);
            float topInset = Mathf.Max(0f, (Screen.height - area.yMax) * scale);
            float bottomInset = Mathf.Max(0f, area.yMin * scale);

            // Minimum notch / status padding on portrait phones
            if (Screen.height > Screen.width)
            {
                topInset = Mathf.Max(topInset, 6f);
                bottomInset = Mathf.Max(bottomInset, 8f);
            }

            safe.style.left = leftInset;
            safe.style.right = rightInset;
            safe.style.top = topInset;
            safe.style.bottom = bottomInset;
        }

        /// <summary>
        /// Pointer-over brightening and tactile touch-press scale for buttons.
        /// </summary>
        public static void AddHover(Button button, bool accent = false)
        {
            Color baseColor = accent ? Accent : ButtonBg;
            Color hoverColor = Color.Lerp(baseColor, Color.white, 0.18f);
            button.RegisterCallback<PointerEnterEvent>(_ => button.style.backgroundColor = hoverColor);
            button.RegisterCallback<PointerLeaveEvent>(_ =>
            {
                button.style.backgroundColor = baseColor;
                button.style.scale = new Scale(Vector3.one);
            });
            button.RegisterCallback<PointerDownEvent>(_ => button.style.scale = new Scale(new Vector3(0.96f, 0.96f, 1f)));
            button.RegisterCallback<PointerUpEvent>(_ => button.style.scale = new Scale(Vector3.one));
        }

        /// <summary>
        /// Fade + rise entrance. The stagger delay is baked into one animation
        /// instead of a scheduled callback, so an element can never be stranded
        /// at zero opacity if its delayed callback is dropped.
        /// </summary>
        public static void PlayEnter(VisualElement element, int delayMs)
        {
            PlayEnter(element, delayMs, ENTER_SLIDE_PX);
        }

        /// <summary>
        /// Same entrance with an explicit rise distance, for panels that need a
        /// taller travel than a list row.
        /// </summary>
        public static void PlayEnter(VisualElement element, int delayMs, float slidePixels)
        {
            float delay = delayMs < 0 ? 0f : delayMs;
            float total = delay + ENTER_MS;
            element.style.opacity = 0f;
            element.style.translate = new Translate(0f, slidePixels);
            element.experimental.animation.Start(0f, 1f, (int)total, (target, value) =>
            {
                float linear = Mathf.Clamp01((value * total - delay) / ENTER_MS);
                float remaining = 1f - linear;
                float eased = 1f - remaining * remaining * remaining;
                target.style.opacity = eased;
                target.style.translate = new Translate(0f, slidePixels * (1f - eased));
            });
        }
    }
}
