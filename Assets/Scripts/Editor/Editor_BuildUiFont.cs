#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityEngine.TextCore.Text;
using UnityEngine.TextCore.LowLevel;
using UnityEngine.UIElements;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Builds the project's own UI font asset and binds it to RaceHudPanelSettings.
    ///
    /// WHY THIS EXISTS. RaceHudPanelSettings had no textSettings, so every screen drew
    /// with whatever font asset the PLATFORM defaults to - and that is not the same asset
    /// on both sides. In the editor it resolves to the Windows system fonts (Inter, Arial,
    /// Segoe, all 1024x1024 with multi-atlas enabled); in an Android player it resolves to
    /// the runtime default, whose atlas fills. Measured on a Pixel 9 Pro:
    ///
    ///   I/Unity: Texture is too small, consider reducing the sampling point size or
    ///            augmenting the atlas's size in the Font Asset
    ///
    /// and the consequence was not merely ugly text. Elements in this UI size to their
    /// content, so an element whose glyphs never made it into the atlas collapses: on
    /// device the three map cards rendered as a blank gap, and two of the ten roster rows
    /// had no height at all - IsaacBox and MojucuBoy could be raced but could not be seen
    /// or configured. Nothing failed in the editor, and the only clue in the log was that
    /// one warning.
    ///
    /// The fix is to stop depending on the platform default: ship one font asset, with an
    /// atlas big enough for this UI and multi-atlas support so it can grow rather than
    /// silently drop glyphs.
    ///
    /// LiberationSans is Unity's own bundled font (SIL Open Font License), so this adds no
    /// new redistribution question.
    ///
    /// Invoke: unity command eval --code "PoRacer.EditorTools.Editor_BuildUiFont.Build()"
    /// Re-run after changing the size ramp in UiTheme - the sampling point size below wants
    /// to sit at or above the largest token that is drawn often.
    /// </summary>
    public static class Editor_BuildUiFont
    {
        private const string FONT_DIR = "Assets/UI";
        private const string FONT_ASSET_PATH = FONT_DIR + "/PoRacerFont SDF.asset";
        private const string TEXT_SETTINGS_PATH = FONT_DIR + "/PoRacerTextSettings.asset";
        private const string PANEL_SETTINGS_PATH = FONT_DIR + "/RaceHudPanelSettings.asset";

        // 90 covers FONT_COUNTDOWN (68 units) at the 1.25 scale ceiling. Glyphs are SDF, so
        // this is the rasterisation size, not a cap on how large text can draw.
        private const int SAMPLING_POINT_SIZE = 90;
        private const int ATLAS_PADDING = 9;
        // 2048, not the default 1024: the whole ramp at this sampling size did not fit in
        // 1024, which is the failure this exists to remove. Multi-atlas below is the
        // belt-and-braces - it lets the asset add a second texture instead of dropping a
        // glyph, which is what turns a font problem into an invisible-UI problem.
        private const int ATLAS_SIZE = 2048;

        public static string Build()
        {
            Font source = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (source == null)
            {
                // Older editors named it Arial.ttf.
                source = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
            if (source == null)
            {
                return "ABORT: no builtin font found (tried LegacyRuntime.ttf, Arial.ttf)";
            }

            FontAsset font = FontAsset.CreateFontAsset(
                source,
                SAMPLING_POINT_SIZE,
                ATLAS_PADDING,
                GlyphRenderMode.SDFAA,
                ATLAS_SIZE,
                ATLAS_SIZE,
                AtlasPopulationMode.Dynamic,
                enableMultiAtlasSupport: true);
            if (font == null)
            {
                return "ABORT: CreateFontAsset returned null";
            }
            font.name = "PoRacerFont SDF";

            // Replace in place so the .meta GUID survives and the PanelSettings reference
            // does not break on a rebuild.
            var existing = AssetDatabase.LoadAssetAtPath<FontAsset>(FONT_ASSET_PATH);
            if (existing != null)
            {
                EditorUtility.CopySerialized(font, existing);
                font = existing;
                EditorUtility.SetDirty(font);
            }
            else
            {
                AssetDatabase.CreateAsset(font, FONT_ASSET_PATH);
            }
            // The atlas texture and material are sub-assets; without this they are lost.
            if (font.atlasTextures != null)
            {
                foreach (Texture2D atlas in font.atlasTextures)
                {
                    if (atlas != null && !AssetDatabase.IsSubAsset(atlas)
                        && AssetDatabase.GetAssetPath(atlas) != FONT_ASSET_PATH)
                    {
                        AssetDatabase.AddObjectToAsset(atlas, font);
                    }
                }
            }
            if (font.material != null && AssetDatabase.GetAssetPath(font.material) != FONT_ASSET_PATH)
            {
                AssetDatabase.AddObjectToAsset(font.material, font);
            }

            var settings = AssetDatabase.LoadAssetAtPath<PanelTextSettings>(TEXT_SETTINGS_PATH);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<PanelTextSettings>();
                AssetDatabase.CreateAsset(settings, TEXT_SETTINGS_PATH);
            }
            settings.defaultFontAsset = font;
            EditorUtility.SetDirty(settings);

            var panel = AssetDatabase.LoadAssetAtPath<PanelSettings>(PANEL_SETTINGS_PATH);
            if (panel == null)
            {
                return "ABORT: " + PANEL_SETTINGS_PATH + " not found";
            }
            panel.textSettings = settings;
            EditorUtility.SetDirty(panel);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            return string.Format(
                "UI FONT RESULT: {0} atlas {1}x{2} multiAtlas={3} | textSettings bound to {4}",
                font.name, font.atlasWidth, font.atlasHeight,
                font.isMultiAtlasTexturesEnabled, panel.name);
        }
    }
}
#endif
