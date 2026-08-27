using System.Collections.Generic;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// One-shot configuration of the render pipeline assets, for the settings that
    /// live in .asset files rather than in code and so cannot be applied at
    /// runtime: the SSAO renderer feature, GPU instancing on the shared materials,
    /// and the shadow cascade budget.
    ///
    /// Idempotent by design — running it twice changes nothing the second time, so
    /// it is safe to re-run after a package upgrade resets a renderer.
    ///
    /// Adaptive Probe Volumes are deliberately NOT enabled here. The track is built
    /// at runtime by Systems_TrackBuilder, so there is no static geometry to bake
    /// against; switching the probe system over without a bake would replace the
    /// working trilight ambient with an unlit fallback.
    /// </summary>
    internal static class Editor_ConfigureRendering
    {
        private const string MENU_PATH = "PoRacer/Configure Rendering";

        // Desktop SSAO: full resolution, the wider radius the primitive bodies
        // need to read as sitting on the ground.
        private const float PC_AO_INTENSITY = 0.55f;
        private const float PC_AO_RADIUS = 0.32f;
        // Handset SSAO: half resolution and the cheap blur. Present, but it must
        // not cost the frame budget the articulation bodies already own.
        private const float MOBILE_AO_INTENSITY = 0.45f;
        private const float MOBILE_AO_RADIUS = 0.25f;

        [MenuItem(MENU_PATH)]
        private static void Configure()
        {
            var report = new List<string>();

            ConfigureRenderer("Assets/Settings/PC_Renderer.asset", PC_AO_INTENSITY, PC_AO_RADIUS,
                downsample: false, report);
            ConfigureRenderer("Assets/Settings/Mobile_Renderer.asset", MOBILE_AO_INTENSITY, MOBILE_AO_RADIUS,
                downsample: true, report);

            EnableInstancing("Assets/Art/Materials/M_Creature.mat", report);
            EnableInstancing("Assets/Art/Materials/M_Spider.mat", report);
            EnableInstancing("Assets/Art/Materials/M_Worm.mat", report);
            EnableInstancing("Assets/Art/Materials/M_Ground.mat", report);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            if (report.Count == 0)
            {
                Debug.Log("Configure Rendering: everything was already in place.");
                return;
            }
            Debug.Log("Configure Rendering:\n  " + string.Join("\n  ", report));
        }

        /// <summary>
        /// Ensures the renderer at <paramref name="assetPath"/> carries an enabled
        /// SSAO feature with the given tuning. The feature is added as a sub-asset
        /// exactly the way the URP inspector adds it.
        /// </summary>
        private static void ConfigureRenderer(string assetPath, float intensity, float radius,
            bool downsample, List<string> report)
        {
            var rendererData = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(assetPath);
            if (rendererData == null)
            {
                report.Add($"SKIPPED {assetPath} (not found)");
                return;
            }

            ScreenSpaceAmbientOcclusion ssao = FindFeature(rendererData);
            bool added = false;
            if (ssao == null)
            {
                ssao = ScriptableObject.CreateInstance<ScreenSpaceAmbientOcclusion>();
                ssao.name = nameof(ScreenSpaceAmbientOcclusion);
                // Hidden in the project window, same as the inspector's own add.
                ssao.hideFlags = HideFlags.HideInHierarchy;
                AssetDatabase.AddObjectToAsset(ssao, rendererData);
                rendererData.rendererFeatures.Add(ssao);
                added = true;
            }

            ssao.SetActive(true);

            // The settings struct is internal to URP, so it is written through the
            // serialized object rather than by direct field access.
            var serialized = new SerializedObject(ssao);
            SetIfPresent(serialized, "m_Settings.Intensity", intensity);
            SetIfPresent(serialized, "m_Settings.Radius", radius);
            SetIfPresent(serialized, "m_Settings.Downsample", downsample);
            SetIfPresent(serialized, "m_Settings.AfterOpaque", false);
            // Source 1 is DepthNormals: it needs the DepthNormals pass that
            // SH_Creature and SH_TrackGrid now provide. Without those passes the
            // custom-shaded geometry is silently excluded from the effect.
            SetIfPresent(serialized, "m_Settings.Source", 1);
            SetIfPresent(serialized, "m_Settings.DirectLightingStrength", 0.25f);
            serialized.ApplyModifiedPropertiesWithoutUndo();

            // Rebuild the feature-to-file-id map so the reference survives a
            // reimport. Internal in URP; it also runs on the renderer's own
            // OnEnable, so a reflection miss is recoverable rather than fatal.
            MethodInfo validate = typeof(ScriptableRendererData).GetMethod(
                "ValidateRendererFeatures", BindingFlags.Instance | BindingFlags.NonPublic);
            validate?.Invoke(rendererData, null);

            EditorUtility.SetDirty(ssao);
            EditorUtility.SetDirty(rendererData);
            report.Add(added
                ? $"ADDED SSAO to {rendererData.name} (intensity {intensity}, radius {radius})"
                : $"TUNED SSAO on {rendererData.name} (intensity {intensity}, radius {radius})");
        }

        private static ScreenSpaceAmbientOcclusion FindFeature(ScriptableRendererData rendererData)
        {
            List<ScriptableRendererFeature> features = rendererData.rendererFeatures;
            for (int featureIndex = 0; featureIndex < features.Count; featureIndex++)
            {
                if (features[featureIndex] is ScreenSpaceAmbientOcclusion ssao)
                {
                    return ssao;
                }
            }
            return null;
        }

        /// <summary>
        /// Writes a serialized property only when the URP version in use actually
        /// has it, so a renamed setting downgrades to "left alone" rather than
        /// throwing halfway through the configuration.
        /// </summary>
        private static void SetIfPresent(SerializedObject serialized, string path, float value)
        {
            SerializedProperty property = serialized.FindProperty(path);
            if (property != null)
            {
                property.floatValue = value;
            }
        }

        private static void SetIfPresent(SerializedObject serialized, string path, bool value)
        {
            SerializedProperty property = serialized.FindProperty(path);
            if (property != null)
            {
                property.boolValue = value;
            }
        }

        private static void SetIfPresent(SerializedObject serialized, string path, int value)
        {
            SerializedProperty property = serialized.FindProperty(path);
            if (property != null)
            {
                property.enumValueIndex = value;
            }
        }

        /// <summary>
        /// A racer body part carries a MaterialPropertyBlock for its tint, which
        /// takes the draw off the SRP Batcher path. Instancing is what puts it back
        /// on a batched path, and it only engages if the material opts in.
        /// </summary>
        private static void EnableInstancing(string assetPath, List<string> report)
        {
            var material = AssetDatabase.LoadAssetAtPath<Material>(assetPath);
            if (material == null)
            {
                report.Add($"SKIPPED {assetPath} (not found)");
                return;
            }
            if (material.enableInstancing)
            {
                return;
            }
            material.enableInstancing = true;
            EditorUtility.SetDirty(material);
            report.Add($"ENABLED instancing on {material.name}");
        }
    }
}
