using UnityEditor;
using UnityEditor.Build;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// One-shot configuration of everything Google Play needs from PlayerSettings:
    /// application id, version, SDK levels, architecture, orientation, and the
    /// launcher icons.
    ///
    /// It is a MENU ITEM rather than an [InitializeOnLoad] because it writes
    /// ProjectSettings.asset, and settings that rewrite themselves on every domain
    /// reload cannot be overridden by hand. Run it once, or again after changing the
    /// icon art; the build tools re-assert identity and signing on every build but
    /// deliberately never touch the icons.
    ///
    /// Icon layout mirrors what Android consumes:
    ///   Adaptive (API 26+) — two layers, background first then foreground, at the
    ///     six densities Unity asks for. The foreground art must stay inside the
    ///     middle 66% of its canvas; launchers mask the rest away, and every OEM
    ///     masks it to a different shape.
    ///   Round / Legacy — the single pre-adaptive bitmap, for older launchers.
    /// </summary>
    public static class Editor_ConfigureAndroidRelease
    {
        private const string ICON_DIR = "Assets/Icons/";
        private const string ADAPTIVE_BACKGROUND = ICON_DIR + "AppIcon_Adaptive_Background.png";
        private const string ADAPTIVE_FOREGROUND = ICON_DIR + "AppIcon_Adaptive_Foreground.png";
        private const string LEGACY = ICON_DIR + "AppIcon_Legacy.png";

        private const string VERSION = "1.0.0";
        private const int VERSION_CODE = 1;

        public static void Apply()
        {
            PlayerSettings.companyName = "Punkouter Software";
            PlayerSettings.productName = "PoRacer";
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, Editor_BuildAndroidAAB.APP_ID);
            PlayerSettings.bundleVersion = VERSION;
            PlayerSettings.Android.bundleVersionCode = VERSION_CODE;

            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
            PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions)36;
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.SetIl2CppCompilerConfiguration(NamedBuildTarget.Android, Il2CppCompilerConfiguration.Release);

            // Portrait-only: this project's UI is laid out against a 9:16 reference,
            // so a landscape rotation is not a degraded experience, it is a broken one.
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;
            PlayerSettings.allowedAutorotateToPortrait = true;
            PlayerSettings.allowedAutorotateToPortraitUpsideDown = false;
            PlayerSettings.allowedAutorotateToLandscapeLeft = false;
            PlayerSettings.allowedAutorotateToLandscapeRight = false;


            // --- Download size ---
            // The first signed bundle came out at 109 MB, and the build report put
            // 261.8 MB of its 306 MB uncompressed payload in Textures, almost all
            // of it Boy_Character_mujoco.glb expanding from 30 MB on disk to 250 MB
            // in the build. Nothing was setting a mobile texture format, so those
            // textures shipped effectively uncompressed. ASTC is the right target
            // for an ARM64, API 26+ app: universally supported on that floor, and
            // far smaller than the uncompressed fallback.
            EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;

            // --- Code size ---
            // libil2cpp.so was 112.8 MB uncompressed with managed stripping unset,
            // which leaves it at Minimal. Low is safe here ONLY because Assets/link.xml
            // preserves the assemblies that are reached by reflection rather than by
            // a static reference - VContainer injection, MessagePipe brokers,
            // ML-Agents policy selection, Inference Engine backends, MuJoCo
            // components. Do not raise this past Low without extending that file
            // and testing on a device: the failure mode is a build that installs
            // and then dies on a null resolve, which never reproduces in the editor.
            PlayerSettings.SetManagedStrippingLevel(NamedBuildTarget.Android, ManagedStrippingLevel.Low);
            PlayerSettings.SetIl2CppCodeGeneration(NamedBuildTarget.Android, Il2CppCodeGeneration.OptimizeSize);

            // Signing paths only — Unity never serializes the passwords.
            PlayerSettings.Android.useCustomKeystore = true;
            PlayerSettings.Android.keystoreName = Editor_BuildAndroidAAB.KEYSTORE_PATH;
            PlayerSettings.Android.keyaliasName = Editor_BuildAndroidAAB.KEYALIAS;

            string iconReport = ApplyIcons();
            AssetDatabase.SaveAssets();

            Debug.Log($"ANDROID CONFIG RESULT: id={Editor_BuildAndroidAAB.APP_ID} v{VERSION} " +
                      $"(code {VERSION_CODE}) min=26 target=36 arch=ARM64 IL2CPP " +
                      $"tex=ASTC strip=Low codegen=OptimizeSize | {iconReport}");
        }

        private static string ApplyIcons()
        {
            var background = AssetDatabase.LoadAssetAtPath<Texture2D>(ADAPTIVE_BACKGROUND);
            var foreground = AssetDatabase.LoadAssetAtPath<Texture2D>(ADAPTIVE_FOREGROUND);
            var legacy = AssetDatabase.LoadAssetAtPath<Texture2D>(LEGACY);

            if (background == null || foreground == null || legacy == null)
            {
                return "ICONS SKIPPED — missing art under " + ICON_DIR;
            }

            var report = new System.Text.StringBuilder("icons:");
            foreach (PlatformIconKind kind in PlayerSettings.GetSupportedIconKinds(NamedBuildTarget.Android))
            {
                PlatformIcon[] slots = PlayerSettings.GetPlatformIcons(NamedBuildTarget.Android, kind);
                foreach (PlatformIcon slot in slots)
                {
                    // Two layers means adaptive; Unity orders them background-first,
                    // which is the order the generated XML references them in.
                    if (slot.maxLayerCount >= 2)
                    {
                        slot.SetTextures(background, foreground);
                    }
                    else
                    {
                        slot.SetTextures(legacy);
                    }
                }
                PlayerSettings.SetPlatformIcons(NamedBuildTarget.Android, kind, slots);
                report.Append($" {kind}x{slots.Length}");
            }
            return report.ToString();
        }
    }
}
