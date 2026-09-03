using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Builds the Android APK for on-device testing — the artifact you sideload
    /// with adb, as opposed to the .aab that goes to Play. Same scenes and same
    /// signing key as the bundle, so what you test is what you ship.
    /// Reports through the "BUILD RESULT:" line in the editor log.
    /// </summary>
    public static class Editor_BuildAndroid
    {
        private const string OUTPUT_PATH = "Builds/Android/PoRacer.apk";

        [MenuItem("PoRacer/Build Android APK")]
        public static void Build()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("BUILD RESULT: Aborted — exit Play mode first.");
                return;
            }

            string password = Editor_BuildAndroidAAB.ResolveKeystorePassword();
            if (string.IsNullOrEmpty(password))
            {
                Debug.LogError("BUILD RESULT: Aborted — no keystore password. Set " +
                               Editor_BuildAndroidAAB.PASS_ENV_VAR + " or create the .pass file beside " +
                               Editor_BuildAndroidAAB.KEYSTORE_PATH);
                return;
            }

            // A present .pass with a missing .jks — a half-restored release folder —
            // satisfies the check above, so without this guard the build runs for
            // minutes and then dies inside Gradle on a signing error.
            if (!System.IO.File.Exists(Editor_BuildAndroidAAB.KEYSTORE_PATH))
            {
                Debug.LogError("BUILD RESULT: Aborted — keystore not found at " +
                               Editor_BuildAndroidAAB.KEYSTORE_PATH);
                return;
            }

            var scenes = Editor_BuildAndroidAAB.ResolveShipScenes();
            if (scenes == null)
            {
                return;
            }

            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, Editor_BuildAndroidAAB.APP_ID);
            PlayerSettings.Android.useCustomKeystore = true;
            PlayerSettings.Android.keystoreName = Editor_BuildAndroidAAB.KEYSTORE_PATH;
            PlayerSettings.Android.keystorePass = password;
            PlayerSettings.Android.keyaliasName = Editor_BuildAndroidAAB.KEYALIAS;
            PlayerSettings.Android.keyaliasPass = password;

            // The AAB builder leaves buildAppBundle = true persisted in
            // EditorUserBuildSettings. Without this the "APK" build silently emits an
            // app bundle to PoRacer.apk, which adb cannot install.
            EditorUserBuildSettings.buildAppBundle = false;

            // Same texture format and stripping as the bundle. A sideloaded APK is
            // how the release build gets checked on real hardware, so it has to be
            // built the same way - an APK with different compression or stripping
            // would be testing something the store never receives.
            EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;
            PlayerSettings.SetManagedStrippingLevel(NamedBuildTarget.Android, ManagedStrippingLevel.Low);
            PlayerSettings.SetIl2CppCodeGeneration(NamedBuildTarget.Android, Il2CppCodeGeneration.OptimizeSize);

            var options = new BuildPlayerOptions
            {
                scenes = scenes.ToArray(),
                locationPathName = OUTPUT_PATH,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None,
            };

            BuildReport report;
            try
            {
                report = BuildPipeline.BuildPlayer(options);
            }
            finally
            {
                PlayerSettings.Android.keystorePass = string.Empty;
                PlayerSettings.Android.keyaliasPass = string.Empty;
            }

            BuildSummary summary = report.summary;
            Debug.Log($"BUILD RESULT: {summary.result} | errors={summary.totalErrors} | " +
                      $"size={summary.totalSize / (1024 * 1024)}MB | " +
                      $"time={summary.totalTime.TotalMinutes:F1}min | {OUTPUT_PATH}");
        }
    }
}
