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

        public static void Build()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("BUILD RESULT: Aborted — exit Play mode first.");
                return;
            }

            // --- Signing ---
            // The APK is the SIDELOAD artifact: its only job is to run on a device
            // over adb. The .aab is what reaches Play, and only the .aab needs the
            // upload key. So a missing release key is not fatal here — it downgrades
            // this build to Gradle's debug key rather than aborting.
            //
            // This is deliberately NOT symmetric with Editor_BuildAndroidAAB, which
            // still aborts: a bundle signed with a debug key is worthless to Play and
            // shipping one silently would be the real failure. Here the trade is the
            // other way round. The one cost is that a debug-signed APK cannot install
            // OVER a release-signed one — Android rejects the signature change — so
            // the installer has to uninstall first, losing app data.
            string password = Editor_BuildAndroidAAB.ResolveKeystorePassword();
            bool keystorePresent = System.IO.File.Exists(Editor_BuildAndroidAAB.KEYSTORE_PATH);
            bool releaseSigned = keystorePresent && !string.IsNullOrEmpty(password);

            if (!releaseSigned)
            {
                string why = !keystorePresent
                    ? "keystore not found at " + Editor_BuildAndroidAAB.KEYSTORE_PATH
                    : "no keystore password (set " + Editor_BuildAndroidAAB.PASS_ENV_VAR +
                      " or create the .pass file beside the keystore)";
                Debug.LogWarning(
                    "BUILD SIGNING: falling back to the Android DEBUG key — " + why + ".\n" +
                    "This APK is for on-device testing only. It will NOT install over a " +
                    "release-signed build (uninstall first), and it must never be used as " +
                    "a Play artifact. Restore the upload key before building the .aab.");
            }

            var scenes = Editor_BuildAndroidAAB.ResolveShipScenes();
            if (scenes == null)
            {
                return;
            }

            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, Editor_BuildAndroidAAB.APP_ID);
            PlayerSettings.Android.useCustomKeystore = releaseSigned;
            if (releaseSigned)
            {
                PlayerSettings.Android.keystoreName = Editor_BuildAndroidAAB.KEYSTORE_PATH;
                PlayerSettings.Android.keystorePass = password;
                PlayerSettings.Android.keyaliasName = Editor_BuildAndroidAAB.KEYALIAS;
                PlayerSettings.Android.keyaliasPass = password;
            }
            else
            {
                // Leaving a stale keystore path on the settings while useCustomKeystore
                // is false is how a "debug" build still ends up in Gradle's signing
                // config. Clear the whole block, not just the flag.
                PlayerSettings.Android.keystoreName = string.Empty;
                PlayerSettings.Android.keystorePass = string.Empty;
                PlayerSettings.Android.keyaliasName = string.Empty;
                PlayerSettings.Android.keyaliasPass = string.Empty;
            }

            // Every device install gets its own version code. Android refuses to
            // install a lower code over a higher one, so a build that reuses the
            // previous code is a build that silently fails to land on a device that
            // already has one. Bumping here rather than in the configure step means
            // it tracks ARTIFACTS, which is what the device actually sees.
            int versionCode = Editor_BuildAndroidAAB.BumpVersionCode();

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
                      $"time={summary.totalTime.TotalMinutes:F1}min | " +
                      $"v{PlayerSettings.bundleVersion} code={versionCode} | " +
                      $"signing={(releaseSigned ? "upload-key" : "DEBUG-KEY (device test only)")} | " +
                      $"{OUTPUT_PATH}");
        }
    }
}
