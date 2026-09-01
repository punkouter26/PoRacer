using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Builds the signed release Android App Bundle (.aab) for Google Play.
    ///
    /// The upload keystore and its password live OUTSIDE the repo so neither can
    /// be committed:
    ///     C:/Users/punko/Downloads/PoRacer-Release/poracer-upload.jks
    ///     C:/Users/punko/Downloads/PoRacer-Release/poracer-upload.pass   (password, one line)
    ///
    /// Unity deliberately does not serialize keystore passwords into
    /// ProjectSettings.asset, so they must be supplied at build time — that is what
    /// the .pass file (or the PORACER_KEYSTORE_PASS environment variable, which wins if
    /// set) is for. Without either the build ABORTS, rather than producing an
    /// unsigned artifact that Play would reject minutes later.
    ///
    /// Invoked from the PoRacer menu, or headlessly with -executeMethod
    /// PoRacer.EditorTools.Editor_BuildAndroidAAB.Build. Either way the outcome is the
    /// "AAB BUILD RESULT:" line in the editor log.
    /// </summary>
    public static class Editor_BuildAndroidAAB
    {
        private const string OUTPUT_PATH = "Builds/Android/PoRacer.aab";

        /// PERMANENT once the first bundle is uploaded — Play keys the app on it.
        internal const string APP_ID = "com.punkoutersoftware.poracer";

        // internal, not private: the APK builder signs with the same upload key, so
        // an APK installed over a Play build does not hit a signature mismatch.
        internal const string KEYSTORE_PATH = "C:/Users/punko/Downloads/PoRacer-Release/poracer-upload.jks";
        internal const string KEYALIAS = "poracer-upload";
        internal const string PASS_ENV_VAR = "PORACER_KEYSTORE_PASS";

        // Google Play requires new apps and updates to target API 36 from 2026-08-31.
        private const int TARGET_SDK = 36;
        private const int MIN_SDK = 26;

        /// The scenes that belong in a PLAYER, in boot order — index 0 is what the
        /// app opens into. This is an explicit list rather than whatever is ticked
        /// in Build Settings, because Build Settings also carries the training
        /// scenes (SCN_TRAIN_ALL, SCN_TRAIN_HUMANOIDS and SCN_TRAIN_WORM_ROUGH) and shipping those would
        /// both bloat the bundle and, depending on order, boot a tester into a
        /// training rig.
        internal static readonly string[] SHIP_SCENES =
        {
            "Assets/Scenes/SCN_RACE_FLAT.unity",
        };

        [MenuItem("PoRacer/Build Android AAB (Play release)")]
        public static void Build()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("AAB BUILD RESULT: Aborted — exit Play mode first.");
                return;
            }

            string password = ResolveKeystorePassword();
            if (string.IsNullOrEmpty(password))
            {
                Debug.LogError(
                    "AAB BUILD RESULT: Aborted — no keystore password. Set " + PASS_ENV_VAR +
                    " or put the password on the first line of " +
                    Path.ChangeExtension(KEYSTORE_PATH, null) + ".pass");
                return;
            }

            if (!File.Exists(KEYSTORE_PATH))
            {
                Debug.LogError("AAB BUILD RESULT: Aborted — keystore not found at " + KEYSTORE_PATH);
                return;
            }

            List<string> scenes = ResolveShipScenes();
            if (scenes == null)
            {
                return;
            }

            // --- Identity ---
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android, APP_ID);
            PlayerSettings.companyName = "Punkouter Software";
            PlayerSettings.productName = "PoRacer";

            // --- Signing ---
            PlayerSettings.Android.useCustomKeystore = true;
            PlayerSettings.Android.keystoreName = KEYSTORE_PATH;
            PlayerSettings.Android.keystorePass = password;
            PlayerSettings.Android.keyaliasName = KEYALIAS;
            PlayerSettings.Android.keyaliasPass = password;

            // --- Play requirements ---
            PlayerSettings.Android.minSdkVersion = (AndroidSdkVersions)MIN_SDK;
            PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions)TARGET_SDK;
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.SetIl2CppCompilerConfiguration(NamedBuildTarget.Android, Il2CppCompilerConfiguration.Release);

            // .aab, not .apk. This is the switch Play cares about most.
            EditorUserBuildSettings.buildAppBundle = true;
            EditorUserBuildSettings.androidBuildType = AndroidBuildType.Release;
            EditorUserBuildSettings.development = false;

            Directory.CreateDirectory(Path.GetDirectoryName(OUTPUT_PATH));

            var options = new BuildPlayerOptions
            {
                scenes = scenes.ToArray(),
                locationPathName = OUTPUT_PATH,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None,
            };

            Debug.Log($"AAB BUILD START: {APP_ID} v{PlayerSettings.bundleVersion} " +
                      $"(code {PlayerSettings.Android.bundleVersionCode}) " +
                      $"target={TARGET_SDK} min={MIN_SDK} scenes={scenes.Count} " +
                      $"boot={Path.GetFileNameWithoutExtension(scenes[0])}");

            BuildReport report;
            try
            {
                report = BuildPipeline.BuildPlayer(options);
            }
            finally
            {
                // Do not leave the password sitting in the in-memory PlayerSettings.
                PlayerSettings.Android.keystorePass = string.Empty;
                PlayerSettings.Android.keyaliasPass = string.Empty;
            }

            BuildSummary summary = report.summary;
            Debug.Log($"AAB BUILD RESULT: {summary.result} | errors={summary.totalErrors} | " +
                      $"size={summary.totalSize / (1024 * 1024)}MB | " +
                      $"time={summary.totalTime.TotalMinutes:F1}min | {summary.outputPath}");
        }

        /// Verifies every shipping scene is actually on disk and returns them in
        /// boot order. A missing scene is an abort rather than a warning: a bundle
        /// silently short one scene is a crash on a tester's phone, discovered a
        /// day later.
        internal static List<string> ResolveShipScenes()
        {
            var scenes = new List<string>();
            foreach (string path in SHIP_SCENES)
            {
                if (!File.Exists(path))
                {
                    Debug.LogError("BUILD RESULT: Aborted — shipping scene not found: " + path);
                    return null;
                }
                scenes.Add(path);
            }
            return scenes;
        }

        /// Environment variable wins; otherwise read &lt;keystore&gt;.pass beside the keystore.
        internal static string ResolveKeystorePassword()
        {
            string fromEnv = Environment.GetEnvironmentVariable(PASS_ENV_VAR);
            if (!string.IsNullOrEmpty(fromEnv))
            {
                return fromEnv.Trim();
            }

            string passFile = Path.ChangeExtension(KEYSTORE_PATH, null) + ".pass";
            if (File.Exists(passFile))
            {
                return File.ReadAllText(passFile).Trim();
            }

            string sibling = Path.Combine(Path.GetDirectoryName(KEYSTORE_PATH) ?? ".", "keystore.pass");
            return File.Exists(sibling) ? File.ReadAllText(sibling).Trim() : null;
        }
    }
}
