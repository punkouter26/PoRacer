#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Fire-and-poll wrapper around the two Android builders, so a build can be driven
    /// from MCP / the Unity CLI.
    ///
    /// WHY THIS EXISTS. The CLI's `eval` runs on the main thread with a 5 second budget.
    /// A player build takes minutes, so calling Editor_BuildAndroid.Build() through eval
    /// does not slow down - it fails outright with "Main thread operation timed out after
    /// 5000ms", and no artifact is written. With the editor menu items gone, that left no
    /// way to build through the bridge at all except by re-implementing the builders'
    /// settings by hand against the CLI's generic `build` command - which does no signing
    /// and no scene curation, so what it produces is not the artifact that ships.
    ///
    /// <see cref="Start"/> returns immediately, having queued the real builder on the next
    /// editor tick; <see cref="Status"/> is then polled the same way the CLI's own
    /// build_status is:
    ///
    ///   unity command eval --code "PoRacer.EditorTools.Editor_BuildAsync.Start(\"apk\")"
    ///   unity command eval --code "PoRacer.EditorTools.Editor_BuildAsync.Status()"
    ///
    /// Status() is NOT answerable while the build is running: BuildPipeline owns the main
    /// thread, so the polling eval hits the same 5 s timeout and errors. That error means
    /// the build is healthy, not broken. Poll the ARTIFACT's mtime from the shell while it
    /// runs, and call Status() once it stops moving to get the verdict.
    ///
    /// Success is judged by the ARTIFACT, not by the return of Build(): both builders
    /// abort early and return normally when the editor is in play mode or the keystore is
    /// missing, so a caller that trusts control flow reports a build that never happened.
    /// This compares the output file's timestamp and size across the call instead.
    /// </summary>
    public static class Editor_BuildAsync
    {
        private const string APK_PATH = "Builds/Android/PoRacer.apk";
        private const string AAB_PATH = "Builds/Android/PoRacer.aab";
        // Training envs. These MUST go through Editor_BuildSharedTrainingScene, which names
        // its scene explicitly - the CLI's generic `build` command silently ignores --scenes
        // and ships whatever EditorBuildSettings holds. That produced an "env" running
        // SCN_RACE_FLAT: no agents, so the Academy never initialised, so mlagents-learn sat
        // there until it timed out with UnityTimeOutException and no clue in any log.
        private const string ALLENV_PATH = "Builds/AllEnv/AllEnv.exe";
        private const string FOCUSEDENV_PATH = "Builds/FocusedEnv/FocusedEnv.exe";

        private static string _state = "idle";
        private static string _detail = string.Empty;

        /// <summary>
        /// Queues a build. <paramref name="target"/> is "apk" or "aab". Returns the state
        /// the caller should expect to see from <see cref="Status"/>, or a refusal.
        /// </summary>
        public static string Start(string target)
        {
            if (_state == "running")
            {
                return "REFUSED: a build is already running — poll Status() until it is not";
            }

            string kind = (target ?? string.Empty).Trim().ToLowerInvariant();
            if (kind != "apk" && kind != "aab" && kind != "allenv" && kind != "focusedenv")
            {
                return "REFUSED: target must be \"apk\", \"aab\", \"allenv\" or \"focusedenv\", got \""
                       + target + "\"";
            }

            // Both builders abort on this, but they do it minutes of Gradle later in the
            // AAB case and it reads as a mystery no-op. Say it up front.
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "REFUSED: exit play mode first";
            }

            string path = kind == "apk" ? APK_PATH
                        : kind == "aab" ? AAB_PATH
                        : kind == "allenv" ? ALLENV_PATH
                        : FOCUSEDENV_PATH;
            DateTime before = File.Exists(path) ? File.GetLastWriteTimeUtc(path) : DateTime.MinValue;

            _state = "running";
            _detail = kind + " queued at " + DateTime.Now.ToString("HH:mm:ss");

            // A one-shot EditorApplication.update handler, NOT delayCall. Measured
            // through the CLI bridge on 2026-09-03: an update handler fires, and
            // delayCall never does - a delayCall build sat in "running" forever and
            // wrote nothing. Unsubscribe first thing so it runs exactly once.
            EditorApplication.CallbackFunction tick = null;
            tick = () =>
            {
                EditorApplication.update -= tick;
                Run(kind, path, before);
            };
            EditorApplication.update += tick;

            return "queued " + kind + " — poll Editor_BuildAsync.Status()";
        }

        /// <summary>Current state: idle | running | succeeded | failed, plus detail.</summary>
        public static string Status()
        {
            return _state + (string.IsNullOrEmpty(_detail) ? string.Empty : " | " + _detail);
        }

        private static void Run(string kind, string path, DateTime before)
        {
            var started = DateTime.Now;
            try
            {
                switch (kind)
                {
                    case "apk":
                        Editor_BuildAndroid.Build();
                        break;
                    case "aab":
                        Editor_BuildAndroidAAB.Build();
                        break;
                    case "allenv":
                        PoRacer.Editor.Editor_BuildSharedTrainingScene.BuildEnv();
                        break;
                    default:
                        PoRacer.Editor.Editor_BuildSharedTrainingScene.BuildFocusedEnv();
                        break;
                }
            }
            catch (Exception e)
            {
                _state = "failed";
                _detail = kind + " threw: " + e.GetType().Name + ": " + e.Message;
                Debug.LogError("ASYNC BUILD RESULT: " + _detail);
                return;
            }

            // The artifact is the evidence. An unchanged timestamp means the builder
            // returned without writing - an early abort it already logged the reason for.
            var info = new FileInfo(path);
            if (!info.Exists || File.GetLastWriteTimeUtc(path) <= before)
            {
                _state = "failed";
                _detail = kind + " wrote no new artifact at " + path +
                          " — check the console for the builder's abort reason";
                Debug.LogError("ASYNC BUILD RESULT: " + _detail);
                return;
            }

            _state = "succeeded";
            _detail = string.Format("{0} {1:0.0} MB at {2} ({3:0.0} min)",
                kind, info.Length / 1048576.0, info.LastWriteTime.ToString("HH:mm:ss"),
                (DateTime.Now - started).TotalMinutes);
            Debug.Log("ASYNC BUILD RESULT: " + _detail);
        }
    }
}
#endif
