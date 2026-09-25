using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using PoRacer.Systems;
using Unity.Profiling;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.TestTools;
using VContainer;
using VContainer.Unity;
using Object = UnityEngine.Object;

namespace PoRacer.Tests
{
    /// <summary>
    /// Plays every non-training scene for a fixed window and fails on any error or
    /// exception it logs. The menu-driven race scene is started the way the menu starts
    /// it. Each run writes per-frame telemetry (frame time, GC per frame, draw calls) to
    /// Logs/SmokeRun/&lt;scene&gt;.csv and one summary row per scene to Logs/SmokeRun/summary.md.
    /// </summary>
    public sealed class SceneSmokeTests
    {
        private const float SETTLE_SECONDS = 2f;
        private const float PLAY_SECONDS = 45f;
        private const int TIMEOUT_MS = 180000;
        private const string REPORT_FOLDER = "Logs/SmokeRun";
        private const string CANARY = "[SmokeRun] log capture check";

        private static readonly string[] ScenePaths =
        {
            "Assets/Scenes/SCN_RACE_FLAT.unity",
            "Assets/Scenes/SCN_WORM_RACE.unity",
            "Assets/Scenes/SCN_QUAD_RACE.unity",
            "Assets/Scenes/SCN_TEST_MOJUCUBOY.unity",
            "Assets/Scenes/SCN_TEST_MOJUCUBOY_RIG.unity"
        };

        private readonly List<string> _errors = new();
        private readonly HashSet<string> _warnings = new();
        private bool _canaryHeard;

        [UnityTest, Timeout(TIMEOUT_MS)]
        public IEnumerator Scene_PlaysWithoutErrors([ValueSource(nameof(ScenePaths))] string scenePath)
        {
            EditorSceneManager.OpenScene(scenePath);
            LogAssert.ignoreFailingMessages = true;

            // Entering play mode reloads the domain, which drops any log handler added
            // before it and resets this fixture, so capture starts once play has begun.
            yield return new EnterPlayMode();
            _errors.Clear();
            _warnings.Clear();
            _canaryHeard = false;
            Application.logMessageReceived += OnLog;
            Debug.LogWarning(CANARY);

            yield return WaitRealtime(SETTLE_SECONDS);
            StartMenuRaceIfPresent();

            var frameMs = new List<float>(4096);
            var gcBytes = new List<long>(4096);
            var drawCalls = new List<long>(4096);
            using (ProfilerRecorder gcRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Allocated In Frame"))
            {
                float endTime = Time.realtimeSinceStartup + PLAY_SECONDS;
                while (Time.realtimeSinceStartup < endTime)
                {
                    yield return null;
                    frameMs.Add(Time.unscaledDeltaTime * 1000f);
                    gcBytes.Add(gcRecorder.LastValue);
                    drawCalls.Add(UnityStats.drawCalls);
                }
            }

            Application.logMessageReceived -= OnLog;
            yield return new ExitPlayMode();
            LogAssert.ignoreFailingMessages = false;

            WriteReport(Path.GetFileNameWithoutExtension(scenePath), frameMs, gcBytes, drawCalls);
            Assert.That(_canaryHeard, Is.True, "Log capture is broken: the canary warning never arrived.");
            Assert.That(_errors, Is.Empty, string.Join("\n\n", _errors));
        }

        private void OnLog(string condition, string stackTrace, LogType type)
        {
            if (condition == CANARY)
            {
                _canaryHeard = true;
            }
            else if (type == LogType.Error || type == LogType.Exception || type == LogType.Assert)
            {
                _errors.Add($"[{type}] {condition}\n{stackTrace}");
            }
            else if (type == LogType.Warning)
            {
                _warnings.Add(condition);
            }
        }

        private static IEnumerator WaitRealtime(float seconds)
        {
            float endTime = Time.realtimeSinceStartup + seconds;
            while (Time.realtimeSinceStartup < endTime)
            {
                yield return null;
            }
        }

        // SCN_RACE_FLAT opens on its menu; the race scenes start their series by themselves.
        private static void StartMenuRaceIfPresent()
        {
            LifetimeScope[] scopes = Object.FindObjectsByType<LifetimeScope>(FindObjectsSortMode.None);
            for (int scopeIndex = 0; scopeIndex < scopes.Length; scopeIndex++)
            {
                IObjectResolver container = scopes[scopeIndex].Container;
                if (container != null && container.TryResolve(out Systems_Spawn spawn))
                {
                    spawn.BeginRacing();
                    return;
                }
            }
        }

        private void WriteReport(string sceneName, List<float> frameMs, List<long> gcBytes, List<long> drawCalls)
        {
            Directory.CreateDirectory(REPORT_FOLDER);
            var csv = new StringBuilder("frame,frame_ms,gc_bytes,draw_calls\n");
            for (int frameIndex = 0; frameIndex < frameMs.Count; frameIndex++)
            {
                csv.Append(frameIndex).Append(',')
                    .Append(frameMs[frameIndex].ToString("F2", CultureInfo.InvariantCulture)).Append(',')
                    .Append(gcBytes[frameIndex]).Append(',')
                    .Append(drawCalls[frameIndex]).Append('\n');
            }
            File.WriteAllText(Path.Combine(REPORT_FOLDER, sceneName + ".csv"), csv.ToString());

            var sorted = new List<float>(frameMs);
            sorted.Sort();
            float p99 = sorted.Count > 0 ? sorted[Mathf.Min(sorted.Count - 1, (int)(sorted.Count * 0.99f))] : 0f;
            float max = sorted.Count > 0 ? sorted[sorted.Count - 1] : 0f;
            double totalMs = 0;
            long totalGc = 0;
            long totalDraws = 0;
            int gcFrames = 0;
            for (int frameIndex = 0; frameIndex < frameMs.Count; frameIndex++)
            {
                totalMs += frameMs[frameIndex];
                totalGc += gcBytes[frameIndex];
                totalDraws += drawCalls[frameIndex];
                if (gcBytes[frameIndex] > 0)
                {
                    gcFrames++;
                }
            }
            int frames = Mathf.Max(1, frameMs.Count);
            string summaryPath = Path.Combine(REPORT_FOLDER, "summary.md");
            if (!File.Exists(summaryPath))
            {
                File.WriteAllText(summaryPath,
                    "| When | Scene | Frames | Avg FPS | p99 ms | Max ms | GC KB/frame | GC frames | Draw calls | Errors | Warnings |\n" +
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
            }
            File.AppendAllText(summaryPath, string.Format(CultureInfo.InvariantCulture,
                "| {0:yyyy-MM-dd HH:mm} | {1} | {2} | {3:F1} | {4:F1} | {5:F1} | {6:F2} | {7} | {8} | {9} | {10} |\n",
                DateTime.Now, sceneName, frameMs.Count, 1000.0 * frames / Math.Max(1.0, totalMs), p99, max,
                totalGc / 1024.0 / frames, gcFrames, totalDraws / frames, _errors.Count, _warnings.Count));
        }
    }
}
