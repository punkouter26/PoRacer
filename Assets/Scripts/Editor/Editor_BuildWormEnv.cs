using UnityEditor;
using UnityEngine;

namespace PoRacer.Editor
{
    /// <summary>
    /// Builds the headless training environment from an explicit scene list so the
    /// training build never depends on the race app's EditorBuildSettings order.
    /// </summary>
    public static class Editor_BuildWormEnv
    {
        [MenuItem("PoRacer/Build Worm Training Env")]
        public static void Build()
        {
            BuildEnv("Assets/Scenes/SCN_TRAIN_WORM.unity", "Builds/WormEnv/WormEnv.exe");
        }

        [MenuItem("PoRacer/Build Spider Training Env")]
        public static void BuildSpider()
        {
            BuildEnv("Assets/Scenes/SCN_TRAIN_SPIDER.unity", "Builds/SpiderEnv/SpiderEnv.exe");
        }

        // Secondary output path so a terrain env can build while SpiderEnv.exe is training.
        [MenuItem("PoRacer/Build Spider Training Env (alt path)")]
        public static void BuildSpiderAlt()
        {
            BuildEnv("Assets/Scenes/SCN_TRAIN_SPIDER.unity", "Builds/SpiderEnv2/SpiderEnv.exe");
        }

        private static void BuildEnv(string scenePath, string outputPath)
        {
            var options = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = outputPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"Training env build: {report.summary.result}, {report.summary.totalErrors} errors, output {outputPath}");
        }
    }
}
