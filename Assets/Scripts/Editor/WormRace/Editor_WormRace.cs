#if UNITY_EDITOR
using System;
using System.IO;
using PoRacer.WormRace;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using VContainer;
using Object = UnityEngine.Object;

namespace PoRacer.WormRace.EditorTools
{
    /// <summary>
    /// CLI driver for SCN_WORM_RACE, in the pattern of Editor_SmokeRace / Editor_WalkExam:
    /// opens the scene, tells the next play session what to do (WormRaceRequest), enters
    /// play mode, watches WormRaceModel until the series or self-test is finished, then
    /// leaves play mode. Results are in Logs/wormrace_*.json; <see cref="Status"/> prints
    /// the latest one.
    ///
    /// Entering play mode reloads the domain, so the job lives in SessionState and
    /// <see cref="Resume"/> re-arms the watcher on the other side of the reload.
    ///
    /// Invoke:
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Start(5);"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"zero\");"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"yaw\");"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.SelfTest(\"pitch\");"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Status();"
    ///   unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_WormRace.Stop();"
    /// </summary>
    public static class Editor_WormRace
    {
        private const string JOB_KEY = "PoRacer.WormRace.Job";
        private const string LAST_KEY = "PoRacer.WormRace.LastResult";
        // Per race: 60 s limit + 3 s countdown + 4 s results hold + spawn, with headroom
        // for an editor that cannot keep real time at 200 Hz physics.
        private const float SECONDS_PER_RACE_BUDGET = 120f;
        private const float SELF_TEST_OVERHEAD_SECONDS = 60f;
        private const float STARTUP_BUDGET_SECONDS = 60f;

        [Serializable]
        private sealed class Job
        {
            public string mode;
            public float amount;
            public string startedAt;
            public double startedAtEditorTime;
            public float budgetSeconds;
        }

        private static Job _job;
        private static WormRaceModel _model;

        /// <summary>Runs <paramref name="races"/> races back to back.</summary>
        public static string Start(int races = 5)
        {
            int count = Mathf.Max(1, races);
            return Launch(WormRaceMode.Race, count, STARTUP_BUDGET_SECONDS + count * SECONDS_PER_RACE_BUDGET);
        }

        /// <summary>
        /// Every racer in WormRaceSettings is tested, each in its own physics, and every one
        /// must pass (a racer without a brain too: the tests use fixed actions).
        /// "zero": every worm at zero action must lie still and straight on the floor.
        /// "yaw":  j0_yaw = +0.5 rad must swing segment 1 to Unity +x.
        /// "pitch": j0_pitch = +0.5 rad must lift segment 1 relative to the head.
        /// <paramref name="seconds"/> 0 uses the defaults (5 s zero, 2 s sign tests).
        /// </summary>
        public static string SelfTest(string mode = "zero", float seconds = 0f)
        {
            WormRaceMode parsed;
            switch ((mode ?? string.Empty).Trim().ToLowerInvariant())
            {
                case "zero":
                    parsed = WormRaceMode.ZeroActionTest;
                    break;
                case "yaw":
                    parsed = WormRaceMode.YawSignTest;
                    break;
                case "pitch":
                    parsed = WormRaceMode.PitchSignTest;
                    break;
                default:
                    return $"unknown self-test '{mode}': use \"zero\", \"yaw\" or \"pitch\"";
            }
            return Launch(parsed, Mathf.Max(0f, seconds),
                          STARTUP_BUDGET_SECONDS + SELF_TEST_OVERHEAD_SECONDS + Mathf.Max(0f, seconds));
        }

        public static string Status()
        {
            string pending = SessionState.GetString(JOB_KEY, string.Empty);
            if (!string.IsNullOrEmpty(pending))
            {
                Job job = JsonUtility.FromJson<Job>(pending);
                double elapsed = EditorApplication.timeSinceStartup - job.startedAtEditorTime;
                string phase = _model != null
                    ? $"{_model.Phase} race {_model.RaceNumber}/{_model.PlannedRaces} | {_model.Message}"
                    : "scope not resolved yet";
                return $"running | {job.mode} {job.amount} | {elapsed:0}s of {job.budgetSeconds:0}s budget | {phase}";
            }
            string last = SessionState.GetString(LAST_KEY, string.Empty);
            if (string.IsNullOrEmpty(last))
            {
                return "idle | no worm race this session";
            }
            string[] lines = last.Split('\n');
            string path = lines.Length > 1 ? lines[lines.Length - 1] : string.Empty;
            if (!string.IsNullOrEmpty(path) && File.Exists(path))
            {
                return "done | " + last + "\n" + File.ReadAllText(path);
            }
            return "done | " + last;
        }

        /// <summary>Abandons a running job and leaves play mode.</summary>
        public static string Stop()
        {
            if (string.IsNullOrEmpty(SessionState.GetString(JOB_KEY, string.Empty)))
            {
                return "nothing running";
            }
            Finish("stopped by request", _model != null ? _model.ReportPath : string.Empty);
            return "stopped";
        }

        private static string Launch(WormRaceMode mode, float amount, float budgetSeconds)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "already in play mode - stop it first";
            }
            if (!File.Exists(WormRacePaths.SCENE))
            {
                return $"no scene at {WormRacePaths.SCENE}: run "
                     + "PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build() first";
            }
            if (EditorSceneManager.GetActiveScene().isDirty)
            {
                return "the open scene has unsaved changes - save or discard them first "
                     + "(opening another scene would raise a modal prompt)";
            }
            EditorSceneManager.OpenScene(WormRacePaths.SCENE, OpenSceneMode.Single);

            var job = new Job
            {
                mode = mode.ToString(),
                amount = amount,
                startedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                startedAtEditorTime = EditorApplication.timeSinceStartup,
                budgetSeconds = budgetSeconds,
            };
            SessionState.SetString(JOB_KEY, JsonUtility.ToJson(job));
            SessionState.EraseString(LAST_KEY);
            // Survives the domain reload that entering play mode performs; the running
            // scene consumes and clears it (WormRaceRequest.TryConsume).
            Environment.SetEnvironmentVariable(WormRaceRequest.ENVIRONMENT_VARIABLE,
                                               WormRaceRequest.Format(mode, amount));
            EditorApplication.isPlaying = true;
            return $"queued | {WormRacePaths.SCENE} | {mode} {amount} | budget {budgetSeconds:0}s - poll Status()";
        }

        [InitializeOnLoadMethod]
        private static void Resume()
        {
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return;
            }
            string pending = SessionState.GetString(JOB_KEY, string.Empty);
            if (string.IsNullOrEmpty(pending))
            {
                return;
            }
            _job = JsonUtility.FromJson<Job>(pending);
            _model = null;
            EditorApplication.update += Tick;
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
        }

        private static void Tick()
        {
            if (!EditorApplication.isPlaying || _job == null)
            {
                return;
            }
            if (_model == null)
            {
                _model = ResolveModel();
            }
            double elapsed = EditorApplication.timeSinceStartup - _job.startedAtEditorTime;
            if (_model != null && _model.IsFinished)
            {
                Finish($"{_model.Phase} | {_model.Message}", _model.ReportPath);
                return;
            }
            if (elapsed > _job.budgetSeconds)
            {
                Finish($"timeout after {elapsed:0}s | phase {(_model != null ? _model.Phase.ToString() : "no scope")}",
                       _model != null ? _model.ReportPath : string.Empty);
            }
        }

        private static WormRaceModel ResolveModel()
        {
            var scope = Object.FindAnyObjectByType<WormRaceLifetimeScope>();
            if (scope == null || scope.Container == null)
            {
                return null;
            }
            return scope.Container.Resolve<WormRaceModel>();
        }

        private static void OnPlayModeChanged(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.ExitingPlayMode
                && !string.IsNullOrEmpty(SessionState.GetString(JOB_KEY, string.Empty)))
            {
                Finish("play mode exited before the job finished", _model != null ? _model.ReportPath : string.Empty);
            }
        }

        private static void Finish(string summary, string reportPath)
        {
            SessionState.EraseString(JOB_KEY);
            SessionState.SetString(LAST_KEY, summary + "\n" + (reportPath ?? string.Empty));
            Environment.SetEnvironmentVariable(WormRaceRequest.ENVIRONMENT_VARIABLE, null);
            EditorApplication.update -= Tick;
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            _job = null;
            if (EditorApplication.isPlaying)
            {
                EditorApplication.isPlaying = false;
            }
        }
    }
}
#endif
