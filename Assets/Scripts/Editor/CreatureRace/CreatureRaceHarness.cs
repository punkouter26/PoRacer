#if UNITY_EDITOR
using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using VContainer;
using VContainer.Unity;
using Object = UnityEngine.Object;

namespace PoRacer.CreatureRace.EditorTools
{
    /// <summary>
    /// CLI driver shared by every creature race scene, in the pattern of Editor_SmokeRace /
    /// Editor_WalkExam: opens the scene, tells the next play session what to do
    /// (CreatureRaceRequest), enters play mode, watches CreatureRaceModel until the series or
    /// self-test is finished, then leaves play mode. Results are in Logs/&lt;prefix&gt;_*.json;
    /// <see cref="Status"/> prints the latest one.
    ///
    /// Each creature exposes thin wrappers (Editor_WormRace, Editor_QuadRace) that pass their
    /// scene and settings. One job runs at a time, whatever the creature.
    ///
    /// Entering play mode reloads the domain, so the job lives in SessionState and
    /// <see cref="Resume"/> re-arms the watcher on the other side of the reload.
    /// </summary>
    public static class CreatureRaceHarness
    {
        private const string JOB_KEY = "PoRacer.CreatureRace.Job";
        private const string LAST_KEY = "PoRacer.CreatureRace.LastResult";
        // Per race: the time limit twice over (countdown, results hold, spawn, and an editor
        // that cannot keep real time at 200 Hz physics), plus a start-up allowance.
        private const float RACE_BUDGET_FACTOR = 2f;
        private const float SELF_TEST_OVERHEAD_SECONDS = 60f;
        private const float STARTUP_BUDGET_SECONDS = 60f;

        [Serializable]
        private sealed class Job
        {
            public string scene;
            public string label;
            public string mode;
            public float amount;
            public string startedAt;
            public double startedAtEditorTime;
            public float budgetSeconds;
        }

        private static Job _job;
        private static CreatureRaceModel _model;

        /// <summary>Runs <paramref name="races"/> races back to back in <paramref name="scenePath"/>.</summary>
        public static string Race(string scenePath, string label, CreatureRaceConfig config, int races)
        {
            int count = Mathf.Max(1, races);
            float perRace = config != null ? config.TimeLimitSeconds * RACE_BUDGET_FACTOR : 120f;
            return Launch(scenePath, label, CreatureRaceRequest.RACE_MODE, count, STARTUP_BUDGET_SECONDS + count * perRace);
        }

        /// <summary>
        /// Runs one of the scene's self-tests, by name or alias ("zero", "yaw", "hip", ...).
        /// <paramref name="seconds"/> 0 uses the test's default.
        /// </summary>
        public static string SelfTest(string scenePath, string label, CreatureRaceConfig config, string key, float seconds)
        {
            CreatureSelfTestDefinition test = config?.FindSelfTest((key ?? string.Empty).Trim());
            if (test == null)
            {
                return $"unknown self-test '{key}': use one of {TestList(config)} (re-run the scene builder if the list is empty)";
            }
            float duration = seconds > 0f ? seconds : test.DefaultSeconds;
            return Launch(scenePath, label, test.Name, Mathf.Max(0f, seconds),
                          STARTUP_BUDGET_SECONDS + SELF_TEST_OVERHEAD_SECONDS + duration);
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
                return $"running | {job.label} {job.mode} {job.amount} | {elapsed:0}s of {job.budgetSeconds:0}s budget | {phase}";
            }
            string last = SessionState.GetString(LAST_KEY, string.Empty);
            if (string.IsNullOrEmpty(last))
            {
                return "idle | no creature race this session";
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

        private static string Launch(string scenePath, string label, string mode, float amount, float budgetSeconds)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "already in play mode - stop it first";
            }
            if (!File.Exists(scenePath))
            {
                return $"no scene at {scenePath}: run the {label} scene builder first";
            }
            if (EditorSceneManager.GetActiveScene().isDirty)
            {
                return "the open scene has unsaved changes - save or discard them first "
                     + "(opening another scene would raise a modal prompt)";
            }
            EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

            var job = new Job
            {
                scene = scenePath,
                label = label,
                mode = mode,
                amount = amount,
                startedAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                startedAtEditorTime = EditorApplication.timeSinceStartup,
                budgetSeconds = budgetSeconds,
            };
            SessionState.SetString(JOB_KEY, JsonUtility.ToJson(job));
            SessionState.EraseString(LAST_KEY);
            // Survives the domain reload that entering play mode performs; the running
            // scene consumes and clears it (CreatureRaceRequest.TryConsume).
            Environment.SetEnvironmentVariable(CreatureRaceRequest.ENVIRONMENT_VARIABLE,
                                               CreatureRaceRequest.Format(mode, amount));
            EditorApplication.isPlaying = true;
            return $"queued | {scenePath} | {mode} {amount} | budget {budgetSeconds:0}s - poll Status()";
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

        private static CreatureRaceModel ResolveModel()
        {
            LifetimeScope[] scopes = Object.FindObjectsByType<LifetimeScope>();
            for (int index = 0; index < scopes.Length; index++)
            {
                IObjectResolver container = scopes[index].Container;
                if (container != null && container.TryResolve(out CreatureRaceModel model))
                {
                    return model;
                }
            }
            return null;
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
            Environment.SetEnvironmentVariable(CreatureRaceRequest.ENVIRONMENT_VARIABLE, null);
            EditorApplication.update -= Tick;
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            _job = null;
            if (EditorApplication.isPlaying)
            {
                EditorApplication.isPlaying = false;
            }
        }

        private static string TestList(CreatureRaceConfig config)
        {
            if (config == null || config.SelfTests.Count == 0)
            {
                return "(none)";
            }
            var text = new StringBuilder();
            for (int index = 0; index < config.SelfTests.Count; index++)
            {
                if (index > 0)
                {
                    text.Append(", ");
                }
                text.Append('"').Append(config.SelfTests[index].Alias).Append('"');
            }
            return text.ToString();
        }
    }
}
#endif
