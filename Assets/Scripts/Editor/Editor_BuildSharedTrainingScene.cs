#if UNITY_EDITOR
using PoRacer.Systems;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoRacer.Editor
{
    /// <summary>
    /// Builds SCN_TRAIN_ALL: one shared training scene where every creature learns
    /// on two ground variants — flat with blocking walls (TrackKind.Walls) and the
    /// race map's lumpy terrain with chunky objects (TrackKind.Lumpy). Two areas per
    /// creature, laid
    /// out on a grid far enough apart that physics never interacts across areas.
    /// Also provides the headless env build for it.
    /// </summary>
    public static class Editor_BuildSharedTrainingScene
    {
        private const float AREA_SPACING_X = 40f;
        private const float AREA_SPACING_Z = 60f;

        private static readonly string[] CreaturePrefabs =
        {
            "Assets/Prefabs/Worm_v01.prefab",
            "Assets/Prefabs/Spider_v01.prefab",
            "Assets/Prefabs/Hexapod_v01.prefab",
            "Assets/Prefabs/Quad_v01.prefab",
            "Assets/Prefabs/Centipede_v01.prefab",
            "Assets/Prefabs/Crab_v01.prefab",
            "Assets/Prefabs/Kangaroo_v01.prefab",
            "Assets/Prefabs/Blob_v01.prefab"
        };

        /// <summary>
        /// The .glb bipeds train in their own scene and their own run. They share
        /// nothing with the coded-gait fleet — different behaviors, different
        /// areas — and mixing them only splits one time box across a much harder
        /// problem, starving both halves.
        /// </summary>
        private static readonly string[] HumanoidPrefabs =
        {
            "Assets/Prefabs/Grandma_v01.prefab",
            "Assets/Prefabs/Grandpa_v01.prefab",
            "Assets/Prefabs/Matt_v01.prefab",
            "Assets/Prefabs/Nick_v01.prefab"
        };

        private static readonly TrackKind[] Variants = { TrackKind.Walls, TrackKind.Lumpy };

        public static void BuildScene()
        {
            BuildSceneFrom(CreaturePrefabs, "Assets/Scenes/SCN_TRAIN_ALL.unity");
        }

        public static void BuildHumanoidScene()
        {
            BuildSceneFrom(HumanoidPrefabs, "Assets/Scenes/SCN_TRAIN_HUMANOIDS.unity");
        }

        private static void BuildSceneFrom(string[] prefabPaths, string scenePath)
        {
            (Material ground, Material obstacle, PhysicsMaterial physics) = ReadWormAreaMaterials();

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            int built = 0;
            for (int creatureIndex = 0; creatureIndex < prefabPaths.Length; creatureIndex++)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPaths[creatureIndex]);
                if (prefab == null)
                {
                    Debug.LogWarning($"Missing prefab {prefabPaths[creatureIndex]} — run 'Build Creature Prefabs' first. Skipping.");
                    continue;
                }
                for (int variantIndex = 0; variantIndex < Variants.Length; variantIndex++)
                {
                    BuildArea(prefab, Variants[variantIndex],
                        new Vector3(creatureIndex * AREA_SPACING_X, 0f, variantIndex * AREA_SPACING_Z),
                        ground, obstacle, physics);
                    built++;
                }
            }
            EditorSceneManager.SaveScene(scene, scenePath);
            Debug.Log($"{System.IO.Path.GetFileNameWithoutExtension(scenePath)} saved with {built} training areas.");
        }

        public static void BuildEnv()
        {
            BuildEnvFrom("Assets/Scenes/SCN_TRAIN_ALL.unity", "Builds/AllEnv/AllEnv.exe");
        }

        /// <summary>
        /// Builds a scene holding only the named creatures, for a run that concentrates a
        /// time box on a few of them instead of spreading it across the whole fleet.
        ///
        /// The point is throughput, not tidiness. Every behaviour in the shared scene trains
        /// simultaneously and max_steps is PER BEHAVIOUR, so halving the roster roughly halves
        /// the areas each env instance has to simulate and lets more instances run at once -
        /// the surviving behaviours get several times the steps in the same wall clock. With no
        /// checkpoints to warm-start from, that concentration is the only way a fresh run beats
        /// brains that already had eight hours.
        ///
        /// The config must declare exactly what the scene contains: mlagents-learn aborts on
        /// the first behaviour the env reports that the config does not name.
        ///
        /// <paramref name="names"/> is comma-separated and matches the prefab stem, e.g.
        /// "Worm,Spider,Centipede,Crab".
        /// </summary>
        public static string BuildFocusedScene(string names)
        {
            string[] wanted = (names ?? string.Empty).Split(',');
            var paths = new System.Collections.Generic.List<string>();
            var missing = new System.Collections.Generic.List<string>();
            foreach (string raw in wanted)
            {
                string name = raw.Trim();
                if (name.Length == 0) continue;
                string path = "Assets/Prefabs/" + name + "_v01.prefab";
                if (AssetDatabase.LoadAssetAtPath<GameObject>(path) == null)
                {
                    missing.Add(path);
                    continue;
                }
                paths.Add(path);
            }
            if (missing.Count > 0)
            {
                return "ABORT: prefab(s) not found: " + string.Join(", ", missing);
            }
            if (paths.Count == 0)
            {
                return "ABORT: no creature names given";
            }

            BuildSceneFrom(paths.ToArray(), FocusedScenePath);
            return "focused scene built with " + paths.Count + " creatures: " + string.Join(", ", paths);
        }

        public static void BuildFocusedEnv()
        {
            BuildEnvFrom(FocusedScenePath, "Builds/FocusedEnv/FocusedEnv.exe");
        }

        private const string FocusedScenePath = "Assets/Scenes/SCN_TRAIN_FOCUSED.unity";

        public static void BuildHumanoidEnv()
        {
            BuildEnvFrom("Assets/Scenes/SCN_TRAIN_HUMANOIDS.unity", "Builds/HumanoidEnv/HumanoidEnv.exe");
        }

        // Folded in from the old Editor_BuildWormEnv, whose other two menu items
        // pointed at SCN_TRAIN_WORM and SCN_TRAIN_SPIDER - scenes that no longer
        // exist. Every env build now lives here and names a scene that does.
        public static void BuildWormRoughEnv()
        {
            BuildEnvFrom("Assets/Scenes/SCN_TRAIN_WORM_ROUGH.unity", "Builds/WormRoughEnv/WormRoughEnv.exe");
        }

        private static void BuildEnvFrom(string scenePath, string outputPath)
        {
            var options = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = outputPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"Env build {outputPath}: {report.summary.result}, {report.summary.totalErrors} errors.");
        }

        private static void BuildArea(GameObject prefab, TrackKind kind, Vector3 origin,
            Material ground, Material obstacle, PhysicsMaterial physics)
        {
            var area = new GameObject($"Area_{prefab.name}_{kind}");
            area.transform.position = origin;

            var trackRoot = new GameObject("TrackRoot");
            trackRoot.transform.SetParent(area.transform, false);

            float spawnY = Mathf.Max(prefab.transform.position.y, 0.2f) + 0.05f;
            var spawnPoint = new GameObject("SpawnPoint");
            spawnPoint.transform.SetParent(area.transform, false);
            spawnPoint.transform.localPosition = new Vector3(0f, spawnY, 0f);

            var goal = new GameObject("Goal");
            goal.transform.SetParent(area.transform, false);
            goal.transform.localPosition = new Vector3(0f, 0.5f, 6f);

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, area.transform);
            instance.transform.position = origin + new Vector3(0f, spawnY, 0f);

            var trainingArea = area.AddComponent<Systems_TrainingArea>();
            var so = new SerializedObject(trainingArea);
            so.FindProperty("_agent").objectReferenceValue = instance.GetComponentInChildren<Unity.MLAgents.Agent>(true);
            so.FindProperty("_goal").objectReferenceValue = goal.transform;
            so.FindProperty("_spawnPoint").objectReferenceValue = spawnPoint.transform;
            so.FindProperty("_trackKind").enumValueIndex = (int)kind;
            so.FindProperty("_trackRoot").objectReferenceValue = trackRoot.transform;
            so.FindProperty("_groundMaterial").objectReferenceValue = ground;
            so.FindProperty("_obstacleMaterial").objectReferenceValue = obstacle;
            so.FindProperty("_physicsMaterial").objectReferenceValue = physics;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        private static (Material, Material, PhysicsMaterial) ReadWormAreaMaterials()
        {
            // Same assets the retired per-creature training scenes wired up;
            // loaded directly so this builder depends on no other scene.
            Material ground = AssetDatabase.LoadAssetAtPath<Material>("Assets/Art/Materials/M_Ground.mat");
            Material obstacle = AssetDatabase.LoadAssetAtPath<Material>("Assets/Art/Materials/M_Worm.mat");
            return (ground, obstacle, null);
        }
    }
}
#endif
