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
            "Assets/Prefabs/Snake_v01.prefab",
            "Assets/Prefabs/Centipede_v01.prefab",
            "Assets/Prefabs/Crab_v01.prefab",
            "Assets/Prefabs/Kangaroo_v01.prefab",
            "Assets/Prefabs/Blob_v01.prefab"
        };

        private static readonly TrackKind[] Variants = { TrackKind.Walls, TrackKind.Lumpy };

        [MenuItem("PoRacer/Build Shared Training Scene (SCN_TRAIN_ALL)")]
        public static void BuildScene()
        {
            (Material ground, Material obstacle, PhysicsMaterial physics) = ReadWormAreaMaterials();

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            int built = 0;
            for (int creatureIndex = 0; creatureIndex < CreaturePrefabs.Length; creatureIndex++)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(CreaturePrefabs[creatureIndex]);
                if (prefab == null)
                {
                    Debug.LogWarning($"Missing prefab {CreaturePrefabs[creatureIndex]} — run 'Build Creature Prefabs' first. Skipping.");
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
            EditorSceneManager.SaveScene(scene, "Assets/Scenes/SCN_TRAIN_ALL.unity");
            Debug.Log($"SCN_TRAIN_ALL saved with {built} training areas.");
        }

        [MenuItem("PoRacer/Build All-Creatures Training Env")]
        public static void BuildEnv()
        {
            var options = new BuildPlayerOptions
            {
                scenes = new[] { "Assets/Scenes/SCN_TRAIN_ALL.unity" },
                locationPathName = "Builds/AllEnv/AllEnv.exe",
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"All-creatures env build: {report.summary.result}, {report.summary.totalErrors} errors.");
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
            Scene wormScene = EditorSceneManager.OpenScene("Assets/Scenes/SCN_TRAIN_WORM.unity", OpenSceneMode.Additive);
            Material ground = null;
            Material obstacle = null;
            PhysicsMaterial physics = null;
            foreach (GameObject rootGo in wormScene.GetRootGameObjects())
            {
                var area = rootGo.GetComponentInChildren<Systems_TrainingArea>(true);
                if (area != null)
                {
                    var so = new SerializedObject(area);
                    ground = so.FindProperty("_groundMaterial").objectReferenceValue as Material;
                    obstacle = so.FindProperty("_obstacleMaterial").objectReferenceValue as Material;
                    physics = so.FindProperty("_physicsMaterial").objectReferenceValue as PhysicsMaterial;
                    break;
                }
            }
            EditorSceneManager.CloseScene(wormScene, true);
            return (ground, obstacle, physics);
        }
    }
}
#endif
