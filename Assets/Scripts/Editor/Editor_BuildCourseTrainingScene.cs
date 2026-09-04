#if UNITY_EDITOR
using PoRacer.Systems;
using PoRacer.Views;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PoRacer.EditorTools
{
    /// <summary>
    /// Builds SCN_TRAIN_ACROBAT: one copy of the Acrobat course per creature, each
    /// with a Systems_CourseTrainingArea driving that creature along the
    /// centreline. The brains shipped today were trained on flat builder ground
    /// and mostly cannot climb this course; this scene is where the Acrobat
    /// brains come from.
    ///
    /// The config must declare exactly the behaviours the scene contains:
    /// mlagents-learn aborts on the first one the env reports that the config
    /// does not name (Config/AcrobatLoco01.yaml names these four).
    ///
    /// Invoke: unity command eval --code "return PoRacer.EditorTools.Editor_BuildCourseTrainingScene.Build();"
    ///         then Editor_BuildAsync-style: Editor_BuildCourseTrainingScene.BuildEnv() (queue it; it is a player build)
    /// </summary>
    public static class Editor_BuildCourseTrainingScene
    {
        private const string SCENE_PATH = "Assets/Scenes/SCN_TRAIN_ACROBAT.unity";
        private const string ENV_PATH = "Builds/AcrobatEnv/AcrobatEnv.exe";
        // The course is 142 m across; areas sit well apart so nothing on one
        // mountain can see or touch the next.
        private const float AREA_SPACING_X = 260f;

        private static readonly string[] CreaturePrefabs =
        {
            "Assets/Prefabs/Centipede_v01.prefab",
            "Assets/Prefabs/Crab_v01.prefab",
            "Assets/Prefabs/Hexapod_v01.prefab",
            "Assets/Prefabs/Quad_v01.prefab",
        };

        public static string Build()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "cannot build while in play mode";
            }
            GameObject courseAsset = Editor_BuildCourseTrack.LoadAsset();
            if (courseAsset == null)
            {
                return "course GLB missing";
            }
            // The race scene's RaceTrackView carries no physics material either, so
            // the course colliders train on the same default friction they race on.
            PhysicsMaterial physics = null;

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            var report = new System.Text.StringBuilder();
            int built = 0;
            for (int creatureIndex = 0; creatureIndex < CreaturePrefabs.Length; creatureIndex++)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(CreaturePrefabs[creatureIndex]);
                if (prefab == null)
                {
                    report.Append("\nmissing prefab ").Append(CreaturePrefabs[creatureIndex]);
                    continue;
                }
                var area = new GameObject($"Area_{prefab.name}_Acrobat");
                area.transform.position = new Vector3(creatureIndex * AREA_SPACING_X, 0f, 0f);

                var courseInstance = (GameObject)PrefabUtility.InstantiatePrefab(courseAsset, area.transform);
                courseInstance.name = "Course";
                courseInstance.transform.localPosition = Vector3.zero;
                courseInstance.transform.localScale = Vector3.one * Editor_BuildCourseTrack.COURSE_SCALE;
                string error = Editor_BuildCourseTrack.Install(courseInstance, physics, out RaceCourseView course, out string summary);
                if (error != null)
                {
                    return error;
                }

                var creature = (GameObject)PrefabUtility.InstantiatePrefab(prefab, area.transform);
                float spawnHeight = Mathf.Max(prefab.transform.position.y, 0.2f);
                creature.transform.position = course.Path.Start + Vector3.up * (spawnHeight + 0.05f);
                creature.transform.rotation = Quaternion.LookRotation(course.Path.HeadingAt(0f), Vector3.up) * prefab.transform.rotation;

                var trainingArea = area.AddComponent<Systems_CourseTrainingArea>();
                var so = new SerializedObject(trainingArea);
                so.FindProperty("_agent").objectReferenceValue = creature.GetComponentInChildren<Unity.MLAgents.Agent>(true);
                so.FindProperty("_course").objectReferenceValue = course;
                so.FindProperty("_spawnHeight").floatValue = spawnHeight;
                so.ApplyModifiedPropertiesWithoutUndo();
                report.Append('\n').Append(area.name).Append(": ").Append(summary);
                built++;
            }
            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            return $"{SCENE_PATH} saved with {built} course areas{report}";
        }

        /// <summary>Player build of the training scene; takes minutes, so queue it rather than eval it directly.</summary>
        public static void BuildEnv()
        {
            var options = new BuildPlayerOptions
            {
                scenes = new[] { SCENE_PATH },
                locationPathName = ENV_PATH,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"Env build {ENV_PATH}: {report.summary.result}, {report.summary.totalErrors} errors.");
        }
    }
}
#endif
