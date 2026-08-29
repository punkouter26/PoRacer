using System.Collections;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
#if ISAAC_SPIDER_INFERENCE
using Unity.InferenceEngine;
#endif
#if URDF_IMPORTER
using Unity.Robotics.UrdfImporter;
#endif

namespace IsaacSpider.Editor
{
    /// <summary>
    /// Prefab builder + scene spawner for the Isaac spider. Never creates a scene, never touches
    /// project settings, layers, build settings or existing assets.
    /// </summary>
    public static class IsaacSpiderSetup
    {
        public const string FOLDER = "Assets/unity_export/IsaacSpider";
        public const string URDF_PATH = FOLDER + "/robot/spider.urdf";
        public const string ONNX_PATH = FOLDER + "/spider.onnx";
        public const string REFERENCE_PATH = FOLDER + "/isaac_reference.json";
        public const string PHYSICS_MATERIAL_PATH = FOLDER + "/PM_IsaacSpider.physicsMaterial";
        public const string PREFAB_PATH = FOLDER + "/IsaacSpider.prefab";
        public const string PREFAB_NAME = "IsaacSpider";
        private const string MENU = "PoRacer/Isaac Spider/";

        // ------------------------------------------------------------------ 1. prefab
        [MenuItem(MENU + "Build Prefab", priority = 0)]
        public static void BuildPrefab()
        {
            GameObject prefab = BuildPrefabAsset();
            if (prefab != null)
            {
                Selection.activeObject = prefab;
                EditorGUIUtility.PingObject(prefab);
            }
        }

        /// <summary>Builds and saves the prefab; returns the saved prefab asset (null on failure).</summary>
        public static GameObject BuildPrefabAsset()
        {
            if (!File.Exists(URDF_PATH))
            {
                Debug.LogError($"[IsaacSpider] URDF not found at {URDF_PATH}");
                return null;
            }
            string urdfXml = File.ReadAllText(URDF_PATH);
            PhysicsMaterial physicsMaterial = GetOrCreatePhysicsMaterial();
            GameObject root = BuildRig(urdfXml, physicsMaterial);
            if (root == null)
            {
                return null;
            }
            IsaacSpiderAgent agent = root.GetComponent<IsaacSpiderAgent>();
            if (agent == null)
            {
                agent = root.AddComponent<IsaacSpiderAgent>();
            }
            var serialized = new SerializedObject(agent);
#if ISAAC_SPIDER_INFERENCE
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(ONNX_PATH);
            if (model == null)
            {
                Debug.LogWarning($"[IsaacSpider] {ONNX_PATH} did not import as a ModelAsset; the prefab is saved without a policy.");
            }
            serialized.FindProperty("_model").objectReferenceValue = model;
#endif
            serialized.FindProperty("_isaacReference").objectReferenceValue = AssetDatabase.LoadAssetAtPath<TextAsset>(REFERENCE_PATH);
            serialized.ApplyModifiedPropertiesWithoutUndo();
            // Capture raw URDF mass/inertia into the serialized arrays, then apply env.yaml values + floors.
            agent.ApplyPhysicsSettings();
            EditorUtility.SetDirty(agent);

            GameObject saved = PrefabUtility.SaveAsPrefabAsset(root, PREFAB_PATH, out bool success);
            Object.DestroyImmediate(root);
            if (!success)
            {
                Debug.LogError($"[IsaacSpider] failed to save {PREFAB_PATH}");
                return null;
            }
            AssetDatabase.SaveAssets();
            Debug.Log($"[IsaacSpider] prefab saved to {PREFAB_PATH} ({saved.GetComponentsInChildren<ArticulationBody>(true).Length} bodies, {saved.GetComponentsInChildren<Collider>(true).Length} colliders, importer={(UsesUrdfImporter ? "URDF Importer" : "built-in URDF parser")})");
            return saved;
        }

        public static bool UsesUrdfImporter
        {
            get
            {
#if URDF_IMPORTER
                return true;
#else
                return false;
#endif
            }
        }

        private static GameObject BuildRig(string urdfXml, PhysicsMaterial physicsMaterial)
        {
#if URDF_IMPORTER
            var settings = new ImportSettings
            {
                choosenAxis = ImportSettings.axisType.yAxis,
                convexMethod = ImportSettings.convexDecomposer.unity,
            };
            IEnumerator<GameObject> routine = UrdfRobotExtensions.Create(URDF_PATH, settings);
            GameObject imported = null;
            while (routine.MoveNext())
            {
                imported = routine.Current;
            }
            if (imported == null)
            {
                Debug.LogError("[IsaacSpider] UrdfRobotExtensions.Create returned nothing.");
                return null;
            }
            imported.name = PREFAB_NAME;
            // The importer's runtime helpers set jointFriction / angularDamping = 10 in Start. Remove them.
            RemoveAll<Controller>(imported);
            RemoveAll<JointControl>(imported);
            RemoveAll<FKRobot>(imported);
            RemoveAll<UrdfRobot>(imported);
            int replaced = IsaacSpiderRigBuilder.ReplaceCollidersOnImportedRig(imported, urdfXml, physicsMaterial);
            Debug.Log($"[IsaacSpider] replaced scaled importer colliders on {replaced} links with unscaled primitives.");
            ArticulationBody rootBody = imported.GetComponentInChildren<ArticulationBody>();
            if (rootBody != null)
            {
                rootBody.immovable = false;
            }
            return imported;
#else
            return IsaacSpiderRigBuilder.Build(urdfXml, PREFAB_NAME, physicsMaterial);
#endif
        }

#if URDF_IMPORTER
        private static void RemoveAll<T>(GameObject root) where T : Component
        {
            T[] found = root.GetComponentsInChildren<T>(true);
            for (int index = 0; index < found.Length; index++)
            {
                Object.DestroyImmediate(found[index]);
            }
        }
#endif

        /// <summary>env.yaml ground material: static 0.5, dynamic 0.5, restitution 0. Minimum combine so the pair friction stays 0.5 against the project's 0.6 default ground.</summary>
        public static PhysicsMaterial GetOrCreatePhysicsMaterial()
        {
            var existing = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(PHYSICS_MATERIAL_PATH);
            if (existing != null)
            {
                return existing;
            }
            var material = new PhysicsMaterial("PM_IsaacSpider")
            {
                staticFriction = 0.5f,
                dynamicFriction = 0.5f,
                bounciness = 0f,
                frictionCombine = PhysicsMaterialCombine.Minimum,
                bounceCombine = PhysicsMaterialCombine.Minimum,
            };
            AssetDatabase.CreateAsset(material, PHYSICS_MATERIAL_PATH);
            return material;
        }

        // ------------------------------------------------------------------ 2. spawn into the open scene
        [MenuItem(MENU + "Spawn Into Open Scene (target = selection)", priority = 10)]
        public static void SpawnIntoOpenScene()
        {
            IsaacSpiderSpawnWindow.Open();
        }

        /// <summary>Instantiates the prefab into the active scene. Ground/light/camera are added only when the scene has none.</summary>
        public static GameObject Spawn(Vector3 position, float spawnHeight, Transform target)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PREFAB_PATH);
            if (prefab == null)
            {
                prefab = BuildPrefabAsset();
                if (prefab == null)
                {
                    return null;
                }
            }
            Scene scene = SceneManager.GetActiveScene();
            EnsureSceneEssentials(scene);
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            instance.transform.position = new Vector3(position.x, position.y + spawnHeight, position.z);
            Undo.RegisterCreatedObjectUndo(instance, "Spawn Isaac Spider");
            if (target != null)
            {
                var serialized = new SerializedObject(instance.GetComponent<IsaacSpiderAgent>());
                serialized.FindProperty("_target").objectReferenceValue = target;
                serialized.ApplyModifiedProperties();
            }
            EditorSceneManager.MarkSceneDirty(scene);
            Selection.activeGameObject = instance;
            Debug.Log($"[IsaacSpider] spawned {instance.name} at {instance.transform.position} in '{scene.name}' target={(target != null ? target.name : "ring sampler")}");
            return instance;
        }

        private static void EnsureSceneEssentials(Scene scene)
        {
            if (Object.FindAnyObjectByType<Collider>() == null)
            {
                var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
                ground.name = "IsaacSpider_Ground";
                ground.transform.localScale = new Vector3(4f, 1f, 4f);
                ground.GetComponent<Collider>().sharedMaterial = GetOrCreatePhysicsMaterial();
                SceneManager.MoveGameObjectToScene(ground, scene);
                Undo.RegisterCreatedObjectUndo(ground, "IsaacSpider ground");
            }
            if (Object.FindAnyObjectByType<Light>() == null)
            {
                var lightGo = new GameObject("IsaacSpider_Light");
                Light light = lightGo.AddComponent<Light>();
                light.type = LightType.Directional;
                lightGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
                SceneManager.MoveGameObjectToScene(lightGo, scene);
                Undo.RegisterCreatedObjectUndo(lightGo, "IsaacSpider light");
            }
            if (Object.FindAnyObjectByType<Camera>() == null)
            {
                var cameraGo = new GameObject("IsaacSpider_Camera");
                cameraGo.AddComponent<Camera>();
                cameraGo.transform.position = new Vector3(0f, 1.5f, -3f);
                cameraGo.transform.LookAt(Vector3.zero);
                SceneManager.MoveGameObjectToScene(cameraGo, scene);
                Undo.RegisterCreatedObjectUndo(cameraGo, "IsaacSpider camera");
            }
        }

        // ------------------------------------------------------------------ 3. edit-mode reference check
        [MenuItem(MENU + "Run Isaac Reference Check", priority = 20)]
        public static void RunReferenceCheck()
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PREFAB_PATH);
            if (prefab == null)
            {
                Debug.LogError("[IsaacSpider] build the prefab first.");
                return;
            }
            GameObject temp = Object.Instantiate(prefab);
            temp.hideFlags = HideFlags.HideAndDontSave;
            try
            {
                IsaacSpiderAgent agent = temp.GetComponent<IsaacSpiderAgent>();
                float worst = agent.RunReferenceCheck(out int steps);
                Debug.Log($"[IsaacSpider] reference check over {steps} steps: max |diff| = {worst:E2}");
                agent.ReleaseWorker();
            }
            finally
            {
                Object.DestroyImmediate(temp);
            }
        }
    }

    /// <summary>Small window: position, spawn height and target (defaults to the current selection).</summary>
    public sealed class IsaacSpiderSpawnWindow : EditorWindow
    {
        private Vector3 _position = Vector3.zero;
        private float _spawnHeight = 0.18f; // env.yaml init_state.pos z
        private Transform _target;

        public static void Open()
        {
            var window = GetWindow<IsaacSpiderSpawnWindow>("Isaac Spider");
            window._target = Selection.activeTransform;
            window.Show();
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("Spawns IsaacSpider.prefab into the currently open scene.", EditorStyles.wordWrappedLabel);
            _position = EditorGUILayout.Vector3Field("Ground position", _position);
            _spawnHeight = EditorGUILayout.FloatField("Spawn height (m)", _spawnHeight);
            _target = (Transform)EditorGUILayout.ObjectField("Target (optional)", _target, typeof(Transform), true);
            if (GUILayout.Button("Use selection as target") && Selection.activeTransform != null)
            {
                _target = Selection.activeTransform;
            }
            EditorGUILayout.Space();
            if (GUILayout.Button("Spawn"))
            {
                IsaacSpiderSetup.Spawn(_position, _spawnHeight, _target);
            }
        }
    }
}
