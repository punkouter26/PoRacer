using System.IO;
using Creature.MojucuBoy;
using Mujoco;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace CreatureEditor
{
    /// <summary>
    /// Builds MojucuBoy's race-ready scene: the MJCF physics rig, the authored GLB
    /// character bound to it, the policy controller, and a camera.
    ///
    /// Re-runnable. Every step is derived from files on disk -- mojucuboy_unity.xml,
    /// mojucuboy_rig.json, Boy_Character_mujoco.glb -- so re-running after a re-export
    /// reproduces a matching scene rather than patching a stale one.
    /// </summary>
    public static class MojucuBoySetup
    {
        private const string SCENE_PATH = "Assets/Agents/MojucuBoy_v01/SCN_MOJUCUBOY_RACE.unity";
        private const string MJCF_PATH = "Assets/Agents/MojucuBoy_v01/mojucuboy_unity.xml";
        private const string RIG_JSON = "Assets/Agents/MojucuBoy_v01/mojucuboy_rig.json";
        private const string GLB_PATH = "Assets/Boy_Character_mujoco.glb";
        private const string ONNX_PATH = "Assets/Agents/MojucuBoy_v01/mojucuboy_policy.onnx";
        private const string IMPORT_ASSETS = "Assets/Local/MjImports/mojucuboy";

        [MenuItem("PoRacer/Creatures/Build Boy Race Scene")]
        public static string Build()
        {
            var log = new System.Text.StringBuilder();

            // MjImporterWithAssets throws if its generated materials already exist,
            // then ImportString swallows the exception and returns null. Clear first
            // so this stays re-runnable.
            if (AssetDatabase.IsValidFolder(IMPORT_ASSETS))
            {
                AssetDatabase.DeleteAsset(IMPORT_ASSETS);
                AssetDatabase.Refresh();
            }

            Scene scene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene, NewSceneMode.Single);

            MjEngineTool.LoadPlugins();
            var importer = new MjImporterWithAssets();
            string mjcf = File.ReadAllText(MJCF_PATH);
            GameObject physicsRoot = importer.ImportString(mjcf, "mojucuboy", MJCF_PATH);
            if (physicsRoot == null)
            {
                return "ABORT: MJCF import returned null; check the console for the "
                     + "exception ImportString swallowed.";
            }
            log.Append($"imported physics rig: {physicsRoot.name}\n");

            var settings = Object.FindAnyObjectByType<MjGlobalSettings>();
            if (settings != null)
            {
                settings.UseRawGameObjectNames = true;
            }

            // URP does not support the built-in Standard shader the importer assigns;
            // it renders magenta. The collision primitives are debug geometry anyway,
            // so they are hidden here and the authored character is what is seen.
            int hidden = 0;
            foreach (MjGeom geom in Object.FindObjectsByType<MjGeom>(FindObjectsInactive.Include))
            {
                var renderer = geom.GetComponent<MeshRenderer>();
                if (renderer != null && geom.name != "floor")
                {
                    renderer.enabled = false;
                    hidden++;
                }
            }
            log.Append($"hid {hidden} collision-geom renderers\n");

            // The authored character. The MJCF is built with a 180 degree facing yaw
            // (build_mjcf.py) so the rig faces MuJoCo +Y == Unity +Z, the race
            // direction; the GLB is authored unyawed, so it needs the same rotation
            // before the bind-pose offsets are captured.
            var glb = AssetDatabase.LoadAssetAtPath<GameObject>(GLB_PATH);
            if (glb == null)
            {
                return $"ABORT: {GLB_PATH} not found.";
            }
            var character = (GameObject)PrefabUtility.InstantiatePrefab(glb);
            character.name = "BoyCharacter";
            character.transform.SetParent(physicsRoot.transform, false);
            character.transform.SetLocalPositionAndRotation(
                Vector3.zero, Quaternion.Euler(0f, 180f, 0f));
            log.Append("instantiated character, yawed 180 to match the MJCF facing\n");

            // The scene needs an explicit MjScene: nothing steps MuJoCo without it,
            // and relying on MjScene.Instance to create one lazily is how the
            // "singleton, yet multiple instances found" error happens -- the getter
            // constructs one, so whoever asks second throws.
            if (!MjScene.InstanceExists)
            {
                var sceneObject = new GameObject("MjScene");
                sceneObject.AddComponent<MjScene>();
                log.Append("added MjScene\n");
            }

            var controller = physicsRoot.AddComponent<MojucuBoyController>();
            var rigJson = AssetDatabase.LoadAssetAtPath<TextAsset>(RIG_JSON);
            Assign(controller, "_rigJson", rigJson);
            log.Append($"rig json: {(rigJson == null ? "MISSING" : RIG_JSON)}\n");

#if CREATURE_HAS_INFERENCE
            var model = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(ONNX_PATH);
            Assign(controller, "_modelAsset", model);
            log.Append($"policy: {(model == null ? "MISSING -- will hold the stance" : ONNX_PATH)}\n");
#else
            log.Append("BUILT WITHOUT CREATURE_HAS_INFERENCE -- the racer cannot think.\n");
#endif

            var binder = physicsRoot.AddComponent<MojucuBoyVisualBinder>();
            Assign(binder, "_skinRoot", character.transform);
            Assign(binder, "_physicsRoot", physicsRoot.transform);

            var cameraObject = new GameObject("Main Camera");
            var camera = cameraObject.AddComponent<Camera>();
            cameraObject.tag = "MainCamera";
            cameraObject.transform.SetPositionAndRotation(
                new Vector3(2.6f, 1.5f, 2.6f), Quaternion.Euler(12f, -135f, 0f));
            camera.nearClipPlane = 0.05f;

            var lightObject = new GameObject("Directional Light");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.shadows = LightShadows.Soft;
            lightObject.transform.rotation = Quaternion.Euler(48f, -30f, 0f);

            EditorSceneManager.SaveScene(scene, SCENE_PATH);
            log.Append($"saved {SCENE_PATH}\n");
            return log.ToString();
        }

        private static void Assign(Object target, string field, Object value)
        {
            var serialised = new SerializedObject(target);
            SerializedProperty property = serialised.FindProperty(field);
            if (property == null)
            {
                Debug.LogError($"field '{field}' not found on {target.GetType().Name}");
                return;
            }
            property.objectReferenceValue = value;
            serialised.ApplyModifiedPropertiesWithoutUndo();
        }
    }
}
