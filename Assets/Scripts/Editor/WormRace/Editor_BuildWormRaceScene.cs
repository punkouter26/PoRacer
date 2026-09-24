#if UNITY_EDITOR
using System;
using System.IO;
using System.Text;
using PoRacer.WormRace;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;
using Object = UnityEngine.Object;

namespace PoRacer.WormRace.EditorTools
{
    /// <summary>
    /// Builds Assets/Scenes/SCN_WORM_RACE.unity and the assets it needs, from scratch,
    /// re-runnably (AGENTS rule G: the track, lines, camera, light, HUD and LifetimeScope
    /// are authored scene objects; only the worms and the MuJoCo world are created at race
    /// start, see WormSpawnSystem).
    ///
    /// Creates or updates, keeping GUIDs:
    ///   Assets/WormRace/WormRaceSettings.asset        wired to the rig, brains, materials
    ///   Assets/WormRace/PM_WormRace.physicMaterial    0.9 / 0.9, no bounce (worm_rig.json)
    ///   Assets/WormRace/Materials/*.mat               BLUE MuJoCo, ORANGE Isaac, track
    ///   Assets/WormRace/UI/WormRacePanelSettings.asset
    /// Red and green are never used (AGENTS rule D): they belong to heuristic bots and the
    /// baseline RL racer.
    ///
    /// Invoke: unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build();"
    /// </summary>
    public static class Editor_BuildWormRaceScene
    {
        private const string URP_LIT = "Universal Render Pipeline/Lit";
        private const string SOURCE_PANEL_SETTINGS = "Assets/UI/RaceHudPanelSettings.asset";
        private const string DEFAULT_THEME = "Assets/UI/DefaultRuntimeTheme.tss";
        private const string PANEL_SETTINGS = WormRacePaths.UI + "/WormRacePanelSettings.asset";
        // The legacy extension on purpose: a "*.physicsMaterial" file gets DefaultImporter
        // on this Unity version and never loads (Assets/unity_export/IsaacBox/Runtime/IsaacBoxPaths.cs).
        private const string PHYSICS_MATERIAL = WormRacePaths.ROOT + "/PM_WormRace.physicMaterial";
        private const string MATERIAL_MUJOCO = WormRacePaths.MATERIALS + "/M_WormMuJoCo_Blue.mat";
        private const string MATERIAL_ISAAC = WormRacePaths.MATERIALS + "/M_WormIsaac_Orange.mat";
        private const string MATERIAL_GROUND = WormRacePaths.MATERIALS + "/M_WormTrack_Ground.mat";
        private const string MATERIAL_WHITE = WormRacePaths.MATERIALS + "/M_WormTrack_LineWhite.mat";
        private const string MATERIAL_BLACK = WormRacePaths.MATERIALS + "/M_WormTrack_LineBlack.mat";

        private const float DEFAULT_FRICTION = 0.9f;
        private const float GROUND_THICKNESS = 0.2f;
        private const float GROUND_SIDE_MARGIN = 1.5f;
        private const float GROUND_END_MARGIN = 3f;
        private const float LINE_WIDTH = 0.05f;
        private const float LINE_HEIGHT = 0.004f;
        private const float CROSS_LINE_HEIGHT = 0.005f;
        private const float START_LINE_DEPTH = 0.06f;
        private const float LANE_LINE_OVERHANG = 2f;
        private const float CHECKER_TILE = 0.25f;
        private const int CHECKER_ROWS = 2;
        private const float CAMERA_NEAR = 0.05f;
        private const float CAMERA_FAR = 200f;
        private const float CAMERA_FIELD_OF_VIEW = 50f;
        private const int REFERENCE_WIDTH = 1280;
        private const int REFERENCE_HEIGHT = 720;

        private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
        private static readonly int SmoothnessId = Shader.PropertyToID("_Smoothness");
        private static readonly Color MujocoBlue = new(0.16f, 0.45f, 0.95f);
        private static readonly Color IsaacOrange = new(1.0f, 0.55f, 0.10f);
        private static readonly Color GroundGrey = new(0.30f, 0.32f, 0.35f);
        private static readonly Color LineWhite = new(0.92f, 0.92f, 0.92f);
        private static readonly Color LineBlack = new(0.07f, 0.07f, 0.08f);
        private static readonly Vector3 CameraOffset = new(5.5f, 3.2f, -3.0f);
        private static readonly Vector3 LightEuler = new(50f, -30f, 0f);

        [MenuItem("PoRacer/Worm Race/Build Scene")]
        public static string Build()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                return "ABORT: stop play mode first.";
            }
            if (EditorSceneManager.GetActiveScene().isDirty)
            {
                return "ABORT: the open scene has unsaved changes - save or discard them first "
                     + "(creating a new scene would raise a modal prompt).";
            }

            var log = new StringBuilder();
            EnsureFolder(WormRacePaths.ROOT);
            EnsureFolder(WormRacePaths.MATERIALS);
            EnsureFolder(WormRacePaths.BRAINS);
            EnsureFolder(WormRacePaths.UI);

            Shader lit = Shader.Find(URP_LIT);
            if (lit == null)
            {
                return $"ABORT: shader '{URP_LIT}' not found; is URP installed and active?";
            }
            Material blue = UpsertMaterial(MATERIAL_MUJOCO, lit, MujocoBlue, 0.45f);
            Material orange = UpsertMaterial(MATERIAL_ISAAC, lit, IsaacOrange, 0.45f);
            Material ground = UpsertMaterial(MATERIAL_GROUND, lit, GroundGrey, 0.15f);
            Material white = UpsertMaterial(MATERIAL_WHITE, lit, LineWhite, 0.2f);
            Material black = UpsertMaterial(MATERIAL_BLACK, lit, LineBlack, 0.2f);
            log.Append("materials: blue (MuJoCo), orange (Isaac), ground, line white/black\n");

            var rigJson = AssetDatabase.LoadAssetAtPath<TextAsset>(WormRacePaths.RIG_JSON);
            float friction = ReadFriction(rigJson);
            PhysicsMaterial physicsMaterial = UpsertPhysicsMaterial(friction);
            log.Append($"physics material: {PHYSICS_MATERIAL} ({friction:0.00} static/dynamic, no bounce)\n");

            PanelSettings panel = UpsertPanelSettings(log);
            WormRaceSettings settings = UpsertSettings(rigJson, blue, orange, physicsMaterial, log);
            AssetDatabase.SaveAssets();

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            BuildScope(settings);
            BuildLight();
            BuildCamera(settings);
            BuildTrack(settings, ground, white, black, physicsMaterial);
            BuildHud(panel);
            log.Append("scene objects: WormRaceLifetimeScope, Directional Light, Main Camera, Track, WormRaceHud\n");

            if (!EditorSceneManager.SaveScene(scene, WormRacePaths.SCENE))
            {
                return log.Append("ABORT: could not save ").Append(WormRacePaths.SCENE).ToString();
            }
            log.Append("saved ").Append(WormRacePaths.SCENE).Append('\n');
            return log.ToString();
        }

        // --------------------------------------------------------------- assets --

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path))
            {
                return;
            }
            string parent = Path.GetDirectoryName(path)?.Replace('\\', '/');
            string leaf = Path.GetFileName(path);
            if (!string.IsNullOrEmpty(parent) && !AssetDatabase.IsValidFolder(parent))
            {
                EnsureFolder(parent);
            }
            AssetDatabase.CreateFolder(parent, leaf);
        }

        private static Material UpsertMaterial(string path, Shader shader, Color color, float smoothness)
        {
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            bool isNew = material == null;
            if (isNew)
            {
                material = new Material(shader);
            }
            else if (material.shader != shader)
            {
                material.shader = shader;
            }
            material.SetColor(BaseColorId, color);
            material.SetFloat(SmoothnessId, smoothness);
            material.enableInstancing = true;
            if (isNew)
            {
                AssetDatabase.CreateAsset(material, path);
            }
            else
            {
                EditorUtility.SetDirty(material);
            }
            return material;
        }

        private static PhysicsMaterial UpsertPhysicsMaterial(float friction)
        {
            var material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(PHYSICS_MATERIAL);
            bool isNew = material == null;
            if (isNew)
            {
                material = new PhysicsMaterial("PM_WormRace");
            }
            // WORM_SPEC detail 14: every worm-floor and worm-worm pair at 0.9. MuJoCo takes
            // the larger of two frictions; Maximum reproduces that in PhysX.
            material.staticFriction = friction;
            material.dynamicFriction = friction;
            material.bounciness = 0f;
            material.frictionCombine = PhysicsMaterialCombine.Maximum;
            material.bounceCombine = PhysicsMaterialCombine.Minimum;
            if (isNew)
            {
                AssetDatabase.CreateAsset(material, PHYSICS_MATERIAL);
            }
            else
            {
                EditorUtility.SetDirty(material);
            }
            return AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(PHYSICS_MATERIAL);
        }

        private static PanelSettings UpsertPanelSettings(StringBuilder log)
        {
            var panel = AssetDatabase.LoadAssetAtPath<PanelSettings>(PANEL_SETTINGS);
            if (panel == null)
            {
                // Copy the race HUD's settings so the project font and theme come along.
                var source = AssetDatabase.LoadAssetAtPath<PanelSettings>(SOURCE_PANEL_SETTINGS);
                panel = source != null ? Object.Instantiate(source) : ScriptableObject.CreateInstance<PanelSettings>();
                AssetDatabase.CreateAsset(panel, PANEL_SETTINGS);
            }
            if (panel.themeStyleSheet == null)
            {
                panel.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(DEFAULT_THEME);
            }
            // Landscape desktop HUD, unlike the portrait race HUD it was copied from.
            panel.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            panel.referenceResolution = new Vector2Int(REFERENCE_WIDTH, REFERENCE_HEIGHT);
            panel.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight;
            panel.match = 0.5f;
            EditorUtility.SetDirty(panel);
            log.Append("panel settings: ").Append(PANEL_SETTINGS)
               .Append(panel.themeStyleSheet == null ? " (NO THEME - HUD text may not render)\n" : "\n");
            return panel;
        }

        private static WormRaceSettings UpsertSettings(TextAsset rigJson, Material blue, Material orange,
                                                       PhysicsMaterial physicsMaterial, StringBuilder log)
        {
            var settings = AssetDatabase.LoadAssetAtPath<WormRaceSettings>(WormRacePaths.SETTINGS);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<WormRaceSettings>();
                AssetDatabase.CreateAsset(settings, WormRacePaths.SETTINGS);
            }
            var mujocoBrain = AssetDatabase.LoadAssetAtPath<ModelAsset>(WormRacePaths.MUJOCO_BRAIN);
            var isaacBrain = AssetDatabase.LoadAssetAtPath<ModelAsset>(WormRacePaths.ISAAC_BRAIN);

            var serialized = new SerializedObject(settings);
            Assign(serialized, "_rigJson", rigJson);
            Assign(serialized, "_mujocoBrain", mujocoBrain);
            Assign(serialized, "_isaacBrain", isaacBrain);
            Assign(serialized, "_mujocoMaterial", blue);
            Assign(serialized, "_isaacMaterial", orange);
            Assign(serialized, "_wormPhysicsMaterial", physicsMaterial);
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(settings);

            log.Append("settings: ").Append(WormRacePaths.SETTINGS).Append('\n');
            log.Append("  rig:          ").Append(rigJson != null ? WormRacePaths.RIG_JSON : "MISSING - copy training/worm/worm_rig.json to " + WormRacePaths.RIG_JSON).Append('\n');
            log.Append("  MuJoCo brain: ").Append(mujocoBrain != null ? WormRacePaths.MUJOCO_BRAIN : "MISSING - copy training/worm/export/worm_mujoco.onnx to " + WormRacePaths.MUJOCO_BRAIN).Append('\n');
            log.Append("  Isaac brain:  ").Append(isaacBrain != null ? WormRacePaths.ISAAC_BRAIN : "MISSING - copy training/worm/export/worm_isaac.onnx to " + WormRacePaths.ISAAC_BRAIN).Append('\n');
            return settings;
        }

        private static void Assign(SerializedObject serialized, string field, Object value)
        {
            SerializedProperty property = serialized.FindProperty(field);
            if (property == null)
            {
                Debug.LogError($"[WormRace] field '{field}' not found on {serialized.targetObject.GetType().Name}");
                return;
            }
            property.objectReferenceValue = value;
        }

        private static float ReadFriction(TextAsset rigJson)
        {
            if (rigJson == null)
            {
                return DEFAULT_FRICTION;
            }
            try
            {
                FrictionOnly parsed = JsonUtility.FromJson<FrictionOnly>(rigJson.text);
                return parsed != null && parsed.friction > 0f ? parsed.friction : DEFAULT_FRICTION;
            }
            catch (ArgumentException)
            {
                return DEFAULT_FRICTION;
            }
        }

        // ---------------------------------------------------------------- scene --

        private static void BuildScope(WormRaceSettings settings)
        {
            var scopeObject = new GameObject("WormRaceLifetimeScope");
            var scope = scopeObject.AddComponent<WormRaceLifetimeScope>();
            var serialized = new SerializedObject(scope);
            Assign(serialized, "_settings", settings);
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void BuildLight()
        {
            var lightObject = new GameObject("Directional Light");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.shadows = LightShadows.Soft;
            light.intensity = 1.1f;
            lightObject.transform.rotation = Quaternion.Euler(LightEuler);
        }

        private static void BuildCamera(WormRaceSettings settings)
        {
            var cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            var camera = cameraObject.AddComponent<Camera>();
            camera.nearClipPlane = CAMERA_NEAR;
            camera.farClipPlane = CAMERA_FAR;
            camera.fieldOfView = CAMERA_FIELD_OF_VIEW;
            cameraObject.AddComponent<AudioListener>();
            cameraObject.AddComponent<WormRaceCameraView>();

            var focus = new Vector3(0f, 0f, settings.StartLineZ);
            cameraObject.transform.position = focus + CameraOffset;
            cameraObject.transform.rotation = Quaternion.LookRotation(
                focus + Vector3.forward * 1.5f - cameraObject.transform.position, Vector3.up);
        }

        private static void BuildTrack(WormRaceSettings settings, Material ground, Material white, Material black,
                                       PhysicsMaterial physicsMaterial)
        {
            var track = new GameObject("Track");
            float spacing = settings.LaneSpacing;
            float start = settings.StartLineZ;
            float finish = start + settings.TrackLength;
            float laneCount = WormRaceModel.RACER_COUNT;
            float halfWidth = laneCount * spacing * 0.5f;

            // Ground: the only collider on the track. Its top face is y = 0, where the
            // MuJoCo world puts its own plane.
            float groundStart = start - GROUND_END_MARGIN;
            float groundEnd = finish + GROUND_END_MARGIN;
            GameObject groundObject = Block("Ground", track.transform, ground,
                new Vector3(0f, -GROUND_THICKNESS * 0.5f, (groundStart + groundEnd) * 0.5f),
                new Vector3(2f * (halfWidth + GROUND_SIDE_MARGIN), GROUND_THICKNESS, groundEnd - groundStart),
                keepCollider: true);
            groundObject.GetComponent<BoxCollider>().sharedMaterial = physicsMaterial;

            // Lane lines: both outer edges and every divider.
            float lineStart = start - LANE_LINE_OVERHANG;
            float lineEnd = finish + LANE_LINE_OVERHANG;
            for (int edge = 0; edge <= WormRaceModel.RACER_COUNT; edge++)
            {
                float x = -halfWidth + edge * spacing;
                Block($"LaneLine_{edge}", track.transform, white,
                      new Vector3(x, LINE_HEIGHT * 0.5f, (lineStart + lineEnd) * 0.5f),
                      new Vector3(LINE_WIDTH, LINE_HEIGHT, lineEnd - lineStart),
                      keepCollider: false);
            }

            Block("StartLine", track.transform, white,
                  new Vector3(0f, CROSS_LINE_HEIGHT * 0.5f, start),
                  new Vector3(2f * halfWidth + LINE_WIDTH, CROSS_LINE_HEIGHT, START_LINE_DEPTH),
                  keepCollider: false);

            // Chequered finish, leading edge exactly on the finish line (the nose that
            // reaches it first wins).
            var finishLine = new GameObject("FinishLine");
            finishLine.transform.SetParent(track.transform, false);
            int columns = Mathf.Max(2, Mathf.RoundToInt(2f * halfWidth / CHECKER_TILE));
            float tileWidth = 2f * halfWidth / columns;
            for (int row = 0; row < CHECKER_ROWS; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    Material tileMaterial = (row + column) % 2 == 0 ? white : black;
                    Block($"Tile_{row}_{column}", finishLine.transform, tileMaterial,
                          new Vector3(-halfWidth + (column + 0.5f) * tileWidth, CROSS_LINE_HEIGHT * 0.5f,
                                      finish + (row + 0.5f) * CHECKER_TILE),
                          new Vector3(tileWidth, CROSS_LINE_HEIGHT, CHECKER_TILE),
                          keepCollider: false);
                }
            }

            MarkStatic(track);
        }

        private static GameObject Block(string name, Transform parent, Material material, Vector3 position,
                                        Vector3 size, bool keepCollider)
        {
            GameObject block = GameObject.CreatePrimitive(PrimitiveType.Cube);
            block.name = name;
            block.transform.SetParent(parent, false);
            block.transform.localPosition = position;
            block.transform.localScale = size;
            block.GetComponent<MeshRenderer>().sharedMaterial = material;
            if (!keepCollider)
            {
                Object.DestroyImmediate(block.GetComponent<BoxCollider>());
            }
            return block;
        }

        private static void MarkStatic(GameObject root)
        {
            Transform[] all = root.GetComponentsInChildren<Transform>(true);
            for (int index = 0; index < all.Length; index++)
            {
                GameObjectUtility.SetStaticEditorFlags(all[index].gameObject, StaticEditorFlags.BatchingStatic);
            }
        }

        private static void BuildHud(PanelSettings panel)
        {
            var hudObject = new GameObject("WormRaceHud");
            var document = hudObject.AddComponent<UIDocument>();
            document.panelSettings = panel;
            hudObject.AddComponent<WormRaceHudView>();
        }

        [Serializable]
        private sealed class FrictionOnly
        {
#pragma warning disable 0649
            public float friction;
#pragma warning restore 0649
        }
    }
}
#endif
