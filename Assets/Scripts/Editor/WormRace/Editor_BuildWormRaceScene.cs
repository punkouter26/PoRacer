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
    ///   Assets/WormRace/WormRaceSettings.asset        wired to the rig, the racer list, materials
    ///   Assets/WormRace/PM_WormRace.physicMaterial    0.9 / 0.9, no bounce (worm_rig.json)
    ///   Assets/WormRace/Materials/*.mat               one per racer (BLUE, ORANGE, PURPLE), track
    ///   Assets/WormRace/UI/WormRacePanelSettings.asset
    /// Red and green are never used (AGENTS rule D): they belong to heuristic bots and the
    /// baseline RL racer.
    ///
    /// The racer list (one lane each) is seeded from <see cref="Roster"/>: names, methods,
    /// colours, materials and brains are refreshed on every run, but a racer's Physics
    /// choice, once made in the Inspector, is never overwritten. Racers added by hand after
    /// the known ones are kept, and get a lane too.
    ///
    /// ORDER MATTERS: the new scene is created BEFORE any asset is loaded. NewScene in
    /// Single mode unloads every asset nothing in memory references, and the settings asset
    /// and the panel settings (referenced only by this method's locals) were being destroyed
    /// under their C# wrappers; assigning such a fake-null object through SerializedObject
    /// writes nothing, so the scope's _settings and the HUD's PanelSettings were saved as
    /// {fileID: 0}. Build() now also checks the saved file for both links.
    ///
    /// Invoke: unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build();"
    /// </summary>
    public static class Editor_BuildWormRaceScene
    {
        /// <summary>A known racer, in lane order. Physics is only the default for a new entry.</summary>
        private sealed class RacerSpec
        {
            public string name;
            public string method;
            public string brainFile;
            public string materialPath;
            public Color color;
            public int defaultPhysics;
        }

        // WormPhysicsKind values (the enum is internal to the runtime assembly).
        private const int PHYSICS_MUJOCO = 0;
        private const int PHYSICS_PHYSX = 1;
        private const string EXPORT_FOLDER = "training/worm/export/";
        private const string URP_LIT = "Universal Render Pipeline/Lit";
        private const string SOURCE_PANEL_SETTINGS = "Assets/UI/RaceHudPanelSettings.asset";
        private const string DEFAULT_THEME = "Assets/UI/DefaultRuntimeTheme.tss";
        private const string PANEL_SETTINGS = WormRacePaths.UI + "/WormRacePanelSettings.asset";
        // The legacy extension on purpose: a "*.physicsMaterial" file gets DefaultImporter
        // on this Unity version and never loads (Assets/unity_export/IsaacBox/Runtime/IsaacBoxPaths.cs).
        private const string PHYSICS_MATERIAL = WormRacePaths.ROOT + "/PM_WormRace.physicMaterial";
        private const string MATERIAL_MUJOCO = WormRacePaths.MATERIALS + "/M_WormMuJoCo_Blue.mat";
        private const string MATERIAL_ISAAC = WormRacePaths.MATERIALS + "/M_WormIsaac_Orange.mat";
        private const string MATERIAL_ISAACLAB3 = WormRacePaths.MATERIALS + "/M_WormIsaacLab3_Purple.mat";
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
        private static readonly Color IsaacLab3Purple = new(0.60f, 0.30f, 0.85f);
        private static readonly Color GroundGrey = new(0.30f, 0.32f, 0.35f);
        private static readonly Color LineWhite = new(0.92f, 0.92f, 0.92f);
        private static readonly Color LineBlack = new(0.07f, 0.07f, 0.08f);
        private static readonly Vector3 CameraOffset = new(5.5f, 3.2f, -3.0f);
        private static readonly Vector3 LightEuler = new(50f, -30f, 0f);

        /// <summary>
        /// The known racers, lane 0 first. Lane 2's brain comes from Isaac Lab 3 on Newton's
        /// MuJoCo-Warp solver, hence MuJoCo by default; switch it to PhysX in the settings if
        /// that trainer falls back to PhysX.
        /// </summary>
        private static readonly RacerSpec[] Roster =
        {
            new()
            {
                name = "MuJoCo worm", method = "MuJoCo", brainFile = "worm_mujoco.onnx",
                materialPath = MATERIAL_MUJOCO, color = MujocoBlue, defaultPhysics = PHYSICS_MUJOCO,
            },
            new()
            {
                name = "Isaac worm", method = "Isaac Lab", brainFile = "worm_isaac.onnx",
                materialPath = MATERIAL_ISAAC, color = IsaacOrange, defaultPhysics = PHYSICS_PHYSX,
            },
            new()
            {
                name = "Isaac3Worm", method = "Isaac Lab 3", brainFile = "worm_isaaclab3.onnx",
                materialPath = MATERIAL_ISAACLAB3, color = IsaacLab3Purple, defaultPhysics = PHYSICS_MUJOCO,
            },
        };

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
            Shader lit = Shader.Find(URP_LIT);
            if (lit == null)
            {
                return $"ABORT: shader '{URP_LIT}' not found; is URP installed and active?";
            }

            // FIRST, before any asset is loaded or created: NewScene (Single) unloads every
            // asset nothing references, which destroyed the settings and panel assets held
            // only in locals here and left the scene's links at {fileID: 0}. See the class
            // summary.
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var log = new StringBuilder();
            EnsureFolder(WormRacePaths.ROOT);
            EnsureFolder(WormRacePaths.MATERIALS);
            EnsureFolder(WormRacePaths.BRAINS);
            EnsureFolder(WormRacePaths.UI);

            var racerMaterials = new Material[Roster.Length];
            for (int index = 0; index < Roster.Length; index++)
            {
                racerMaterials[index] = UpsertMaterial(Roster[index].materialPath, lit, Roster[index].color, 0.45f);
            }
            Material ground = UpsertMaterial(MATERIAL_GROUND, lit, GroundGrey, 0.15f);
            Material white = UpsertMaterial(MATERIAL_WHITE, lit, LineWhite, 0.2f);
            Material black = UpsertMaterial(MATERIAL_BLACK, lit, LineBlack, 0.2f);
            log.Append("materials: blue (MuJoCo), orange (Isaac), purple (Isaac Lab 3), ground, line white/black\n");

            var rigJson = AssetDatabase.LoadAssetAtPath<TextAsset>(WormRacePaths.RIG_JSON);
            float friction = ReadFriction(rigJson);
            PhysicsMaterial physicsMaterial = UpsertPhysicsMaterial(friction);
            log.Append($"physics material: {PHYSICS_MATERIAL} ({friction:0.00} static/dynamic, no bounce)\n");

            PanelSettings panel = UpsertPanelSettings(log);
            WormRaceSettings settings = UpsertSettings(rigJson, racerMaterials, physicsMaterial, log);
            AssetDatabase.SaveAssets();

            WormRaceLifetimeScope scope = BuildScope(settings);
            BuildLight();
            BuildCamera(settings);
            BuildTrack(settings, ground, white, black, physicsMaterial);
            UIDocument hud = BuildHud(panel);
            log.Append("scene objects: WormRaceLifetimeScope, Directional Light, Main Camera, Track (")
               .Append(settings.LaneCount).Append(" lanes), WormRaceHud\n");

            string unlinked = UnlinkedReferences(scope, settings, hud, panel);
            if (unlinked.Length > 0)
            {
                return log.Append("ABORT before saving: ").Append(unlinked).ToString();
            }
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene, WormRacePaths.SCENE))
            {
                return log.Append("ABORT: could not save ").Append(WormRacePaths.SCENE).ToString();
            }
            string missingOnDisk = LinksMissingOnDisk(settings, panel);
            if (missingOnDisk.Length > 0)
            {
                return log.Append("ERROR: saved, but ").Append(missingOnDisk).ToString();
            }
            log.Append("saved ").Append(WormRacePaths.SCENE)
               .Append(" (checked on disk: scope -> settings, HUD -> panel settings)\n");
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

        private static WormRaceSettings UpsertSettings(TextAsset rigJson, Material[] racerMaterials,
                                                       PhysicsMaterial physicsMaterial, StringBuilder log)
        {
            var settings = AssetDatabase.LoadAssetAtPath<WormRaceSettings>(WormRacePaths.SETTINGS);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<WormRaceSettings>();
                AssetDatabase.CreateAsset(settings, WormRacePaths.SETTINGS);
            }

            var serialized = new SerializedObject(settings);
            Assign(serialized, "_rigJson", rigJson);
            Assign(serialized, "_wormPhysicsMaterial", physicsMaterial);
            log.Append("settings: ").Append(WormRacePaths.SETTINGS).Append('\n');
            log.Append("  rig: ").Append(rigJson != null ? WormRacePaths.RIG_JSON : "MISSING - copy training/worm/worm_rig.json to " + WormRacePaths.RIG_JSON).Append('\n');
            UpsertRacers(serialized, racerMaterials, log);
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(settings);
            return settings;
        }

        /// <summary>
        /// Puts the known racers in lanes 0..n-1 of _racers, in <see cref="Roster"/> order.
        /// An existing entry (matched by name) keeps its Physics choice; a new one gets the
        /// roster's default. Entries the roster does not know stay, after the known ones.
        /// </summary>
        private static void UpsertRacers(SerializedObject serialized, Material[] racerMaterials, StringBuilder log)
        {
            SerializedProperty racers = serialized.FindProperty("_racers");
            if (racers == null || !racers.isArray)
            {
                Debug.LogError("[WormRace] WormRaceSettings has no _racers list");
                return;
            }
            for (int lane = 0; lane < Roster.Length; lane++)
            {
                RacerSpec spec = Roster[lane];
                int existing = FindRacer(racers, spec.name, lane);
                bool isNew = existing < 0;
                if (isNew)
                {
                    racers.InsertArrayElementAtIndex(Mathf.Min(lane, racers.arraySize));
                    existing = Mathf.Min(lane, racers.arraySize - 1);
                }
                if (existing != lane)
                {
                    racers.MoveArrayElement(existing, lane);
                }

                string brainPath = WormRacePaths.BRAINS + "/" + spec.brainFile;
                var brain = AssetDatabase.LoadAssetAtPath<ModelAsset>(brainPath);
                SerializedProperty entry = racers.GetArrayElementAtIndex(lane);
                entry.FindPropertyRelative("_name").stringValue = spec.name;
                entry.FindPropertyRelative("_method").stringValue = spec.method;
                entry.FindPropertyRelative("_brainFile").stringValue = spec.brainFile;
                entry.FindPropertyRelative("_brain").objectReferenceValue = brain;
                entry.FindPropertyRelative("_material").objectReferenceValue = racerMaterials[lane];
                entry.FindPropertyRelative("_color").colorValue = spec.color;
                SerializedProperty physics = entry.FindPropertyRelative("_physics");
                if (isNew)
                {
                    physics.enumValueIndex = spec.defaultPhysics;
                }

                log.Append("  lane ").Append(lane).Append(": ").Append(spec.name).Append(" (")
                   .Append(spec.method).Append(", ")
                   .Append(physics.enumValueIndex == PHYSICS_PHYSX ? "PhysX ArticulationBody" : "MuJoCo plug-in")
                   .Append(isNew ? ", new" : string.Empty).Append(") brain: ")
                   .Append(brain != null
                       ? brainPath
                       : "MISSING - copy " + EXPORT_FOLDER + spec.brainFile + " to " + brainPath
                         + " (the worm lies still, HUD says NO BRAIN)")
                   .Append('\n');
            }
            for (int lane = Roster.Length; lane < racers.arraySize; lane++)
            {
                log.Append("  lane ").Append(lane).Append(": ")
                   .Append(racers.GetArrayElementAtIndex(lane).FindPropertyRelative("_name").stringValue)
                   .Append(" (added by hand, kept as is)\n");
            }
        }

        /// <summary>Index of the entry called <paramref name="name"/> at or after <paramref name="from"/>, or -1.</summary>
        private static int FindRacer(SerializedProperty racers, string name, int from)
        {
            for (int index = from; index < racers.arraySize; index++)
            {
                if (racers.GetArrayElementAtIndex(index).FindPropertyRelative("_name").stringValue == name)
                {
                    return index;
                }
            }
            return -1;
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

        private static WormRaceLifetimeScope BuildScope(WormRaceSettings settings)
        {
            var scopeObject = new GameObject("WormRaceLifetimeScope");
            var scope = scopeObject.AddComponent<WormRaceLifetimeScope>();
            var serialized = new SerializedObject(scope);
            Assign(serialized, "_settings", settings);
            serialized.ApplyModifiedPropertiesWithoutUndo();
            return scope;
        }

        /// <summary>The two links the old builder lost, re-read from the live objects.</summary>
        private static string UnlinkedReferences(WormRaceLifetimeScope scope, WormRaceSettings settings,
                                                 UIDocument hud, PanelSettings panel)
        {
            var problems = new StringBuilder();
            if (settings == null)
            {
                problems.Append("the settings asset is not loaded; ");
            }
            else if (new SerializedObject(scope).FindProperty("_settings").objectReferenceValue != settings)
            {
                problems.Append("WormRaceLifetimeScope._settings did not take the settings asset; ");
            }
            if (panel == null)
            {
                problems.Append("the panel settings asset is not loaded; ");
            }
            else if (hud.panelSettings != panel)
            {
                problems.Append("the HUD's UIDocument did not take the panel settings; ");
            }
            return problems.ToString();
        }

        /// <summary>Reads the saved .unity file back and looks for both links by GUID.</summary>
        private static string LinksMissingOnDisk(WormRaceSettings settings, PanelSettings panel)
        {
            string text = File.ReadAllText(WormRacePaths.SCENE);
            var problems = new StringBuilder();
            string settingsGuid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(settings));
            if (!text.Contains("_settings: {fileID: 11400000, guid: " + settingsGuid))
            {
                problems.Append("the saved scene has no _settings link to ").Append(WormRacePaths.SETTINGS).Append("; ");
            }
            string panelGuid = AssetDatabase.AssetPathToGUID(AssetDatabase.GetAssetPath(panel));
            if (!text.Contains("m_PanelSettings: {fileID: 11400000, guid: " + panelGuid))
            {
                problems.Append("the saved scene has no PanelSettings link to ").Append(PANEL_SETTINGS).Append("; ");
            }
            return problems.ToString();
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
            int laneCount = Mathf.Max(1, settings.LaneCount);
            // Lanes are centred on x = 0 (WormRaceSettings.LaneX), one lane width each.
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
            for (int edge = 0; edge <= laneCount; edge++)
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

        private static UIDocument BuildHud(PanelSettings panel)
        {
            var hudObject = new GameObject("WormRaceHud");
            var document = hudObject.AddComponent<UIDocument>();
            document.panelSettings = panel;
            hudObject.AddComponent<WormRaceHudView>();
            return document;
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
