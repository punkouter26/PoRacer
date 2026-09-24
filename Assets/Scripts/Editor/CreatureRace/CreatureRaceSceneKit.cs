#if UNITY_EDITOR
using System.IO;
using System.Text;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using Object = UnityEngine.Object;

namespace PoRacer.CreatureRace.EditorTools
{
    /// <summary>
    /// What every creature race scene builder shares (AGENTS rule G: the track, lines,
    /// camera, light, HUD and lifetime scope are authored scene objects; only the racers and
    /// the MuJoCo world are created at race start). A builder creates its scene FIRST and only
    /// then loads or creates assets: NewScene (Single) unloads every asset nothing references,
    /// and a settings asset held only in a local would be destroyed under its C# wrapper and
    /// saved as {fileID: 0} (the worm builder's old bug, training/worm/RESULTS.md problem 4).
    /// After saving, <see cref="LinksMissingOnDisk"/> reads the .unity file back to prove the
    /// links landed.
    /// </summary>
    public static class CreatureRaceSceneKit
    {
        public const string URP_LIT = "Universal Render Pipeline/Lit";

        private const string SOURCE_PANEL_SETTINGS = "Assets/UI/RaceHudPanelSettings.asset";
        private const string DEFAULT_THEME = "Assets/UI/DefaultRuntimeTheme.tss";
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
        private const float LIGHT_INTENSITY = 1.1f;

        private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
        private static readonly int SmoothnessId = Shader.PropertyToID("_Smoothness");
        private static readonly Vector3 LightEuler = new(50f, -30f, 0f);

        /// <summary>A known racer, in lane order. Physics is only the default for a new entry.</summary>
        public sealed class RacerSpec
        {
            public RacerSpec(string name, string method, string brainFile, string materialPath, Color color,
                             CreaturePhysicsKind defaultPhysics)
            {
                Name = name;
                Method = method;
                BrainFile = brainFile;
                MaterialPath = materialPath;
                Color = color;
                DefaultPhysics = defaultPhysics;
            }

            public string Name { get; }
            public string Method { get; }
            public string BrainFile { get; }
            public string MaterialPath { get; }
            public Color Color { get; }
            public CreaturePhysicsKind DefaultPhysics { get; }
        }

        /// <summary>One self-test as a builder writes it (mirrors CreatureSelfTestDefinition).</summary>
        public sealed class SelfTestSpec
        {
            public string Name { get; set; } = string.Empty;
            public string Alias { get; set; } = string.Empty;
            public CreatureSelfTestKind Kind { get; set; }
            public float DefaultSeconds { get; set; } = 2f;
            public string Expectation { get; set; } = string.Empty;
            public string Joint { get; set; } = string.Empty;
            public float TargetRad { get; set; } = 0.5f;
            public float MinJointRad { get; set; } = 0.3f;
            public string ProbeBodyA { get; set; } = string.Empty;
            public string ProbeBodyB { get; set; } = string.Empty;
            public Vector3 ProbePoint { get; set; }
            public Vector3 ProbeAxis { get; set; } = Vector3.forward;
            public float MinProbeOffset { get; set; } = 0.02f;
            public bool ProbeRelativeToRelease { get; set; }
            public string ProbeLabel { get; set; } = string.Empty;
            public float MaxLeadDisplacement { get; set; } = 0.02f;
            public float MaxSpeed { get; set; } = 0.01f;
            public float MaxJointFromRest { get; set; } = 0.05f;
            public float ExpectedReferenceHeight { get; set; }
            public float ReferenceHeightTolerance { get; set; } = 0.01f;
            public float MinUpright { get; set; } = -1f;
        }

        // --------------------------------------------------------------- assets --

        public static void EnsureFolder(string path)
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

        public static Material UpsertMaterial(string path, Shader shader, Color color, float smoothness)
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

        /// <summary>
        /// Static = dynamic friction, no bounce, Maximum combine (MuJoCo takes the larger of two
        /// frictions). Use the legacy ".physicMaterial" extension: "*.physicsMaterial" gets
        /// DefaultImporter on this Unity version and never loads.
        /// </summary>
        public static PhysicsMaterial UpsertPhysicsMaterial(string path, string name, float friction)
        {
            var material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
            bool isNew = material == null;
            if (isNew)
            {
                material = new PhysicsMaterial(name);
            }
            material.staticFriction = friction;
            material.dynamicFriction = friction;
            material.bounciness = 0f;
            material.frictionCombine = PhysicsMaterialCombine.Maximum;
            material.bounceCombine = PhysicsMaterialCombine.Minimum;
            if (isNew)
            {
                AssetDatabase.CreateAsset(material, path);
            }
            else
            {
                EditorUtility.SetDirty(material);
            }
            return AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
        }

        /// <summary>A landscape desktop HUD panel, copied from the race HUD's so the project font and theme come along.</summary>
        public static PanelSettings UpsertPanelSettings(string path, StringBuilder log)
        {
            var panel = AssetDatabase.LoadAssetAtPath<PanelSettings>(path);
            if (panel == null)
            {
                var source = AssetDatabase.LoadAssetAtPath<PanelSettings>(SOURCE_PANEL_SETTINGS);
                panel = source != null ? Object.Instantiate(source) : ScriptableObject.CreateInstance<PanelSettings>();
                AssetDatabase.CreateAsset(panel, path);
            }
            if (panel.themeStyleSheet == null)
            {
                panel.themeStyleSheet = AssetDatabase.LoadAssetAtPath<ThemeStyleSheet>(DEFAULT_THEME);
            }
            panel.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            panel.referenceResolution = new Vector2Int(REFERENCE_WIDTH, REFERENCE_HEIGHT);
            panel.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight;
            panel.match = 0.5f;
            EditorUtility.SetDirty(panel);
            log.Append("panel settings: ").Append(path)
               .Append(panel.themeStyleSheet == null ? " (NO THEME - HUD text may not render)\n" : "\n");
            return panel;
        }

        /// <summary>
        /// Copies <paramref name="source"/> (outside Assets/, e.g. a trainer's export) over
        /// <paramref name="destination"/> when it is missing or differs, and imports it.
        /// Returns "copied", "up to date" or "no source".
        /// </summary>
        public static string SyncFile(string source, string destination)
        {
            if (!File.Exists(source))
            {
                return "no source";
            }
            if (File.Exists(destination) && FilesEqual(source, destination))
            {
                return "up to date";
            }
            File.Copy(source, destination, true);
            AssetDatabase.ImportAsset(destination, ImportAssetOptions.ForceUpdate);
            return "copied";
        }

        // --------------------------------------------------- serialized config --

        public static void SetObject(SerializedProperty parent, string field, Object value)
        {
            Find(parent, field).objectReferenceValue = value;
        }

        public static void SetString(SerializedProperty parent, string field, string value)
        {
            Find(parent, field).stringValue = value;
        }

        public static void SetFloat(SerializedProperty parent, string field, float value)
        {
            Find(parent, field).floatValue = value;
        }

        public static void SetInt(SerializedProperty parent, string field, int value)
        {
            Find(parent, field).intValue = value;
        }

        public static void SetBool(SerializedProperty parent, string field, bool value)
        {
            Find(parent, field).boolValue = value;
        }

        public static void SetVector3(SerializedProperty parent, string field, Vector3 value)
        {
            Find(parent, field).vector3Value = value;
        }

        /// <summary>
        /// Puts the known racers in lanes 0..n-1 of the config's racer list, in roster order.
        /// An existing entry (matched by name) keeps its Physics choice; a new one gets the
        /// roster's default. Entries the roster does not know stay, after the known ones.
        /// </summary>
        public static void UpsertRacers(SerializedProperty config, RacerSpec[] roster, Material[] materials,
                                        string brainsFolder, string exportFolder, StringBuilder log)
        {
            SerializedProperty racers = Find(config, "_racers");
            for (int lane = 0; lane < roster.Length; lane++)
            {
                RacerSpec spec = roster[lane];
                int existing = FindRacer(racers, spec.Name, lane);
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

                string brainPath = brainsFolder + "/" + spec.BrainFile;
                var brain = AssetDatabase.LoadAssetAtPath<ModelAsset>(brainPath);
                SerializedProperty entry = racers.GetArrayElementAtIndex(lane);
                SetString(entry, "_name", spec.Name);
                SetString(entry, "_method", spec.Method);
                SetString(entry, "_brainFile", spec.BrainFile);
                SetObject(entry, "_brain", brain);
                SetObject(entry, "_material", materials[lane]);
                Find(entry, "_color").colorValue = spec.Color;
                SerializedProperty physics = Find(entry, "_physics");
                if (isNew)
                {
                    physics.enumValueIndex = (int)spec.DefaultPhysics;
                }

                log.Append("  lane ").Append(lane).Append(": ").Append(spec.Name).Append(" (")
                   .Append(spec.Method).Append(", ")
                   .Append(physics.enumValueIndex == (int)CreaturePhysicsKind.PhysxArticulation
                       ? "PhysX ArticulationBody"
                       : "MuJoCo plug-in")
                   .Append(isNew ? ", new" : string.Empty).Append(") brain: ")
                   .Append(brain != null
                       ? brainPath
                       : "MISSING - copy " + exportFolder + spec.BrainFile + " to " + brainPath
                         + " (the racer holds its rest pose, HUD says NO BRAIN)")
                   .Append('\n');
            }
            for (int lane = roster.Length; lane < racers.arraySize; lane++)
            {
                log.Append("  lane ").Append(lane).Append(": ")
                   .Append(Find(racers.GetArrayElementAtIndex(lane), "_name").stringValue)
                   .Append(" (added by hand, kept as is)\n");
            }
        }

        /// <summary>Replaces the config's self-test list with <paramref name="tests"/>.</summary>
        public static void WriteSelfTests(SerializedProperty config, SelfTestSpec[] tests)
        {
            SerializedProperty list = Find(config, "_selfTests");
            list.arraySize = tests.Length;
            for (int index = 0; index < tests.Length; index++)
            {
                SelfTestSpec test = tests[index];
                SerializedProperty entry = list.GetArrayElementAtIndex(index);
                SetString(entry, "_name", test.Name);
                SetString(entry, "_alias", test.Alias);
                Find(entry, "_kind").enumValueIndex = (int)test.Kind;
                SetFloat(entry, "_defaultSeconds", test.DefaultSeconds);
                SetString(entry, "_expectation", test.Expectation);
                SetString(entry, "_joint", test.Joint);
                SetFloat(entry, "_targetRad", test.TargetRad);
                SetFloat(entry, "_minJointRad", test.MinJointRad);
                SetString(entry, "_probeBodyA", test.ProbeBodyA);
                SetString(entry, "_probeBodyB", test.ProbeBodyB);
                SetVector3(entry, "_probePoint", test.ProbePoint);
                SetVector3(entry, "_probeAxis", test.ProbeAxis);
                SetFloat(entry, "_minProbeOffset", test.MinProbeOffset);
                SetBool(entry, "_probeRelativeToRelease", test.ProbeRelativeToRelease);
                SetString(entry, "_probeLabel", test.ProbeLabel);
                SetFloat(entry, "_maxLeadDisplacement", test.MaxLeadDisplacement);
                SetFloat(entry, "_maxSpeed", test.MaxSpeed);
                SetFloat(entry, "_maxJointFromRest", test.MaxJointFromRest);
                SetFloat(entry, "_expectedReferenceHeight", test.ExpectedReferenceHeight);
                SetFloat(entry, "_referenceHeightTolerance", test.ReferenceHeightTolerance);
                SetFloat(entry, "_minUpright", test.MinUpright);
            }
        }

        // ---------------------------------------------------------------- scene --

        public static void BuildLight()
        {
            var lightObject = new GameObject("Directional Light");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.shadows = LightShadows.Soft;
            light.intensity = LIGHT_INTENSITY;
            lightObject.transform.rotation = Quaternion.Euler(LightEuler);
        }

        /// <summary>The follow camera, placed where it starts (the start line) and told its offset.</summary>
        public static void BuildCamera(CreatureRaceConfig config, Vector3 offset, float lookAhead)
        {
            var cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            var camera = cameraObject.AddComponent<Camera>();
            camera.nearClipPlane = CAMERA_NEAR;
            camera.farClipPlane = CAMERA_FAR;
            camera.fieldOfView = CAMERA_FIELD_OF_VIEW;
            cameraObject.AddComponent<AudioListener>();
            var view = cameraObject.AddComponent<CreatureRaceCameraView>();
            var serialized = new SerializedObject(view);
            serialized.FindProperty("_offset").vector3Value = offset;
            serialized.FindProperty("_lookAhead").floatValue = lookAhead;
            serialized.ApplyModifiedPropertiesWithoutUndo();

            var focus = new Vector3(0f, 0f, config.StartLineZ);
            cameraObject.transform.position = focus + offset;
            cameraObject.transform.rotation = Quaternion.LookRotation(
                focus + Vector3.forward * lookAhead - cameraObject.transform.position, Vector3.up);
        }

        /// <summary>
        /// Ground (the only track collider, top at y = 0 where the MuJoCo world puts its plane),
        /// one white line per lane edge, a white start line and a chequered finish line.
        /// </summary>
        public static void BuildTrack(CreatureRaceConfig config, Material ground, Material white, Material black,
                                      PhysicsMaterial physicsMaterial)
        {
            var track = new GameObject("Track");
            float spacing = config.LaneSpacing;
            float start = config.StartLineZ;
            float finish = start + config.TrackLength;
            int laneCount = Mathf.Max(1, config.LaneCount);
            // Lanes are centred on x = 0 (CreatureRaceConfig.LaneX), one lane width each.
            float halfWidth = laneCount * spacing * 0.5f;

            float groundStart = start - GROUND_END_MARGIN;
            float groundEnd = finish + GROUND_END_MARGIN;
            GameObject groundObject = Block("Ground", track.transform, ground,
                new Vector3(0f, -GROUND_THICKNESS * 0.5f, (groundStart + groundEnd) * 0.5f),
                new Vector3(2f * (halfWidth + GROUND_SIDE_MARGIN), GROUND_THICKNESS, groundEnd - groundStart),
                keepCollider: true);
            groundObject.GetComponent<BoxCollider>().sharedMaterial = physicsMaterial;

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

            // Chequered finish, leading edge exactly on the finish line.
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

            Transform[] all = track.GetComponentsInChildren<Transform>(true);
            for (int index = 0; index < all.Length; index++)
            {
                GameObjectUtility.SetStaticEditorFlags(all[index].gameObject, StaticEditorFlags.BatchingStatic);
            }
        }

        public static UIDocument BuildHud(string name, PanelSettings panel)
        {
            var hudObject = new GameObject(name);
            var document = hudObject.AddComponent<UIDocument>();
            document.panelSettings = panel;
            hudObject.AddComponent<CreatureRaceHudView>();
            return document;
        }

        /// <summary>The two links the worm builder once lost, re-read from the live objects.</summary>
        public static string UnlinkedReferences(Component scope, string settingsField, Object settings,
                                                UIDocument hud, PanelSettings panel)
        {
            var problems = new StringBuilder();
            if (settings == null)
            {
                problems.Append("the settings asset is not loaded; ");
            }
            else if (new SerializedObject(scope).FindProperty(settingsField).objectReferenceValue != settings)
            {
                problems.Append(scope.GetType().Name).Append('.').Append(settingsField)
                        .Append(" did not take the settings asset; ");
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
        public static string LinksMissingOnDisk(string scenePath, string settingsField, Object settings,
                                                PanelSettings panel)
        {
            string text = File.ReadAllText(scenePath);
            var problems = new StringBuilder();
            string settingsPath = AssetDatabase.GetAssetPath(settings);
            string settingsGuid = AssetDatabase.AssetPathToGUID(settingsPath);
            if (!text.Contains(settingsField + ": {fileID: 11400000, guid: " + settingsGuid))
            {
                problems.Append("the saved scene has no ").Append(settingsField).Append(" link to ")
                        .Append(settingsPath).Append("; ");
            }
            string panelPath = AssetDatabase.GetAssetPath(panel);
            string panelGuid = AssetDatabase.AssetPathToGUID(panelPath);
            if (!text.Contains("m_PanelSettings: {fileID: 11400000, guid: " + panelGuid))
            {
                problems.Append("the saved scene has no PanelSettings link to ").Append(panelPath).Append("; ");
            }
            return problems.ToString();
        }

        // -------------------------------------------------------------- helpers --

        private static SerializedProperty Find(SerializedProperty parent, string field)
        {
            SerializedProperty property = parent.FindPropertyRelative(field);
            if (property == null)
            {
                throw new System.ArgumentException($"serialized field '{field}' not found under '{parent.propertyPath}'");
            }
            return property;
        }

        private static int FindRacer(SerializedProperty racers, string name, int from)
        {
            for (int index = from; index < racers.arraySize; index++)
            {
                if (Find(racers.GetArrayElementAtIndex(index), "_name").stringValue == name)
                {
                    return index;
                }
            }
            return -1;
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

        private static bool FilesEqual(string first, string second)
        {
            var infoFirst = new FileInfo(first);
            var infoSecond = new FileInfo(second);
            if (infoFirst.Length != infoSecond.Length)
            {
                return false;
            }
            byte[] bytesFirst = File.ReadAllBytes(first);
            byte[] bytesSecond = File.ReadAllBytes(second);
            for (int index = 0; index < bytesFirst.Length; index++)
            {
                if (bytesFirst[index] != bytesSecond[index])
                {
                    return false;
                }
            }
            return true;
        }
    }
}
#endif
