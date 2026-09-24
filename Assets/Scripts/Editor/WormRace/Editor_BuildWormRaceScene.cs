#if UNITY_EDITOR
using System;
using System.Text;
using PoRacer.CreatureRace;
using PoRacer.CreatureRace.EditorTools;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace PoRacer.WormRace.EditorTools
{
    /// <summary>
    /// Builds Assets/Scenes/SCN_WORM_RACE.unity and the assets it needs, from scratch,
    /// re-runnably (AGENTS rule G: the track, lines, camera, light, HUD and LifetimeScope are
    /// authored scene objects; only the worms and the MuJoCo world are created at race start,
    /// see WormSpawnSystem). The shared parts come from the creature template's
    /// CreatureRaceSceneKit; this file holds only what is the worm's.
    ///
    /// Creates or updates, keeping GUIDs:
    ///   Assets/WormRace/WormRaceSettings.asset        its Race config (rig, racers, track, timing,
    ///                                                 observation, self-tests, labels) + PhysX material
    ///   Assets/WormRace/PM_WormRace.physicMaterial    0.9 / 0.9, no bounce (worm_rig.json)
    ///   Assets/WormRace/Materials/*.mat               one per racer (BLUE, ORANGE, PURPLE), track
    ///   Assets/WormRace/UI/WormRacePanelSettings.asset
    /// Red and green are never used (AGENTS rule D).
    ///
    /// The racer list is seeded from <see cref="Roster"/>: names, methods, colours, materials and
    /// brains are refreshed on every run, but a racer's Physics choice, once made in the
    /// Inspector, is never overwritten. Racers added by hand after the known ones are kept.
    ///
    /// ORDER MATTERS: the new scene is created BEFORE any asset is loaded (see the kit), and the
    /// saved file is read back to check the scope -> settings and HUD -> panel links.
    ///
    /// Invoke: unity cmd eval --code "return PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build();"
    /// </summary>
    public static class Editor_BuildWormRaceScene
    {
        private const string REBUILD_COMMAND = "PoRacer.WormRace.EditorTools.Editor_BuildWormRaceScene.Build()";
        private const string EXPORT_FOLDER = "training/worm/export/";
        private const string PANEL_SETTINGS = WormRacePaths.UI + "/WormRacePanelSettings.asset";
        private const string PHYSICS_MATERIAL = WormRacePaths.ROOT + "/PM_WormRace.physicMaterial";
        private const string MATERIAL_MUJOCO = WormRacePaths.MATERIALS + "/M_WormMuJoCo_Blue.mat";
        private const string MATERIAL_ISAAC = WormRacePaths.MATERIALS + "/M_WormIsaac_Orange.mat";
        private const string MATERIAL_ISAACLAB3 = WormRacePaths.MATERIALS + "/M_WormIsaacLab3_Purple.mat";
        private const string MATERIAL_GROUND = WormRacePaths.MATERIALS + "/M_WormTrack_Ground.mat";
        private const string MATERIAL_WHITE = WormRacePaths.MATERIALS + "/M_WormTrack_LineWhite.mat";
        private const string MATERIAL_BLACK = WormRacePaths.MATERIALS + "/M_WormTrack_LineBlack.mat";
        private const string SETTINGS_FIELD = "_settings";
        private const string CONFIG_FIELD = "_race";

        private const float DEFAULT_FRICTION = 0.9f;
        private const float DEFAULT_RADIUS = 0.045f;
        private const float SIGN_TEST_RAD = 0.5f;

        private static readonly Color MujocoBlue = new(0.16f, 0.45f, 0.95f);
        private static readonly Color IsaacOrange = new(1.0f, 0.55f, 0.10f);
        private static readonly Color IsaacLab3Purple = new(0.60f, 0.30f, 0.85f);
        private static readonly Color GroundGrey = new(0.30f, 0.32f, 0.35f);
        private static readonly Color LineWhite = new(0.92f, 0.92f, 0.92f);
        private static readonly Color LineBlack = new(0.07f, 0.07f, 0.08f);
        private static readonly Vector3 CameraOffset = new(5.5f, 3.2f, -3.0f);
        private const float CAMERA_LOOK_AHEAD = 1.5f;

        /// <summary>
        /// The known racers, lane 0 first. Lane 2's brain comes from Isaac Lab 3 on Newton's
        /// MuJoCo-Warp solver, hence MuJoCo by default; switch it to PhysX in the settings if
        /// that trainer falls back to PhysX.
        /// </summary>
        private static readonly CreatureRaceSceneKit.RacerSpec[] Roster =
        {
            new("MuJoCo worm", "MuJoCo", "worm_mujoco.onnx", MATERIAL_MUJOCO, MujocoBlue,
                CreaturePhysicsKind.MujocoPlugin),
            new("Isaac worm", "Isaac Lab", "worm_isaac.onnx", MATERIAL_ISAAC, IsaacOrange,
                CreaturePhysicsKind.PhysxArticulation),
            new("Isaac3Worm", "Isaac Lab 3", "worm_isaaclab3.onnx", MATERIAL_ISAACLAB3, IsaacLab3Purple,
                CreaturePhysicsKind.MujocoPlugin),
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
            Shader lit = Shader.Find(CreatureRaceSceneKit.URP_LIT);
            if (lit == null)
            {
                return $"ABORT: shader '{CreatureRaceSceneKit.URP_LIT}' not found; is URP installed and active?";
            }

            // FIRST, before any asset is loaded or created (CreatureRaceSceneKit summary).
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var log = new StringBuilder();
            CreatureRaceSceneKit.EnsureFolder(WormRacePaths.ROOT);
            CreatureRaceSceneKit.EnsureFolder(WormRacePaths.MATERIALS);
            CreatureRaceSceneKit.EnsureFolder(WormRacePaths.BRAINS);
            CreatureRaceSceneKit.EnsureFolder(WormRacePaths.UI);

            var racerMaterials = new Material[Roster.Length];
            for (int index = 0; index < Roster.Length; index++)
            {
                racerMaterials[index] = CreatureRaceSceneKit.UpsertMaterial(Roster[index].MaterialPath, lit,
                                                                            Roster[index].Color, 0.45f);
            }
            Material ground = CreatureRaceSceneKit.UpsertMaterial(MATERIAL_GROUND, lit, GroundGrey, 0.15f);
            Material white = CreatureRaceSceneKit.UpsertMaterial(MATERIAL_WHITE, lit, LineWhite, 0.2f);
            Material black = CreatureRaceSceneKit.UpsertMaterial(MATERIAL_BLACK, lit, LineBlack, 0.2f);
            log.Append("materials: blue (MuJoCo), orange (Isaac), purple (Isaac Lab 3), ground, line white/black\n");

            var rigJson = AssetDatabase.LoadAssetAtPath<TextAsset>(WormRacePaths.RIG_JSON);
            RigNumbers numbers = ReadRigNumbers(rigJson);
            PhysicsMaterial physicsMaterial = CreatureRaceSceneKit.UpsertPhysicsMaterial(
                PHYSICS_MATERIAL, "PM_WormRace", numbers.friction);
            log.Append($"physics material: {PHYSICS_MATERIAL} ({numbers.friction:0.00} static/dynamic, no bounce)\n");

            PanelSettings panel = CreatureRaceSceneKit.UpsertPanelSettings(PANEL_SETTINGS, log);
            WormRaceSettings settings = UpsertSettings(rigJson, numbers, racerMaterials, physicsMaterial, log);
            AssetDatabase.SaveAssets();

            WormRaceLifetimeScope scope = BuildScope(settings);
            CreatureRaceSceneKit.BuildLight();
            CreatureRaceSceneKit.BuildCamera(settings.Race, CameraOffset, CAMERA_LOOK_AHEAD);
            CreatureRaceSceneKit.BuildTrack(settings.Race, ground, white, black, physicsMaterial);
            UIDocument hud = CreatureRaceSceneKit.BuildHud("WormRaceHud", panel);
            log.Append("scene objects: WormRaceLifetimeScope, Directional Light, Main Camera, Track (")
               .Append(settings.Race.LaneCount).Append(" lanes), WormRaceHud\n");

            string unlinked = CreatureRaceSceneKit.UnlinkedReferences(scope, SETTINGS_FIELD, settings, hud, panel);
            if (unlinked.Length > 0)
            {
                return log.Append("ABORT before saving: ").Append(unlinked).ToString();
            }
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene, WormRacePaths.SCENE))
            {
                return log.Append("ABORT: could not save ").Append(WormRacePaths.SCENE).ToString();
            }
            string missingOnDisk = CreatureRaceSceneKit.LinksMissingOnDisk(WormRacePaths.SCENE, SETTINGS_FIELD,
                                                                          settings, panel);
            if (missingOnDisk.Length > 0)
            {
                return log.Append("ERROR: saved, but ").Append(missingOnDisk).ToString();
            }
            log.Append("saved ").Append(WormRacePaths.SCENE)
               .Append(" (checked on disk: scope -> settings, HUD -> panel settings)\n");
            return log.ToString();
        }

        // --------------------------------------------------------------- assets --

        private static WormRaceSettings UpsertSettings(TextAsset rigJson, RigNumbers numbers, Material[] racerMaterials,
                                                       PhysicsMaterial physicsMaterial, StringBuilder log)
        {
            var settings = AssetDatabase.LoadAssetAtPath<WormRaceSettings>(WormRacePaths.SETTINGS);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<WormRaceSettings>();
                AssetDatabase.CreateAsset(settings, WormRacePaths.SETTINGS);
            }

            var serialized = new SerializedObject(settings);
            serialized.FindProperty("_wormPhysicsMaterial").objectReferenceValue = physicsMaterial;
            SerializedProperty config = serialized.FindProperty(CONFIG_FIELD);
            log.Append("settings: ").Append(WormRacePaths.SETTINGS).Append('\n');
            log.Append("  rig: ").Append(rigJson != null
                ? WormRacePaths.RIG_JSON
                : "MISSING - copy training/worm/worm_rig.json to " + WormRacePaths.RIG_JSON).Append('\n');

            CreatureRaceSceneKit.SetObject(config, "_rigJson", rigJson);
            // WORM_SPEC: the nose is the tip of segment 0; the observation is built in segment 2.
            CreatureRaceSceneKit.SetString(config, "_leadBody", "seg0");
            SerializedProperty observation = config.FindPropertyRelative("_observation");
            CreatureRaceSceneKit.SetString(observation, "_referenceBody", "seg2");
            CreatureRaceSceneKit.SetFloat(observation, "_linearVelocityScale", 0.5f);
            CreatureRaceSceneKit.SetFloat(observation, "_angularVelocityScale", 0.25f);
            CreatureRaceSceneKit.SetFloat(observation, "_jointVelocityScale", 0.1f);
            CreatureRaceSceneKit.SetBool(observation, "_jointPositionsRelativeToRest", false);
            CreatureRaceSceneKit.SetBool(observation, "_includeTargetSpeed", false);
            CreatureRaceSceneKit.SetFloat(observation, "_targetSpeed", 0f);
            CreatureRaceSceneKit.SetFloat(observation, "_targetSpeedDivisor", 2f);
            CreatureRaceSceneKit.SetBool(config, "_previousActionClipped", true);

            CreatureRaceSceneKit.UpsertRacers(config, Roster, racerMaterials, WormRacePaths.BRAINS, EXPORT_FOLDER, log);

            CreatureRaceSceneKit.SetFloat(config, "_trackLength", 20f);
            CreatureRaceSceneKit.SetFloat(config, "_laneSpacing", 2f);
            CreatureRaceSceneKit.SetFloat(config, "_startLineZ", 0f);
            CreatureRaceSceneKit.SetFloat(config, "_startGap", 0.01f);
            CreatureRaceSceneKit.SetInt(config, "_countdownSeconds", 3);
            CreatureRaceSceneKit.SetFloat(config, "_timeLimitSeconds", 60f);
            CreatureRaceSceneKit.SetFloat(config, "_resultsHoldSeconds", 4f);
            CreatureRaceSceneKit.SetInt(config, "_seriesLength", 5);
            CreatureRaceSceneKit.SetBool(config, "_autoStartSeries", true);
            CreatureRaceSceneKit.SetFloat(config, "_selfTestSettleSeconds", 1f);
            // A worm rolls and has no 'up': falling is not tracked (-1).
            CreatureRaceSceneKit.SetFloat(config, "_fallenUprightThreshold", -1f);
            CreatureRaceSceneKit.WriteSelfTests(config, SelfTests(numbers.radius));
            CreatureRaceSceneKit.SetInt(config, "_mujocoSolverIterations", 10);
            CreatureRaceSceneKit.SetBool(config, "_dumpMujocoMjcf", true);
            CreatureRaceSceneKit.SetString(config, "_title", "WORM RACE");
            CreatureRaceSceneKit.SetString(config, "_reportPrefix", "wormrace");
            CreatureRaceSceneKit.SetString(config, "_finishRule",
                                           "first segment-0 nose past the finish line; time limit ranks by distance");
            CreatureRaceSceneKit.SetString(config, "_holdLabel", "holds straight");
            CreatureRaceSceneKit.SetString(config, "_brainsFolder", WormRacePaths.BRAINS);
            CreatureRaceSceneKit.SetString(config, "_rebuildCommand", REBUILD_COMMAND);

            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(settings);
            log.Append("  self-tests: zero, yaw, pitch\n");
            return settings;
        }

        /// <summary>
        /// The three worm self-tests (training/worm/UNITY_RACE.md section 3), exactly as the
        /// worm's own race system judged them before the creature template.
        /// </summary>
        private static CreatureRaceSceneKit.SelfTestSpec[] SelfTests(float radius)
        {
            return new[]
            {
                new CreatureRaceSceneKit.SelfTestSpec
                {
                    Name = "ZeroActionTest",
                    Alias = "zero",
                    Kind = CreatureSelfTestKind.RestPose,
                    DefaultSeconds = 5f,
                    Expectation = "zero action: nose moves < 2 cm, speed < 1 cm/s, every |joint| < 0.05 rad, "
                                + $"segment 2 centre at the capsule radius ({radius:0.000} m) +/- 1 cm, for every "
                                + "racer in its own physics",
                    MaxLeadDisplacement = 0.02f,
                    MaxSpeed = 0.01f,
                    MaxJointFromRest = 0.05f,
                    ExpectedReferenceHeight = radius,
                    ReferenceHeightTolerance = 0.01f,
                    MinUpright = -1f,
                },
                new CreatureRaceSceneKit.SelfTestSpec
                {
                    Name = "YawSignTest",
                    Alias = "yaw",
                    Kind = CreatureSelfTestKind.JointSign,
                    DefaultSeconds = 2f,
                    Expectation = "MuJoCo j0_yaw = +0.5 rad swings segment 1 toward MuJoCo -y = Unity +x (the worm's "
                                + "right when facing +Z): j0_yaw reads > +0.3 rad and segment 1 sits > 2 cm to the "
                                + "right of the head, for every racer in its own physics",
                    Joint = "j0_yaw",
                    TargetRad = SIGN_TEST_RAD,
                    MinJointRad = 0.3f,
                    ProbeBodyA = "seg0",
                    ProbeBodyB = "seg1",
                    ProbeAxis = new Vector3(0f, -1f, 0f),
                    MinProbeOffset = 0.02f,
                    ProbeLabel = "segment1 to the worm's right (Unity +x)",
                },
                new CreatureRaceSceneKit.SelfTestSpec
                {
                    Name = "PitchSignTest",
                    Alias = "pitch",
                    Kind = CreatureSelfTestKind.JointSign,
                    DefaultSeconds = 2f,
                    Expectation = "MuJoCo j0_pitch = +0.5 rad rotates segment 1 about +y, lifting it relative to the "
                                + "head: j0_pitch reads > +0.3 rad and segment 1 sits > 2 cm above the head's "
                                + "plane, for every racer in its own physics",
                    Joint = "j0_pitch",
                    TargetRad = SIGN_TEST_RAD,
                    MinJointRad = 0.3f,
                    ProbeBodyA = "seg0",
                    ProbeBodyB = "seg1",
                    ProbeAxis = new Vector3(0f, 0f, 1f),
                    MinProbeOffset = 0.02f,
                    ProbeLabel = "segment1 above the head's plane",
                },
            };
        }

        private static RigNumbers ReadRigNumbers(TextAsset rigJson)
        {
            var fallback = new RigNumbers { friction = DEFAULT_FRICTION, radius = DEFAULT_RADIUS };
            if (rigJson == null)
            {
                return fallback;
            }
            try
            {
                RigNumbers parsed = JsonUtility.FromJson<RigNumbers>(rigJson.text);
                return new RigNumbers
                {
                    friction = parsed != null && parsed.friction > 0f ? parsed.friction : DEFAULT_FRICTION,
                    radius = parsed != null && parsed.radius > 0f ? parsed.radius : DEFAULT_RADIUS,
                };
            }
            catch (ArgumentException)
            {
                return fallback;
            }
        }

        // ---------------------------------------------------------------- scene --

        private static WormRaceLifetimeScope BuildScope(WormRaceSettings settings)
        {
            var scopeObject = new GameObject("WormRaceLifetimeScope");
            var scope = scopeObject.AddComponent<WormRaceLifetimeScope>();
            var serialized = new SerializedObject(scope);
            serialized.FindProperty(SETTINGS_FIELD).objectReferenceValue = settings;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            return scope;
        }

        /// <summary>worm_rig.json's friction and capsule radius, read by name.</summary>
        [Serializable]
        private sealed class RigNumbers
        {
#pragma warning disable 0649
            public float friction;
            public float radius;
#pragma warning restore 0649
        }
    }
}
#endif
