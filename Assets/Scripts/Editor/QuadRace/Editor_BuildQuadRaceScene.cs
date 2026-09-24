#if UNITY_EDITOR
using System.Text;
using PoRacer.CreatureRace;
using PoRacer.CreatureRace.EditorTools;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

namespace PoRacer.QuadRace.EditorTools
{
    /// <summary>
    /// Builds Assets/Scenes/SCN_QUAD_RACE.unity: two quads trained on MuJoCo Warp race 30 m on
    /// the MuJoCo plug-in, lanes 3 m apart (a quad is about 1 m wide), 60 s limit. Everything
    /// the quad needs at runtime is the creature template; this builder only writes its
    /// settings and authors the scene (AGENTS rule G), re-runnably, keeping GUIDs:
    ///
    ///   Assets/QuadRace/quad_rig.json                 copied from training/quad/quad_rig.json when it changes
    ///   Assets/QuadRace/Brains/quad_*.onnx            copied from training/quad/export/ when present
    ///   Assets/QuadRace/QuadRaceSettings.asset        racers, track, timing, 36-float observation, self-tests
    ///   Assets/QuadRace/PM_QuadRace.physicMaterial    the rig's floor friction, no bounce
    ///   Assets/QuadRace/Materials/*.mat               BLUE (MuJoCo), PURPLE (Isaac Lab 3), track
    ///   Assets/QuadRace/UI/QuadRacePanelSettings.asset
    ///
    /// Red and green are never used (AGENTS rule D; quad.xml's green body is a viewer colour).
    /// A missing brain is not an error: that quad stands at rest and the HUD says NO BRAIN.
    ///
    /// Invoke: unity cmd eval --code "return PoRacer.QuadRace.EditorTools.Editor_BuildQuadRaceScene.Build();"
    /// </summary>
    public static class Editor_BuildQuadRaceScene
    {
        private const string REBUILD_COMMAND = "PoRacer.QuadRace.EditorTools.Editor_BuildQuadRaceScene.Build()";
        private const string PANEL_SETTINGS = QuadRacePaths.UI + "/QuadRacePanelSettings.asset";
        private const string PHYSICS_MATERIAL = QuadRacePaths.ROOT + "/PM_QuadRace.physicMaterial";
        private const string MATERIAL_MUJOCO = QuadRacePaths.MATERIALS + "/M_QuadMuJoCo_Blue.mat";
        private const string MATERIAL_ISAACLAB3 = QuadRacePaths.MATERIALS + "/M_QuadIsaacLab3_Purple.mat";
        private const string MATERIAL_GROUND = QuadRacePaths.MATERIALS + "/M_QuadTrack_Ground.mat";
        private const string MATERIAL_WHITE = QuadRacePaths.MATERIALS + "/M_QuadTrack_LineWhite.mat";
        private const string MATERIAL_BLACK = QuadRacePaths.MATERIALS + "/M_QuadTrack_LineBlack.mat";
        private const string SETTINGS_FIELD = "_settings";
        private const string CONFIG_FIELD = "_race";

        // Quad_v01 names (training/quad/quad_rig.json legNames), checked against the rig below.
        private const string DEFAULT_TORSO = "Quad_v01";
        private const string TEST_HIP = "Upper_-1_1";
        private const string TEST_KNEE = "Lower_-1_1";
        private const string TEST_LEG_LABEL = "front-left";
        private const float DEFAULT_REST_HEIGHT = 0.9f;
        private const float DEFAULT_FRICTION = 0.9f;
        private static readonly Vector3 DefaultFootPoint = new(0f, 0f, -0.108f);

        private const float TRACK_LENGTH = 30f;
        private const float LANE_SPACING = 3f;
        private const float TIME_LIMIT = 60f;
        private const float SIGN_TEST_RAD = 0.5f;

        private static readonly Color MujocoBlue = new(0.16f, 0.45f, 0.95f);
        private static readonly Color IsaacLab3Purple = new(0.60f, 0.30f, 0.85f);
        private static readonly Color GroundGrey = new(0.30f, 0.32f, 0.35f);
        private static readonly Color LineWhite = new(0.92f, 0.92f, 0.92f);
        private static readonly Color LineBlack = new(0.07f, 0.07f, 0.08f);
        private static readonly Vector3 CameraOffset = new(8f, 4.5f, -5f);
        private const float CAMERA_LOOK_AHEAD = 3f;

        private static readonly CreatureRaceSceneKit.RacerSpec[] Roster =
        {
            new("Quad (MuJoCo)", "MuJoCo", "quad_mujoco.onnx", MATERIAL_MUJOCO, MujocoBlue,
                CreaturePhysicsKind.MujocoPlugin),
            new("Quad (Isaac Lab 3)", "Isaac Lab 3", "quad_isaaclab3.onnx", MATERIAL_ISAACLAB3, IsaacLab3Purple,
                CreaturePhysicsKind.MujocoPlugin),
        };

        [MenuItem("PoRacer/Quad Race/Build Scene")]
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
            CreatureRaceSceneKit.EnsureFolder(QuadRacePaths.ROOT);
            CreatureRaceSceneKit.EnsureFolder(QuadRacePaths.MATERIALS);
            CreatureRaceSceneKit.EnsureFolder(QuadRacePaths.BRAINS);
            CreatureRaceSceneKit.EnsureFolder(QuadRacePaths.UI);

            log.Append("rig: ").Append(QuadRacePaths.TRAINING_RIG_JSON).Append(" -> ").Append(QuadRacePaths.RIG_JSON)
               .Append(" (").Append(CreatureRaceSceneKit.SyncFile(QuadRacePaths.TRAINING_RIG_JSON, QuadRacePaths.RIG_JSON))
               .Append(")\n");
            for (int lane = 0; lane < Roster.Length; lane++)
            {
                string source = QuadRacePaths.TRAINING_EXPORT + Roster[lane].BrainFile;
                string target = QuadRacePaths.BRAINS + "/" + Roster[lane].BrainFile;
                log.Append("brain: ").Append(source).Append(" -> ").Append(target).Append(" (")
                   .Append(CreatureRaceSceneKit.SyncFile(source, target)).Append(")\n");
            }

            var rigJson = AssetDatabase.LoadAssetAtPath<TextAsset>(QuadRacePaths.RIG_JSON);
            RigFacts facts = ReadRig(rigJson, log);

            var racerMaterials = new Material[Roster.Length];
            for (int index = 0; index < Roster.Length; index++)
            {
                racerMaterials[index] = CreatureRaceSceneKit.UpsertMaterial(Roster[index].MaterialPath, lit,
                                                                            Roster[index].Color, 0.45f);
            }
            Material ground = CreatureRaceSceneKit.UpsertMaterial(MATERIAL_GROUND, lit, GroundGrey, 0.15f);
            Material white = CreatureRaceSceneKit.UpsertMaterial(MATERIAL_WHITE, lit, LineWhite, 0.2f);
            Material black = CreatureRaceSceneKit.UpsertMaterial(MATERIAL_BLACK, lit, LineBlack, 0.2f);
            log.Append("materials: blue (MuJoCo), purple (Isaac Lab 3), ground, line white/black\n");
            PhysicsMaterial physicsMaterial = CreatureRaceSceneKit.UpsertPhysicsMaterial(
                PHYSICS_MATERIAL, "PM_QuadRace", facts.FloorFriction);
            log.Append($"physics material: {PHYSICS_MATERIAL} ({facts.FloorFriction:0.00}, no bounce)\n");

            PanelSettings panel = CreatureRaceSceneKit.UpsertPanelSettings(PANEL_SETTINGS, log);
            CreatureRaceSettings settings = UpsertSettings(rigJson, facts, racerMaterials, log);
            AssetDatabase.SaveAssets();

            CreatureRaceLifetimeScope scope = BuildScope(settings);
            CreatureRaceSceneKit.BuildLight();
            CreatureRaceSceneKit.BuildCamera(settings.Race, CameraOffset, CAMERA_LOOK_AHEAD);
            CreatureRaceSceneKit.BuildTrack(settings.Race, ground, white, black, physicsMaterial);
            UIDocument hud = CreatureRaceSceneKit.BuildHud("QuadRaceHud", panel);
            log.Append("scene objects: QuadRaceLifetimeScope, Directional Light, Main Camera, Track (")
               .Append(settings.Race.LaneCount).Append(" lanes, ").Append(TRACK_LENGTH).Append(" m), QuadRaceHud\n");

            string unlinked = CreatureRaceSceneKit.UnlinkedReferences(scope, SETTINGS_FIELD, settings, hud, panel);
            if (unlinked.Length > 0)
            {
                return log.Append("ABORT before saving: ").Append(unlinked).ToString();
            }
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene, QuadRacePaths.SCENE))
            {
                return log.Append("ABORT: could not save ").Append(QuadRacePaths.SCENE).ToString();
            }
            string missingOnDisk = CreatureRaceSceneKit.LinksMissingOnDisk(QuadRacePaths.SCENE, SETTINGS_FIELD,
                                                                          settings, panel);
            if (missingOnDisk.Length > 0)
            {
                return log.Append("ERROR: saved, but ").Append(missingOnDisk).ToString();
            }
            log.Append("saved ").Append(QuadRacePaths.SCENE)
               .Append(" (checked on disk: scope -> settings, HUD -> panel settings)\n");
            return log.ToString();
        }

        // --------------------------------------------------------------- assets --

        /// <summary>What the builder needs to know about the rig, from the runtime's own parser.</summary>
        private sealed class RigFacts
        {
            public string Torso { get; set; } = DEFAULT_TORSO;
            public float RestHeight { get; set; } = DEFAULT_REST_HEIGHT;
            public float FloorFriction { get; set; } = DEFAULT_FRICTION;
            public Vector3 FootPoint { get; set; } = DefaultFootPoint;
        }

        private static RigFacts ReadRig(TextAsset rigJson, StringBuilder log)
        {
            var facts = new RigFacts();
            if (rigJson == null)
            {
                log.Append("  MISSING: ").Append(QuadRacePaths.RIG_JSON).Append(" - the other agent's build_quad.py "
                         + "writes training/quad/quad_rig.json; re-run this builder once it exists. The scene is wired "
                         + "to the path and reports the missing rig at play time.\n");
                return facts;
            }
            if (!CreatureRigParser.TryParse(rigJson.text, out CreatureRig rig, out string error))
            {
                log.Append("  ERROR: ").Append(QuadRacePaths.RIG_JSON).Append(" does not load: ").Append(error).Append('\n');
                return facts;
            }
            facts.Torso = rig.Bodies[rig.TorsoBody].Name;
            if (float.IsFinite(rig.RestRootHeight))
            {
                facts.RestHeight = rig.RestRootHeight;
            }
            facts.FloorFriction = rig.FloorContact.Friction.x;
            int knee = rig.FindBody(TEST_KNEE);
            facts.FootPoint = knee >= 0 ? LowestPoint(rig, knee) : DefaultFootPoint;

            // QUAD_SPEC: 9 + 3 x actions + goal (2) + target speed (1) = 36 for eight joints.
            int observationSize = 9 + 3 * rig.ActionSize + 3;
            log.Append("  rig '").Append(rig.Name).Append("': ").Append(rig.Bodies.Count).Append(" bodies, ")
               .Append(rig.Geoms.Count).Append(" geoms, ").Append(rig.Joints.Count).Append(" hinges, ")
               .Append(rig.ActionSize).Append(" actions, torso ").Append(facts.Torso)
               .Append(", nose ").Append(rig.ForwardExtent(rig.TorsoBody).ToString("0.000")).Append(" m ahead, rest ")
               .Append(facts.RestHeight.ToString("0.00")).Append(" m, spawn ")
               .Append(float.IsFinite(rig.SpawnRootHeight) ? rig.SpawnRootHeight.ToString("0.00") : "pos").Append(" m, ")
               .Append("target speed ").Append(float.IsFinite(rig.TargetSpeed) ? rig.TargetSpeed.ToString("0.00") : "none")
               .Append(" m/s, observation ").Append(observationSize).Append(" floats\n");
            if (rig.FindAction(TEST_HIP) < 0 || rig.FindAction(TEST_KNEE) < 0 || knee < 0)
            {
                log.Append("  WARNING: the self-tests drive ").Append(TEST_HIP).Append(" and ").Append(TEST_KNEE)
                   .Append(", which this rig does not have; hip/knee tests will report an error.\n");
            }
            return facts;
        }

        /// <summary>The lowest end of a body's first capsule, body frame: the foot of a lower leg.</summary>
        private static Vector3 LowestPoint(CreatureRig rig, int body)
        {
            for (int geomIndex = 0; geomIndex < rig.Geoms.Count; geomIndex++)
            {
                CreatureGeomDef geom = rig.Geoms[geomIndex];
                if (geom.Body != body || geom.Type != CreatureGeomType.Capsule || !geom.HasFromTo)
                {
                    continue;
                }
                return geom.From.z < geom.To.z ? geom.From : geom.To;
            }
            return DefaultFootPoint;
        }

        private static CreatureRaceSettings UpsertSettings(TextAsset rigJson, RigFacts facts, Material[] racerMaterials,
                                                           StringBuilder log)
        {
            var settings = AssetDatabase.LoadAssetAtPath<CreatureRaceSettings>(QuadRacePaths.SETTINGS);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<CreatureRaceSettings>();
                AssetDatabase.CreateAsset(settings, QuadRacePaths.SETTINGS);
            }

            var serialized = new SerializedObject(settings);
            SerializedProperty config = serialized.FindProperty(CONFIG_FIELD);
            log.Append("settings: ").Append(QuadRacePaths.SETTINGS).Append('\n');

            CreatureRaceSceneKit.SetObject(config, "_rigJson", rigJson);
            // QUAD_SPEC: the torso is the reference body; its front is the finish point.
            CreatureRaceSceneKit.SetString(config, "_leadBody", string.Empty);
            SerializedProperty observation = config.FindPropertyRelative("_observation");
            CreatureRaceSceneKit.SetString(observation, "_referenceBody", string.Empty);
            CreatureRaceSceneKit.SetFloat(observation, "_linearVelocityScale", 0.5f);
            CreatureRaceSceneKit.SetFloat(observation, "_angularVelocityScale", 0.25f);
            CreatureRaceSceneKit.SetFloat(observation, "_jointVelocityScale", 0.1f);
            CreatureRaceSceneKit.SetBool(observation, "_jointPositionsRelativeToRest", false);
            CreatureRaceSceneKit.SetBool(observation, "_includeTargetSpeed", true);
            CreatureRaceSceneKit.SetFloat(observation, "_targetSpeed", 0f);
            CreatureRaceSceneKit.SetFloat(observation, "_targetSpeedDivisor", 2f);
            CreatureRaceSceneKit.SetBool(config, "_previousActionClipped", true);

            CreatureRaceSceneKit.UpsertRacers(config, Roster, racerMaterials, QuadRacePaths.BRAINS,
                                              QuadRacePaths.TRAINING_EXPORT, log);

            CreatureRaceSceneKit.SetFloat(config, "_trackLength", TRACK_LENGTH);
            CreatureRaceSceneKit.SetFloat(config, "_laneSpacing", LANE_SPACING);
            CreatureRaceSceneKit.SetFloat(config, "_startLineZ", 0f);
            CreatureRaceSceneKit.SetFloat(config, "_startGap", 0.01f);
            CreatureRaceSceneKit.SetInt(config, "_countdownSeconds", 3);
            CreatureRaceSceneKit.SetFloat(config, "_timeLimitSeconds", TIME_LIMIT);
            CreatureRaceSceneKit.SetFloat(config, "_resultsHoldSeconds", 4f);
            CreatureRaceSceneKit.SetInt(config, "_seriesLength", 5);
            CreatureRaceSceneKit.SetBool(config, "_autoStartSeries", true);
            CreatureRaceSceneKit.SetFloat(config, "_selfTestSettleSeconds", 1f);
            CreatureRaceSceneKit.SetFloat(config, "_fallenUprightThreshold", 0.5f);
            CreatureRaceSceneKit.WriteSelfTests(config, SelfTests(facts));
            CreatureRaceSceneKit.SetInt(config, "_mujocoSolverIterations", 10);
            CreatureRaceSceneKit.SetBool(config, "_dumpMujocoMjcf", true);
            CreatureRaceSceneKit.SetString(config, "_title", "QUAD RACE");
            CreatureRaceSceneKit.SetString(config, "_reportPrefix", "quadrace");
            CreatureRaceSceneKit.SetString(config, "_finishRule",
                                           "first torso front past the finish line; time limit ranks by distance; "
                                         + "a fallen quad is never stood up (rule H)");
            CreatureRaceSceneKit.SetString(config, "_holdLabel", "stands at rest");
            CreatureRaceSceneKit.SetString(config, "_brainsFolder", QuadRacePaths.BRAINS);
            CreatureRaceSceneKit.SetString(config, "_rebuildCommand", REBUILD_COMMAND);

            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(settings);
            log.Append("  track ").Append(TRACK_LENGTH).Append(" m, lanes ").Append(LANE_SPACING)
               .Append(" m apart, limit ").Append(TIME_LIMIT).Append(" s; self-tests: zero, hip, knee\n");
            return settings;
        }

        /// <summary>
        /// Legged self-tests. zero: every servo at rest must leave the quad standing still,
        /// upright, at its rest height. hip / knee: +0.5 rad on the front-left hip or knee must
        /// read positive and swing the front-left foot BACKWARD relative to the torso (MuJoCo
        /// -x): training/bugs/README.md's worked example, "+20 degrees swings the foot toward
        /// MuJoCo -X (backward)", for both joints since both turn about +y.
        /// </summary>
        private static CreatureRaceSceneKit.SelfTestSpec[] SelfTests(RigFacts facts)
        {
            return new[]
            {
                new CreatureRaceSceneKit.SelfTestSpec
                {
                    Name = "ZeroActionTest",
                    Alias = "zero",
                    Kind = CreatureSelfTestKind.RestPose,
                    DefaultSeconds = 5f,
                    Expectation = "zero action (every servo at its rest angle): the quad stands - chest moves < 5 cm, "
                                + "speed < 5 cm/s, every joint within 0.1 rad of rest, torso at the rest height "
                                + $"({facts.RestHeight:0.00} m) +/- 5 cm and upright (up . world up >= 0.9), for every racer",
                    MaxLeadDisplacement = 0.05f,
                    MaxSpeed = 0.05f,
                    MaxJointFromRest = 0.1f,
                    ExpectedReferenceHeight = facts.RestHeight,
                    ReferenceHeightTolerance = 0.05f,
                    MinUpright = 0.9f,
                },
                SignTest("HipSignTest", "hip", TEST_HIP, "hip", facts),
                SignTest("KneeSignTest", "knee", TEST_KNEE, "knee", facts),
            };
        }

        private static CreatureRaceSceneKit.SelfTestSpec SignTest(string name, string alias, string joint, string what,
                                                                  RigFacts facts)
        {
            return new CreatureRaceSceneKit.SelfTestSpec
            {
                Name = name,
                Alias = alias,
                Kind = CreatureSelfTestKind.JointSign,
                DefaultSeconds = 2f,
                Expectation = $"MuJoCo {joint} ({TEST_LEG_LABEL} {what}) = rest + 0.5 rad turns the leg about +y, "
                            + $"swinging the {TEST_LEG_LABEL} foot toward MuJoCo -x (backward, training/bugs/README.md): "
                            + $"{joint} reads > +0.3 rad and the foot moves > 2 cm backward relative to the torso, "
                            + "for every racer",
                Joint = joint,
                TargetRad = SIGN_TEST_RAD,
                MinJointRad = 0.3f,
                ProbeBodyA = facts.Torso,
                ProbeBodyB = TEST_KNEE,
                ProbePoint = facts.FootPoint,
                ProbeAxis = new Vector3(-1f, 0f, 0f),
                MinProbeOffset = 0.02f,
                ProbeRelativeToRelease = true,
                ProbeLabel = $"{TEST_LEG_LABEL} foot moved backward (MuJoCo -x)",
            };
        }

        // ---------------------------------------------------------------- scene --

        private static CreatureRaceLifetimeScope BuildScope(CreatureRaceSettings settings)
        {
            var scopeObject = new GameObject("QuadRaceLifetimeScope");
            var scope = scopeObject.AddComponent<CreatureRaceLifetimeScope>();
            var serialized = new SerializedObject(scope);
            serialized.FindProperty(SETTINGS_FIELD).objectReferenceValue = settings;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            return scope;
        }
    }
}
#endif
