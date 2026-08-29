using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

using PoRacer.IsaacPorts;

namespace MujocoBiped.EditorTools
{
    /// <summary>
    /// Authoring tools for the MujocoBiped creature. Everything this writes lands inside
    /// Assets/unity_export/MujocoBiped/ - it never edits a project setting, a layer, a
    /// build setting, another agent, or another scene's assets.
    ///
    /// Menu:
    ///   MujocoBiped / Rebuild Rig Asset From JSON  - MujocoBiped_rig.json -> the asset
    ///   MujocoBiped / Build Prefab                 - rig asset -> prefab + physics material
    ///   MujocoBiped / Spawn Into Open Scene        - window; spawns, leaves the scene dirty
    ///   MujocoBiped / Run Reference Check          - edit-mode ONNX check vs the recording
    /// </summary>
    public static class MujocoBipedSetup
    {
        public const string Root = MujocoBipedPaths.Root;
        public const string RigJsonPath = MujocoBipedPaths.RigJson;
        public const string RigAssetPath = MujocoBipedPaths.RigAsset;
        public const string PrefabPath = MujocoBipedPaths.Prefab;
        public const string MaterialPath = MujocoBipedPaths.Material;
        public const string OnnxPath = MujocoBipedPaths.Onnx;

        // ------------------------------------------------------------- rig asset --
        [MenuItem("MujocoBiped/Rebuild Rig Asset From JSON", priority = 0)]
        public static void RebuildRigAsset()
        {
            var asset = BuildRigAssetFromJson();
            if (asset == null) return;
            Selection.activeObject = asset;
            EditorGUIUtility.PingObject(asset);

            int dummies = 0, colliders = 0;
            foreach (var l in asset.links)
            {
                if (l.isDummy) dummies++;
                colliders += l.geoms?.Length ?? 0;
            }
            Debug.Log($"[MujocoBiped] rig asset rebuilt: {asset.links.Length} links " +
                      $"({asset.links.Length - dummies} real + {dummies} single-DOF " +
                      $"placeholders), {asset.jointOrder.Length} joints, {colliders} " +
                      $"collision shapes -> {RigAssetPath}");
        }

        public static MujocoBipedRigAsset BuildRigAssetFromJson()
        {
            if (!File.Exists(RigJsonPath))
            {
                Debug.LogError($"[MujocoBiped] {RigJsonPath} not found. Run: python extract_rig.py");
                return null;
            }

            var raw = MiniJson.Parse(File.ReadAllText(RigJsonPath)) as Dictionary<string, object>;
            if (raw == null)
            {
                Debug.LogError("[MujocoBiped] rig JSON did not parse to an object.");
                return null;
            }

            var asset = AssetDatabase.LoadAssetAtPath<MujocoBipedRigAsset>(RigAssetPath);
            bool isNew = asset == null;
            if (isNew) asset = ScriptableObject.CreateInstance<MujocoBipedRigAsset>();

            asset.source = MiniJson.Str(raw, "source");
            asset.jointOrder = MiniJson.StrArray(raw, "jointOrder");
            asset.obsDim = (int)MiniJson.Num(raw, "obsDim");
            asset.actDim = (int)MiniJson.Num(raw, "actDim");

            var timing = MiniJson.Obj(raw, "timing");
            asset.policyDt = MiniJson.Num(timing, "policyDt");
            asset.mujocoPhysicsDt = MiniJson.Num(timing, "mujocoPhysicsDt");
            asset.mujocoFrameSkip = (int)MiniJson.Num(timing, "mujocoFrameSkip");

            asset.spawnPosMuj = MiniJson.Vec3(MiniJson.Obj(raw, "spawn"), "posMuj");

            var obs = MiniJson.Obj(raw, "observation");
            asset.clipLinVel = MiniJson.Num(obs, "clipLinVel");
            asset.clipAngVel = MiniJson.Num(obs, "clipAngVel");
            asset.clipJointVel = MiniJson.Num(obs, "clipJointVel");
            asset.maxTargetDistance = MiniJson.Num(obs, "maxTargetDistance");
            asset.angularVelocityIsDoubleRotated =
                MiniJson.Bool(obs, "angularVelocityIsDoubleRotated");

            var task = MiniJson.Obj(raw, "task");
            asset.reachRadiusM = MiniJson.Num(task, "reachRadiusM");
            var dr = MiniJson.Arr(task, "targetDistanceRangeM");
            if (dr != null && dr.Count >= 2)
                asset.targetDistanceRangeM = new Vector2((float)(double)dr[0], (float)(double)dr[1]);
            asset.targetAngleRangeRad = MiniJson.Num(task, "targetAngleRangeRad");
            asset.targetHeightMuj = MiniJson.Num(task, "targetHeightMuj");
            var hz = MiniJson.Arr(task, "healthyZRange");
            if (hz != null && hz.Count >= 2)
                asset.healthyZRange = new Vector2((float)(double)hz[0], (float)(double)hz[1]);
            asset.minUprightness = MiniJson.Num(task, "minUprightness");
            asset.maxEpisodeSteps = (int)MiniJson.Num(task, "maxEpisodeSteps");

            var ph = MiniJson.Obj(raw, "physics");
            asset.physics = new MujocoBipedPhysicsDef
            {
                gravityMuj = MiniJson.Vec3(ph, "gravityMuj"),
                floorFriction = MiniJson.Num(ph, "floorFriction"),
                footFriction = MiniJson.Num(ph, "footFriction"),
                bodyFriction = MiniJson.Num(ph, "bodyFriction"),
                effectiveFootGroundFriction = MiniJson.Num(ph, "effectiveFootGroundFriction"),
                maxJointVelocity = MiniJson.Num(ph, "maxJointVelocity"),
                maxAngularVelocity = MiniJson.Num(ph, "maxAngularVelocity"),
                maxLinearVelocity = MiniJson.Num(ph, "maxLinearVelocity"),
                maxDepenetrationVelocity = MiniJson.Num(ph, "maxDepenetrationVelocity"),
                linearDamping = MiniJson.Num(ph, "linearDamping"),
                angularDamping = MiniJson.Num(ph, "angularDamping"),
                jointFriction = MiniJson.Num(ph, "jointFriction"),
                solverPositionIterations = (int)MiniJson.Num(ph, "solverPositionIterations"),
                solverVelocityIterations = (int)MiniJson.Num(ph, "solverVelocityIterations"),
                contactOffset = MiniJson.Num(ph, "contactOffset"),
                restOffset = MiniJson.Num(ph, "restOffset"),
                selfCollisionExcludesParentChildOnly =
                    MiniJson.Bool(ph, "selfCollisionExcludesParentChildOnly"),
                dummyLinkMass = MiniJson.Num(ph, "dummyLinkMass"),
                inertiaFloor = MiniJson.Num(ph, "inertiaFloor"),
            };

            var ev = MiniJson.Obj(raw, "eval");
            asset.mujocoTargetsReachedPerEpisode = MiniJson.Num(ev, "targetsReachedPerEpisode");
            asset.mujocoEpisodeLengthSteps = MiniJson.Num(ev, "episodeLengthSteps");
            asset.mujocoMeanClosingSpeed = MiniJson.Num(ev, "meanClosingSpeed");
            asset.mujocoSurvivedFullEpisodeFraction =
                MiniJson.Num(ev, "survivedFullEpisodeFraction");

            var links = MiniJson.Arr(raw, "links");
            asset.links = new MujocoBipedLinkDef[links.Count];
            for (int i = 0; i < links.Count; i++)
            {
                var b = links[i] as Dictionary<string, object>;
                var def = new MujocoBipedLinkDef
                {
                    name = MiniJson.Str(b, "name"),
                    parent = MiniJson.Str(b, "parent"),
                    isRoot = MiniJson.Bool(b, "isRoot"),
                    isDummy = MiniJson.Bool(b, "isDummy"),
                    mjBody = MiniJson.Str(b, "mjBody"),
                    mass = MiniJson.Num(b, "mass"),
                    comMuj = MiniJson.Vec3(b, "comMuj"),
                    inertiaDiagMuj = MiniJson.Vec3(b, "inertiaDiagMuj"),
                    localPosMuj = MiniJson.Vec3(b, "localPosMuj"),
                    localRotMujWxyz = MiniJson.Vec4(b, "localRotMujWxyz"),
                    armatureFoldExact = MiniJson.Num(b, "armatureFoldExact"),
                    armatureFoldNaive = MiniJson.Num(b, "armatureFoldNaive"),
                };

                var j = MiniJson.Obj(b, "joint");
                if (j != null)
                {
                    def.hasJoint = true;
                    def.joint = new MujocoBipedJointDef
                    {
                        name = MiniJson.Str(j, "name"),
                        index = (int)MiniJson.Num(j, "index"),
                        axisInChildMuj = MiniJson.Vec3(j, "axisInChildMuj"),
                        lowerRad = MiniJson.Num(j, "lowerRad"),
                        upperRad = MiniJson.Num(j, "upperRad"),
                        damping = MiniJson.Num(j, "damping"),
                        armature = MiniJson.Num(j, "armature"),
                        gear = MiniJson.Num(j, "gear"),
                        ctrlLower = MiniJson.Num(j, "ctrlLower"),
                        ctrlUpper = MiniJson.Num(j, "ctrlUpper"),
                    };
                }

                var geoms = MiniJson.Arr(b, "geoms");
                def.geoms = new MujocoBipedGeomDef[geoms?.Count ?? 0];
                for (int g = 0; g < def.geoms.Length; g++)
                {
                    var gd = geoms[g] as Dictionary<string, object>;
                    def.geoms[g] = new MujocoBipedGeomDef
                    {
                        name = MiniJson.Str(gd, "name"),
                        kind = MiniJson.Str(gd, "kind"),
                        a = MiniJson.Vec3(gd, "a"),
                        b = MiniJson.Vec3(gd, "b"),
                        radius = MiniJson.Num(gd, "r"),
                        pos = MiniJson.Vec3(gd, "pos"),
                        half = MiniJson.Vec3(gd, "half"),
                        friction = FrictionFor(MiniJson.Str(gd, "name"), asset),
                    };
                }

                asset.links[i] = def;
            }

            if (isNew) AssetDatabase.CreateAsset(asset, RigAssetPath);
            else EditorUtility.SetDirty(asset);
            AssetDatabase.SaveAssets();
            return asset;
        }

        /// <summary>The MJCF overrides friction only on the feet; everything else defaults.</summary>
        static float FrictionFor(string geomName, MujocoBipedRigAsset rig)
            => geomName != null && geomName.StartsWith("foot")
                ? rig.physics.footFriction
                : rig.physics.bodyFriction;

        // -------------------------------------------------------- physics material --
        /// <summary>
        /// The creature's own material.
        ///
        /// MuJoCo combines two geoms' friction by taking the ELEMENTWISE MAXIMUM, so the
        /// foot/floor pair ran at max(foot 1.2, floor 1.0) = 1.2 during training. Unity's
        /// PhysX has no Maximum-with-a-cap; its combine modes are Average, Multiply,
        /// Minimum, Maximum, and the HIGHER enum value wins a mismatched pair.
        ///
        /// This ships Minimum, which is the conservative choice: it can never produce MORE
        /// grip than the scene's own ground offers, so dropping the creature into someone
        /// else's level cannot make that level's ground unexpectedly sticky. The cost is
        /// real and worth stating plainly - against a ground with no material at all,
        /// Unity's default is 0.6, so the effective pair friction is 0.6 rather than
        /// MuJoCo's 1.2. SCN_RACE_FLAT builds its ground with a null physics material, so
        /// that is exactly the case there.
        ///
        /// Maximum at 1.2 would reproduce MuJoCo's rule exactly against any ground. The
        /// rung-6 sweep measures both; README_UNITY.md carries the result and the
        /// one-line change.
        /// </summary>
        public static PhysicsMaterial CreateOrUpdateMaterial(MujocoBipedRigAsset rig)
        {
            var mat = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(MaterialPath);
            bool isNew = mat == null;
            if (isNew) mat = new PhysicsMaterial("PM_MujocoBiped");

            mat.staticFriction = rig.physics.effectiveFootGroundFriction;
            mat.dynamicFriction = rig.physics.effectiveFootGroundFriction;
            mat.bounciness = 0f;
            mat.frictionCombine = PhysicsMaterialCombine.Minimum;
            mat.bounceCombine = PhysicsMaterialCombine.Minimum;

            Directory.CreateDirectory(Root);
            // CreateAsset logs a complaint about the ".physicsMaterial" extension - it
            // wants ".asset" - but it does write a correct, loadable file, and
            // ".physicsMaterial" is the extension Unity's own Create menu produces. The
            // asset is not in the database until it has been imported, so a Load straight
            // after SaveAssets returns null; import it explicitly rather than guessing.
            if (isNew) AssetDatabase.CreateAsset(mat, MaterialPath);
            else EditorUtility.SetDirty(mat);
            AssetDatabase.SaveAssets();
            AssetDatabase.ImportAsset(MaterialPath, ImportAssetOptions.ForceUpdate);

            var back = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(MaterialPath);
            if (back == null)
                Debug.LogError($"[MujocoBiped] {MaterialPath} did not import - the colliders " +
                               "would ship with no friction at all.");
            return back;
        }

        // ----------------------------------------------------------- build prefab --
        [MenuItem("MujocoBiped/Build Prefab", priority = 1)]
        public static void BuildPrefab()
        {
            var rig = AssetDatabase.LoadAssetAtPath<MujocoBipedRigAsset>(RigAssetPath)
                      ?? BuildRigAssetFromJson();
            if (rig == null) return;

            var mat = CreateOrUpdateMaterial(rig);
            if (mat == null)
            {
                Debug.LogError("[MujocoBiped] aborting Build Prefab: the physics material did " +
                               "not import, so the colliders would ship with no friction.");
                return;
            }

            GameObject root = null;
            try
            {
                root = MujocoBipedRigBuilder.Build(rig, MujocoBipedRigBuilder.ArmatureMode.Exact, mat);

                var agent = root.AddComponent<MujocoBipedAgent>();
                agent.rig = rig;
#if ISAACPORTS_HAS_INFERENCE
                agent.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(OnnxPath);
                if (agent.modelAsset == null)
                    Debug.LogWarning($"[MujocoBiped] {OnnxPath} did not import as a ModelAsset. " +
                                     "Assign it on the prefab by hand.");
#endif
                root.AddComponent<MujocoBipedTargetSampler>();

                Directory.CreateDirectory(Root);
                var prefab = PrefabUtility.SaveAsPrefabAsset(root, PrefabPath, out bool ok);
                if (!ok || prefab == null)
                {
                    Debug.LogError($"[MujocoBiped] failed to save {PrefabPath}");
                    return;
                }

                AssetDatabase.SaveAssets();
                Selection.activeObject = prefab;
                EditorGUIUtility.PingObject(prefab);

                var bodies = root.GetComponentsInChildren<ArticulationBody>(true);
                float mass = 0f;
                foreach (var b in bodies) mass += b.mass;
                Debug.Log($"[MujocoBiped] prefab built -> {PrefabPath}\n" +
                          $"  links {bodies.Length}, colliders " +
                          $"{root.GetComponentsInChildren<Collider>(true).Length}, " +
                          $"total mass {mass:F3} kg\n" +
                          $"  material {mat.staticFriction:F2}/{mat.dynamicFriction:F2} " +
                          $"({mat.frictionCombine} combine)\n" +
                          $"  armature fold: Exact");
            }
            finally
            {
                if (root != null) UnityEngine.Object.DestroyImmediate(root);
            }
        }

        // ------------------------------------------------------- reference check --
        [MenuItem("MujocoBiped/Run Reference Check", priority = 20)]
        public static void RunReferenceCheck()
        {
#if ISAACPORTS_HAS_INFERENCE
            if (!MujocoBipedPaths.TryLoadReference(out var obs, out var acts, out string err))
            {
                Debug.LogError($"[MujocoBiped] {err}");
                return;
            }

            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(OnnxPath);
            if (modelAsset == null)
            {
                Debug.LogError($"[MujocoBiped] {OnnxPath} did not import as a ModelAsset.");
                return;
            }

            var model = ModelLoader.Load(modelAsset);
            using var worker = new Worker(model, BackendType.CPU);
            using var input = new Tensor<float>(new TensorShape(1, obs[0].Length));

            float worst = 0f;
            int worstStep = 0, worstIndex = 0;
            for (int s = 0; s < obs.Length; s++)
            {
                input.Upload(obs[s]);
                worker.Schedule(input);
                float[] got = (worker.PeekOutput() as Tensor<float>).DownloadToArray();
                for (int i = 0; i < acts[s].Length; i++)
                {
                    float d = Mathf.Abs(got[i] - acts[s][i]);
                    if (d > worst) { worst = d; worstStep = s; worstIndex = i; }
                }
            }

            string verdict = worst <= 1e-4f ? "PASS" : "FAIL";
            Debug.Log($"[MujocoBiped] reference check {verdict}: max abs diff {worst:E3} over " +
                      $"{obs.Length} recorded steps (worst at step {worstStep}, action index " +
                      $"{worstIndex}). Tolerance 1e-4; check_onnx.py reports 7.749e-07 for the " +
                      "same data under onnxruntime, so anything materially larger is an " +
                      "Inference Engine difference, not a model one.");
#else
            Debug.LogError("[MujocoBiped] com.unity.ai.inference is not installed.");
#endif
        }

        // ------------------------------------------------------------ spawn window --
        [MenuItem("MujocoBiped/Spawn Into Open Scene", priority = 40)]
        public static void OpenSpawnWindow() => MujocoBipedSpawnWindow.Open();
    }

    /// <summary>
    /// Places the prefab in the open scene. Leaves the scene DIRTY and never saves it -
    /// saving someone's scene is not this tool's call.
    /// </summary>
    public class MujocoBipedSpawnWindow : EditorWindow
    {
        Vector3 _position = new Vector3(-7f, 0.88f, -2f);
        float _yawDegrees;
        Transform _target;
        bool _useSceneFinishLine = true;
        int _count = 1;
        float _spacing = 1.5f;

        public static void Open()
        {
            var w = GetWindow<MujocoBipedSpawnWindow>(true, "Spawn MujocoBiped", true);
            w.minSize = new Vector2(430f, 290f);
            w.Show();
        }

        void OnGUI()
        {
            EditorGUILayout.HelpBox(
                "Spawns MujocoBiped.prefab into the OPEN scene and leaves the scene dirty. " +
                "Nothing is saved for you.\n\n" +
                "The default position is lane 0 of SCN_RACE_FLAT (x = -7) at z = -2, two " +
                "metres behind the start grid, with y = 0.88 - the export's init_qpos " +
                "height, measured to the TORSO ORIGIN, not to the feet.",
                MessageType.Info);

            _position = EditorGUILayout.Vector3Field("Position", _position);
            _yawDegrees = EditorGUILayout.FloatField("Yaw (degrees)", _yawDegrees);

            _count = Mathf.Max(1, EditorGUILayout.IntField("Count", _count));
            if (_count > 1) _spacing = EditorGUILayout.FloatField("Spacing (m, along x)", _spacing);

            EditorGUILayout.Space();
            _useSceneFinishLine = EditorGUILayout.Toggle(
                new GUIContent("Chase scene FinishLine",
                    "Wires the agent's target to a GameObject named FinishLine in the open " +
                    "scene. Off, or if there is none, the prefab's own target sampler drives " +
                    "it instead."),
                _useSceneFinishLine);
            if (!_useSceneFinishLine)
                _target = (Transform)EditorGUILayout.ObjectField("Target", _target,
                    typeof(Transform), true);

            EditorGUILayout.Space();
            if (GUILayout.Button("Spawn", GUILayout.Height(30f))) Spawn();
        }

        void Spawn()
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(MujocoBipedPaths.Prefab);
            if (prefab == null)
            {
                Debug.LogError($"[MujocoBiped] {MujocoBipedPaths.Prefab} not found. Run " +
                               "MujocoBiped > Build Prefab first.");
                return;
            }

            Transform target = _target;
            if (_useSceneFinishLine)
            {
                var found = GameObject.Find("FinishLine");
                if (found != null) target = found.transform;
                else Debug.LogWarning("[MujocoBiped] no GameObject named 'FinishLine' in the " +
                                      "open scene; falling back to the target sampler.");
            }

            for (int i = 0; i < _count; i++)
            {
                var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                go.transform.position = _position + new Vector3(i * _spacing, 0f, 0f);
                go.transform.rotation = Quaternion.Euler(0f, _yawDegrees, 0f);
                if (_count > 1) go.name = $"{prefab.name}_{i}";

                var agent = go.GetComponent<MujocoBipedAgent>();
                if (agent != null && target != null) agent.target = target;

                Undo.RegisterCreatedObjectUndo(go, "Spawn MujocoBiped");
                if (i == 0) Selection.activeGameObject = go;
            }

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Debug.Log($"[MujocoBiped] spawned {_count} at {_position} " +
                      $"(target: {(target != null ? target.name : "sampler")}). The scene is " +
                      "dirty and has NOT been saved.");
        }
    }
}
