using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

using PoRacer.IsaacPorts;

namespace IsaacH1.EditorTools
{
    /// <summary>
    /// Authoring tools for the IsaacH1 creature. Everything this writes lands inside
    /// Assets/unity_export/IsaacH1/ - it never edits a project setting, a layer, a build
    /// setting, another agent, or another scene's assets.
    ///
    /// Entry points. These carried [MenuItem] attributes until 2026-09-03; the editor UI
    /// was removed so that all authoring goes through MCP / the Unity CLI. Call them with
    ///   unity command eval --code "IsaacH1.EditorTools.IsaacH1Setup.RebuildRigAsset()"
    /// NOTE the CLI's eval has a 5 s main-thread budget, so anything long-running (a
    /// player build) has to go through the async build/build_status pair instead.
    ///
    ///   RebuildRigAsset()       - IsaacH1_rig.json -> IsaacH1Rig.asset
    ///   BuildPrefab()           - rig asset -> IsaacH1.prefab + material
    ///   RunReferenceCheckMenu() - edit-mode ONNX check vs the recording
    ///   OpenSpawnWindow()       - the interactive spawn window; needs a human at the editor
    /// </summary>
    public static class IsaacH1Setup
    {
        // Paths and the reference loaders live in the RUNTIME assembly (IsaacH1Paths)
        // so the PlayMode test assembly can reach them without being Editor-only.
        public const string Root = IsaacH1Paths.Root;
        public const string RigJsonPath = IsaacH1Paths.RigJson;
        public const string RigAssetPath = IsaacH1Paths.RigAsset;
        public const string PrefabPath = IsaacH1Paths.Prefab;
        public const string MaterialPath = IsaacH1Paths.Material;
        public const string OnnxPath = IsaacH1Paths.Onnx;
        public const string ReferencePath = IsaacH1Paths.Reference;

        // ------------------------------------------------------------ rig asset --
        public static void RebuildRigAsset()
        {
            var asset = BuildRigAssetFromJson();
            if (asset == null) return;
            Selection.activeObject = asset;
            EditorGUIUtility.PingObject(asset);
            Debug.Log($"[IsaacH1] rig asset rebuilt: {asset.bodies.Length} bodies, " +
                      $"{asset.jointOrder.Length} joints, " +
                      $"{CountColliders(asset)} collision shapes -> {RigAssetPath}");
        }

        static int CountColliders(IsaacH1RigAsset a)
        {
            int n = 0;
            foreach (var b in a.bodies) n += b.colliders?.Length ?? 0;
            return n;
        }

        public static IsaacH1RigAsset BuildRigAssetFromJson()
        {
            if (!File.Exists(RigJsonPath))
            {
                Debug.LogError($"[IsaacH1] {RigJsonPath} not found. Run extract_rig.py first.");
                return null;
            }

            var raw = MiniJson.Parse(File.ReadAllText(RigJsonPath)) as Dictionary<string, object>;
            if (raw == null) { Debug.LogError("[IsaacH1] rig JSON did not parse to an object."); return null; }

            var asset = AssetDatabase.LoadAssetAtPath<IsaacH1RigAsset>(RigAssetPath);
            bool isNew = asset == null;
            if (isNew) asset = ScriptableObject.CreateInstance<IsaacH1RigAsset>();

            asset.sourceTask = MiniJson.Str(raw, "sourceTask");
            asset.jointOrder = MiniJson.StrArray(raw, "jointOrder");
            asset.bodyOrder = MiniJson.StrArray(raw, "bodyOrder");
            asset.obsDim = (int)MiniJson.Num(raw, "obsDim");
            asset.actDim = (int)MiniJson.Num(raw, "actDim");
            asset.actionScale = MiniJson.Num(raw, "actionScale");
            asset.useDefaultOffset = MiniJson.Bool(raw, "useDefaultOffset");

            var timing = MiniJson.Obj(raw, "timing");
            asset.policyDt = MiniJson.Num(timing, "policyDt");
            asset.isaacPhysicsDt = MiniJson.Num(timing, "isaacPhysicsDt");
            asset.isaacDecimation = (int)MiniJson.Num(timing, "isaacDecimation");

            var spawn = MiniJson.Obj(raw, "spawn");
            asset.spawnPosIsaac = MiniJson.Vec3(spawn, "posIsaac");

            var ph = MiniJson.Obj(raw, "physics");
            var p = new IsaacH1PhysicsDef
            {
                gravityIsaac = MiniJson.Vec3(ph, "gravity"),
                groundStaticFriction = MiniJson.Num(ph, "groundStaticFriction"),
                groundDynamicFriction = MiniJson.Num(ph, "groundDynamicFriction"),
                groundRestitution = MiniJson.Num(ph, "groundRestitution"),
                robotStaticFriction = MiniJson.Num(ph, "robotStaticFriction"),
                robotDynamicFriction = MiniJson.Num(ph, "robotDynamicFriction"),
                robotRestitution = MiniJson.Num(ph, "robotRestitution"),
                frictionCombineMode = MiniJson.Str(ph, "frictionCombineMode"),
                maxLinearVelocity = MiniJson.Num(ph, "maxLinearVelocity"),
                maxAngularVelocity = MiniJson.Num(ph, "maxAngularVelocity"),
                maxDepenetrationVelocity = MiniJson.Num(ph, "maxDepenetrationVelocity"),
                linearDamping = MiniJson.Num(ph, "linearDamping"),
                angularDamping = MiniJson.Num(ph, "angularDamping"),
                jointFriction = MiniJson.Num(ph, "jointFriction"),
                solverPositionIterations = (int)MiniJson.Num(ph, "solverPositionIterations"),
                solverVelocityIterations = (int)MiniJson.Num(ph, "solverVelocityIterations"),
                enabledSelfCollisions = MiniJson.Bool(ph, "enabledSelfCollisions"),
                contactOffset = MiniJson.Num(ph, "contactOffset"),
                restOffset = MiniJson.Num(ph, "restOffset"),
                isaacSolverType = MiniJson.Str(ph, "isaacSolverType"),
            };
            asset.physics = p;

            var ev = MiniJson.Obj(raw, "eval");
            asset.isaacMeanSpeed = MiniJson.Num(ev, "meanSpeed");
            asset.isaacMeanLinVelTrackingError = MiniJson.Num(ev, "meanLinVelTrackingError");
            asset.isaacFallsPerRobotPerMinute = MiniJson.Num(ev, "fallsPerRobotPerMinute");
            asset.referenceCommand = MiniJson.Vec3(ev, "referenceCommand");

            var dev = MiniJson.Obj(raw, "deviationsFromUrdf");
            asset.torsoMassNominal = MiniJson.Num(dev, "torsoMassNominalUsd");
            asset.torsoMassInReferenceRecording = MiniJson.Num(dev, "torsoMassInReferenceRecording");

            var bodies = MiniJson.Arr(raw, "bodies");
            asset.bodies = new IsaacH1BodyDef[bodies.Count];
            for (int i = 0; i < bodies.Count; i++)
            {
                var b = bodies[i] as Dictionary<string, object>;
                var def = new IsaacH1BodyDef
                {
                    name = MiniJson.Str(b, "name"),
                    parent = MiniJson.Str(b, "parent"),
                    isRoot = MiniJson.Bool(b, "isRoot"),
                    mass = MiniJson.Num(b, "mass"),
                    comIsaac = MiniJson.Vec3(b, "com"),
                    inertiaDiagIsaac = MiniJson.Vec3(b, "inertiaDiag"),
                    urdfMass = MiniJson.Num(b, "urdfMass"),
                    urdfInertiaDiagIsaac = MiniJson.Vec3(b, "urdfInertiaDiag"),
                    localPosIsaac = MiniJson.Vec3(b, "localPos"),
                    localRotIsaacWxyz = MiniJson.Vec4(b, "localRotWxyz"),
                };

                var j = MiniJson.Obj(b, "joint");
                if (j != null)
                {
                    def.hasJoint = true;
                    def.joint = new IsaacH1JointDef
                    {
                        name = MiniJson.Str(j, "name"),
                        index = (int)MiniJson.Num(j, "index"),
                        axisInChildIsaac = MiniJson.Vec3(j, "axisInChild"),
                        lowerRad = MiniJson.Num(j, "lowerRad"),
                        upperRad = MiniJson.Num(j, "upperRad"),
                        stiffness = MiniJson.Num(j, "stiffness"),
                        damping = MiniJson.Num(j, "damping"),
                        effortLimit = MiniJson.Num(j, "effortLimit"),
                        defaultPosRad = MiniJson.Num(j, "defaultPosRad"),
                        armature = MiniJson.Num(j, "armature"),
                        urdfVelocityLimit = MiniJson.Num(j, "urdfVelocityLimit"),
                        urdfEffortLimit = MiniJson.Num(j, "urdfEffortLimit"),
                    };
                }

                var cols = MiniJson.Arr(b, "colliders");
                def.colliders = new IsaacH1ColliderDef[cols?.Count ?? 0];
                for (int c = 0; c < def.colliders.Length; c++)
                {
                    var cd = cols[c] as Dictionary<string, object>;
                    def.colliders[c] = new IsaacH1ColliderDef
                    {
                        centerIsaac = MiniJson.Vec3(cd, "center"),
                        sizeIsaac = MiniJson.Vec3(cd, "size"),
                        sourceApproximation = MiniJson.Str(cd, "sourceApproximation"),
                        sourceVertexCount = (int)MiniJson.Num(cd, "sourceVertexCount"),
                    };
                }

                var vis = MiniJson.Arr(b, "visualProxies");
                def.visuals = new IsaacH1VisualDef[vis?.Count ?? 0];
                for (int c = 0; c < def.visuals.Length; c++)
                {
                    var vd = vis[c] as Dictionary<string, object>;
                    def.visuals[c] = new IsaacH1VisualDef
                    {
                        kind = MiniJson.Str(vd, "kind"),
                        originIsaac = MiniJson.Vec3(vd, "origin"),
                        rpy = MiniJson.Vec3(vd, "rpy"),
                        size = MiniJson.Vec3(vd, "size"),
                        radius = MiniJson.Num(vd, "radius"),
                        length = MiniJson.Num(vd, "length"),
                    };
                }

                asset.bodies[i] = def;
            }

            if (isNew) AssetDatabase.CreateAsset(asset, RigAssetPath);
            else EditorUtility.SetDirty(asset);
            AssetDatabase.SaveAssets();
            return asset;
        }

        // -------------------------------------------------------- physics material --
        /// <summary>
        /// The creature's own material. Isaac's startup physics_material event set the
        /// ROBOT's shapes to 0.8 static / 0.6 dynamic (a degenerate range, so a fixed
        /// value); the ground stayed at 1.0/1.0 and PhysX combined them by multiply,
        /// giving pair friction 0.8/0.6.
        ///
        /// Unity ships Minimum combine. With mu_creature = 0.8/0.6, Minimum reproduces
        /// multiply exactly against any ground with mu >= the creature's - including the
        /// 1.0/1.0 ground this tool creates - and degrades gracefully (never ABOVE the
        /// Isaac pair value) against a slipperier existing ground. In Unity 6 the combine
        /// enum is Average=0, Multiply=1, Minimum=2, Maximum=3 and the HIGHER value wins a
        /// mismatched pair, so Minimum also beats a scene material asking for Average or
        /// Multiply. Only a Maximum material overrides it - noted in CONTRACT.md.
        ///
        /// </summary>
        public static PhysicsMaterial CreateOrUpdateMaterial(IsaacH1RigAsset rig)
        {
            var mat = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(MaterialPath);
            bool isNew = mat == null;
            if (isNew) mat = new PhysicsMaterial("PM_IsaacH1");

            mat.staticFriction = rig.physics.robotStaticFriction;
            mat.dynamicFriction = rig.physics.robotDynamicFriction;
            mat.bounciness = rig.physics.robotRestitution;
            mat.frictionCombine = PhysicsMaterialCombine.Minimum;
            mat.bounceCombine = PhysicsMaterialCombine.Minimum;

            Directory.CreateDirectory(Root);
            if (isNew) AssetDatabase.CreateAsset(mat, MaterialPath);
            else EditorUtility.SetDirty(mat);
            AssetDatabase.SaveAssets();

            var back = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(MaterialPath);
            if (back == null)
                Debug.LogError($"[IsaacH1] {MaterialPath} did not import - colliders would " +
                               "ship with no friction.");
            return back;
        }

        // ----------------------------------------------------------- build prefab --
        public static void BuildPrefab()
        {
            var rig = AssetDatabase.LoadAssetAtPath<IsaacH1RigAsset>(RigAssetPath)
                      ?? BuildRigAssetFromJson();
            if (rig == null) return;

            var mat = CreateOrUpdateMaterial(rig);
            if (mat == null)
            {
                Debug.LogError("[IsaacH1] aborting Build Prefab: the physics material did " +
                               "not import, so colliders would ship with no friction.");
                return;
            }

            GameObject root = null;
            try
            {
                var meshLib = AssetDatabase.LoadAssetAtPath<IsaacH1MeshLibrary>(
                    IsaacH1MeshImporter.LibraryPath);
                if (meshLib == null)
                    Debug.Log("[IsaacH1] no mesh library found - building with primitive " +
                              "visual proxies. Run: python extract_meshes.py, then " +
                              "IsaacH1MeshImporter.ImportMeshes(), for the real Isaac geometry.");

                root = IsaacH1RigBuilder.Build(rig, IsaacH1Agent.ArmatureMode.None, meshLib);

                foreach (var c in root.GetComponentsInChildren<Collider>(true))
                    c.sharedMaterial = mat;

                var agent = root.AddComponent<IsaacH1Agent>();
                agent.rig = rig;
#if ISAACPORTS_HAS_INFERENCE
                agent.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(OnnxPath);
                if (agent.modelAsset == null)
                    Debug.LogWarning($"[IsaacH1] {OnnxPath} did not import as a ModelAsset. " +
                                     "Assign it on the prefab by hand.");
#endif
                root.AddComponent<IsaacH1RingTargetSampler>();

                Directory.CreateDirectory(Root);
                var prefab = PrefabUtility.SaveAsPrefabAsset(root, PrefabPath, out bool ok);
                if (!ok || prefab == null)
                {
                    Debug.LogError($"[IsaacH1] failed to save {PrefabPath}");
                    return;
                }

                AssetDatabase.SaveAssets();
                Selection.activeObject = prefab;
                EditorGUIUtility.PingObject(prefab);
                Debug.Log($"[IsaacH1] prefab built -> {PrefabPath}\n" +
                          $"  bodies {root.GetComponentsInChildren<ArticulationBody>(true).Length}, " +
                          $"colliders {root.GetComponentsInChildren<Collider>(true).Length}, " +
                          $"material {mat.staticFriction:F2}/{mat.dynamicFriction:F2} " +
                          $"({mat.frictionCombine} combine)\n" +
                          $"  visuals: {(meshLib != null ? $"ORIGINAL Isaac meshes ({meshLib.totalVertices:N0} verts, {meshLib.totalTriangles:N0} tris)" : "primitive proxies")}");
            }
            finally
            {
                if (root != null) UnityEngine.Object.DestroyImmediate(root);
            }
        }

        // --------------------------------------------------------------- spawning --
        /// <summary>Builds the ground/light/camera the scene is missing - and nothing else.</summary>
        internal static void EnsureSceneEssentials(Scene scene, IsaacH1RigAsset rig,
                                                   List<string> createdLog)
        {
            bool hasGround = false, hasLight = false, hasCamera = false;
            foreach (var go in scene.GetRootGameObjects())
            {
                foreach (var c in go.GetComponentsInChildren<Collider>(true))
                {
                    // any large, roughly horizontal static collider counts as ground
                    if (c.GetComponentInParent<IsaacH1Agent>() != null) continue;
                    var b = c.bounds;
                    if (b.size.x > 2f && b.size.z > 2f) { hasGround = true; break; }
                }
                if (go.GetComponentInChildren<Terrain>(true) != null) hasGround = true;
                if (go.GetComponentInChildren<Light>(true) != null) hasLight = true;
                if (go.GetComponentInChildren<Camera>(true) != null) hasCamera = true;
            }

            if (!hasGround)
            {
                var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
                ground.name = "IsaacH1_Ground";
                ground.transform.localScale = new Vector3(20f, 1f, 20f); // 200 x 200 m
                ground.isStatic = true;
                // Same material as the creature so Minimum combine reproduces Isaac's
                // multiply of 0.8*1.0 / 0.6*1.0 exactly.
                var gm = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(MaterialPath);
                if (gm != null) ground.GetComponent<Collider>().sharedMaterial = gm;
                ground.GetComponent<Collider>().contactOffset = rig.physics.contactOffset;
                createdLog.Add("IsaacH1_Ground (200x200 m plane, PM_IsaacH1, contactOffset "
                               + rig.physics.contactOffset.ToString("F3") + ")");
            }

            if (!hasLight)
            {
                var lightGo = new GameObject("IsaacH1_Light");
                var l = lightGo.AddComponent<Light>();
                l.type = LightType.Directional;
                l.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
                createdLog.Add("IsaacH1_Light (directional)");
            }

            if (!hasCamera)
            {
                var camGo = new GameObject("IsaacH1_Camera");
                camGo.AddComponent<Camera>();
                camGo.transform.position = new Vector3(4f, 2.5f, -4f);
                camGo.transform.LookAt(new Vector3(0f, 1f, 0f));
                createdLog.Add("IsaacH1_Camera");
            }
        }

        public static void OpenSpawnWindow() => IsaacH1SpawnWindow.Open();

        /// <summary>
        /// Spawns into the CURRENTLY OPEN scene and leaves it dirty without saving.
        /// Never calls EditorSceneManager.NewScene.
        /// </summary>
        public static GameObject SpawnIntoOpenScene(Vector3 position, Transform target,
                                                    bool createEssentials)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
            if (prefab == null)
            {
                Debug.LogError($"[IsaacH1] {PrefabPath} not found. Run IsaacH1Setup.BuildPrefab().");
                return null;
            }

            var scene = SceneManager.GetActiveScene();
            if (!scene.IsValid())
            {
                Debug.LogError("[IsaacH1] no valid open scene to spawn into.");
                return null;
            }

            var rig = AssetDatabase.LoadAssetAtPath<IsaacH1RigAsset>(RigAssetPath);
            var created = new List<string>();
            if (createEssentials && rig != null) EnsureSceneEssentials(scene, rig, created);

            var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            go.transform.position = position;
            go.transform.rotation = Quaternion.identity;

            var agent = go.GetComponent<IsaacH1Agent>();
            if (agent != null && target != null) agent.target = target;

            Undo.RegisterCreatedObjectUndo(go, "Spawn IsaacH1");
            Selection.activeGameObject = go;
            EditorSceneManager.MarkSceneDirty(scene);   // dirty on purpose; not saved

            Debug.Log($"[IsaacH1] spawned '{go.name}' at {position} in '{scene.name}'. " +
                      (created.Count > 0 ? "Created: " + string.Join(", ", created) + ". " : "") +
                      (target != null ? $"Target: {target.name}. " : "Target: ring sampler fallback. ") +
                      "The scene is left DIRTY and has not been saved.");
            return go;
        }

        // -------------------------------------------------------- reference check --
        public static void RunReferenceCheckMenu()
        {
#if ISAACPORTS_HAS_INFERENCE
            if (!TryLoadReference(out var obs, out var acts, out string err))
            {
                Debug.LogError($"[IsaacH1] {err}");
                return;
            }

            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(OnnxPath);
            if (modelAsset == null) { Debug.LogError($"[IsaacH1] {OnnxPath} is not a ModelAsset."); return; }

            var model = ModelLoader.Load(modelAsset);
            var worker = new Worker(model, BackendType.CPU);
            var input = new Tensor<float>(new TensorShape(1, obs[0].Length));
            try
            {
                float worst = 0f;
                int worstStep = -1, worstIdx = -1;
                for (int s = 0; s < obs.Length; s++)
                {
                    input.Upload(obs[s]);
                    worker.Schedule(input);
                    var output = worker.PeekOutput() as Tensor<float>;
                    float[] got = output.DownloadToArray();
                    for (int i = 0; i < acts[s].Length; i++)
                    {
                        float d = Mathf.Abs(got[i] - acts[s][i]);
                        if (d > worst) { worst = d; worstStep = s; worstIdx = i; }
                    }
                }
                string verdict = worst < 1e-4f ? "PASS" : "FAIL";
                Debug.Log($"[IsaacH1] reference check (edit mode, BackendType.CPU): {verdict}\n" +
                          $"  {obs.Length} recorded observations through IsaacH1.onnx\n" +
                          $"  max abs diff {worst:E3} (step {worstStep}, action {worstIdx}), " +
                          $"gate 1e-4\n" +
                          $"  check_onnx.py reports 2.384E-006 for the same data under onnxruntime.");
            }
            finally
            {
                input.Dispose();
                worker.Dispose();
            }
#else
            Debug.LogError("[IsaacH1] com.unity.ai.inference is not installed.");
#endif
        }

        /// <summary>Forwards to <see cref="IsaacH1Paths.TryLoadReference"/>.</summary>
        public static bool TryLoadReference(out float[][] obs, out float[][] actions, out string error)
            => IsaacH1Paths.TryLoadReference(out obs, out actions, out error);

    }

    /// <summary>Spawn window. Position, height, and target = the current selection.</summary>
    public class IsaacH1SpawnWindow : EditorWindow
    {
        Vector3 _position = new Vector3(0f, 1.05f, -2f);
        bool _useExportHeight = true;
        Transform _target;
        bool _createEssentials = true;

        public static void Open()
        {
            var w = GetWindow<IsaacH1SpawnWindow>(true, "Spawn IsaacH1", true);
            w.minSize = new Vector2(380f, 250f);
            w.PickSelectionAsTarget();
            w.Show();
        }

        void PickSelectionAsTarget()
        {
            if (Selection.activeTransform != null &&
                Selection.activeTransform.GetComponentInParent<IsaacH1Agent>() == null)
                _target = Selection.activeTransform;
        }

        void OnGUI()
        {
            var rig = AssetDatabase.LoadAssetAtPath<IsaacH1RigAsset>(IsaacH1Setup.RigAssetPath);
            float spawnHeight = rig != null ? rig.spawnPosIsaac.z : 1.05f;

            EditorGUILayout.HelpBox(
                "Spawns the IsaacH1 prefab into the scene that is already open. The scene is " +
                "left dirty and is never saved for you. A ground, light or camera is created " +
                "only if the scene has none.", MessageType.Info);

            _useExportHeight = EditorGUILayout.Toggle(
                new GUIContent("Use export spawn height",
                    $"Isaac init_pos z = {spawnHeight:F3} m"), _useExportHeight);

            _position = EditorGUILayout.Vector3Field("Position", _position);
            if (_useExportHeight) _position.y = spawnHeight;

            _target = (Transform)EditorGUILayout.ObjectField(
                new GUIContent("Target", "Leave empty to use the ring sampler fallback."),
                _target, typeof(Transform), true);

            if (GUILayout.Button("Use current selection as target")) PickSelectionAsTarget();

            _createEssentials = EditorGUILayout.Toggle(
                new GUIContent("Create ground/light/camera if missing"), _createEssentials);

            EditorGUILayout.Space();
            var scene = SceneManager.GetActiveScene();
            EditorGUILayout.LabelField("Open scene", scene.IsValid() ? scene.name : "(none)");
            EditorGUILayout.LabelField("Fixed timestep", Time.fixedDeltaTime.ToString("F5") + " s");
            if (rig != null)
            {
                float ratio = rig.policyDt / Time.fixedDeltaTime;
                EditorGUILayout.LabelField("policy_dt / fixedDeltaTime",
                    $"{ratio:F4}  ->  decimation {Mathf.Max(1, Mathf.RoundToInt(ratio))}" +
                    (Mathf.Abs(ratio - Mathf.Round(ratio)) > 1e-4f ? "  (NOT an integer)" : ""));
            }

            EditorGUILayout.Space();
            using (new EditorGUI.DisabledScope(!scene.IsValid()))
            {
                if (GUILayout.Button("Spawn", GUILayout.Height(30f)))
                {
                    IsaacH1Setup.SpawnIntoOpenScene(_position, _target, _createEssentials);
                    Close();
                }
            }
        }
    }
}
