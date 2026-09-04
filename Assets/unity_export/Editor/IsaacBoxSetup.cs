using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

using IsaacBox;
using PoRacer.IsaacPorts;

namespace IsaacBox.EditorTools
{
    /// <summary>
    /// Authoring tools for the IsaacBox creature. Everything this writes lands inside
    /// Assets/unity_export/IsaacBox/ - it never edits a project setting, a layer, a build
    /// setting, another agent, or another scene's assets.
    ///
    /// Entry points. These carried [MenuItem] attributes until 2026-09-03; the editor UI
    /// was removed so that all authoring goes through MCP / the Unity CLI. Call them with
    ///   unity command eval --code "IsaacBox.EditorTools.IsaacBoxSetup.RebuildRigAsset()"
    /// NOTE the CLI's eval has a 5 s main-thread budget, so anything long-running (a
    /// player build) has to go through the async build/build_status pair instead.
    ///
    ///   RebuildRigAsset()          - isaacbox_rig.json -> IsaacBoxRig.asset
    ///   BuildPrefab()              - rig asset + IsaacBox_Character.fbx -> IsaacBox.prefab + material
    ///   SpawnIntoOpenSceneDefaults() - spawns, leaves the scene dirty
    ///   RunReferenceCheckMenu()    - edit-mode ONNX check vs the recording
    ///   OpenSpawnWindow()          - the interactive spawn window; needs a human at the editor
    /// </summary>
    public static class IsaacBoxSetup
    {
        public const string Root = IsaacBoxPaths.Root;

        // ------------------------------------------------------------ rig asset --
        public static void RebuildRigAsset()
        {
            var asset = BuildRigAssetFromJson();
            if (asset == null) return;
            Selection.activeObject = asset;
            EditorGUIUtility.PingObject(asset);
            Debug.Log($"[IsaacBox] rig asset rebuilt: {asset.bodies.Length} bodies, {asset.jointOrder.Length} joints, " +
                      $"{CountColliders(asset)} collision shapes, {asset.totalMass:F1} kg -> {IsaacBoxPaths.RigAsset}");
        }

        static int CountColliders(IsaacBoxRigAsset a)
        {
            int n = 0;
            foreach (var b in a.bodies) n += b.colliders?.Length ?? 0;
            return n;
        }

        public static IsaacBoxRigAsset BuildRigAssetFromJson()
        {
            if (!File.Exists(IsaacBoxPaths.RigJson))
            {
                Debug.LogError($"[IsaacBox] {IsaacBoxPaths.RigJson} not found. Run ISAAC/boy_rig/build_boy_rig.py first.");
                return null;
            }

            var raw = MiniJson.Parse(File.ReadAllText(IsaacBoxPaths.RigJson)) as Dictionary<string, object>;
            if (raw == null) { Debug.LogError("[IsaacBox] rig JSON did not parse to an object."); return null; }

            var asset = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset);
            bool isNew = asset == null;
            if (isNew) asset = ScriptableObject.CreateInstance<IsaacBoxRigAsset>();

            asset.sourceTask = MiniJson.Str(raw, "sourceTask");
            asset.trainTask = MiniJson.Str(raw, "trainTask");
            asset.sourceModel = MiniJson.Str(raw, "sourceModel");
            asset.checkpoint = raw.ContainsKey("checkpoint") ? MiniJson.Str(raw, "checkpoint") : "";
            asset.jointOrder = MiniJson.StrArray(raw, "jointOrder");
            asset.bodyOrder = MiniJson.StrArray(raw, "bodyOrder");
            asset.skinBones = MiniJson.StrArray(raw, "skinBones");
            asset.obsDim = (int)MiniJson.Num(raw, "obsDim");
            asset.actDim = (int)MiniJson.Num(raw, "actDim");
            asset.actionScale = MiniJson.Num(raw, "actionScale");
            asset.useDefaultOffset = MiniJson.Bool(raw, "useDefaultOffset");
            asset.totalMass = MiniJson.Num(raw, "totalMass");

            var timing = MiniJson.Obj(raw, "timing");
            asset.policyDt = MiniJson.Num(timing, "policyDt");
            asset.isaacPhysicsDt = MiniJson.Num(timing, "isaacPhysicsDt");
            asset.isaacDecimation = (int)MiniJson.Num(timing, "isaacDecimation");
            asset.episodeLengthS = MiniJson.Num(timing, "episodeLengthS");

            var spawn = MiniJson.Obj(raw, "spawn");
            asset.spawnPosIsaac = MiniJson.Vec3(spawn, "posIsaac");
            asset.hipsHeightAtDefaultPoseRest = MiniJson.Num(spawn, "hipsHeightAtDefaultPoseRest");
            asset.hipsHeightAtZeroPoseRest = MiniJson.Num(spawn, "hipsHeightAtZeroPoseRest");

            var chase = MiniJson.Obj(raw, "chase");
            var radius = MiniJson.FloatArray(chase, "targetRadiusRange");
            var resample = MiniJson.FloatArray(chase, "resampleRangeS");
            asset.chase = new BoyChaseDef
            {
                targetObsClip = MiniJson.Num(chase, "targetObsClip"),
                targetRadiusMin = radius[0],
                targetRadiusMax = radius[1],
                reachRadius = MiniJson.Num(chase, "reachRadius"),
                resampleSecondsMin = resample[0],
                resampleSecondsMax = resample[1],
                targetSpeed = MiniJson.Num(chase, "targetSpeed"),
            };

            var ph = MiniJson.Obj(raw, "physics");
            asset.physics = new BoyPhysicsDef
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

            var ev = MiniJson.Obj(raw, "eval");
            asset.isaacMeanSpeed = MiniJson.Num(ev, "meanSpeed");
            asset.isaacMeanSpeedTowardTarget = ev.ContainsKey("meanSpeedTowardTarget") ? MiniJson.Num(ev, "meanSpeedTowardTarget") : 0f;
            asset.isaacFallsPerRobotPerMinute = MiniJson.Num(ev, "fallsPerRobotPerMinute");
            asset.isaacTargetsReachedPerMinute = MiniJson.Num(ev, "targetsReachedPerMinute");
            asset.isaacReferenceForwardSpeed = ev.ContainsKey("referenceForwardSpeed") ? MiniJson.Num(ev, "referenceForwardSpeed") : 0f;
            asset.isaacReferenceTargetDistance = ev.ContainsKey("referenceTargetDistance") ? MiniJson.Num(ev, "referenceTargetDistance") : 8f;

            var bodies = MiniJson.Arr(raw, "bodies");
            asset.bodies = new BoyBodyDef[bodies.Count];
            for (int i = 0; i < bodies.Count; i++)
            {
                var b = bodies[i] as Dictionary<string, object>;
                var def = new BoyBodyDef
                {
                    name = MiniJson.Str(b, "name"),
                    parent = MiniJson.Str(b, "parent"),
                    isRoot = MiniJson.Bool(b, "isRoot"),
                    boneName = MiniJson.Str(b, "boneName") ?? "",
                    mass = MiniJson.Num(b, "mass"),
                    comIsaac = MiniJson.Vec3(b, "com"),
                    inertiaDiagIsaac = MiniJson.Vec3(b, "inertiaDiag"),
                    worldPosIsaac = MiniJson.Vec3(b, "worldPos"),
                    localPosIsaac = MiniJson.Vec3(b, "localPos"),
                    localRotIsaacWxyz = MiniJson.Vec4(b, "localRotWxyz"),
                };

                var j = MiniJson.Obj(b, "joint");
                if (j != null)
                {
                    def.hasJoint = true;
                    def.joint = new BoyJointDef
                    {
                        name = MiniJson.Str(j, "name"),
                        index = (int)MiniJson.Num(j, "index"),
                        family = MiniJson.Str(j, "family"),
                        axisInChildIsaac = MiniJson.Vec3(j, "axisInChild"),
                        lowerRad = MiniJson.Num(j, "lowerRad"),
                        upperRad = MiniJson.Num(j, "upperRad"),
                        stiffness = MiniJson.Num(j, "stiffness"),
                        damping = MiniJson.Num(j, "damping"),
                        effortLimit = MiniJson.Num(j, "effortLimit"),
                        defaultPosRad = MiniJson.Num(j, "defaultPosRad"),
                        armature = MiniJson.Num(j, "armature"),
                    };
                }

                var cols = MiniJson.Arr(b, "colliders");
                def.colliders = new BoyColliderDef[cols?.Count ?? 0];
                for (int c = 0; c < def.colliders.Length; c++)
                {
                    var cd = cols[c] as Dictionary<string, object>;
                    var col = new BoyColliderDef
                    {
                        kind = MiniJson.Str(cd, "kind"),
                        centerIsaac = MiniJson.Vec3(cd, "center"),
                    };
                    if (col.kind == "box") col.sizeIsaac = MiniJson.Vec3(cd, "size");
                    if (col.kind == "sphere" || col.kind == "capsule") col.radius = MiniJson.Num(cd, "radius");
                    if (col.kind == "capsule")
                    {
                        col.height = MiniJson.Num(cd, "height");
                        col.axis = MiniJson.Str(cd, "axis");
                    }
                    def.colliders[c] = col;
                }

                asset.bodies[i] = def;
            }

            if (isNew) AssetDatabase.CreateAsset(asset, IsaacBoxPaths.RigAsset);
            else EditorUtility.SetDirty(asset);
            AssetDatabase.SaveAssets();
            return asset;
        }

        // -------------------------------------------------------- physics material --
        /// <summary>
        /// The creature's own material: the Play task's fixed 0.8/0.6. Unity ships Minimum
        /// combine, which reproduces Isaac's multiply against a 1.0/1.0 ground exactly and
        /// never exceeds the Isaac pair value against a slipperier one (see the H1 port).
        /// </summary>
        public static PhysicsMaterial CreateOrUpdateMaterial(IsaacBoxRigAsset rig)
        {
            var mat = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacBoxPaths.Material);
            bool isNew = mat == null;
            if (isNew) mat = new PhysicsMaterial("PM_IsaacBox");

            mat.staticFriction = rig.physics.robotStaticFriction;
            mat.dynamicFriction = rig.physics.robotDynamicFriction;
            mat.bounciness = rig.physics.robotRestitution;
            mat.frictionCombine = PhysicsMaterialCombine.Minimum;
            mat.bounceCombine = PhysicsMaterialCombine.Minimum;

            Directory.CreateDirectory(Root);
            if (isNew) AssetDatabase.CreateAsset(mat, IsaacBoxPaths.Material);
            else EditorUtility.SetDirty(mat);
            AssetDatabase.SaveAssets();

            var back = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacBoxPaths.Material);
            if (back == null)
                Debug.LogError($"[IsaacBox] {IsaacBoxPaths.Material} did not import - colliders would ship with no friction.");
            return back;
        }

        // ----------------------------------------------------------- build prefab --
        public static void BuildPrefab() => BuildPrefab(true);

        /// <summary>Builds IsaacBox.prefab. Returns the saved prefab asset, or null.</summary>
        public static GameObject BuildPrefab(bool withSkin)
        {
            var rig = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset) ?? BuildRigAssetFromJson();
            if (rig == null) return null;

            var mat = CreateOrUpdateMaterial(rig);
            if (mat == null)
            {
                Debug.LogError("[IsaacBox] aborting Build Prefab: the physics material did not import.");
                return null;
            }

            GameObject fbx = null;
            if (withSkin)
            {
                fbx = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBoxPaths.Fbx);
                if (fbx == null)
                    Debug.LogWarning($"[IsaacBox] {IsaacBoxPaths.Fbx} not found; building the physics rig without its skin.");
            }

            GameObject root = null;
            try
            {
                root = IsaacBoxRigBuilder.Build(rig, fbx, out string report);

                foreach (var c in root.GetComponentsInChildren<Collider>(true))
                    c.sharedMaterial = mat;

                // AddComponent runs Awake IMMEDIATELY, while `rig` is still null - so the agent
                // hits its own "no rig asset; disabling" guard and sets enabled = false. Assigning
                // the rig on the next line does not undo that, and SaveAsPrefabAsset then bakes a
                // DISABLED agent into the prefab. A disabled agent skips ApplyPerBodyOverrides
                // (Awake returns before it) and never reaches Start, so decimation stays at its
                // field initializer of 1 and the policy never runs: the creature ships as a
                // ragdoll that fails ten rungs of the ladder with no error at test time.
                // Re-enable explicitly AFTER the rig is set, and keep it the last word.
                var agent = root.AddComponent<IsaacBoxAgent>();
                agent.rig = rig;
                agent.enabled = true;
#if ISAACPORTS_HAS_INFERENCE
                agent.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(IsaacBoxPaths.Onnx);
                if (agent.modelAsset == null)
                    Debug.LogWarning($"[IsaacBox] {IsaacBoxPaths.Onnx} is not present yet (run ISAAC/scripts/export_bundle.py). " +
                                     "The prefab holds the default pose until a brain is assigned.");
#endif
                var sampler = root.AddComponent<IsaacBoxTargetSampler>();
                sampler.ConfigureFrom(rig);

                // The FBX has no textures at all, so its imported materials render flat. Put the
                // authored materials (built from the GLB twin's images) on before saving, or every
                // prefab rebuild silently reverts the creature to untextured.
                int textured = IsaacBoxMaterials.Apply(root);

                Directory.CreateDirectory(Root);
                var prefab = PrefabUtility.SaveAsPrefabAsset(root, IsaacBoxPaths.Prefab, out bool ok);
                if (!ok || prefab == null)
                {
                    Debug.LogError($"[IsaacBox] failed to save {IsaacBoxPaths.Prefab}");
                    return null;
                }

                AssetDatabase.SaveAssets();
                Selection.activeObject = prefab;
                EditorGUIUtility.PingObject(prefab);
                Debug.Log($"[IsaacBox] prefab built -> {IsaacBoxPaths.Prefab}\n{report}" +
                          $"  materials: {textured} renderers textured from the GLB\n" + 
                          $"  bodies {root.GetComponentsInChildren<ArticulationBody>(true).Length}, " +
                          $"colliders {root.GetComponentsInChildren<Collider>(true).Length}, " +
                          $"material {mat.staticFriction:F2}/{mat.dynamicFriction:F2} ({mat.frictionCombine} combine)");
                return prefab;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[IsaacBox] Build Prefab failed: {e.Message}");
                return null;
            }
            finally
            {
                if (root != null) UnityEngine.Object.DestroyImmediate(root);
            }
        }

        // --------------------------------------------------------------- spawning --
        internal static void EnsureSceneEssentials(Scene scene, IsaacBoxRigAsset rig, List<string> createdLog)
        {
            bool hasGround = false, hasLight = false, hasCamera = false;
            foreach (var go in scene.GetRootGameObjects())
            {
                foreach (var c in go.GetComponentsInChildren<Collider>(true))
                {
                    if (c.GetComponentInParent<IsaacBoxAgent>() != null) continue;
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
                ground.name = "IsaacBox_Ground";
                ground.transform.localScale = new Vector3(20f, 1f, 20f);
                ground.isStatic = true;
                var gm = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacBoxPaths.Material);
                if (gm != null) ground.GetComponent<Collider>().sharedMaterial = gm;
                ground.GetComponent<Collider>().contactOffset = rig.physics.contactOffset;
                createdLog.Add("IsaacBox_Ground (200x200 m plane, PM_IsaacBox)");
            }

            if (!hasLight)
            {
                var lightGo = new GameObject("IsaacBox_Light");
                var l = lightGo.AddComponent<Light>();
                l.type = LightType.Directional;
                l.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
                createdLog.Add("IsaacBox_Light (directional)");
            }

            if (!hasCamera)
            {
                var camGo = new GameObject("IsaacBox_Camera");
                camGo.AddComponent<Camera>();
                camGo.transform.position = new Vector3(3f, 2f, -3f);
                camGo.transform.LookAt(new Vector3(0f, 0.8f, 0f));
                createdLog.Add("IsaacBox_Camera");
            }
        }

        public static void OpenSpawnWindow() => IsaacBoxSpawnWindow.Open();

        /// <summary>
        /// Window-free twin of "Spawn Into Open Scene", carrying the same defaults the window
        /// ships with. The menu item above opens an EditorWindow, and a modal window stalls the
        /// `unity command menu` bridge, so automated setup needs an entry point that just runs.
        /// </summary>
        public static void SpawnIntoOpenSceneDefaults()
        {
            var rig = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset);
            float y = rig != null ? rig.spawnPosIsaac.z : 0.764f;
            SpawnIntoOpenScene(new Vector3(0f, y, -2f), null, true);
        }

        public static GameObject SpawnIntoOpenScene(Vector3 position, Transform target, bool createEssentials)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBoxPaths.Prefab);
            if (prefab == null)
            {
                Debug.LogError($"[IsaacBox] {IsaacBoxPaths.Prefab} not found. Run IsaacBoxSetup.BuildPrefab().");
                return null;
            }

            var scene = SceneManager.GetActiveScene();
            if (!scene.IsValid())
            {
                Debug.LogError("[IsaacBox] no valid open scene to spawn into.");
                return null;
            }

            var rig = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset);
            var created = new List<string>();
            if (createEssentials && rig != null) EnsureSceneEssentials(scene, rig, created);

            var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            go.transform.position = position;
            go.transform.rotation = Quaternion.identity;

            var agent = go.GetComponent<IsaacBoxAgent>();
            if (agent != null && target != null) agent.target = target;

            Undo.RegisterCreatedObjectUndo(go, "Spawn IsaacBox");
            Selection.activeGameObject = go;
            EditorSceneManager.MarkSceneDirty(scene);

            Debug.Log($"[IsaacBox] spawned '{go.name}' at {position} in '{scene.name}'. " +
                      (created.Count > 0 ? "Created: " + string.Join(", ", created) + ". " : "") +
                      (target != null ? $"Target: {target.name}. " : "Target: ring sampler fallback. ") +
                      "The scene is left DIRTY and has not been saved.");
            return go;
        }

        // -------------------------------------------------------- reference check --
        public static void RunReferenceCheckMenu()
        {
#if ISAACPORTS_HAS_INFERENCE
            if (!IsaacBoxPaths.TryLoadReference(out var obs, out var acts, out string err))
            {
                Debug.LogError($"[IsaacBox] {err}");
                return;
            }

            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(IsaacBoxPaths.Onnx);
            if (modelAsset == null) { Debug.LogError($"[IsaacBox] {IsaacBoxPaths.Onnx} is not a ModelAsset."); return; }

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
                Debug.Log($"[IsaacBox] reference check (edit mode, BackendType.CPU): {verdict}\n" +
                          $"  {obs.Length} recorded observations through IsaacBox.onnx\n" +
                          $"  max abs diff {worst:E3} (step {worstStep}, action {worstIdx}), gate 1e-4");
            }
            finally
            {
                input.Dispose();
                worker.Dispose();
            }
#else
            Debug.LogError("[IsaacBox] com.unity.ai.inference is not installed.");
#endif
        }
    }

    /// <summary>Spawn window. Position, height, and target = the current selection.</summary>
    public class IsaacBoxSpawnWindow : EditorWindow
    {
        Vector3 _position = new Vector3(0f, 0.764f, -2f);
        bool _useExportHeight = true;
        Transform _target;
        bool _createEssentials = true;

        public static void Open()
        {
            var w = GetWindow<IsaacBoxSpawnWindow>(true, "Spawn IsaacBox", true);
            w.minSize = new Vector2(380f, 250f);
            w.PickSelectionAsTarget();
            w.Show();
        }

        void PickSelectionAsTarget()
        {
            if (Selection.activeTransform != null &&
                Selection.activeTransform.GetComponentInParent<IsaacBoxAgent>() == null)
                _target = Selection.activeTransform;
        }

        void OnGUI()
        {
            var rig = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset);
            float spawnHeight = rig != null ? rig.spawnPosIsaac.z : 0.764f;

            EditorGUILayout.HelpBox(
                "Spawns the IsaacBox prefab into the scene that is already open. The scene is left " +
                "dirty and is never saved for you. A ground, light or camera is created only if " +
                "the scene has none.", MessageType.Info);

            _useExportHeight = EditorGUILayout.Toggle(
                new GUIContent("Use export spawn height", $"Isaac init_pos z = {spawnHeight:F3} m"), _useExportHeight);

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
                    IsaacBoxSetup.SpawnIntoOpenScene(_position, _target, _createEssentials);
                    Close();
                }
            }
        }
    }
}
