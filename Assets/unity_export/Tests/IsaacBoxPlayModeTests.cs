using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;

using IsaacBox;
using PoRacer.IsaacPorts;

namespace IsaacBox.Tests
{
    /// <summary>
    /// The rung ladder for the IsaacBox port.
    ///
    /// STRICT (fail the run): kinematics vs the independent Python FK, skin attachment,
    /// per-body overrides, rest height (rung 1), zero-g momentum (rung 2), gain units
    /// (rung 2b), observation sanity, decimation logging, and - once a brain exists -
    /// rung 0 (ONNX vs recording) and rung 5 (closed-loop chase).
    ///
    /// INFORMATIVE (always pass, print numbers): rungs 3, 4, 6 and the perf test.
    ///
    /// Rungs that need IsaacBox.onnx / isaac_reference.json report Assert.Ignore until
    /// ISAAC/scripts/export_bundle.py has produced them, so the rig can be validated before
    /// training and the missing brain shows up as "skipped", never as a green tick.
    ///
    /// SetUp pins Time.fixedDeltaTime to Isaac's 0.005 s (which is also this project's
    /// value); TearDown restores whatever it was.
    /// </summary>
    public class IsaacBoxPlayModeTests
    {
        static readonly float ProjectFixedDt = 0.005f;

        IsaacBoxRigAsset _rig;
        float _savedFixedDt;
        readonly List<GameObject> _spawned = new List<GameObject>();

        [SetUp]
        public void SetUp()
        {
            _savedFixedDt = Time.fixedDeltaTime;
            _rig = AssetDatabase.LoadAssetAtPath<IsaacBoxRigAsset>(IsaacBoxPaths.RigAsset);
            Assert.IsNotNull(_rig, $"rig asset missing at {IsaacBoxPaths.RigAsset} - run IsaacBoxSetup.RebuildRigAsset()");
            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var go in _spawned)
            {
                if (go == null) continue;
                var a = go.GetComponent<IsaacBoxAgent>();
                if (a != null) a.ReleaseWorker();
                Object.DestroyImmediate(go);
            }
            _spawned.Clear();
            Time.fixedDeltaTime = _savedFixedDt;
            Physics.gravity = new Vector3(0f, -9.81f, 0f);
            LogAssert.ignoreFailingMessages = false;
        }

        static void AllowDivergenceLogs() => LogAssert.ignoreFailingMessages = true;

        bool HasBrain => File.Exists(IsaacBoxPaths.Onnx) && File.Exists(IsaacBoxPaths.Reference);

        static string NoBrainMessage =>
            $"no trained brain yet: {IsaacBoxPaths.Onnx} / {IsaacBoxPaths.Reference} missing. Run " +
            "ISAAC/scripts/train.py then export_bundle.py, then IsaacBoxSetup.BuildPrefab().";

        // ------------------------------------------------------------- utilities --
        GameObject SpawnGround()
        {
            var g = GameObject.CreatePrimitive(PrimitiveType.Plane);
            g.name = "TestGround";
            g.transform.localScale = new Vector3(20f, 1f, 20f);
            var mat = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacBoxPaths.Material);
            var col = g.GetComponent<Collider>();
            if (mat != null) col.sharedMaterial = mat;
            col.contactOffset = _rig.physics.contactOffset;
            _spawned.Add(g);
            return g;
        }

        IsaacBoxAgent SpawnCreature(Vector3 pos, bool ground = true)
        {
            if (ground) SpawnGround();
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBoxPaths.Prefab);
            Assert.IsNotNull(prefab, $"prefab missing at {IsaacBoxPaths.Prefab} - run IsaacBoxSetup.BuildPrefab()");
            var go = Object.Instantiate(prefab, pos, Quaternion.identity);
            go.name = "IsaacBox_test";
            _spawned.Add(go);
            var agent = go.GetComponent<IsaacBoxAgent>();
            Assert.IsNotNull(agent, "prefab has no IsaacBoxAgent");
            return agent;
        }

        static IEnumerator Steps(int n)
        {
            for (int i = 0; i < n; i++) yield return new WaitForFixedUpdate();
        }

        static void SetZeroGravity(IsaacBoxAgent agent)
        {
            foreach (var b in agent.GetComponentsInChildren<ArticulationBody>(true))
                b.useGravity = false;
        }

        int JointIndex(string jointName)
        {
            for (int i = 0; i < _rig.jointOrder.Length; i++)
                if (_rig.jointOrder[i] == jointName) return i;
            Assert.Fail($"rig has no joint named {jointName}");
            return -1;
        }

        Transform SetTarget(IsaacBoxAgent agent, Vector3 world)
        {
            var t = new GameObject("target").transform;
            t.position = world;
            _spawned.Add(t.gameObject);
            agent.target = t;
            var sampler = agent.GetComponent<IsaacBoxTargetSampler>();
            if (sampler != null) sampler.enabled = false;
            return t;
        }

        /// <summary>A skeleton bone: a transform with this name that carries no ArticulationBody.</summary>
        static Transform FindBone(Transform root, string name)
        {
            if (root.name == name && root.GetComponent<ArticulationBody>() == null) return root;
            foreach (Transform c in root)
            {
                var r = FindBone(c, name);
                if (r != null) return r;
            }
            return null;
        }

        static Transform FindDeep(Transform root, string name)
        {
            if (root.name == name) return root;
            foreach (Transform c in root)
            {
                var r = FindDeep(c, name);
                if (r != null) return r;
            }
            return null;
        }

        // =========================================================== rung 0: ONNX ==
        [UnityTest]
        public IEnumerator Rung0_OnnxInEngineMatchesRecordedActions()
        {
            if (!HasBrain) Assert.Ignore(NoBrainMessage);
            var agent = SpawnCreature(new Vector3(0f, 2f, 0f), ground: false);
            yield return Steps(2);

            Assert.IsTrue(IsaacBoxPaths.TryLoadReference(out var obs, out var acts, out string err), err);
            Assert.AreEqual(_rig.obsDim, obs[0].Length, "recorded obs width != rig obsDim");
            Assert.AreEqual(_rig.actDim, acts[0].Length, "recorded action width != rig actDim");

            float worst = agent.RunReferenceCheck(obs, acts);
            Debug.Log($"[rung 0] in-engine ONNX vs {obs.Length} recorded actions: max abs diff {worst:E3}  (gate 1e-4)");
            Assert.Less(worst, 1e-4f, "Inference Engine disagrees with the recorded actions; " +
                                      "this is a tensor-layout, normaliser or backend problem, not physics.");
        }

        // ======================================= skin: bones ride on the links (strict) ==
        [UnityTest]
        public IEnumerator Skin_BonesAreParentedToTheirLinks()
        {
            var agent = SpawnCreature(new Vector3(0f, 2f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(1);

            var renderers = agent.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            Assert.Greater(renderers.Length, 0, "the prefab carries no skinned mesh - was IsaacBox_Character.fbx found at build time?");

            int attached = 0;
            var lines = new List<string>();
            var links = new Dictionary<string, Transform>();
            foreach (var b in agent.GetComponentsInChildren<ArticulationBody>(true)) links[b.name] = b.transform;
            foreach (var def in _rig.bodies)
            {
                if (string.IsNullOrEmpty(def.boneName)) continue;
                Assert.IsTrue(links.TryGetValue(def.name, out var link), $"link '{def.name}' missing from the prefab");
                // the hips and spine bones share their link's name: a bone is a transform WITHOUT a body
                var bone = FindBone(agent.transform, def.boneName);
                Assert.IsNotNull(bone, $"bone '{def.boneName}' missing from the prefab");
                Assert.AreEqual(link, bone.parent, $"bone '{def.boneName}' is not a direct child of link '{def.name}'");
                float off = (bone.position - link.position).magnitude;
                lines.Add($"    {def.boneName,-14} on {def.name,-12} offset {off * 1000f:F2} mm");
                Assert.Less(off, IsaacBoxRigBuilderGateM, $"bone '{def.boneName}' sits {off * 1000f:F1} mm off its link origin");
                attached++;
            }
            foreach (var bn in _rig.skinBones)
                Assert.IsNotNull(FindBone(agent.transform, bn), $"skin bone '{bn}' is missing; the mesh would tear");

            int verts = 0;
            foreach (var r in renderers) if (r.sharedMesh != null) verts += r.sharedMesh.vertexCount;
            Debug.Log($"[skin] {attached} articulated bones on their links, {renderers.Length} skinned renderers, " +
                      $"{verts:N0} vertices:\n" + string.Join("\n", lines));
        }

        const float IsaacBoxRigBuilderGateM = 0.005f;

        // ============ per-body overrides are runtime-only, so assert them live (strict) ==
        [UnityTest]
        public IEnumerator PerBodyOverrides_AreLiveAtRuntime()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(2);

            var p = _rig.physics;
            var bodies = agent.GetComponentsInChildren<ArticulationBody>(true);
            Assert.AreEqual(_rig.bodies.Length, bodies.Length, "wrong body count");

            float massSum = 0f;
            foreach (var b in bodies)
            {
                massSum += b.mass;
                Assert.AreEqual(p.maxAngularVelocity, b.maxAngularVelocity, 1e-3f, $"{b.name}.maxAngularVelocity");
                Assert.AreEqual(p.maxLinearVelocity, b.maxLinearVelocity, 1e-3f, $"{b.name}.maxLinearVelocity");
                Assert.AreEqual(p.maxDepenetrationVelocity, b.maxDepenetrationVelocity, 1e-3f, $"{b.name}.maxDepenetrationVelocity");
                Assert.AreEqual(p.linearDamping, b.linearDamping, 1e-4f, $"{b.name}.linearDamping");
                Assert.AreEqual(p.angularDamping, b.angularDamping, 1e-4f, $"{b.name}.angularDamping");
                Assert.AreEqual(p.jointFriction, b.jointFriction, 1e-4f, $"{b.name}.jointFriction");
            }
            Assert.AreEqual(_rig.totalMass, massSum, 1e-2f, "total mass differs from the rig");

            foreach (var c in agent.GetComponentsInChildren<Collider>(true))
            {
                Assert.AreEqual(p.contactOffset, c.contactOffset, 1e-4f, $"{c.name}.contactOffset");
                Assert.IsNotNull(c.sharedMaterial, $"{c.name} has no physics material");
                Assert.AreEqual(p.robotStaticFriction, c.sharedMaterial.staticFriction, 1e-4f);
                Assert.AreEqual(p.robotDynamicFriction, c.sharedMaterial.dynamicFriction, 1e-4f);
            }

            for (int j = 0; j < agent.Joints.Count; j++)
                Assert.AreEqual(agent.maxJointVelocity, agent.Joints[j].maxJointVelocity, 1e-3f,
                    $"joint {_rig.jointOrder[j]}.maxJointVelocity");

            var cols = agent.GetComponentsInChildren<Collider>(true);
            int pairs = 0;
            for (int i = 0; i < cols.Length; i++)
                for (int k = i + 1; k < cols.Length; k++)
                    if (!Physics.GetIgnoreCollision(cols[i], cols[k])) pairs++;

            var root = agent.Root;
            Debug.Log($"[overrides] live values on {bodies.Length} bodies / {cols.Length} colliders, {massSum:F2} kg:\n" +
                      $"    maxAngularVelocity {root.maxAngularVelocity:F0}  maxJointVelocity {agent.Joints[0].maxJointVelocity:F0}  " +
                      $"maxDepenetrationVel {root.maxDepenetrationVelocity:F2}  contactOffset {cols[0].contactOffset:F3}\n" +
                      $"    solverIterations {root.solverIterations}/{root.solverVelocityIterations}  " +
                      $"material {cols[0].sharedMaterial.staticFriction:F2}/{cols[0].sharedMaterial.dynamicFriction:F2} " +
                      $"{cols[0].sharedMaterial.frictionCombine}  un-ignored self pairs {pairs} (must be 0)");

            Assert.AreEqual(0, pairs, "self-collision is enabled somewhere; the Isaac cfg says enabled_self_collisions: false");
            agent.ResolveSolverIterations(out int expectPos, out int expectVel);
            Assert.AreEqual(expectPos, root.solverIterations, "root solverIterations");
            Assert.AreEqual(expectVel, root.solverVelocityIterations, "root solverVelocityIterations");
            Assert.AreEqual(p.solverPositionIterations, expectPos, "at Isaac's own step the resolved count must equal the cfg's 4");
        }

        // ===================================================== kinematics (strict) ==
        [UnityTest]
        public IEnumerator Kinematics_MatchesIndependentPythonForwardKinematics()
        {
            Assert.IsTrue(File.Exists(IsaacBoxPaths.KinematicsReference),
                $"{IsaacBoxPaths.KinematicsReference} missing - run ISAAC/boy_rig/build_boy_rig.py");
            var root = MiniJson.Parse(File.ReadAllText(IsaacBoxPaths.KinematicsReference)) as Dictionary<string, object>;
            var bodyOrder = MiniJson.StrArray(root, "bodyOrder");
            var jointOrder = MiniJson.StrArray(root, "jointOrder");
            var poses = MiniJson.Arr(root, "poses");

            var agent = SpawnCreature(new Vector3(0f, 3f, 0f), ground: false);
            SetZeroGravity(agent);
            agent.enabled = false;                      // no policy, no drive updates
            var rootBody = agent.Root;
            rootBody.immovable = true;
            var links = new Dictionary<string, Transform>();
            foreach (var b in agent.GetComponentsInChildren<ArticulationBody>(true))
                links[b.name] = b.transform;
            var jointByName = new Dictionary<string, ArticulationBody>();
            for (int j = 0; j < agent.Joints.Count; j++) jointByName[_rig.jointOrder[j]] = agent.Joints[j];
            yield return Steps(2);

            float worstAll = 0f;
            string worstWhere = "";
            var lines = new List<string>();

            foreach (var po in poses)
            {
                var p = po as Dictionary<string, object>;
                string poseName = MiniJson.Str(p, "name");
                float[] q = MiniJson.FloatArray(p, "jointPosRad");
                var expected = MiniJson.Arr(p, "linkPosInHipsIsaac");

                for (int j = 0; j < jointOrder.Length; j++)
                {
                    var b = jointByName[jointOrder[j]];
                    b.jointPosition = new ArticulationReducedSpace(q[j]);
                    b.jointVelocity = new ArticulationReducedSpace(0f);
                    var d = b.xDrive;
                    d.target = q[j] * Mathf.Rad2Deg;
                    b.xDrive = d;
                }
                yield return Steps(3);

                float worst = 0f;
                string where = "";
                var hips = links["hips"];
                for (int i = 0; i < bodyOrder.Length; i++)
                {
                    var e = expected[i] as List<object>;
                    var exp = new Vector3((float)(double)e[0], (float)(double)e[1], (float)(double)e[2]);
                    Vector3 gotUnity = hips.InverseTransformPoint(links[bodyOrder[i]].position);
                    Vector3 got = IsaacFrameMap.PosToIsaac(gotUnity);
                    float d2 = (got - exp).magnitude;
                    if (d2 > worst) { worst = d2; where = bodyOrder[i]; }
                }
                lines.Add($"    {poseName,-8} worst {worst * 1000f:F4} mm on {where}");
                if (worst > worstAll) { worstAll = worst; worstWhere = $"{poseName}/{where}"; }
            }

            Debug.Log($"[kinematics] Unity rig vs independent Python FK, {poses.Count} poses x {bodyOrder.Length} links:\n" +
                      string.Join("\n", lines) + $"\n    WORST {worstAll * 1000f:F4} mm at {worstWhere}   (gate 1.000 mm)");

            Assert.Less(worstAll, 1e-3f,
                "Unity link positions disagree with the Python FK. The frame map, an anchor axis, " +
                "or a joint sign is wrong - fix this before believing any dynamics result.");
        }

        // ============================================== rung 1: rest height (strict) ==
        /// <summary>
        /// Do the LEG DRIVES carry the body at the exported rest height? A biped's ankle drive
        /// (kp 30 N.m/rad here, 20 on the H1) cannot hold a 45 kg inverted pendulum: without
        /// a policy the creature topples in about a second - measured, and the same physics
        /// Isaac has, where the trained policy does the balancing. So this rung fixes the
        /// balancing away with a kinematic assist (root tilt rate and planar drift zeroed
        /// every step; height is left entirely to the legs), switches fall recovery OFF so it
        /// cannot fake a pass, and asks only the question a rest-height rung can answer.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung1_LegsHoldTheBodyAtTheExportedRestHeight()
        {
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            agent.target = null;
            agent.autoRecoverFromFalls = false;
            var sampler = agent.GetComponent<IsaacBoxTargetSampler>();
            if (sampler != null) sampler.enabled = false;   // hold pose: no target, no policy

            int n = Mathf.CeilToInt(3f / Time.fixedDeltaTime);
            var root = agent.Root;
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                // balance assist: keep the hips level and in place, let gravity act vertically
                root.angularVelocity = Vector3.zero;
                Vector3 v = root.linearVelocity;
                root.linearVelocity = new Vector3(0f, v.y, 0f);
            }

            float hipsY = root.transform.position.y;
            float comY = agent.CenterOfMassPosition.y;
            float upright = Vector3.Dot(root.transform.up, Vector3.up);
            float lo = _rig.hipsHeightAtDefaultPoseRest - 0.12f;
            float hi = _rig.hipsHeightAtDefaultPoseRest + 0.06f;
            float worstSag = 0f; string sagJoint = "";
            for (int j = 0; j < agent.Joints.Count; j++)
            {
                float err = agent.Joints[j].jointPosition[0] - agent.Joints[j].xDrive.target * Mathf.Deg2Rad;
                if (Mathf.Abs(err) > Mathf.Abs(worstSag)) { worstSag = err; sagJoint = _rig.jointOrder[j]; }
            }
            Debug.Log($"[rung 1] after 3 s standing with the balance assist (no policy, no recovery):\n" +
                      $"    hips y = {hipsY:F4} m, CoM y = {comY:F4} m, upright {upright:F3}\n" +
                      $"    expected band [{lo:F3}, {hi:F3}] m   (FK rest at the default pose " +
                      $"{_rig.hipsHeightAtDefaultPoseRest:F4} m, spawn {_rig.spawnPosIsaac.z:F4} m)\n" +
                      $"    worst drive sag {worstSag:F4} rad ({worstSag * Mathf.Rad2Deg:F1} deg) on {sagJoint}, " +
                      $"recoveries {agent.Recoveries} (must be 0)");

            Assert.AreEqual(0, agent.Recoveries, "fall recovery fired; the number above is not a rest height");
            Assert.Greater(upright, 0.95f, "the assist failed to keep the hips level");
            Assert.Greater(hipsY, lo, "the creature collapsed - legs are not holding it up");
            Assert.Less(hipsY, hi, "the creature is floating or was launched");
        }

        /// <summary>
        /// The companion measurement: the SAME rig with nothing helping it. Informative, and
        /// expected to fall - this is the number the policy has to beat. If it ever stands on
        /// its own the gains changed.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung1b_UnassistedStandingIsNotStable()
        {
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            agent.target = null;
            agent.autoRecoverFromFalls = false;
            var sampler = agent.GetComponent<IsaacBoxTargetSampler>();
            if (sampler != null) sampler.enabled = false;

            float tFall = -1f;
            int n = Mathf.CeilToInt(3f / Time.fixedDeltaTime);
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                if (tFall < 0f && Vector3.Dot(agent.Root.transform.up, Vector3.up) < 0.5f) tFall = i * Time.fixedDeltaTime;
            }
            float upright = Vector3.Dot(agent.Root.transform.up, Vector3.up);
            Debug.Log($"[rung 1b] unassisted, policy OFF, recovery OFF, 3 s: " +
                      (tFall < 0f ? "did not fall" : $"fell at t = {tFall:F2} s") +
                      $", final upright {upright:F3}, hips y {agent.Root.transform.position.y:F3} m. " +
                      "Expected to fall: ankle kp 30 N.m/rad vs ~335 N.m/rad of gravitational " +
                      "destabilisation for a 45 kg body - the trained policy balances, as in Isaac.");
            Assert.Pass("informative");
        }

        // ======================================= rung 2: zero-g single joint (strict) ==
        [UnityTest]
        public IEnumerator Rung2_ZeroGravitySingleJointDoesNotMoveTheCentreOfMass()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            int J = JointIndex("knee_L");
            agent.actionOverride = IsaacBoxAgent.ActionOverride.Constant;
            agent.overrideJointIndex = J;
            agent.overrideAmplitude = 1f;

            yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));

            float worst = 0f;
            int n = Mathf.CeilToInt(2f / Time.fixedDeltaTime);
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                worst = Mathf.Max(worst, agent.CenterOfMassVelocity.magnitude);
            }

            Debug.Log($"[rung 2] zero-g, driving joint {_rig.jointOrder[J]} only, fdt {Time.fixedDeltaTime:F5}: " +
                      $"max |vCoM| = {worst:F5} m/s  (gate 0.02)");
            Assert.Less(worst, 0.02f, "the whole-body centre of mass accelerated with no external force: " +
                                      "momentum pumping in the articulation solver.");
        }

        // ------------------------- rung 2b: which units do drive gains use? -----------
        [UnityTest]
        public IEnumerator Rung2b_GainUnitsCalibration()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            agent.enabled = false;
            agent.Root.immovable = true;

            int J = JointIndex("elbow_L");            // a leaf: least chain reaction
            var body = agent.Joints[J];
            var def = _rig.JointBodiesInIsaacOrder()[J].joint;
            float kpNominal = def.stiffness;
            float I = agent.InertiaAboutJointAxis(J);
            float q0 = def.defaultPosRad;
            const float err = 0.2f;

            body.jointPosition = new ArticulationReducedSpace(q0);
            body.jointVelocity = new ArticulationReducedSpace(0f);
            var d = body.xDrive;
            d.damping = 0f;
            d.forceLimit = 1e9f;
            d.target = (q0 + err) * Mathf.Rad2Deg;
            body.xDrive = d;
            yield return new WaitForFixedUpdate();

            float w1 = body.jointVelocity[0];
            float alpha = w1 / Time.fixedDeltaTime;
            float tau = alpha * I;
            float kpMeasured = tau / err;

            float ifRadians = kpNominal;
            float ifDegrees = kpNominal * Mathf.Rad2Deg;
            bool radians = Mathf.Abs(kpMeasured - ifRadians) < Mathf.Abs(kpMeasured - ifDegrees);

            Debug.Log($"[rung 2b] ArticulationDrive gain-unit calibration on {_rig.jointOrder[J]}:\n" +
                      $"    kp (Isaac) {kpNominal:F1}   I about axis {I:F5} kg.m2   error {err:F3} rad\n" +
                      $"    omega after 1 step {w1:F5} rad/s -> torque {tau:F3} N.m -> kp measured {kpMeasured:F2}\n" +
                      $"    radian hypothesis {ifRadians:F2}, degree hypothesis {ifDegrees:F2}\n" +
                      $"    VERDICT: gains are {(radians ? "RADIAN" : "DEGREE")}-based; agent ships GainUnits.{agent.gainUnits}");

            Assert.IsTrue(radians == (agent.gainUnits == IsaacBoxAgent.GainUnits.Radians),
                $"measured kp {kpMeasured:F2} says the drive is {(radians ? "RADIAN" : "DEGREE")}-based, but the agent " +
                $"ships GainUnits.{agent.gainUnits}. Every joint would be off by 57.3x.");
        }

        // ================== observation sanity at a known state (strict) =============
        [UnityTest]
        public IEnumerator Observations_AreCorrectAtTheSpawnPose()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            // a target straight ahead at 3 m: target_pos_b must read (3, 0, 0) in Isaac terms
            SetTarget(agent, new Vector3(0f, 5f, 3f));
            agent.actionOverride = IsaacBoxAgent.ActionOverride.Constant;
            agent.overrideAmplitude = 0f;           // hold the default pose exactly
            yield return Steps(4);

            var o = agent.LatestObservation;
            Assert.IsNotNull(o, "no observation was built");
            int n = _rig.actDim;

            float maxJp = 0f, maxJv = 0f, maxAct = 0f;
            for (int j = 0; j < n; j++)
            {
                maxJp = Mathf.Max(maxJp, Mathf.Abs(o[12 + j]));
                maxJv = Mathf.Max(maxJv, Mathf.Abs(o[12 + n + j]));
                maxAct = Mathf.Max(maxAct, Mathf.Abs(o[12 + 2 * n + j]));
            }
            Debug.Log($"[obs] at the spawn pose, zero-g, target 3 m ahead:\n" +
                      $"    base_lin_vel      {o[0]:F4} {o[1]:F4} {o[2]:F4}   expect ~0\n" +
                      $"    base_ang_vel      {o[3]:F4} {o[4]:F4} {o[5]:F4}   expect ~0\n" +
                      $"    projected_gravity {o[6]:F4} {o[7]:F4} {o[8]:F4}   expect 0 0 -1\n" +
                      $"    target_pos_b      {o[9]:F4} {o[10]:F4} {o[11]:F4}   expect 3 0 0\n" +
                      $"    max |joint_pos|   {maxJp:F4}   expect ~0 (we are at the default pose)\n" +
                      $"    max |joint_vel|   {maxJv:F4}   expect ~0\n" +
                      $"    max |action|      {maxAct:F4}   (override holds 0)");

            Assert.Less(Mathf.Abs(o[6]), 0.05f, "projected_gravity.x should be ~0 while upright");
            Assert.Less(Mathf.Abs(o[7]), 0.05f, "projected_gravity.y should be ~0 while upright");
            Assert.Less(Mathf.Abs(o[8] + 1f), 0.05f, "projected_gravity.z should be -1 while upright");
            Assert.AreEqual(3f, o[9], 0.05f, "target_pos_b.x: a target straight ahead must be +X in Isaac");
            Assert.Less(Mathf.Abs(o[10]), 0.05f, "target_pos_b.y should be 0 for a target straight ahead");
            Assert.Less(maxJp, 0.05f, "joint_pos is relative to the default pose and we are AT it");
        }

        [UnityTest]
        public IEnumerator Observations_TargetIsClippedAndLeftIsPositiveY()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            // 20 m to the creature's LEFT (Unity -X): Isaac +Y, clipped to the obs radius
            SetTarget(agent, new Vector3(-20f, 5f, 0f));
            agent.actionOverride = IsaacBoxAgent.ActionOverride.Constant;
            agent.overrideAmplitude = 0f;
            yield return Steps(4);

            var o = agent.LatestObservation;
            float clip = _rig.chase.targetObsClip;
            Debug.Log($"[obs] target 20 m to the left: target_pos_b = ({o[9]:F3}, {o[10]:F3}, {o[11]:F3}), clip {clip}");
            Assert.AreEqual(clip, o[10], 0.05f, "a target on the creature's left must read +Y in Isaac, at the clip radius");
            Assert.Less(Mathf.Abs(o[9]), 0.05f);
        }

        // ============================ rung 3: zero-g square wave (informative) ========
        [UnityTest]
        public IEnumerator Rung3_ZeroGravitySquareWaveMomentumPumping()
        {
            AllowDivergenceLogs();
            var rows = new List<string>();
            float[] steps = { _rig.isaacPhysicsDt, 1f / 100f, 1f / 500f };
            string[] labels = { "1/200 (Isaac + project)", "1/100", "1/500" };

            for (int c = 0; c < steps.Length; c++)
            {
                Time.fixedDeltaTime = steps[c];
                var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
                SetZeroGravity(agent);
                agent.actionOverride = IsaacBoxAgent.ActionOverride.SquareWave;
                agent.overrideJointIndex = -1;
                agent.overrideAmplitude = 1f;
                agent.overrideSquareWavePeriod = 0.4f;

                yield return Steps(Mathf.CeilToInt(0.5f / Time.fixedDeltaTime));

                float worst = 0f;
                Vector3 p0 = agent.CenterOfMassPosition;
                int n = Mathf.CeilToInt(3f / Time.fixedDeltaTime);
                for (int i = 0; i < n; i++)
                {
                    yield return new WaitForFixedUpdate();
                    worst = Mathf.Max(worst, agent.CenterOfMassVelocity.magnitude);
                }
                float drift = (agent.CenterOfMassPosition - p0).magnitude;
                rows.Add($"    {labels[c],-24} fdt {steps[c]:F5}  max|vCoM| {worst,8:F4} m/s   CoM drift over 3 s {drift,7:F4} m");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
            Debug.Log($"[rung 3] zero-g bang-bang square wave on all {_rig.actDim} joints. With no contacts " +
                      "|vCoM| must stay 0 - anything else is momentum pumping:\n" + string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ================================ rung 4: zero-g policy (informative) =========
        [UnityTest]
        public IEnumerator Rung4_ZeroGravityPolicyRuns()
        {
            if (!HasBrain) Assert.Ignore(NoBrainMessage);
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            SetTarget(agent, new Vector3(0f, 5f, 30f));

            yield return Steps(Mathf.CeilToInt(5f / Time.fixedDeltaTime));

            float maxAbsAction = 0f;
            var a = agent.LatestAction;
            for (int i = 0; i < a.Length; i++) maxAbsAction = Mathf.Max(maxAbsAction, Mathf.Abs(a[i]));
            float maxJointVel = 0f;
            for (int i = 0; i < agent.Joints.Count; i++)
                maxJointVel = Mathf.Max(maxJointVel, Mathf.Abs(agent.Joints[i].jointVelocity[0]));

            Debug.Log($"[rung 4] zero-g, policy driving, 5 s at fdt {Time.fixedDeltaTime:F5}:\n" +
                      $"    policy steps {agent.PolicySteps} (decimation {agent.Decimation})  max |action| {maxAbsAction:F3}  " +
                      $"max |jointVel| {maxJointVel:F3} rad/s  |vCoM| {agent.CenterOfMassVelocity.magnitude:F4} m/s");
            Assert.IsFalse(float.IsNaN(maxAbsAction), "policy produced NaN");
            Assert.Pass("informative");
        }

        // ================================== rung 5: locomotion (strict) ==============
        [UnityTest]
        public IEnumerator Rung5_WalksTowardsADistantTarget()
        {
            if (!HasBrain) Assert.Ignore(NoBrainMessage);
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            var target = SetTarget(agent, new Vector3(0f, 0f, 20f));   // >= 10 m, so closed/time == speed

            Vector3 p0 = agent.Root.transform.position;
            float d0 = Vector2.Distance(new Vector2(p0.x, p0.z), new Vector2(target.position.x, target.position.z));

            const float duration = 12f;
            yield return Steps(Mathf.CeilToInt(duration / Time.fixedDeltaTime));

            Vector3 p1 = agent.Root.transform.position;
            float d1 = Vector2.Distance(new Vector2(p1.x, p1.z), new Vector2(target.position.x, target.position.z));
            float closed = d0 - d1;
            float speed = closed / duration;
            float uprightDot = Vector3.Dot(agent.Root.transform.up, Vector3.up);

            Debug.Log($"[rung 5] {duration:F0} s chasing a target {d0:F1} m away, fdt {Time.fixedDeltaTime:F5}:\n" +
                      $"    distance closed {closed:F3} m (gate > 1.0)   mean speed {speed:F3} m/s   hips height {p1.y:F3} m\n" +
                      $"    upright dot {uprightDot:F3} (gate > 0.5)   recoveries {agent.Recoveries}\n" +
                      $"    Isaac reference {_rig.isaacReferenceForwardSpeed:F3} m/s toward a fixed target; " +
                      $"eval mean {_rig.isaacMeanSpeedTowardTarget:F3} m/s toward random targets");

            Assert.Greater(uprightDot, 0.5f, "the creature fell over");
            Assert.Greater(closed, 1.0f, "the creature did not make meaningful progress");
        }

        // ============================ decimation logging (strict on the message) =====
        [UnityTest]
        public IEnumerator Decimation_LogsAnErrorWhenTheStepDoesNotDividePolicyDt()
        {
            Time.fixedDeltaTime = 0.03f;
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(
                @"policy_dt / Time\.fixedDeltaTime = 0\.020000 / 0\.030000 = 0\.666\d+, which is NOT an integer"));

            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(3);

            Debug.Log($"[decimation] fdt 0.03 -> ratio {_rig.policyDt / 0.03f:F4}, agent ran with decimation " +
                      $"{agent.Decimation} and logged the error above.");
            Assert.AreEqual(1, agent.Decimation);
            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
        }

        [UnityTest]
        public IEnumerator Decimation_IsExactAtTheProjectStep()
        {
            Time.fixedDeltaTime = ProjectFixedDt;
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(3);

            Assert.AreEqual(_rig.isaacDecimation, agent.Decimation,
                "at the project step of 0.005 s the policy must run every 4th fixed step, exactly as Isaac did");
            Debug.Log($"[decimation] project step {ProjectFixedDt:F5}: ratio {_rig.policyDt / ProjectFixedDt:F4} -> " +
                      $"decimation {agent.Decimation}, matching Isaac. No warning, no error.");
        }

        // ================================ rung 6: speed parity (informative) =========
        [UnityTest]
        public IEnumerator Rung6_SpeedParityAgainstIsaac()
        {
            if (!HasBrain) Assert.Ignore(NoBrainMessage);
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            SetTarget(agent, new Vector3(0f, 0f, 40f));

            yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));   // settle
            Vector3 p0 = agent.Root.transform.position;
            const float duration = 15f;
            yield return Steps(Mathf.CeilToInt(duration / Time.fixedDeltaTime));
            Vector3 p1 = agent.Root.transform.position;

            float travelled = Vector2.Distance(new Vector2(p0.x, p0.z), new Vector2(p1.x, p1.z));
            float speed = travelled / duration;
            float isaac = _rig.isaacReferenceForwardSpeed > 0f ? _rig.isaacReferenceForwardSpeed : _rig.isaacMeanSpeedTowardTarget;
            float parity = isaac > 0f ? speed / isaac : float.NaN;

            Debug.Log($"[rung 6] speed parity toward a distant target:\n" +
                      $"    Unity  {speed:F3} m/s over {duration:F0} s ({travelled:F2} m), recoveries {agent.Recoveries}\n" +
                      $"    Isaac  {isaac:F3} m/s (reference run, fixed target {_rig.isaacReferenceTargetDistance:F0} m ahead)\n" +
                      $"    parity {parity:P1}   Isaac eval: {_rig.isaacFallsPerRobotPerMinute:F3} falls/robot/min, " +
                      $"{_rig.isaacTargetsReachedPerMinute:F2} targets/robot/min");
            Assert.Pass("informative");
        }

        // ============= drive tracking under load, policy OFF (informative) ===========
        [UnityTest]
        public IEnumerator Diag_DriveTrackingUnderGravity()
        {
            var rows = new List<string>();
            int[] iters = { 4, 16, 64 };

            foreach (int it in iters)
            {
                var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
                agent.enabled = false;
                foreach (var b in agent.GetComponentsInChildren<ArticulationBody>(true))
                {
                    b.solverIterations = it;
                    b.solverVelocityIterations = it;
                }

                yield return Steps(Mathf.CeilToInt(3f / Time.fixedDeltaTime));

                float worst = 0f; int argw = 0; float sumAbs = 0f;
                for (int j = 0; j < agent.Joints.Count; j++)
                {
                    var b = agent.Joints[j];
                    float tgt = b.xDrive.target * Mathf.Deg2Rad;
                    float err = b.jointPosition[0] - tgt;
                    sumAbs += Mathf.Abs(err);
                    if (Mathf.Abs(err) > Mathf.Abs(worst)) { worst = err; argw = j; }
                }
                var hips = agent.Root.transform.position;
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                rows.Add($"    solverIter {it,2}/{it,-2}  hips h {hips.y,6:F4} m  upright {up,6:F3}  " +
                         $"worst joint error {worst,7:F4} rad ({worst * Mathf.Rad2Deg,6:F2} deg) on {_rig.jointOrder[argw],-16} " +
                         $"mean|err| {sumAbs / agent.Joints.Count,7:F4} rad");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Debug.Log($"[diag-drive] holding the DEFAULT pose under gravity, policy OFF, fdt {Time.fixedDeltaTime:F5}. " +
                      $"FK rest height {_rig.hipsHeightAtDefaultPoseRest:F4} m.\n" + string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ==================================== perf: N creatures (informative) ========
        [UnityTest]
        public IEnumerator Perf_EightCreaturesAtTheProjectStep()
        {
            const int N = 8;
            Time.fixedDeltaTime = ProjectFixedDt;
            SpawnGround();

            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(IsaacBoxPaths.Prefab);
            var agents = new List<IsaacBoxAgent>();
            for (int i = 0; i < N; i++)
            {
                var go = Object.Instantiate(prefab,
                    new Vector3((i % 4) * 2.5f - 3.75f, _rig.spawnPosIsaac.z, (i / 4) * 2.5f), Quaternion.identity);
                _spawned.Add(go);
                var a = go.GetComponent<IsaacBoxAgent>();
                SetTarget(a, new Vector3(0f, 0f, 40f));
                agents.Add(a);
            }

            yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));

            int frames = Mathf.CeilToInt(6f / Time.fixedDeltaTime);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < frames; i++) yield return new WaitForFixedUpdate();
            sw.Stop();

            double msPerStep = sw.Elapsed.TotalMilliseconds / frames;
            double msPerRenderedFrame = msPerStep * (1.0 / 60.0) / Time.fixedDeltaTime;
            double budgetMs = 1000.0 / 60.0;
            int standing = 0;
            foreach (var a in agents)
                if (Vector3.Dot(a.Root.transform.up, Vector3.up) > 0.5f) standing++;

            Debug.Log($"[perf] {N} Boys at fdt {Time.fixedDeltaTime:F5}, decimation {agents[0].Decimation}, " +
                      $"brain {(HasBrain ? "ON" : "OFF (holding pose)")}, {frames} fixed steps:\n" +
                      $"    wall time per fixed step {msPerStep:F3} ms -> {msPerRenderedFrame:F3} ms per 60 FPS frame " +
                      $"(budget {budgetMs:F3} ms, headroom {budgetMs / msPerRenderedFrame:F2}x)\n" +
                      $"    still upright after 8 s {standing}/{N}\n" +
                      "    NOTE: measured inside the editor with the test runner attached - a floor, not a shipping number.");
            Assert.Pass("informative");
        }
    }
}
