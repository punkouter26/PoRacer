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

using PoRacer.IsaacPorts;

namespace IsaacH1.Tests
{
    /// <summary>
    /// The rung ladder. Rungs 0, 1, 2, 5 and the kinematics check are STRICT gates and
    /// will fail the run. Rungs 3, 4, 6 and the perf test are INFORMATIVE: they always
    /// pass and exist to print numbers, because their thresholds depend on the project's
    /// physics step, which this package is not allowed to change.
    ///
    /// SetUp pins Time.fixedDeltaTime to the exact divisor of policy_dt that reproduces
    /// Isaac (0.005 s, decimation 4); TearDown restores the project value. Individual
    /// tests that are about the step itself override it and restore it themselves.
    /// </summary>
    public class IsaacH1PlayModeTests
    {
        const string RigAssetPath = IsaacH1Paths.RigAsset;
        const string PrefabPath = IsaacH1Paths.Prefab;
        const string KinematicsPath = IsaacH1Paths.KinematicsReference;

        // The project's own value, captured once so TearDown can always put it back.
        static readonly float ProjectFixedDt = 0.02f;

        IsaacH1RigAsset _rig;
        float _savedFixedDt;
        readonly List<GameObject> _spawned = new List<GameObject>();

        [SetUp]
        public void SetUp()
        {
            _savedFixedDt = Time.fixedDeltaTime;
            _rig = AssetDatabase.LoadAssetAtPath<IsaacH1RigAsset>(RigAssetPath);
            Assert.IsNotNull(_rig, $"rig asset missing at {RigAssetPath} - run " +
                                   "IsaacH1 > Rebuild Rig Asset From JSON");
            // The exact divisor: policy_dt / isaacDecimation == Isaac's own physics step.
            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (var go in _spawned)
            {
                if (go == null) continue;
                var a = go.GetComponent<IsaacH1Agent>();
                if (a != null) a.ReleaseWorker();
                Object.DestroyImmediate(go);
            }
            _spawned.Clear();
            Time.fixedDeltaTime = _savedFixedDt;
            Physics.gravity = new Vector3(0f, -9.81f, 0f);
            // Only the two divergence diagnostics ever set this; clearing it here means a
            // strict rung can never inherit a relaxed log gate from the test before it.
            LogAssert.ignoreFailingMessages = false;
        }

        /// <summary>
        /// Lets a test that DELIBERATELY diverges keep running. When the articulation is
        /// driven bang-bang (rung 3) or with a knowingly-inadequate solver
        /// (<see cref="Diag_ProjectStepRescueAttempts"/>) the creature is flung far from
        /// the origin, and Unity's renderer emits a native
        /// "[Assert] Invalid worldAABB" that the Test Runner counts as an unhandled log.
        /// That assert is a rendering-bounds complaint about the 534k-triangle visual
        /// meshes, not a measurement: the numbers these tests print are still the numbers.
        /// Strict rungs never call this - they must fail on unexpected logs.
        /// </summary>
        static void AllowDivergenceLogs() => LogAssert.ignoreFailingMessages = true;

        // ------------------------------------------------------------- utilities --
        GameObject SpawnGround()
        {
            var g = GameObject.CreatePrimitive(PrimitiveType.Plane);
            g.name = "TestGround";
            g.transform.localScale = new Vector3(20f, 1f, 20f);
            var mat = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(IsaacH1Paths.Material);
            var col = g.GetComponent<Collider>();
            if (mat != null) col.sharedMaterial = mat;
            col.contactOffset = _rig.physics.contactOffset;
            _spawned.Add(g);
            return g;
        }

        IsaacH1Agent SpawnCreature(Vector3 pos, bool ground = true)
        {
            if (ground) SpawnGround();
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
            Assert.IsNotNull(prefab, $"prefab missing at {PrefabPath} - run IsaacH1 > Build Prefab");
            var go = Object.Instantiate(prefab, pos, Quaternion.identity);
            go.name = "IsaacH1_test";
            _spawned.Add(go);
            var agent = go.GetComponent<IsaacH1Agent>();
            Assert.IsNotNull(agent, "prefab has no IsaacH1Agent");
            return agent;
        }

        static IEnumerator Steps(int n)
        {
            for (int i = 0; i < n; i++) yield return new WaitForFixedUpdate();
        }

        /// <summary>Turns gravity off for one creature without touching Physics.gravity.</summary>
        static void SetZeroGravity(IsaacH1Agent agent)
        {
            foreach (var b in agent.GetComponentsInChildren<ArticulationBody>(true))
                b.useGravity = false;
        }

        static float PlanarSpeed(Vector3 v) => new Vector2(v.x, v.z).magnitude;

        // =========================================================== rung 0: ONNX ==
        [UnityTest]
        public IEnumerator Rung0_OnnxInEngineMatchesRecordedActions()
        {
            var agent = SpawnCreature(new Vector3(0f, 1.05f, 0f), ground: false);
            yield return Steps(2);

            Assert.IsTrue(IsaacH1Paths.TryLoadReference(out var obs, out var acts, out string err), err);

            float worst = agent.RunReferenceCheck(obs, acts);
            Debug.Log($"[rung 0] in-engine ONNX vs {obs.Length} recorded actions: " +
                      $"max abs diff {worst:E3}  (gate 1e-4; check_onnx.py reports 2.384E-006 " +
                      "for the same data under onnxruntime)");
            Assert.Less(worst, 1e-4f, "Inference Engine disagrees with the recorded actions; " +
                                      "this is a tensor-layout or backend problem, not physics.");
        }

        // ============ per-body overrides are runtime-only, so assert them live (strict) ==
        [UnityTest]
        public IEnumerator PerBodyOverrides_AreLiveAtRuntime()
        {
            // Unity serialises m_Mass / m_InertiaTensor / m_CenterOfMass on an
            // ArticulationBody but NOT contactOffset, solverIterations,
            // maxJointVelocity, maxAngular/LinearVelocity or maxDepenetrationVelocity.
            // Inspecting the prefab therefore proves nothing about those - they exist
            // only if IsaacH1Agent.ApplyPerBodyOverrides ran. Hence this test.
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(2);

            var p = _rig.physics;
            var bodies = agent.GetComponentsInChildren<ArticulationBody>(true);
            Assert.AreEqual(_rig.bodies.Length, bodies.Length, "wrong body count");

            foreach (var b in bodies)
            {
                Assert.AreEqual(p.maxAngularVelocity, b.maxAngularVelocity, 1e-3f,
                    $"{b.name}.maxAngularVelocity (project default is 50, env.yaml says 1000)");
                Assert.AreEqual(p.maxLinearVelocity, b.maxLinearVelocity, 1e-3f, $"{b.name}.maxLinearVelocity");
                Assert.AreEqual(p.maxDepenetrationVelocity, b.maxDepenetrationVelocity, 1e-3f,
                    $"{b.name}.maxDepenetrationVelocity");
                Assert.AreEqual(p.linearDamping, b.linearDamping, 1e-4f, $"{b.name}.linearDamping");
                Assert.AreEqual(p.angularDamping, b.angularDamping, 1e-4f, $"{b.name}.angularDamping");
                Assert.AreEqual(p.jointFriction, b.jointFriction, 1e-4f, $"{b.name}.jointFriction");
            }

            foreach (var c in agent.GetComponentsInChildren<Collider>(true))
            {
                Assert.AreEqual(p.contactOffset, c.contactOffset, 1e-4f,
                    $"{c.name}.contactOffset (Unity project default is 0.01, Isaac ran 0.02)");
                Assert.IsNotNull(c.sharedMaterial, $"{c.name} has no physics material");
                Assert.AreEqual(p.robotStaticFriction, c.sharedMaterial.staticFriction, 1e-4f);
                Assert.AreEqual(p.robotDynamicFriction, c.sharedMaterial.dynamicFriction, 1e-4f);
            }

            // maxJointVelocity: Isaac left velocity_limit_sim null, so this must be the
            // link angular cap, NOT Unity's default 100 and NOT the URDF limit.
            for (int j = 0; j < agent.Joints.Count; j++)
                Assert.AreEqual(agent.maxJointVelocity, agent.Joints[j].maxJointVelocity, 1e-3f,
                    $"joint {_rig.jointOrder[j]}.maxJointVelocity");

            // self-collision off: only 3 shapes exist and none may collide with another
            var cols = agent.GetComponentsInChildren<Collider>(true);
            int pairs = 0;
            for (int i = 0; i < cols.Length; i++)
                for (int k = i + 1; k < cols.Length; k++)
                    if (!Physics.GetIgnoreCollision(cols[i], cols[k])) pairs++;

            var root = agent.Root;
            Debug.Log($"[overrides] live values on {bodies.Length} bodies / {cols.Length} colliders:\n" +
                      $"    maxAngularVelocity   {root.maxAngularVelocity:F0}   (project default 50)\n" +
                      $"    maxJointVelocity     {agent.Joints[0].maxJointVelocity:F0}   (Unity default 100)\n" +
                      $"    maxDepenetrationVel  {root.maxDepenetrationVelocity:F2}\n" +
                      $"    contactOffset        {cols[0].contactOffset:F3}   (project default 0.01)\n" +
                      $"    solverIterations     {root.solverIterations}/{root.solverVelocityIterations}   " +
                      $"(project default 12/4, env.yaml 4/4; PhysX applies these per ARTICULATION from the root)\n" +
                      $"    material             {cols[0].sharedMaterial.staticFriction:F2}/" +
                      $"{cols[0].sharedMaterial.dynamicFriction:F2} {cols[0].sharedMaterial.frictionCombine}\n" +
                      $"    un-ignored self pairs {pairs} (must be 0)\n" +
                      $"    inertia w/ armature   {agent.Joints[11].inertiaTensor}  " +
                      $"(left_knee raw + 0.1 on the joint axis)");

            Assert.AreEqual(0, pairs, "self-collision is enabled somewhere; env.yaml says " +
                                      "enabled_self_collisions: false");
            // At Isaac's own step AutoScaleWithStep resolves to env.yaml's 4/4; at a
            // coarser step it raises them, which is a per-body override and legitimate.
            agent.ResolveSolverIterations(out int expectPos, out int expectVel);
            Assert.AreEqual(expectPos, root.solverIterations, "root solverIterations");
            Assert.AreEqual(expectVel, root.solverVelocityIterations, "root solverVelocityIterations");
            Assert.AreEqual(p.solverPositionIterations, expectPos,
                "at Isaac's own step the resolved count must equal env.yaml's 4");
        }

        // ===================================================== kinematics (strict) ==
        [UnityTest]
        public IEnumerator Kinematics_MatchesIndependentUrdfForwardKinematics()
        {
            Assert.IsTrue(File.Exists(KinematicsPath),
                $"{KinematicsPath} missing - run gen_kinematics_reference.py");
            var root = MiniJson.Parse(File.ReadAllText(KinematicsPath)) as Dictionary<string, object>;
            var bodyOrder = MiniJson.StrArray(root, "bodyOrder");
            var poses = MiniJson.Arr(root, "poses");

            var agent = SpawnCreature(new Vector3(0f, 3f, 0f), ground: false);
            SetZeroGravity(agent);
            agent.enabled = false;                      // no policy, no drive updates
            var rootBody = agent.Root;
            rootBody.immovable = true;                  // isolate pure kinematics
            var joints = agent.Joints;
            var links = new Dictionary<string, Transform>();
            foreach (var b in agent.GetComponentsInChildren<ArticulationBody>(true))
                links[b.name] = b.transform;
            yield return Steps(2);

            float worstAll = 0f;
            string worstWhere = "";
            var lines = new List<string>();

            foreach (var po in poses)
            {
                var p = po as Dictionary<string, object>;
                string poseName = MiniJson.Str(p, "name");
                float[] q = MiniJson.FloatArray(p, "jointPosRad");
                var expected = MiniJson.Arr(p, "linkPosInPelvisIsaac");

                for (int j = 0; j < joints.Count; j++)
                {
                    var b = joints[j];
                    b.jointPosition = new ArticulationReducedSpace(q[j]);
                    b.jointVelocity = new ArticulationReducedSpace(0f);
                    var d = b.xDrive;
                    d.target = q[j] * Mathf.Rad2Deg;    // no restoring torque while we look
                    b.xDrive = d;
                }
                yield return Steps(3);

                float worst = 0f;
                string where = "";
                var pelvis = links["pelvis"];
                for (int i = 0; i < bodyOrder.Length; i++)
                {
                    var e = expected[i] as List<object>;
                    var exp = new Vector3((float)(double)e[0], (float)(double)e[1], (float)(double)e[2]);
                    Vector3 gotUnity = pelvis.InverseTransformPoint(links[bodyOrder[i]].position);
                    Vector3 got = IsaacH1FrameMap.PosToIsaac(gotUnity);
                    float d2 = (got - exp).magnitude;
                    if (d2 > worst) { worst = d2; where = bodyOrder[i]; }
                }
                lines.Add($"    {poseName,-11} worst {worst * 1000f:F4} mm on {where}");
                if (worst > worstAll) { worstAll = worst; worstWhere = $"{poseName}/{where}"; }
            }

            Debug.Log($"[kinematics] Unity rig vs independent Python URDF FK, 3 poses x " +
                      $"{bodyOrder.Length} links:\n" + string.Join("\n", lines) +
                      $"\n    WORST {worstAll * 1000f:F4} mm at {worstWhere}   (gate 1.000 mm)");

            Assert.Less(worstAll, 1e-3f,
                "Unity link positions disagree with an independent URDF FK. That means the " +
                "frame map, an anchor axis, or a joint sign is wrong - fix this before " +
                "believing any dynamics result.");
        }

        // ============================================== rung 1: rest height (strict) ==
        [UnityTest]
        public IEnumerator Rung1_SettlesAtTheExportedRestHeight()
        {
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            agent.commandSpeed = 0f;                    // stand, do not walk
            agent.target = null;
            var sampler = agent.GetComponent<IsaacH1RingTargetSampler>();
            if (sampler != null) sampler.enabled = false;

            yield return Steps(Mathf.CeilToInt(3f / Time.fixedDeltaTime));

            float pelvisY = agent.Root.transform.position.y;
            float comY = agent.CenterOfMassPosition.y;
            // rig_audit.py section E: feet just touch with the pelvis at 0.97837 m; the
            // recording's walking mean is 0.95439 m. Standing should land between them,
            // with slack for drive sag under load.
            const float lo = 0.80f, hi = 1.05f;
            Debug.Log($"[rung 1] after 3 s standing: pelvis y = {pelvisY:F4} m, CoM y = {comY:F4} m\n" +
                      $"    expected band [{lo:F2}, {hi:F2}] m   " +
                      $"(FK rest 0.9784 m, Isaac walking mean 0.9544 m, spawn " +
                      $"{_rig.spawnPosIsaac.z:F4} m)");

            Assert.Greater(pelvisY, lo, "the creature collapsed - legs are not holding it up");
            Assert.Less(pelvisY, hi, "the creature is floating or was launched");
        }

        // ======================================= rung 2: zero-g single joint (strict) ==
        [UnityTest]
        public IEnumerator Rung2_ZeroGravitySingleJointDoesNotMoveTheCentreOfMass()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            agent.actionOverride = IsaacH1Agent.ActionOverride.Constant;
            agent.overrideJointIndex = 11;              // left_knee: heavy subtree, kp 200
            agent.overrideAmplitude = 1f;

            yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));

            float worst = 0f;
            int n = Mathf.CeilToInt(2f / Time.fixedDeltaTime);
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                worst = Mathf.Max(worst, agent.CenterOfMassVelocity.magnitude);
            }

            Debug.Log($"[rung 2] zero-g, driving joint {agent.rig.jointOrder[11]} only, " +
                      $"fdt {Time.fixedDeltaTime:F5}: max |vCoM| = {worst:F5} m/s  (gate 0.02)");
            Assert.Less(worst, 0.02f,
                "the whole-body centre of mass accelerated with no external force. That is " +
                "momentum pumping in the articulation solver, not physics.");
        }

        // ------------------------- rung 2b: which units do drive gains use? -----------
        [UnityTest]
        public IEnumerator Rung2b_GainUnitsCalibration()
        {
            // Unity's drive TARGET is unambiguously in degrees while jointPosition is in
            // radians. If the drive also applies stiffness against a DEGREE error, then
            // feeding Isaac's kp straight in makes every joint 180/pi = 57.3x too stiff.
            // Measure the torque the drive actually produces: from rest, one fixed step of
            // a known position error gives tau = I * (dOmega / dt), so
            //     kp_effective = tau / error_rad.
            // The two hypotheses differ by 57.3x, so even a large error in I cannot
            // confuse them.
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            agent.enabled = false;                 // no policy writing targets
            agent.Root.immovable = true;

            const int J = 17;                      // left_elbow: a leaf, so least chain reaction
            var body = agent.Joints[J];
            float kpNominal = _rig.Body("left_elbow_link").joint.stiffness;
            float I = agent.InertiaAboutJointAxis(J);
            float q0 = _rig.Body("left_elbow_link").joint.defaultPosRad;
            const float err = 0.2f;                // rad, well inside the limits

            body.jointPosition = new ArticulationReducedSpace(q0);
            body.jointVelocity = new ArticulationReducedSpace(0f);
            var d = body.xDrive;
            d.damping = 0f;                        // isolate the stiffness term
            d.forceLimit = 1e9f;                   // do not clip what we are measuring
            d.target = (q0 + err) * Mathf.Rad2Deg;
            body.xDrive = d;
            yield return new WaitForFixedUpdate();

            float w1 = body.jointVelocity[0];
            float alpha = w1 / Time.fixedDeltaTime;
            float tau = alpha * I;
            float kpMeasured = tau / err;

            float ifRadians = kpNominal;                       // tau = kp * err_rad
            float ifDegrees = kpNominal * Mathf.Rad2Deg;       // tau = kp * err_deg
            bool radians = Mathf.Abs(kpMeasured - ifRadians) < Mathf.Abs(kpMeasured - ifDegrees);

            Debug.Log($"[rung 2b] ArticulationDrive gain-unit calibration on {_rig.jointOrder[J]}:\n" +
                      $"    kp (Isaac)            {kpNominal:F1}\n" +
                      $"    I about joint axis    {I:F5} kg.m2 (incl. folded armature)\n" +
                      $"    step position error   {err:F3} rad = {err * Mathf.Rad2Deg:F2} deg\n" +
                      $"    omega after 1 step    {w1:F5} rad/s at dt {Time.fixedDeltaTime:F5}\n" +
                      $"    => torque             {tau:F3} N.m\n" +
                      $"    => kp measured        {kpMeasured:F2}\n" +
                      $"       if gains are RADIAN-based, expect {ifRadians:F2}\n" +
                      $"       if gains are DEGREE-based, expect {ifDegrees:F2}\n" +
                      $"    VERDICT: gains are {(radians ? "RADIAN" : "DEGREE")}-based; " +
                      $"ship GainUnits.{(radians ? "Radians" : "Degrees")} " +
                      $"(currently {agent.gainUnits})");

            Assert.IsTrue(radians == (agent.gainUnits == IsaacH1Agent.GainUnits.Radians),
                $"measured kp {kpMeasured:F2} says the drive is " +
                $"{(radians ? "RADIAN" : "DEGREE")}-based, but the agent ships " +
                $"GainUnits.{agent.gainUnits}. Every joint would be off by 57.3x.");
        }

        // ================== observation sanity at a known state (strict) =============
        [UnityTest]
        public IEnumerator Observations_AreCorrectAtTheSpawnPose()
        {
            // At the spawn pose, standing upright and still, every term of the 69-vector
            // has a value we know independently:
            //   base_lin_vel      ~ 0
            //   base_ang_vel      ~ 0
            //   projected_gravity = (0, 0, -1)      <- upright, Isaac Z-up
            //   velocity_commands = whatever we set
            //   joint_pos (rel)   = 0               <- we are AT the default pose
            //   joint_vel         ~ 0
            //   actions           = 0               <- no policy step has run yet
            // Anything else means a frame-map or indexing error that rungs 1-5 would only
            // show as "it falls over eventually".
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            agent.commandSpeed = 0f;
            agent.target = null;
            var sampler = agent.GetComponent<IsaacH1RingTargetSampler>();
            if (sampler != null) sampler.enabled = false;
            agent.actionOverride = IsaacH1Agent.ActionOverride.Constant;
            agent.overrideAmplitude = 0f;           // hold the default pose exactly
            yield return Steps(4);

            var o = agent.LatestObservation;
            Assert.IsNotNull(o, "no observation was built");

            var sb = new System.Text.StringBuilder();
            sb.Append($"[obs] at the spawn pose, zero-g, no command:" + "\n");
            sb.Append($"    base_lin_vel      {o[0]:F4} {o[1]:F4} {o[2]:F4}   expect ~0" + "\n");
            sb.Append($"    base_ang_vel      {o[3]:F4} {o[4]:F4} {o[5]:F4}   expect ~0" + "\n");
            sb.Append($"    projected_gravity {o[6]:F4} {o[7]:F4} {o[8]:F4}   expect 0 0 -1" + "\n");
            sb.Append($"    velocity_commands {o[9]:F4} {o[10]:F4} {o[11]:F4}" + "\n");
            float maxJp = 0f, maxJv = 0f, maxAct = 0f;
            for (int j = 0; j < 19; j++)
            {
                maxJp = Mathf.Max(maxJp, Mathf.Abs(o[12 + j]));
                maxJv = Mathf.Max(maxJv, Mathf.Abs(o[31 + j]));
                maxAct = Mathf.Max(maxAct, Mathf.Abs(o[50 + j]));
            }
            sb.Append($"    max |joint_pos|   {maxJp:F4}   expect ~0 (we are at the default pose)" + "\n");
            sb.Append($"    max |joint_vel|   {maxJv:F4}   expect ~0" + "\n");
            sb.Append($"    max |action|      {maxAct:F4}   (override holds 0)");
            Debug.Log(sb.ToString());

            Assert.Less(Mathf.Abs(o[6]), 0.05f, "projected_gravity.x should be ~0 while upright");
            Assert.Less(Mathf.Abs(o[7]), 0.05f, "projected_gravity.y should be ~0 while upright");
            Assert.Less(Mathf.Abs(o[8] + 1f), 0.05f,
                "projected_gravity.z should be -1 while upright; if it is +1 the frame map " +
                "or the gravity direction is inverted");
            Assert.Less(maxJp, 0.05f, "joint_pos is relative to the default pose and we are AT it");
        }

        // ================== walking failure diagnostic (informative) =================
        [UnityTest]
        public IEnumerator Diag_WalkTimeSeries()
        {
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            var target = new GameObject("t").transform;
            target.position = new Vector3(0f, 0f, 40f);
            _spawned.Add(target.gameObject);
            agent.target = target;
            agent.commandSpeed = 1f;

            var rows = new List<string>();
            int perSample = Mathf.CeilToInt(0.25f / Time.fixedDeltaTime);
            for (int k = 0; k < 24; k++)
            {
                yield return Steps(perSample);
                var o = agent.LatestObservation;
                float maxJv = 0f, maxAct = 0f;
                int argJv = 0;
                for (int j = 0; j < 19; j++)
                {
                    if (Mathf.Abs(o[31 + j]) > maxJv) { maxJv = Mathf.Abs(o[31 + j]); argJv = j; }
                    maxAct = Mathf.Max(maxAct, Mathf.Abs(agent.LatestAction[j]));
                }
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                var p = agent.Root.transform.position;
                rows.Add($"    t={k * 0.25f,5:F2}  h={p.y,6:F3}  up={up,6:F3}  " +
                         $"z={p.z,7:F3}  vB=({o[0],6:F2},{o[1],6:F2},{o[2],6:F2})  " +
                         $"pg=({o[6],5:F2},{o[7],5:F2},{o[8],5:F2})  " +
                         $"|jv|max={maxJv,6:F2}@{_rig.jointOrder[argJv]}  |a|max={maxAct,5:F2}");
            }

            Debug.Log($"[diag] walking time series, fdt {Time.fixedDeltaTime:F5}, " +
                      $"command vx=1. Isaac recording for comparison: h~0.954, up~1.0, " +
                      $"vB.x~0.90, |jv|max peak 20.3, |a| within a few units.:\n" +
                      string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ============= drive tracking under load, policy OFF (informative) ===========
        [UnityTest]
        public IEnumerator Diag_DriveTrackingUnderGravity()
        {
            // Isolates the actuator from the policy: hold the DEFAULT pose under gravity
            // and measure how far each joint sags from its commanded target. Isaac holds
            // the torso within 5.4 deg of vertical while walking; if the drives sag badly
            // just standing, no policy can recover.
            var rows = new List<string>();
            int[] iters = { 4, 16, 64 };

            foreach (int it in iters)
            {
                SpawnGround();
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
                var go = Object.Instantiate(prefab, new Vector3(0f, _rig.spawnPosIsaac.z, 0f),
                                            Quaternion.identity);
                _spawned.Add(go);
                var agent = go.GetComponent<IsaacH1Agent>();
                agent.enabled = false;                    // NO policy: drives hold the default pose
                foreach (var b in go.GetComponentsInChildren<ArticulationBody>(true))
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
                var pel = agent.Root.transform.position;
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                rows.Add($"    solverIter {it,2}/{it,-2}  pelvis h {pel.y,6:F4} m  upright {up,6:F3}  " +
                         $"worst joint error {worst,7:F4} rad ({worst * Mathf.Rad2Deg,6:F2} deg) " +
                         $"on {_rig.jointOrder[argw],-20} mean|err| {sumAbs / 19f,7:F4} rad");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Debug.Log($"[diag-drive] holding the DEFAULT pose under gravity, policy OFF, " +
                      $"fdt {Time.fixedDeltaTime:F5}.\n" +
                      $"    FK rest height (feet just touching) 0.9784 m; Isaac walking mean 0.9544 m.\n" +
                      string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ============================ rung 3: zero-g square wave (informative) ========
        [UnityTest]
        public IEnumerator Rung3_ZeroGravitySquareWaveMomentumPumping()
        {
            AllowDivergenceLogs();   // pumping is the measurement; being flung is expected
            var rows = new List<string>();
            // Every step here divides policy_dt EXACTLY, so the agent logs no decimation
            // error and the decimation is honest. RIG_AUDIT.md's explicit-PD bound quotes
            // 1/120 and 1/480, but 0.02/(1/120) = 2.4 and 0.02/(1/480) = 9.6 are not
            // integers - 1/100 (dec 2) and 1/500 (dec 10) are the nearest exact divisors,
            // and 1/500 is strictly finer than the 1/480 the audit recommends.
            float[] steps = { ProjectFixedDt, _rig.isaacPhysicsDt, 1f / 100f, 1f / 500f };
            string[] labels = { "project 1/50 drive", "1/200 drive (Isaac)", "1/100 drive", "1/500 torque" };

            for (int c = 0; c < steps.Length; c++)
            {
                Time.fixedDeltaTime = steps[c];
                var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
                SetZeroGravity(agent);
                if (c == 3) { agent.actuatorMode = IsaacH1Agent.ActuatorMode.ExplicitTorquePD; agent.Reconfigure(); }
                agent.actionOverride = IsaacH1Agent.ActionOverride.SquareWave;
                agent.overrideJointIndex = -1;          // bang-bang on EVERY joint
                agent.overrideAmplitude = 1f;
                agent.overrideSquareWavePeriod = 0.4f;

                yield return Steps(Mathf.CeilToInt(0.5f / Time.fixedDeltaTime));

                float worst = 0f, drift;
                Vector3 p0 = agent.CenterOfMassPosition;
                int n = Mathf.CeilToInt(3f / Time.fixedDeltaTime);
                for (int i = 0; i < n; i++)
                {
                    yield return new WaitForFixedUpdate();
                    worst = Mathf.Max(worst, agent.CenterOfMassVelocity.magnitude);
                }
                drift = (agent.CenterOfMassPosition - p0).magnitude;

                rows.Add($"    {labels[c],-22} fdt {steps[c]:F5}  max|vCoM| {worst,8:F4} m/s   " +
                         $"CoM drift over 3 s {drift,7:F4} m");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
            Debug.Log("[rung 3] zero-g bang-bang square wave on all 19 joints. In zero gravity " +
                      "with no contacts, |vCoM| must stay at 0 - anything else is PhysX 4.1 " +
                      "momentum pumping, and it grows with the step:\n" + string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ================================ rung 4: zero-g policy (informative) =========
        [UnityTest]
        public IEnumerator Rung4_ZeroGravityPolicyRuns()
        {
            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            var target = new GameObject("target").transform;
            target.position = new Vector3(0f, 5f, 30f);
            _spawned.Add(target.gameObject);
            agent.target = target;

            yield return Steps(Mathf.CeilToInt(5f / Time.fixedDeltaTime));

            float maxAbsAction = 0f;
            var a = agent.LatestAction;
            for (int i = 0; i < a.Length; i++) maxAbsAction = Mathf.Max(maxAbsAction, Mathf.Abs(a[i]));
            float maxJointVel = 0f;
            for (int i = 0; i < agent.Joints.Count; i++)
                maxJointVel = Mathf.Max(maxJointVel, Mathf.Abs(agent.Joints[i].jointVelocity[0]));

            Debug.Log($"[rung 4] zero-g, policy driving, 5 s at fdt {Time.fixedDeltaTime:F5}:\n" +
                      $"    policy steps      {agent.PolicySteps}  (decimation {agent.Decimation})\n" +
                      $"    max |action|      {maxAbsAction:F3}\n" +
                      $"    max |jointVel|    {maxJointVel:F3} rad/s  " +
                      $"(Isaac recording peak 20.33)\n" +
                      $"    |vCoM|            {agent.CenterOfMassVelocity.magnitude:F4} m/s\n" +
                      $"    command           vx {agent.Command.x:F2} wz {agent.Command.z:F2}");
            Assert.IsFalse(float.IsNaN(maxAbsAction), "policy produced NaN");
            Assert.Pass("informative");
        }

        // ================================== rung 5: locomotion (strict) ==============
        [UnityTest]
        public IEnumerator Rung5_WalksTowardsADistantTarget()
        {
            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            var target = new GameObject("target").transform;
            target.position = new Vector3(0f, 0f, 20f);   // >= 10 m, so closed/time == speed
            _spawned.Add(target.gameObject);
            agent.target = target;
            agent.commandSpeed = 1f;

            Vector3 p0 = agent.Root.transform.position;
            float d0 = Vector2.Distance(new Vector2(p0.x, p0.z),
                                        new Vector2(target.position.x, target.position.z));

            const float duration = 12f;
            yield return Steps(Mathf.CeilToInt(duration / Time.fixedDeltaTime));

            Vector3 p1 = agent.Root.transform.position;
            float d1 = Vector2.Distance(new Vector2(p1.x, p1.z),
                                        new Vector2(target.position.x, target.position.z));
            float closed = d0 - d1;
            float speed = closed / duration;
            float uprightDot = Vector3.Dot(agent.Root.transform.up, Vector3.up);

            Debug.Log($"[rung 5] {duration:F0} s chasing a target {d0:F1} m away, " +
                      $"fdt {Time.fixedDeltaTime:F5}, {agent.actuatorMode}:\n" +
                      $"    distance closed   {closed:F3} m   (gate > 1.0)\n" +
                      $"    mean speed        {speed:F3} m/s\n" +
                      $"    pelvis height     {p1.y:F3} m\n" +
                      $"    upright dot       {uprightDot:F3}   (gate > 0.5)\n" +
                      $"    Isaac reference   0.895 m/s at command (1,0,0); " +
                      $"eval mean {_rig.isaacMeanSpeed:F3} m/s over random commands");

            Assert.Greater(uprightDot, 0.5f, "the creature fell over");
            Assert.Greater(closed, 1.0f, "the creature did not make meaningful progress");
        }

        // ----------------- rung 5 at other steps / actuators (informative) -----------
        [UnityTest]
        public IEnumerator Rung5b_LocomotionAcrossStepsAndActuators()
        {
            var rows = new List<string>();
            // 1/500 (dec 10) rather than the audit's 1/480: it is finer AND divides
            // policy_dt exactly, so the control rate stays a true 50 Hz.
            float[] steps = { ProjectFixedDt, _rig.isaacPhysicsDt, 1f / 500f, 1f / 1000f };
            var modes = new[]
            {
                IsaacH1Agent.ActuatorMode.ArticulationDrive,
                IsaacH1Agent.ActuatorMode.ArticulationDrive,
                IsaacH1Agent.ActuatorMode.ExplicitTorquePD,
                IsaacH1Agent.ActuatorMode.ExplicitTorquePD,
            };
            string[] labels = { "project 1/50 drive", "1/200 drive", "1/500 torque", "1/1000 torque" };

            for (int c = 0; c < steps.Length; c++)
            {
                Time.fixedDeltaTime = steps[c];
                var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
                agent.actuatorMode = modes[c];
                agent.Reconfigure();
                var target = new GameObject("t").transform;
                target.position = new Vector3(0f, 0f, 20f);
                _spawned.Add(target.gameObject);
                agent.target = target;

                Vector3 p0 = agent.Root.transform.position;
                const float duration = 10f;
                yield return Steps(Mathf.CeilToInt(duration / Time.fixedDeltaTime));
                Vector3 p1 = agent.Root.transform.position;

                var goal = new Vector2(target.position.x, target.position.z);
                float closed = Vector2.Distance(new Vector2(p0.x, p0.z), goal)
                             - Vector2.Distance(new Vector2(p1.x, p1.z), goal);
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                rows.Add($"    {labels[c],-20} closed {closed,7:F3} m   speed {closed / duration,6:F3} m/s" +
                         $"   upright {up,6:F3}   height {p1.y,5:F3} m");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
            Debug.Log("[rung 5b] locomotion across steps and actuator models (10 s each):\n" +
                      string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ============================ decimation logging (strict on the message) =====
        [UnityTest]
        public IEnumerator Decimation_LogsAnErrorWhenTheStepDoesNotDividePolicyDt()
        {
            // 0.03 s: policy_dt / fdt = 0.6667, not an integer. The agent must say so, name
            // the exact ratio, and propose the nearest exact divisor - then run anyway.
            Time.fixedDeltaTime = 0.03f;

            // LogAssert.Expect is the only thing that consumes a LogError; ignoreFailingMessages
            // does NOT silence one, so the expectation has to match.
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(
                @"policy_dt / Time\.fixedDeltaTime = 0\.020000 / 0\.030000 = 0\.666\d+, which is NOT an integer"));

            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(3);

            Debug.Log($"[decimation] fdt 0.03 -> ratio {_rig.policyDt / 0.03f:F4}, " +
                      $"agent ran with decimation {agent.Decimation} and logged the error above. " +
                      $"Nearest exact divisor: {_rig.policyDt / Mathf.CeilToInt(_rig.policyDt / 0.03f):F6} s.");
            Assert.AreEqual(1, agent.Decimation);
            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
        }

        [UnityTest]
        public IEnumerator Decimation_IsExactAtTheProjectStep()
        {
            Time.fixedDeltaTime = ProjectFixedDt;
            // An exact integer ratio, but not Isaac's decimation -> warning, never an error.
            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex(
                @"is an exact integer.*Isaac ran decimation 4"));

            var agent = SpawnCreature(new Vector3(0f, 5f, 0f), ground: false);
            SetZeroGravity(agent);
            yield return Steps(3);

            Assert.AreEqual(1, agent.Decimation,
                "at the project step of 0.02 s the policy runs every fixed step");
            Debug.Log($"[decimation] project step {ProjectFixedDt:F5}: ratio " +
                      $"{_rig.policyDt / ProjectFixedDt:F4} -> decimation {agent.Decimation}, " +
                      "no LogError. Isaac used decimation 4 at 0.005 s.");
            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
        }

        // ================================ rung 6: speed parity (informative) =========
        [UnityTest]
        public IEnumerator Rung6_SpeedParityAgainstIsaac()
        {
            const float IsaacSpeedAtCommand1 = 0.8954f;   // measured from the recording

            var agent = SpawnCreature(new Vector3(0f, _rig.spawnPosIsaac.z, 0f));
            var target = new GameObject("t").transform;
            target.position = new Vector3(0f, 0f, 40f);
            _spawned.Add(target.gameObject);
            agent.target = target;
            agent.commandSpeed = 1f;

            yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));   // settle
            Vector3 p0 = agent.Root.transform.position;
            const float duration = 15f;
            yield return Steps(Mathf.CeilToInt(duration / Time.fixedDeltaTime));
            Vector3 p1 = agent.Root.transform.position;

            float travelled = Vector2.Distance(new Vector2(p0.x, p0.z), new Vector2(p1.x, p1.z));
            float speed = travelled / duration;
            float parity = speed / IsaacSpeedAtCommand1;

            Debug.Log($"[rung 6] speed parity at command vx = 1.0:\n" +
                      $"    Unity  {speed:F3} m/s over {duration:F0} s ({travelled:F2} m)\n" +
                      $"    Isaac  {IsaacSpeedAtCommand1:F3} m/s (250-step recording, same command)\n" +
                      $"    parity {parity:P1}   " +
                      (parity < 0.5f
                          ? "BELOW 50% - run Rung6b_ConfigurationSweep and read the table"
                          : "at or above 50%"));
            Assert.Pass("informative");
        }

        // --------------------- rung 6b: configuration sweep (informative) ------------
        [UnityTest]
        public IEnumerator Rung6b_ConfigurationSweep()
        {
            var rows = new List<string>();

            // Each entry mutates one variable away from the shipped configuration.
            var cases = new (string label, System.Action<IsaacH1Agent, GameObject> apply)[]
            {
                ("shipped (box foot, min combine, offset 0.02, no floor, drive, armature None)", null),
                ("foot box -> sole-only thin box", (a, g) => ResizeFeet(a, 0.02f)),
                ("ground material: Average combine", (a, g) => SetGroundCombine(g, PhysicsMaterialCombine.Average)),
                ("ground material: Multiply combine (Isaac's own)", (a, g) => SetGroundCombine(g, PhysicsMaterialCombine.Multiply)),
                ("contact offset 0.01 (Unity project default)", (a, g) => SetContactOffset(a, g, 0.01f)),
                ("inertia floor ON (1e-4)", (a, g) => { a.applyInertiaFloor = true; }),
                ("armature: None (shipped)", (a, g) => { a.armatureMode = IsaacH1Agent.ArmatureMode.None; }),
                ("armature: FoldIntoInertia (naive)", (a, g) => { a.armatureMode = IsaacH1Agent.ArmatureMode.FoldIntoInertia; }),
                ("armature: FoldDistalOnly (exact for parallel runs)", (a, g) => { a.armatureMode = IsaacH1Agent.ArmatureMode.FoldDistalOnly; }),
                ("torso mass = the recording's 15.333 kg", (a, g) => { a.torsoMassScale = 0.8619f; }),
            };

            foreach (var (label, apply) in cases)
            {
                var ground = SpawnGround();
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
                var go = Object.Instantiate(prefab, new Vector3(0f, _rig.spawnPosIsaac.z, 0f),
                                            Quaternion.identity);
                _spawned.Add(go);
                var agent = go.GetComponent<IsaacH1Agent>();
                apply?.Invoke(agent, ground);
                agent.Reconfigure();   // Awake already ran; re-apply the mutated fields

                var target = new GameObject("t").transform;
                target.position = new Vector3(0f, 0f, 40f);
                _spawned.Add(target.gameObject);
                agent.target = target;

                yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));
                Vector3 p0 = agent.Root.transform.position;
                const float duration = 10f;
                yield return Steps(Mathf.CeilToInt(duration / Time.fixedDeltaTime));
                Vector3 p1 = agent.Root.transform.position;

                float speed = Vector2.Distance(new Vector2(p0.x, p0.z), new Vector2(p1.x, p1.z)) / duration;
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                rows.Add($"    {label,-52} {speed,6:F3} m/s  upright {up,6:F3}  " +
                         $"h {p1.y,5:F3}  parity {speed / 0.8954f,6:P0}");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Debug.Log("[rung 6b] configuration sweep, 10 s each, one variable at a time " +
                      "(Isaac reference 0.895 m/s):\n" + string.Join("\n", rows));
            Assert.Pass("informative");
        }

        static void ResizeFeet(IsaacH1Agent a, float height)
        {
            foreach (var name in new[] { "left_ankle_link", "right_ankle_link" })
            {
                var t = FindDeep(a.transform, name);
                if (t == null) continue;
                var bc = t.GetComponent<BoxCollider>();
                if (bc == null) continue;
                float bottom = bc.center.y - bc.size.y * 0.5f;
                var s = bc.size; s.y = height; bc.size = s;
                var c = bc.center; c.y = bottom + height * 0.5f; bc.center = c;
            }
        }

        static void SetGroundCombine(GameObject ground, PhysicsMaterialCombine combine)
        {
            var col = ground.GetComponent<Collider>();
            var m = new PhysicsMaterial("sweep")
            {
                staticFriction = 1f,
                dynamicFriction = 1f,
                frictionCombine = combine,
            };
            col.sharedMaterial = m;
        }

        static void SetContactOffset(IsaacH1Agent a, GameObject ground, float offset)
        {
            foreach (var c in a.GetComponentsInChildren<Collider>(true)) c.contactOffset = offset;
            ground.GetComponent<Collider>().contactOffset = offset;
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

        // ====== can the PROJECT step be rescued without a global change? (informative) =
        [UnityTest]
        public IEnumerator Diag_ProjectStepRescueAttempts()
        {
            // rung 5b shows the creature walks at 1/200 but falls at this project's
            // 1/50. Before proposing a global fixedDeltaTime change, try everything that
            // is PER-BODY and therefore shippable without touching project settings.
            AllowDivergenceLogs();   // the deliberately-inadequate solver cases fall apart
            var rows = new List<string>();
            var cases = new (string label, System.Action<IsaacH1Agent> apply)[]
            {
                ("baseline (solver 4/4, as Isaac)", null),
                ("solver 16/16", a => SetSolver(a, 16, 16)),
                ("solver 32/32", a => SetSolver(a, 32, 32)),
                ("solver 48/48", a => SetSolver(a, 48, 48)),
                ("solver 64/64", a => SetSolver(a, 64, 64)),
                ("solver 96/96", a => SetSolver(a, 96, 96)),
                ("solver 128/128", a => SetSolver(a, 128, 128)),
            };

            foreach (var (label, apply) in cases)
            {
                Time.fixedDeltaTime = ProjectFixedDt;
                SpawnGround();
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
                var go = Object.Instantiate(prefab, new Vector3(0f, _rig.spawnPosIsaac.z, 0f),
                                            Quaternion.identity);
                _spawned.Add(go);
                var agent = go.GetComponent<IsaacH1Agent>();
                apply?.Invoke(agent);
                var target = new GameObject("t").transform;
                target.position = new Vector3(0f, 0f, 40f);
                _spawned.Add(target.gameObject);
                agent.target = target;

                yield return Steps(Mathf.CeilToInt(1f / Time.fixedDeltaTime));
                Vector3 p0 = agent.Root.transform.position;
                const float dur = 20f;   // long enough to expose a marginal survivor
                yield return Steps(Mathf.CeilToInt(dur / Time.fixedDeltaTime));
                Vector3 p1 = agent.Root.transform.position;
                float sp = Vector2.Distance(new Vector2(p0.x, p0.z), new Vector2(p1.x, p1.z)) / dur;
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                rows.Add($"    {label,-38} {sp,6:F3} m/s  upright {up,6:F3}  h {p1.y,5:F3}  " +
                         $"parity {sp / 0.8954f,5:P0}");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Time.fixedDeltaTime = _rig.isaacPhysicsDt;
            Debug.Log($"[diag-projectstep] locomotion at THIS PROJECT'S step " +
                      $"({ProjectFixedDt:F5} s, decimation 1), 20 s each. Everything tried " +
                      $"here is a PER-BODY override - no project setting is touched.\n" +
                      $"    For reference, the same run at 1/200 (Isaac's step) gives " +
                      $"1.001 m/s upright 0.999.\n" +
                      string.Join("\n", rows));
            Assert.Pass("informative");
        }

        static void SetSolver(IsaacH1Agent a, int pos, int vel)
        {
            foreach (var b in a.GetComponentsInChildren<ArticulationBody>(true))
            { b.solverIterations = pos; b.solverVelocityIterations = vel; }
        }

        static void SetDepen(IsaacH1Agent a, float v)
        {
            foreach (var b in a.GetComponentsInChildren<ArticulationBody>(true))
                b.maxDepenetrationVelocity = v;
        }

        // ============== turning / sustained-run robustness (informative) =============
        [UnityTest, Timeout(900000)]   // 4 x 60 s runs exceed the 180 s default
        public IEnumerator Diag_TurningAndSustainedRun()
        {
            // rung 5 walks toward a FIXED target, i.e. in a straight line. The shipped
            // scene uses the ring sampler, which snaps to a new random point every 10 s
            // and can demand a near-max yaw rate at full forward speed - the corner of the
            // trained command distribution. Playing the scene showed a fall after ~13 m.
            // This measures time-to-first-fall over a long run and whether backing off
            // either half of that command keeps it upright.
            var rows = new List<string>();
            var cases = new (string label, System.Action<IsaacH1Agent> apply)[]
            {
                ("ring, turnSlowdown 0 + no recovery (old)", a => { a.turnSlowdown = 0f; a.autoRecoverFromFalls = false; }),
                ("ring, turnSlowdown 0 + recovery ON", a => { a.turnSlowdown = 0f; }),
                ("ring, SHIPPED (turnSlowdown 0.5 + recovery)", null),
                ("fixed straight target (rung 5 style)", a => { }),
            };

            for (int c = 0; c < cases.Length; c++)
            {
                SpawnGround();
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
                var go = Object.Instantiate(prefab, new Vector3(0f, _rig.spawnPosIsaac.z, 0f),
                                            Quaternion.identity);
                _spawned.Add(go);
                var agent = go.GetComponent<IsaacH1Agent>();
                var sampler = go.GetComponent<IsaacH1RingTargetSampler>();
                if (c == 3)
                {
                    if (sampler != null) sampler.enabled = false;
                    var t = new GameObject("t").transform;
                    t.position = new Vector3(0f, 0f, 300f);
                    _spawned.Add(t.gameObject);
                    agent.target = t;
                }
                else if (sampler != null)
                {
                    sampler.seed = 20260828;   // same draws for every ring case
                }
                cases[c].apply?.Invoke(agent);
                agent.Reconfigure();

                float tFall = -1f, dist = 0f, maxYaw = 0f;
                Vector3 prev = agent.Root.transform.position;
                int sample = Mathf.CeilToInt(0.1f / Time.fixedDeltaTime);
                int nSamples = Mathf.CeilToInt(60f / (sample * Time.fixedDeltaTime));
                for (int i = 0; i < nSamples; i++)
                {
                    yield return Steps(sample);
                    var p = agent.Root.transform.position;
                    dist += new Vector2(p.x - prev.x, p.z - prev.z).magnitude;
                    prev = p;
                    maxYaw = Mathf.Max(maxYaw, Mathf.Abs(agent.Command.z));
                    if (tFall < 0f && Vector3.Dot(agent.Root.transform.up, Vector3.up) < 0.5f)
                        tFall = i * sample * Time.fixedDeltaTime;
                }
                float up = Vector3.Dot(agent.Root.transform.up, Vector3.up);
                rows.Add($"    {cases[c].label,-44} " +
                         (tFall < 0f ? "no fall in 60 s" : $"first fall t={tFall,5:F1} s") +
                         $"   travelled {dist,6:F1} m   recoveries {agent.Recoveries,2}   " +
                         $"final upright {up,5:F2}   max|wz| {maxYaw,4:F2}");

                foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
                _spawned.Clear();
                yield return null;
            }

            Debug.Log($"[diag-turn] 60 s runs at fdt {Time.fixedDeltaTime:F5}. Isaac's own eval " +
                      $"reports {_rig.isaacFallsPerRobotPerMinute:F3} falls/robot/minute, i.e. " +
                      $"about one fall per 8 minutes.:\n" + string.Join("\n", rows));
            Assert.Pass("informative");
        }

        // ==================================== perf: N creatures (informative) ========
        [UnityTest]
        public IEnumerator Perf_EightCreaturesAtTheProjectStep()
        {
            yield return PerfRun(ProjectFixedDt, "PROJECT step (shipping config)");
            yield return PerfRun(_rig.isaacPhysicsDt, "Isaac step 1/200");
            // This project runs TGS with 12/4 defaults, so Diag_ProjectStepRescueAttempts
            // finds Isaac's own 4/4 already walks at 0.02 s (0.969 m/s, upright 0.992) -
            // unlike the PGS project this rig was first built in, which needed >= 48.
            // 4/4 is 16x less solver work per body, and the shipping config lands exactly
            // on the 60 FPS budget, so measure what IsaacExact buys.
            yield return PerfRun(ProjectFixedDt, "PROJECT step, solver IsaacExact 4/4",
                                 IsaacH1Agent.SolverIterationMode.IsaacExact);
        }

        IEnumerator PerfRun(float fdt, string label,
                            IsaacH1Agent.SolverIterationMode? solverMode = null)
        {
            const int N = 8;
            Time.fixedDeltaTime = fdt;
            SpawnGround();

            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
            var agents = new List<IsaacH1Agent>();
            for (int i = 0; i < N; i++)
            {
                var go = Object.Instantiate(prefab,
                    new Vector3((i % 4) * 2.5f - 3.75f, _rig.spawnPosIsaac.z, (i / 4) * 2.5f),
                    Quaternion.identity);
                _spawned.Add(go);
                var a = go.GetComponent<IsaacH1Agent>();
                var t = new GameObject($"t{i}").transform;
                t.position = new Vector3(0f, 0f, 40f);
                _spawned.Add(t.gameObject);
                a.target = t;
                if (solverMode.HasValue)
                {
                    a.solverIterationMode = solverMode.Value;
                    a.Reconfigure();
                }
                agents.Add(a);
            }

            yield return Steps(Mathf.CeilToInt(2f / Time.fixedDeltaTime));   // warm up

            int frames = Mathf.CeilToInt(6f / Time.fixedDeltaTime);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < frames; i++) yield return new WaitForFixedUpdate();
            sw.Stop();

            double msPerStep = sw.Elapsed.TotalMilliseconds / frames;
            double simSeconds = frames * Time.fixedDeltaTime;
            double realtimeFactor = simSeconds / sw.Elapsed.TotalSeconds;
            double msPerRenderedFrame = msPerStep * (1.0 / 60.0) / Time.fixedDeltaTime;
            double budgetMs = 1000.0 / 60.0;
            int standing = 0;
            foreach (var a in agents)
                if (Vector3.Dot(a.Root.transform.up, Vector3.up) > 0.5f) standing++;

            var a0 = agents[0];
            Debug.Log($"[perf] {label}: {N} creatures, CPU backend, fdt {Time.fixedDeltaTime:F5}, " +
                      $"solverIter {a0.SolverIterationsInUse}, decimation {a0.Decimation}, " +
                      $"{frames} fixed steps:\n" +
                      $"    wall time per fixed step  {msPerStep:F3} ms\n" +
                      $"    physics cost per 60 FPS frame {msPerRenderedFrame:F3} ms " +
                      $"(= {1.0 / 60.0 / Time.fixedDeltaTime:F2} fixed steps per frame)\n" +
                      $"    60 FPS budget             {budgetMs:F3} ms\n" +
                      $"    headroom                  {budgetMs / msPerRenderedFrame:F2}x   " +
                      $"({(msPerRenderedFrame < budgetMs ? "within budget" : "OVER budget")})\n" +
                      $"    real-time factor          {realtimeFactor:F2}x\n" +
                      $"    still upright after 8 s   {standing}/{N}\n" +
                      "    NOTE: measured inside the editor with the test runner attached, " +
                      "so this is a floor, not a shipping number.");

            foreach (var g in _spawned) if (g != null) Object.DestroyImmediate(g);
            _spawned.Clear();
            yield return null;
        }
    }
}
