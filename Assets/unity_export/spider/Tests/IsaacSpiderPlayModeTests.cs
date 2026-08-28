using System.Collections;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor;
#endif
using Debug = UnityEngine.Debug;

namespace IsaacSpider.Tests
{
    /// <summary>
    /// The stability-triage ladder from README_UNITY.md as PlayMode tests. Spawns the prefab (or builds the
    /// rig from the URDF when the prefab is missing) into the test runner's own empty scene. Rungs 0/1/2/5
    /// are strict gates; substep / actuator variants are informative and log their numbers.
    /// </summary>
    public sealed class IsaacSpiderPlayModeTests
    {
        private const string PREFAB_PATH = "Assets/unity_export/spider/IsaacSpider.prefab";
        private const string URDF_PATH = "Assets/unity_export/spider/robot/spider.urdf";
        private const string ONNX_PATH = "Assets/unity_export/spider/spider.onnx";
        private const string REFERENCE_PATH = "Assets/unity_export/spider/isaac_reference.json";

        private float _savedFixedDelta;
        private Vector3 _savedGravity;
        private GameObject _ground;
        private readonly System.Collections.Generic.List<GameObject> _spawned = new();
        private readonly float[] _obs = new float[IsaacSpiderAgent.OBS_DIM];

        [SetUp]
        public void SetUp()
        {
            _savedFixedDelta = Time.fixedDeltaTime;
            _savedGravity = Physics.gravity;
            Physics.gravity = new Vector3(0f, -9.81f, 0f);
            // 1/60 s is the coarsest step that divides policy_dt exactly; rungs that want another step set it themselves.
            Time.fixedDeltaTime = 1f / 60f;
        }

        private static void ExpectDecimationError()
        {
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex("not an integer multiple of Time.fixedDeltaTime"));
        }

        [TearDown]
        public void TearDown()
        {
            for (int index = 0; index < _spawned.Count; index++)
            {
                if (_spawned[index] != null)
                {
                    Object.Destroy(_spawned[index]);
                }
            }
            _spawned.Clear();
            if (_ground != null)
            {
                Object.Destroy(_ground);
            }
            Time.fixedDeltaTime = _savedFixedDelta;
            Physics.gravity = _savedGravity;
            LogAssert.ignoreFailingMessages = false;
        }

        // ------------------------------------------------------------------ rung 0
        [UnityTest]
        public IEnumerator Rung0_OnnxInEngineMatchesIsaacRecording()
        {
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 1f, 0f), IsaacSpiderAgent.ActuatorMode.Off, true);
            yield return null;
            if (!agent.HasPolicy)
            {
                Assert.Inconclusive("No ModelAsset on the prefab (spider.onnx not imported?).");
            }
            float worst = agent.RunReferenceCheck(out int steps);
            Debug.Log($"[Rung0] {steps} steps, max |onnx - isaac| = {worst:E2}");
            Assert.That(steps, Is.EqualTo(200));
            Assert.That(worst, Is.LessThan(1e-4f));
        }

        // ------------------------------------------------------------------ frame / sign check
        [UnityTest]
        public IEnumerator Kinematics_JointSignAndFrameMapMatchIsaacFk()
        {
            IsaacSpiderAgent agent = Spawn(Vector3.zero, IsaacSpiderAgent.ActuatorMode.Off, true);
            agent.Root.immovable = true;
            yield return new WaitForFixedUpdate();
            // Expected Unity positions of the L1 tibia collider centre from an independent Python FK of the URDF (rig_audit.py conventions).
            yield return CheckPose(agent, 0.4f, 0.5f, new Vector3(-0.15203f, -0.03818f, 0.15879f));
            yield return CheckPose(agent, -0.6f, -0.8f, new Vector3(-0.02537f, 0.05998f, 0.33431f));
            yield return CheckPose(agent, 0f, 0f, new Vector3(-0.14305f, -0.0208f, 0.24773f));
        }

        private IEnumerator CheckPose(IsaacSpiderAgent agent, float hip, float knee, Vector3 expected)
        {
            agent.JointBodies[0].jointPosition = new ArticulationReducedSpace(hip);
            agent.JointBodies[1].jointPosition = new ArticulationReducedSpace(knee);
            agent.JointBodies[0].jointVelocity = new ArticulationReducedSpace(0f);
            agent.JointBodies[1].jointVelocity = new ArticulationReducedSpace(0f);
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            Transform collider = agent.JointBodies[1].transform.Find("col_L1_tibia");
            Assert.That(collider, Is.Not.Null, "tibia collider holder missing");
            Vector3 actual = collider.position - agent.Root.transform.position;
            Debug.Log($"[Kinematics] hip={hip} knee={knee}: expected {expected:F4} actual {actual:F4} (Δ {(actual - expected).magnitude * 1000f:0.0} mm)");
            Assert.That((actual - expected).magnitude, Is.LessThan(0.005f), $"hip={hip} knee={knee}");
        }

        // ------------------------------------------------------------------ rung 1
        [UnityTest]
        public IEnumerator Rung1_RestHeightWithDrivesHoldingZeroPose()
        {
            CreateGround();
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 0.18f, 0f), IsaacSpiderAgent.ActuatorMode.ArticulationDrive, false);
            agent.SetActionOverride(IsaacSpiderAgent.ActionOverrideMode.Constant, Vector4.zero, 1f, -1);
            yield return new WaitForSeconds(3f);
            float height = agent.BodyHeight;
            Debug.Log($"[Rung1] body height after 3 s at dt={Time.fixedDeltaTime:0.00000}: {height:0.000} m (Isaac standing height 0.141 m)  |vCoM| = {agent.CenterOfMassVelocity().magnitude:0.000}");
            Assert.That(height, Is.InRange(0.115f, 0.17f));
            Assert.That(agent.CenterOfMassVelocity().magnitude, Is.LessThan(0.05f));
        }

        [UnityTest]
        public IEnumerator Rung1_Informative_FreeJointsRestHeight()
        {
            CreateGround();
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 0.18f, 0f), IsaacSpiderAgent.ActuatorMode.Off, false);
            yield return new WaitForSeconds(3f);
            Debug.Log($"[Rung1-free] drives off: body height {agent.BodyHeight:0.000} m, |vCoM| {agent.CenterOfMassVelocity().magnitude:0.000}");
            Assert.That(agent.BodyHeight, Is.GreaterThan(0.0f));
        }

        // ------------------------------------------------------------------ rung 2
        [UnityTest]
        public IEnumerator Rung2_ZeroG_SingleJointStep_NoCoMMomentum_Drive()
        {
            yield return ZeroGSingleJoint(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, Time.fixedDeltaTime, true);
        }

        [UnityTest]
        public IEnumerator Rung2_Informative_Torque_1_480()
        {
            yield return ZeroGSingleJoint(IsaacSpiderAgent.ActuatorMode.TorqueCSharp, 1f / 480f, false);
        }

        [UnityTest]
        public IEnumerator Rung2_Informative_Torque_1_240()
        {
            yield return ZeroGSingleJoint(IsaacSpiderAgent.ActuatorMode.TorqueCSharp, 1f / 240f, false);
        }

        private IEnumerator ZeroGSingleJoint(IsaacSpiderAgent.ActuatorMode mode, float dt, bool strict)
        {
            Time.fixedDeltaTime = dt;
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 1f, 0f), mode, true);
            agent.SetActionOverride(IsaacSpiderAgent.ActionOverrideMode.Constant, new Vector4(0.8f, 0.8f, 0f, 0f), 1f, 1);
            Vector3 start = agent.Root.transform.position;
            yield return new WaitForSeconds(2f);
            Vector3 com = agent.CenterOfMassVelocity();
            float drift = (agent.Root.transform.position - start).magnitude;
            float knee = agent.ReadJointPosition(1);
            Debug.Log($"[Rung2 {mode} dt={dt:0.00000}] |vCoM| = {com.magnitude:0.0000} m/s, root drift {drift:0.0000} m, knee = {knee:0.000} rad (target 0.64), max|q̇| {agent.MaxJointSpeedSeen:0.0}");
            Assert.That(float.IsFinite(com.magnitude), "CoM velocity is NaN/inf");
            if (strict)
            {
                Assert.That(com.magnitude, Is.LessThan(0.02f), "a joint torque must not create momentum");
                Assert.That(drift, Is.LessThan(0.03f));
                Assert.That(knee, Is.EqualTo(0.64f).Within(0.05f), "drive should reach the target");
            }
        }

        // ------------------------------------------------------------------ rung 3
        [UnityTest]
        public IEnumerator Rung3_ZeroG_SquareWaveAllJoints_Drive()
        {
            yield return ZeroGSquareWave(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, Time.fixedDeltaTime);
        }

        [UnityTest]
        public IEnumerator Rung3_Informative_Torque_1_480()
        {
            yield return ZeroGSquareWave(IsaacSpiderAgent.ActuatorMode.TorqueCSharp, 1f / 480f);
        }

        [UnityTest]
        public IEnumerator Rung3_Informative_Drive_1_120()
        {
            yield return ZeroGSquareWave(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, 1f / 120f);
        }

        [UnityTest]
        public IEnumerator Rung3_Informative_Drive_ProjectDefault_0_02()
        {
            yield return ZeroGSquareWave(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, 0.02f);
        }

        private IEnumerator ZeroGSquareWave(IsaacSpiderAgent.ActuatorMode mode, float dt)
        {
            Time.fixedDeltaTime = dt;
            if (Mathf.Abs(dt - 0.02f) < 1e-6f)
            {
                ExpectDecimationError();
            }
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 1f, 0f), mode, true);
            agent.SetActionOverride(IsaacSpiderAgent.ActionOverrideMode.SquareWave, new Vector4(0.8f, 0.8f, -0.8f, -0.8f), 0.5f, -1);
            yield return new WaitForSeconds(3f);
            Vector3 com = agent.CenterOfMassVelocity();
            Debug.Log($"[Rung3 {mode} dt={dt:0.00000}] |vCoM| = {com.magnitude:0.0000} m/s, height {agent.BodyHeight:0.000}, max|q̇| {agent.MaxJointSpeedSeen:0.0} rad/s");
            Assert.That(float.IsFinite(com.magnitude) && float.IsFinite(agent.BodyHeight), "diverged");
            Assert.That(com.magnitude, Is.LessThan(0.5f), "bang-bang joints created macroscopic momentum");
        }

        // ------------------------------------------------------------------ rung 4
        [UnityTest]
        public IEnumerator Rung4_Informative_ZeroG_FullPolicy()
        {
            Time.fixedDeltaTime = 1f / 60f;
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 1f, 0f), IsaacSpiderAgent.ActuatorMode.ArticulationDrive, true);
            if (!agent.HasPolicy)
            {
                Assert.Inconclusive("no policy");
            }
            yield return new WaitForSeconds(3f);
            Vector3 com = agent.CenterOfMassVelocity();
            Debug.Log($"[Rung4] zero-g full policy 3 s: |vCoM| = {com.magnitude:0.0000}, height {agent.BodyHeight:0.000}, max|q̇| {agent.MaxJointSpeedSeen:0.0}, decimation {agent.Decimation}");
            Assert.That(float.IsFinite(com.magnitude) && float.IsFinite(agent.BodyHeight), "diverged");
        }

        // ------------------------------------------------------------------ rung 5
        [UnityTest]
        public IEnumerator Rung5_Gravity_PolicyWalksToTarget_Drive_1_60()
        {
            yield return WalkToTarget(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, 1f / 60f, true);
        }

        [UnityTest]
        public IEnumerator Rung5_Informative_Drive_ProjectDefault_0_02()
        {
            yield return WalkToTarget(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, 0.02f, false);
        }

        [UnityTest]
        public IEnumerator Rung5_Informative_Torque_1_480()
        {
            yield return WalkToTarget(IsaacSpiderAgent.ActuatorMode.TorqueCSharp, 1f / 480f, false);
        }

        [UnityTest]
        public IEnumerator Rung5_Informative_Drive_1_120_IsaacStep()
        {
            yield return WalkToTarget(IsaacSpiderAgent.ActuatorMode.ArticulationDrive, 1f / 120f, false);
        }

        private IEnumerator WalkToTarget(IsaacSpiderAgent.ActuatorMode mode, float dt, bool strict)
        {
            Time.fixedDeltaTime = dt;
            CreateGround();
            var targetGo = new GameObject("Target");
            _spawned.Add(targetGo);
            targetGo.transform.position = new Vector3(0f, 0.12f, 10f); // far enough that the spider never arrives inside the window: closed / time = speed
            if (Mathf.Abs(dt - 0.02f) < 1e-6f)
            {
                ExpectDecimationError();
            }
            IsaacSpiderAgent agent = Spawn(new Vector3(0f, 0.18f, 0f), mode, false);
            if (!agent.HasPolicy)
            {
                Assert.Inconclusive("no policy");
            }
            agent.SetTarget(targetGo.transform);
            float startDistance = PlanarDistance(agent.Root.transform.position, targetGo.transform.position);
            const float SECONDS = 8f;
            float minHeight = float.MaxValue;
            float elapsed = 0f;
            while (elapsed < SECONDS)
            {
                yield return new WaitForFixedUpdate();
                elapsed += Time.fixedDeltaTime;
                minHeight = Mathf.Min(minHeight, agent.BodyHeight);
                if (!float.IsFinite(agent.BodyHeight) || agent.BodyHeight > 5f)
                {
                    break;
                }
            }
            agent.CopyObservation(_obs);
            float endDistance = PlanarDistance(agent.Root.transform.position, targetGo.transform.position);
            float closed = startDistance - endDistance;
            Debug.Log($"[Rung5 {mode} dt={dt:0.00000} decim={agent.Decimation}] closed {closed:0.00} m of {startDistance:0.00} in {elapsed:0.0} s " +
                      $"(≈{closed / elapsed:0.00} m/s; Isaac eval 2.9 m/s), end height {agent.BodyHeight:0.000}, min height {minHeight:0.000}, upright g_z={_obs[40]:0.00}, max|q̇| {agent.MaxJointSpeedSeen:0.0}");
            Assert.That(float.IsFinite(agent.BodyHeight) && agent.BodyHeight < 5f, "diverged / flew");
            if (strict)
            {
                Assert.That(_obs[40], Is.LessThan(-0.5f), "spider is not upright");
                Assert.That(agent.BodyHeight, Is.GreaterThan(0.06f), "collapsed");
                Assert.That(closed, Is.GreaterThan(1.0f), "did not walk toward the target");
            }
        }

        // ------------------------------------------------------------------ perf (informative)
        [UnityTest]
        public IEnumerator Perf_Informative_EightSpidersAt1_60()
        {
            Time.fixedDeltaTime = 1f / 60f;
            CreateGround();
            for (int index = 0; index < 8; index++)
            {
                IsaacSpiderAgent agent = Spawn(new Vector3((index % 4) * 1.5f - 2.25f, 0.18f, (index / 4) * 1.5f), IsaacSpiderAgent.ActuatorMode.ArticulationDrive, false);
                agent.ShowGui = false;
            }
            yield return new WaitForSeconds(1f);
            var watch = Stopwatch.StartNew();
            int frames = 0;
            float start = Time.realtimeSinceStartup;
            while (Time.realtimeSinceStartup - start < 3f)
            {
                yield return null;
                frames++;
            }
            watch.Stop();
            Debug.Log($"[Perf] 8 spiders, drive, policy, dt=1/60: {frames} frames in {watch.ElapsedMilliseconds} ms = {(float)watch.ElapsedMilliseconds / frames:0.00} ms/frame (editor, includes test-runner overhead)");
            Assert.That(frames, Is.GreaterThan(0));
        }

        // ------------------------------------------------------------------ helpers
        private static float PlanarDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f;
            b.y = 0f;
            return Vector3.Distance(a, b);
        }

        private void CreateGround()
        {
            if (_ground != null)
            {
                return;
            }
            // Same construction as the race track: a plane with a solid slab collider whose top is y = 0 and no physics material (Unity default 0.6/0.6).
            _ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            _ground.name = "TestGround";
            _ground.transform.localScale = new Vector3(4f, 1f, 4f);
            Object.Destroy(_ground.GetComponent<MeshCollider>());
            var slab = _ground.AddComponent<BoxCollider>();
            slab.size = new Vector3(10f, 1f, 10f);
            slab.center = new Vector3(0f, -0.5f, 0f);
        }

        private IsaacSpiderAgent Spawn(Vector3 position, IsaacSpiderAgent.ActuatorMode mode, bool zeroGravity)
        {
            GameObject instance = null;
#if UNITY_EDITOR
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PREFAB_PATH);
            if (prefab != null)
            {
                instance = Object.Instantiate(prefab, position, Quaternion.identity);
            }
            else
            {
                instance = BuildFromUrdf(position);
            }
#endif
            Assert.That(instance, Is.Not.Null, "could not spawn the spider (prefab missing and URDF fallback unavailable)");
            _spawned.Add(instance);
            IsaacSpiderAgent agent = instance.GetComponent<IsaacSpiderAgent>();
            Assert.That(agent.IsReady, "agent failed to bind its rig (see console)");
            agent.ShowGui = false;
            agent.Actuator = mode;
            agent.ZeroGravity = zeroGravity;
            return agent;
        }

#if UNITY_EDITOR
        private static GameObject BuildFromUrdf(Vector3 position)
        {
            if (!File.Exists(URDF_PATH))
            {
                return null;
            }
            GameObject root = IsaacSpiderRigBuilder.Build(File.ReadAllText(URDF_PATH), "IsaacSpider(built)", null);
            root.SetActive(false);
            root.transform.position = position;
            var agent = root.AddComponent<IsaacSpiderAgent>();
            var serialized = new SerializedObject(agent);
#if ISAAC_SPIDER_INFERENCE
            serialized.FindProperty("_model").objectReferenceValue = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(ONNX_PATH);
#endif
            serialized.FindProperty("_isaacReference").objectReferenceValue = AssetDatabase.LoadAssetAtPath<TextAsset>(REFERENCE_PATH);
            serialized.ApplyModifiedPropertiesWithoutUndo();
            root.SetActive(true);
            return root;
        }
#endif
    }
}
