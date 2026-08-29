using System.Collections;
using System.Collections.Generic;
using System.Text;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace MujocoBiped.Tests
{
    /// <summary>
    /// The triage ladder for the MujocoBiped port, rungs 0 to 6.
    ///
    /// Each rung isolates one layer, so a failure names its own cause instead of leaving
    /// you to guess between "the model is wrong", "the frame map is wrong" and "PhysX is
    /// unstable":
    ///
    ///   0  inference       the ONNX in Unity vs the actions MuJoCo recorded
    ///   K  kinematics      the frame map, against an independent Python FK of the MJCF
    ///   1  statics         does it stand where MuJoCo says it stands
    ///   2  momentum        zero-g, one joint: internal torque must not move the CoM
    ///   3  stability       zero-g, square wave, across candidate timesteps
    ///   4  policy sanity   zero-g, the real policy driving, no blow-ups
    ///   5  locomotion      full gravity, does it walk toward a goal
    ///   6  parity + sweep  measured speed against MuJoCo's own 1.15 m/s
    ///
    /// SetUp pins Time.fixedDeltaTime to the project value, which for this project is
    /// already MuJoCo's own 0.005 s; TearDown always restores it, including for the rungs
    /// that deliberately sweep it.
    /// </summary>
    [TestFixture]
    public class MujocoBipedPlayModeTests
    {
        const float Gate_ReferenceAction = 1e-4f;
        const float Gate_KinematicsMetres = 1e-3f;
        const float Gate_ZeroGCoMVelocity = 0.02f;
        const float Gate_SpeedParityFraction = 0.50f;
        const float Rung5_MinDistanceClosed = 1.0f;
        const float Rung5_TargetDistance = 10.0f;

        MujocoBipedRigAsset _rig;
        GameObject _creature;
        GameObject _ground;
        float _savedFixedDt;
        Vector3 _savedGravity;

        [SetUp]
        public void SetUp()
        {
            _savedFixedDt = Time.fixedDeltaTime;
            _savedGravity = Physics.gravity;
#if UNITY_EDITOR
            _rig = AssetDatabase.LoadAssetAtPath<MujocoBipedRigAsset>(MujocoBipedPaths.RigAsset);
#endif
            if (_rig == null)
                Assert.Ignore($"{MujocoBipedPaths.RigAsset} not found - run " +
                              "MujocoBiped > Rebuild Rig Asset From JSON.");
        }

        [TearDown]
        public void TearDown()
        {
            if (_creature != null)
            {
                var agent = _creature.GetComponent<MujocoBipedAgent>();
                if (agent != null) agent.ReleaseWorker();
                Object.DestroyImmediate(_creature);
                _creature = null;
            }
            if (_ground != null) { Object.DestroyImmediate(_ground); _ground = null; }

            // Rungs 3 and 6 sweep the timestep; nothing may leak out of a test.
            Time.fixedDeltaTime = _savedFixedDt;
            Physics.gravity = _savedGravity;
        }

        // ------------------------------------------------------------- scaffolding --
        /// <summary>
        /// A solid slab whose TOP is at y = 0, not a Plane primitive: a plane's MeshCollider
        /// is paper-thin and a 40 kg biped's feet press straight through it. This mirrors
        /// how Systems_TrackBuilder builds SCN_RACE_FLAT's ground.
        /// </summary>
        void MakeGround(float friction = 1.0f,
                        PhysicsMaterialCombine combine = PhysicsMaterialCombine.Average)
        {
            _ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _ground.name = "TestGround";
            _ground.transform.position = new Vector3(0f, -0.5f, 0f);
            _ground.transform.localScale = new Vector3(200f, 1f, 200f);

            var mat = new PhysicsMaterial("TestGround")
            {
                staticFriction = friction,
                dynamicFriction = friction,
                bounciness = 0f,
                frictionCombine = combine,
                bounceCombine = PhysicsMaterialCombine.Minimum,
            };
            _ground.GetComponent<Collider>().sharedMaterial = mat;
        }

        MujocoBipedAgent SpawnCreature(Vector3 position, bool zeroGravity = false,
                                       Transform target = null)
        {
#if UNITY_EDITOR
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(MujocoBipedPaths.Prefab);
            if (prefab == null)
                Assert.Ignore($"{MujocoBipedPaths.Prefab} not found - run " +
                              "MujocoBiped > Build Prefab.");
            _creature = Object.Instantiate(prefab, position, Quaternion.identity);
#else
            Assert.Ignore("these tests need the Editor to load the prefab");
            return null;
#endif
            _creature.name = "MujocoBiped_test";
            var agent = _creature.GetComponent<MujocoBipedAgent>();
            Assert.That(agent, Is.Not.Null, "the prefab has no MujocoBipedAgent");

            // Instantiate runs Awake immediately, so anything set here needs Reconfigure.
            agent.zeroGravity = zeroGravity;
            agent.target = target;
            agent.autoRecoverFromFalls = false;      // a test measures falls, it does not hide them
            agent.Reconfigure();
            return agent;
        }

        static IEnumerator Steps(int n)
        {
            for (int i = 0; i < n; i++) yield return new WaitForFixedUpdate();
        }

        static int StepsFor(float seconds) => Mathf.CeilToInt(seconds / Time.fixedDeltaTime);

        // ============================================================ rung 0: model ==
        /// <summary>
        /// The ONNX, through Unity's own Inference Engine worker, against the actions
        /// MuJoCo recorded. Nothing here touches physics: if this passes and the creature
        /// still walks wrong, the fault is in the physics and only in the physics.
        ///
        /// check_onnx.py reports 7.749e-07 for the same data under onnxruntime, so a
        /// materially larger number here is an Inference Engine difference, not a model one.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung0_InferenceMatchesRecordedActions()
        {
            if (!MujocoBipedPaths.TryLoadReference(out var obs, out var acts, out string err))
                Assert.Ignore(err);

            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            yield return null;      // let Start() build the worker

            float worst = agent.RunReferenceCheck(obs, acts);
            if (float.IsNaN(worst))
                Assert.Ignore("no ModelAsset assigned, or Inference Engine is missing");

            Debug.Log($"[rung 0] in-engine inference over {obs.Length} recorded steps: " +
                      $"max abs action diff {worst:E3} (gate {Gate_ReferenceAction:E0}, " +
                      "onnxruntime reports 7.749E-007 for the same data)");
            Assert.That(worst, Is.LessThan(Gate_ReferenceAction),
                $"the policy disagrees with the recording by {worst:E3}. The model or the " +
                "tensor plumbing is wrong; no physics fix will help.");
        }

        // ======================================================= rung K: kinematics ==
        /// <summary>
        /// The frame map, against a forward kinematics computed independently from the
        /// MJCF by gen_kinematics_reference.py - never from MujocoBiped_rig.json, so an
        /// error shared by the extraction and the builder cannot hide.
        ///
        /// Joint limits are freed for the duration: MuJoCo's limits are soft constraints
        /// and its own zero pose sits 2 degrees past the knee limit, which Unity's hard
        /// limits would drag the rig off before it could be measured.
        /// </summary>
        [UnityTest]
        public IEnumerator RungK_ForwardKinematicsMatchesIndependentPythonFk()
        {
            if (!MujocoBipedPaths.TryLoadKinematics(out var poses, out float tol, out string err))
                Assert.Ignore(err);
            tol = Mathf.Max(tol, Gate_KinematicsMetres);

            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            agent.enabled = false;                  // no policy, no torque - pure kinematics
            agent.SetJointLimitsEnabled(false);
            yield return new WaitForFixedUpdate();

            float worst = 0f;
            string worstWhere = "";
            var log = new StringBuilder();

            foreach (var pose in poses)
            {
                // Place the root exactly where the Python FK put it, then pose the joints.
                Vector4 q = pose.rootQuatMujWxyz;
                agent.Root.TeleportRoot(MujocoBipedFrameMap.Pos(pose.rootPosMuj),
                                        MujocoBipedFrameMap.RotFromWxyz(q.x, q.y, q.z, q.w));
                agent.Root.linearVelocity = Vector3.zero;
                agent.Root.angularVelocity = Vector3.zero;
                agent.SetJointPositionsRad(pose.jointsRad);
                yield return new WaitForFixedUpdate();

                float poseWorst = 0f;
                for (int b = 0; b < pose.bodyNames.Length; b++)
                {
                    var body = agent.FindBody(pose.bodyNames[b]);
                    Assert.That(body, Is.Not.Null,
                        $"the rig has no link named '{pose.bodyNames[b]}'");

                    Vector3 expect = MujocoBipedFrameMap.Pos(pose.bodyPosMuj[b]);
                    float d = Vector3.Distance(body.transform.position, expect);
                    poseWorst = Mathf.Max(poseWorst, d);
                    if (d > worst) { worst = d; worstWhere = $"{pose.label}/{pose.bodyNames[b]}"; }
                }
                log.Append($"\n  {pose.label,-18} worst body position error {poseWorst * 1000f:F4} mm");
            }

            agent.SetJointLimitsEnabled(true);

            Debug.Log($"[rung K] forward kinematics vs independent Python FK of the MJCF, " +
                      $"{poses.Length} poses:{log}\n  overall {worst * 1000f:F4} mm at " +
                      $"{worstWhere} (gate {tol * 1000f:F2} mm)");
            Assert.That(worst, Is.LessThan(tol),
                $"body positions are off by {worst * 1000f:F3} mm at {worstWhere}. The frame " +
                "map, a joint axis, a body offset or the joint composition order is wrong.");
        }

        // ====================================================== rung O: observations ==
        /// <summary>
        /// Put the Unity rig into a state MuJoCo actually recorded, rebuild the 49-float
        /// observation from it, and compare term by term against what MuJoCo wrote.
        ///
        /// This is the rung that catches the errors nothing else can. Rung 0 proves the
        /// model is intact and rung K proves the frame map moves bodies correctly, but
        /// neither would notice an angular velocity rotated once instead of twice, a
        /// linear velocity read at the centre of mass instead of the body origin, or a
        /// target direction rotated by the full orientation instead of yaw alone. Every
        /// one of those is silent - the vector has a plausible magnitude and is wrong only
        /// when the torso is tilted, which is exactly when the policy depends on it.
        ///
        /// Reported per term, because which term is off names the bug.
        /// </summary>
        [UnityTest]
        public IEnumerator RungO_ObservationMatchesRecordedObservations()
        {
            if (!MujocoBipedPaths.TryLoadStates(out var steps, out string err))
                Assert.Ignore(err);

            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            agent.enabled = false;                  // no policy, no torque
            agent.SetJointLimitsEnabled(false);     // MuJoCo's limits are soft; its states can sit outside

            var goal = new GameObject("ObsGoal");
            agent.target = goal.transform;
            yield return new WaitForFixedUpdate();

            // Term name, first index, length - the layout from CONTRACT.md section 2.
            var terms = new (string name, int start, int len)[]
            {
                ("torso_height", 0, 1),
                ("projected_gravity", 1, 3),
                ("linear_velocity", 4, 3),
                ("angular_velocity", 7, 3),
                ("joint_positions", 10, 12),
                ("joint_velocities", 22, 12),
                ("target_direction", 34, 2),
                ("target_distance", 36, 1),
                ("last_action", 37, 12),
            };
            var worst = new float[terms.Length];
            var worstStep = new int[terms.Length];

            // Every recorded step - nothing is stepped in the loop, so all 150 are cheap.

            for (int i = 1; i < steps.Length; i++)
            {
                var s = steps[i];
                agent.SetMujocoState(s.rootPosMuj, s.rootQuatMujWxyz, s.rootLinVelWorldMuj,
                                     s.rootAngVelBodyLocalMuj, s.jointPositionsRad,
                                     s.jointVelocitiesRaw);
                agent.SetLastActionForTest(steps[i - 1].action);
                goal.transform.position = MujocoBipedFrameMap.Pos(s.targetPosMuj);

                // Deliberately NO physics step between setting the state and reading it.
                // One 5 ms tick integrates the recorded velocities forward - 10 m/s of
                // torso speed is 5 cm and 37 rad/s of joint speed is 10.6 degrees - and
                // that integration error would swamp the frame error this test exists to
                // measure. Everything the observation reads is set directly: TeleportRoot
                // writes the root transform, the velocity setters write the root's
                // velocities, and jointPosition/jointVelocity are the articulation's own
                // reduced state.
                float[] got = agent.CaptureObservation();
                for (int t = 0; t < terms.Length; t++)
                {
                    for (int k = 0; k < terms[t].len; k++)
                    {
                        int idx = terms[t].start + k;
                        float d = Mathf.Abs(got[idx] - s.observation[idx]);
                        if (d > worst[t]) { worst[t] = d; worstStep[t] = i; }
                    }
                }
            }

            agent.SetJointLimitsEnabled(true);
            Object.DestroyImmediate(goal);

            var log = new StringBuilder();
            float overall = 0f;
            string overallTerm = "";
            for (int t = 0; t < terms.Length; t++)
            {
                log.Append($"\n  {terms[t].name,-20} max abs error {worst[t]:E3}  " +
                           $"(worst at recorded step {worstStep[t]})");
                if (worst[t] > overall) { overall = worst[t]; overallTerm = terms[t].name; }
            }

            // Float precision through TeleportRoot and the reduced-coordinate state, not
            // physics: nothing is integrated between writing the state and reading it.
            const float gate = 1e-3f;
            Debug.Log($"[rung O] observation parity against MuJoCo's own recorded " +
                      $"observations, {steps.Length - 1} states:{log}\n" +
                      $"  overall {overall:E3} in '{overallTerm}' (gate {gate:E0})");

            Assert.That(overall, Is.LessThan(gate),
                $"the '{overallTerm}' term differs from MuJoCo by {overall:E3}. The policy " +
                "is being fed something it was not trained on; see CONTRACT.md section 2 " +
                "for what that term is supposed to be.");
        }

        // ========================================================== rung 1: statics ==
        /// <summary>
        /// Statics, in two parts, neither of which is "does a passive biped stand up".
        ///
        /// It cannot. With `stiffness = 0` the only thing acting on a knee is 1 N.m.s/rad
        /// of damping, and the knee's range runs to -150 degrees, so a torque-free rig
        /// folds under its own weight. That is correct physics, not a defect - MuJoCo does
        /// the same. What is worth asserting is the GEOMETRY and the CONTACT:
        ///
        ///   a) at the spawn pose, with the torso origin at MuJoCo's init_qpos height of
        ///      0.88 m, the lowest point of the rig sits on the ground. If leg lengths,
        ///      the spawn height or the foot collider are wrong, this is where it shows.
        ///   b) after settling, the rig rests ON the floor rather than sinking through it.
        ///
        /// MuJoCo's own init_qpos puts the feet about 5 mm BELOW the floor and lets its
        /// soft contacts resolve it, so (a) allows for that.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung1_StandsAtTheMujocoSpawnHeightAndRestsOnTheGround()
        {
            MakeGround();
            var agent = SpawnCreature(new Vector3(0f, _rig.SpawnHeight, 0f));
            agent.enabled = false;              // gravity and contacts only, no torque

            // (a) Geometry, before a single tick of dynamics.
            float lowestAtSpawn = LowestColliderPoint(agent);
            float torsoAtSpawn = agent.Root.transform.position.y;

            yield return Steps(StepsFor(3f));

            // (b) Contact, after settling.
            float lowestSettled = LowestColliderPoint(agent);
            float torsoSettled = agent.Root.transform.position.y;
            Vector3 v = agent.CenterOfMassVelocity;

            Debug.Log($"[rung 1] statics at fdt {Time.fixedDeltaTime:F5}:\n" +
                      $"  at the spawn pose: torso origin {torsoAtSpawn:F4} m " +
                      $"(MuJoCo init_qpos {_rig.SpawnHeight:F4}), lowest collider point " +
                      $"{lowestAtSpawn * 1000f:+0.00;-0.00} mm relative to the ground\n" +
                      $"  after 3 s passive: torso origin {torsoSettled:F4} m, lowest point " +
                      $"{lowestSettled * 1000f:+0.00;-0.00} mm, |v_CoM| {v.magnitude:F4} m/s, " +
                      $"upright {agent.Uprightness:F4}\n" +
                      $"  A torque-free biped folds - stiffness is 0 and the knee range runs " +
                      $"to -150 deg, so the torso dropping here is expected. Rung 5 is where " +
                      $"standing is judged.");

            Assert.That(Mathf.Abs(torsoAtSpawn - _rig.SpawnHeight), Is.LessThan(1e-4f),
                $"the torso origin spawned at {torsoAtSpawn:F4} m, not MuJoCo's " +
                $"{_rig.SpawnHeight:F4} m. The root link's offset is being applied on top " +
                "of the spawn position - for a free joint MuJoCo folds the body's own pos " +
                "into qpos[0:3], so those are the same number, not two.");
            Assert.That(lowestAtSpawn, Is.GreaterThan(-0.02f).And.LessThan(0.02f),
                $"at the spawn pose the lowest point of the rig is {lowestAtSpawn * 1000f:F1} mm " +
                "from the ground. MuJoCo's init_qpos puts it about -5 mm; anything else means " +
                "the leg lengths, the spawn height or the foot collider are wrong.");
            Assert.That(lowestSettled, Is.GreaterThan(-0.05f),
                $"the rig sank {(-lowestSettled) * 1000f:F0} mm into the ground. Contact is " +
                "not holding - check the foot collider and contactOffset.");
            Assert.That(v.magnitude, Is.LessThan(0.1f),
                $"still moving at {v.magnitude:F3} m/s after 3 s - it is not settling.");
        }

        /// <summary>Lowest point of any collider on the rig, in world Y. Ground top is 0.</summary>
        static float LowestColliderPoint(MujocoBipedAgent agent)
        {
            float lowest = float.MaxValue;
            foreach (var c in agent.GetComponentsInChildren<Collider>(true))
                lowest = Mathf.Min(lowest, c.bounds.min.y);
            return lowest;
        }

        // ========================================================= rung 2: momentum ==
        /// <summary>
        /// Zero gravity, no ground, one joint driven. Internal torque cannot change total
        /// linear momentum, so the centre of mass must stay put. Anything else is PhysX
        /// manufacturing momentum, and it manufactures it into the gait too.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung2_ZeroGSingleJointConservesLinearMomentum()
        {
            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            agent.actionOverride = MujocoBipedAgent.ActionOverride.Constant;
            agent.overrideJointIndex = 3;               // knee_l, the biggest single mass swing
            agent.overrideAmplitude = 1f;               // full gear: 110 N.m
            yield return Steps(StepsFor(0.5f));         // let the transient go

            float worst = 0f;
            int n = StepsFor(2f);
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                worst = Mathf.Max(worst, agent.CenterOfMassVelocity.magnitude);
            }

            Debug.Log($"[rung 2] zero-g, joint {agent.overrideJointIndex} " +
                      $"({_rig.jointOrder[agent.overrideJointIndex]}) at full torque, " +
                      $"fdt {Time.fixedDeltaTime:F5}: max |v_CoM| {worst:F5} m/s " +
                      $"(gate {Gate_ZeroGCoMVelocity})");
            Assert.That(worst, Is.LessThan(Gate_ZeroGCoMVelocity),
                $"the centre of mass drifted at {worst:F4} m/s under purely internal " +
                "torque. PhysX is pumping momentum - suspect the inertia tensors, the " +
                "solver iteration count, or a link whose inertia is too small for the step.");
        }

        // ================================================ rung D: actuator calibration ==
        /// <summary>
        /// Two questions this rig cannot be trusted without answering.
        ///
        /// 1. Does the actuator torque reach the solver at all, and through which API?
        ///    `ArticulationBody.jointForce` writes into the joint's reduced space, which
        ///    is exactly MuJoCo's actuator semantics; an equal-and-opposite `AddTorque`
        ///    pair on the child and its parent is the route the export's own README
        ///    recommends. Both are measured against a no-torque control.
        ///
        /// 2. Does `ArticulationDrive.damping` act on radians per second or degrees per
        ///    second? This has to be measured, not assumed: `jointPosition` and
        ///    `jointVelocity` are unambiguously radians while `xDrive.target` and the
        ///    limits are unambiguously degrees, so the drive straddles both conventions
        ///    and the gain's own units are documented as neither. The two hypotheses differ
        ///    by 180/pi = 57.3x - MuJoCo's 1.0 N.m.s/rad would become 57.3, and the
        ///    creature would wade rather than walk.
        ///
        /// SELF-COLLISION IS OFF for this test and the joint limits are freed. With both
        /// on, a joint driven at full torque swings its own shin into the pelvis within
        /// 50 ms and jams, and every reading below collapses to a number that looks
        /// exactly like "the torque never arrived". That false negative cost real time -
        /// see Diag_SingleJointTorqueTimeSeries for the trace that separated the two.
        /// </summary>
        [UnityTest]
        public IEnumerator RungD_ActuatorReachesTheSolverAndDampingIsPerRadian()
        {
            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            agent.selfCollisionMode = MujocoBipedAgent.SelfCollisionMode.None;

            const int joint = 3;                       // knee_l, gear 110
            const float action = 0.1f;                 // 11 N.m
            const float torque = action * 110f;

            float jointForceQd = 0f;     // jointForce, damping 0     -> should accelerate
            float noTorqueQd = 0f;       // no torque at all          -> the control
            float pairQd = 0f;           // AddTorque pair, damping 0 -> should accelerate
            float dampedQd = 0f;         // AddTorque pair, damping 1 -> steady state

            for (int c = 0; c < 4; c++)
            {
                agent.actuatorMode = c == 0
                    ? MujocoBipedAgent.ActuatorMode.DirectTorqueExplicitDamping
                    : MujocoBipedAgent.ActuatorMode.TorquePairImplicitDamping;
                agent.actionOverride = MujocoBipedAgent.ActionOverride.Constant;
                agent.overrideJointIndex = joint;
                agent.overrideAmplitude = c == 1 ? 0f : action;
                agent.Reconfigure();
                agent.SetJointLimitsEnabled(false);

                if (c != 3)
                {
                    // Suppress the damping so A, B and C show the bare torque response.
                    foreach (var b in agent.Joints)
                    {
                        var dr = b.xDrive;
                        dr.damping = 0f;
                        b.xDrive = dr;
                    }
                }

                // Config D needs to reach terminal velocity (tau = kd * qd); the others
                // are read while still accelerating, which is the point.
                yield return Steps(StepsFor(c == 3 ? 3f : 0.4f));

                float acc = 0f;
                int n = StepsFor(c == 3 ? 1f : 0.2f);
                for (int i = 0; i < n; i++)
                {
                    yield return new WaitForFixedUpdate();
                    acc += Mathf.Abs(agent.Joints[joint].jointVelocity[0]);
                }
                acc /= n;
                if (c == 0) jointForceQd = acc;
                else if (c == 1) noTorqueQd = acc;
                else if (c == 2) pairQd = acc;
                else dampedQd = acc;
            }

            agent.SetJointLimitsEnabled(true);

            float kdEffective = dampedQd > 1e-6f ? torque / dampedQd : float.PositiveInfinity;
            float ifRadians = torque / 1.0f;                       // 11 rad/s
            float ifDegrees = torque / Mathf.Rad2Deg;              // 0.192 rad/s

            Debug.Log($"[rung D] actuator calibration on {_rig.jointOrder[joint]} " +
                      $"({torque:F1} N.m, zero gravity, limits freed, self-collision off):\n" +
                      $"  A  jointForce,     damping 0  mean |qd| {jointForceQd,9:F4} rad/s\n" +
                      $"  B  no torque       (control)  mean |qd| {noTorqueQd,9:F4} rad/s\n" +
                      $"  C  AddTorque pair, damping 0  mean |qd| {pairQd,9:F4} rad/s\n" +
                      $"  D  AddTorque pair, damping 1  mean |qd| {dampedQd,9:F4} rad/s " +
                      "(terminal velocity, calibrates the units)\n" +
                      $"  If damping is per RADIAN/s, D should read {ifRadians:F4} rad/s " +
                      "(kd = 1.0, MuJoCo's value).\n" +
                      $"  If damping is per DEGREE/s, D should read {ifDegrees:F4} rad/s " +
                      "(kd = 57.3 effective).\n" +
                      $"  Implied kd_effective {kdEffective:F3} N.m.s/rad.");

            Assert.That(jointForceQd, Is.GreaterThan(10f * Mathf.Max(noTorqueQd, 1e-4f)),
                $"jointForce moved the joint at {jointForceQd:F4} rad/s against " +
                $"{noTorqueQd:F4} rad/s with no torque at all - it is not reaching the solver.");
            Assert.That(pairQd, Is.GreaterThan(10f * Mathf.Max(noTorqueQd, 1e-4f)),
                $"the AddTorque pair moved the joint at {pairQd:F4} rad/s against " +
                $"{noTorqueQd:F4} rad/s with no torque at all - it is not reaching the solver.");
            Assert.That(kdEffective, Is.LessThan(5f),
                $"the drive behaves as {kdEffective:F1} N.m.s/rad where MuJoCo used 1.0. " +
                "If that is close to 57, ArticulationDrive.damping is per degree/s and " +
                "every damping value must be scaled by Mathf.Deg2Rad before it is written.");
        }

        /// <summary>
        /// Diagnostic time series for one joint under constant torque. Not a gate - it
        /// exists so a "the joint barely moves" reading can be told apart from "the joint
        /// moves and then something stops it", which a windowed mean cannot distinguish.
        /// </summary>
        [UnityTest]
        public IEnumerator Diag_SingleJointTorqueTimeSeries()
        {
            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            agent.actuatorMode = MujocoBipedAgent.ActuatorMode.TorquePairImplicitDamping;
            agent.selfCollisionMode = MujocoBipedAgent.SelfCollisionMode.None;
            agent.Reconfigure();
            agent.SetJointLimitsEnabled(false);
            foreach (var b in agent.Joints)
            {
                var d = b.xDrive;
                d.damping = 0f;
                b.xDrive = d;
            }

            const int joint = 3;                       // knee_l, gear 110
            agent.actionOverride = MujocoBipedAgent.ActionOverride.Constant;
            agent.overrideJointIndex = joint;
            agent.overrideAmplitude = 1f;              // 110 N.m, the joint's full torque

            var log = new StringBuilder();
            var jb = agent.Joints[joint];
            log.Append($"\n  {"t (s)",8} {"qpos (rad)",12} {"qvel (rad/s)",14} {"action",8}");
            for (int i = 0; i <= 40; i++)
            {
                if (i > 0) yield return Steps(StepsFor(0.025f));
                log.Append($"\n  {i * 0.025f,8:F3} {jb.jointPosition[0],12:F5} " +
                           $"{jb.jointVelocity[0],14:F5} {agent.LatestAction[joint],8:F3}");
            }

            agent.SetJointLimitsEnabled(true);
            Debug.Log($"[diag] {_rig.jointOrder[joint]} under a constant 110 N.m torque pair, " +
                      $"zero gravity, limits freed, self-collision off:{log}");
            Assert.Pass("diagnostic only");
        }

        // ======================================================== rung 3: stability ==
        /// <summary>
        /// Zero gravity, a square wave on every joint, across candidate timesteps. Not a
        /// pass/fail on the physics so much as a measurement of how much headroom the
        /// project step has - and a direct check of RIG_AUDIT section C, which predicts
        /// the explicit-damping path rings but does not diverge at 0.005 s.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung3_ZeroGSquareWaveStabilityAcrossTimesteps()
        {
            var cases = new List<(string label, float dt, MujocoBipedAgent.ActuatorMode mode)>
            {
                ("project 0.005 shipped (jointForce)", _savedFixedDt,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping),
                ("1/120 shipped (jointForce)", 1f / 120f,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping),
                ("1/240 shipped (jointForce)", 1f / 240f,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping),
                ("project 0.005 torque pair", _savedFixedDt,
                    MujocoBipedAgent.ActuatorMode.TorquePairImplicitDamping),
                ("1/480 explicit torque", 1f / 480f,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueExplicitDamping),
                ("project 0.005 explicit torque", _savedFixedDt,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueExplicitDamping),
            };

            var log = new StringBuilder();
            var divergedAtProjectStep = new List<string>();

            foreach (var c in cases)
            {
                Time.fixedDeltaTime = c.dt;
                var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
                agent.actuatorMode = c.mode;
                agent.actionOverride = MujocoBipedAgent.ActionOverride.SquareWave;
                agent.overrideJointIndex = -1;
                agent.overrideAmplitude = 1f;
                agent.overrideSquareWavePeriod = 0.5f;
                agent.Reconfigure();

                yield return Steps(StepsFor(0.5f));

                float peakCoM = 0f, peakJoint = 0f;
                bool diverged = false;
                int n = StepsFor(3f);
                for (int i = 0; i < n && !diverged; i++)
                {
                    yield return new WaitForFixedUpdate();
                    peakCoM = Mathf.Max(peakCoM, agent.CenterOfMassVelocity.magnitude);
                    for (int j = 0; j < agent.Joints.Count; j++)
                        peakJoint = Mathf.Max(peakJoint, Mathf.Abs(agent.Joints[j].jointVelocity[0]));
                    diverged = float.IsNaN(peakCoM) || float.IsInfinity(peakCoM) || peakJoint > 500f;
                }

                if (diverged && Mathf.Abs(c.dt - _savedFixedDt) < 1e-9f
                    && c.mode == MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping)
                    divergedAtProjectStep.Add(c.label);
                log.Append($"\n  {c.label,-30} peak |v_CoM| {peakCoM,8:F4} m/s   " +
                           $"peak |qd| {peakJoint,8:F2} rad/s{(diverged ? "   DIVERGED" : "")}");

                agent.ReleaseWorker();
                Object.DestroyImmediate(_creature);
                _creature = null;
            }

            Time.fixedDeltaTime = _savedFixedDt;
            Debug.Log($"[rung 3] zero-g square wave, all 12 joints at full torque, 3 s each:" +
                      $"{log}\n" +
                      "  Recorded MuJoCo peak joint velocity for reference: 37.14 rad/s.\n" +
                      "  This is a HEADROOM measurement. Twelve joints square-waving at full " +
                      "torque in free\n  fall is far outside anything MuJoCo simulated - " +
                      "under the real policy the peak is\n  43 rad/s (rung 4) - so the " +
                      "off-step and off-actuator rows are there to show where\n  the margin " +
                      "runs out, not to be passed. Only the SHIPPED actuator at the PROJECT " +
                      "step\n  is asserted, because that is the one combination this " +
                      "creature actually runs.");
            Assert.That(divergedAtProjectStep, Is.Empty,
                "the PROJECT timestep diverged under " +
                string.Join(", ", divergedAtProjectStep) +
                " - that is the step this creature ships at, so this is a real defect, not " +
                "a headroom note.");
        }

        // ==================================================== rung 4: policy sanity ==
        /// <summary>
        /// Zero gravity, the real policy driving. Out of distribution by construction - the
        /// policy has never seen free fall - so this is not a behaviour test. It asks one
        /// question: does the full loop run for five seconds without producing a NaN, an
        /// infinity or a joint velocity an order of magnitude past anything MuJoCo recorded.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung4_ZeroGPolicyActuationStaysFinite()
        {
            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            yield return null;

            float peakJoint = 0f, peakAction = 0f;
            bool bad = false;
            int n = StepsFor(5f);
            for (int i = 0; i < n && !bad; i++)
            {
                yield return new WaitForFixedUpdate();
                for (int j = 0; j < agent.Joints.Count; j++)
                {
                    float qd = agent.Joints[j].jointVelocity[0];
                    if (float.IsNaN(qd) || float.IsInfinity(qd)) { bad = true; break; }
                    peakJoint = Mathf.Max(peakJoint, Mathf.Abs(qd));
                }
                var a = agent.LatestAction;
                for (int j = 0; j < a.Length; j++) peakAction = Mathf.Max(peakAction, Mathf.Abs(a[j]));
            }

            Debug.Log($"[rung 4] zero-g with the policy driving, 5 s at fdt " +
                      $"{Time.fixedDeltaTime:F5}:\n" +
                      $"  policy steps {agent.PolicySteps} (decimation {agent.Decimation})\n" +
                      $"  peak |qd| {peakJoint:F2} rad/s   peak |action| {peakAction:F3}\n" +
                      $"  torso height {agent.TorsoHeight:F3} m   upright {agent.Uprightness:F3}");

            Assert.That(bad, Is.False, "a joint velocity went NaN or infinite");
            Assert.That(peakAction, Is.LessThanOrEqualTo(1.0001f),
                $"the network emitted {peakAction:F4}, outside the [-1, 1] its own Clip node " +
                "should guarantee - the wrong output tensor is being read.");
            Assert.That(agent.PolicySteps, Is.GreaterThan(n / agent.Decimation - 2),
                "the policy did not tick once per decimation window");
            Assert.That(peakJoint, Is.LessThan(400f),
                $"joints reached {peakJoint:F1} rad/s, more than 10x anything MuJoCo " +
                "recorded (37.14 rad/s) - the rig is unstable, not merely out of distribution.");
        }

        // ======================================================= rung 5: locomotion ==
        /// <summary>
        /// Full gravity, real ground, a goal 10 m ahead. The first rung where the creature
        /// has to actually walk.
        ///
        /// The gate is deliberately modest - one metre closed while staying upright - because
        /// MuJoCo's own policy fell before the 25 s timeout in 70% of its evaluation
        /// rollouts. Rung 6 is where speed is judged.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung5_WalksTowardTheTargetUnderGravity()
        {
            MakeGround();

            var goal = new GameObject("TestGoal");
            goal.transform.position = new Vector3(0f, 0.02f, Rung5_TargetDistance);

            var agent = SpawnCreature(new Vector3(0f, _rig.SpawnHeight, 0f), target: goal.transform);
            yield return null;

            Vector3 start = agent.Root.transform.position;
            float startDist = PlanarDistance(start, goal.transform.position);
            float bestDist = startDist;
            float uprightAt = agent.Uprightness;
            float fellAt = -1f;

            int n = StepsFor(20f);
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                float d = PlanarDistance(agent.Root.transform.position, goal.transform.position);
                bestDist = Mathf.Min(bestDist, d);
                if (fellAt < 0f && !agent.IsHealthy) fellAt = i * Time.fixedDeltaTime;
                if (fellAt < 0f) uprightAt = agent.Uprightness;
            }

            float closed = startDist - bestDist;
            Vector3 end = agent.Root.transform.position;
            Object.DestroyImmediate(goal);

            Debug.Log($"[rung 5] 20 s under gravity toward a goal {Rung5_TargetDistance:F1} m " +
                      $"away, fdt {Time.fixedDeltaTime:F5}:\n" +
                      $"  start {start:F2} -> end {end:F2}\n" +
                      $"  distance closed {closed:F2} m (best approach {bestDist:F2} m)\n" +
                      $"  upright while healthy {uprightAt:F3}   " +
                      $"{(fellAt < 0f ? "stayed healthy for the full 20 s" : $"left the healthy band at {fellAt:F1} s")}\n" +
                      $"  MuJoCo's own eval survived the full 25 s in only " +
                      $"{_rig.mujocoSurvivedFullEpisodeFraction:P0} of rollouts.");

            Assert.That(closed, Is.GreaterThan(Rung5_MinDistanceClosed),
                $"only {closed:F2} m of the {startDist:F1} m was closed. The creature is not " +
                "walking - check rungs 0, K and 1 before touching any gain.");
            Assert.That(uprightAt, Is.GreaterThan(_rig.minUprightness),
                $"uprightness fell to {uprightAt:F3}, below MuJoCo's own termination " +
                $"threshold of {_rig.minUprightness:F2}, while still counted healthy.");
        }

        // ============================================== rung 6: speed parity + sweep ==
        /// <summary>
        /// Measured closing speed against MuJoCo's own 1.15 m/s. If Unity manages less than
        /// half of that, the sweep below runs and reports a comparison table rather than
        /// leaving a bare number - the failure message is meant to be actionable.
        /// </summary>
        [UnityTest]
        public IEnumerator Rung6_SpeedParityAgainstMujocoBaseline()
        {
            float speed = 0f;
            yield return MeasureClosingSpeed(
                friction: _rig.physics.effectiveFootGroundFriction,
                groundCombine: PhysicsMaterialCombine.Average,
                armature: MujocoBipedRigBuilder.ArmatureMode.Exact,
                actuator: MujocoBipedAgent.ActuatorMode.TorquePairImplicitDamping,
                contactOffset: _rig.physics.contactOffset,
                result: v => speed = v);

            float parity = speed / _rig.mujocoMeanClosingSpeed;
            Debug.Log($"[rung 6] closing speed {speed:F3} m/s against MuJoCo's " +
                      $"{_rig.mujocoMeanClosingSpeed:F2} m/s = {parity:P0} parity " +
                      $"(gate {Gate_SpeedParityFraction:P0})");

            if (parity >= Gate_SpeedParityFraction)
            {
                Assert.Pass($"speed parity {parity:P0}");
                yield break;
            }

            // Under the gate: sweep, so the report says WHICH knob matters.
            var rows = new List<string>();
            rows.Add($"  {"configuration",-46} {"speed m/s",10} {"parity",8}");
            rows.Add($"  {"baseline (shipped)",-46} {speed,10:F3} {parity,8:P0}");

            var sweep = new List<(string label, float fric, PhysicsMaterialCombine comb,
                                  MujocoBipedRigBuilder.ArmatureMode arm,
                                  MujocoBipedAgent.ActuatorMode act, float offset)>
            {
                ("ground friction 1.2, Maximum combine (MuJoCo's rule)", 1.2f,
                    PhysicsMaterialCombine.Maximum, MujocoBipedRigBuilder.ArmatureMode.Exact,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping, 0.01f),
                ("ground friction 0.6 (Unity default, no material)", 0.6f,
                    PhysicsMaterialCombine.Average, MujocoBipedRigBuilder.ArmatureMode.Exact,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping, 0.01f),
                ("contact offset 0.02", 1.2f,
                    PhysicsMaterialCombine.Average, MujocoBipedRigBuilder.ArmatureMode.Exact,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping, 0.02f),
                ("armature None", 1.2f,
                    PhysicsMaterialCombine.Average, MujocoBipedRigBuilder.ArmatureMode.None,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping, 0.01f),
                ("armature Naive (over-counts parallel runs)", 1.2f,
                    PhysicsMaterialCombine.Average, MujocoBipedRigBuilder.ArmatureMode.Naive,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueImplicitDamping, 0.01f),
                ("explicit torque damping (jointForce)", 1.2f,
                    PhysicsMaterialCombine.Average, MujocoBipedRigBuilder.ArmatureMode.Exact,
                    MujocoBipedAgent.ActuatorMode.DirectTorqueExplicitDamping, 0.01f),
                ("AddTorque pair + implicit damping", 1.2f,
                    PhysicsMaterialCombine.Average, MujocoBipedRigBuilder.ArmatureMode.Exact,
                    MujocoBipedAgent.ActuatorMode.TorquePairImplicitDamping, 0.01f),
            };

            float best = speed;
            string bestLabel = "baseline (shipped)";
            foreach (var s in sweep)
            {
                float v = 0f;
                yield return MeasureClosingSpeed(s.fric, s.comb, s.arm, s.act, s.offset,
                                                 r => v = r);
                rows.Add($"  {s.label,-46} {v,10:F3} {v / _rig.mujocoMeanClosingSpeed,8:P0}");
                if (v > best) { best = v; bestLabel = s.label; }
            }

            string table = string.Join("\n", rows);
            Debug.Log($"[rung 6] speed parity sweep (MuJoCo baseline " +
                      $"{_rig.mujocoMeanClosingSpeed:F2} m/s):\n{table}\n" +
                      $"  best: {bestLabel} at {best:F3} m/s " +
                      $"({best / _rig.mujocoMeanClosingSpeed:P0})");

            Assert.Fail($"speed parity {parity:P0} is below the {Gate_SpeedParityFraction:P0} " +
                        $"gate. The sweep above found '{bestLabel}' best at {best:F3} m/s " +
                        $"({best / _rig.mujocoMeanClosingSpeed:P0}).");
        }

        /// <summary>
        /// One 15 s run toward a distant goal, reporting metres closed per second. The goal
        /// is placed far enough that the creature never reaches it, so the number is a
        /// speed and not a race time.
        /// </summary>
        IEnumerator MeasureClosingSpeed(float friction, PhysicsMaterialCombine groundCombine,
                                        MujocoBipedRigBuilder.ArmatureMode armature,
                                        MujocoBipedAgent.ActuatorMode actuator,
                                        float contactOffset,
                                        System.Action<float> result)
        {
            if (_creature != null)
            {
                var old = _creature.GetComponent<MujocoBipedAgent>();
                if (old != null) old.ReleaseWorker();
                Object.DestroyImmediate(_creature);
                _creature = null;
            }
            if (_ground != null) { Object.DestroyImmediate(_ground); _ground = null; }

            MakeGround(friction, groundCombine);

            var goal = new GameObject("SpeedGoal");
            goal.transform.position = new Vector3(0f, 0.02f, 40f);

            var agent = SpawnCreature(new Vector3(0f, _rig.SpawnHeight, 0f), target: goal.transform);
            agent.armatureMode = armature;
            agent.actuatorMode = actuator;
            foreach (var c in agent.GetComponentsInChildren<Collider>(true))
                c.contactOffset = contactOffset;
            agent.Reconfigure();
            yield return null;

            const float duration = 15f;
            Vector3 start = agent.Root.transform.position;
            float startDist = PlanarDistance(start, goal.transform.position);
            float bestDist = startDist;

            int n = StepsFor(duration);
            for (int i = 0; i < n; i++)
            {
                yield return new WaitForFixedUpdate();
                bestDist = Mathf.Min(bestDist,
                    PlanarDistance(agent.Root.transform.position, goal.transform.position));
            }

            Object.DestroyImmediate(goal);
            result(Mathf.Max(0f, (startDist - bestDist) / duration));
        }

        static float PlanarDistance(Vector3 a, Vector3 b)
        {
            a.y = 0f;
            b.y = 0f;
            return Vector3.Distance(a, b);
        }

        // ============================================================ perf: 8 at 60 ==
        /// <summary>
        /// The stated budget: 8 creatures at 60 FPS, CPU inference backend, measured in
        /// the editor.
        ///
        /// Measures the two costs separately, because they scale differently and only one
        /// of them is this port's doing. Physics runs at 200 Hz - five FixedUpdates per
        /// 60 Hz frame - while inference runs at 40 Hz, one per creature per five physics
        /// ticks, so the frame budget is dominated by PhysX articulation solving rather
        /// than by the network.
        /// </summary>
        [UnityTest]
        public IEnumerator Perf_EightCreaturesHoldSixtyFps()
        {
            const int count = 8;
            const float budgetMs = 1000f / 60f;

            MakeGround();
            var goal = new GameObject("PerfGoal");
            goal.transform.position = new Vector3(0f, 0.02f, 40f);

            var crowd = new List<GameObject>(count);
            var agents = new List<MujocoBipedAgent>(count);
#if UNITY_EDITOR
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(MujocoBipedPaths.Prefab);
            if (prefab == null) Assert.Ignore("prefab not built");
            for (int i = 0; i < count; i++)
            {
                var go = Object.Instantiate(prefab,
                    new Vector3((i - count / 2) * 1.5f, _rig.SpawnHeight, 0f), Quaternion.identity);
                var a = go.GetComponent<MujocoBipedAgent>();
                a.target = goal.transform;
                a.autoRecoverFromFalls = true;
                a.Reconfigure();
                crowd.Add(go);
                agents.Add(a);
            }
#endif
            yield return Steps(StepsFor(1f));          // warm up: JIT, worker allocation, contacts

            var sw = new System.Diagnostics.Stopwatch();
            double totalMs = 0;
            int frames = 0;
            float worstFrameMs = 0f;

            for (int f = 0; f < 120; f++)
            {
                sw.Restart();
                yield return null;                      // one rendered frame, physics included
                sw.Stop();
                float ms = (float)sw.Elapsed.TotalMilliseconds;
                totalMs += ms;
                worstFrameMs = Mathf.Max(worstFrameMs, ms);
                frames++;
            }

            float meanMs = (float)(totalMs / frames);
            int policySteps = 0;
            foreach (var a in agents) policySteps += a.PolicySteps;

            foreach (var go in crowd)
            {
                var a = go.GetComponent<MujocoBipedAgent>();
                if (a != null) a.ReleaseWorker();
                Object.DestroyImmediate(go);
            }
            Object.DestroyImmediate(goal);

            Debug.Log($"[perf] {count} creatures, {frames} frames, fdt " +
                      $"{Time.fixedDeltaTime:F5} ({1f / Time.fixedDeltaTime:F0} Hz physics, " +
                      $"{1f / _rig.policyDt:F0} Hz policy, CPU backend):\n" +
                      $"  mean frame {meanMs:F2} ms ({1000f / meanMs:F1} FPS)   " +
                      $"worst {worstFrameMs:F2} ms\n" +
                      $"  budget {budgetMs:F2} ms for 60 FPS -> " +
                      $"{(meanMs <= budgetMs ? "WITHIN" : "OVER")} by " +
                      $"{Mathf.Abs(meanMs - budgetMs):F2} ms\n" +
                      $"  headroom {budgetMs / Mathf.Max(meanMs, 1e-3f):F2}x -> about " +
                      $"{Mathf.FloorToInt(count * budgetMs / Mathf.Max(meanMs, 1e-3f))} " +
                      $"creatures at 60 FPS\n" +
                      $"  {policySteps} policy evaluations total. Editor timings include " +
                      "the editor's own overhead, so a player build is faster.");

            Assert.That(meanMs, Is.LessThan(budgetMs),
                $"{count} creatures averaged {meanMs:F2} ms per frame " +
                $"({1000f / meanMs:F1} FPS), over the {budgetMs:F2} ms budget for 60 FPS.");
        }

        // ==================================================== decimation diagnostics ==
        /// <summary>
        /// The project step divides policy_dt exactly, so this fixture would never exercise
        /// the error path. Force a step that does not divide it and prove the agent says so
        /// loudly rather than quietly running at the wrong control rate.
        /// </summary>
        [UnityTest]
        public IEnumerator Decimation_NonIntegerRatioIsReportedAsAnError()
        {
            Time.fixedDeltaTime = 0.02f;             // 0.025 / 0.02 = 1.25
            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(
                @"policy_dt / Time\.fixedDeltaTime = 0\.025000 / 0\.020000 = 1\.25\d*, " +
                @"which is NOT an integer"));

            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            yield return null;

            Assert.That(agent.Decimation, Is.EqualTo(1),
                "1.25 rounds to 1, so the control rate should be 50 Hz, not 40 Hz");
            Time.fixedDeltaTime = _savedFixedDt;
        }

        /// <summary>The project step and the policy rate agree - the case that actually ships.</summary>
        [UnityTest]
        public IEnumerator Decimation_ProjectStepDividesPolicyDtExactly()
        {
            var agent = SpawnCreature(new Vector3(0f, 50f, 0f), zeroGravity: true);
            yield return null;

            float ratio = _rig.policyDt / _savedFixedDt;
            Debug.Log($"[decimation] policy_dt {_rig.policyDt:F4} s / fixedDeltaTime " +
                      $"{_savedFixedDt:F5} s = {ratio:F4} -> decimation {agent.Decimation}, " +
                      $"control rate {1f / (agent.Decimation * _savedFixedDt):F1} Hz " +
                      $"(MuJoCo ran {_rig.mujocoFrameSkip} substeps at " +
                      $"{_rig.mujocoPhysicsDt:F4} s).");

            Assert.That(Mathf.Abs(ratio - Mathf.Round(ratio)), Is.LessThan(1e-4f),
                $"policy_dt / fixedDeltaTime = {ratio:F6} is not an integer");
            Assert.That(agent.Decimation, Is.EqualTo(_rig.mujocoFrameSkip),
                "this project's fixed step is MuJoCo's own, so the decimation should equal " +
                "MuJoCo's frame skip exactly");
        }
    }
}
