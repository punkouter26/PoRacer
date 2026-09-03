using System;
using System.Collections.Generic;
using UnityEngine;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

using PoRacer.IsaacPorts;

namespace MujocoBiped
{
    /// <summary>
    /// Runs the biped_sentis MuJoCo PPO policy on a Unity ArticulationBody rig through
    /// Inference Engine.
    ///
    /// NOT through ML-Agents: this ONNX takes a bare <c>obs</c> float32[1,49] and returns
    /// a bare <c>action</c> float32[1,12]. It has no <c>obs_0</c>, no
    /// <c>continuous_actions</c>, no <c>version_number</c> and no <c>memory_size</c>
    /// tensors, so it cannot bind to BehaviorParameters at all. The project's ML-Agents
    /// creatures are untouched by this component.
    ///
    /// Contract, in one place:
    ///   obs     float32[1, 49]  built index-by-index as env.py's _get_obs()
    ///   action  float32[1, 12]  torque_i = action_i * gear_i, direct torque, no PD
    ///   rate    40 Hz policy; decimation = round(policyDt / Time.fixedDeltaTime)
    ///
    /// This component never writes a project-wide setting. Every physics value it needs
    /// that differs from the project default is applied per body or per collider.
    /// See CONTRACT.md and README_UNITY.md.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("MujocoBiped/MujocoBiped Agent")]
    public class MujocoBipedAgent : MonoBehaviour
    {
        /// <summary>
        /// How MuJoCo's passive joint damping reaches the solver. The ACTUATOR is always
        /// direct torque - the MJCF has motor actuators, not position servos, and there
        /// is no PD loop anywhere in the policy.
        /// </summary>
        public enum ActuatorMode
        {
            /// <summary>
            /// Damping through the ArticulationDrive (stiffness 0, damping = MJCF
            /// damping), which PhysX integrates IMPLICITLY and is unconditionally stable.
            /// Torque through jointForce, which writes into the joint's REDUCED space and
            /// is therefore exactly MuJoCo's actuator semantics. Shipped default.
            /// </summary>
            DirectTorqueImplicitDamping = 0,

            /// <summary>
            /// Diagnostic: drive damping zeroed and tau = action*gear - kd*qd written
            /// explicitly in C# through jointForce. A forward-Euler feedback term, stable
            /// only while kd*dt/I_joint &lt; 2 - RIG_AUDIT.md section C tabulates that per
            /// joint and per timestep. Use it to tell a drive problem from a torque problem.
            /// </summary>
            DirectTorqueExplicitDamping = 1,

            /// <summary>
            /// Torque applied as an equal-and-opposite PAIR through AddTorque on the child
            /// link and its parent, with MuJoCo's passive damping left to the implicit
            /// drive. This is the route the export's own README recommends, and unlike a
            /// single-link AddTorque it is a genuinely INTERNAL torque - the reaction lands
            /// on the parent rather than on the world, so rung 2's momentum check holds.
            ///
            /// Measured equal to jointForce on speed (rung 6: 0.248 m/s both) and
            /// materially WORSE under extreme actuation (rung 3: the project step diverges
            /// here and does not on jointForce), so it is the alternative, not the default.
            /// Reach for it if a future rig turns out not to respond to jointForce.
            /// </summary>
            TorquePairImplicitDamping = 2,
        }

        /// <summary>
        /// Whether ArticulationDrive.damping is applied against an error in radians per
        /// second or degrees per second.
        ///
        /// This is not a preference - it is a measured property of PhysX that Unity does
        /// not document. `ArticulationBody.jointPosition` and `jointVelocity` are
        /// unambiguously radians while `xDrive.target` and the limits are unambiguously
        /// degrees, so the drive straddles both conventions and the gain's own units
        /// belong to neither by inspection.
        ///
        /// Measured by rung D: writing MuJoCo's damping of 1.0 straight into the drive
        /// produces an effective 98 N.m.s/rad, roughly the 180/pi = 57.3 a degrees
        /// convention predicts (the rest is drive coupling through the parent). At 37 rad/s
        /// - the fastest joint velocity MuJoCo ever recorded - that is 3600 N.m of damping
        /// against a 110 N.m actuator. The creature wades instead of walking.
        /// </summary>
        public enum GainUnits
        {
            /// <summary>Write MuJoCo's damping straight through. Measured WRONG here;
            /// kept so rung D has something to contrast against.</summary>
            Radians = 0,

            /// <summary>Scale by Mathf.Deg2Rad, so the effective damping is MuJoCo's.
            /// Shipped default, chosen by measurement.</summary>
            Degrees = 1,
        }

        /// <summary>
        /// Which point's velocity fills obs[4:7]. MuJoCo's qvel[0:3] is unambiguously the
        /// body-frame ORIGIN, but ArticulationBody.linearVelocity reports the centre of
        /// mass, and the torso's CoM sits 0.185 m above its origin - so at 5 rad/s of
        /// pitch the two readings differ by nearly 1 m/s.
        /// </summary>
        public enum LinearVelocityReference
        {
            /// <summary>d/dt of qpos[0:3]. What MuJoCo recorded. Default.</summary>
            BodyFrameOrigin = 0,

            /// <summary>ArticulationBody.linearVelocity as-is. For the sweep.</summary>
            CenterOfMass = 1,
        }

        /// <summary>
        /// MuJoCo excludes only DIRECT parent-child geom pairs; everything else collided
        /// during training. That exclusion is not optional here - see
        /// <see cref="ApplySelfCollisionFiltering"/>.
        /// </summary>
        public enum SelfCollisionMode
        {
            /// <summary>Exclude parent-child pairs only. What MuJoCo did. Default.</summary>
            MujocoFaithful = 0,

            /// <summary>Exclude every internal pair. Cheaper, lets the legs pass through
            /// each other - a visible difference from training.</summary>
            None = 1,
        }

        public enum ActionOverride
        {
            None = 0,
            Constant = 1,
            SquareWave = 2,
        }

        // ------------------------------------------------------------------ setup --
        [Header("Policy")]
#if ISAACPORTS_HAS_INFERENCE
        [Tooltip("MujocoBiped.onnx - obs float32[1,49] -> action float32[1,12].")]
        public ModelAsset modelAsset;
#endif
        [Tooltip("Generated from MujocoBiped_rig.json. Every MuJoCo value this agent needs.")]
        public MujocoBipedRigAsset rig;

        [Header("Target")]
        [Tooltip("Explicit chase target. Takes priority over any ITargetProvider.")]
        public Transform target;

        [Header("Actuation")]
        public ActuatorMode actuatorMode = ActuatorMode.DirectTorqueImplicitDamping;

        [Tooltip("ArticulationDrive.damping units. Measured, not assumed - rung D. " +
                 "Degrees scales MuJoCo's damping by Deg2Rad so the joints are not " +
                 "over-damped by ~57x.")]
        public GainUnits gainUnits = GainUnits.Degrees;

        [Tooltip("Exact = every joint's H[i][i] gains exactly its MJCF armature. " +
                 "RIG_AUDIT.md section A.")]
        public MujocoBipedRigBuilder.ArmatureMode armatureMode =
            MujocoBipedRigBuilder.ArmatureMode.Exact;

        [Tooltip("No MJCF joint has a velocity limit, so this ships OFF and " +
                 "maxJointVelocity is a safety valve. RIG_AUDIT.md section B.")]
        public bool enforceVelocityLimit = false;

        [Tooltip("Applied when enforceVelocityLimit is false. Recorded peak is 37.14 rad/s.")]
        public float maxJointVelocity = 200f;

        [Tooltip("Scale on every actuator gear. 1 = MuJoCo's own torques. For the sweep.")]
        public float torqueScale = 1f;

        [Header("Observation")]
        public LinearVelocityReference linearVelocityReference =
            LinearVelocityReference.BodyFrameOrigin;

        [Header("Collision")]
        public SelfCollisionMode selfCollisionMode = SelfCollisionMode.MujocoFaithful;

        [Header("Fall recovery")]
        [Tooltip("env.py ends the episode when the torso leaves [0.55, 1.1] m or tips " +
                 "past 66 degrees, and resets. Unity has no episode, so without this a " +
                 "single fall leaves the creature on the floor for good.")]
        // OFF by default: a racer that falls stays on the ground, and getting
        // up is something the policy has to do for itself. Only a harness that
        // is deliberately measuring something else should turn this on.
        public bool autoRecoverFromFalls = false;

        [Tooltip("env.py min_uprightness. upright = R[2,2] = dot(root up, world up).")]
        [Range(0f, 1f)] public float fallUprightThreshold = 0.4f;

        [Tooltip("Seconds outside the healthy band before a recovery is triggered.")]
        public float fallGraceSeconds = 1f;

        [Header("Diagnostics")]
        public bool debugLogObservations = false;
        public int debugLogEveryNSteps = 40;

        public ActionOverride actionOverride = ActionOverride.None;

        [Tooltip("-1 applies the override to every joint; otherwise only this index.")]
        public int overrideJointIndex = -1;

        public float overrideAmplitude = 1f;
        public float overrideSquareWavePeriod = 1f;

        [Tooltip("Per-body useGravity = false. Never touches project-wide Physics.gravity.")]
        public bool zeroGravity = false;

        public bool showOnGuiReadout = false;

        // ------------------------------------------------------------------ state --
        ArticulationBody _root;
        ArticulationBody[] _joints;            // indexed by MuJoCo joint index
        ArticulationBody[] _allBodies;
        Collider[] _allColliders;
        MujocoBipedJointDef[] _jointDefs;
        MujocoBipedLinkDef[] _linkDefs;        // parallel to _allBodies
        float[] _gear, _damping;
        Vector3[] _jointAxisLocal;             // joint axis in the CHILD link's own frame
        ArticulationBody[] _jointParents;      // the link the reaction torque belongs to
        float[] _obs;
        float[] _action;

        int _substep;
        int _decimation = 1;
        int _policySteps;
        float _wallTime;
        float _unhealthyFor;
        int _recoveries;
        Vector3 _homePosition;
        bool _homeCaptured;
        bool _ready;
        bool _hierarchyOk;

#if ISAACPORTS_HAS_INFERENCE
        Model _model;
        Worker _worker;
        Tensor<float> _input;
#endif

        public int Decimation => _decimation;
        public int PolicySteps => _policySteps;
        public int Recoveries => _recoveries;
        public bool IsReady => _ready;
        public ArticulationBody Root => _root;
        public IReadOnlyList<ArticulationBody> Joints => _joints;
        public float[] LatestObservation => _obs;
        public float[] LatestAction => _action;

        /// <summary>Whole-creature centre of mass velocity, mass-weighted.</summary>
        public Vector3 CenterOfMassVelocity
        {
            get
            {
                if (_allBodies == null) return Vector3.zero;
                Vector3 p = Vector3.zero;
                float m = 0f;
                for (int i = 0; i < _allBodies.Length; i++)
                {
                    p += _allBodies[i].linearVelocity * _allBodies[i].mass;
                    m += _allBodies[i].mass;
                }
                return m > 0f ? p / m : Vector3.zero;
            }
        }

        public Vector3 CenterOfMassPosition
        {
            get
            {
                if (_allBodies == null) return transform.position;
                Vector3 p = Vector3.zero;
                float m = 0f;
                for (int i = 0; i < _allBodies.Length; i++)
                {
                    p += _allBodies[i].worldCenterOfMass * _allBodies[i].mass;
                    m += _allBodies[i].mass;
                }
                return m > 0f ? p / m : transform.position;
            }
        }

        /// <summary>env.py's uprightness term: R[2,2], the torso's up axis against world up.</summary>
        public float Uprightness => _root != null
            ? Vector3.Dot(_root.transform.up, Vector3.up)
            : 0f;

        /// <summary>obs[0]: the torso body-frame origin height, which is what qpos[2] is.</summary>
        public float TorsoHeight => _root != null ? _root.transform.position.y : 0f;

        public bool IsHealthy => _root != null
                                 && TorsoHeight > rig.healthyZRange.x
                                 && TorsoHeight < rig.healthyZRange.y
                                 && Uprightness > rig.minUprightness;

        // ------------------------------------------------------------------- init --
        void Awake()
        {
            if (rig == null || rig.links == null || rig.links.Length == 0)
            {
                Debug.LogError($"[{name}] MujocoBipedAgent has no rig asset; disabling.", this);
                enabled = false;
                return;
            }

            useGUILayout = false;
            CacheHierarchy();
            if (!_hierarchyOk) { enabled = false; return; }

            ApplyPerBodyOverrides();
            ApplySelfCollisionFiltering();
            ResetToSpawnPose();
        }

        void CacheHierarchy()
        {
            _hierarchyOk = false;
            _allBodies = GetComponentsInChildren<ArticulationBody>(true);
            _allColliders = GetComponentsInChildren<Collider>(true);

            var byName = new Dictionary<string, ArticulationBody>(_allBodies.Length);
            for (int i = 0; i < _allBodies.Length; i++) byName[_allBodies[i].name] = _allBodies[i];

            int n = rig.jointOrder.Length;
            _joints = new ArticulationBody[n];
            _jointDefs = new MujocoBipedJointDef[n];
            _gear = new float[n];
            _damping = new float[n];
            _jointAxisLocal = new Vector3[n];
            _jointParents = new ArticulationBody[n];
            _linkDefs = new MujocoBipedLinkDef[_allBodies.Length];

            for (int i = 0; i < rig.links.Length; i++)
            {
                var def = rig.links[i];
                if (!byName.TryGetValue(def.name, out var body))
                {
                    Debug.LogError($"[{name}] the rig expects a link named '{def.name}' but " +
                                   "the hierarchy has none. Rebuild the prefab with " +
                                   "MujocoBiped > Build Prefab.", this);
                    return;
                }

                for (int b = 0; b < _allBodies.Length; b++)
                {
                    if (ReferenceEquals(_allBodies[b], body)) { _linkDefs[b] = def; break; }
                }

                if (def.isRoot) _root = body;
                if (!def.hasJoint) continue;

                int j = def.joint.index;
                _joints[j] = body;
                _jointDefs[j] = def.joint;
                _gear[j] = def.joint.gear;
                _damping[j] = def.joint.damping;
                _jointAxisLocal[j] = MujocoBipedFrameMap.Axis(def.joint.axisInChildMuj).normalized;
                _jointParents[j] = body.transform.parent != null
                    ? body.transform.parent.GetComponentInParent<ArticulationBody>()
                    : null;
            }

            if (_root == null)
            {
                Debug.LogError($"[{name}] no articulation root found in the hierarchy.", this);
                return;
            }
            for (int j = 0; j < _joints.Length; j++)
            {
                if (_joints[j] == null)
                {
                    Debug.LogError($"[{name}] no link carries joint index {j} " +
                                   $"('{rig.jointOrder[j]}').", this);
                    return;
                }
            }

            _obs = new float[rig.obsDim];
            _action = new float[rig.actDim];
            _hierarchyOk = true;
        }

        /// <summary>
        /// Everything here comes from the MJCF and is applied PER BODY / PER COLLIDER.
        /// Nothing in this method may become a project-wide setting.
        /// </summary>
        void ApplyPerBodyOverrides()
        {
            var p = rig.physics;
            for (int i = 0; i < _allBodies.Length; i++)
            {
                var b = _allBodies[i];
                b.linearDamping = p.linearDamping;
                b.angularDamping = p.angularDamping;
                b.jointFriction = p.jointFriction;
                b.maxLinearVelocity = p.maxLinearVelocity;
                b.maxAngularVelocity = p.maxAngularVelocity;
                b.maxDepenetrationVelocity = p.maxDepenetrationVelocity;
                b.solverIterations = p.solverPositionIterations;
                b.solverVelocityIterations = p.solverVelocityIterations;
                b.useGravity = !zeroGravity;
            }

            for (int j = 0; j < _joints.Length; j++)
            {
                // No MJCF joint carries a velocity limit, so "enforce" means the
                // Unity-side valve; otherwise the link angular cap.
                _joints[j].maxJointVelocity = enforceVelocityLimit
                    ? maxJointVelocity
                    : p.maxAngularVelocity;
            }

            for (int i = 0; i < _allColliders.Length; i++)
                _allColliders[i].contactOffset = p.contactOffset;

            ApplyMassAndInertia();
            ApplyDriveGains();
        }

        /// <summary>
        /// Re-applies mass, centre of mass and inertia through the same code the builder
        /// used, so changing armatureMode on an instance actually changes the physics
        /// rather than silently disagreeing with the baked prefab.
        /// </summary>
        void ApplyMassAndInertia()
        {
            for (int i = 0; i < _allBodies.Length; i++)
            {
                var def = _linkDefs[i];
                if (def == null) continue;
                MujocoBipedRigBuilder.ComposeMassAndInertia(def, rig, armatureMode,
                    out float mass, out Vector3 com, out Vector3 inertia);
                _allBodies[i].mass = mass;
                _allBodies[i].centerOfMass = com;
                _allBodies[i].inertiaTensor = inertia;
                _allBodies[i].inertiaTensorRotation = Quaternion.identity;
            }
        }

        /// <summary>Deg2Rad when the drive turns out to be a per-degree gain. See
        /// <see cref="GainUnits"/>.</summary>
        public float DampingScale => gainUnits == GainUnits.Degrees ? Mathf.Deg2Rad : 1f;

        void ApplyDriveGains()
        {
            bool implicitDamping = actuatorMode != ActuatorMode.DirectTorqueExplicitDamping;
            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                var d = b.xDrive;
                var def = _jointDefs[j];

                d.lowerLimit = def.lowerRad * Mathf.Rad2Deg;
                d.upperLimit = def.upperRad * Mathf.Rad2Deg;

                // Stiffness stays zero in BOTH modes. The policy emits torque; a position
                // servo would fight it.
                d.stiffness = 0f;
                d.damping = implicitDamping ? def.damping * DampingScale : 0f;
                d.target = 0f;
                d.targetVelocity = 0f;
                d.driveType = ArticulationDriveType.Force;
                // Large but finite - float.MaxValue destabilises PhysX at some timesteps.
                // See MujocoBipedRigBuilder.DriveForceLimit.
                d.forceLimit = MujocoBipedRigBuilder.DriveForceLimit;
                b.xDrive = d;
            }
        }

        /// <summary>
        /// MuJoCo excludes DIRECT parent-child geom pairs and nothing else. This is not a
        /// tuning knob: the pelvis capsule (r = 0.085, spanning y = -0.09..0.09) and each
        /// thigh capsule (r = 0.06, starting 0.02 m away at the hip) overlap by more than
        /// 0.12 m at the spawn pose. PhysX would normally suppress that as an adjacent
        /// articulation pair, but the two links are NOT adjacent here - the dummy chain
        /// that carries hip_z and hip_x sits between them - so without this filtering the
        /// creature detonates on the first physics tick.
        /// </summary>
        void ApplySelfCollisionFiltering()
        {
            int pairs = 0;
            for (int i = 0; i < _allColliders.Length; i++)
            {
                for (int k = i + 1; k < _allColliders.Length; k++)
                {
                    if (!ShouldIgnorePair(_allColliders[i], _allColliders[k])) continue;
                    Physics.IgnoreCollision(_allColliders[i], _allColliders[k], true);
                    pairs++;
                }
            }
            if (debugLogObservations)
                Debug.Log($"[{name}] self-collision: ignored {pairs} of " +
                          $"{_allColliders.Length * (_allColliders.Length - 1) / 2} pairs " +
                          $"({selfCollisionMode}).", this);
        }

        bool ShouldIgnorePair(Collider a, Collider b)
        {
            if (selfCollisionMode == SelfCollisionMode.None) return true;

            string bodyA = MujocoBodyOf(a), bodyB = MujocoBodyOf(b);
            if (bodyA == null || bodyB == null) return false;
            if (bodyA == bodyB) return true;                       // MuJoCo: same body
            return IsMujocoParentChild(bodyA, bodyB);
        }

        /// <summary>The MuJoCo body a collider belongs to, via its owning ArticulationBody.</summary>
        string MujocoBodyOf(Collider c)
        {
            var body = c.GetComponentInParent<ArticulationBody>();
            if (body == null) return null;
            for (int i = 0; i < _allBodies.Length; i++)
            {
                if (ReferenceEquals(_allBodies[i], body))
                    return _linkDefs[i] != null ? _linkDefs[i].mjBody : null;
            }
            return null;
        }

        /// <summary>
        /// Direct parent-child in the MUJOCO body tree, which is not the same as adjacency
        /// in the Unity link chain: torso and thigh_l are parent and child in MuJoCo but
        /// three links apart in Unity.
        /// </summary>
        bool IsMujocoParentChild(string a, string b)
        {
            return MujocoParentOf(a) == b || MujocoParentOf(b) == a;
        }

        string MujocoParentOf(string mjBody)
        {
            // The first link of a MuJoCo body carries the body's own offset; its Unity
            // parent belongs to the parent MuJoCo body.
            for (int i = 0; i < rig.links.Length; i++)
            {
                var l = rig.links[i];
                if (l.mjBody != mjBody || l.isRoot) continue;
                var parent = rig.Link(l.parent);
                if (parent != null && parent.mjBody != mjBody) return parent.mjBody;
            }
            return null;
        }

        void Start()
        {
            ComputeDecimation();
            if (_root != null)
            {
                _homePosition = _root.transform.position;
                _homeCaptured = true;
            }
#if ISAACPORTS_HAS_INFERENCE
            if (modelAsset == null)
            {
                Debug.LogError($"[{name}] no ModelAsset assigned; the creature will not move.", this);
                return;
            }
            _model = ModelLoader.Load(modelAsset);
            _worker = new Worker(_model, BackendType.CPU);
            _input = new Tensor<float>(new TensorShape(1, rig.obsDim));
            _policySteps = 0;
            _ready = true;
#else
            Debug.LogError($"[{name}] com.unity.ai.inference is not installed, so no policy " +
                           "can run. Add it via Window > Package Manager.", this);
#endif
        }

        void ComputeDecimation()
        {
            float fdt = Time.fixedDeltaTime;
            float ratio = rig.policyDt / fdt;
            _decimation = Mathf.Max(1, Mathf.RoundToInt(ratio));

            if (Mathf.Abs(ratio - _decimation) > 1e-4f)
            {
                // The only global change worth proposing is the nearest step that divides
                // policy_dt exactly. Ship anyway, with rounded decimation.
                int k = Mathf.Max(1, Mathf.CeilToInt(ratio));
                Debug.LogError(
                    $"[{name}] policy_dt / Time.fixedDeltaTime = {rig.policyDt:F6} / {fdt:F6} = " +
                    $"{ratio:F6}, which is NOT an integer. Running with rounded decimation " +
                    $"{_decimation}, so the control rate is {1f / (_decimation * fdt):F3} Hz " +
                    $"instead of {1f / rig.policyDt:F3} Hz. The nearest fixed step that divides " +
                    $"policy_dt exactly is {rig.policyDt / k:F6} s (decimation {k}). This agent " +
                    "will not change Time.fixedDeltaTime - that is a project-wide setting.", this);
            }
            else if (Mathf.Abs(fdt - rig.mujocoPhysicsDt) > 1e-6f)
            {
                Debug.LogWarning(
                    $"[{name}] the control rate is exact at {1f / rig.policyDt:F1} Hz " +
                    $"(decimation {_decimation}), but MuJoCo integrated at " +
                    $"{rig.mujocoPhysicsDt:F6} s and this project runs {fdt:F6} s, so each " +
                    $"control step gets {_decimation} substeps instead of " +
                    $"{rig.mujocoFrameSkip}. Fidelity, not correctness.", this);
            }
        }

        // ------------------------------------------------------------------- loop --
        void FixedUpdate()
        {
            if (!_ready) return;

            if (_substep == 0)
            {
                BuildObservations();
                RunPolicy();
                _policySteps++;
            }

            // MuJoCo holds ctrl constant across all frame_skip substeps, so the torque is
            // re-applied every physics tick, not only on policy ticks.
            ApplyTorques();

            if (autoRecoverFromFalls) TickFallRecovery();

            _substep++;
            if (_substep >= _decimation) _substep = 0;
            _wallTime += Time.fixedDeltaTime;
        }

        /// <summary>
        /// Stands the creature back up, the way env.py's healthy-band termination resets
        /// an episode. Keeps the planar position and the heading, so it carries on from
        /// where it went down rather than teleporting home.
        /// </summary>
        void TickFallRecovery()
        {
            if (IsHealthy) { _unhealthyFor = 0f; return; }

            _unhealthyFor += Time.fixedDeltaTime;
            if (_unhealthyFor < fallGraceSeconds) return;
            _unhealthyFor = 0f;
            _recoveries++;

            Vector3 p = _root.transform.position;
            float yaw = _root.transform.eulerAngles.y;
            RespawnAt(GroundedRespawnPoint(p), Quaternion.Euler(0f, yaw, 0f));

            Debug.LogWarning($"[{name}] left the healthy band (height {TorsoHeight:F2} m, " +
                             $"upright {Uprightness:F2}); stood back up. Recovery " +
                             $"#{_recoveries}. env.py would have ended the episode here - " +
                             $"its own eval survived the full 25 s in only " +
                             $"{rig.mujocoSurvivedFullEpisodeFraction:P0} of rollouts.", this);
        }

        /// <summary>
        /// Filled index-by-index in exactly the order env.py's _get_obs concatenates its
        /// terms. Every clip is part of the trained contract, not a safety measure: 33 of
        /// the 1800 recorded joint-velocity samples exceed the +/-20 clip, so the policy
        /// has only ever seen the clipped value.
        /// </summary>
        void BuildObservations()
        {
            Quaternion invRot = Quaternion.Inverse(_root.transform.rotation);

            // [0] torso_height = qpos[2]: the body-frame ORIGIN, not the centre of mass.
            _obs[0] = _root.transform.position.y;

            // [1:4] projected_gravity = R^T (0,0,-1). Read from Physics.gravity so a
            // rotated-gravity scene stays correct. zeroGravity is a per-body flag and
            // deliberately does NOT change this term: the policy uses it to sense
            // orientation, not weight, and rung 2 depends on that still being true.
            Vector3 gWorld = Physics.gravity.sqrMagnitude > 1e-8f
                ? Physics.gravity.normalized
                : Vector3.down;
            Vector3 g = MujocoBipedFrameMap.PosToMujoco(invRot * gWorld);
            _obs[1] = g.x; _obs[2] = g.y; _obs[3] = g.z;

            // [4:7] linear_velocity = clip(R^T qvel[0:3], +/-10). qvel[0:3] is the WORLD
            // velocity of the body-frame origin; ArticulationBody.linearVelocity is the
            // centre-of-mass velocity, so subtract w x r to get back to the origin.
            Vector3 vWorld = _root.linearVelocity;
            if (linearVelocityReference == LinearVelocityReference.BodyFrameOrigin)
            {
                // r = worldCenterOfMass - transform.position, but written from the
                // transform so it is correct the instant the root is moved. Reading
                // ArticulationBody.worldCenterOfMass instead makes this term lag a physics
                // step behind a teleport, which the observation-parity test catches
                // immediately and a running creature would only show as a subtle wrongness.
                Vector3 r = _root.transform.rotation * _root.centerOfMass;
                vWorld -= Vector3.Cross(_root.angularVelocity, r);
            }
            Vector3 v = MujocoBipedFrameMap.PosToMujoco(invRot * vWorld);
            _obs[4] = Mathf.Clamp(v.x, -rig.clipLinVel, rig.clipLinVel);
            _obs[5] = Mathf.Clamp(v.y, -rig.clipLinVel, rig.clipLinVel);
            _obs[6] = Mathf.Clamp(v.z, -rig.clipLinVel, rig.clipLinVel);

            // [7:10] angular_velocity. MuJoCo stores a free joint's qvel[3:6] in the BODY
            // frame, and env.py applies rot.T to it anyway - so this term is rotated into
            // the torso frame TWICE. That is not defensible modelling, but the policy
            // trained on it for 14.5M steps, so reproduce it exactly. Proven in
            // RIG_AUDIT.md section D. Angular velocity is a pseudovector, hence
            // AxisToMujoco rather than PosToMujoco.
            Vector3 wLocal = invRot * _root.angularVelocity;
            if (rig.angularVelocityIsDoubleRotated) wLocal = invRot * wLocal;
            Vector3 w = MujocoBipedFrameMap.AxisToMujoco(wLocal);
            _obs[7] = Mathf.Clamp(w.x, -rig.clipAngVel, rig.clipAngVel);
            _obs[8] = Mathf.Clamp(w.y, -rig.clipAngVel, rig.clipAngVel);
            _obs[9] = Mathf.Clamp(w.z, -rig.clipAngVel, rig.clipAngVel);

            // [10:22] qpos[7:] and [22:34] clip(qvel[6:], +/-20). Both radians:
            // ArticulationBody.jointPosition/jointVelocity are radians for a revolute
            // joint even though xDrive limits are degrees. No sign flip is needed because
            // every anchor's +X was built at -M*axis.
            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                _obs[10 + j] = b.jointPosition[0];
                _obs[22 + j] = Mathf.Clamp(b.jointVelocity[0], -rig.clipJointVel, rig.clipJointVel);
            }

            // [34:36] target_direction and [36] target_distance, both PLANAR and both
            // rotated by yaw ONLY - never by the full orientation, which is what keeps the
            // policy heading-invariant.
            Vector3 here = _root.transform.position;
            float dist = 0f;
            float dirX = 0f, dirY = 0f;
            if (TryGetTargetWorld(out Vector3 tgt))
            {
                Vector3 to = tgt - here;
                to.y = 0f;
                dist = to.magnitude;
                Vector3 unit = to / Mathf.Max(dist, 1e-6f);
                // MuJoCo XY of a Unity XZ direction: (x, y)_muj = (z, -x)_unity.
                float wx = unit.z, wy = -unit.x;
                float heading = MujocoBipedFrameMap.HeadingRad(_root.transform.rotation);
                float c = Mathf.Cos(-heading), s = Mathf.Sin(-heading);
                dirX = c * wx - s * wy;
                dirY = s * wx + c * wy;
            }
            _obs[34] = dirX;
            _obs[35] = dirY;
            _obs[36] = Mathf.Min(dist, rig.maxTargetDistance);

            // [37:49] the previous step's network output, post-clamp. Zeros after a reset.
            Array.Copy(_action, 0, _obs, 37, rig.actDim);

            if (debugLogObservations && _policySteps % Mathf.Max(1, debugLogEveryNSteps) == 0)
                LogObservations();
        }

        bool TryGetTargetWorld(out Vector3 world)
        {
            if (target != null) { world = target.position; return true; }
            // GetComponent ignores the enabled flag, so check it - a disabled provider
            // must not keep steering the creature.
            var provider = GetComponent<ITargetProvider>();
            if (provider != null && (!(provider is Behaviour bh) || bh.isActiveAndEnabled))
                return provider.TryGetTarget(out world);
            world = default;
            return false;
        }

        void RunPolicy()
        {
#if ISAACPORTS_HAS_INFERENCE
            _input.Upload(_obs);
            _worker.Schedule(_input);
            var output = _worker.PeekOutput() as Tensor<float>;
            // Complete once, then index. DownloadToArray allocates a managed array every
            // call and is reserved for the reference check.
            output.CompleteAllPendingOperations();
            for (int i = 0; i < rig.actDim; i++) _action[i] = output[0, i];
#endif
            ApplyActionOverride();
        }

        void ApplyActionOverride()
        {
            if (actionOverride == ActionOverride.None) return;

            float value;
            if (actionOverride == ActionOverride.Constant)
            {
                value = overrideAmplitude;
            }
            else
            {
                float period = Mathf.Max(1e-4f, overrideSquareWavePeriod);
                bool high = (_wallTime % period) < period * 0.5f;
                value = high ? overrideAmplitude : -overrideAmplitude;
            }

            for (int i = 0; i < _action.Length; i++)
                _action[i] = (overrideJointIndex < 0 || overrideJointIndex == i) ? value : 0f;
        }

        /// <summary>
        /// torque_i = action_i * gear_i. No target integration, no PD, no clamp beyond the
        /// [-1, 1] the network already applies - the MJCF actuators are motors.
        /// </summary>
        void ApplyTorques()
        {
            bool explicitDamping = actuatorMode == ActuatorMode.DirectTorqueExplicitDamping;
            bool pair = actuatorMode == ActuatorMode.TorquePairImplicitDamping;

            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                float tau = _action[j] * _gear[j] * torqueScale;
                if (explicitDamping) tau -= _damping[j] * b.jointVelocity[0];

                if (!pair)
                {
                    b.jointForce = new ArticulationReducedSpace(tau);
                    continue;
                }

                // Equal and opposite about the joint axis, in world space. The axis is
                // stored in the child's own frame, so rotate it out by the child's current
                // rotation rather than caching a world axis - the leg swings.
                Vector3 axisWorld = b.transform.rotation * _jointAxisLocal[j];
                Vector3 t = axisWorld * tau;
                b.AddTorque(t, ForceMode.Force);
                // The reaction belongs on the parent, not on the world. Without it this is
                // an EXTERNAL torque and the creature can spin itself up from nothing.
                if (_jointParents[j] != null) _jointParents[j].AddTorque(-t, ForceMode.Force);
            }
        }

        // -------------------------------------------------------------- utilities --
        /// <summary>
        /// Re-applies every setting otherwise read only in Awake. Instantiate() runs Awake
        /// immediately, so a field changed on the component straight afterwards -
        /// armatureMode, actuatorMode, zeroGravity, selfCollisionMode - would otherwise be
        /// silently ignored. The rung-6 sweep depends on this.
        /// </summary>
        public void Reconfigure()
        {
            if (_allBodies == null) return;
            ApplyPerBodyOverrides();
            ApplySelfCollisionFiltering();
            ResetToSpawnPose();
        }

        /// <summary>
        /// init_qpos wants every hinge at zero - but the knees cannot go there. Their MJCF
        /// range is [-150, -2] degrees, so zero is 2 degrees OUTSIDE it, and MuJoCo's reset
        /// noise pushes them further out still (the recording starts with knee_r at
        /// +0.0138 rad). MuJoCo gets away with that because its joint limits are soft
        /// constraints it can violate and then relax; Unity's ArticulationDrive limits are
        /// hard, and a jointPosition set outside them is fought by the solver from the
        /// first tick.
        ///
        /// So the spawn pose is init_qpos CLAMPED into each joint's own range. For the
        /// knees that is 2 degrees of bend, which costs 0.24 mm of standing height.
        /// </summary>
        public float SpawnPoseRad(int jointIndex)
        {
            var d = _jointDefs[jointIndex];
            return Mathf.Clamp(0f, d.lowerRad, d.upperRad);
        }

        /// <summary>Drives every joint to the clamped init_qpos and clears velocities.</summary>
        public void ResetToSpawnPose()
        {
            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                if (b == null) continue;
                b.jointPosition = new ArticulationReducedSpace(SpawnPoseRad(j));
                b.jointVelocity = new ArticulationReducedSpace(0f);
                b.jointForce = new ArticulationReducedSpace(0f);
                var d = b.xDrive;
                d.target = 0f;
                b.xDrive = d;
            }
            if (_action != null) Array.Clear(_action, 0, _action.Length);
            _substep = 0;
            _wallTime = 0f;
            _unhealthyFor = 0f;
        }

        /// <summary>Teleports the whole articulation without the solver fighting it.</summary>
        /// <summary>
        /// Where to stand the creature back up. Recovery used to keep the planar position
        /// and only reset the height, which is right for a creature that fell over on the
        /// track - but a creature that walked off the edge of the ground has no floor under
        /// that position, so it free-fell, respawned mid-air at the same x/z, and fell
        /// again forever. Observed in SCN_RACE_FLAT: 24 consecutive recoveries at -6.9 m,
        /// one every 1.26 s, which is exactly free fall plus fallGraceSeconds.
        ///
        /// So probe for ground first, and fall back to where the rig started if there is
        /// none. Only the planar position falls back - the height always comes from the
        /// rig's own spawn height.
        /// </summary>
        Vector3 GroundedRespawnPoint(Vector3 current)
        {
            const float PROBE_START_HEIGHT = 50f;
            const float PROBE_DISTANCE = 200f;

            var above = new Vector3(current.x, PROBE_START_HEIGHT, current.z);
            if (Physics.Raycast(above, Vector3.down, out RaycastHit hit, PROBE_DISTANCE,
                                ~0, QueryTriggerInteraction.Ignore)
                && !hit.transform.IsChildOf(transform))
            {
                return new Vector3(current.x, hit.point.y + rig.SpawnHeight, current.z);
            }

            if (_homeCaptured)
            {
                return new Vector3(_homePosition.x, rig.SpawnHeight, _homePosition.z);
            }

            return new Vector3(current.x, rig.SpawnHeight, current.z);
        }

        public void RespawnAt(Vector3 position, Quaternion rotation)
        {
            if (_root == null) return;
            _root.TeleportRoot(position, rotation);
            _root.linearVelocity = Vector3.zero;
            _root.angularVelocity = Vector3.zero;
            ResetToSpawnPose();
        }

        /// <summary>
        /// Drives every joint to a pose in MuJoCo joint order, for the kinematics test.
        /// Positions are radians and are applied with the same sign convention the
        /// observations read back, which is the point of the test.
        /// </summary>
        public void SetJointPositionsRad(float[] radians)
        {
            if (_joints == null || radians == null) return;
            int n = Mathf.Min(radians.Length, _joints.Length);
            for (int j = 0; j < n; j++)
            {
                var b = _joints[j];
                b.jointPosition = new ArticulationReducedSpace(radians[j]);
                b.jointVelocity = new ArticulationReducedSpace(0f);
                b.jointForce = new ArticulationReducedSpace(0f);
            }
        }

        /// <summary>
        /// Rebuilds the observation vector from the rig's CURRENT state and returns it.
        /// The observation-parity test uses this to put the rig into a state MuJoCo
        /// recorded and check that the 49 floats come back the way MuJoCo wrote them -
        /// which is the only way to catch a sign or a frame error in a term the policy
        /// merely degrades on rather than crashes on.
        /// </summary>
        public float[] CaptureObservation()
        {
            BuildObservations();
            return _obs;
        }

        /// <summary>Injects obs[37:49] for the parity test, which replays recorded steps
        /// out of order and so cannot rely on the agent's own last action.</summary>
        public void SetLastActionForTest(float[] action)
        {
            if (action == null || _action == null) return;
            Array.Copy(action, _action, Mathf.Min(action.Length, _action.Length));
        }

        /// <summary>
        /// Places the rig in a MuJoCo state: root pose and velocity, joint positions and
        /// velocities. Everything in, everything out is MuJoCo convention - positions and
        /// linear velocity as true vectors, the angular velocity BODY-LOCAL exactly as
        /// qvel[3:6] stores it.
        /// </summary>
        public void SetMujocoState(Vector3 rootPosMuj, Vector4 rootQuatMujWxyz,
                                   Vector3 rootLinVelWorldMuj, Vector3 rootAngVelBodyLocalMuj,
                                   float[] jointPosRad, float[] jointVelRad)
        {
            Quaternion rot = MujocoBipedFrameMap.RotFromWxyz(
                rootQuatMujWxyz.x, rootQuatMujWxyz.y, rootQuatMujWxyz.z, rootQuatMujWxyz.w);
            Vector3 pos = MujocoBipedFrameMap.Pos(rootPosMuj);

            // TeleportRoot moves the articulation without the solver fighting it, but it
            // does not update the Transform until the next physics step - and this method
            // exists precisely so the observation can be read WITHOUT stepping. So write
            // the Transform too, then sync it into PhysX (this project runs with
            // autoSyncTransforms off, so that sync is not automatic).
            _root.TeleportRoot(pos, rot);
            _root.transform.SetPositionAndRotation(pos, rot);
            Physics.SyncTransforms();

            // qvel[3:6] is body-local, so rotate it into the world before handing it to
            // Unity, which reports angularVelocity in world coordinates.
            Vector3 wWorld = rot * MujocoBipedFrameMap.Axis(rootAngVelBodyLocalMuj);
            _root.angularVelocity = wWorld;

            // qvel[0:3] is the ORIGIN's world velocity; ArticulationBody.linearVelocity is
            // the centre of mass's, so add w x r on the way in - the exact inverse of what
            // BuildObservations subtracts on the way out, and written the same way so the
            // two cannot drift apart.
            Vector3 vOrigin = MujocoBipedFrameMap.Pos(rootLinVelWorldMuj);
            _root.linearVelocity = vOrigin + Vector3.Cross(wWorld, rot * _root.centerOfMass);

            for (int j = 0; j < _joints.Length; j++)
            {
                _joints[j].jointPosition = new ArticulationReducedSpace(jointPosRad[j]);
                _joints[j].jointVelocity = new ArticulationReducedSpace(jointVelRad[j]);
                _joints[j].jointForce = new ArticulationReducedSpace(0f);
            }
        }

        /// <summary>
        /// Frees or re-limits every joint. The kinematics test needs this: MuJoCo's joint
        /// limits are soft constraints, so its own reference poses can sit outside them
        /// (the zero pose has both knees 2 degrees past their limit), and Unity's hard
        /// limits would drag the rig off that pose before it could be measured. Freeing
        /// the twist lock isolates the frame map, which is what that test is about.
        /// Always restore it - a rig left unlimited will hyperextend under load.
        /// </summary>
        public void SetJointLimitsEnabled(bool enabledLimits)
        {
            for (int j = 0; j < _joints.Length; j++)
            {
                _joints[j].twistLock = enabledLimits
                    ? ArticulationDofLock.LimitedMotion
                    : ArticulationDofLock.FreeMotion;
            }
        }

        /// <summary>The link named <paramref name="n"/>, or null.</summary>
        public ArticulationBody FindBody(string n)
        {
            if (_allBodies == null) return null;
            for (int i = 0; i < _allBodies.Length; i++)
                if (_allBodies[i].name == n) return _allBodies[i];
            return null;
        }

        /// <summary>
        /// Feeds the recorded observations through the live worker and returns the max abs
        /// difference against the recorded actions. Isolates the inference path from the
        /// physics path: this matching while the creature still walks wrong means the
        /// problem is physics, and only physics.
        /// </summary>
        public float RunReferenceCheck(float[][] recordedObs, float[][] recordedActions)
        {
#if ISAACPORTS_HAS_INFERENCE
            if (!_ready)
            {
                if (modelAsset == null) return float.NaN;
                _model = ModelLoader.Load(modelAsset);
                _worker = new Worker(_model, BackendType.CPU);
                _input = new Tensor<float>(new TensorShape(1, rig.obsDim));
                _ready = true;
            }

            float worst = 0f;
            for (int s = 0; s < recordedObs.Length; s++)
            {
                _input.Upload(recordedObs[s]);
                _worker.Schedule(_input);
                var output = _worker.PeekOutput() as Tensor<float>;
                // DownloadToArray is fine here - this is the reference check, not the loop.
                float[] got = output.DownloadToArray();
                for (int i = 0; i < recordedActions[s].Length; i++)
                    worst = Mathf.Max(worst, Mathf.Abs(got[i] - recordedActions[s][i]));
            }
            return worst;
#else
            return float.NaN;
#endif
        }

        void LogObservations()
        {
            var sb = new System.Text.StringBuilder(512);
            sb.Append('[').Append(name).Append("] step ").Append(_policySteps)
              .Append("  height ").Append(_obs[0].ToString("F3"))
              .Append("  upright ").Append(Uprightness.ToString("F3"))
              .Append("\n  grav  ").Append(V3(_obs, 1))
              .Append("\n  linv  ").Append(V3(_obs, 4))
              .Append("\n  angv  ").Append(V3(_obs, 7))
              .Append("\n  tgt   dir (").Append(_obs[34].ToString("F3")).Append(", ")
              .Append(_obs[35].ToString("F3")).Append(")  dist ")
              .Append(_obs[36].ToString("F2"));
            sb.Append("\n  qpos ");
            for (int j = 0; j < rig.actDim; j++) sb.Append(_obs[10 + j].ToString("F2")).Append(' ');
            sb.Append("\n  act  ");
            for (int j = 0; j < rig.actDim; j++) sb.Append(_action[j].ToString("F2")).Append(' ');
            Debug.Log(sb.ToString(), this);
        }

        static string V3(float[] a, int i)
            => $"({a[i]:F3}, {a[i + 1]:F3}, {a[i + 2]:F3})";

        void OnGUI()
        {
            if (!showOnGuiReadout) return;
            Vector3 v = CenterOfMassVelocity;
            GUI.Label(new Rect(10f, 10f, 520f, 130f),
                $"{name}   step {_policySteps}   decimation {_decimation}\n" +
                $"CoM velocity  {v.magnitude:F2} m/s   ({v.x:F2}, {v.y:F2}, {v.z:F2})\n" +
                $"torso height  {TorsoHeight:F3} m   upright {Uprightness:F3}   " +
                $"healthy {IsHealthy}\n" +
                $"target dist   {(_obs != null ? _obs[36] : 0f):F2} m   recoveries {_recoveries}\n" +
                $"actuator {actuatorMode}   armature {armatureMode}");
        }

        public void ReleaseWorker()
        {
#if ISAACPORTS_HAS_INFERENCE
            _worker?.Dispose();
            _worker = null;
            _input?.Dispose();
            _input = null;
            _model = null;
#endif
            _ready = false;
        }

        void OnDestroy() => ReleaseWorker();
    }
}
