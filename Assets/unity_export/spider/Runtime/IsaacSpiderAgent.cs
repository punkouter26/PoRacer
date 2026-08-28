using System;
using System.Text;
using UnityEngine;
#if ISAAC_SPIDER_INFERENCE
using Unity.InferenceEngine;
#endif

namespace IsaacSpider
{
    /// <summary>
    /// Runs the Isaac Lab spider walk-to-target policy (RSL-RL PPO export, <c>spider.onnx</c>) on a
    /// PhysX 4 ArticulationBody rig with the Inference Engine CPU backend. Self-contained: no
    /// ML-Agents, no VContainer. Numbers come from <c>checkpoint/params/env.yaml</c> and
    /// <c>export_report.json</c>; see CONTRACT.md for the observation layout and frame map.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class IsaacSpiderAgent : MonoBehaviour
    {
        public enum ActuatorMode
        {
            /// <summary>PhysX implicit spring-damper (ArticulationDrive) — what Isaac's ImplicitActuator is. Stable at any fixed step.</summary>
            ArticulationDrive,
            /// <summary>Explicit C# PD torque via jointForce every physics step. Needs Time.fixedDeltaTime ≤ 1/480 (see RIG_AUDIT.md).</summary>
            TorqueCSharp,
            /// <summary>No actuation at all (rig audit rung 1: free joints).</summary>
            Off,
        }

        public enum ActionOverrideMode { None, Constant, SquareWave }

        public const int OBS_DIM = 59;
        public const int ACT_DIM = 16;

        /// <summary>Joint order of the policy (export_report.json → joint_order).</summary>
        public static readonly string[] JointOrder =
        {
            "L1_hip", "L1_knee", "L2_hip", "L2_knee", "L3_hip", "L3_knee", "L4_hip", "L4_knee",
            "R1_hip", "R1_knee", "R2_hip", "R2_knee", "R3_hip", "R3_knee", "R4_hip", "R4_knee",
        };

        // ------------------------------------------------------------------ policy
        [Header("Policy")]
#if ISAAC_SPIDER_INFERENCE
        [SerializeField] private ModelAsset _model;
#endif
        [SerializeField] private TextAsset _isaacReference;
        [Tooltip("export_report.json → policy_dt (1/30 s). decimation = policyDt / Time.fixedDeltaTime and must be an integer.")]
        [SerializeField] private float _policyDt = 1f / 30f;
        [Tooltip("env.yaml → action_scale: q_target = actionScale · clamp(action, -1, 1) [rad]")]
        [SerializeField] private float _actionScale = 0.8f;

        // ------------------------------------------------------------------ actuator
        [Header("Actuator (env.yaml → actuators.legs)")]
        [SerializeField] private ActuatorMode _actuatorMode = ActuatorMode.ArticulationDrive;
        [SerializeField] private float _stiffness = 25f;
        [SerializeField] private float _damping = 1f;
        [SerializeField] private float _effortLimit = 15f;
        [Tooltip("env.yaml → velocity_limit_sim. The Isaac recording shows |q̇| up to 90 rad/s, so Isaac did NOT enforce it; leave enforcement off to match.")]
        [SerializeField] private float _velocityLimit = 12f;
        [SerializeField] private bool _enforceVelocityLimit = false;

        // ------------------------------------------------------------------ per-body physics
        [Header("Per-body physics (env.yaml → rigid_props / articulation_props)")]
        [SerializeField] private float _maxLinearVelocity = 100f;
        [SerializeField] private float _maxAngularVelocity = 100f;
        [SerializeField] private float _maxDepenetrationVelocity = 10f;
        [Tooltip("Isaac reference reached 90 rad/s: the joint cap must not bite. 100 rad/s = the link angular cap.")]
        [SerializeField] private float _maxJointVelocity = 100f;
        [SerializeField] private float _jointFriction = 0f;
        [SerializeField] private float _linearDamping = 0f;
        [Tooltip("PhysX default angular damping (env.yaml leaves it null).")]
        [SerializeField] private float _angularDamping = 0.05f;
        [SerializeField] private int _solverIterations = 8;
        [SerializeField] private int _solverVelocityIterations = 0;
        [Tooltip("Isaac/PhysX 5 default contact offset is 0.02 m; the project default is 0.01. Applied per collider, not project-wide.")]
        [SerializeField] private float _contactOffset = 0.02f;

        [Header("Rig floors (RIG_AUDIT.md). 0 = raw URDF")]
        [SerializeField] private float _massFloor = 0f;
        [SerializeField] private float _inertiaFloor = 1e-4f;
        [SerializeField] private string _layerName = "IsaacSpider";

        // ------------------------------------------------------------------ target
        [Header("Target")]
        [Tooltip("Optional. When unset and no ITargetProvider is assigned, the Isaac ring sampler (1.5–3.5 m) is used.")]
        [SerializeField] private Transform _target;
        [SerializeField] private Vector2 _targetRadiusRange = new Vector2(1.5f, 3.5f);
        [SerializeField] private float _reachThreshold = 0.3f;
        [SerializeField] private int _ringSamplerSeed = 42;

        [Header("Scene integration")]
        [Tooltip("Hold the root immovable (policy idle) until a downward probe finds ground. Needed in scenes that build their ground at runtime (PoRacer builds the track on START RACING). Re-holds and resets the pose if the spider ever falls below spawn − 3 m.")]
        [SerializeField] private bool _holdUntilGrounded = true;
        [Tooltip("How far below the body to look for anything to land on. Long on purpose: stacked spawn towers start racers metres up.")]
        [SerializeField] private float _groundProbeDistance = 50f;

        // ------------------------------------------------------------------ diagnostics
        [Header("Diagnostics")]
        [SerializeField] private bool _debugLogObservations;
        [SerializeField] private ActionOverrideMode _actionOverride = ActionOverrideMode.None;
        [Tooltip("x = hips high/constant, y = knees high/constant, z = hips low, w = knees low (square wave).")]
        [SerializeField] private Vector4 _overrideValues = new Vector4(0.5f, 0.5f, -0.5f, -0.5f);
        [SerializeField] private float _overridePeriod = 1f;
        [Tooltip("-1 = all joints; otherwise only this joint index gets the override, the others 0.")]
        [SerializeField] private int _overrideSingleJoint = -1;
        [SerializeField] private bool _zeroGravity;
        [SerializeField] private bool _showGui = true;

        // ------------------------------------------------------------------ runtime state
        private ArticulationBody _root;
        private ArticulationBody[] _bodies;
        private readonly ArticulationBody[] _joints = new ArticulationBody[ACT_DIM];
        private Collider[] _colliders;
        // Raw URDF values captured by the prefab builder so floors can be re-applied (and removed) at runtime.
        [SerializeField, HideInInspector] private float[] _urdfMass;
        [SerializeField, HideInInspector] private Vector3[] _urdfInertia;
        private readonly float[] _obs = new float[OBS_DIM];
        private readonly float[] _action = new float[ACT_DIM];
        private readonly float[] _prevAction = new float[ACT_DIM];
        private readonly float[] _jointTarget = new float[ACT_DIM];
        private int _decimation = 1;
        private int _stepCounter;
        private bool _ready;
        private Vector3 _ringOrigin;
        private Vector3 _ringTarget;
        private System.Random _ringRandom;
        private int _reached;
        private float _maxJointSpeedSeen;
        private ITargetProvider _targetProvider;
        private bool _held;
        private Vector3 _spawnPosition;
        private Quaternion _spawnRotation;
        private readonly RaycastHit[] _probeHits = new RaycastHit[32];
        private float _flippedSeconds;
        private StringBuilder _log;
        private readonly GUIContent _guiContent = new GUIContent();
        private string _guiText = string.Empty;
        private float _nextGuiRefresh;
#if ISAAC_SPIDER_INFERENCE
        private Worker _worker;
        private Tensor<float> _input;
#endif

        // ------------------------------------------------------------------ properties (read by tests / spawner)
        public ArticulationBody Root => _root;
        public ArticulationBody[] Bodies => _bodies;
        public ArticulationBody[] JointBodies => _joints;
        public int Decimation => _decimation;
        public int TargetsReached => _reached;
        public float MaxJointSpeedSeen => _maxJointSpeedSeen;
        public bool IsReady => _ready;
        /// <summary>Belly-up for more than a second - Isaac terminates the episode here; the policy cannot recover.</summary>
        public bool IsFlipped => _flippedSeconds > 1f;
        /// <summary>When true the agent resets its own pose after a flip/fall. A race adapter turns this off and lets RacerView rescue.</summary>
        public bool AutoRecover { get; set; } = true;
        public float BodyHeight => _root != null ? _root.transform.position.y : 0f;
        public Vector3 CurrentTargetWorld => ResolveTargetWorld();
        public ITargetProvider TargetProvider { get => _targetProvider; set => _targetProvider = value; }
        public ActuatorMode Actuator { get => _actuatorMode; set { _actuatorMode = value; if (_ready) { ConfigureDrives(); } } }
        public bool ZeroGravity
        {
            get => _zeroGravity;
            set
            {
                _zeroGravity = value;
                if (_ready)
                {
                    ApplyPhysicsSettings();
                    if (value && _held)
                    {
                        Release(); // nothing to fall through in zero-g
                    }
                }
            }
        }
        public bool ShowGui { get => _showGui; set => _showGui = value; }
        public float MassFloor { get => _massFloor; set => _massFloor = value; }
        public float InertiaFloor { get => _inertiaFloor; set => _inertiaFloor = value; }
        public bool HasPolicy
        {
            get
            {
#if ISAAC_SPIDER_INFERENCE
                return _model != null;
#else
                return false;
#endif
            }
        }

        // ------------------------------------------------------------------ unity lifecycle
        private void Awake()
        {
            _log = new StringBuilder(1024);
            if (!TryBindRig())
            {
                enabled = false;
                return;
            }
            CaptureUrdfValuesIfMissing();
            ApplyPhysicsSettings();
            ApplyCollisionFilters();
            ConfigureDrives();
            ResolveDecimation();
            _ringOrigin = _root.transform.position;
            _ringRandom = new System.Random(_ringSamplerSeed);
            SampleRingTarget();
            CreateWorker();
            _spawnPosition = _root.transform.position;
            _spawnRotation = _root.transform.rotation;
            if (_holdUntilGrounded && !_zeroGravity)
            {
                Hold();
            }
            _ready = true;
        }

        private void FixedUpdate()
        {
            if (!_ready)
            {
                return;
            }
            if (_holdUntilGrounded)
            {
                if (_held)
                {
                    if (ProbeGround())
                    {
                        Release();
                    }
                    return;
                }
                if (AutoRecover && _root.transform.position.y < _spawnPosition.y - 3f)
                {
                    Debug.LogWarning($"[{name}] fell below the spawn height (no ground?) — resetting to the spawn pose and holding until ground appears.");
                    ResetPoseAndHold();
                    return;
                }
                // Isaac terminates the episode when the body flips (projected gravity z > 0); the policy never learned to
                // recover, so after 1 s belly-up put it back on its feet at the spawn pose.
                _flippedSeconds = _root.transform.up.y < 0f ? _flippedSeconds + Time.fixedDeltaTime : 0f;
                if (AutoRecover && _flippedSeconds > 1f)
                {
                    Debug.LogWarning($"[{name}] flipped for > 1 s — resetting to the spawn pose.");
                    _flippedSeconds = 0f;
                    ResetPoseAndHold();
                    return;
                }
            }
            if (_stepCounter % _decimation == 0)
            {
                BuildObservation();
                ComputeAction();
                for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
                {
                    _jointTarget[jointIndex] = _actionScale * _action[jointIndex];
                }
                if (_actuatorMode == ActuatorMode.ArticulationDrive)
                {
                    for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
                    {
                        ArticulationDrive drive = _joints[jointIndex].xDrive;
                        drive.target = _jointTarget[jointIndex] * Mathf.Rad2Deg;
                        _joints[jointIndex].xDrive = drive;
                    }
                }
            }
            _stepCounter++;
            if (_actuatorMode == ActuatorMode.TorqueCSharp)
            {
                ApplyTorque();
            }
            TrackJointSpeed();
            UpdateTargetReach();
        }

        private void OnDestroy() => ReleaseWorker();

        /// <summary>Disposes the Inference Engine worker and input tensor (also used by the edit-mode reference check).</summary>
        public void ReleaseWorker()
        {
#if ISAAC_SPIDER_INFERENCE
            _input?.Dispose();
            _worker?.Dispose();
            _input = null;
            _worker = null;
#endif
        }

        private void OnValidate()
        {
            if (Application.isPlaying && _ready)
            {
                ApplyPhysicsSettings();
                ConfigureDrives();
            }
        }

        private void OnGUI()
        {
            if (!_showGui || !_ready)
            {
                return;
            }
            if (Time.unscaledTime >= _nextGuiRefresh)
            {
                _nextGuiRefresh = Time.unscaledTime + 0.25f;
                Vector3 com = CenterOfMassVelocity();
                _log.Clear();
                _log.Append(name).Append("  mode=").Append(_actuatorMode)
                    .Append("  dt=").Append(Time.fixedDeltaTime.ToString("0.00000"))
                    .Append("  decim=").Append(_decimation)
                    .Append("  h=").Append(BodyHeight.ToString("0.000"))
                    .Append("  |vCoM|=").Append(com.magnitude.ToString("0.000"))
                    .Append(" (").Append(com.x.ToString("0.00")).Append(',').Append(com.y.ToString("0.00")).Append(',').Append(com.z.ToString("0.00")).Append(')')
                    .Append("  reached=").Append(_reached)
                    .Append("  max|q̇|=").Append(_maxJointSpeedSeen.ToString("0.0"))
                    .Append("  override=").Append(_actionOverride)
                    .Append(_zeroGravity ? "  ZERO-G" : string.Empty)
                    .Append(HasPolicy ? string.Empty : "  NO MODEL");
                _guiText = _log.ToString();
            }
            _guiContent.text = _guiText;
            GUI.Label(new Rect(8f, 8f, Screen.width - 16f, 24f), _guiContent);
        }

        // ------------------------------------------------------------------ public API
        public void SetTarget(Transform target) => _target = target;

        /// <summary>Call after something outside rewrote the joint drives (race quirks scale stiffness/force): adopt them as the new baseline.</summary>
        public void NotifyDrivesChanged()
        {
            if (_joints[0] == null)
            {
                return;
            }
            ArticulationDrive drive = _joints[0].xDrive;
            if (_actuatorMode == ActuatorMode.ArticulationDrive && drive.stiffness > 0f)
            {
                _stiffness = drive.stiffness;
                _damping = drive.damping;
            }
            if (float.IsFinite(drive.forceLimit) && drive.forceLimit > 0f)
            {
                _effortLimit = drive.forceLimit;
            }
        }

        public void SetActionOverride(ActionOverrideMode mode, Vector4 values, float period, int singleJoint)
        {
            _actionOverride = mode;
            _overrideValues = values;
            _overridePeriod = Mathf.Max(0.01f, period);
            _overrideSingleJoint = singleJoint;
        }

        public void CopyObservation(float[] destination) => Array.Copy(_obs, destination, OBS_DIM);
        public void CopyLastAction(float[] destination) => Array.Copy(_action, destination, ACT_DIM);

        public float ReadJointPosition(int jointIndex) => _joints[jointIndex].jointPosition[0];
        public float ReadJointVelocity(int jointIndex) => _joints[jointIndex].jointVelocity[0];

        /// <summary>Mass-weighted velocity of the whole rig.</summary>
        public Vector3 CenterOfMassVelocity()
        {
            Vector3 momentum = Vector3.zero;
            float totalMass = 0f;
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                momentum += _bodies[bodyIndex].linearVelocity * _bodies[bodyIndex].mass;
                totalMass += _bodies[bodyIndex].mass;
            }
            return totalMass > 0f ? momentum / totalMass : Vector3.zero;
        }

        /// <summary>Per-body values from env.yaml + the rig-audit floors. Re-applicable at runtime.</summary>
        [ContextMenu("Apply physics settings")]
        public void ApplyPhysicsSettings()
        {
            if (_bodies == null && !TryBindRig())
            {
                return;
            }
            CaptureUrdfValuesIfMissing();
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                ArticulationBody body = _bodies[bodyIndex];
                body.automaticCenterOfMass = false;
                body.automaticInertiaTensor = false;
                body.mass = _massFloor > 0f ? Mathf.Max(_urdfMass[bodyIndex], _massFloor) : _urdfMass[bodyIndex];
                Vector3 inertia = _urdfInertia[bodyIndex];
                if (_inertiaFloor > 0f)
                {
                    inertia = new Vector3(Mathf.Max(inertia.x, _inertiaFloor), Mathf.Max(inertia.y, _inertiaFloor), Mathf.Max(inertia.z, _inertiaFloor));
                }
                body.inertiaTensor = inertia;
                body.maxLinearVelocity = _maxLinearVelocity;
                body.maxAngularVelocity = _maxAngularVelocity;
                body.maxDepenetrationVelocity = _maxDepenetrationVelocity;
                body.maxJointVelocity = _maxJointVelocity;
                body.jointFriction = _jointFriction;
                body.linearDamping = _linearDamping;
                body.angularDamping = _angularDamping;
                body.solverIterations = _solverIterations;
                body.solverVelocityIterations = _solverVelocityIterations;
                body.useGravity = !_zeroGravity;
                body.collisionDetectionMode = CollisionDetectionMode.Discrete; // env.yaml enable_ccd: false
            }
            if (_colliders != null)
            {
                for (int colliderIndex = 0; colliderIndex < _colliders.Length; colliderIndex++)
                {
                    _colliders[colliderIndex].contactOffset = _contactOffset;
                }
            }
        }

        /// <summary>Isaac: enabled_self_collisions = false. Also parks the rig on its own layer when the project defines one.</summary>
        public void ApplyCollisionFilters()
        {
            _colliders = GetComponentsInChildren<Collider>(true);
            for (int a = 0; a < _colliders.Length; a++)
            {
                for (int b = a + 1; b < _colliders.Length; b++)
                {
                    Physics.IgnoreCollision(_colliders[a], _colliders[b], true);
                }
            }
            int layer = LayerMask.NameToLayer(_layerName);
            if (layer < 0)
            {
                Debug.Log($"[{name}] layer '{_layerName}' is not defined in this project; staying on layer {LayerMask.LayerToName(gameObject.layer)}. Add it in Tags & Layers to filter the spider in the collision matrix.");
                return;
            }
            Transform[] all = GetComponentsInChildren<Transform>(true);
            for (int index = 0; index < all.Length; index++)
            {
                all[index].gameObject.layer = layer;
            }
        }

        /// <summary>Runs every recorded Isaac observation through the Worker and returns max |onnx − isaac|.</summary>
        public float RunReferenceCheck(out int steps)
        {
            steps = 0;
#if ISAAC_SPIDER_INFERENCE
            if (_isaacReference == null || _model == null)
            {
                Debug.LogWarning($"[{name}] reference check needs both the reference JSON and the model.");
                return float.NaN;
            }
            if (_worker == null)
            {
                CreateWorker();
            }
            ReferenceFile file = JsonUtility.FromJson<ReferenceFile>(_isaacReference.text);
            float worst = 0f;
            var output = new float[ACT_DIM];
            for (int stepIndex = 0; stepIndex < file.steps.Length; stepIndex++)
            {
                ReferenceStep step = file.steps[stepIndex];
                _input.Upload(step.obs);
                _worker.Schedule(_input);
                ReadOutput(output);
                for (int actionIndex = 0; actionIndex < ACT_DIM; actionIndex++)
                {
                    worst = Mathf.Max(worst, Mathf.Abs(output[actionIndex] - step.action[actionIndex]));
                }
            }
            steps = file.steps.Length;
            Debug.Log($"[{name}] reference check: {steps} steps, max |onnx - isaac| = {worst:E2} ({(worst < 1e-4f ? "PASS" : "FAIL")})");
            return worst;
#else
            return float.NaN;
#endif
        }

        [ContextMenu("Run Isaac reference check")]
        private void RunReferenceCheckMenu() => RunReferenceCheck(out _);

        // ------------------------------------------------------------------ rig binding
        private bool TryBindRig()
        {
            _bodies = GetComponentsInChildren<ArticulationBody>(true);
            if (_bodies.Length == 0)
            {
                Debug.LogError($"[{name}] no ArticulationBody rig under this object.");
                return false;
            }
            _root = null;
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                if (_bodies[bodyIndex].isRoot)
                {
                    _root = _bodies[bodyIndex];
                }
            }
            if (_root == null)
            {
                Debug.LogError($"[{name}] no root ArticulationBody found.");
                return false;
            }
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                string linkName = LinkNameForJoint(JointOrder[jointIndex]);
                _joints[jointIndex] = null;
                for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
                {
                    if (_bodies[bodyIndex].name == linkName)
                    {
                        _joints[jointIndex] = _bodies[bodyIndex];
                        break;
                    }
                }
                if (_joints[jointIndex] == null)
                {
                    Debug.LogError($"[{name}] link '{linkName}' (joint {JointOrder[jointIndex]}) not found in the rig.");
                    return false;
                }
                if (_joints[jointIndex].dofCount != 1)
                {
                    Debug.LogError($"[{name}] link '{linkName}' has {_joints[jointIndex].dofCount} DoF, expected a revolute joint (1).");
                    return false;
                }
            }
            return true;
        }

        /// <summary>Joint L1_hip drives link L1_femur; L1_knee drives L1_tibia (URDF child links).</summary>
        public static string LinkNameForJoint(string jointName)
        {
            int underscore = jointName.IndexOf('_');
            string leg = jointName.Substring(0, underscore);
            return jointName.EndsWith("hip") ? leg + "_femur" : leg + "_tibia";
        }

        private void CaptureUrdfValuesIfMissing()
        {
            if (_urdfMass != null && _urdfMass.Length == _bodies.Length)
            {
                return;
            }
            _urdfMass = new float[_bodies.Length];
            _urdfInertia = new Vector3[_bodies.Length];
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                _urdfMass[bodyIndex] = _bodies[bodyIndex].mass;
                _urdfInertia[bodyIndex] = _bodies[bodyIndex].inertiaTensor;
            }
        }

        private void ConfigureDrives()
        {
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                drive.driveType = ArticulationDriveType.Force;
                switch (_actuatorMode)
                {
                    case ActuatorMode.ArticulationDrive:
                        drive.stiffness = _stiffness;
                        drive.damping = _damping;
                        drive.forceLimit = _effortLimit;
                        break;
                    default:
                        drive.stiffness = 0f;
                        drive.damping = 0f;
                        drive.forceLimit = _effortLimit;
                        drive.target = 0f;
                        break;
                }
                drive.targetVelocity = 0f;
                _joints[jointIndex].xDrive = drive;
                if (_actuatorMode != ActuatorMode.TorqueCSharp)
                {
                    _joints[jointIndex].jointForce = new ArticulationReducedSpace(0f);
                }
            }
        }

        private void ResolveDecimation()
        {
            float ratio = _policyDt / Time.fixedDeltaTime;
            _decimation = Mathf.Max(1, Mathf.RoundToInt(ratio));
            if (Mathf.Abs(ratio - _decimation) > 1e-3f)
            {
                Debug.LogError($"[{name}] policyDt {_policyDt:0.00000} s is not an integer multiple of Time.fixedDeltaTime {Time.fixedDeltaTime:0.00000} s (ratio {ratio:0.000}). " +
                               $"Running the policy every {_decimation} steps = {_decimation * Time.fixedDeltaTime * 1000f:0.0} ms instead of {_policyDt * 1000f:0.0} ms — joint velocities and gait timing will differ from training. " +
                               "Set Fixed Timestep to 1/60 (drive mode) or 1/480 (torque mode) in Project Settings → Time.");
            }
            const float KNEE_INERTIA = 2.711e-3f; // RIG_AUDIT.md: tibia about the knee, parallel-axis
            float bound = _damping * Time.fixedDeltaTime / KNEE_INERTIA;
            if (_actuatorMode == ActuatorMode.TorqueCSharp && bound >= 1f)
            {
                Debug.LogWarning($"[{name}] explicit torque PD: kd·dt/I_knee = {bound:0.00} at dt = {Time.fixedDeltaTime:0.00000} s (must stay < 2, < 1 to be safe). " +
                                 "The rig audit requires Fixed Timestep = 1/480 s (0.0020833) for this mode; this script does not change it because it is a project-wide setting. " +
                                 "Use ActuatorMode.ArticulationDrive at coarser steps.");
            }
        }

        private void CreateWorker()
        {
#if ISAAC_SPIDER_INFERENCE
            if (_model == null)
            {
                Debug.LogWarning($"[{name}] no ModelAsset assigned — the policy is bypassed (actions = override or zero).");
                return;
            }
            Model model = ModelLoader.Load(_model);
            if (model.inputs.Count != 1 || model.inputs[0].name != "obs" || model.outputs.Count != 1 || model.outputs[0].name != "actions")
            {
                Debug.LogError($"[{name}] unexpected ONNX I/O: inputs={model.inputs.Count} '{(model.inputs.Count > 0 ? model.inputs[0].name : "-")}', outputs={model.outputs.Count} '{(model.outputs.Count > 0 ? model.outputs[0].name : "-")}'. Expected obs[1,59] → actions[1,16].");
            }
            _worker = new Worker(model, BackendType.CPU);
            _input = new Tensor<float>(new TensorShape(1, OBS_DIM));
#endif
        }

        // ------------------------------------------------------------------ observation
        /// <summary>
        /// Mirrors SpiderEnv._get_observations(). All vectors are converted Unity (Y-up, LH) → Isaac (Z-up, RH)
        /// with the URDF Importer map: Isaac (x, y, z) = Unity (z, -x, y). Angular velocity is a pseudo-vector, so the
        /// handedness flip negates it as well: ω_isaac = -(ω_u.z, -ω_u.x, ω_u.y).
        /// </summary>
        private void BuildObservation()
        {
            Transform bodyTransform = _root.transform;
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                // [0..15]  joint positions [rad], policy joint order
                _obs[jointIndex] = _joints[jointIndex].jointPosition[0];
                // [16..31] joint velocities [rad/s] × 0.1
                _obs[16 + jointIndex] = _joints[jointIndex].jointVelocity[0] * 0.1f;
            }
            // [32..34] root linear velocity in the body frame [m/s]
            Vector3 vLocal = bodyTransform.InverseTransformDirection(_root.linearVelocity);
            Vector3 vIsaac = IsaacSpiderRigBuilder.UnityToRos(vLocal);
            _obs[32] = vIsaac.x; _obs[33] = vIsaac.y; _obs[34] = vIsaac.z;
            // [35..37] root angular velocity in the body frame [rad/s] × 0.2 — pseudo-vector sign flip
            Vector3 wLocal = bodyTransform.InverseTransformDirection(_root.angularVelocity);
            Vector3 wIsaac = -IsaacSpiderRigBuilder.UnityToRos(wLocal);
            _obs[35] = wIsaac.x * 0.2f; _obs[36] = wIsaac.y * 0.2f; _obs[37] = wIsaac.z * 0.2f;
            // [38..40] gravity direction in the body frame (unit; (0, 0, -1) when upright)
            Vector3 gLocal = bodyTransform.InverseTransformDirection(Vector3.down);
            Vector3 gIsaac = IsaacSpiderRigBuilder.UnityToRos(gLocal);
            _obs[38] = gIsaac.x; _obs[39] = gIsaac.y; _obs[40] = gIsaac.z;
            // [41..42] target offset in the body's yaw-only frame: x = forward (Isaac +x = Unity +z), y = left (Isaac +y = Unity -x)
            Vector3 delta = ResolveTargetWorld() - bodyTransform.position;
            Vector3 forward = bodyTransform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 1e-8f)
            {
                forward = Vector3.forward;
            }
            forward.Normalize();
            var left = new Vector3(-forward.z, 0f, forward.x);
            _obs[41] = delta.x * forward.x + delta.z * forward.z;
            _obs[42] = delta.x * left.x + delta.z * left.z;
            // [43..58] previous (clamped) action
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                _obs[43 + jointIndex] = _prevAction[jointIndex];
            }
            if (_debugLogObservations)
            {
                _log.Clear();
                _log.Append("obs[").Append(_stepCounter).Append("] ");
                for (int index = 0; index < OBS_DIM; index++)
                {
                    _log.Append(_obs[index].ToString("0.000")).Append(' ');
                }
                Debug.Log(_log.ToString());
            }
        }

        private void ComputeAction()
        {
            switch (_actionOverride)
            {
                case ActionOverrideMode.Constant:
                    FillOverride(_overrideValues.x, _overrideValues.y);
                    break;
                case ActionOverrideMode.SquareWave:
                    bool high = Mathf.Repeat(Time.fixedTime, _overridePeriod) < _overridePeriod * 0.5f;
                    FillOverride(high ? _overrideValues.x : _overrideValues.z, high ? _overrideValues.y : _overrideValues.w);
                    break;
                default:
#if ISAAC_SPIDER_INFERENCE
                    if (_worker != null)
                    {
                        _input.Upload(_obs);
                        _worker.Schedule(_input);
                        ReadOutput(_action);
                    }
                    else
#endif
                    {
                        Array.Clear(_action, 0, ACT_DIM);
                    }
                    break;
            }
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                _action[jointIndex] = Mathf.Clamp(_action[jointIndex], -1f, 1f);
                _prevAction[jointIndex] = _action[jointIndex];
            }
        }

        private void FillOverride(float hipValue, float kneeValue)
        {
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                bool selected = _overrideSingleJoint < 0 || _overrideSingleJoint == jointIndex;
                _action[jointIndex] = selected ? ((jointIndex & 1) == 0 ? hipValue : kneeValue) : 0f;
            }
        }

#if ISAAC_SPIDER_INFERENCE
        private void ReadOutput(float[] destination)
        {
            var output = _worker.PeekOutput() as Tensor<float>;
            // CPU backend schedules jobs lazily; finish them before indexing (no allocation, unlike DownloadToArray).
            output.CompleteAllPendingOperations();
            for (int actionIndex = 0; actionIndex < ACT_DIM; actionIndex++)
            {
                destination[actionIndex] = output[0, actionIndex];
            }
        }
#endif

        // ------------------------------------------------------------------ actuator
        /// <summary>Isaac ImplicitActuator math, explicit: τ = clip(kp·(q_t − q) − kd·q̇, ±effort), every physics step.</summary>
        private void ApplyTorque()
        {
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                ArticulationBody joint = _joints[jointIndex];
                float q = joint.jointPosition[0];
                float qd = joint.jointVelocity[0];
                float torque = Mathf.Clamp(_stiffness * (_jointTarget[jointIndex] - q) - _damping * qd, -_effortLimit, _effortLimit);
                if (_enforceVelocityLimit && ((qd > _velocityLimit && torque > 0f) || (qd < -_velocityLimit && torque < 0f)))
                {
                    torque = 0f;
                }
                joint.jointForce = new ArticulationReducedSpace(torque);
            }
        }

        private void TrackJointSpeed()
        {
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                float speed = Mathf.Abs(_joints[jointIndex].jointVelocity[0]);
                if (speed > _maxJointSpeedSeen)
                {
                    _maxJointSpeedSeen = speed;
                }
            }
        }

        // ------------------------------------------------------------------ hold until grounded
        public bool IsHeld => _held;

        private void Hold()
        {
            _held = true;
            _root.immovable = true;
        }

        private void Release()
        {
            _held = false;
            _root.immovable = false;
            _stepCounter = 0;
        }

        private void ResetPoseAndHold()
        {
            _root.TeleportRoot(_spawnPosition, _spawnRotation);
            for (int bodyIndex = 0; bodyIndex < _bodies.Length; bodyIndex++)
            {
                _bodies[bodyIndex].linearVelocity = Vector3.zero;
                _bodies[bodyIndex].angularVelocity = Vector3.zero;
            }
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                _joints[jointIndex].jointPosition = new ArticulationReducedSpace(0f);
                _joints[jointIndex].jointVelocity = new ArticulationReducedSpace(0f);
                _prevAction[jointIndex] = 0f;
                _jointTarget[jointIndex] = 0f;
            }
            Hold();
        }

        /// <summary>True when something that is not part of this rig lies within the probe distance below the body.</summary>
        private bool ProbeGround()
        {
            int count = Physics.RaycastNonAlloc(_root.transform.position, Vector3.down, _probeHits, _groundProbeDistance, ~0, QueryTriggerInteraction.Ignore);
            for (int hitIndex = 0; hitIndex < count; hitIndex++)
            {
                if (!_probeHits[hitIndex].collider.transform.IsChildOf(transform))
                {
                    return true;
                }
            }
            return false;
        }

        // ------------------------------------------------------------------ target
        private Vector3 ResolveTargetWorld()
        {
            if (_targetProvider != null && _targetProvider.TryGetTargetPosition(out Vector3 provided))
            {
                return provided;
            }
            if (_target != null)
            {
                return _target.position;
            }
            return _ringTarget;
        }

        private void UpdateTargetReach()
        {
            Vector3 delta = ResolveTargetWorld() - _root.transform.position;
            delta.y = 0f;
            if (delta.sqrMagnitude >= _reachThreshold * _reachThreshold)
            {
                return;
            }
            bool external = _target != null || _targetProvider != null;
            if (!external)
            {
                _reached++;
                SampleRingTarget();
            }
        }

        private void SampleRingTarget()
        {
            float radius = Mathf.Lerp(_targetRadiusRange.x, _targetRadiusRange.y, (float)_ringRandom.NextDouble());
            float angle = (float)(_ringRandom.NextDouble() * Mathf.PI * 2.0 - Mathf.PI);
            _ringTarget = _ringOrigin + new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
            _ringTarget.y = 0.12f;
        }

        // ------------------------------------------------------------------ reference file (JsonUtility)
        [Serializable]
        private sealed class ReferenceFile
        {
            public string note;
            public ReferenceStep[] steps;
        }

        [Serializable]
        private sealed class ReferenceStep
        {
            public int step;
            public float t;
            public float[] obs;
            public float[] action;
            public float[] root_pos_w;
            public float[] root_quat_w_xyzw;
            public float[] joint_pos;
            public float[] target_rel;
            public float[] joint_vel;
            public float[] action_clamped;
        }
    }
}
