using System.Text;
using UnityEngine;
#if ISAAC_BIPED2_INFERENCE
using Unity.InferenceEngine;
#endif

using PoRacer.IsaacPorts;

namespace IsaacBiped2
{
    /// <summary>
    /// Runs the Isaac Lab biped walk-to-target policy (RSL-RL PPO, exported to ONNX) on Unity's
    /// Inference Engine. Not an ML-Agents agent: it owns its own observation vector, worker and
    /// PD loop, exactly as the spider and H1 exports do.
    ///
    /// Contract (unity_export/IsaacBiped2/export_report.json):
    ///   obs    float32[1, 42]  see <see cref="BuildObservation"/> for the layout
    ///   action float32[1, 10]  joint target = defaultPose + 0.5 * clamp(action, -1, 1) [rad]
    ///   rate   50 Hz policy; Isaac ran 200 Hz physics with decimation 4
    ///
    /// <b>Step-rate caveat.</b> The project runs at Time.fixedDeltaTime = 0.02 s, so the control
    /// rate is exactly right (decimation 1) but physics is 4x coarser than Isaac's 0.005 s. The
    /// standing pose was measured in Isaac to be stable only at ~170 Hz and above — below that the
    /// ankle drive cannot hold the forward lean and the rig topples. Solver iterations are scaled
    /// up to compensate (see <see cref="ResolveSolverIterations"/>), which is the same mitigation
    /// the H1 export uses. This component will not change Time.fixedDeltaTime: that is a
    /// project-wide setting shared with every ML-Agents racer.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class IsaacBiped2Agent : MonoBehaviour
    {
        public const int OBS_DIM = 42;
        public const int ACT_DIM = 10;

        /// <summary>Policy joint order (export_report.json → joint_order).</summary>
        public static readonly string[] JointOrder =
        {
            "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
            "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"
        };

        /// <summary>Nominal standing crouch [rad]; actions are offsets around THIS, not around zero.</summary>
        public static readonly float[] DefaultPose =
        {
            0f, 0f, -0.25f, 0.50f, -0.25f,
            0f, 0f, -0.25f, 0.50f, -0.25f
        };

        /// <summary>Per-joint drive gains, in policy joint order (Isaac ImplicitActuatorCfg).</summary>
        private static readonly float[] Stiffness = { 100f, 100f, 150f, 150f, 80f, 100f, 100f, 150f, 150f, 80f };
        private static readonly float[] Damping = { 4f, 4f, 5f, 5f, 3f, 4f, 4f, 5f, 5f, 3f };
        private static readonly float[] EffortLimit = { 100f, 100f, 150f, 150f, 80f, 100f, 100f, 150f, 150f, 80f };

        // ------------------------------------------------------------------ policy
        [Header("Policy")]
#if ISAAC_BIPED2_INFERENCE
        [SerializeField] private ModelAsset _model;
#endif
        [Tooltip("Isaac policy step. 0.02 s = 50 Hz.")]
        [SerializeField] private float _policyDt = 0.02f;
        [Tooltip("Isaac physics step the policy was trained against, for the fidelity warning.")]
        [SerializeField] private float _isaacPhysicsDt = 0.005f;
        [SerializeField] private float _actionScale = 0.5f;

        // ------------------------------------------------------------------ physics
        [Header("Physics")]
        [SerializeField] private float _maxLinearVelocity = 100f;
        [SerializeField] private float _maxAngularVelocity = 100f;
        [SerializeField] private float _maxDepenetrationVelocity = 1f;
        [SerializeField] private float _maxJointVelocity = 100f;
        [SerializeField] private float _angularDamping = 0.05f;
        [SerializeField] private int _solverPositionIterations = 8;
        [SerializeField] private int _solverVelocityIterations = 0;
        [Tooltip("Raise solver iterations when the project step is coarser than Isaac's.")]
        [SerializeField] private bool _autoScaleSolverIterations = true;

        [Tooltip("Scales every drive gain. Unity's ArticulationDrive answers a commanded target " +
                 "more stiffly than Isaac's ImplicitActuator: under the identical standing load the " +
                 "ankle droops 0.030 rad here against 0.088 rad in Isaac. 1 = as trained.")]
        [SerializeField] private float _gainScale = 1f;

        [Tooltip("Distance at which contacts begin generating force. Unity defaults to 0.01 m, " +
                 "PhysX under Isaac to 0.02 — an unmatched offset changes exactly when a foot starts " +
                 "pushing back, which is the moment a biped's balance is decided.")]
        [SerializeField] private float _contactOffset = 0.02f;

        [Tooltip("Minimum principal inertia [kg*m^2] for the collider-less hub links only. They are " +
                 "modelling artifacts that exist to carry the hip yaw/roll DOFs, and at ~1e-4 they " +
                 "are 15x worse conditioned than any real segment. 0 disables.")]
        [SerializeField] private float _hubInertiaFloor = 0.01f;

        // ------------------------------------------------------------------ target
        [Header("Target")]
        [SerializeField] private Transform _target;
        [SerializeField] private float _reachThreshold = 0.4f;

        // ------------------------------------------------------------------ lifecycle
        [Header("Lifecycle")]
        [Tooltip("Pin the root until ground exists underneath. SCN_RACE_FLAT builds its track " +
                 "only when the race starts, so a racer spawned earlier would fall out of the world.")]
        [SerializeField] private bool _holdUntilGrounded = true;
        [SerializeField] private float _groundProbeDistance = 50f;
        [Tooltip("Hold this long after ground appears so the drives can reach the standing " +
                 "crouch before the policy takes over.")]
        [SerializeField] private float _settleSeconds = 0.5f;
        [SerializeField] private bool _autoRecoverFromFalls = true;
        [SerializeField] private bool _showOnGuiReadout;

        // ------------------------------------------------------------------ runtime state
        private ArticulationBody _root;
        private readonly ArticulationBody[] _joints = new ArticulationBody[ACT_DIM];
        private readonly float[] _obs = new float[OBS_DIM];
        private readonly float[] _action = new float[ACT_DIM];
        private readonly float[] _prevAction = new float[ACT_DIM];
        private int _decimation = 1;
        private int _stepCounter;
        private bool _ready;
        private bool _held;
        private Vector3 _spawnPosition;
        private Quaternion _spawnRotation;
        private readonly RaycastHit[] _probeHits = new RaycastHit[32];
        private float _flippedSeconds;
        private float _settledFor;
        private ITargetProvider _targetProvider;
        private int _reached;
        private string _guiText = string.Empty;
#if ISAAC_BIPED2_INFERENCE
        private Worker _worker;
        private Tensor<float> _input;
#endif

        /// <summary>Assigned by the race adapter; overrides any <see cref="ITargetProvider"/>.</summary>
        public Transform target
        {
            get => _target;
            set => _target = value;
        }

        public bool autoRecoverFromFalls
        {
            get => _autoRecoverFromFalls;
            set => _autoRecoverFromFalls = value;
        }

        public bool showOnGuiReadout
        {
            get => _showOnGuiReadout;
            set => _showOnGuiReadout = value;
        }

        public ArticulationBody Root => _root;

        public int Decimation => _decimation;

        public int TargetsReached => _reached;

        private void Awake()
        {
            _root = GetComponentInChildren<ArticulationBody>();
            if (_root == null)
            {
                Debug.LogError($"[{name}] no ArticulationBody under this object; the rig was not built.", this);
                return;
            }
            _targetProvider = GetComponent<ITargetProvider>();
            ResolveJoints();
            ApplyPhysicsSettings();
            ConfigureDrives();
            ResolveDecimation();
            BuildWorker();
            _spawnPosition = _root.transform.position;
            _spawnRotation = _root.transform.rotation;
            if (!CheckStepIsSolvable())
            {
                enabled = false;
                return;
            }
            _ready = true;
        }

        private void Start()
        {
            if (_holdUntilGrounded && _ready)
            {
                _held = true;
                _root.immovable = true;
            }
        }

        private void OnDestroy() => ReleaseWorker();

        // ------------------------------------------------------------------ setup
        private void ResolveJoints()
        {
            ArticulationBody[] bodies = _root.GetComponentsInChildren<ArticulationBody>(true);
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                for (int bodyIndex = 0; bodyIndex < bodies.Length; bodyIndex++)
                {
                    // Rig link names carry the joint name of the joint that drives them: the
                    // builder names each link after its URDF link, whose joint is <link>'s parent
                    // joint. Match on the link that the policy joint drives.
                    if (bodies[bodyIndex].name == LinkDrivenBy(JointOrder[jointIndex]))
                    {
                        _joints[jointIndex] = bodies[bodyIndex];
                        break;
                    }
                }
                if (_joints[jointIndex] == null)
                {
                    Debug.LogError($"[{name}] joint '{JointOrder[jointIndex]}' not found in the rig.", this);
                }
            }
        }

        /// <summary>The URDF child link each policy joint drives (biped.urdf).</summary>
        private static string LinkDrivenBy(string jointName)
        {
            string side = jointName.Substring(0, 2);           // "L_" or "R_"
            string joint = jointName.Substring(2);
            switch (joint)
            {
                case "hip_yaw": return side + "hip_yaw_link";
                case "hip_roll": return side + "hip_roll_link";
                case "hip_pitch": return side + "thigh";
                case "knee": return side + "shank";
                default: return side + "foot";
            }
        }

        private void ApplyPhysicsSettings()
        {
            ResolveSolverIterations(out int position, out int velocity);
            ArticulationBody[] bodies = _root.GetComponentsInChildren<ArticulationBody>(true);
            for (int index = 0; index < bodies.Length; index++)
            {
                ArticulationBody body = bodies[index];
                body.maxLinearVelocity = _maxLinearVelocity;
                body.maxAngularVelocity = _maxAngularVelocity;
                body.maxDepenetrationVelocity = _maxDepenetrationVelocity;
                body.maxJointVelocity = _maxJointVelocity;
                body.angularDamping = _angularDamping;
                body.linearDamping = 0f;
                body.jointFriction = 0f;
                body.solverIterations = position;
                body.solverVelocityIterations = velocity;
                // Only the hubs: a body with no collider contributes no contact and no visible
                // volume, so inflating its inertia is a solver aid, not a change to the physics the
                // policy was trained against. Applying the same floor to the thigh or foot would
                // multiply a real segment's inertia by 20x and break the gait outright.
                if (_contactOffset > 0f)
                {
                    Collider[] bodyColliders = body.GetComponentsInChildren<Collider>();
                    for (int c = 0; c < bodyColliders.Length; c++)
                    {
                        if (bodyColliders[c].GetComponentInParent<ArticulationBody>() == body)
                        {
                            bodyColliders[c].contactOffset = _contactOffset;
                        }
                    }
                }
                if (_hubInertiaFloor > 0f && body.GetComponentInChildren<Collider>() == null)
                {
                    Vector3 inertia = body.inertiaTensor;
                    body.inertiaTensor = new Vector3(
                        Mathf.Max(inertia.x, _hubInertiaFloor),
                        Mathf.Max(inertia.y, _hubInertiaFloor),
                        Mathf.Max(inertia.z, _hubInertiaFloor));
                }
            }
        }

        /// <summary>
        /// A position drive is only solvable when the joint inertia is large enough for the step:
        /// roughly I &gt;= k*dt^2. Below that the articulation diverges — measured here as joint
        /// positions reaching ~1e12 rad and the root going NaN within seconds, which corrupts the
        /// whole racer. Report it plainly instead of letting the scene fill with NaN.
        /// </summary>
        private bool CheckStepIsSolvable()
        {
            float dt = Time.fixedDeltaTime;
            if (dt <= _isaacPhysicsDt * 1.5f)
            {
                return true;
            }
            // Measured, not guessed: at 0.005 s the rig stands at 0.658 m (Isaac: 0.657) and the
            // policy drives it. At the project's 0.02 s it diverges — joint positions reach ~1e12
            // rad and the root goes NaN within seconds, which also poisons anything reading its
            // transform. The limbs are thin hand-built primitives whose inertias are 30-400x too
            // small to solve a 100-150 N*m/rad drive at a 20 ms step; the H1 gets away with 0.02
            // only because its links carry real USD inertias.
            Debug.LogError(
                $"[{name}] disabled: this rig needs Time.fixedDeltaTime <= {_isaacPhysicsDt:F4} s and the " +
                $"project is running {dt:F4} s. At this step the articulation diverges to NaN rather " +
                $"than merely walking badly, so the agent will not run. Set Fixed Timestep to " +
                $"{_isaacPhysicsDt:F4} in Project Settings > Time (this is project-wide and affects " +
                $"every other racer, which is why this component will not change it itself).", this);
            return false;
        }

        /// <summary>
        /// Isaac's own iteration counts are correct at Isaac's step. At a coarser project step the
        /// per-step drive and contact error grows roughly with the square of the step, so the counts
        /// are scaled by that ratio — the same compensation the H1 export measured.
        /// </summary>
        private void ResolveSolverIterations(out int position, out int velocity)
        {
            if (!_autoScaleSolverIterations)
            {
                position = _solverPositionIterations;
                velocity = _solverVelocityIterations;
                return;
            }
            float ratio = Time.fixedDeltaTime / Mathf.Max(1e-6f, _isaacPhysicsDt);
            if (ratio <= 1.01f)
            {
                position = _solverPositionIterations;
                velocity = _solverVelocityIterations;
                return;
            }
            int scaled = Mathf.CeilToInt(_solverPositionIterations * ratio * ratio);
            position = Mathf.Clamp(Mathf.Max(48, scaled), 4, 96);
            velocity = position;
        }

        private void ConfigureDrives()
        {
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                ArticulationBody joint = _joints[jointIndex];
                if (joint == null)
                {
                    continue;
                }
                ArticulationDrive drive = joint.xDrive;
                drive.driveType = ArticulationDriveType.Force;
                // Gains pass through unscaled: the H1 calibration measured Unity's articulation
                // drive to apply stiffness per RADIAN, matching Isaac, even though the target is
                // expressed in degrees.
                drive.stiffness = Stiffness[jointIndex] * _gainScale;
                drive.damping = Damping[jointIndex] * _gainScale;
                drive.forceLimit = EffortLimit[jointIndex];
                drive.target = DefaultPose[jointIndex] * Mathf.Rad2Deg;
                drive.targetVelocity = 0f;
                joint.xDrive = drive;
            }
        }

        private void ResolveDecimation()
        {
            float ratio = _policyDt / Time.fixedDeltaTime;
            _decimation = Mathf.Max(1, Mathf.RoundToInt(ratio));
            if (Mathf.Abs(ratio - _decimation) > 1e-3f)
            {
                Debug.LogError(
                    $"[{name}] policyDt {_policyDt:F5} s is not an integer multiple of " +
                    $"Time.fixedDeltaTime {Time.fixedDeltaTime:F5} s (ratio {ratio:F3}). Running at " +
                    $"decimation {_decimation}, so the control rate is " +
                    $"{1f / (_decimation * Time.fixedDeltaTime):F1} Hz instead of {1f / _policyDt:F1} Hz.",
                    this);
            }
            if (Time.fixedDeltaTime > _isaacPhysicsDt * 1.01f)
            {
                Debug.LogWarning(
                    $"[{name}] physics runs at {1f / Time.fixedDeltaTime:F0} Hz but the policy was " +
                    $"trained at {1f / _isaacPhysicsDt:F0} Hz. The standing pose was measured stable " +
                    $"only at ~170 Hz and above, so expect degraded balance. Solver iterations have " +
                    $"been raised to compensate; lowering Time.fixedDeltaTime would fix it properly " +
                    $"but is a project-wide change this component will not make.", this);
            }
        }

        private void BuildWorker()
        {
#if ISAAC_BIPED2_INFERENCE
            if (_model == null)
            {
                Debug.LogError($"[{name}] no ModelAsset assigned; the biped will stand still.", this);
                return;
            }
            Model runtimeModel = ModelLoader.Load(_model);
            _worker = new Worker(runtimeModel, BackendType.CPU);
            _input = new Tensor<float>(new TensorShape(1, OBS_DIM));
#endif
        }

        public void ReleaseWorker()
        {
#if ISAAC_BIPED2_INFERENCE
            _input?.Dispose();
            _worker?.Dispose();
            _input = null;
            _worker = null;
#endif
        }

        // ------------------------------------------------------------------ loop
        private void FixedUpdate()
        {
            if (!_ready)
            {
                return;
            }
            if (_held)
            {
                // Ground has to exist before the racer is dropped, and the drives need a moment to
                // pull the legs from the authored straight pose into the standing crouch. Releasing
                // the instant the track appears hands the policy a stance it never trained on.
                if (ProbeGround())
                {
                    _settledFor += Time.fixedDeltaTime;
                    if (_settledFor >= _settleSeconds)
                    {
                        Release();
                    }
                }
                else
                {
                    _settledFor = 0f;
                }
                return;
            }
            if (_autoRecoverFromFalls)
            {
                if (_root.transform.position.y < _spawnPosition.y - 3f)
                {
                    ResetPoseAndHold();
                    return;
                }
                // Isaac ends the episode when the torso tips past ~66 degrees; the policy never
                // learned to get up, so put it back on its feet rather than let it flail.
                _flippedSeconds = _root.transform.up.y < 0.4f ? _flippedSeconds + Time.fixedDeltaTime : 0f;
                if (_flippedSeconds > 1.5f)
                {
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
                    ArticulationBody joint = _joints[jointIndex];
                    if (joint == null)
                    {
                        continue;
                    }
                    float clamped = Mathf.Clamp(_action[jointIndex], -1f, 1f);
                    float targetRad = DefaultPose[jointIndex] + _actionScale * clamped;
                    ArticulationDrive drive = joint.xDrive;
                    drive.target = targetRad * Mathf.Rad2Deg;   // Unity drive targets are DEGREES
                    joint.xDrive = drive;
                    _prevAction[jointIndex] = clamped;
                }
            }
            _stepCounter++;
            UpdateTargetReach();
        }

        /// <summary>
        /// obs[0..9]   joint positions minus the standing pose [rad]
        /// obs[10..19] joint velocities [rad/s] x 0.1
        /// obs[20..22] torso linear velocity in the body frame [m/s], Isaac axes
        /// obs[23..25] torso angular velocity in the body frame [rad/s] x 0.25, Isaac axes
        /// obs[26..28] gravity direction in the body frame (0, 0, -1 when upright)
        /// obs[29..30] unit heading to the target in the torso's yaw-only frame
        /// obs[31]     distance to the target / 5, clamped to 1
        /// obs[32..41] previous clamped action
        /// </summary>
        private void BuildObservation()
        {
            Transform bodyTransform = _root.transform;
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                ArticulationBody joint = _joints[jointIndex];
                if (joint == null)
                {
                    continue;
                }
                _obs[jointIndex] = joint.jointPosition[0] - DefaultPose[jointIndex];
                _obs[10 + jointIndex] = joint.jointVelocity[0] * 0.1f;
            }
            Vector3 vLocal = bodyTransform.InverseTransformDirection(_root.linearVelocity);
            Vector3 vIsaac = IsaacBiped2RigBuilder.UnityToRos(vLocal);
            _obs[20] = vIsaac.x; _obs[21] = vIsaac.y; _obs[22] = vIsaac.z;
            // Angular velocity is a pseudo-vector: the handedness flip negates it.
            Vector3 wLocal = bodyTransform.InverseTransformDirection(_root.angularVelocity);
            Vector3 wIsaac = -IsaacBiped2RigBuilder.UnityToRos(wLocal);
            _obs[23] = wIsaac.x * 0.25f; _obs[24] = wIsaac.y * 0.25f; _obs[25] = wIsaac.z * 0.25f;
            Vector3 gLocal = bodyTransform.InverseTransformDirection(Vector3.down);
            Vector3 gIsaac = IsaacBiped2RigBuilder.UnityToRos(gLocal);
            _obs[26] = gIsaac.x; _obs[27] = gIsaac.y; _obs[28] = gIsaac.z;
            // Yaw-only frame: Isaac +x is Unity +z (forward), Isaac +y is Unity -x (left).
            Vector3 delta = ResolveTargetWorld() - bodyTransform.position;
            Vector3 forward = bodyTransform.forward;
            forward.y = 0f;
            if (forward.sqrMagnitude < 1e-8f)
            {
                forward = Vector3.forward;
            }
            forward.Normalize();
            var left = new Vector3(-forward.z, 0f, forward.x);
            float ahead = delta.x * forward.x + delta.z * forward.z;
            float sideways = delta.x * left.x + delta.z * left.z;
            float distance = Mathf.Sqrt(ahead * ahead + sideways * sideways);
            _obs[29] = ahead / (distance + 1e-6f);
            _obs[30] = sideways / (distance + 1e-6f);
            _obs[31] = Mathf.Min(distance / 5f, 1f);
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                _obs[32 + jointIndex] = _prevAction[jointIndex];
            }
            // Belt and braces: the articulation can report a non-finite velocity on the frame it
            // is un-pinned, and feeding that to the network produces NaN actions.
            for (int index = 0; index < OBS_DIM; index++)
            {
                if (float.IsNaN(_obs[index]) || float.IsInfinity(_obs[index]))
                {
                    _obs[index] = 0f;
                }
            }
        }

        private void ComputeAction()
        {
#if ISAAC_BIPED2_INFERENCE
            if (_worker == null)
            {
                System.Array.Clear(_action, 0, ACT_DIM);
                return;
            }
            _input.Upload(_obs);
            _worker.Schedule(_input);
            var output = _worker.PeekOutput() as Tensor<float>;
            // The CPU backend schedules lazily; finish the jobs before indexing. Indexing avoids
            // the per-step allocation that DownloadToArray would cost.
            output.CompleteAllPendingOperations();
            for (int index = 0; index < ACT_DIM; index++)
            {
                float value = output[0, index];
                // A single non-finite action would be written into _prevAction, fed back in as
                // obs[32..41] on the next step and produce NaN again forever — the policy would
                // latch off for the rest of the run. Drop it to zero instead so one bad frame
                // (typically the release frame, when the solver has just been un-pinned) cannot
                // permanently disable the racer.
                _action[index] = float.IsNaN(value) || float.IsInfinity(value) ? 0f : value;
            }
#else
            System.Array.Clear(_action, 0, ACT_DIM);
#endif
        }

        // ------------------------------------------------------------------ target
        private Vector3 ResolveTargetWorld()
        {
            if (_target != null)
            {
                return _target.position;
            }
            if (_targetProvider != null && _targetProvider.TryGetTarget(out Vector3 provided))
            {
                return provided;
            }
            return _root.transform.position + _root.transform.forward * 5f;
        }

        private void UpdateTargetReach()
        {
            Vector3 delta = ResolveTargetWorld() - _root.transform.position;
            delta.y = 0f;
            if (delta.magnitude < _reachThreshold)
            {
                _reached++;
            }
        }

        // ------------------------------------------------------------------ hold and recover
        private bool ProbeGround()
        {
            int hits = Physics.RaycastNonAlloc(
                _root.transform.position + Vector3.up * 0.1f,
                Vector3.down,
                _probeHits,
                _groundProbeDistance,
                ~0,
                QueryTriggerInteraction.Ignore);
            for (int index = 0; index < hits; index++)
            {
                if (_probeHits[index].collider.GetComponentInParent<ArticulationBody>() == null)
                {
                    return true;
                }
            }
            return false;
        }

        private void Release()
        {
            _held = false;
            _root.immovable = false;
            _root.TeleportRoot(_root.transform.position, _root.transform.rotation);
            ApplyNominalPose();
            _root.WakeUp();
        }

        /// <summary>
        /// Drive the joints to the standing crouch the policy was trained to start from. The rig is
        /// authored at all-zero joints (legs straight), 0.25 rad away at both hips, knees and
        /// ankles — the policy would be reasoning about a pose it never saw, from a stance whose
        /// soles are not flat.
        ///
        /// This only sets the drive targets and lets PhysX walk the joints there over the hold
        /// frames. Writing <c>jointPosition</c> per body instead corrupts the articulation's
        /// internal cache: doing that here blew the joint positions up to ~1e12 rad within a few
        /// frames. The supported alternative, the root's reduced-space arrays, needs a DOF offset
        /// this Unity version does not expose — and the drives get there on their own anyway,
        /// because the racer is held for several frames before it is released.
        /// </summary>
        private void ApplyNominalPose() => ConfigureDrives();

        private void ResetPoseAndHold()
        {
            _root.immovable = true;
            _root.TeleportRoot(_spawnPosition, _spawnRotation);
            for (int jointIndex = 0; jointIndex < ACT_DIM; jointIndex++)
            {
                _prevAction[jointIndex] = 0f;
                _action[jointIndex] = 0f;
            }
            ApplyNominalPose();
            _settledFor = 0f;
            _held = true;
        }

        /// <summary>Re-apply drives after something external rewrote them (spawn, fatigue).</summary>
        public void ReapplyDrives() => ConfigureDrives();

        private void OnGUI()
        {
            if (!_showOnGuiReadout || !_ready)
            {
                return;
            }
            var builder = new StringBuilder();
            builder.Append("IsaacBiped2  y=").Append(_root.transform.position.y.ToString("0.00"))
                .Append("  up=").Append(_root.transform.up.y.ToString("0.00"))
                .Append("  reached=").Append(_reached)
                .Append(_held ? "  HELD" : string.Empty);
            _guiText = builder.ToString();
            GUI.Label(new Rect(10f, 10f, 600f, 24f), _guiText);
        }
    }
}
