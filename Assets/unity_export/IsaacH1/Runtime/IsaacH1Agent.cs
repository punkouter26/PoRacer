using System;
using System.Collections.Generic;
using UnityEngine;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

using PoRacer.IsaacPorts;

namespace IsaacH1
{
    /// <summary>
    /// Runs the Isaac Lab RSL-RL locomotion policy for the Unitree H1 on a Unity
    /// ArticulationBody rig, through Inference Engine (NOT ML-Agents - the RSL-RL ONNX
    /// has no obs_0 / continuous_actions / version_number / memory_size tensors and
    /// cannot be attached to BehaviorParameters).
    ///
    /// Contract, in one place:
    ///   obs    float32[1, 69]  built index-by-index as Isaac's _get_observations()
    ///   action float32[1, 19]  joint_position_target[i] = default[i] + 0.5 * action[i]
    ///   rate   50 Hz policy; decimation = round(policyDt / Time.fixedDeltaTime)
    ///
    /// This component never writes a project-wide setting. Every physics value it needs
    /// that differs from the project default is applied per body or per collider.
    /// See CONTRACT.md and README_UNITY.md.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("IsaacH1/IsaacH1 Agent")]
    public class IsaacH1Agent : MonoBehaviour
    {
        public enum ActuatorMode
        {
            /// <summary>PhysX implicit spring-damper = Isaac's ImplicitActuator. Default.</summary>
            ArticulationDrive = 0,

            /// <summary>Diagnostic only: explicit tau = clip(kp*(q*-q) - kd*qd, +/-effort)
            /// written through ArticulationBody.jointForce. Conditionally stable - see
            /// RIG_AUDIT.md section C for the required substep.</summary>
            ExplicitTorquePD = 1,
        }

        /// <summary>
        /// Whether Unity's ArticulationDrive applies stiffness/damping against an error
        /// expressed in radians or in degrees. Unity's drive TARGET is unambiguously in
        /// degrees while jointPosition is in radians, so the gain convention has to be
        /// measured, not assumed. The rung-2 calibration test measures the real natural
        /// frequency and compares it against <see cref="PredictedNaturalFrequency"/> for
        /// both conventions; the shipped default is whatever that test reported.
        /// </summary>
        public enum GainUnits
        {
            /// <summary>stiffness is used as-is (torque per radian).</summary>
            Radians = 0,

            /// <summary>stiffness is scaled by Deg2Rad (torque per degree).</summary>
            Degrees = 1,
        }

        /// <summary>
        /// PhysX's articulation armature adds to the joint-space mass matrix DIAGONAL
        /// (H[i][i] += armature). Unity exposes no such field, so it can only be
        /// approximated through link inertia - and the obvious approximation is wrong.
        /// </summary>
        public enum ArmatureMode
        {
            /// <summary>
            /// Ignore the USD armature. MEASURED BEST and shipped by default: 1.046 m/s
            /// and upright, 117% of the Isaac reference (RIG_AUDIT.md / README_UNITY.md).
            /// It under-damps the joints relative to Isaac, which only matters for the
            /// ExplicitTorquePD diagnostic path.
            /// </summary>
            None = 0,

            /// <summary>
            /// Add armature*a*a^T to every jointed link's inertia. DO NOT USE for this
            /// rig: link inertia is SPATIAL, so a run of parallel-axis joints
            /// (hip_pitch -> knee -> ankle all rotate about the same axis) accumulates
            /// the armature of every descendant. Leg-swing inertia comes out ~3x too
            /// high, the swing leg cannot get in front of the body, and the creature
            /// pitches forward and face-plants at ~2 s. Measured: 0.094 m/s, 11% parity.
            /// </summary>
            FoldIntoInertia = 1,

            /// <summary>
            /// Fold the armature only into the DISTAL end of each parallel-axis run.
            /// For a chain of parallel joints the contribution to H[i][i] is the sum over
            /// the subtree, so folding A into the last link alone yields exactly
            /// H[i][i] += A for every joint in the run - the triangular solve of
            /// FoldIntoInertia's over-count.
            /// </summary>
            FoldDistalOnly = 2,
        }

        /// <summary>
        /// PhysX solver iteration counts. These are a PER-BODY override, so raising them
        /// is allowed where changing Time.fixedDeltaTime is not.
        /// </summary>
        public enum SolverIterationMode
        {
            /// <summary>4/4, exactly env.yaml. Correct only at Isaac's own 0.005 s step.</summary>
            IsaacExact = 0,

            /// <summary>
            /// Scale with how much coarser this project's step is than Isaac's. Default.
            /// Measured at the 0.02 s project step: 4, 16 and 32 iterations all fall over
            /// within 20 s, while 48, 64 and 96 all walk upright at 1.14-1.26 m/s. The
            /// floor of 48 comes straight from that measurement.
            /// </summary>
            AutoScaleWithStep = 1,

            /// <summary>Use manualSolverIterations / manualSolverVelocityIterations.</summary>
            Manual = 2,
        }

        public enum ActionOverride
        {
            None = 0,
            Constant = 1,
            SquareWave = 2,
        }

        public enum BaseVelocityReference
        {
            /// <summary>Matches the recording 2x better than LinkOrigin - see CONTRACT.md.</summary>
            CenterOfMass = 0,
            LinkOrigin = 1,
        }

        // ------------------------------------------------------------------ setup --
        [Header("Policy")]
#if ISAACPORTS_HAS_INFERENCE
        [Tooltip("IsaacH1.onnx - obs float32[1,69] -> actions float32[1,19].")]
        public ModelAsset modelAsset;
#endif
        [Tooltip("Generated from IsaacH1_rig.json. Holds every Isaac value this agent needs.")]
        public IsaacH1RigAsset rig;

        [Header("Target")]
        [Tooltip("Explicit chase target. Takes priority over any ITargetProvider.")]
        public Transform target;

        [Tooltip("Isaac trained lin_vel_x in [0, 1] m/s.")]
        [Range(0f, 1f)] public float commandSpeed = 1f;

        [Tooltip("Isaac's heading_control_stiffness.")]
        public float headingStiffness = 0.5f;

        [Tooltip("Isaac's ang_vel_z command range is [-1, 1] rad/s.")]
        public float maxYawRate = 1f;

        [Tooltip("Inside this radius the forward command tapers to zero. 0 = never stop.")]
        public float arriveRadius = 0f;

        [Tooltip("Reduce the forward command as the yaw command saturates. Isaac sampled " +
                 "lin_vel_x and ang_vel_z INDEPENDENTLY, so full speed together with a " +
                 "full-rate turn is a rare corner of the training distribution - and " +
                 "measurably the one that puts this policy on the floor (fell at 12.5 s " +
                 "with this at 0; no fall in 60 s at 0.5). 0 disables.")]
        [Range(0f, 1f)] public float turnSlowdown = 0.5f;

        [Header("Fall recovery")]
        [Tooltip("Isaac terminates the episode on base contact and resets. Unity has no " +
                 "such mechanism, so without this a single fall leaves the creature on " +
                 "the floor for good.")]
        public bool autoRecoverFromFalls = true;

        [Tooltip("upright = dot(root.up, world up). Below this counts as fallen.")]
        [Range(0f, 1f)] public float fallUprightThreshold = 0.4f;

        [Tooltip("Seconds below the threshold before a recovery is triggered.")]
        public float fallGraceSeconds = 1.0f;

        [Header("Actuation")]
        public ActuatorMode actuatorMode = ActuatorMode.ArticulationDrive;
        public GainUnits gainUnits = GainUnits.Radians;
        public ArmatureMode armatureMode = ArmatureMode.None;

        [Tooltip("Isaac set velocity_limit_sim: null and the recording exceeds the URDF " +
                 "limit (knee 20.33 > 14 rad/s), so this ships OFF. See RIG_AUDIT.md section B.")]
        public bool enforceVelocityLimit = false;

        [Tooltip("Applied when enforceVelocityLimit is false: the link angular cap from " +
                 "env.yaml (max_angular_velocity = 1000 rad/s).")]
        public float maxJointVelocity = 1000f;

        [Tooltip("Per-body PhysX solver iterations. Isaac ran 4/4 at 0.005 s; a coarser " +
                 "step needs more. See RIG_AUDIT.md and README_UNITY.md.")]
        public SolverIterationMode solverIterationMode = SolverIterationMode.AutoScaleWithStep;
        public int manualSolverIterations = 4;
        public int manualSolverVelocityIterations = 4;

        [Header("Mass / inertia floors")]
        [Tooltip("No link needs a floor for this rig (smallest inertia 2.14e-4, 2.1x the " +
                 "1e-4 threshold) - RIG_AUDIT.md section A. Raw values stay serialised on " +
                 "the rig asset either way, so a floor is always re-appliable.")]
        public bool applyInertiaFloor = false;
        public float inertiaFloor = 1e-4f;
        public bool applyMassFloor = false;
        public float massFloor = 0.05f;

        [Tooltip("1.0 = the nominal USD torso mass. The reference recording drew 0.8619 " +
                 "from the add_base_mass event; set that to replay it exactly.")]
        public float torsoMassScale = 1f;

        [Header("Observation")]
        public BaseVelocityReference baseVelocityReference = BaseVelocityReference.CenterOfMass;

        [Header("Diagnostics")]
        public bool debugLogObservations = false;
        public int debugLogEveryNSteps = 50;
        public ActionOverride actionOverride = ActionOverride.None;

        [Tooltip("-1 applies the override to every joint; otherwise only this joint index.")]
        public int overrideJointIndex = -1;

        public float overrideAmplitude = 1f;
        public float overrideSquareWavePeriod = 1f;

        [Tooltip("Per-body useGravity = false. Never touches project-wide Physics.gravity.")]
        public bool zeroGravity = false;

        public bool showOnGuiReadout = false;

        // ------------------------------------------------------------------ state --
        ArticulationBody _root;
        ArticulationBody[] _joints;       // indexed by Isaac joint index
        ArticulationBody[] _allBodies;
        Collider[] _allColliders;
        IsaacH1JointDef[] _jointDefs; // indexed by Isaac joint index
        float[] _defaultPos;
        float[] _kp, _kd, _effort;
        float[] _obs;
        float[] _action;                  // the raw policy output, fed back as obs[50:69]
        float[] _jointTargetRad;
        Vector3 _command;                 // (vx, vy, wz) in Isaac convention
        int _substep;
        int _decimation = 1;
        float _wallTime;
        int _policySteps;
        float _fallenFor;
        int _recoveries;
        bool _ready;
        Vector3 _homePosition;
        bool _homeCaptured;

#if ISAACPORTS_HAS_INFERENCE
        Model _model;
        Worker _worker;
        Tensor<float> _input;
#endif

        public int Decimation => _decimation;

        /// <summary>The per-body solver iteration count actually applied.</summary>
        public int SolverIterationsInUse
        {
            get { ResolveSolverIterations(out int p, out _); return p; }
        }

        public int PolicySteps => _policySteps;

        /// <summary>How many times this creature has been stood back up.</summary>
        public int Recoveries => _recoveries;
        public ArticulationBody Root => _root;
        public IReadOnlyList<ArticulationBody> Joints => _joints;
        public Vector3 Command => _command;
        public float[] LatestObservation => _obs;
        public float[] LatestAction => _action;
        public bool IsReady => _ready;

        /// <summary>Whole-creature centre-of-mass velocity, mass-weighted.</summary>
        public Vector3 CenterOfMassVelocity
        {
            get
            {
                if (_allBodies == null) return Vector3.zero;
                Vector3 p = Vector3.zero;
                float m = 0f;
                for (int i = 0; i < _allBodies.Length; i++)
                {
                    var b = _allBodies[i];
                    p += b.linearVelocity * b.mass;
                    m += b.mass;
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
                    var b = _allBodies[i];
                    p += b.worldCenterOfMass * b.mass;
                    m += b.mass;
                }
                return m > 0f ? p / m : transform.position;
            }
        }

        // ------------------------------------------------------------------- init --
        void Awake()
        {
            if (rig == null || rig.bodies == null || rig.bodies.Length == 0)
            {
                Debug.LogError($"[{name}] IsaacH1Agent has no rig asset; disabling.", this);
                enabled = false;
                return;
            }

            useGUILayout = false; // no IMGUI layout pass; OnGUI early-outs when unused
            CacheHierarchy();
            if (!enabled) return;

            ApplyPerBodyOverrides();
            ApplySelfCollisionFiltering();
            ApplyLayer();
            ResetToDefaultPose();
        }

        void CacheHierarchy()
        {
            _allBodies = GetComponentsInChildren<ArticulationBody>(true);
            _allColliders = GetComponentsInChildren<Collider>(true);

            var byName = new Dictionary<string, ArticulationBody>(_allBodies.Length);
            for (int i = 0; i < _allBodies.Length; i++) byName[_allBodies[i].name] = _allBodies[i];

            int n = rig.jointOrder.Length;
            _joints = new ArticulationBody[n];
            _jointDefs = new IsaacH1JointDef[n];
            _defaultPos = new float[n];
            _kp = new float[n];
            _kd = new float[n];
            _effort = new float[n];
            _jointTargetRad = new float[n];

            for (int i = 0; i < rig.bodies.Length; i++)
            {
                var def = rig.bodies[i];
                if (!byName.TryGetValue(def.name, out var body))
                {
                    Debug.LogError($"[{name}] rig expects a body named '{def.name}' but the " +
                                   "hierarchy has none. Rebuild the prefab with " +
                                   "IsaacH1 > Build Prefab.", this);
                    enabled = false;
                    return;
                }

                if (def.isRoot) _root = body;
                if (!def.hasJoint) continue;

                int j = def.joint.index;
                _joints[j] = body;
                _jointDefs[j] = def.joint;
                _defaultPos[j] = def.joint.defaultPosRad;
                _kp[j] = def.joint.stiffness;
                _kd[j] = def.joint.damping;
                _effort[j] = def.joint.effortLimit;
                _jointTargetRad[j] = def.joint.defaultPosRad;
            }

            if (_root == null)
            {
                Debug.LogError($"[{name}] no articulation root found in the hierarchy.", this);
                enabled = false;
                return;
            }

            _obs = new float[rig.obsDim];
            _action = new float[rig.actDim];
        }

        /// <summary>
        /// Every value here comes from env.yaml and is applied PER BODY / PER COLLIDER.
        /// Nothing in this method may become a project-wide setting.
        /// </summary>
        void ApplyPerBodyOverrides()
        {
            var p = rig.physics;
            ResolveSolverIterations(out int pos, out int vel);
            for (int i = 0; i < _allBodies.Length; i++)
            {
                var b = _allBodies[i];
                b.linearDamping = p.linearDamping;
                b.angularDamping = p.angularDamping;
                b.jointFriction = p.jointFriction;
                b.maxLinearVelocity = p.maxLinearVelocity;
                b.maxAngularVelocity = p.maxAngularVelocity;
                b.maxDepenetrationVelocity = p.maxDepenetrationVelocity;
                b.solverIterations = pos;
                b.solverVelocityIterations = vel;
                b.useGravity = !zeroGravity;
            }

            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                if (b == null) continue;
                var def = _jointDefs[j];
                b.maxJointVelocity = enforceVelocityLimit ? def.urdfVelocityLimit : maxJointVelocity;
            }

            for (int i = 0; i < _allColliders.Length; i++)
                _allColliders[i].contactOffset = p.contactOffset;

            ApplyMassAndInertia();
            ApplyDriveGains();
        }

        /// <summary>
        /// Chooses the per-body solver iteration counts. Never touches
        /// Physics.defaultSolverIterations - that is project-wide.
        /// </summary>
        public void ResolveSolverIterations(out int position, out int velocity)
        {
            var p = rig.physics;
            switch (solverIterationMode)
            {
                case SolverIterationMode.Manual:
                    position = Mathf.Max(1, manualSolverIterations);
                    velocity = Mathf.Max(0, manualSolverVelocityIterations);
                    return;

                case SolverIterationMode.AutoScaleWithStep:
                {
                    float ratio = Time.fixedDeltaTime / Mathf.Max(1e-6f, rig.isaacPhysicsDt);
                    if (ratio <= 1.01f)
                    {
                        // At (or finer than) Isaac's step, Isaac's own counts are correct.
                        position = p.solverPositionIterations;
                        velocity = p.solverVelocityIterations;
                        return;
                    }
                    // Contact/drive error per step grows roughly with the square of the
                    // step, and 48 is the measured floor at the 4x-coarser project step.
                    int scaled = Mathf.CeilToInt(p.solverPositionIterations * ratio * ratio);
                    position = Mathf.Clamp(Mathf.Max(48, scaled), 4, 96);
                    velocity = position;
                    return;
                }

                default:
                    position = p.solverPositionIterations;
                    velocity = p.solverVelocityIterations;
                    return;
            }
        }

        void ApplyMassAndInertia()
        {
            for (int i = 0; i < rig.bodies.Length; i++)
            {
                var def = rig.bodies[i];
                var b = FindBody(def.name);
                if (b == null) continue;

                float mass = def.mass;
                if (def.name == "torso_link") mass *= torsoMassScale;
                if (applyMassFloor) mass = Mathf.Max(mass, massFloor);
                b.mass = mass;

                // Always compose and write: mass/inertia floors and the armature fold
                // have to agree, and Unity does NOT serialise most per-body physics
                // properties, so Awake is the only place these can be guaranteed.
                Vector3 axisUnity = def.hasJoint
                    ? IsaacH1FrameMap.Axis(def.joint.axisInChildIsaac).normalized
                    : Vector3.right;
                IsaacH1Inertia.Compose(
                    def.inertiaDiagIsaac,
                    applyInertiaFloor, inertiaFloor,
                    def.hasJoint && armatureMode != ArmatureMode.None,
                    axisUnity,
                    ArmatureFor(def),
                    out Vector3 diag, out Quaternion diagRot);
                b.inertiaTensor = diag;
                b.inertiaTensorRotation = diagRot;
                b.centerOfMass = IsaacH1FrameMap.Pos(def.comIsaac);
            }
        }

        /// <summary>
        /// How much armature to fold into this link, given the mode. FoldDistalOnly
        /// returns zero for a link whose own joint axis is parallel to a CHILD joint's
        /// axis, because that child (or its own descendant) already carries it.
        /// </summary>
        float ArmatureFor(IsaacH1BodyDef def)
        {
            if (!def.hasJoint || armatureMode == ArmatureMode.None) return 0f;
            if (armatureMode == ArmatureMode.FoldIntoInertia) return def.joint.armature;

            Vector3 a = def.joint.axisInChildIsaac.normalized;
            for (int i = 0; i < rig.bodies.Length; i++)
            {
                var c = rig.bodies[i];
                if (!c.hasJoint || c.parent != def.name) continue;
                // the child's axis lives in the CHILD frame; localRot maps it to ours
                Vector4 q = c.localRotIsaacWxyz;               // (w, x, y, z)
                Quaternion rot = new Quaternion(q.y, q.z, q.w, q.x);   // -> (x, y, z, w)
                Vector3 ac = (rot * c.joint.axisInChildIsaac).normalized;
                if (Mathf.Abs(Vector3.Dot(a, ac)) > 0.999f) return 0f;
            }
            return def.joint.armature;
        }

        void ApplyDriveGains()
        {
            float gainScale = gainUnits == GainUnits.Degrees ? Mathf.Deg2Rad : 1f;
            bool explicitPd = actuatorMode == ActuatorMode.ExplicitTorquePD;

            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                if (b == null) continue;
                var def = _jointDefs[j];

                var d = b.xDrive;
                d.lowerLimit = def.lowerRad * Mathf.Rad2Deg;
                d.upperLimit = def.upperRad * Mathf.Rad2Deg;
                d.driveType = ArticulationDriveType.Force;
                // Explicit PD must not fight the implicit drive: zero it out entirely.
                d.stiffness = explicitPd ? 0f : _kp[j] * gainScale;
                d.damping = explicitPd ? 0f : _kd[j] * gainScale;
                d.forceLimit = explicitPd ? 0f : _effort[j];
                d.target = _defaultPos[j] * Mathf.Rad2Deg;
                d.targetVelocity = 0f;
                b.xDrive = d;
            }

            if (explicitPd)
            {
                // The single-joint bound kd*dt/I is optimistic for a serial chain; the
                // recommendation carries a 4x margin. RIG_AUDIT.md section C.
                // MEASURED, not predicted: at 1/500 the creature diverges (rung 3 max
                // |vCoM| 34.2 m/s, 58.7 m of drift in 3 s of zero-gravity bang-bang; rung
                // 5b falls at once). At 1/1000 it walks: 0.956 m/s, upright 0.991. That
                // matches RIG_AUDIT.md section C's no-armature column, which is the one
                // that applies because armatureMode ships as None.
                Debug.LogWarning(
                    $"[{name}] ActuatorMode.ExplicitTorquePD is a DIAGNOSTIC path and is only " +
                    $"conditionally stable. MEASURED on this rig: it DIVERGES at a 1/500 s " +
                    $"step and walks at 1/1000 s. The current step is " +
                    $"{Time.fixedDeltaTime:F5} s" +
                    (Time.fixedDeltaTime > 1f / 1000f + 1e-6f
                        ? " - COARSER than the 1/1000 s that was measured stable, so expect " +
                          "divergence."
                        : " - at or finer than the measured-stable 1/1000 s.") +
                    $" The shipped default is ArticulationDrive, which is an implicit " +
                    $"spring-damper and has no such bound. This agent does NOT change " +
                    $"Time.fixedDeltaTime; run it in a scene whose step you set yourself.", this);
            }
        }

        void ApplySelfCollisionFiltering()
        {
            // env.yaml: enabled_self_collisions: false. Only 3 shapes exist (torso and the
            // two feet), so this is 3 pairs - but the loop is general.
            if (rig.physics.enabledSelfCollisions) return;
            for (int i = 0; i < _allColliders.Length; i++)
                for (int k = i + 1; k < _allColliders.Length; k++)
                    Physics.IgnoreCollision(_allColliders[i], _allColliders[k], true);
        }

        void ApplyLayer()
        {
            int layer = LayerMask.NameToLayer("IsaacCreature");
            if (layer < 0)
            {
                Debug.Log($"[{name}] no 'IsaacCreature' layer is defined in this project; " +
                          "staying on 'Default'. Adding a layer is a project-settings change " +
                          "and is left for you to confirm (see README_UNITY.md).", this);
                return;
            }
            var all = GetComponentsInChildren<Transform>(true);
            for (int i = 0; i < all.Length; i++) all[i].gameObject.layer = layer;
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
                float proposed = rig.policyDt / k;
                Debug.LogError(
                    $"[{name}] policy_dt / Time.fixedDeltaTime = {rig.policyDt:F6} / {fdt:F6} = " +
                    $"{ratio:F6}, which is NOT an integer. Running with rounded decimation " +
                    $"{_decimation}, so the control rate is {1f / (_decimation * fdt):F3} Hz " +
                    $"instead of {1f / rig.policyDt:F3} Hz. The nearest fixed step that divides " +
                    $"policy_dt exactly is {proposed:F6} s (decimation {k}). This agent will not " +
                    $"change Time.fixedDeltaTime - that is a project-wide setting.", this);
            }
            else if (_decimation != rig.isaacDecimation)
            {
                Debug.LogWarning(
                    $"[{name}] policy_dt / Time.fixedDeltaTime = {ratio:F6} is an exact integer, " +
                    $"so the control rate is correct at {1f / rig.policyDt:F1} Hz with decimation " +
                    $"{_decimation}. But Isaac ran decimation {rig.isaacDecimation} at " +
                    $"{rig.isaacPhysicsDt:F6} s: the PD drive got {rig.isaacDecimation} physics " +
                    $"ticks per policy step and here it gets {_decimation}. Fidelity, not " +
                    $"correctness. Setting the project step to {rig.isaacPhysicsDt:F6} s " +
                    $"reproduces Isaac exactly and still divides policy_dt.\n" +
                    $"Solver iterations were raised to {SolverIterationsInUse} " +
                    $"(SolverIterationMode.{solverIterationMode}) to compensate; at 4/4 this " +
                    $"creature falls over at this step. This is a PER-BODY override - no " +
                    $"project setting was changed.", this);
            }
        }

        // ------------------------------------------------------------------- loop --
        void FixedUpdate()
        {
            if (!_ready) return;

            if (_substep == 0)
            {
                UpdateCommand();
                BuildObservations();
                RunPolicy();
                ApplyActionToTargets();
                _policySteps++;
            }

            if (actuatorMode == ActuatorMode.ExplicitTorquePD) ApplyExplicitTorques();

            if (autoRecoverFromFalls) TickFallRecovery();

            _substep++;
            if (_substep >= _decimation) _substep = 0;
            _wallTime += Time.fixedDeltaTime;
        }

        /// <summary>
        /// Where to stand the creature back up. Recovery used to keep the planar position
        /// and only reset the height, which is right for a creature that fell over on the
        /// track - but a creature that walked off the edge of the ground has no floor under
        /// that position, so it respawns mid-air at the same x/z and falls again. The sister
        /// port MujocoBiped did exactly that in SCN_RACE_FLAT: 24 consecutive recoveries at
        /// -6.9 m, one every 1.26 s, which is free fall plus fallGraceSeconds.
        ///
        /// So probe for ground first, and fall back to where the rig started if there is
        /// none. Only the planar position falls back - the height still comes from the
        /// rig's Isaac spawn pose.
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
                return new Vector3(current.x, hit.point.y + rig.spawnPosIsaac.z, current.z);
            }

            if (_homeCaptured)
            {
                return new Vector3(_homePosition.x, rig.spawnPosIsaac.z, _homePosition.z);
            }

            return new Vector3(current.x, rig.spawnPosIsaac.z, current.z);
        }

        /// <summary>
        /// Stands the creature back up after a fall, the way Isaac's `base_contact`
        /// termination resets an episode. Keeps the planar position and the heading, so
        /// the creature carries on from where it went down rather than teleporting home.
        /// </summary>
        void TickFallRecovery()
        {
            float upright = Vector3.Dot(_root.transform.up, Vector3.up);
            if (upright >= fallUprightThreshold)
            {
                _fallenFor = 0f;
                return;
            }

            _fallenFor += Time.fixedDeltaTime;
            if (_fallenFor < fallGraceSeconds) return;

            _fallenFor = 0f;
            _recoveries++;

            Vector3 p = _root.transform.position;
            float yaw = _root.transform.eulerAngles.y;
            Vector3 stand = GroundedRespawnPoint(p);
            var rot = Quaternion.Euler(0f, yaw, 0f);

            // TeleportRoot moves the whole articulation without the solver fighting it.
            _root.TeleportRoot(stand, rot);
            _root.linearVelocity = Vector3.zero;
            _root.angularVelocity = Vector3.zero;
            ResetToDefaultPose();

            Debug.LogWarning($"[{name}] fell (upright {upright:F2}); stood back up at " +
                             $"{stand:F2}. Recovery #{_recoveries}. Isaac would have ended " +
                             $"the episode here - its own eval logs " +
                             $"{rig.isaacFallsPerRobotPerMinute:F3} falls/robot/minute.", this);
        }

        void UpdateCommand()
        {
            if (!TryGetTargetWorld(out Vector3 tgt))
            {
                _command = Vector3.zero;
                return;
            }

            Vector3 here = _root.worldCenterOfMass;
            Vector3 to = tgt - here;
            to.y = 0f;
            float dist = to.magnitude;
            if (dist < 1e-4f) { _command = Vector3.zero; return; }

            Vector3 fwd = _root.transform.rotation * Vector3.forward; // Isaac +X == Unity +Z
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-8f) { _command = Vector3.zero; return; }

            // SignedAngle is positive when the target lies to Unity's +X (the creature's
            // right). Turning right is a NEGATIVE yaw rate in Isaac's right-handed Z-up
            // convention, hence the minus.
            float headingErr = -Vector3.SignedAngle(fwd.normalized, to / dist, Vector3.up) * Mathf.Deg2Rad;
            float wz = Mathf.Clamp(headingStiffness * headingErr, -maxYawRate, maxYawRate);

            float vx = commandSpeed;
            if (arriveRadius > 0f) vx *= Mathf.Clamp01(dist / arriveRadius);
            // Keep (vx, wz) inside the region Isaac actually sampled.
            if (turnSlowdown > 0f && maxYawRate > 1e-4f)
                vx *= Mathf.Lerp(1f, 1f - turnSlowdown, Mathf.Abs(wz) / maxYawRate);

            // Isaac trained lin_vel_y in [0, 0] - the command is strictly forward + yaw.
            _command = new Vector3(vx, 0f, wz);
        }

        bool TryGetTargetWorld(out Vector3 world)
        {
            if (target != null) { world = target.position; return true; }
            // GetComponent ignores the enabled flag, so check it: a disabled provider
            // must not keep steering the creature.
            var provider = GetComponent<ITargetProvider>();
            if (provider != null && (!(provider is Behaviour bh) || bh.isActiveAndEnabled))
                return provider.TryGetTarget(out world);
            world = default;
            return false;
        }

        /// <summary>
        /// Filled index-by-index in exactly the order Isaac's _get_observations()
        /// concatenates its terms. enable_corruption is false in the play task, so no
        /// noise is added; every term has scale: null and clip: null, so no scaling
        /// either. Layout is asserted against the rig asset at build time.
        /// </summary>
        void BuildObservations()
        {
            Quaternion invRot = Quaternion.Inverse(_root.transform.rotation);

            // [0:3] base_lin_vel - the ROOT BODY's velocity in the base frame.
            // Measured against the recording: the centre-of-mass reading fits 2x better
            // than the link-origin reading (0.0089 vs 0.0179 m/s mean residual), and
            // ArticulationBody.linearVelocity is already the CoM velocity.
            Vector3 vWorld = _root.linearVelocity;
            if (baseVelocityReference == BaseVelocityReference.LinkOrigin)
                vWorld -= Vector3.Cross(_root.angularVelocity,
                                        _root.worldCenterOfMass - _root.transform.position);
            Vector3 v = IsaacH1FrameMap.PosToIsaac(invRot * vWorld);
            _obs[0] = v.x; _obs[1] = v.y; _obs[2] = v.z;

            // [3:6] base_ang_vel - pseudovector, so AxisToIsaac not PosToIsaac.
            Vector3 w = IsaacH1FrameMap.AxisToIsaac(invRot * _root.angularVelocity);
            _obs[3] = w.x; _obs[4] = w.y; _obs[5] = w.z;

            // [6:9] projected_gravity - the world gravity DIRECTION in the base frame,
            // normalised. (0,0,-1) while upright. Read from Physics.gravity rather than
            // Vector3.down so a rotated-gravity scene stays correct; zeroGravity is a
            // per-body flag and deliberately does not change this term, because the
            // policy uses it to sense orientation, not weight.
            Vector3 gWorld = Physics.gravity.sqrMagnitude > 1e-8f
                ? Physics.gravity.normalized
                : Vector3.down;
            Vector3 g = IsaacH1FrameMap.PosToIsaac(invRot * gWorld);
            _obs[6] = g.x; _obs[7] = g.y; _obs[8] = g.z;

            // [9:12] velocity_commands - already in Isaac convention.
            _obs[9] = _command.x; _obs[10] = _command.y; _obs[11] = _command.z;

            // [12:31] joint_pos, RELATIVE to the default pose (joint_pos_rel).
            // [31:50] joint_vel (joint_vel_rel; the default joint velocity is zero).
            // Both are radians: ArticulationBody.jointPosition/jointVelocity are in
            // radians for a revolute joint even though xDrive.target is in degrees.
            // Signs need no flip because every anchor X is built at -M*axis.
            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                _obs[12 + j] = b.jointPosition[0] - _defaultPos[j];
                _obs[31 + j] = b.jointVelocity[0];
            }

            // [50:69] the previous RAW action, before scale and offset.
            Array.Copy(_action, 0, _obs, 50, rig.actDim);

            if (debugLogObservations && _policySteps % Mathf.Max(1, debugLogEveryNSteps) == 0)
                LogObservations();
        }

        void RunPolicy()
        {
#if ISAACPORTS_HAS_INFERENCE
            _input.Upload(_obs);
            _worker.Schedule(_input);
            var output = _worker.PeekOutput() as Tensor<float>;
            // Complete once, then index. DownloadToArray() allocates a managed array
            // every call and is reserved for the reference check.
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

        void ApplyActionToTargets()
        {
            // joint_position_target[i] = default[i] + scale * action[i]   (clip: null)
            for (int j = 0; j < _joints.Length; j++)
            {
                float t = rig.actionScale * _action[j];
                if (rig.useDefaultOffset) t += _defaultPos[j];
                _jointTargetRad[j] = t;

                if (actuatorMode == ActuatorMode.ArticulationDrive)
                {
                    var b = _joints[j];
                    var d = b.xDrive;
                    d.target = t * Mathf.Rad2Deg;   // Unity drive targets are DEGREES
                    b.xDrive = d;
                }
            }
        }

        void ApplyExplicitTorques()
        {
            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                float q = b.jointPosition[0];
                float qd = b.jointVelocity[0];
                float tau = Mathf.Clamp(_kp[j] * (_jointTargetRad[j] - q) - _kd[j] * qd,
                                        -_effort[j], _effort[j]);
                b.jointForce = new ArticulationReducedSpace(tau);
            }
        }

        // -------------------------------------------------------------- utilities --
        ArticulationBody FindBody(string n)
        {
            for (int i = 0; i < _allBodies.Length; i++)
                if (_allBodies[i].name == n) return _allBodies[i];
            return null;
        }

        /// <summary>
        /// Re-applies every setting that is otherwise read only in Awake: per-body
        /// overrides, mass/inertia (floors + armature fold), drive gains, self-collision
        /// filtering, and the default pose.
        ///
        /// Instantiate() runs Awake immediately, so a field changed on the component
        /// straight afterwards - armatureMode, applyInertiaFloor, torsoMassScale,
        /// actuatorMode, gainUnits - would otherwise be silently ignored. The sweep test
        /// and anyone tweaking values in the inspector at runtime must call this.
        /// </summary>
        public void Reconfigure()
        {
            if (_allBodies == null) return;
            ApplyPerBodyOverrides();
            ApplySelfCollisionFiltering();
            ResetToDefaultPose();
        }

        /// <summary>Drives every joint back to its Isaac default pose and clears velocities.</summary>
        public void ResetToDefaultPose()
        {
            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                if (b == null) continue;
                b.jointPosition = new ArticulationReducedSpace(_defaultPos[j]);
                b.jointVelocity = new ArticulationReducedSpace(0f);
                b.jointForce = new ArticulationReducedSpace(0f);
                var d = b.xDrive;
                d.target = _defaultPos[j] * Mathf.Rad2Deg;
                b.xDrive = d;
                _jointTargetRad[j] = _defaultPos[j];
            }
            if (_action != null) Array.Clear(_action, 0, _action.Length);
            _substep = 0;
            _wallTime = 0f;
        }

        /// <summary>
        /// Feeds the recorded observations through the live worker and returns the max
        /// abs difference against the recorded actions. Isolates the inference path from
        /// the physics path; should match check_onnx.py to ~1e-6.
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

        /// <summary>
        /// PREDICTED natural frequency w_n = sqrt(kp_effective / I) for one joint, under
        /// each gain convention. Nothing is measured here - the rung-2 calibration test
        /// measures the real w_n from the physics and compares against these two, which
        /// differ by sqrt(180/pi) = 7.6x and are therefore trivially separable.
        /// </summary>
        public void PredictedNaturalFrequency(int jointIndex, out float ifRadians, out float ifDegrees)
        {
            float I = Mathf.Max(1e-9f, InertiaAboutJointAxis(jointIndex));
            float kp = Mathf.Max(1e-9f, _kp[jointIndex]);
            ifRadians = Mathf.Sqrt(kp / I);
            ifDegrees = Mathf.Sqrt(kp * Mathf.Deg2Rad / I);
        }

        /// <summary>
        /// The joint's own link inertia about its rotation axis, in Unity's frame. This
        /// is the single-body value; the rung-2 test isolates one joint in zero gravity
        /// so the rest of the chain does not contribute.
        /// </summary>
        public float InertiaAboutJointAxis(int jointIndex)
        {
            var b = _joints[jointIndex];
            // Unity's revolute axis is the anchor frame's local X.
            Vector3 axisLocal = b.anchorRotation * Vector3.right;
            Vector3 it = b.inertiaTensor;
            Quaternion r = b.inertiaTensorRotation;
            Vector3 a = Quaternion.Inverse(r) * axisLocal;
            return it.x * a.x * a.x + it.y * a.y * a.y + it.z * a.z * a.z;
        }

        void LogObservations()
        {
            var sb = new System.Text.StringBuilder(512);
            sb.Append($"[{name}] step {_policySteps} t={_wallTime:F2}\n");
            sb.Append($"  base_lin_vel      {_obs[0]:F3} {_obs[1]:F3} {_obs[2]:F3}\n");
            sb.Append($"  base_ang_vel      {_obs[3]:F3} {_obs[4]:F3} {_obs[5]:F3}\n");
            sb.Append($"  projected_gravity {_obs[6]:F3} {_obs[7]:F3} {_obs[8]:F3}\n");
            sb.Append($"  velocity_commands {_obs[9]:F3} {_obs[10]:F3} {_obs[11]:F3}\n");
            sb.Append("  joint_pos        ");
            for (int j = 0; j < rig.actDim; j++) sb.Append($" {_obs[12 + j]:F2}");
            sb.Append("\n  joint_vel        ");
            for (int j = 0; j < rig.actDim; j++) sb.Append($" {_obs[31 + j]:F2}");
            sb.Append("\n  action           ");
            for (int j = 0; j < rig.actDim; j++) sb.Append($" {_action[j]:F2}");
            Debug.Log(sb.ToString(), this);
        }

        void OnGUI()
        {
            if (!showOnGuiReadout) return;
            Vector3 v = CenterOfMassVelocity;
            Vector3 flat = new Vector3(v.x, 0f, v.z);
            GUI.Label(new Rect(10f, 10f, 900f, 22f),
                $"{name}  step {_policySteps}  dec {_decimation}  fdt {Time.fixedDeltaTime:F4}  " +
                $"|vCoM| {v.magnitude:F3} m/s  (planar {flat.magnitude:F3})  " +
                $"CoM y {CenterOfMassPosition.y:F3}  cmd vx {_command.x:F2} wz {_command.z:F2}  " +
                $"{actuatorMode}");
        }

        /// <summary>Disposes the Inference Engine worker and input tensor. Idempotent.</summary>
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

        void OnDisable()
        {
            // Leave the worker alive across a disable/enable cycle, but make sure a
            // domain reload or a destroyed scene cannot leak it.
            if (!Application.isPlaying) ReleaseWorker();
        }
    }
}
