using System;
using System.Collections.Generic;
using UnityEngine;
#if ISAACPORTS_HAS_INFERENCE
using Unity.InferenceEngine;
#endif

using PoRacer.IsaacPorts;

namespace IsaacBox
{
    /// <summary>
    /// Runs the Isaac Lab RSL-RL target-chasing policy for the IsaacBox on a Unity
    /// ArticulationBody rig, through Inference Engine (NOT ML-Agents - the RSL-RL ONNX
    /// has no obs_0 / continuous_actions / version_number / memory_size tensors and
    /// cannot be attached to BehaviorParameters).
    ///
    /// Contract, in one place (CONTRACT.md is the long form):
    ///   obs    float32[1, 75]  base_lin_vel(3) base_ang_vel(3) projected_gravity(3)
    ///                          target_pos_b(3) joint_pos_rel(21) joint_vel(21) actions(21)
    ///   action float32[1, 21]  joint_position_target[i] = default[i] + 0.5 * action[i]
    ///   rate   50 Hz policy; decimation = round(policyDt / Time.fixedDeltaTime)
    ///   norm   baked into the ONNX; raw observations go in
    ///
    /// This component never writes a project-wide setting. Every physics value it needs
    /// that differs from the project default is applied per body or per collider.
    /// </summary>
    [DisallowMultipleComponent]
    [AddComponentMenu("IsaacBox/IsaacBox Agent")]
    public class IsaacBoxAgent : MonoBehaviour
    {
        /// <summary>
        /// Whether Unity's ArticulationDrive applies stiffness against a radian or a degree
        /// error. The drive TARGET is in degrees while jointPosition is in radians, so the
        /// gain convention is measured by the rung-2b test rather than assumed. The H1 port
        /// measured RADIANS on this Unity version; the same drive code runs here.
        /// </summary>
        public enum GainUnits
        {
            Radians = 0,
            Degrees = 1,
        }

        /// <summary>
        /// PhysX solver iteration counts. A PER-BODY override, so raising them is allowed
        /// where changing Time.fixedDeltaTime is not.
        /// </summary>
        public enum SolverIterationMode
        {
            /// <summary>4/4, exactly the Isaac cfg. Correct at Isaac's own 0.005 s step.</summary>
            IsaacExact = 0,

            /// <summary>Scale with how much coarser the step is than Isaac's. Default.</summary>
            AutoScaleWithStep = 1,

            Manual = 2,
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
        [Tooltip("IsaacBox.onnx - obs float32[1,75] -> actions float32[1,21]. Normaliser baked in.")]
        public ModelAsset modelAsset;
#endif
        [Tooltip("Generated from isaacbox_rig.json. Holds every Isaac value this agent needs.")]
        public IsaacBoxRigAsset rig;

        [Header("Target")]
        [Tooltip("Explicit chase target. Takes priority over any ITargetProvider.")]
        public Transform target;

        [Tooltip("With no target at all, hold the default pose with the drives instead of " +
                 "running the policy on a zero target vector (which it never saw in training).")]
        public bool holdPoseWithoutTarget = true;

        [Header("Fall recovery")]
        [Tooltip("Stands the agent back up after it has been down for fallGraceSeconds. " +
                 "OFF by default, and it must stay off for anything that races: a racer " +
                 "that falls stays on the ground, and getting up is something the policy " +
                 "has to do for itself. Standing it up for free makes a policy that never " +
                 "learned to recover look identical to one that did.")]
        public bool autoRecoverFromFalls = false;

        [Tooltip("upright = dot(root.up, world up). Below this counts as fallen.")]
        [Range(0f, 1f)] public float fallUprightThreshold = 0.4f;

        [Tooltip("Seconds below the threshold before a recovery is triggered.")]
        public float fallGraceSeconds = 1.0f;

        [Header("Actuation")]
        public GainUnits gainUnits = GainUnits.Radians;

        [Tooltip("Per-body PhysX solver iterations. Isaac ran 4/4 at 0.005 s.")]
        public SolverIterationMode solverIterationMode = SolverIterationMode.AutoScaleWithStep;
        public int manualSolverIterations = 4;
        public int manualSolverVelocityIterations = 4;

        [Tooltip("Isaac set velocity_limit_sim: null, so the cap is the link angular cap.")]
        public float maxJointVelocity = 1000f;

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
        BoyJointDef[] _jointDefs;
        float[] _defaultPos;
        float[] _kp, _kd, _effort;
        float[] _obs;
        float[] _action;                  // raw policy output, fed back as the last obs block
        float[] _jointTargetRad;
        Vector3 _targetObs;               // target_pos_b, Isaac convention, clipped
        bool _hasTarget;
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
        public int PolicySteps => _policySteps;
        public int Recoveries => _recoveries;
        public ArticulationBody Root => _root;
        public IReadOnlyList<ArticulationBody> Joints => _joints;
        public Vector3 TargetObservation => _targetObs;
        public bool HasTarget => _hasTarget;
        public float[] LatestObservation => _obs;
        public float[] LatestAction => _action;
        public bool IsReady => _ready;

        public int SolverIterationsInUse
        {
            get { ResolveSolverIterations(out int p, out _); return p; }
        }

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
                Debug.LogError($"[{name}] IsaacBoxAgent has no rig asset; disabling.", this);
                enabled = false;
                return;
            }

            useGUILayout = false;
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
            _jointDefs = new BoyJointDef[n];
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
                                   "hierarchy has none. Rebuild the prefab with IsaacBox > Build Prefab.", this);
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
        /// Every value here comes from the Isaac cfg and is applied PER BODY / PER COLLIDER.
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
                b.maxJointVelocity = maxJointVelocity;
            }

            for (int i = 0; i < _allColliders.Length; i++)
                _allColliders[i].contactOffset = p.contactOffset;

            ApplyMassAndInertia();
            ApplyDriveGains();
        }

        /// <summary>Chooses the per-body solver iteration counts. Never touches Physics.defaultSolverIterations.</summary>
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
                        position = p.solverPositionIterations;
                        velocity = p.solverVelocityIterations;
                        return;
                    }
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

                // Unity does NOT serialise most per-body physics properties, so Awake is
                // the only place these are guaranteed. Every link frame is world-aligned
                // and every tensor is diagonal in it, so the diagonal permutes under M.
                b.mass = def.mass;
                b.inertiaTensor = IsaacFrameMap.InertiaDiagToUnity(def.inertiaDiagIsaac);
                b.inertiaTensorRotation = Quaternion.identity;
                b.centerOfMass = IsaacFrameMap.Pos(def.comIsaac);
            }
        }

        void ApplyDriveGains()
        {
            float gainScale = gainUnits == GainUnits.Degrees ? Mathf.Deg2Rad : 1f;

            for (int j = 0; j < _joints.Length; j++)
            {
                var b = _joints[j];
                if (b == null) continue;
                var def = _jointDefs[j];

                var d = b.xDrive;
                d.lowerLimit = def.lowerRad * Mathf.Rad2Deg;
                d.upperLimit = def.upperRad * Mathf.Rad2Deg;
                d.driveType = ArticulationDriveType.Force;
                d.stiffness = _kp[j] * gainScale;
                d.damping = _kd[j] * gainScale;
                d.forceLimit = _effort[j];
                d.target = _defaultPos[j] * Mathf.Rad2Deg;
                d.targetVelocity = 0f;
                b.xDrive = d;
            }
        }

        void ApplySelfCollisionFiltering()
        {
            // Isaac: enabled_self_collisions false. Arms hang inside the torso box's
            // margin in the default pose, so this is load-bearing, not cosmetic.
            if (rig.physics.enabledSelfCollisions) return;
            for (int i = 0; i < _allColliders.Length; i++)
                for (int k = i + 1; k < _allColliders.Length; k++)
                    Physics.IgnoreCollision(_allColliders[i], _allColliders[k], true);
        }

        void ApplyLayer()
        {
            int layer = LayerMask.NameToLayer("IsaacCreature");
            if (layer < 0) return;   // no such layer in this project; staying on Default is fine
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
                // Not an error: until export_bundle.py has run there is no IsaacBox.onnx, and the
                // rig tests (rest height, kinematics, drives) must still be able to run.
                Debug.LogWarning($"[{name}] no ModelAsset assigned; holding the default pose. " +
                                 "Run ISAAC/scripts/export_bundle.py and IsaacBox > Build Prefab.", this);
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
                    $"ticks per policy step and here it gets {_decimation}. Solver iterations were " +
                    $"raised to {SolverIterationsInUse} (SolverIterationMode.{solverIterationMode}) " +
                    $"to compensate - a PER-BODY override, no project setting was changed.", this);
            }
        }

        // ------------------------------------------------------------------- loop --
        void FixedUpdate()
        {
            if (_substep == 0)
            {
                UpdateTarget();
                if (_ready && (_hasTarget || !holdPoseWithoutTarget || actionOverride != ActionOverride.None))
                {
                    BuildObservations();
                    RunPolicy();
                    ApplyActionToTargets();
                    _policySteps++;
                }
                else if (!_ready && actionOverride != ActionOverride.None)
                {
                    // no policy, but the diagnostics still need to drive the joints
                    BuildObservations();
                    ApplyActionOverride();
                    ApplyActionToTargets();
                    _policySteps++;
                }
            }

            if (autoRecoverFromFalls) TickFallRecovery();

            _substep++;
            if (_substep >= _decimation) _substep = 0;
            _wallTime += Time.fixedDeltaTime;
        }

        /// <summary>
        /// Where to stand the creature back up: on the ground under its current planar
        /// position if there is any, else where it started (a creature that walked off the
        /// edge would otherwise respawn mid-air and fall again).
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

            _root.TeleportRoot(stand, rot);
            _root.linearVelocity = Vector3.zero;
            _root.angularVelocity = Vector3.zero;
            ResetToDefaultPose();

            Debug.LogWarning($"[{name}] fell (upright {upright:F2}); stood back up at " +
                             $"{stand:F2}. Recovery #{_recoveries}. Isaac would have ended the " +
                             $"episode here - its own eval logs " +
                             $"{rig.isaacFallsPerRobotPerMinute:F3} falls/robot/minute.", this);
        }

        /// <summary>
        /// target_pos_b exactly as TargetPositionCommand computes it: (target - root) rotated
        /// into the base frame, then scaled down to the clip radius when longer than it.
        /// </summary>
        void UpdateTarget()
        {
            if (!TryGetTargetWorld(out Vector3 tgt))
            {
                _hasTarget = false;
                _targetObs = Vector3.zero;
                return;
            }
            _hasTarget = true;

            // Isaac keeps the target at hip height on flat ground; the chase term only ever
            // measures planar distance, so use the root's own height to keep z ~ 0.
            Vector3 delta = tgt - _root.transform.position;
            delta.y = 0f;
            Vector3 inBase = Quaternion.Inverse(_root.transform.rotation) * delta;
            Vector3 isaac = IsaacFrameMap.PosToIsaac(inBase);
            float clip = rig.chase.targetObsClip;
            float mag = isaac.magnitude;
            if (mag > clip && mag > 1e-6f) isaac *= clip / mag;
            _targetObs = isaac;
        }

        bool TryGetTargetWorld(out Vector3 world)
        {
            if (target != null) { world = target.position; return true; }
            var provider = GetComponent<ITargetProvider>();
            if (provider != null && (!(provider is Behaviour bh) || bh.isActiveAndEnabled))
                return provider.TryGetTarget(out world);
            world = default;
            return false;
        }

        /// <summary>
        /// Filled index-by-index in exactly the order the Isaac observation group
        /// concatenates its terms. No scaling: the normaliser lives inside the ONNX.
        /// </summary>
        void BuildObservations()
        {
            Quaternion invRot = Quaternion.Inverse(_root.transform.rotation);
            int n = rig.actDim;

            // [0:3] base_lin_vel - root CoM velocity in the base frame (fits the H1 recording
            // 2x better than the link-origin reading; ArticulationBody.linearVelocity is CoM).
            Vector3 v = IsaacFrameMap.PosToIsaac(invRot * _root.linearVelocity);
            _obs[0] = v.x; _obs[1] = v.y; _obs[2] = v.z;

            // [3:6] base_ang_vel - pseudovector.
            Vector3 w = IsaacFrameMap.AxisToIsaac(invRot * _root.angularVelocity);
            _obs[3] = w.x; _obs[4] = w.y; _obs[5] = w.z;

            // [6:9] projected_gravity - unit gravity direction in the base frame; (0,0,-1)
            // while upright. zeroGravity is per-body and deliberately does not change this.
            Vector3 gWorld = Physics.gravity.sqrMagnitude > 1e-8f ? Physics.gravity.normalized : Vector3.down;
            Vector3 g = IsaacFrameMap.PosToIsaac(invRot * gWorld);
            _obs[6] = g.x; _obs[7] = g.y; _obs[8] = g.z;

            // [9:12] target_pos_b - already Isaac convention and clipped.
            _obs[9] = _targetObs.x; _obs[10] = _targetObs.y; _obs[11] = _targetObs.z;

            // [12:33] joint_pos relative to default, [33:54] joint_vel. Radians: jointPosition
            // is radians even though xDrive.target is degrees. No sign flips - every anchor X
            // is built at -M*axis.
            for (int j = 0; j < n; j++)
            {
                var b = _joints[j];
                _obs[12 + j] = b.jointPosition[0] - _defaultPos[j];
                _obs[12 + n + j] = b.jointVelocity[0];
            }

            // [54:75] the previous RAW action.
            Array.Copy(_action, 0, _obs, 12 + 2 * n, n);

            if (debugLogObservations && _policySteps % Mathf.Max(1, debugLogEveryNSteps) == 0)
                LogObservations();
        }

        void RunPolicy()
        {
#if ISAACPORTS_HAS_INFERENCE
            _input.Upload(_obs);
            _worker.Schedule(_input);
            var output = _worker.PeekOutput() as Tensor<float>;
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
            // joint_position_target[i] = default[i] + scale * action[i]
            for (int j = 0; j < _joints.Length; j++)
            {
                float t = rig.actionScale * _action[j];
                if (rig.useDefaultOffset) t += _defaultPos[j];
                _jointTargetRad[j] = t;

                var b = _joints[j];
                var d = b.xDrive;         // copy, so a quirk's stiffness/forceLimit edit survives
                d.target = t * Mathf.Rad2Deg;
                b.xDrive = d;
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
        /// Re-applies every setting that is otherwise read only in Awake. Instantiate()
        /// runs Awake immediately, so a field changed on the component straight afterwards
        /// (gainUnits, solverIterationMode, zeroGravity) is silently ignored without this.
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
        /// Feeds the recorded observations through the live worker and returns the max abs
        /// difference against the recorded actions. Isolates inference from physics.
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
                float[] got = output.DownloadToArray();
                for (int i = 0; i < recordedActions[s].Length; i++)
                    worst = Mathf.Max(worst, Mathf.Abs(got[i] - recordedActions[s][i]));
            }
            return worst;
#else
            return float.NaN;
#endif
        }

        /// <summary>The joint's own link inertia about its rotation axis, in Unity's frame.</summary>
        public float InertiaAboutJointAxis(int jointIndex)
        {
            var b = _joints[jointIndex];
            Vector3 axisLocal = b.anchorRotation * Vector3.right;
            Vector3 it = b.inertiaTensor;
            Quaternion r = b.inertiaTensorRotation;
            Vector3 a = Quaternion.Inverse(r) * axisLocal;
            return it.x * a.x * a.x + it.y * a.y * a.y + it.z * a.z * a.z;
        }

        void LogObservations()
        {
            int n = rig.actDim;
            var sb = new System.Text.StringBuilder(768);
            sb.Append($"[{name}] step {_policySteps} t={_wallTime:F2}\n");
            sb.Append($"  base_lin_vel      {_obs[0]:F3} {_obs[1]:F3} {_obs[2]:F3}\n");
            sb.Append($"  base_ang_vel      {_obs[3]:F3} {_obs[4]:F3} {_obs[5]:F3}\n");
            sb.Append($"  projected_gravity {_obs[6]:F3} {_obs[7]:F3} {_obs[8]:F3}\n");
            sb.Append($"  target_pos_b      {_obs[9]:F3} {_obs[10]:F3} {_obs[11]:F3}\n");
            sb.Append("  joint_pos        ");
            for (int j = 0; j < n; j++) sb.Append($" {_obs[12 + j]:F2}");
            sb.Append("\n  joint_vel        ");
            for (int j = 0; j < n; j++) sb.Append($" {_obs[12 + n + j]:F2}");
            sb.Append("\n  action           ");
            for (int j = 0; j < n; j++) sb.Append($" {_action[j]:F2}");
            Debug.Log(sb.ToString(), this);
        }

        void OnGUI()
        {
            if (!showOnGuiReadout) return;
            Vector3 v = CenterOfMassVelocity;
            Vector3 flat = new Vector3(v.x, 0f, v.z);
            Vector3 wv = _root != null ? _root.angularVelocity : Vector3.zero;
            GUI.Label(new Rect(10f, 10f, 1000f, 22f),
                $"{name}  step {_policySteps}  dec {_decimation}  fdt {Time.fixedDeltaTime:F4}  " +
                $"|vCoM| {v.magnitude:F3} m/s (planar {flat.magnitude:F3})  |w| {wv.magnitude:F2} rad/s  " +
                $"CoM y {CenterOfMassPosition.y:F3}  target_b ({_targetObs.x:F2},{_targetObs.y:F2})  " +
                $"{(_hasTarget ? "chasing" : "no target")}  recoveries {_recoveries}");
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
            if (!Application.isPlaying) ReleaseWorker();
        }
    }
}
