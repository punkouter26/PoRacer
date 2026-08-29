using PoRacer.Rewards;
using PoRacer.Sensors;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Generic N-joint locomotion agent shared by every creature, worm and spider
    /// included, with joint count, limits, and gait tables configured per prefab
    /// (MVS carve-out: ML-Agents Agents own their observation/action/reward logic).
    ///
    /// Observations (N*3 + 19): per joint: normalized position + velocity + ground
    /// contact; root up vector (3); root height (1); goal direction in root space
    /// (3); normalized goal distance (1); root velocity in root space (3); root
    /// angular velocity in root space (3); stamina (1); terrain look-ahead height
    /// probes (4).
    /// Actions (N continuous, [-1,1]): drive target per joint, scaled by
    /// _jointDriveScale (degrees for revolute joints, metres for prismatic).
    ///
    /// Fatigue: applied torque above the sustainable fraction drains a stamina
    /// pool; torque below it recovers. Low stamina scales joint stiffness and
    /// force limit toward MIN_POWER_FACTOR. Load is read from applied torque, not
    /// the action vector (project rule): isometric bracing tires a creature even
    /// when its actions barely move. Identical in training and races, so the
    /// policy learns to pace itself.
    /// </summary>
    public sealed class Agent_Creature : Agent, ICreatureAgent
    {
        private const float MAX_JOINT_VELOCITY = 10f; // rad/s (or m/s), normalization only
        private const float MAX_ROOT_SPEED = 2f;      // m/s, normalization only
        private const float GOAL_DISTANCE_NORM = 20f;
        // Far past any reachable distance: only a diverged body gets here.
        private const float DIVERGED_GOAL_DISTANCE = 500f;
        private const float MAX_ANGULAR_VELOCITY = 20f;
        // Terrain look-ahead: ground height relative to the root, sampled at fixed
        // distances along the flattened forward direction.
        private const float PROBE_RAY_HEIGHT = 3f;
        private const float PROBE_RAY_RANGE = 6f;
        private const float PROBE_HEIGHT_NORM = 2f;
        private static readonly float[] ProbeDistances = { 0.5f, 1.2f, 2.2f, 3.5f };
        // Fatigue tuning. At full overload (normalized torque 1.0) a fresh pool
        // empties in ~4 s; a resting creature refills it in ~6.7 s. An empty pool
        // still leaves MIN_POWER_FACTOR of authored joint power, so tired racers
        // slow down instead of collapsing.
        private const float SUSTAINABLE_TORQUE_FRACTION = 0.5f;
        private const float STAMINA_DRAIN_PER_SECOND = 0.25f;
        private const float STAMINA_RECOVERY_PER_SECOND = 0.15f;
        private const float MIN_POWER_FACTOR = 0.55f;
        // Drives are only rewritten when the power factor moved this much, so a
        // steady cruise costs no drive writes at all.
        private const float POWER_FACTOR_WRITE_EPSILON = 0.005f;

        [SerializeField] private ArticulationBody _root;
        [SerializeField] private ArticulationBody[] _joints;
        [SerializeField] private Transform _goal;
        [SerializeField] private float _jointDriveScale = 45f;
        [SerializeField] private float _maxJointTorque = 100f; // N*m, matches the prefab's drive forceLimit
        // Coded gait: action[i] = amplitude[i] * sin(2*pi*frequency*t + phase[i]).
        // Authored per creature by Editor_BuildCreatures (tripod, trot, serpentine, ...).
        // Used as the scripted alternative brain and as the demo source for BC/GAIL.
        [SerializeField] private float _gaitFrequency = 0.8f;
        [SerializeField] private float[] _gaitPhases;
        [SerializeField] private float[] _gaitAmplitudes;
        // Per-joint DC offset so crouched stances (bent knees) are expressible.
        [SerializeField] private float[] _gaitOffsets;

        private readonly Reward_WormLoco _reward = new();
        private System.Action _areaReset;
        private bool _failed;
        private Sensor_LimbContact[] _limbContacts;
        private float[] _previousActions;
        private float _stamina = 1f;
        private float _lastPowerFactor = 1f;
        // Baseline drives for fatigue scaling, captured lazily on the first write
        // so spawn-time quirk scaling (applied a frame after instantiation) is
        // already included. NotifyDrivesChanged() invalidates the capture.
        private float[] _baseStiffness;
        private float[] _baseForceLimit;
        private bool _driveBaselineCaptured;
        private Quaternion _restRotation = Quaternion.identity;

        public bool Failed => _failed;

        public Quaternion RestRotation => _restRotation;

        int ICreatureAgent.MaxStep
        {
            get => MaxStep;
            set => MaxStep = value;
        }

        public ArticulationBody Root => _root;

        /// <summary>Articulation root's transform; the prefab root only when there is no articulation.</summary>
        public Transform Body => _root != null ? _root.transform : transform;

        public void SetGoal(Transform goal) => _goal = goal;

        public void SetAreaResetCallback(System.Action areaReset) => _areaReset = areaReset;

        public void NotifyDrivesChanged()
        {
            _driveBaselineCaptured = false;
            _lastPowerFactor = 1f;
        }

        public override void Initialize()
        {
            // Captured before the first episode reset can move it: this is the
            // authored rest pose the whole rig was laid out around.
            _restRotation = _root.transform.localRotation;
            _root.maxAngularVelocity = MAX_ANGULAR_VELOCITY;
            _previousActions = new float[_joints.Length];
            _limbContacts = new Sensor_LimbContact[_joints.Length];
            _baseStiffness = new float[_joints.Length];
            _baseForceLimit = new float[_joints.Length];
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                _joints[jointIndex].maxAngularVelocity = MAX_ANGULAR_VELOCITY;
                GameObject jointGo = _joints[jointIndex].gameObject;
                Sensor_LimbContact contact = jointGo.GetComponent<Sensor_LimbContact>();
                _limbContacts[jointIndex] = contact != null ? contact : jointGo.AddComponent<Sensor_LimbContact>();
            }
        }

        public override void OnEpisodeBegin()
        {
            // Fatigue cleared before motors are restored (project rule): undo any
            // fatigue drive scaling first, then let the area reset re-author
            // drives and quirks on top of clean values.
            RestoreDriveBaseline();
            _stamina = 1f;
            _lastPowerFactor = 1f;
            _driveBaselineCaptured = false;
            if (_previousActions != null)
            {
                System.Array.Clear(_previousActions, 0, _previousActions.Length);
            }
            _areaReset?.Invoke();
            _failed = false;
            _reward.Reset(DistanceToGoal());
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Every value passes through Safe(): one exploded body must never feed
            // NaN into the sensor, which would abort the Academy step for ALL agents.
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                ArticulationBody joint = _joints[jointIndex];
                float jointPosition = joint.jointPosition.dofCount > 0 ? joint.jointPosition[0] : 0f;
                float jointVelocity = joint.jointVelocity.dofCount > 0 ? joint.jointVelocity[0] : 0f;
                sensor.AddObservation(Safe(Mathf.Clamp(jointPosition / (_jointDriveScale * Mathf.Deg2Rad), -1f, 1f)));
                sensor.AddObservation(Safe(Mathf.Clamp(jointVelocity / MAX_JOINT_VELOCITY, -1f, 1f)));
                sensor.AddObservation(_limbContacts[jointIndex] != null && _limbContacts[jointIndex].IsGrounded ? 1f : 0f);
            }

            Transform rootTransform = _root.transform;
            sensor.AddObservation(SafeVector(rootTransform.up));
            sensor.AddObservation(Safe(rootTransform.position.y));

            Vector3 toGoal = _goal != null ? _goal.position - rootTransform.position : Vector3.zero;
            Vector3 localDirection = rootTransform.InverseTransformDirection(toGoal.normalized);
            sensor.AddObservation(SafeVector(localDirection));
            sensor.AddObservation(Safe(Mathf.Clamp01(toGoal.magnitude / GOAL_DISTANCE_NORM)));

            Vector3 localVelocity = rootTransform.InverseTransformDirection(_root.linearVelocity);
            sensor.AddObservation(SafeVector(Vector3.ClampMagnitude(localVelocity / MAX_ROOT_SPEED, 1f)));

            Vector3 localAngularVelocity = rootTransform.InverseTransformDirection(_root.angularVelocity);
            sensor.AddObservation(SafeVector(Vector3.ClampMagnitude(localAngularVelocity / MAX_ANGULAR_VELOCITY, 1f)));

            sensor.AddObservation(Safe(_stamina));

            Vector3 forward = rootTransform.forward;
            forward.y = 0f;
            forward = forward.sqrMagnitude > 0.001f ? forward.normalized : Vector3.forward;
            float rootY = rootTransform.position.y;
            for (int probeIndex = 0; probeIndex < ProbeDistances.Length; probeIndex++)
            {
                Vector3 origin = rootTransform.position
                    + forward * ProbeDistances[probeIndex] + Vector3.up * PROBE_RAY_HEIGHT;
                float relativeHeight = 0f;
                if (Physics.Raycast(origin, Vector3.down, out RaycastHit hit, PROBE_RAY_RANGE))
                {
                    relativeHeight = hit.point.y - rootY;
                }
                sensor.AddObservation(Safe(Mathf.Clamp(relativeHeight / PROBE_HEIGHT_NORM, -1f, 1f)));
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Runs every FixedUpdate (TakeActionsBetweenDecisions holds targets between decisions).
            float jerkSum = 0f;
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                float clamped = Mathf.Clamp(actions.ContinuousActions[jointIndex], -1f, 1f);
                jerkSum += Mathf.Abs(clamped - _previousActions[jointIndex]);
                _previousActions[jointIndex] = clamped;
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                drive.target = clamped * _jointDriveScale;
                _joints[jointIndex].xDrive = drive;
            }
            float actionJerk = jerkSum / _joints.Length;

            Vector3 rootPosition = _root.transform.position;
            Vector3 rootUp = _root.transform.up;
            if (float.IsNaN(rootPosition.x) || float.IsNaN(rootPosition.y) || float.IsNaN(rootPosition.z)
                || float.IsNaN(rootUp.x))
            {
                _failed = true;
                SetReward(Reward_WormLoco.OUT_OF_BOUNDS_REWARD);
                EndEpisode();
                return;
            }

            float distance = DistanceToGoal();
            float normalizedTorque = ComputeNormalizedTorque();
            float skateVelocity = ComputeSkateVelocity();
            UpdateStamina(normalizedTorque);
            ApplyFatigueToDrives();
            float uprightDot = Vector3.Dot(_root.transform.up, Vector3.up);
            AddReward(_reward.Step(distance, normalizedTorque, uprightDot, actionJerk, skateVelocity));
            LogRewardComponents();

            if (_reward.ReachedGoal(distance))
            {
                AddReward(Reward_WormLoco.GOAL_REWARD);
                EndEpisode();
            }
            else if (rootPosition.y < -1f || distance > DIVERGED_GOAL_DISTANCE)
            {
                // A solver divergence reaches absurd-but-finite coordinates long
                // before it reaches NaN, and every step in between feeds garbage
                // into the trainer. Measured against the goal so it holds wherever
                // a training area sits in the world.
                _failed = true;
                AddReward(Reward_WormLoco.OUT_OF_BOUNDS_REWARD);
                EndEpisode();
            }
            else if (_reward.NoProgressExceeded)
            {
                AddReward(Reward_WormLoco.STALL_REWARD);
                EpisodeInterrupted();
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Per-creature coded gait from the phase/amplitude tables; falls back to
            // a traveling wave when no gait was authored.
            ActionSegment<float> continuous = actionsOut.ContinuousActions;
            float angle = 2f * Mathf.PI * _gaitFrequency * Time.fixedTime;
            bool hasGait = _gaitPhases != null && _gaitPhases.Length == _joints.Length
                && _gaitAmplitudes != null && _gaitAmplitudes.Length == _joints.Length;
            bool hasOffsets = _gaitOffsets != null && _gaitOffsets.Length == _joints.Length;
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                float offset = hasOffsets ? _gaitOffsets[jointIndex] : 0f;
                continuous[jointIndex] = hasGait
                    ? _gaitAmplitudes[jointIndex] * Mathf.Sin(angle + _gaitPhases[jointIndex]) + offset
                    : Mathf.Sin(angle - jointIndex * 1.1f);
            }
        }

        private float ComputeSkateVelocity()
        {
            if (_limbContacts == null || _joints == null)
            {
                return 0f;
            }
            float skateSum = 0f;
            int groundedCount = 0;
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                if (_limbContacts[jointIndex] != null && _limbContacts[jointIndex].IsGrounded)
                {
                    Vector3 linearVelocity = _joints[jointIndex].linearVelocity;
                    linearVelocity.y = 0f;
                    skateSum += linearVelocity.magnitude;
                    groundedCount++;
                }
            }
            return groundedCount > 0 ? skateSum / groundedCount : 0f;
        }

        private float ComputeNormalizedTorque()
        {
            // Real applied torque (jointForce), not the action target — isometric
            // bracing can hold near-maximum torque while the action barely moves.
            float torqueSum = 0f;
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                ArticulationReducedSpace jointForce = _joints[jointIndex].jointForce;
                torqueSum += jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
            }
            return torqueSum / (_joints.Length * _maxJointTorque);
        }

        private void UpdateStamina(float normalizedTorque)
        {
            float overload = (Mathf.Clamp01(normalizedTorque) - SUSTAINABLE_TORQUE_FRACTION)
                / (1f - SUSTAINABLE_TORQUE_FRACTION);
            _stamina += overload > 0f
                ? -overload * STAMINA_DRAIN_PER_SECOND * Time.fixedDeltaTime
                : STAMINA_RECOVERY_PER_SECOND * Time.fixedDeltaTime;
            _stamina = Mathf.Clamp01(_stamina);
        }

        private void ApplyFatigueToDrives()
        {
            float factor = Mathf.Lerp(MIN_POWER_FACTOR, 1f, _stamina);
            if (Mathf.Abs(factor - _lastPowerFactor) < POWER_FACTOR_WRITE_EPSILON)
            {
                return;
            }
            if (!_driveBaselineCaptured)
            {
                CaptureDriveBaseline();
            }
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                float scaledStiffness = _baseStiffness[jointIndex] * factor;
                float scaledForceLimit = _baseForceLimit[jointIndex] * factor;
                if (float.IsFinite(scaledStiffness))
                {
                    drive.stiffness = scaledStiffness;
                }
                if (float.IsFinite(scaledForceLimit))
                {
                    drive.forceLimit = scaledForceLimit;
                }
                _joints[jointIndex].xDrive = drive;
            }
            _lastPowerFactor = factor;
        }

        private void CaptureDriveBaseline()
        {
            // Unwind whatever factor is currently applied so the baseline always
            // stores the drives at full power, quirks included.
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                _baseStiffness[jointIndex] = drive.stiffness / _lastPowerFactor;
                _baseForceLimit[jointIndex] = drive.forceLimit / _lastPowerFactor;
            }
            _driveBaselineCaptured = true;
        }

        private void RestoreDriveBaseline()
        {
            if (!_driveBaselineCaptured)
            {
                return;
            }
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                if (float.IsFinite(_baseStiffness[jointIndex]))
                {
                    drive.stiffness = _baseStiffness[jointIndex];
                }
                if (float.IsFinite(_baseForceLimit[jointIndex]))
                {
                    drive.forceLimit = _baseForceLimit[jointIndex];
                }
                _joints[jointIndex].xDrive = drive;
            }
        }

        private void LogRewardComponents()
        {
            // Only meaningful with a trainer attached; hundreds of inference-only
            // racers must never pay for stats nobody is recording.
            if (!Academy.IsInitialized || !Academy.Instance.IsCommunicatorOn)
            {
                return;
            }
            Unity.MLAgents.StatsRecorder stats = Academy.Instance.StatsRecorder;
            stats.Add("Reward/Progress", _reward.LastProgressReward);
            stats.Add("Reward/EfficiencyPenalty", _reward.LastEfficiencyPenalty);
            stats.Add("Reward/UprightBonus", _reward.LastUprightBonus);
            stats.Add("Reward/JerkPenalty", _reward.LastJerkPenalty);
            stats.Add("Reward/SkatePenalty", _reward.LastSkatePenalty);
            stats.Add("Reward/TimePenalty", -Reward_WormLoco.TIME_PENALTY);
            stats.Add("Fatigue/Stamina", _stamina);
        }

        private static float Safe(float value)
        {
            return float.IsNaN(value) || float.IsInfinity(value) ? 0f : value;
        }

        private static Vector3 SafeVector(Vector3 value)
        {
            return new Vector3(Safe(value.x), Safe(value.y), Safe(value.z));
        }

        private float DistanceToGoal()
        {
            if (_goal == null)
            {
                return GOAL_DISTANCE_NORM;
            }
            Vector3 toGoal = _goal.position - _root.transform.position;
            toGoal.y = 0f;
            return toGoal.magnitude;
        }
    }
}
