using PoRacer.Rewards;
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
    /// Observations (N*2 + 11): per joint: normalized position + velocity; root up
    /// vector (3); root height (1); goal direction in root space (3); normalized goal
    /// distance (1); root velocity in root space (3).
    /// Actions (N continuous, [-1,1]): drive target per joint, scaled by
    /// _jointDriveScale (degrees for revolute joints, metres for prismatic).
    /// </summary>
    public sealed class Agent_Creature : Agent, ICreatureAgent
    {
        private const float MAX_JOINT_VELOCITY = 10f; // rad/s (or m/s), normalization only
        private const float MAX_ROOT_SPEED = 2f;      // m/s, normalization only
        private const float GOAL_DISTANCE_NORM = 20f;
        private const float MAX_ANGULAR_VELOCITY = 20f;
        // Terrain look-ahead: ground height relative to the root, sampled at fixed
        // distances along the flattened forward direction.
        private const float PROBE_RAY_HEIGHT = 3f;
        private const float PROBE_RAY_RANGE = 6f;
        private const float PROBE_HEIGHT_NORM = 2f;
        private static readonly float[] ProbeDistances = { 0.5f, 1.2f, 2.2f, 3.5f };

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
        // Adds 4 look-ahead height observations. Changes the observation size:
        // enable only together with a matching VectorObservationSize and a retrain.
        [SerializeField] private bool _useTerrainProbes;

        private readonly Reward_WormLoco _reward = new();
        private System.Action _areaReset;
        private bool _failed;

        public bool Failed => _failed;

        int ICreatureAgent.MaxStep
        {
            get => MaxStep;
            set => MaxStep = value;
        }

        public ArticulationBody Root => _root;

        public void SetGoal(Transform goal) => _goal = goal;

        public void SetAreaResetCallback(System.Action areaReset) => _areaReset = areaReset;

        public override void Initialize()
        {
            _root.maxAngularVelocity = MAX_ANGULAR_VELOCITY;
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                _joints[jointIndex].maxAngularVelocity = MAX_ANGULAR_VELOCITY;
            }
        }

        public override void OnEpisodeBegin()
        {
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

            if (_useTerrainProbes)
            {
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
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Runs every FixedUpdate (TakeActionsBetweenDecisions holds targets between decisions).
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                float clamped = Mathf.Clamp(actions.ContinuousActions[jointIndex], -1f, 1f);
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                drive.target = clamped * _jointDriveScale;
                _joints[jointIndex].xDrive = drive;
            }

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
            float uprightDot = Vector3.Dot(_root.transform.up, Vector3.up);
            AddReward(_reward.Step(distance, normalizedTorque, uprightDot));
            LogRewardComponents();

            if (_reward.ReachedGoal(distance))
            {
                AddReward(Reward_WormLoco.GOAL_REWARD);
                EndEpisode();
            }
            else if (rootPosition.y < -1f)
            {
                _failed = true;
                AddReward(Reward_WormLoco.OUT_OF_BOUNDS_REWARD);
                EndEpisode();
            }
            else if (_reward.NoProgressExceeded)
            {
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
            stats.Add("Reward/TimePenalty", -Reward_WormLoco.TIME_PENALTY);
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
