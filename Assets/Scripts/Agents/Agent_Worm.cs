using PoRacer.Rewards;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Torque-driven worm. Movement comes only from ArticulationBody joint drives —
    /// never from forces on the root (MVS carve-out: ML-Agents Agents own their
    /// observation/action/reward logic; see docs/Plan-P1-Worm.md D6).
    ///
    /// Observations (21): per joint (5): normalized angle + normalized velocity;
    /// root up vector (3); root height (1); goal direction in root space (3);
    /// normalized goal distance (1); root velocity in root space (3).
    /// Actions (5 continuous, [-1,1]): target angle per joint, scaled to the drive limit.
    /// </summary>
    public sealed class Agent_Worm : Agent, ICreatureAgent
    {
        public const int JOINT_COUNT = 5;
        public const int OBSERVATION_COUNT = JOINT_COUNT * 2 + 11;
        public const float JOINT_LIMIT_DEGREES = 45f;
        private const float MAX_JOINT_VELOCITY = 10f; // rad/s, normalization only
        private const float MAX_ROOT_SPEED = 2f;      // m/s, normalization only
        private const float GOAL_DISTANCE_NORM = 20f;
        private const float MAX_ANGULAR_VELOCITY = 20f;
        private const float MAX_JOINT_TORQUE = 30f; // N*m, matches the prefab's xDrive forceLimit; reward normalization only

        [SerializeField] private ArticulationBody _root;
        [SerializeField] private ArticulationBody[] _joints = new ArticulationBody[JOINT_COUNT];
        [SerializeField] private Transform _goal;

        private readonly Reward_WormLoco _reward = new();
        private System.Action _areaReset;
        private bool _failed;

        public bool Failed => _failed;

        int ICreatureAgent.MaxStep
        {
            get => MaxStep;
            set => MaxStep = value;
        }

        public Transform Goal => _goal;

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
                float angle = joint.jointPosition.dofCount > 0 ? joint.jointPosition[0] : 0f;
                float velocity = joint.jointVelocity.dofCount > 0 ? joint.jointVelocity[0] : 0f;
                sensor.AddObservation(Safe(Mathf.Clamp(angle / (JOINT_LIMIT_DEGREES * Mathf.Deg2Rad), -1f, 1f)));
                sensor.AddObservation(Safe(Mathf.Clamp(velocity / MAX_JOINT_VELOCITY, -1f, 1f)));
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
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Runs every FixedUpdate (TakeActionsBetweenDecisions holds targets between decisions).
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                float clamped = Mathf.Clamp(actions.ContinuousActions[jointIndex], -1f, 1f);
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                drive.target = clamped * JOINT_LIMIT_DEGREES;
                _joints[jointIndex].xDrive = drive;
            }

            Vector3 rootPosition = _root.transform.position;
            if (float.IsNaN(rootPosition.x) || float.IsNaN(rootPosition.y) || float.IsNaN(rootPosition.z))
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
            return torqueSum / (JOINT_COUNT * MAX_JOINT_TORQUE);
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
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Scripted traveling sine wave: doubles as the pre-training gait validation.
            ActionSegment<float> continuous = actionsOut.ContinuousActions;
            float time = Time.fixedTime;
            for (int jointIndex = 0; jointIndex < JOINT_COUNT; jointIndex++)
            {
                continuous[jointIndex] = Mathf.Sin(time * 5f - jointIndex * 1.1f);
            }
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
