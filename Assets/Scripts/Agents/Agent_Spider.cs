using PoRacer.Rewards;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace PoRacer.Agents
{
    /// <summary>
    /// Four-legged walker. 8 revolute joints: per leg a hip swing (around Y)
    /// and a knee lift (around Z). Torque-driven only, same MVS carve-out and
    /// NaN-safety rules as Agent_Worm (docs/Plan-P1-Worm.md D6/D14).
    ///
    /// Observations (27): per joint (8): normalized angle + velocity; root up (3);
    /// root height (1); goal direction in root space (3); normalized goal
    /// distance (1); root velocity in root space (3).
    /// Actions (8 continuous, [-1,1]): target angle per joint scaled to limits.
    /// </summary>
    public sealed class Agent_Spider : Agent, ICreatureAgent
    {
        public const int JOINT_COUNT = 8;
        public const float JOINT_LIMIT_DEGREES = 40f;
        private const float MAX_JOINT_VELOCITY = 12f;
        private const float MAX_ROOT_SPEED = 3f;
        private const float GOAL_DISTANCE_NORM = 20f;
        private const float MAX_ANGULAR_VELOCITY = 20f;
        private const float MAX_JOINT_TORQUE = 600f; // N*m, matches the prefab's xDrive forceLimit; reward normalization only

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
            for (int jointIndex = 0; jointIndex < _joints.Length; jointIndex++)
            {
                float clamped = Mathf.Clamp(actions.ContinuousActions[jointIndex], -1f, 1f);
                ArticulationDrive drive = _joints[jointIndex].xDrive;
                drive.target = clamped * JOINT_LIMIT_DEGREES;
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
            // Diagonal trot: legs 0/3 (front-left, back-right) move opposite to 1/2.
            // Joint order: FL hip, FL knee, FR hip, FR knee, BL hip, BL knee, BR hip, BR knee.
            ActionSegment<float> continuous = actionsOut.ContinuousActions;
            float phase = Time.fixedTime * 5f;
            for (int legIndex = 0; legIndex < 4; legIndex++)
            {
                bool diagonalA = legIndex == 0 || legIndex == 3;
                float legPhase = diagonalA ? phase : phase + Mathf.PI;
                continuous[legIndex * 2] = Mathf.Sin(legPhase) * 0.8f;             // hip swing
                continuous[legIndex * 2 + 1] = Mathf.Cos(legPhase) * 0.6f - 0.2f;  // knee lift
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
