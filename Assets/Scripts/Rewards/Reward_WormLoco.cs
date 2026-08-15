namespace PoRacer.Rewards
{
    /// <summary>
    /// Locomotion reward shaping, kept as plain C# so it is unit-testable.
    /// Progress reward is potential-based: the sum over an episode equals
    /// meters of net approach toward the goal times PROGRESS_SCALE.
    /// Efficiency and upright terms are shaping-only (training signal); they
    /// never touch inference, so they are safe to retune without invalidating
    /// an already-trained .onnx brain until the next training run.
    /// </summary>
    public sealed class Reward_WormLoco
    {
        public const float PROGRESS_SCALE = 1f;
        public const float GOAL_REWARD = 10f;
        public const float OUT_OF_BOUNDS_REWARD = -1f;
        public const float GOAL_RADIUS_METERS = 0.5f;
        public const int NO_PROGRESS_LIMIT_STEPS = 1000; // 20 s at 0.02 s per step
        public const float MAX_STEP_DELTA_METERS = 0.2f; // physics-glitch clamp: real max is ~0.04 m per 0.02 s step
        public const float ENERGY_PENALTY_SCALE = 0.05f;
        public const float UPRIGHT_BONUS_SCALE = 0.02f;
        private const float IMPROVEMENT_EPSILON = 0.05f;

        private float _previousDistance;
        private float _bestDistance;
        private int _stepsSinceImprovement;

        public bool NoProgressExceeded => _stepsSinceImprovement >= NO_PROGRESS_LIMIT_STEPS;

        /// <summary>Reward components from the last Step() call, for StatsRecorder logging.</summary>
        public float LastProgressReward { get; private set; }

        public float LastEfficiencyPenalty { get; private set; }

        public float LastUprightBonus { get; private set; }

        public void Reset(float initialDistance)
        {
            _previousDistance = initialDistance;
            _bestDistance = initialDistance;
            _stepsSinceImprovement = 0;
            LastProgressReward = 0f;
            LastEfficiencyPenalty = 0f;
            LastUprightBonus = 0f;
        }

        /// <param name="currentDistance">Current distance to goal, meters.</param>
        /// <param name="normalizedTorque">
        /// Real applied joint torque (not the action/target), summed across joints and
        /// normalized to roughly [0,1]. Isometric bracing near a joint's torque limit
        /// must show up here even when the action itself barely changes.
        /// </param>
        /// <param name="uprightDot">Dot of root up-vector with world up; 1 = upright.</param>
        public float Step(float currentDistance, float normalizedTorque, float uprightDot)
        {
            float delta = _previousDistance - currentDistance;
            delta = delta > MAX_STEP_DELTA_METERS ? MAX_STEP_DELTA_METERS
                : delta < -MAX_STEP_DELTA_METERS ? -MAX_STEP_DELTA_METERS : delta;
            _previousDistance = currentDistance;
            if (currentDistance < _bestDistance - IMPROVEMENT_EPSILON)
            {
                _bestDistance = currentDistance;
                _stepsSinceImprovement = 0;
            }
            else
            {
                _stepsSinceImprovement++;
            }

            float clampedTorque = normalizedTorque < 0f ? 0f : normalizedTorque > 1f ? 1f : normalizedTorque;
            float uprightPositive = uprightDot > 0f ? uprightDot : 0f;

            LastProgressReward = delta * PROGRESS_SCALE;
            LastEfficiencyPenalty = -ENERGY_PENALTY_SCALE * clampedTorque;
            LastUprightBonus = UPRIGHT_BONUS_SCALE * uprightPositive;

            return LastProgressReward + LastEfficiencyPenalty + LastUprightBonus;
        }

        public bool ReachedGoal(float currentDistance) => currentDistance <= GOAL_RADIUS_METERS;
    }
}
