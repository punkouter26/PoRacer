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
        // Giving up must never be reward-neutral: without this, freezing beats
        // trying-and-falling (-1) in expectation, which trains passive brains.
        // Milder than OUT_OF_BOUNDS_REWARD so trying stays the better gamble.
        public const float STALL_REWARD = -0.5f;
        public const float MAX_STEP_DELTA_METERS = 0.2f; // physics-glitch clamp: real max is ~0.04 m per 0.02 s step
        public const float ENERGY_PENALTY_SCALE = 0.05f;
        public const float UPRIGHT_BONUS_SCALE = 0.02f;
        // Races are won on time: a constant per-step cost makes a fast finish
        // strictly better than a slow one even when both cover the same meters.
        // Full episode (3000 steps) costs 6, well under the goal bonus of 10.
        public const float TIME_PENALTY = 0.002f;
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
            // A NaN initial distance (agent reset while its articulation was still
            // corrupt) would poison every later delta even with sane inputs.
            if (float.IsNaN(initialDistance) || float.IsInfinity(initialDistance))
            {
                initialDistance = 20f;
            }
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
            // NaN/Inf firewall: an exploding articulation can produce NaN distance,
            // torque, or up-vector for a frame before the agent's failure check
            // catches it. A single NaN reward aborts the whole training run
            // (mlagents raises on NaN), so sanitize every input here — the one
            // choke point all creature agents share.
            if (float.IsNaN(currentDistance) || float.IsInfinity(currentDistance))
            {
                currentDistance = _previousDistance;
            }
            if (float.IsNaN(normalizedTorque) || float.IsInfinity(normalizedTorque))
            {
                normalizedTorque = 0f;
            }
            if (float.IsNaN(uprightDot) || float.IsInfinity(uprightDot))
            {
                uprightDot = 0f;
            }

            float delta = _previousDistance - currentDistance;
            // Final belt: the ternary clamp below passes NaN through (NaN fails every
            // comparison), so a poisoned stored state must be zeroed here.
            if (float.IsNaN(delta))
            {
                delta = 0f;
                _previousDistance = currentDistance;
            }
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

            return LastProgressReward + LastEfficiencyPenalty + LastUprightBonus - TIME_PENALTY;
        }

        public bool ReachedGoal(float currentDistance) => currentDistance <= GOAL_RADIUS_METERS;
    }
}
