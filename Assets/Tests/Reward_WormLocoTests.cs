using NUnit.Framework;
using PoRacer.Rewards;

namespace PoRacer.Tests
{
    public sealed class Reward_WormLocoTests
    {
        [Test]
        public void Step_RewardsApproach_PenalizesRetreat()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            Assert.That(sut.Step(9.9f, 0f, 1f, 0f),
                Is.EqualTo(0.1f * Reward_WormLoco.PROGRESS_SCALE - Reward_WormLoco.TIME_PENALTY).Within(0.0001f));
            Assert.That(sut.Step(9.95f, 0f, 1f, 0f), Is.LessThan(0f));
        }

        [Test]
        public void EpisodeRewardSum_EqualsNetProgressMinusTimeCost()
        {
            // Step deltas stay under the physics-glitch clamp, so the sum equals net
            // approach minus the constant per-step time cost. Torque and uprightness
            // are passed at their FREE values so this stays a pure progress-sum check —
            // and note that for uprightness free is 1f (level), not 0f. It was 0f until
            // the 2026-09-10 sign change, when the term stopped paying for being upright
            // and started charging for being tilted.
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            float total = sut.Step(9.9f, 0f, 1f, 0f) + sut.Step(9.95f, 0f, 1f, 0f) + sut.Step(9.8f, 0f, 1f, 0f);

            Assert.That(total,
                Is.EqualTo(0.2f * Reward_WormLoco.PROGRESS_SCALE - 3f * Reward_WormLoco.TIME_PENALTY).Within(0.0001f));
        }

        [Test]
        public void NoProgress_TripsAfterLimit()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            for (int stepIndex = 0; stepIndex < Reward_WormLoco.NO_PROGRESS_LIMIT_STEPS; stepIndex++)
            {
                sut.Step(10f, 0f, 0f, 0f);
            }

            Assert.That(sut.NoProgressExceeded, Is.True);
        }

        [Test]
        public void ProgressResetsNoProgressCounter()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            for (int stepIndex = 0; stepIndex < Reward_WormLoco.NO_PROGRESS_LIMIT_STEPS - 1; stepIndex++)
            {
                sut.Step(10f, 0f, 0f, 0f);
            }
            sut.Step(9f, 0f, 0f, 0f);

            Assert.That(sut.NoProgressExceeded, Is.False);
        }

        [Test]
        public void ReachedGoal_WithinRadius()
        {
            var sut = new Reward_WormLoco();

            Assert.That(sut.ReachedGoal(Reward_WormLoco.GOAL_RADIUS_METERS - 0.01f), Is.True);
            Assert.That(sut.ReachedGoal(Reward_WormLoco.GOAL_RADIUS_METERS + 0.01f), Is.False);
        }

        [Test]
        public void Step_HighTorque_AppliesEfficiencyPenalty()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 1f, 0f, 0f);

            Assert.That(sut.LastEfficiencyPenalty, Is.EqualTo(-Reward_WormLoco.ENERGY_PENALTY_SCALE).Within(0.0001f));
        }

        [Test]
        public void Step_TorqueOutOfRange_IsClampedToUnitRange()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 5f, 0f, 0f);

            Assert.That(sut.LastEfficiencyPenalty, Is.EqualTo(-Reward_WormLoco.ENERGY_PENALTY_SCALE).Within(0.0001f));
        }

        /// <summary>
        /// Level costs NOTHING — it is not paid for. This is the guard on the 2026-09-10
        /// sign change: as a bonus, this term paid +0.005/step for standing perfectly
        /// still, which beat anything these rigs could earn by walking (progress had to
        /// exceed 0.426 m/s to compete; they manage 0.008-0.02). Every creature learned
        /// to stand. If this assertion is ever "fixed" back to expecting a positive
        /// bonus, that is the bug returning.
        /// </summary>
        [Test]
        public void Step_Level_CostsNothing()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 0f, 1f, 0f);

            Assert.That(sut.LastUprightBonus, Is.EqualTo(0f).Within(0.0001f));
        }

        /// <summary>
        /// On its side or beyond costs the full scale. The "do not flop" pressure the
        /// term was added for is unchanged by the sign flip — only the reward for
        /// already being upright is gone.
        /// </summary>
        [Test]
        public void Step_UpsideDown_CostsFullUprightScale()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 0f, -1f, 0f);

            Assert.That(sut.LastUprightBonus, Is.EqualTo(-Reward_WormLoco.UPRIGHT_BONUS_SCALE).Within(0.0001f));
        }

        /// <summary>
        /// The whole point: a level, motionless creature must not turn a profit. Before
        /// the sign change it netted +0.00204/step and the policy correctly exploited it.
        /// </summary>
        [Test]
        public void Step_LevelAndMotionless_IsNotProfitable()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            // Same distance as Reset: no progress. Level, no torque, no jerk, no skate.
            float reward = sut.Step(10f, 0f, 1f, 0f);

            Assert.That(reward, Is.LessThan(0f),
                "standing still must cost something, or every creature learns to stand still");
        }

        [Test]
        public void Step_JerkyActions_ApplyJerkPenalty()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 0f, 0f, 1f);

            Assert.That(sut.LastJerkPenalty, Is.EqualTo(-Reward_WormLoco.JERK_PENALTY_SCALE).Within(0.0001f));
        }

        [Test]
        public void Step_JerkOutOfRange_IsClampedToActionRange()
        {
            // Mean |action delta| can never exceed 2 for [-1, 1] actions; larger
            // values are corrupt input and must be clamped, not amplified.
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 0f, 0f, 50f);

            Assert.That(sut.LastJerkPenalty, Is.EqualTo(-2f * Reward_WormLoco.JERK_PENALTY_SCALE).Within(0.0001f));
        }
    }
}
