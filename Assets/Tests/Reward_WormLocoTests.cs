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

            Assert.That(sut.Step(9.9f, 0f, 0f),
                Is.EqualTo(0.1f * Reward_WormLoco.PROGRESS_SCALE - Reward_WormLoco.TIME_PENALTY).Within(0.0001f));
            Assert.That(sut.Step(9.95f, 0f, 0f), Is.LessThan(0f));
        }

        [Test]
        public void EpisodeRewardSum_EqualsNetProgressMinusTimeCost()
        {
            // Step deltas stay under the physics-glitch clamp, so the sum equals net
            // approach minus the constant per-step time cost. Torque/upright terms
            // are zeroed here so this stays a pure progress-sum check.
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            float total = sut.Step(9.9f, 0f, 0f) + sut.Step(9.95f, 0f, 0f) + sut.Step(9.8f, 0f, 0f);

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
                sut.Step(10f, 0f, 0f);
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
                sut.Step(10f, 0f, 0f);
            }
            sut.Step(9f, 0f, 0f);

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

            sut.Step(10f, 1f, 0f);

            Assert.That(sut.LastEfficiencyPenalty, Is.EqualTo(-Reward_WormLoco.ENERGY_PENALTY_SCALE).Within(0.0001f));
        }

        [Test]
        public void Step_TorqueOutOfRange_IsClampedToUnitRange()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 5f, 0f);

            Assert.That(sut.LastEfficiencyPenalty, Is.EqualTo(-Reward_WormLoco.ENERGY_PENALTY_SCALE).Within(0.0001f));
        }

        [Test]
        public void Step_Upright_AppliesBonus()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 0f, 1f);

            Assert.That(sut.LastUprightBonus, Is.EqualTo(Reward_WormLoco.UPRIGHT_BONUS_SCALE).Within(0.0001f));
        }

        [Test]
        public void Step_UpsideDown_NeverGivesNegativeUprightBonus()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            sut.Step(10f, 0f, -1f);

            Assert.That(sut.LastUprightBonus, Is.EqualTo(0f));
        }
    }
}
