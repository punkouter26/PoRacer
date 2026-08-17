using NUnit.Framework;
using PoRacer.Rewards;

namespace PoRacer.Tests
{
    /// <summary>
    /// A single NaN reward aborts an entire mlagents training run, so the reward
    /// class must never emit one — regardless of which input or stored state was
    /// poisoned by an exploding articulation. Regression tests for two real
    /// training crashes (2026-08-15).
    /// </summary>
    public sealed class RewardNaNGuardTests
    {
        [Test]
        public void Step_NaNInputs_NeverReturnsNaN()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            float reward = sut.Step(float.NaN, float.NaN, float.NaN, float.NaN);

            Assert.That(float.IsNaN(reward), Is.False);
        }

        [Test]
        public void Step_InfinityInputs_NeverReturnsNaN()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            float reward = sut.Step(
                float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, float.PositiveInfinity);

            Assert.That(float.IsNaN(reward), Is.False);
            Assert.That(float.IsInfinity(reward), Is.False);
        }

        [Test]
        public void Reset_WithNaN_DoesNotPoisonNextStep()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(float.NaN); // agent reset while its body was still corrupt

            float reward = sut.Step(10f, 0f, 0f, 0f);

            Assert.That(float.IsNaN(reward), Is.False);
        }

        [Test]
        public void Step_AfterNaNRecovery_ResumesNormalRewards()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);
            sut.Step(float.NaN, 0f, 0f, 0f); // poisoned frame

            float reward = sut.Step(9.9f, 0f, 0f, 0f); // sane frame: 0.1 m approach

            Assert.That(float.IsNaN(reward), Is.False);
            Assert.That(reward, Is.GreaterThan(0f));
        }
    }
}
