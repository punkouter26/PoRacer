using NUnit.Framework;
using PoRacer.Rewards;

namespace PoRacer.Tests
{
    public sealed class RewardClampTests
    {
        [Test]
        public void Step_PhysicsGlitchJump_IsClamped()
        {
            var sut = new Reward_WormLoco();
            sut.Reset(10f);

            float reward = sut.Step(1000000f, 0f, 0f); // exploded body teleports far away

            Assert.That(reward, Is.EqualTo(
                -Reward_WormLoco.MAX_STEP_DELTA_METERS * Reward_WormLoco.PROGRESS_SCALE - Reward_WormLoco.TIME_PENALTY));
        }
    }
}
