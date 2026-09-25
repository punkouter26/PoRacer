using NUnit.Framework;
using PoRacer.Models;
using PoRacer.Systems;

namespace PoRacer.Tests
{
    public sealed class Systems_QualityGovernorTests
    {
        private const float SMOOTH_FRAME = 1f / 60f;
        private const float SLOW_FRAME = 1f / 30f;

        private QualityModel _model;
        private Systems_QualityGovernor _sut;

        [SetUp]
        public void SetUp()
        {
            _model = new QualityModel();
            _sut = new Systems_QualityGovernor(_model, new RaceConfigModel());
        }

        private void Run(float frameSeconds, float totalSeconds)
        {
            int frames = (int)(totalSeconds / frameSeconds) + 1;
            for (int frameIndex = 0; frameIndex < frames; frameIndex++)
            {
                _sut.Sample(frameSeconds);
            }
        }

        [Test]
        public void SustainedSlowFrames_StepDownOneTier()
        {
            Run(SLOW_FRAME, 4.2f);

            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_MEDIUM));
        }

        [Test]
        public void SingleLoadStall_DoesNotStepDown()
        {
            Run(SMOOTH_FRAME, 1f);
            _sut.Sample(4.4f);
            Run(SMOOTH_FRAME, 6f);

            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_HIGH));
        }

        [Test]
        public void SlowBurstShorterThanTwoWindows_DoesNotStepDown()
        {
            Run(SLOW_FRAME, 2.1f);
            Run(SMOOTH_FRAME, 2.1f);
            Run(SLOW_FRAME, 2.1f);

            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_HIGH));
        }

        [Test]
        public void LongCleanRun_StepsBackUp()
        {
            Run(SLOW_FRAME, 4.2f);
            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_MEDIUM));

            Run(SMOOTH_FRAME, 24f);

            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_HIGH));
        }

        [Test]
        public void TierThatFailsRightAfterRising_IsNotRetried()
        {
            Run(SLOW_FRAME, 4.2f);
            Run(SMOOTH_FRAME, 24f);
            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_HIGH));

            // High cannot hold: it drops straight back. Longer than the first slow
            // run because the window it starts in is part smooth.
            Run(SLOW_FRAME, 6.5f);
            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_MEDIUM));

            // However clean the running is now, it stays at medium.
            Run(SMOOTH_FRAME, 60f);
            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_MEDIUM));
        }

        [Test]
        public void NeverStepsBelowMinimum()
        {
            Run(SLOW_FRAME, 60f);

            Assert.That(_model.Tier, Is.EqualTo(QualityModel.TIER_MINIMUM));
        }
    }
}
