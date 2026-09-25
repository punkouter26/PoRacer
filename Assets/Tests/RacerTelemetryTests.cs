using NUnit.Framework;
using PoRacer.Models;

namespace PoRacer.Tests
{
    public sealed class RacerTelemetryTests
    {
        [Test]
        public void HistoryAt_ReturnsOldestFirstBeforeTheBufferWraps()
        {
            var sut = new RacerTelemetry("worm#1");
            sut.PushHistory(1f, 0f, 0f);
            sut.PushHistory(2f, 0f, 0f);

            Assert.That(sut.HistoryCount, Is.EqualTo(2));
            Assert.That(sut.HistoryAt(sut.SpeedHistory, 0), Is.EqualTo(1f));
            Assert.That(sut.HistoryAt(sut.SpeedHistory, 1), Is.EqualTo(2f));
        }

        [Test]
        public void HistoryAt_ReturnsOldestFirstAfterTheBufferWraps()
        {
            var sut = new RacerTelemetry("worm#1");
            for (int sample = 0; sample < RacerTelemetry.HISTORY_LENGTH + 5; sample++)
            {
                sut.PushHistory(sample, 0f, 0f);
            }

            Assert.That(sut.HistoryCount, Is.EqualTo(RacerTelemetry.HISTORY_LENGTH));
            Assert.That(sut.HistoryAt(sut.SpeedHistory, 0), Is.EqualTo(5f));
            Assert.That(sut.HistoryAt(sut.SpeedHistory, RacerTelemetry.HISTORY_LENGTH - 1),
                Is.EqualTo(RacerTelemetry.HISTORY_LENGTH + 4f));
        }

        [Test]
        public void Twitchiness_ScalesJitterAndClampsToTheMeter()
        {
            var sut = new RacerTelemetry("worm#1");

            sut.Jitter = 0f;
            Assert.That(sut.Twitchiness, Is.EqualTo(0f));
            sut.Jitter = RacerTelemetry.JITTER_FULL_SCALE * 0.5f;
            Assert.That(sut.Twitchiness, Is.EqualTo(0.5f).Within(1e-5f));
            sut.Jitter = RacerTelemetry.JITTER_FULL_SCALE * 4f;
            Assert.That(sut.Twitchiness, Is.EqualTo(1f));
        }

        [Test]
        public void CostOfTransport_IsUnknownUntilTheRacerHasCoveredGround()
        {
            var sut = new RacerTelemetry("worm#1") { HasPower = true, MassKg = 10f, EnergyJoules = 98.1f };

            sut.DistanceMeters = 1f;
            Assert.That(sut.CostOfTransport, Is.EqualTo(RacerTelemetry.UNKNOWN));

            sut.DistanceMeters = 5f;
            Assert.That(sut.CostOfTransport, Is.EqualTo(0.2f).Within(1e-4f));
        }
    }
}
