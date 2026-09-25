using NUnit.Framework;
using PoRacer.Systems;

namespace PoRacer.Tests
{
    public sealed class Systems_SkyTests
    {
        [Test]
        public void PickIndex_NeverRepeatsTheCurrentPresetWhenThereIsAnother()
        {
            var rng = new System.Random(7);
            for (int drawIndex = 0; drawIndex < 500; drawIndex++)
            {
                int current = drawIndex % 3;
                Assert.That(Systems_Sky.PickIndex(3, current, rng), Is.Not.EqualTo(current));
            }
        }

        [Test]
        public void PickIndex_ReachesEveryOtherPreset()
        {
            var rng = new System.Random(11);
            var seen = new bool[4];
            for (int drawIndex = 0; drawIndex < 400; drawIndex++)
            {
                seen[Systems_Sky.PickIndex(4, 0, rng)] = true;
            }

            Assert.That(seen[0], Is.False);
            Assert.That(seen[1] && seen[2] && seen[3], Is.True);
        }

        [Test]
        public void PickIndex_SinglePresetPoolAlwaysReturnsIt()
        {
            var rng = new System.Random(3);

            Assert.That(Systems_Sky.PickIndex(1, 0, rng), Is.EqualTo(0));
            Assert.That(Systems_Sky.PickIndex(1, -1, rng), Is.EqualTo(0));
        }
    }
}
