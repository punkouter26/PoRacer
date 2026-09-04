using NUnit.Framework;
using PoRacer.Presentation;

namespace PoRacer.Tests
{
    /// <summary>
    /// Portrait readability floor. The panel scales against a 420 dp reference
    /// width, so a UiTheme size token reads as dp on a phone: body text must not
    /// drop under Android's 14 dp minimum and anything a finger touches must
    /// reach its 48 dp target. In edit mode FontScale is 1, so these are the
    /// base values; the play-mode audit in Editor_SmokeRace checks the rendered
    /// result on top of this.
    /// </summary>
    public sealed class UiThemeTests
    {
        private const float MIN_BODY_DP = 14f;
        private const float MIN_TOUCH_DP = 48f;

        [Test]
        public void EveryFontToken_IsAtLeastAndroidBodyMinimum()
        {
            Assert.That(UiTheme.FONT_XS, Is.GreaterThanOrEqualTo(MIN_BODY_DP), nameof(UiTheme.FONT_XS));
            Assert.That(UiTheme.FONT_SM, Is.GreaterThanOrEqualTo(MIN_BODY_DP), nameof(UiTheme.FONT_SM));
            Assert.That(UiTheme.FONT_MD, Is.GreaterThanOrEqualTo(MIN_BODY_DP), nameof(UiTheme.FONT_MD));
            Assert.That(UiTheme.FONT_LG, Is.GreaterThanOrEqualTo(MIN_BODY_DP), nameof(UiTheme.FONT_LG));
            Assert.That(UiTheme.FONT_TITLE, Is.GreaterThanOrEqualTo(MIN_BODY_DP), nameof(UiTheme.FONT_TITLE));
        }

        [Test]
        public void FontTokens_AreOrdered()
        {
            Assert.That(UiTheme.FONT_XS, Is.LessThanOrEqualTo(UiTheme.FONT_SM));
            Assert.That(UiTheme.FONT_SM, Is.LessThanOrEqualTo(UiTheme.FONT_MD));
            Assert.That(UiTheme.FONT_MD, Is.LessThanOrEqualTo(UiTheme.FONT_LG));
            Assert.That(UiTheme.FONT_LG, Is.LessThanOrEqualTo(UiTheme.FONT_TITLE));
        }

        [Test]
        public void EveryControlToken_IsAtLeastAndroidTouchTarget()
        {
            Assert.That(UiTheme.CONTROL_SM, Is.GreaterThanOrEqualTo(MIN_TOUCH_DP), nameof(UiTheme.CONTROL_SM));
            Assert.That(UiTheme.CONTROL_MD, Is.GreaterThanOrEqualTo(MIN_TOUCH_DP), nameof(UiTheme.CONTROL_MD));
        }

        [Test]
        public void FurnitureNames_AreDistinct()
        {
            var names = new[]
            {
                UiTheme.FURNITURE_TITLE, UiTheme.FURNITURE_FPS, UiTheme.FURNITURE_MENU,
                UiTheme.FURNITURE_DBG, UiTheme.FURNITURE_VERSION,
            };
            Assert.That(names, Is.Unique);
        }
    }
}
