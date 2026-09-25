using PoRacer.Models;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoRacer.Presentation
{
    /// <summary>
    /// One bar per brain output, growing up for a positive command and down for a negative
    /// one around a centre line: what the policy is telling each joint to do right now.
    /// A calm gait shows as bars that sway together; a twitchy one as bars flicking end to end.
    /// </summary>
    public sealed class PolicyBarsGraph : VisualElement
    {
        public static readonly Color PositiveColor = UiTheme.AccentSoft;
        public static readonly Color NegativeColor = UiTheme.NeonCyan;

        private const float BAR_GAP_FRACTION = 0.25f;
        private const float MIN_BAR_HEIGHT = 1f;

        private RacerTelemetry _source;

        public PolicyBarsGraph()
        {
            pickingMode = PickingMode.Ignore;
            generateVisualContent += Draw;
        }

        public void Bind(RacerTelemetry source)
        {
            _source = source;
            MarkDirtyRepaint();
        }

        private void Draw(MeshGenerationContext context)
        {
            Rect rect = contentRect;
            if (rect.width <= 1f || rect.height <= 1f)
            {
                return;
            }
            Painter2D painter = context.painter2D;
            float midY = rect.center.y;
            painter.lineWidth = 1f;
            painter.strokeColor = UiTheme.Divider;
            painter.BeginPath();
            painter.MoveTo(new Vector2(rect.xMin, midY));
            painter.LineTo(new Vector2(rect.xMax, midY));
            painter.Stroke();

            int count = _source != null ? _source.ActionCount : 0;
            if (count == 0)
            {
                return;
            }
            float slot = rect.width / count;
            float barWidth = Mathf.Max(1f, slot * (1f - BAR_GAP_FRACTION));
            float halfHeight = rect.height * 0.5f;
            for (int actionIndex = 0; actionIndex < count; actionIndex++)
            {
                float value = _source.Actions[actionIndex];
                float height = Mathf.Max(MIN_BAR_HEIGHT, Mathf.Abs(value) * halfHeight);
                float left = rect.xMin + actionIndex * slot + (slot - barWidth) * 0.5f;
                float top = value >= 0f ? midY - height : midY;
                painter.fillColor = value >= 0f ? PositiveColor : NegativeColor;
                painter.BeginPath();
                painter.MoveTo(new Vector2(left, top));
                painter.LineTo(new Vector2(left + barWidth, top));
                painter.LineTo(new Vector2(left + barWidth, top + height));
                painter.LineTo(new Vector2(left, top + height));
                painter.ClosePath();
                painter.Fill();
            }
        }
    }
}
