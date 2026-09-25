using PoRacer.Models;
using UnityEngine;
using UnityEngine.UIElements;

namespace PoRacer.Presentation
{
    /// <summary>
    /// Three-line sparkline of a racer's last 15 s: ground speed (scaled to its own peak
    /// in the window), uprightness and joint effort (both 0..1). Drawn with Painter2D, so
    /// it costs one mesh rebuild per new sample and nothing in between.
    /// </summary>
    public sealed class TelemetryGraph : VisualElement
    {
        public static readonly Color SpeedColor = UiTheme.AccentSoft;
        public static readonly Color UprightColor = UiTheme.NeonCyan;
        public static readonly Color EffortColor = new(0.85f, 0.6f, 1f);

        private const float LINE_WIDTH = 2f;
        // Speed is scaled to at least this peak, so a racer crawling at 0.05 m/s does not
        // fill the graph and look fast.
        private const float MIN_SPEED_SCALE = 0.5f;

        private RacerTelemetry _source;
        private int _drawnVersion = -1;

        public TelemetryGraph()
        {
            pickingMode = PickingMode.Ignore;
            generateVisualContent += Draw;
        }

        public void Bind(RacerTelemetry source)
        {
            _source = source;
            _drawnVersion = -1;
            MarkDirtyRepaint();
        }

        /// <summary>Repaints only when a new history sample has landed.</summary>
        public void RefreshIfChanged()
        {
            if (_source != null && _source.HistoryVersion != _drawnVersion)
            {
                MarkDirtyRepaint();
            }
        }

        private void Draw(MeshGenerationContext context)
        {
            Rect rect = contentRect;
            if (_source == null || rect.width <= 1f || rect.height <= 1f)
            {
                return;
            }
            _drawnVersion = _source.HistoryVersion;
            Painter2D painter = context.painter2D;

            // Baseline and mid-line, so "upright 100%" and "half effort" read at a glance.
            painter.lineWidth = 1f;
            painter.strokeColor = UiTheme.Divider;
            painter.BeginPath();
            painter.MoveTo(new Vector2(rect.xMin, rect.yMax - 0.5f));
            painter.LineTo(new Vector2(rect.xMax, rect.yMax - 0.5f));
            painter.MoveTo(new Vector2(rect.xMin, rect.center.y));
            painter.LineTo(new Vector2(rect.xMax, rect.center.y));
            painter.Stroke();

            int count = _source.HistoryCount;
            if (count < 2)
            {
                return;
            }
            float peakSpeed = MIN_SPEED_SCALE;
            for (int age = 0; age < count; age++)
            {
                peakSpeed = Mathf.Max(peakSpeed, _source.HistoryAt(_source.SpeedHistory, age));
            }
            DrawSeries(painter, rect, _source.EffortHistory, count, 1f, EffortColor);
            DrawSeries(painter, rect, _source.UprightHistory, count, 1f, UprightColor);
            DrawSeries(painter, rect, _source.SpeedHistory, count, peakSpeed, SpeedColor);
        }

        private void DrawSeries(Painter2D painter, Rect rect, float[] series, int count, float scale, Color color)
        {
            painter.lineWidth = LINE_WIDTH;
            painter.lineJoin = LineJoin.Round;
            painter.strokeColor = color;
            painter.BeginPath();
            // The newest sample always sits at the right edge; an unfilled window grows in
            // from the right, the way a scrolling chart recorder would.
            float step = rect.width / (RacerTelemetry.HISTORY_LENGTH - 1);
            float startX = rect.xMax - (count - 1) * step;
            for (int age = 0; age < count; age++)
            {
                float value = Mathf.Clamp01(_source.HistoryAt(series, age) / scale);
                var point = new Vector2(startX + age * step, rect.yMax - value * rect.height);
                if (age == 0)
                {
                    painter.MoveTo(point);
                }
                else
                {
                    painter.LineTo(point);
                }
            }
            painter.Stroke();
        }
    }
}
