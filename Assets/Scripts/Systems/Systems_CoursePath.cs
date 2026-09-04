using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Systems
{
    /// <summary>
    /// A race course as a polyline through its centreline: projects a world
    /// position onto the path to read distance-along-course, and samples the
    /// path at a distance to place a look-ahead goal. Pure C#; built once per
    /// course from the authored spline points, read every frame by racers.
    /// </summary>
    public sealed class Systems_CoursePath
    {
        private readonly Vector3[] _points;
        private readonly float[] _cumulative;
        private readonly float _length;

        public Systems_CoursePath(IReadOnlyList<Vector3> points)
        {
            var kept = new List<Vector3>(points.Count);
            for (int pointIndex = 0; pointIndex < points.Count; pointIndex++)
            {
                // Duplicate consecutive points (the authored spline repeats its
                // last knot) would give a zero-length segment and a NaN tangent.
                if (kept.Count == 0 || (points[pointIndex] - kept[kept.Count - 1]).sqrMagnitude > 1e-6f)
                {
                    kept.Add(points[pointIndex]);
                }
            }
            if (kept.Count < 2)
            {
                kept.Add(kept.Count == 0 ? Vector3.zero : kept[0] + Vector3.forward);
            }
            _points = kept.ToArray();
            _cumulative = new float[_points.Length];
            for (int pointIndex = 1; pointIndex < _points.Length; pointIndex++)
            {
                _cumulative[pointIndex] = _cumulative[pointIndex - 1]
                    + Vector3.Distance(_points[pointIndex - 1], _points[pointIndex]);
            }
            _length = _cumulative[_points.Length - 1];
        }

        public float Length => _length;

        public Vector3 Start => _points[0];

        public Vector3 End => _points[_points.Length - 1];

        /// <summary>
        /// Distance along the course of the point on the path nearest to
        /// <paramref name="position"/>. Nearest in three dimensions on purpose:
        /// the switchbacks stack one road over another, and a racer on the upper
        /// one must project onto the upper one, which a flat projection cannot tell apart.
        /// </summary>
        public float Project(Vector3 position)
        {
            float bestDistanceSq = float.MaxValue;
            float bestAlong = 0f;
            for (int segmentIndex = 0; segmentIndex < _points.Length - 1; segmentIndex++)
            {
                Vector3 a = _points[segmentIndex];
                Vector3 ab = _points[segmentIndex + 1] - a;
                float segmentLengthSq = ab.sqrMagnitude;
                float t = segmentLengthSq > 1e-6f
                    ? Mathf.Clamp01(Vector3.Dot(position - a, ab) / segmentLengthSq)
                    : 0f;
                float distanceSq = (position - (a + ab * t)).sqrMagnitude;
                if (distanceSq < bestDistanceSq)
                {
                    bestDistanceSq = distanceSq;
                    bestAlong = _cumulative[segmentIndex] + (_cumulative[segmentIndex + 1] - _cumulative[segmentIndex]) * t;
                }
            }
            return bestAlong;
        }

        /// <summary>Horizontal distance from <paramref name="position"/> to the path.</summary>
        public float LateralDistance(Vector3 position)
        {
            Vector3 onPath = PointAt(Project(position));
            onPath.y = position.y;
            return Vector3.Distance(position, onPath);
        }

        /// <summary>World point <paramref name="distance"/> metres along the course, clamped to its ends.</summary>
        public Vector3 PointAt(float distance)
        {
            int segmentIndex = SegmentAt(distance, out float t);
            return Vector3.LerpUnclamped(_points[segmentIndex], _points[segmentIndex + 1], t);
        }

        /// <summary>Unit direction of travel at <paramref name="distance"/> metres along the course.</summary>
        public Vector3 TangentAt(float distance)
        {
            int segmentIndex = SegmentAt(distance, out _);
            Vector3 tangent = _points[segmentIndex + 1] - _points[segmentIndex];
            return tangent.sqrMagnitude > 1e-6f ? tangent.normalized : Vector3.forward;
        }

        /// <summary>Flattened direction of travel: what "forward" means to a racer facing down the course.</summary>
        public Vector3 HeadingAt(float distance)
        {
            Vector3 tangent = TangentAt(distance);
            tangent.y = 0f;
            return tangent.sqrMagnitude > 1e-6f ? tangent.normalized : Vector3.forward;
        }

        private int SegmentAt(float distance, out float t)
        {
            distance = Mathf.Clamp(distance, 0f, _length);
            int segmentIndex = 0;
            while (segmentIndex < _points.Length - 2 && _cumulative[segmentIndex + 1] < distance)
            {
                segmentIndex++;
            }
            float segmentLength = _cumulative[segmentIndex + 1] - _cumulative[segmentIndex];
            t = segmentLength > 1e-6f ? (distance - _cumulative[segmentIndex]) / segmentLength : 0f;
            return segmentIndex;
        }
    }
}
