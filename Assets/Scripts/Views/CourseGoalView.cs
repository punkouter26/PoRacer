using PoRacer.Agents;
using PoRacer.Systems;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// The carrot. One per racer on an authored course: a goal transform kept
    /// LOOKAHEAD_METERS ahead of the racer along the centreline, so the goal
    /// direction every brain already observes points down the road through
    /// every bend, and the distance the reward reads is the course distance
    /// left rather than the ever-constant gap to a moving marker.
    /// </summary>
    public sealed class CourseGoalView : MonoBehaviour, ICourseProgress
    {
        private const float LOOKAHEAD_METERS = 6f;
        private const float GOAL_HEIGHT = 0.5f;

        private Systems_CoursePath _path;
        private Transform _body;
        private Transform _transform;
        private float _progress;

        public float ProgressMeters => _progress;

        public float RemainingMeters => _path != null ? Mathf.Max(0f, _path.Length - _progress) : 0f;

        public void Initialize(Systems_CoursePath path, Transform body)
        {
            _path = path;
            _body = body;
            _transform = transform;
            Refresh();
        }

        private void FixedUpdate()
        {
            Refresh();
        }

        /// <summary>Re-projects the body now; the training area calls this right after a teleport.</summary>
        public void Refresh()
        {
            if (_path == null || _body == null)
            {
                return;
            }
            _progress = _path.Project(_body.position);
            _transform.position = _path.PointAt(_progress + LOOKAHEAD_METERS) + Vector3.up * GOAL_HEIGHT;
        }
    }
}
