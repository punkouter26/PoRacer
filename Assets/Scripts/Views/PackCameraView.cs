using System.Collections.Generic;
using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Drives the runtime pack camera created by Systems_CameraDirector: a
    /// crane shot behind and above the lead group that widens or tightens so
    /// the racers who matter stay in frame. Stragglers far behind the leader
    /// are dropped from the framing — chasing them made everyone a speck.
    /// Bounds and motion are smoothed so the shot breathes instead of
    /// twitching with the physics. The CinemachineCamera on this object is
    /// passive, so its transform — written here in LateUpdate — is the shot.
    /// </summary>
    public sealed class PackCameraView : MonoBehaviour
    {
        private const float MIN_DISTANCE = 8f;
        private const float MAX_DISTANCE = 45f;
        // Only racers within this many meters of the front runner are framed.
        private const float LEAD_WINDOW_METERS = 14f;
        // Extra room around the bounds so nobody clips the frame edge.
        private const float FRAME_PADDING = 1.1f;
        private const float HEIGHT_RATIO = 0.62f;
        private const float BACK_RATIO = 0.8f;
        // Exponential smoothing rate; higher = snappier.
        private const float SMOOTHING = 2.5f;
        private const float DEFAULT_VERTICAL_TAN = 0.364f;  // fov 40
        private const float DEFAULT_ASPECT = 0.5625f;       // portrait 9:16

        private readonly List<Transform> _targets = new();
        // Set for authored courses: "ahead" is the centreline's heading at the
        // front runner, and the lead window is measured along the course, not +Z.
        private Systems.Systems_CoursePath _course;
        private Camera _mainCamera;
        private Vector3 _center;
        private float _distance = MIN_DISTANCE;
        private bool _hasFrame;

        public void SetTargets(IReadOnlyList<Transform> targets)
        {
            _targets.Clear();
            for (int targetIndex = 0; targetIndex < targets.Count; targetIndex++)
            {
                _targets.Add(targets[targetIndex]);
            }
            _hasFrame = false;
        }

        public void SetCourse(Systems.Systems_CoursePath course)
        {
            _course = course;
            _hasFrame = false;
        }

        private static bool IsFinite(Vector3 position)
        {
            return float.IsFinite(position.x) && float.IsFinite(position.y) && float.IsFinite(position.z);
        }

        private float LeadMetric(Vector3 position)
        {
            return _course != null ? _course.Project(position) : position.z;
        }

        private void Awake()
        {
            _mainCamera = Camera.main;
        }

        private void LateUpdate()
        {
            // Pass 1: the front runner sets the framing window.
            float leaderZ = float.NegativeInfinity;
            for (int targetIndex = 0; targetIndex < _targets.Count; targetIndex++)
            {
                Transform target = _targets[targetIndex];
                if (target == null || !target.gameObject.activeInHierarchy || !IsFinite(target.position))
                {
                    continue;
                }
                leaderZ = Mathf.Max(leaderZ, LeadMetric(target.position));
            }
            if (float.IsNegativeInfinity(leaderZ))
            {
                return;
            }

            // Pass 2: bounds over the lead group only.
            Vector3 min = Vector3.positiveInfinity;
            Vector3 max = Vector3.negativeInfinity;
            int alive = 0;
            for (int targetIndex = 0; targetIndex < _targets.Count; targetIndex++)
            {
                Transform target = _targets[targetIndex];
                if (target == null || !target.gameObject.activeInHierarchy || !IsFinite(target.position)
                    || LeadMetric(target.position) < leaderZ - LEAD_WINDOW_METERS)
                {
                    continue;
                }
                min = Vector3.Min(min, target.position);
                max = Vector3.Max(max, target.position);
                alive++;
            }
            if (alive == 0)
            {
                return;
            }

            Vector3 center = (min + max) * 0.5f;
            Vector3 extents = (max - min) * 0.5f;

            float verticalTan = DEFAULT_VERTICAL_TAN;
            float aspect = DEFAULT_ASPECT;
            if (_mainCamera != null)
            {
                aspect = _mainCamera.aspect;
                // Adaptive vertical FOV: wider in portrait so the shot stays tight without excessive camera back-off
                float targetFov = aspect < 1.0f ? Mathf.Lerp(54f, 42f, aspect) : 40f;
                _mainCamera.fieldOfView = targetFov;
                verticalTan = Mathf.Tan(targetFov * 0.5f * Mathf.Deg2Rad);
            }
            float horizontalTan = verticalTan * aspect;
            // Width must fit the narrow portrait frame; depth and height map
            // mostly onto the vertical axis from this camera angle.
            float widthDistance = extents.x * FRAME_PADDING / Mathf.Max(0.05f, horizontalTan);
            float depthDistance = (extents.z + extents.y) * FRAME_PADDING / Mathf.Max(0.05f, verticalTan);
            float targetDistance = Mathf.Clamp(
                Mathf.Max(widthDistance, depthDistance, MIN_DISTANCE), MIN_DISTANCE, MAX_DISTANCE);

            if (!_hasFrame)
            {
                _center = center;
                _distance = targetDistance;
                _hasFrame = true;
            }
            float blend = 1f - Mathf.Exp(-SMOOTHING * Time.deltaTime);
            _center = Vector3.Lerp(_center, center, blend);
            _distance = Mathf.Lerp(_distance, targetDistance, blend);

            Vector3 ahead = _course != null ? _course.HeadingAt(leaderZ) : Vector3.forward;
            transform.position = _center
                + Vector3.up * (_distance * HEIGHT_RATIO) - ahead * (_distance * BACK_RATIO);
            transform.LookAt(_center + ahead * (_distance * 0.1f));
        }
    }
}
