using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Drives the runtime orbit camera created by Systems_CameraDirector: circles
    /// slowly around a target (the race leader) while smoothing out the target's
    /// physics jitter. The CinemachineCamera on this object is passive, so its
    /// transform — written here in LateUpdate — is the shot.
    /// </summary>
    public sealed class OrbitCameraView : MonoBehaviour
    {
        private const float ORBIT_RADIUS = 4.5f;
        private const float ORBIT_HEIGHT = 2.4f;
        private const float ORBIT_DEGREES_PER_SECOND = 18f;
        // Exponential smoothing rate for the focus point; higher = snappier.
        private const float FOLLOW_SMOOTHING = 4f;
        // Clearance kept between the lens and a keep-out face, so the near plane
        // does not poke through the surface the camera was just pushed out of.
        private const float KEEP_OUT_MARGIN = 0.6f;

        private Transform _target;
        private Vector3 _focusPoint;
        private bool _hasFocus;
        private float _angleDegrees;
        private Bounds _keepOut;
        private bool _hasKeepOut;

        public void SetTarget(Transform target)
        {
            _target = target;
            if (target != null && !_hasFocus)
            {
                _focusPoint = target.position;
                _hasFocus = true;
            }
        }

        /// <summary>
        /// Volume the shot must never enter. Track scenery ships without colliders
        /// (see Systems_TrackBuilder.DecorateTrack), so the finish arch is invisible
        /// to a physics sweep — the director hands the volume over explicitly.
        /// </summary>
        public void SetKeepOut(Bounds keepOut)
        {
            _keepOut = keepOut;
            _hasKeepOut = true;
        }

        public void ClearKeepOut() => _hasKeepOut = false;

        private void LateUpdate()
        {
            if (_target == null)
            {
                return;
            }
            _focusPoint = Vector3.Lerp(
                _focusPoint, _target.position, 1f - Mathf.Exp(-FOLLOW_SMOOTHING * Time.deltaTime));
            _angleDegrees += ORBIT_DEGREES_PER_SECOND * Time.deltaTime;
            float radians = _angleDegrees * Mathf.Deg2Rad;
            // Slow motion pushes the shot in: at timescale 0.35 the orbit tightens
            // ~20%, selling the drama without touching the lens.
            float zoom = Mathf.Lerp(0.7f, 1f, Time.timeScale);
            Vector3 offset = new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians)) * (ORBIT_RADIUS * zoom)
                + Vector3.up * (ORBIT_HEIGHT * zoom);
            transform.position = PushOutOfKeepOut(_focusPoint + offset);
            transform.LookAt(_focusPoint + Vector3.up * 0.5f);
        }

        /// <summary>
        /// Slides a camera position that landed inside the keep-out volume out
        /// through the nearer of its two track-facing walls. The push is along Z
        /// only: the finish gate spans the full track width, so a sideways escape
        /// would swing the shot into the crowd, while front/behind the line is
        /// exactly where a finish camera belongs.
        /// </summary>
        private Vector3 PushOutOfKeepOut(Vector3 position)
        {
            if (!_hasKeepOut || !_keepOut.Contains(position))
            {
                return position;
            }
            float toFront = position.z - (_keepOut.min.z - KEEP_OUT_MARGIN);
            float toBack = (_keepOut.max.z + KEEP_OUT_MARGIN) - position.z;
            position.z = toFront < toBack
                ? _keepOut.min.z - KEEP_OUT_MARGIN
                : _keepOut.max.z + KEEP_OUT_MARGIN;
            return position;
        }
    }
}
