using UnityEngine;

namespace PoRacer.Views
{
    /// <summary>
    /// Drives the runtime orbit camera created by Systems_CameraDirector: a close
    /// shot on a target (the race leader) that cuts between fixed broadcast angles
    /// every few seconds instead of holding one. The CinemachineCamera on this
    /// object is passive, so its transform — written here in LateUpdate — is the
    /// shot.
    /// </summary>
    public sealed class OrbitCameraView : MonoBehaviour
    {
        // Exponential smoothing rate for the focus point; higher = snappier.
        private const float FOLLOW_SMOOTHING = 4f;
        // Clearance kept between the lens and a keep-out face, so the near plane
        // does not poke through the surface the camera was just pushed out of.
        private const float KEEP_OUT_MARGIN = 0.6f;
        // Hold on each angle before cutting to the next. Cuts are hard, the way
        // sports coverage cuts — a blend would read as drifting, not as a new
        // camera. Long enough to register the shot, short enough to stay lively.
        private const float SHOT_SECONDS = 5f;

        /// <summary>
        /// One camera angle. Azimuth is degrees around the target measured from
        /// straight behind it (0 = chase, 90 = its left, 180 = head-on), taken
        /// against world +Z rather than the racer's own facing: a tumbling snake's
        /// forward vector spins, and a shot anchored to it would spin with it.
        /// </summary>
        private readonly struct ShotDef
        {
            public readonly float Radius;
            public readonly float Height;
            public readonly float AzimuthDegrees;
            public readonly float DriftDegreesPerSecond;
            public readonly float LookHeight;

            public ShotDef(float radius, float height, float azimuthDegrees,
                float driftDegreesPerSecond, float lookHeight)
            {
                Radius = radius;
                Height = height;
                AzimuthDegrees = azimuthDegrees;
                DriftDegreesPerSecond = driftDegreesPerSecond;
                LookHeight = lookHeight;
            }
        }

        // Radii are set for a 9:16 frame, where the horizontal half-angle is only
        // ~13.8 deg: visible width is roughly radius * 0.25, so a 3.4 m radius
        // frames 1.7 m and a hexapod does not fit inside it. These keep the
        // subject at roughly half the frame width.
        private static readonly ShotDef[] Shots =
        {
            // Chase: behind and above, the readable "who is winning" shot.
            new(8.0f, 3.0f, 0f, 4f, 0.5f),
            // Low side: down at limb height, where the gait actually reads.
            new(7.0f, 1.4f, 78f, -6f, 0.4f),
            // Crane: high and back, showing ground gained on the field.
            new(9.0f, 6.5f, 20f, 8f, 0.2f),
            // Head-on three-quarter: the racer coming at the lens.
            new(7.5f, 2.2f, 210f, -5f, 0.5f),
            // Full orbit: the original circling shot, kept as the showpiece.
            new(8.0f, 3.4f, 0f, 42f, 0.5f)
        };

        private Transform _target;
        private Vector3 _focusPoint;
        private bool _hasFocus;
        private float _driftDegrees;
        private int _shotIndex;
        private float _shotElapsed;
        private Bounds _keepOut;
        private bool _hasKeepOut;

        public void SetTarget(Transform target)
        {
            bool changedTarget = _target != target;
            _target = target;
            if (target != null && !_hasFocus)
            {
                _focusPoint = target.position;
                _hasFocus = true;
            }
            if (changedTarget)
            {
                // A new leader earns a fresh angle rather than inheriting whatever
                // the last one was halfway through.
                AdvanceShot();
            }
        }

        private void AdvanceShot()
        {
            _shotIndex = Shots.Length == 0 ? 0 : (_shotIndex + 1) % Shots.Length;
            _shotElapsed = 0f;
            _driftDegrees = 0f;
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

            // Unscaled, so a slow-motion finish does not stretch a 5 s hold into 15.
            _shotElapsed += Time.unscaledDeltaTime;
            if (_shotElapsed >= SHOT_SECONDS)
            {
                AdvanceShot();
            }
            ShotDef shot = Shots[_shotIndex];
            _driftDegrees += shot.DriftDegreesPerSecond * Time.deltaTime;

            // Azimuth 0 sits behind the racer, i.e. on the -Z side of it, since the
            // whole field runs toward +Z.
            float radians = (180f + shot.AzimuthDegrees + _driftDegrees) * Mathf.Deg2Rad;
            // Slow motion pushes the shot in: at timescale 0.35 the framing tightens
            // ~20%, selling the drama without touching the lens.
            float zoom = Mathf.Lerp(0.7f, 1f, Time.timeScale);
            Vector3 offset = new Vector3(Mathf.Sin(radians), 0f, Mathf.Cos(radians)) * (shot.Radius * zoom)
                + Vector3.up * (shot.Height * zoom);
            transform.position = PushOutOfKeepOut(_focusPoint + offset);
            transform.LookAt(_focusPoint + Vector3.up * shot.LookHeight);
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
