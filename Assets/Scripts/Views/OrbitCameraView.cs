using PoRacer.Systems;
using Unity.Cinemachine;
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

        // The frame every Shots radius below was authored against: 9:16 at 40 deg
        // vertical, where the horizontal half-tangent is tan(20) * 0.5625 = 0.2047.
        // ApplyLensAndGetFrameScale measures the live frame against this one.
        private const float AUTHORED_FOV = 40f;
        private const float AUTHORED_HORIZONTAL_TAN = 0.20473f;
        // Adaptive portrait FOV, the same curve PackCameraView uses so a cut between
        // the two rigs does not change the apparent lens.
        private const float WIDE_PORTRAIT_FOV = 54f;
        private const float NARROW_PORTRAIT_FOV = 42f;
        // Bounds on the correction. A game view or device reporting a freak aspect
        // must not be able to fling the shot into the next valley or bury it inside
        // the racer; past these the composition is already beyond saving.
        private const float MIN_FRAME_SCALE = 0.6f;
        private const float MAX_FRAME_SCALE = 2f;

        /// <summary>
        /// One camera angle. Azimuth is degrees around the target measured from
        /// straight behind it (0 = chase, 90 = its left, 180 = head-on), taken
        /// against world +Z rather than the racer's own facing: a tumbling centipede's
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

        // Radii are authored for a 9:16 frame, where the horizontal half-angle is
        // only ~13.8 deg: visible width is roughly radius * 0.25, so a 3.4 m radius
        // frames 1.7 m and a hexapod does not fit inside it. These keep the
        // subject at roughly half the frame width.
        //
        // 9:16 is the authoring reference, not an assumption about the device: every
        // radius and height here is multiplied by ApplyLensAndGetFrameScale() so the
        // same composition survives a taller phone or a stretched game view. Tune
        // these against 9:16 and the rest follows.
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

        /// <summary>
        /// One camera angle on a course, placed along the centreline instead of
        /// around a world-space compass. Along is metres up the course from the
        /// racer: negative trails it, positive waits ahead of it.
        /// </summary>
        private readonly struct CourseShotDef
        {
            public readonly float AlongMeters;
            public readonly float Height;
            public readonly float LookHeight;

            public CourseShotDef(float alongMeters, float height, float lookHeight)
            {
                AlongMeters = alongMeters;
                Height = height;
                LookHeight = lookHeight;
            }
        }

        // Course coverage sits ON the road, either trailing the racer or waiting
        // up the track for it, so the sight line runs down the road corridor. On
        // the Apartment circuit the racers run inside a flat: a shot placed at a
        // world-space azimuth spends half the lap looking through a wall or a
        // wardrobe, and no keep-out volume can help because the scenery has no
        // colliders and there are 2100 pieces of it. Staying on the centreline is
        // the only placement that is guaranteed clear, because the road is the one
        // part of the map that is known to be empty.
        private static readonly CourseShotDef[] CourseShots =
        {
            // Chase: trailing and above, the readable "who is winning" shot.
            new(-9f, 3f, 0.6f),
            // Low chase: down at limb height, where the gait reads.
            new(-6f, 1.2f, 0.5f),
            // Head-on: waiting up the road for the racer to come to the lens.
            new(8f, 2.4f, 0.6f),
            // Low head-on: the same, at knee height.
            new(5.5f, 1f, 0.4f),
            // Long crane behind, showing ground gained on the field.
            new(-13f, 5.5f, 0.4f)
        };

        // Below this the shot is effectively inside the racer, which happens when
        // a shot is clamped at either end of the course; back it off along the road.
        private const float MIN_COURSE_SEPARATION = 2f;

        private Transform _target;
        private Vector3 _focusPoint;
        private bool _hasFocus;
        private float _driftDegrees;
        private int _shotIndex;
        private float _shotElapsed;
        private Bounds _keepOut;
        private bool _hasKeepOut;
        private Systems_CoursePath _course;
        private Camera _mainCamera;
        // The vcam this view drives. Same reason PackCameraView holds one: the
        // adaptive FOV must be written to the vcam's lens, never to Camera.main,
        // because CinemachineBrain copies the active vcam's lens onto the Camera in
        // its own LateUpdate and would overwrite it.
        private CinemachineCamera _lensOwner;

        private void Awake()
        {
            _mainCamera = Camera.main;
        }

        /// <summary>The vcam whose lens carries this shot's adaptive portrait FOV.</summary>
        public void BindLens(CinemachineCamera lensOwner)
        {
            _lensOwner = lensOwner;
        }

        /// <summary>
        /// Writes the adaptive portrait FOV and returns the factor the authored shot
        /// distances must be multiplied by to keep the subject the same fraction of
        /// frame WIDTH at the aspect actually being rendered.
        ///
        /// Both halves are why this exists. The FOV curve is copied from
        /// PackCameraView on purpose — the two rigs cut between each other, and two
        /// different focal lengths across a hard cut reads as a lens change nobody
        /// asked for. Until 2026-09-10 this rig had neither: it ran on Cinemachine's
        /// default 40 deg while the pack shot adapted, and since the orbit rig owns
        /// the shot for essentially the whole race, the adapting one was the one
        /// nobody was watching.
        ///
        /// The scale is the other half. <see cref="Shots"/> radii were authored
        /// against a 9:16 frame at 40 deg, where the horizontal half-tangent is
        /// 0.205; at the 0.361 aspect a tall phone or a stretched game view gives,
        /// it is 0.167, so the same radius frames 18% less width and the subject
        /// crops. Scaling the whole shot — radius and height together, so the
        /// elevation angle is preserved and it reads as a dolly, not a crane —
        /// restores the authored composition at any aspect.
        /// </summary>
        private float ApplyLensAndGetFrameScale()
        {
            if (_mainCamera == null)
            {
                return 1f;
            }
            float aspect = _mainCamera.aspect;
            float targetFov = aspect < 1f ? Mathf.Lerp(WIDE_PORTRAIT_FOV, NARROW_PORTRAIT_FOV, aspect) : AUTHORED_FOV;
            if (_lensOwner != null)
            {
                LensSettings lens = _lensOwner.Lens;
                lens.FieldOfView = targetFov;
                _lensOwner.Lens = lens;
            }
            float horizontalTan = Mathf.Tan(targetFov * 0.5f * Mathf.Deg2Rad) * aspect;
            if (horizontalTan <= 0.001f)
            {
                return 1f;
            }
            return Mathf.Clamp(AUTHORED_HORIZONTAL_TAN / horizontalTan, MIN_FRAME_SCALE, MAX_FRAME_SCALE);
        }

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
            int shotCount = _course != null ? CourseShots.Length : Shots.Length;
            _shotIndex = shotCount == 0 ? 0 : (_shotIndex + 1) % shotCount;
            _shotElapsed = 0f;
            _driftDegrees = 0f;
        }

        /// <summary>
        /// Places the shot on the course centreline, ahead of or behind the racer,
        /// and looks back along the road at it. Both ends are clamped by
        /// <see cref="Systems_CoursePath.PointAt"/>, so a shot that would run off
        /// the start or the finish is pulled back onto the road and then pushed
        /// clear of the racer along the local tangent.
        /// </summary>
        private void ApplyCourseShot(float frameScale)
        {
            CourseShotDef shot = CourseShots[_shotIndex % CourseShots.Length];
            float along = _course.Project(_focusPoint);
            float cameraAlong = Mathf.Clamp(along + shot.AlongMeters * frameScale, 0f, _course.Length);
            Vector3 onRoad = _course.PointAt(cameraAlong);

            // Clamping at either end can leave the lens on top of the racer; back
            // it off down the road rather than letting the shot collapse.
            Vector3 flatSeparation = onRoad - _focusPoint;
            flatSeparation.y = 0f;
            if (flatSeparation.magnitude < MIN_COURSE_SEPARATION)
            {
                Vector3 tangent = _course.TangentAt(cameraAlong);
                onRoad += tangent * Mathf.Sign(shot.AlongMeters) * MIN_COURSE_SEPARATION;
            }

            // Slow motion pushes the shot in, matching the world-space shots.
            float zoom = Mathf.Lerp(0.7f, 1f, Time.timeScale) * frameScale;
            transform.position = PushOutOfKeepOut(onRoad + Vector3.up * (shot.Height * zoom));
            transform.LookAt(_focusPoint + Vector3.up * shot.LookHeight);
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

        /// <summary>
        /// The course being raced, or null on a builder map. With a course set the
        /// shot is placed along the centreline rather than at a world-space
        /// azimuth, which is what keeps scenery out from between lens and racer.
        /// </summary>
        public void SetCourse(Systems_CoursePath course) => _course = course;

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
            // Written once per frame, before either shot path places the lens: the
            // FOV goes onto the vcam and the returned factor rescales the authored
            // distances for the aspect actually on screen.
            float frameScale = ApplyLensAndGetFrameScale();
            if (_course != null)
            {
                ApplyCourseShot(frameScale);
                return;
            }
            ShotDef shot = Shots[_shotIndex];
            _driftDegrees += shot.DriftDegreesPerSecond * Time.deltaTime;

            // Azimuth 0 sits behind the racer, i.e. on the -Z side of it, since the
            // whole field runs toward +Z.
            float radians = (180f + shot.AzimuthDegrees + _driftDegrees) * Mathf.Deg2Rad;
            // Slow motion pushes the shot in: at timescale 0.35 the framing tightens
            // ~20%, selling the drama without touching the lens.
            float zoom = Mathf.Lerp(0.7f, 1f, Time.timeScale) * frameScale;
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
